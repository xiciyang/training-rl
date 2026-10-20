#!/usr/bin/env node
/**
 * instrument-calibration —— 长任务 5 · 阶段 3：**偏好仪器（生成长度上限）标定**。
 *
 * ## 为什么需要它
 * LT4 的 RL run（`rl_20260929155357_gr86`）在 `--pref-max-tokens 256` 下
 * 基座 38/48、策略 35/48 条生成被截断（`finish_reason='length'`），
 * 双双超过本包 `MAX_TRUNCATION_RATIO = 0.5` ⇒ 偏好门禁判 **测量不成立**（判据无区分度）。
 *
 * 那是**仪器**的失败，不是"RL 没让模型变好"。本脚本把"换多大的生成长度上限才够"
 * 变成**可复核的数字**：在同一批留出 prompt 上，逐档测两侧的截断率。
 *
 * ## 纪律
 * - **判定仪器只能是 serving**：本脚本只走 `src/evaluate.mjs#evalPreference`
 *   （与 validate 内部**同一个函数、同一批 heldout 前 n 行**），不碰容器内评估。
 * - **不动判据**：不读、不写 `minRewardGainPct` / `z`；截断率闸门常量 `MAX_TRUNCATION_RATIO`
 *   只用来**对照**（标定选点的依据），不修改。
 * - **一个仪器一个值**：选定的 `maxTokens` 会写进证据；后续所有 seed 必须用同一个值
 *   （基线缓存键含 protocol + maxTokens ⇒ 换值必然重测基线，不会混用两个仪器）。
 * - 失败如实落盘（`ok:false` + 原始错误），不假装成功。
 *
 * 用法：
 *   node tests/realmachine/instrument-calibration.mjs --run <runId> \
 *        --max-tokens 256,512,768,1024 --n 48 --out ../../out/rl-instrument-calibration.json
 */
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const TRAINING = join(PKG, '..')
const OUT = join(TRAINING, 'out')

const core = await import(join(PKG, 'src', 'core.mjs'))
const ev = await import(join(PKG, 'src', 'evaluate.mjs'))
const { resolveServingPath, ServingMcpClient, callWithConfirm } = await import(join(PKG, 'src', 'serving-client.mjs'))
const { paths } = await import(join(PKG, 'src', 'runtime.mjs'))

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : d }

const RUN_ID = argOf('--run', null)
const MAX_TOKENS = String(argOf('--max-tokens', '256,512,768,1024')).split(',').map((s) => Number(s.trim())).filter(Number.isFinite)
const N = Number(argOf('--n', '48'))
const EVIDENCE = argOf('--out', join(OUT, 'rl-instrument-calibration.json'))
const DEPLOY_FRACTION = core.DEPLOY_GPU_MEM_FRACTION

const log = (s) => process.stderr.write(`[calib] ${s}\n`)

const result = {
  schema: 'ai-infra/rl-instrument-calibration@1',
  generatedAt: new Date().toISOString(),
  runId: RUN_ID,
  instrument: { protocol: ev.EVAL_PROTOCOL, truncationGate: ev.MAX_TRUNCATION_RATIO, maxTokensSweep: MAX_TOKENS, n: N, evaluator: 'training-rl/src/evaluate.mjs#evalPreference（与 validate 同一函数、同一批 heldout 前 n 行）' },
  baseModel: null,
  policyArtifact: null,
  sweep: [],
  chosen: null,
  ok: false,
  error: null,
}

const write = () => { mkdirSync(dirname(EVIDENCE), { recursive: true }); writeFileSync(EVIDENCE, `${JSON.stringify(result, null, 2)}\n`); log(`证据落盘 ${EVIDENCE}`) }

let client = null
let policyEp = null
let baseEp = null
let policyToken = null
let baseToken = null

try {
  if (!RUN_ID) throw new Error('需要 --run <runId>')
  const rep = await core.getReport({ runId: RUN_ID })
  const run = rep.run
  const mergedSelected = join(run.runDir, 'merged-selected')
  const policyDir = existsSync(join(mergedSelected, 'config.json')) ? mergedSelected : join(run.runDir, 'merged')
  if (!existsSync(join(policyDir, 'config.json'))) throw new Error(`策略产物不合法（缺 config.json）：${policyDir}`)

  result.baseModel = { path: run.baseModel.path, revision: run.baseModel.revision ?? null }
  result.policyArtifact = { dir: policyDir, runStatus: run.status, selectedStep: run.selection?.selected?.step ?? null, selectionRule: run.selection?.selectionRule ?? null }

  const rows = readFileSync(join(run.dataset.dir, 'heldout.jsonl'), 'utf8').trim().split('\n').map((l) => JSON.parse(l))
  log(`留出 prompt ${rows.length} 条（取前 ${N} 条，与 validate 同序）`)

  const sp = await resolveServingPath({ root: paths().root })
  client = await new ServingMcpClient(sp.serverPath).start()
  log(`serving MCP 就绪：${sp.serverPath}`)

  log(`部署策略 ${policyDir} …`)
  const d = await callWithConfirm(client, 'inference_serving_deploy', {
    path: policyDir, owner: 'training-rl', purpose: `LT5 阶段3 仪器标定 ${RUN_ID}`,
    servedModelName: `calib-policy-${RUN_ID}`, gpuMemoryUtilization: DEPLOY_FRACTION, maxNumSeqs: 4,
  }, { log })
  if (d.ok === false) throw new Error(`策略部署失败：${d.code} ${d.message ?? ''}`)
  policyToken = d.ownerToken
  policyEp = { url: d.endpoint.endpoint, model: d.endpoint.servedModelName, id: d.endpoint.id, container: d.endpoint.containerName }
  log(`  → ${policyEp.url}`)

  log(`部署基座 ${run.baseModel.path} …`)
  const b = await callWithConfirm(client, 'inference_serving_deploy', {
    path: run.baseModel.path, owner: 'training-rl', purpose: `LT5 阶段3 仪器标定基线 ${RUN_ID}`,
    servedModelName: `calib-base-${RUN_ID}`, gpuMemoryUtilization: DEPLOY_FRACTION, maxNumSeqs: 4,
  }, { log })
  if (b.ok === false) throw new Error(`基座部署失败：${b.code} ${b.message ?? ''}`)
  baseToken = b.ownerToken
  baseEp = { url: b.endpoint.endpoint, model: b.endpoint.servedModelName, id: b.endpoint.id, container: b.endpoint.containerName }
  log(`  → ${baseEp.url}`)

  for (const mt of MAX_TOKENS) {
    const t0 = Date.now()
    log(`── maxTokens=${mt} ──`)
    const pol = await ev.evalPreference({ endpoint: policyEp.url, model: policyEp.model, rows, limit: N, maxTokens: mt, log: (s) => log(`   [policy] ${s}`) })
    const bas = await ev.evalPreference({ endpoint: baseEp.url, model: baseEp.model, rows, limit: N, maxTokens: mt, log: (s) => log(`   [base] ${s}`) })
    const ratio = (m) => (m.n > 0 ? (m.truncated ?? 0) / m.n : null)
    const row = {
      maxTokens: mt,
      policy: { n: pol.n, nOk: pol.nOk, truncated: pol.truncated, stopped: pol.stopped, truncationRatio: ratio(pol), meanReward: pol.meanReward, protocol: pol.protocol },
      base: { n: bas.n, nOk: bas.nOk, truncated: bas.truncated, stopped: bas.stopped, truncationRatio: ratio(bas), meanReward: bas.meanReward, protocol: bas.protocol },
      clearsGateBothSides: ratio(pol) !== null && ratio(pol) <= ev.MAX_TRUNCATION_RATIO && ratio(bas) !== null && ratio(bas) <= ev.MAX_TRUNCATION_RATIO,
      elapsedMs: Date.now() - t0,
    }
    result.sweep.push(row)
    write()
    log(`   策略 截断 ${row.policy.truncated}/${row.policy.n} (${(row.policy.truncationRatio * 100).toFixed(1)}%)  基座 截断 ${row.base.truncated}/${row.base.n} (${(row.base.truncationRatio * 100).toFixed(1)}%)  清闸=${row.clearsGateBothSides}  ${(row.elapsedMs / 1000).toFixed(0)}s`)
  }

  const chosen = result.sweep.find((r) => r.clearsGateBothSides) ?? null
  result.chosen = chosen
    ? { maxTokens: chosen.maxTokens, reason: '扫描中**最小**的、两侧截断率都 ≤ MAX_TRUNCATION_RATIO(50%) 的生成长度上限' }
    : { maxTokens: null, reason: '扫描区间内两侧截断率**始终**未降到 50% 以下 ⇒ 仅靠提高生成长度上限不足以修复该仪器' }
  result.ok = true
} catch (e) {
  result.error = { code: e?.code ?? null, message: String(e?.message ?? e).slice(0, 1500) }
  result.ok = false
  log(`失败：${result.error.message}`)
} finally {
  if (client) {
    for (const [label, ep, tok] of [['policy', policyEp, policyToken], ['base', baseEp, baseToken]]) {
      if (!ep) continue
      try {
        const r = await callWithConfirm(client, 'inference_serving_teardown', { selector: ep.id, ownerToken: tok }, { log })
        log(`  teardown ${label}: ok=${r.ok} ${r.code ?? ''}`)
      } catch (e) { log(`  teardown ${label} 异常：${e.message}`) }
    }
    await client.close().catch(() => {})
  }
  write()
}

process.stdout.write(`${JSON.stringify({
  ok: result.ok, runId: RUN_ID, chosen: result.chosen,
  sweep: result.sweep.map((r) => ({ mt: r.maxTokens, policyTrunc: r.policy.truncated, baseTrunc: r.base.truncated, n: r.policy.n, clears: r.clearsGateBothSides })),
  error: result.error,
}, null, 2)}\n`)
process.exit(result.ok ? 0 : 1)
