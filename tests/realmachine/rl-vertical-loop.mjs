#!/usr/bin/env node
/**
 * rl-vertical-loop —— **真机**垂直闭环（长任务 3 · 阶段 3 的 P0 验收）。
 *
 * 走的是**已交付的插件 API 路径**（不是另写一套）：`prepare → train → validate`，
 * 与 `longtask4-seed-retrospect.mjs` 的用法同形（那份脚本是本项目认可的"怎么调交付 API"的样例）。
 *
 * ## 为什么要有 `--rehearse`
 * 主 run 要烧十几分钟 GPU。协议层（逐 step 握手 / ack 里带 save / 校准 / 合并 / 指纹）
 * 里的任何一个低级错误都会把这段时间浪费掉。故先跑一个 **2 步的彩排**：
 * 它走完**同一条代码路径**（init → σ_r 校准 → GRPOTrainer → 回调握手 → 落 adapter → 合并 → 指纹），
 * 代价约 0.02 GPU 小时，却能把"协议写错了"与"训练效果不好"两类问题分开。
 *
 * ## 纪律
 * - **判定仪器是 serving**（validate 内部就是这么做的）；
 * - **GPU 串行**：本脚本不自带抢锁逻辑；由调用方保证同一时刻只有一个 GPU 任务
 *   （长任务 3 阶段 3 的约定：等 `out/lt4-seed-retrospect-raw.json` 出现后再开跑）；
 * - 任何失败都**如实写进证据 JSON**（`ok:false` + 原始错误），不假装成功 ——
 *   证据文件必须存在（契约 C13d 会校验它）。
 *
 * 用法：
 *   node tests/realmachine/rl-vertical-loop.mjs --rehearse
 *   node tests/realmachine/rl-vertical-loop.mjs --steps 200 --kl-budget 0.1 --min-gain-pct 5
 *   node tests/realmachine/rl-vertical-loop.mjs --only validate --run <runId>   # 复用已训练的 run
 */
import { writeFileSync, mkdirSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { homedir } from 'node:os'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const TRAINING = join(PKG, '..')
const OUT = join(TRAINING, 'out')

const api = await import(join(PKG, 'src', 'core.mjs'))
const { ERROR_CODES } = await import(join(PKG, 'src', 'version.mjs'))

const argv = process.argv.slice(2)
const has = (k) => argv.includes(k)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : d }

const REHEARSE = has('--rehearse')
const ONLY = argOf('--only', null) // prepare | train | validate（默认全跑）
const RUN_ID = argOf('--run', null)
const STEPS = Number(argOf('--steps', REHEARSE ? '2' : '200'))
const KL_BUDGET = Number(argOf('--kl-budget', '0.1'))
const KL_TOL = Number(argOf('--kl-tolerance-pct', '50'))
const MIN_GAIN = Number(argOf('--min-gain-pct', '5'))
const DATASET = argOf('--dataset', join(homedir(), '.ai-infra/state/datasets/yahma-alpaca-cleaned__sft__s20260929__n1000'))
const EVIDENCE = argOf('--out', REHEARSE
  ? join(OUT, 'rl-rehearsal.json')
  : join(OUT, 'rl-vertical-loop.json'))

const log = (s) => process.stderr.write(`[rl-loop] ${s}\n`)

/** 只对"端点起不来"做有界重试（2 次，间隔 60s）；判据与阈值完全不变。 */
async function retryServing(fn, { attempts = 2, delayMs = 60000 } = {}) {
  let last = null
  for (let i = 1; i <= attempts; i += 1) {
    try { return await fn() } catch (e) {
      last = e
      const transient = e?.code === ERROR_CODES.E_SERVING_UNAVAILABLE || /E_DEGRADE_EXHAUSTED/.test(String(e?.message ?? ''))
      if (!transient || i === attempts) throw e
      log(`部署瞬时失败（${e.code ?? e.message}）⇒ ${delayMs / 1000}s 后重试（${i}/${attempts - 1}）；**判据与阈值不变**`)
      await new Promise((r) => setTimeout(r, delayMs))
    }
  }
  throw last
}
const spend = { train: 0, validate: 0 }

/** 走确认令牌闸门：首次调用会返回 E_CONFIRMATION_REQUIRED + confirmToken，回传即执行。 */
async function withConfirm(fn, args, label) {
  try {
    return await fn(args)
  } catch (e) {
    const token = e?.body?.details?.confirmToken
    if (e?.code === ERROR_CODES.E_CONFIRMATION_REQUIRED && token) {
      log(`${label}: 需确认（${e.code}）⇒ 回传 confirmToken 执行`)
      return await fn({ ...args, confirm: token })
    }
    throw e
  }
}

const result = {
  schema: 'ai-infra/rl-vertical-loop@1',
  generatedAt: new Date().toISOString(),
  mode: REHEARSE ? 'rehearse' : 'main',
  only: ONLY,
  inputs: { dataset: DATASET, steps: STEPS, klBudget: KL_BUDGET, klTolerancePct: KL_TOL, minGainPct: MIN_GAIN },
  ok: false,
  prepare: null,
  train: null,
  validate: null,
  gpuHours: null,
  error: null,
  stages: [],
}

const mark = (stage, ok, extra = {}) => { result.stages.push({ stage, ok, at: new Date().toISOString(), ...extra }) }

const write = () => {
  mkdirSync(OUT, { recursive: true })
  result.gpuHours = Number((spend.train + spend.validate).toFixed(4))
  writeFileSync(EVIDENCE, `${JSON.stringify(result, null, 2)}\n`)
  log(`证据落盘 ${EVIDENCE}（ok=${result.ok}, 累计 ${result.gpuHours} GPU·h）`)
}

try {
  if (!existsSync(join(DATASET, 'lineage.json'))) throw new Error(`数据集目录不合法（缺 lineage.json）：${DATASET}`)

  /* ── prepare ───────────────────────────────────────────────────────── */
  let promptsDir = argOf('--prompts-dir', null)
  // prepare 是纯 CPU（毫秒级）且 train/validate 都依赖它 ⇒ 除 `--only validate` 外一律执行
  if (ONLY !== 'validate') {
    log(`prepare：从既有（已去泄漏）数据集构造 RL prompt 集 …`)
    const p = await api.prepare({
      sourceDatasetDir: DATASET,
      trainPrompts: REHEARSE ? 32 : 400,
      heldoutPrompts: REHEARSE ? 16 : 200,
      maxPromptChars: 1024,
      force: true,
    })
    promptsDir = p.promptsDir
    result.prepare = {
      promptsDir: p.promptsDir, trainPrompts: p.trainPrompts, heldoutPrompts: p.heldoutPrompts,
      leakCheck: p.leakCheck, license: p.license, reward: p.reward, audit: p.audit,
    }
    mark('prepare', true, { train: p.trainPrompts, heldout: p.heldoutPrompts, leakOk: p.leakCheck.ok })
    log(`prepare 完成：train=${p.trainPrompts} heldout=${p.heldoutPrompts} 去泄漏双轨=${JSON.stringify({ h: p.leakCheck.hashOverlap, s: p.leakCheck.substringContained })}`)
  }

  /* ── train ─────────────────────────────────────────────────────────── */
  let runId = RUN_ID
  if (!ONLY || ONLY === 'train') {
    log(`train：${STEPS} 步（KL*=${KL_BUDGET}）…`)
    const t0 = Date.now()
    const r = await withConfirm(api.train, {
      promptsDir,
      baseModel: '<BASE_MODEL>',
      klBudget: KL_BUDGET,
      klTolerancePct: KL_TOL,
      lr: Number(argOf('--lr', '1e-6')),
      batchSize: Number(argOf('--batch', '8')),
      gradAccum: Number(argOf('--grad-accum', '1')),
      numGenerations: Number(argOf('--num-generations', '4')),
      maxCompletionLength: Number(argOf('--max-completion-length', '64')),
      maxSteps: STEPS,
      seed: Number(argOf('--seed', '20260929')),
      checkpointEvery: Number(argOf('--checkpoint-every', REHEARSE ? '1' : '50')),
      checkpointKeep: Number(argOf('--checkpoint-keep', '4')),
      evalEvery: Number(argOf('--eval-every', REHEARSE ? '1' : '50')),
      evalPrompts: Number(argOf('--eval-prompts', REHEARSE ? '4' : '8')),
      calibPrompts: Number(argOf('--calib-prompts', REHEARSE ? '4' : '8')),
      calibGenerations: Number(argOf('--calib-generations', REHEARSE ? '2' : '4')),
      limits: { maxSteps: STEPS, maxGpuHours: 2, concurrency: 1 },
      log: (s) => log(`  [train] ${s}`),
    }, 'train')
    runId = r.runId
    spend.train = Number(r.gpuHours) || 0
    result.train = {
      runId: r.runId, runDir: r.runDir, steps: r.steps, finalLoss: r.finalLoss,
      gpuHours: r.gpuHours, wallMs: Date.now() - t0,
      stopReason: r.stopReason, checkpointCount: r.checkpoints,
      budgetViolation: r.budgetViolation, budgetAssessment: r.budgetAssessment,
      violations: r.violations,
      reward: { kind: r.reward?.kind, version: r.reward?.version, components: r.reward?.components },
      calibration: r.calibration,
      kl: r.kl,
      rewardConformance: r.rewardConformance,
      artifactAssertion: r.artifactAssertion,
      fingerprints: r.fingerprints,
      selection: r.selection,
      audit: r.audit,
    }
    mark('train', true, { runId: r.runId, steps: r.steps, gpuHours: r.gpuHours, sigmaR: r.calibration?.sigmaR, beta: r.calibration?.beta })
    log(`train 完成：runId=${r.runId} steps=${r.steps} σ_r=${r.calibration?.sigmaR} β=${r.calibration?.beta} KL均值=${r.kl?.mean} GPU·h=${r.gpuHours}）
`.trim())
  }
  if (!ONLY || ONLY === 'validate') {
    if (!runId) throw new Error('validate 阶段需要 --run <runId>')
    log(`validate：serving 部署 + 三门禁 …`)
    // 部署瞬时失败（与本机既有服务/并行 workstream 抢内存导致降级链耗尽）时**有界重试**：
    // 判据（阈值/协议/样本量）一个字都不改，重试的只是"起不起得来端点"这件事。
    // 不缓存任何结果 ⇒ 不存在"换个结果更好的仪器"的空间。
    const v = await retryServing(() => withConfirm(api.validate, {
      runId,
      thresholds: { minRewardGainPct: MIN_GAIN },
      nTrain: Number(argOf('--n-train', REHEARSE ? '8' : '48')),
      nVal: Number(argOf('--n-val', REHEARSE ? '8' : '48')),
      nLm: Number(argOf('--n-lm', REHEARSE ? '40' : '100')),
      nMmlu: Number(argOf('--n-mmlu', REHEARSE ? '40' : '100')),
      prefMaxTokens: Number(argOf('--pref-max-tokens', '256')),
      log: (s) => log(`  [validate] ${s}`),
    }, 'validate'))
    spend.validate = (Number(v.ms) || 0) / 3_600_000
    result.validate = v
    mark('validate', v.pass === true, { pass: v.pass, ms: v.ms })
    log(`validate 完成：pass=${v.pass} 偏好门禁=${v.preferenceGate?.pass} 遗忘门禁=${v.forgettingGate?.pass} 过拟合=${v.overfit?.pass}`)
  }

  result.ok = result.stages.every((s) => s.ok === true)
} catch (e) {
  result.error = { code: e?.code ?? null, message: String(e?.message ?? e).slice(0, 1200), body: e?.body ? JSON.parse(JSON.stringify(e.body)).message ?? null : null }
  result.ok = false
  mark('error', false, { code: e?.code ?? null })
  log(`失败：${result.error.code ?? ''} ${result.error.message}`)
  if (e?.body?.details) log(`details: ${JSON.stringify(e.body.details).slice(0, 800)}`)
}

write()
process.stdout.write(`${JSON.stringify({
  ok: result.ok, mode: result.mode, runId: result.train?.runId ?? null,
  steps: result.train?.steps ?? null, gpuHours: result.gpuHours,
  sigmaR: result.train?.calibration?.sigmaR ?? null, beta: result.train?.calibration?.beta ?? null,
  klMean: result.train?.kl?.mean ?? null, artifactOk: result.train?.artifactAssertion?.ok ?? null,
  validatePass: result.validate?.pass ?? null, error: result.error,
}, null, 2)}\n`)
process.exit(result.ok ? 0 : 1)
