/**
 * guards.spec —— 输入处理 / 安全验收 / 健壮性。
 *
 * 全部**不依赖 GPU / 容器 / 网络**：需要外部世界的部分用最小夹具替代，
 * 但夹具**模拟对方的语义模型**（例如假容器 worker 会真的走 stdio 协议、会真的退出）。
 */
import { mkdtempSync, rmSync, writeFileSync, readFileSync, mkdirSync, existsSync, appendFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import * as api from '../../src/core.mjs'
import { ToolError, ERROR_CODES } from '../../src/version.mjs'
import { requireConfirm, tokenFor } from '../../src/confirm.mjs'
import { paths, updateRegistry, readRegistry, audit, auditVerify } from '../../src/runtime.mjs'
import { leakCheck } from '../../src/data/prepare.mjs'
import { ContainerWorker } from '../../src/worker.mjs'
import { preferenceGate, forgettingGate, evalPreference } from '../../src/evaluate.mjs'
import { selectCheckpoint } from '../../src/core.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
export const NAME = 'guards（输入 / 安全 / 健壮性 / O4 选点）'

const attempts = async (fn) => { try { const v = await fn(); return { ok: true, value: v } } catch (e) { return { ok: false, code: e.code, message: String(e.message), body: e.body } } }

const SRC_LINEAGE = {
  schema: 'ai-infra/lineage@1', source: 'fake/dataset', revision: 'abc', seed: 1,
  license: { spdx: 'cc-by-4.0', class: 'permissive', source: 'https://example.invalid/license' },
}

function mkSource(dir, { license = SRC_LINEAGE.license, overlap = false } = {}) {
  mkdirSync(dir, { recursive: true })
  const A = 'Explain in detail how a bicycle derailleur shifts gears under load without dropping the chain.'
  const B = 'Write a short paragraph about the water cycle and why evaporation matters for weather.'
  const C = 'Describe three differences between TCP and UDP with one concrete example for each case.'
  const train = [{ instruction: A, input: '', output: 'x' }, { instruction: B, input: '', output: 'y' }]
  const held = [{ instruction: C, input: '', output: 'z' }]
  if (overlap) {
    // 两条轨道各命中一次：① 完全相同（哈希交集）② 留出行是训练行的子串（行内包含）
    train.push({ instruction: C, input: '', output: 'z2' })
    held.push({ instruction: `${A} Please be concise.`, input: '', output: 'w' })
  }
  writeFileSync(join(dir, 'train.jsonl'), `${train.map((r) => JSON.stringify(r)).join('\n')}\n`)
  writeFileSync(join(dir, 'heldout.jsonl'), `${held.map((r) => JSON.stringify(r)).join('\n')}\n`)
  writeFileSync(join(dir, 'lineage.json'), `${JSON.stringify({ ...SRC_LINEAGE, license }, null, 2)}\n`)
  return dir
}

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tr-guard-'))
  const root = join(tmp, 'ai-infra')
  mkdirSync(join(root, 'state'), { recursive: true })
  const p = paths(root)

  /* ═══════════ 输入处理 ═══════════ */
  // 1) 缺必填参数
  let r = await attempts(() => api.prepare({ root }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_MISSING_PARAM, `1) prepare 缺 sourceDatasetDir → E_MISSING_PARAM（实际 ${r.code}）`)
  r = await attempts(() => api.train({ root, promptsDir: join(tmp, 'nope'), klBudget: 0.1 }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_PROMPTS_INVALID, `1b) train 的 promptsDir 不存在 → E_PROMPTS_INVALID（实际 ${r.code}）`)

  // 2) **KL\* 必须显式给出**（本包不内置默认值 —— 这是相对 SFT 的新增纪律）
  const ds = mkSource(join(tmp, 'src-ok'))
  const prep = await api.prepare({ root, sourceDatasetDir: ds, trainPrompts: 10, heldoutPrompts: 10 })
  t.assert(prep.trainPrompts === 2 && prep.heldoutPrompts === 1, `prepare 产出 train=${prep.trainPrompts} heldout=${prep.heldoutPrompts}`)
  t.assert(prep.leakCheck.ok === true && prep.leakCheck.hashOverlap === 0 && prep.leakCheck.substringContained === 0, 'prepare 的双轨去泄漏复核均为空')
  t.assert(prep.license?.spdx === 'cc-by-4.0', 'prepare 继承上游 license')
  t.assert(prep.reward?.kind === 'rule' && prep.reward?.version === 'rule-reward@1', 'prepare 回报奖励来源与版本')

  r = await attempts(() => api.train({ root, promptsDir: prep.promptsDir, baseModel: '<BASE_MODEL>' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_KL_BUDGET_REQUIRED, `2) 缺 klBudget → E_KL_BUDGET_REQUIRED（实际 ${r.code}）`)
  r = await attempts(() => api.train({ root, promptsDir: prep.promptsDir, baseModel: '<BASE_MODEL>', klBudget: 0 }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_KL_BUDGET_REQUIRED, '2b) klBudget=0 → 仍拒绝（β 会是 Infinity）')
  r = await attempts(() => api.train({ root, promptsDir: prep.promptsDir, baseModel: '<BASE_MODEL>', klBudget: -1 }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_KL_BUDGET_REQUIRED, '2c) klBudget<0 → 仍拒绝')

  // 3) 参数越界（步数/GPU 小时超上限）
  r = await attempts(() => api.train({ root, promptsDir: prep.promptsDir, baseModel: '<BASE_MODEL>', klBudget: 0.1, maxSteps: 9999 }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_BUDGET_EXCEEDED, `3) maxSteps=9999 → E_BUDGET_EXCEEDED（实际 ${r.code}）`)
  r = await attempts(() => api.train({ root, promptsDir: prep.promptsDir, baseModel: '<BASE_MODEL>', klBudget: 0.1, limits: { maxGpuHours: 5 } }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_BUDGET_EXCEEDED, '3b) maxGpuHours>2 → E_BUDGET_EXCEEDED')

  // 4) 基座模型不存在
  r = await attempts(() => api.train({ root, promptsDir: prep.promptsDir, baseModel: join(tmp, 'no-model'), klBudget: 0.1 }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_BASE_MODEL_NOT_FOUND, `4) 基座不存在 → E_BASE_MODEL_NOT_FOUND（实际 ${r.code}）`)

  // 5) 去泄漏闸门（双轨都在 prepare 里 fail-closed）
  const dsLeak = mkSource(join(tmp, 'src-leak'), { overlap: true })
  r = await attempts(() => api.prepare({ root, sourceDatasetDir: dsLeak, trainPrompts: 10, heldoutPrompts: 10 }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_PROMPT_LEAK, `5) 训练/留出有交集 → E_PROMPT_LEAK（实际 ${r.code}）`)
  t.assert(r.body?.details?.hashOverlap >= 1 && r.body?.details?.substringContained >= 1, `5b) 两条轨道都被检出（hash=${r.body?.details?.hashOverlap} substring=${r.body?.details?.substringContained}）`)

  // 5c) 直接测 leakCheck 的双轨语义（不经过 prepare 的封装）
  // 注意：两轨**不是互斥的**。完全相同 ⇒ 也满足"作为子串出现"，故两轨都报（fail-closed 的保守取向）。
  // ② 的独立价值在于抓 ① 抓不到的"部分包含"（见 5d）。
  const lc1 = leakCheck([{ prompt: 'a'.repeat(80) }], [{ prompt: 'a'.repeat(80) }])
  t.assert(lc1.ok === false && lc1.hashOverlap === 1 && lc1.substringContained === 1, '5c) 完全相同 ⇒ 两轨都命中（保守：完全相同也属于"作为子串出现"）')
  const lc2 = leakCheck([{ prompt: `${'a'.repeat(80)} tail` }], [{ prompt: 'a'.repeat(80) }])
  t.assert(lc2.ok === false && lc2.hashOverlap === 0 && lc2.substringContained === 1, '5d) 留出是训练的子串 ⇒ 子串轨命中（哈希轨为空）')
  const lc3 = leakCheck([{ prompt: 'short one' }], [{ prompt: 'short one' }])
  t.assert(lc3.substringContained === 0, '5e) 短于 64 字符的行不进子串轨（避免"短路包含"的假阳性）')

  // 6) 缺 license 的数据集不得用于 RL（不许凭空断言许可）
  const dsNoLic = mkSource(join(tmp, 'src-nolic'), { license: null })
  r = await attempts(() => api.prepare({ root, sourceDatasetDir: dsNoLic, trainPrompts: 5, heldoutPrompts: 5 }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_PROMPTS_INVALID, `6) 上游血缘缺 license → 拒绝（实际 ${r.code}）`)

  // 7) 未知 runId
  r = await attempts(() => api.getReport({ root, runId: 'rl_nope' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_RUN_NOT_FOUND, `7) 未知 runId → E_RUN_NOT_FOUND（实际 ${r.code}）`)
  r = await attempts(() => api.rollback({ root, runId: 'rl_nope' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_RUN_NOT_FOUND, '7b) rollback 未知 runId → E_RUN_NOT_FOUND')

  // 8) validate 的阈值必须显式给出（不内置默认值）
  r = await attempts(() => api.validate({ root, runId: 'rl_nope' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_MISSING_PARAM, `8) validate 缺 thresholds → E_MISSING_PARAM（实际 ${r.code}）`)
  r = await attempts(() => api.validate({ root, runId: 'rl_nope', thresholds: {} }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_MISSING_PARAM, '8b) thresholds.minRewardGainPct 缺失 → E_MISSING_PARAM')

  // 9) 未知工具名
  const { findToolSpec } = await import('../../src/tools.mjs')
  t.assert(findToolSpec('training_rl_nope') === null, '9) 未知工具名返回 null（MCP 侧据此回 -32601）')

  /* ═══════════ 安全验收 ═══════════ */
  r = await attempts(() => requireConfirm({ root, scope: 'train', payload: { a: 1 }, summary: 'x' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_CONFIRMATION_REQUIRED, 'S1) 缺令牌 → E_CONFIRMATION_REQUIRED')
  t.assert(typeof r.body?.details?.confirmToken === 'string' && r.body.details.confirmToken.startsWith('tr_'), 'S1b) 返回 confirmToken（本包前缀 tr_）')
  r = await attempts(() => requireConfirm({ root, scope: 'train', payload: { a: 1 }, confirm: 'tr_wrong', summary: 'x' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_CONFIRMATION_REQUIRED, 'S2) 错令牌 → 仍拒绝')
  const tok = tokenFor('train', { a: 1 })
  r = await attempts(() => requireConfirm({ root, scope: 'train', payload: { a: 1 }, confirm: tok, summary: 'x' }))
  t.assert(r.ok === true && r.value.confirmed === true, 'S3) 正确令牌 → 放行')
  r = await attempts(() => requireConfirm({ root, scope: 'train', payload: { a: 1 }, summary: 'x' }))
  t.assert(r.ok === true && r.value.source === 'memo', 'S3b) 24h 内同类操作免再问（memo）')
  t.assert(tokenFor('train', { a: 1 }) !== tokenFor('train', { a: 2 }), 'S3c) 令牌与目标绑定')
  t.assert(tokenFor('train', { a: 1 }) !== tokenFor('validate', { a: 1 }), 'S3d) 令牌与能力绑定')
  t.assert(tokenFor('train', { a: 1, confirm: 'x' }) === tokenFor('train', { a: 1 }), 'S3e) confirm 字段不进指纹（回传原值不改变目标）')

  // S4) rollback purge 只允许删本包 run 目录（越界删除被拦）
  await updateRegistry(root, (doc) => ({
    ...doc,
    runs: [{
      id: 'rl_x', status: 'trained', startedAt: 'x', baseModel: { path: '/b' }, dataset: { dir: '/d' },
      reward: { kind: 'rule', version: 'rule-reward@1' }, config: {}, budget: { maxSteps: 1, maxGpuHours: 1 }, runDir: '/etc',
    }],
  }), { schemaPath: null })
  r = await attempts(() => api.rollback({ root, runId: 'rl_x', purge: true }))
  t.assert(r.ok === true && r.value.purged.length === 0, 'S4) purge 拒绝删除 runs 根以外目录（/etc 未被删）')
  t.assert(existsSync('/etc'), 'S4b) /etc 仍在（越界删除被拦）')

  // S5) 确认令牌闸门在工具层生效（真调 train 但缺令牌 → 不会真的起容器）
  const { apply, TOOL_NAMES } = await import('../../src/index.mjs')
  const registered = []
  const ctx = {
    logger: { info() {}, warn() {}, error() {}, debug() {}, child() { return this } },
    effect(fn) { return fn() }, get() { return undefined }, on() { return () => {} },
    tools: { register(tool) { registered.push(tool); return () => {} } },
  }
  apply(ctx)
  t.assert(registered.length === TOOL_NAMES.length, 'S5) apply() 注册全部工具（用于令牌闸门的真调测试）')
  const trainTool = registered.find((x) => x.name === 'training_rl_train')
  const env = await trainTool.execute({ root, promptsDir: prep.promptsDir, klBudget: 0.1 })
  t.assert(env.ok === false && env.code === ERROR_CODES.E_CONFIRMATION_REQUIRED, `S5b) 工具层缺令牌 → E_CONFIRMATION_REQUIRED（实际 ${env.code}）`)
  t.assert(env.governance?.confirmationRequired === true && env.governance?.risk === 'write', 'S5c) 治理信封如实标注 risk/confirmationRequired')
  t.assert(typeof env.error?.details?.confirmToken === 'string', 'S5d) 治理信封里带着可回传的 confirmToken')

  /* ═══════════ 健壮性 ═══════════ */
  // R1) 状态文件损坏 → 抛错而不是静默重置
  mkdirSync(p.stateDir, { recursive: true })
  writeFileSync(p.registryFile, '{ not yaml ]')
  r = await attempts(() => readRegistry(root, null))
  t.assert(r.ok === false, 'R1) 损坏的状态文件 → 解析失败（不静默重置为默认值）')
  writeFileSync(p.registryFile, 'version: 1\npkg: "0.1.0"\nruns: []\n')
  const rr = await readRegistry(root, null)
  t.assert(rr.runs.length === 0, 'R1b) 修好后可正常读')

  // R2) 审计链被篡改 → 检出
  const a1 = await audit(root, { capability: 'x', action: 'y', phase: 'end' })
  const shard = join(p.auditDir, `${new Date().toISOString().slice(0, 10)}.jsonl`)
  const lines = readFileSync(shard, 'utf8').split('\n').filter(Boolean)
  const bad = JSON.parse(lines[0]); bad.phase = 'TAMPERED'
  writeFileSync(shard, [JSON.stringify(bad), ...lines.slice(1)].join('\n') + '\n')
  t.assert((await auditVerify(root)).ok === false, 'R2) 审计链被篡改 → verifyAllChains 报 ok=false')
  writeFileSync(shard, `${lines.join('\n')}\n`)
  t.assert((await auditVerify(root)).ok === true, 'R2b) 恢复后链重新通过')
  t.assert(a1.hash.length === 64, 'R2c) 审计哈希是 64 位 sha256')
  appendFileSync(shard, '\n')

  // R3) schema 校验拦截非法状态（用本包自己的 rl-registry schema）
  const RL_SCHEMA = join(HERE, '..', '..', '..', '..', 'inference-contracts', 'state-schema', 'rl-registry.schema.json')
  if (existsSync(RL_SCHEMA)) {
    const badRec = await attempts(() => updateRegistry(root, (doc) => ({ ...doc, runs: [{ id: 'nope', status: 'running' }] }), { schemaPath: RL_SCHEMA }))
    t.assert(badRec.ok === false && badRec.code === ERROR_CODES.E_STATE_CONFLICT, `R3) 非法 run 记录（id 不以 rl_ 开头）被 schema 拦截（实际 ${badRec.code}）`)
    const badStatus = await attempts(() => updateRegistry(root, (doc) => ({
      ...doc,
      runs: [{ id: 'rl_ok', status: 'definitely-not-a-status', startedAt: 'x', baseModel: { path: '/b' }, dataset: { dir: '/d' }, config: {}, budget: { maxSteps: 1, maxGpuHours: 1 } }],
    }), { schemaPath: RL_SCHEMA }))
    t.assert(badStatus.ok === false, 'R3b) 非法 status 被 schema 拦截')
    const good = await attempts(() => updateRegistry(root, (doc) => ({
      ...doc,
      runs: [{
        id: 'rl_ok', status: 'trained', startedAt: 'x', baseModel: { path: '/b' }, dataset: { dir: '/d' },
        reward: { kind: 'rule', version: 'rule-reward@1' }, config: {}, budget: { maxSteps: 1, maxGpuHours: 1 },
        beta: 0.25, kl: { budget: 0.1, formula: 'beta = sigmaR / klBudget', klMean: 0.002 },
        calibration: { sigmaR: 0.025, beta: 0.25, samples: 32 },
      }],
    }), { schemaPath: RL_SCHEMA }))
    t.assert(good.ok === true, `R3c) 合法状态（含 beta/kl/calibration）通过 schema（实际 ${good.code ?? ''} ${good.message ?? ''}）`)
  } else t.assert(false, 'R3) 找不到 rl-registry schema')

  // R4) CAS 冲突
  const cur = await readRegistry(root, null)
  r = await attempts(() => updateRegistry(root, (d) => d, { expectedHash: 'sha256:deadbeef' }))
  t.assert(r.ok === false && r.code === ERROR_CODES.E_STATE_CONFLICT, 'R4) 过期 expectedHash → ERR_STATE_CONFLICT')
  t.assert(cur._hash?.startsWith('sha256:'), 'R4b) 读出的状态带内容哈希（CAS 可用）')

  // R5) worker 协议层（**不依赖 docker** —— CI 里不跑容器；真容器测试在真机 run 里）
  const w = new ContainerWorker({ image: 'alpine:3.20', mounts: ['/a:/b:ro'], gpus: true, entry: '/work/rl_trainer.py' })
  r = await attempts(() => w.call('ping', {}, { timeoutMs: 1000 }))
  t.assert(r.ok === false && /未启动/.test(String(r.message)), 'R5) 未 start 就 call → 响亮失败（不会静默挂起）')
  r = await attempts(() => w.stream('run', {}, { onEvent: async () => ({}) }))
  t.assert(r.ok === false && /未启动/.test(String(r.message)), 'R5b) 未 start 就 stream → 同样响亮失败')
  r = await attempts(() => new ContainerWorker({ image: 'x' }).start().stream('run', {}, {}))
  t.assert(r.ok === false && /onEvent/.test(String(r.message)), 'R5c) stream 缺 onEvent → 响亮失败（不做静默降级）')
  const argv = w.args()
  t.assert(argv[0] === 'run' && argv.includes('--rm') && argv.includes('-i'), 'R5d) 容器参数含 --rm/-i（不残留容器）')
  t.assert(argv.includes('--gpus') && argv[argv.indexOf('--gpus') + 1] === 'all', 'R5e) 显式声明 --gpus all')
  t.assert(argv.includes('--user') && /^\d+:\d+$/.test(argv[argv.indexOf('--user') + 1]), 'R5f) 以宿主 uid:gid 运行（否则容器产出的权重宿主改不动）')
  t.assert(argv.includes('USER=' + (process.env.USER || 'x')) || argv.some((a) => a.startsWith('USER=')), 'R5g) 显式给 USER（非 root uid 缺 /etc/passwd 条目会让 getpass 抛 KeyError）')
  t.assert(argv.some((a) => a.startsWith('LOGNAME=')), 'R5h) 显式给 LOGNAME')
  t.assert(argv.includes('/a:/b:ro'), 'R5i) 挂载原样传递（只读挂载由调用方保证）')
  t.assert(argv[argv.length - 1] === '/work/rl_trainer.py', 'R5j) 入口脚本是最后一个参数')
  await w.stop()

  // R6) 端点不可达 → 评估返回 0 样本而不是伪造数字
  const ep = await evalPreference({ endpoint: 'http://127.0.0.1:1', model: 'm', rows: [{ prompt: 'x' }], limit: 1 })
  t.assert(ep.nOk === 0 && ep.meanReward === null, 'R6) 端点不可达 → meanReward=null（不伪造 0 或 1）')

  /* ═══════════ 门禁与选点（判据来自 testing-core / 冻结参考实现） ═══════════ */
  // G1) 偏好对齐：显著 + 幅度达标 ⇒ 过
  // 注意：测量必须带 `n` 与 `truncated` —— 判据里的**截断率闸门**需要它们（见 evaluate.mjs 的 EVAL_PROTOCOL@2）。
  // 缺字段 ⇒ 无法判定截断率 ⇒ 按"测量不成立"处理（不是"没问题"）。
  const basePref = { meanReward: 0.40, rewards: Array.from({ length: 40 }, (_, i) => 0.4 + (i % 3) * 0.01), n: 40, truncated: 4 }
  const rlPref = { meanReward: 0.52, rewards: basePref.rewards.map((v, i) => v + 0.12 + (i % 5) * 0.001), n: 40, truncated: 3 }
  const pg1 = preferenceGate({ base: basePref, policy: rlPref, thresholds: { minRewardGainPct: 10 }, validity: [{ name: 'v', ok: true }] })
  t.assert(pg1.pass === true && pg1.pairedSignificance.pass === true, 'G1) 显著且幅度达标 ⇒ 偏好门禁通过')
  t.assert(pg1.pairedSignificance.winRate === 1, 'G1b) 40/40 胜出 ⇒ winRate=1')
  // G2) 幅度达标但不显著（噪声大于提升）⇒ 不过（"提升"不得被噪声冒充）
  const rlNoisy = { meanReward: 0.401, rewards: basePref.rewards.map((v, i) => v + (i % 2 === 0 ? 0.05 : -0.048)), n: 40, truncated: 4 }
  const pg2 = preferenceGate({ base: basePref, policy: rlNoisy, thresholds: { minRewardGainPct: 0.1 }, validity: [{ name: 'v', ok: true }] })
  t.assert(pg2.pairedSignificance.pass === false && pg2.pass === false, 'G2) 提升落在噪声内 ⇒ 配对显著性判失败')
  // G3) 测量不成立与数值未达标可区分
  const pg3 = preferenceGate({ base: basePref, policy: rlPref, thresholds: { minRewardGainPct: 10 }, validity: [{ name: 'v', ok: false, detail: 'x' }] })
  t.assert(pg3.pass === false && pg3.validityOk === false && /测量不成立/.test(pg3.reason), 'G3) 测量不成立与数值未达标可区分')
  // G4) 样本不足 ⇒ **未通过**（不是"没问题"）
  const pg4 = preferenceGate({ base: { meanReward: null, rewards: [], n: 0, truncated: 0 }, policy: { meanReward: null, rewards: [], n: 0, truncated: 0 }, thresholds: { minRewardGainPct: 1 }, validity: [] })
  t.assert(pg4.pass === false, 'G4) 配对样本为 0 ⇒ 门禁不通过')
  // G4b) **截断率闸门**（真机第一版 validate 因 47/48 截断而失去区分度，故这条判据是那次的产物）
  const pgTrunc = preferenceGate({ base: { ...basePref, truncated: 39 }, policy: { ...rlPref, truncated: 39 }, thresholds: { minRewardGainPct: 1 }, validity: [{ name: 'v', ok: true }] })
  t.assert(pgTrunc.pass === false && pgTrunc.validityOk === false && /测量不成立/.test(pgTrunc.reason), 'G4b) 截断率超上限 ⇒ 判「测量不成立」（不是"没达标"）')
  t.assert(pgTrunc.checks.some((c) => c.name === 'not-truncated-dominated(base)' && c.ok === false), 'G4b) validity 里如实列出截断率判据')
  // G4c) 缺 n/truncated 字段 ⇒ 同样判「测量不成立」（不把"无法断言"当"没问题"）
  const pgUnknown = preferenceGate({ base: { meanReward: 0.4, rewards: [0.4, 0.5] }, policy: { meanReward: 0.5, rewards: [0.5, 0.6] }, thresholds: { minRewardGainPct: 1 }, validity: [] })
  t.assert(pgUnknown.pass === false && pgUnknown.validityOk === false, 'G4c) 缺截断率信息 ⇒ 测量不成立（fail-closed）')

  // G5-G7) 遗忘门禁
  const fg1 = forgettingGate({
    baseMmlu: { n: 100, accuracy: 0.25 }, rlMmlu: { n: 100, accuracy: 0.25 },
    baseLm: { perText: Array.from({ length: 40 }, () => 2.0), meanNll: 2.0, ppl: Math.exp(2) },
    rlLm: { perText: Array.from({ length: 40 }, () => 2.0), meanNll: 2.0, ppl: Math.exp(2) },
  })
  t.assert(fg1.pass === true, 'G5) 无退化 ⇒ 遗忘门禁通过')
  const fg2 = forgettingGate({
    baseMmlu: { n: 100, accuracy: 0.60 }, rlMmlu: { n: 100, accuracy: 0.10 },
    baseLm: { perText: Array.from({ length: 40 }, () => 2.0), meanNll: 2.0, ppl: Math.exp(2) },
    rlLm: { perText: Array.from({ length: 40 }, () => 2.0), meanNll: 2.0, ppl: Math.exp(2) },
  })
  t.assert(fg2.pass === false && /MMLU/.test(fg2.reasons.join(' ')), 'G6) 显著退化 ⇒ 拒绝，且原因指明是 MMLU')
  const fg3 = forgettingGate({ baseMmlu: { n: 0, accuracy: null }, rlMmlu: { n: 0, accuracy: null }, baseLm: {}, rlLm: {} })
  t.assert(fg3.pass === false && fg3.reasons.length >= 2, 'G7) 样本不足 ⇒ **不通过**（不是"没问题"）')

  // O4) 选点规则：val loss argmin（保序变换后），并列取 payloadHash 字典序小者
  const sel1 = await selectCheckpoint({
    runDir: tmp,
    done: {
      result: { steps: 30 },
      evaluations: [
        { step: 10, split: 'train', metric: 'meanRuleReward', value: 0.5, selectionOnly: true },
        { step: 10, split: 'val', metric: 'meanRuleReward', value: 0.40, selectionOnly: true },
        { step: 20, split: 'train', metric: 'meanRuleReward', value: 0.6, selectionOnly: true },
        { step: 20, split: 'val', metric: 'meanRuleReward', value: 0.55, selectionOnly: true },
        { step: 30, split: 'train', metric: 'meanRuleReward', value: 0.7, selectionOnly: true },
        { step: 30, split: 'val', metric: 'meanRuleReward', value: 0.50, selectionOnly: true },
      ],
    },
    checkpointEvery: 10, maxSteps: 30,
  })
  t.assert(sel1.selectionRule === 'val-loss-argmin' && sel1.selected.step === 20, `O4) 选到 val 奖励最高（= val loss 最低）的 step=20（实际 ${sel1.selected.step}）`)
  t.assert(sel1.curve.selectionOnly === true && sel1.curve.valLoss[1] === -0.55, 'O4b) 曲线标 selectionOnly，且 valLoss = −meanRuleReward（保序变换）')
  t.assert(sel1.selected.isFinalStep === false, 'O4c) 选中早期 checkpoint ⇒ 需要重新合并（validate 侧会做）')

  const sel2 = await selectCheckpoint({ runDir: tmp, done: { result: { steps: 30 }, evaluations: [] }, checkpointEvery: 10, maxSteps: 30 })
  t.assert(sel2.selectionRule === 'final-no-val-signal' && sel2.selected.step === 30, 'O4d) 没有 val 信号 ⇒ 用最终权重且**不做选点声明**')
  t.assert(sel2.selected.selectionOnly === false, 'O4e) 没有选点信号时不冒充 selectionOnly')

  const sel3 = await selectCheckpoint({ runDir: tmp, done: { result: { steps: 10 }, evaluations: [{ step: 10, split: 'val', metric: 'meanRuleReward', value: 0.3, selectionOnly: true }] }, checkpointEvery: 10, maxSteps: 10 })
  t.assert(sel3.selected.isFinalStep === true, 'O4f) 选中最终步 ⇒ 可直接复用 train 时的合并产物')

  rmSync(tmp, { recursive: true, force: true })
}
