#!/usr/bin/env node
/**
 * assemble-evidence —— 把**多次真机 run**的原始证据合成一份最终证据文件
 * （`inference-training/out/rl-vertical-loop.json`），供契约 §10 引用。
 *
 * ## 为什么要"合成"而不是只留最后一次
 * 真机过程里发生了三件事，每一件都必须留在账上：
 *   ① Run 1（β 由声明的 KL*=0.1 推出 = 0.8258）→ 训练通过、**P1 未通过**；
 *      但它的第一次 validate 用的是 `max_tokens=64` 的仪器（47/48 条生成被截断 ⇒ 判据无区分度），
 *      这属于**测量不成立**，必须与"没达标"分开记。
 *   ② `rl-pref-eval@2`（`max_tokens=256`）重判 Run 1 时，策略侧测完了，
 *      基座侧部署两次都撞上 `E_DEGRADE_EXHAUSTED`（与本机既有服务/并行 workstream 抢内存）——
 *      这是**基础设施失败**，不是判定，必须如实标成 `infrastructure-failed`，不能算作"未通过"。
 *   ③ Run 2（β 用**修正后的一阶控制量** 0.0097，即 KL*=8.5）→ 用于区分
 *      "β 造成的" 与 "奖励信号/步长规模造成的"。
 *
 * 用法：node tests/realmachine/assemble-evidence.mjs [--out <path>]
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const OUT = join(PKG, '..', 'out')

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }
const TARGET = argOf('--out', join(OUT, 'rl-vertical-loop.json'))

const read = (f) => {
  const p = join(OUT, f)
  if (!existsSync(p)) return null
  try { return JSON.parse(readFileSync(p, 'utf8')) } catch { return null }
}

const run1 = read('rl-vertical-loop.maxtok64.json')
const v2 = read('rl-vertical-loop.v2.json')
const run2 = read('rl-vertical-loop.v3.json')

if (!run1 && !run2) {
  process.stderr.write('既没有 run1 也没有 run2 的证据文件 —— 不能合成\n')
  process.exit(1)
}

/** 训练侧的"奖励是否真的被推上去"（用 metrics.jsonl 的 first10/last10，而不是单点）。 */
function rewardTrajectory(runDir) {
  const f = join(runDir ?? '', 'metrics.jsonl')
  if (!runDir || !existsSync(f)) return null
  const rows = readFileSync(f, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
  const rew = rows.map((r) => r.reward).filter((x) => typeof x === 'number')
  const kl = rows.map((r) => r.kl).filter((x) => typeof x === 'number')
  const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null)
  const last10 = mean(rew.slice(-10))
  const first10 = mean(rew.slice(0, 10))
  const perBlock = [0, 1, 2, 3].map((i) => mean(rew.slice(i * 50, (i + 1) * 50)))
  return {
    steps: rows.length,
    rewardFirst10: first10, rewardLast10: last10, rewardDeltaFirstLast10: (last10 !== null && first10 !== null) ? last10 - first10 : null,
    rewardDeltaPct: (first10 ? ((last10 - first10) / first10) * 100 : null),
    rewardBlocks50: perBlock,
    klMean: mean(kl), klMax: kl.length ? Math.max(...kl) : null, klLast10: mean(kl.slice(-10)),
  }
}

const summarizeRun = (ev, label) => {
  if (!ev) return { label, present: false }
  const t = ev.train ?? {}
  const v = ev.validate ?? {}
  const vd = v?.preferenceGate?.deltas?.meanReward ?? null
  return {
    label,
    present: true,
    ok: ev.ok === true,
    runId: t.runId ?? ev.runId ?? null,
    klBudget: ev.inputs?.klBudget ?? null,
    steps: t.steps ?? null,
    gpuHours: t.gpuHours ?? ev.gpuHours ?? null,
    calibration: t.calibration ?? null,
    kl: t.kl ? { mean: t.kl.mean, min: t.kl.min, max: t.kl.max, deviationPct: t.kl.deviationPct, withinTolerance: t.kl.withinTolerance } : null,
    beta: t.calibration?.beta ?? null,
    betaRecommendation: t.kl?.recommendation ?? null,
    fingerprints: t.fingerprints ?? null,
    artifactAssertion: t.artifactAssertion ?? null,
    budgetAssessment: t.budgetAssessment ?? null,
    rewardTrajectory: rewardTrajectory(t.runDir),
    rewardConformance: t.rewardConformance ?? null,
    selection: t.selection ? { selectionRule: t.selection.selectionRule, selected: t.selection.selected, curve: t.selection.curve } : null,
    validate: v && v.pass !== undefined ? {
      pass: v.pass,
      preferenceGate: v.preferenceGate,
      forgettingGate: v.forgettingGate,
      overfit: v.overfit,
      base: v.base ?? null, policy: v.policy ?? null,
      error: null,
    } : (ev.error ? { pass: null, infrastructureFailed: true, error: ev.error } : null),
    validateDeltas: vd,
    error: ev.error ?? null,
  }
}

const out = {
  schema: 'ai-infra/rl-vertical-loop@1',
  generatedAt: new Date().toISOString(),
  mode: run2 ? 'main（β 修正后重跑）' : 'main',
  ok: run2 ? run2.ok === true : (run1?.ok === true),
  preregistration: {
    writtenBeforeRun2: true,
    predictions: [
      '预测 1：训练侧规则奖励仍会上升 ≥ +0.02（first10 → last10）——run1 实测 +0.026',
      '预测 2：留出侧偏好提升仍 < 2% 且不显著 ⇒ **β 不是约束**（若变显著则 β 是约束，读法作废）',
      '预测 3：O3 的 train↔val 间隙比 run1 更大（β 变小 ⇒ 漂移更多 ⇒ 间隙更大）',
    ],
    betaTarget: 0.0097,
    klBudgetChosen: 8.5,
    why: 'β_next = β_declared × klMean/KL* = 0.8258130286540805 × 0.0011740167962852866 / 0.1 ≈ 0.009695（修正后的一阶控制量）；通过 KL* = σ_r/β_next ≈ 0.0826/0.0097 ≈ 8.5 走**同一个推导**得到 β≈0.0097，而不是硬编码 β。',
  },
  // 顶层 = 最终判定所用的那一次（供 fill-contract-evidence.mjs 直接读）
  train: run2?.train ?? run1?.train ?? null,
  validate: run2?.validate ?? null,
  gpuHours: Number(((run1?.gpuHours ?? 0) + (run2?.gpuHours ?? 0))) || null,
  inputs: run2?.inputs ?? run1?.inputs ?? null,
  stages: run2?.stages ?? run1?.stages ?? [],
  error: run2?.error ?? run1?.error ?? null,
  runs: [
    summarizeRun(run1, 'run1（β 由声明 KL*=0.1 推出 → 0.8258）'),
    summarizeRun(run2, 'run2（β 由修正后的一阶控制量 → 0.0097，即 KL*=8.5）'),
  ].filter((x) => x.present),
  validateHistory: [],
}

// 判定历史：把三次判定（含基础设施失败）逐条列出，不合并、不美化
const h = []
if (run1?.validate) {
  h.push({
    instrument: 'rl-pref-eval@1（max_tokens=64）',
    run: 'run1', verdict: run1.validate.pass === true ? 'pass' : 'fail',
    note: '47/48 条生成被截断（finish_reason=length）⇒ 判据几乎全并列（27/48 ties）⇒ **测量不成立**，不是"没达标"',
    preferenceGate: run1.validate.preferenceGate,
  })
}
for (const [label, ev] of [['run1 重判(rl-pref-eval@2)', v2], ['run2 判定(rl-pref-eval@2)', run2]]) {
  if (!ev) continue
  if (ev.validate && ev.validate.pass !== undefined) {
    h.push({ instrument: 'rl-pref-eval@2（max_tokens=256）', run: label, verdict: ev.validate.pass === true ? 'pass' : 'fail', preferenceGate: ev.validate.preferenceGate, forgettingGate: ev.validate.forgettingGate, overfit: ev.validate.overfit })
  } else if (ev.error) {
    h.push({ instrument: 'rl-pref-eval@2（max_tokens=256）', run: label, verdict: 'infrastructure-failed', note: `基座/策略端点部署撞上 E_DEGRADE_EXHAUSTED（与既有服务/并行 workstream 抢内存）——**不是判定**`, error: ev.error })
  }
}
out.validateHistory = h

writeFileSync(TARGET, `${JSON.stringify(out, null, 2)}\n`)
process.stdout.write(`${JSON.stringify({
  wrote: TARGET, ok: out.ok, runs: out.runs.map((r) => ({ label: r.label, beta: r.beta, klMean: r.kl?.mean ?? null, trainRewardDelta: r.rewardTrajectory?.rewardDeltaFirstLast10 ?? null, validation: r.validate ? (r.validate.pass === true ? 'PASS' : r.validate.pass === false ? 'FAIL' : 'infra-failed') : null })), validateHistory: out.validateHistory.map((x) => ({ run: x.run, verdict: x.verdict })),
}, null, 2)}\n`)
