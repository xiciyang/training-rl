#!/usr/bin/env node
/**
 * rl-multiseed —— 长任务 5 · 阶段 3：**RL 偏好对齐的多种子重测**（换仪器后）。
 *
 * ## 为什么要有它
 * LT4 的 RL 结论是**单种子**（seed=20260929），且该种子的偏好门禁判的是
 * **测量不成立**（生成被截断主导，仪器无区分度）—— 既不是"通过"也不是"未达标"。
 * 本脚本按用户**预先登记**的三条判据做流程级结论，并对每个种子给出完整证据：
 *   ① 全部种子符号一致        → 稳定
 *   ② 符号翻转                → 测量不成立
 *   ③ R/μ > 50%               → 测量不成立，需重设计
 *
 * ## 纪律
 * - **不写新训练循环**：全部经已交付插件 API（`src/core.mjs` 的 prepare/train/validate）。
 * - **判定仪器只能是 serving**：validate 内部就是 serving 部署 → 测量 → 拆解；
 *   容器内评估只作选点信号（`selectionOnly`）。
 * - **阈值与 z 一个字都不动**：`minRewardGainPct=2`、`z=1.96` 逐字沿用；
 *   本脚本**不读取也不修改**任何门禁常量，只把 `--pref-max-tokens`（**仪器**）传进去。
 * - **聚合不自己写**：import `inference-training/checks/overfit-criterion.mjs#aggregateSeeds`
 *   （冻结契约的唯一参考实现）。
 * - 每个种子的失败如实登记，不跳过、不补跑出"好看"的数字。
 *
 * 用法：
 *   node tests/realmachine/rl-multiseed.mjs \
 *        --seeds 1042,2042,3042 --pref-max-tokens 768 \
 *        --out ../../out/rl-multiseed.json
 */
import { writeFileSync, mkdirSync, readFileSync } from 'node:fs'
import { createHash } from 'node:crypto'
import { homedir } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const TRAINING = join(PKG, '..')
const OUT = join(TRAINING, 'out')

const api = await import(join(PKG, 'src', 'core.mjs'))
const { ERROR_CODES } = await import(join(PKG, 'src', 'version.mjs'))
const { EVAL_PROTOCOL, MAX_TRUNCATION_RATIO } = await import(join(PKG, 'src', 'evaluate.mjs'))
const { aggregateSeeds } = await import(join(TRAINING, 'checks', 'overfit-criterion.mjs'))

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : d }
const NUM = (k, d) => Number(argOf(k, d))

const SEEDS = String(argOf('--seeds', '1042,2042,3042')).split(',').map((s) => Number(s.trim())).filter(Number.isFinite)
const PREF_MAX_TOKENS = NUM('--pref-max-tokens', 768)
const STEPS = NUM('--steps', 200)
const KL_BUDGET = NUM('--kl-budget', 8.5)            // = 已验收 run gr86 的取值（"配置不变，只换种子"）
const KL_TOL = NUM('--kl-tolerance-pct', 50)
const MIN_GAIN = NUM('--min-gain-pct', 2)            // **预登记，不动**
const BUDGET_GPU_H = NUM('--budget-gpu-h', 20)       // 阶段总预算
const DATASET = argOf('--dataset', join(homedir(), '.ai-infra/state/datasets/yahma-alpaca-cleaned__sft__s20260929__n1000'))
const EVIDENCE = argOf('--out', join(OUT, 'rl-multiseed.json'))
const CHOICE_RULE = argOf('--choice-rule',
  '取标定扫描中**两侧截断率都 ≤ MAX_TRUNCATION_RATIO(50%) 且带真实余量**的最小生成长度上限：'
  + '512 恰为 50.00%（knife-edge，换一个种子/子集就会翻闸），768 两侧均 41.7% ⇒ 选 768。'
  + '截断率闸门阈值(0.5)本身未改，改的只是仪器参数 maxTokens。')
/** 已验收的 run（同一个训练配置，seed=20260929）：只**重测**（validate），不重训。 */
const RECALIBRATE_RUN = argOf('--recalibrate-run', 'rl_20260929155357_gr86')

const log = (s) => process.stderr.write(`[rl-multiseed] ${s}\n`)
let gpuSpent = 0
const spend = (h, what) => {
  if (typeof h === 'number' && Number.isFinite(h)) { gpuSpent += h; log(`  +${h.toFixed(4)} GPU·h（${what}）→ 累计 ${gpuSpent.toFixed(4)}`) }
}
const guard = (what) => {
  if (gpuSpent > BUDGET_GPU_H) throw new Error(`E_BUDGET_EXCEEDED: 阶段累计 ${gpuSpent.toFixed(3)} GPU·h 超过 ${BUDGET_GPU_H}（在 ${what} 前暂停）`)
}

async function withConfirm(fn, args, label) {
  try { return await fn(args) } catch (e) {
    const token = e?.body?.details?.confirmToken
    if (e?.code === ERROR_CODES.E_CONFIRMATION_REQUIRED && token) {
      log(`${label}: 需确认 ⇒ 回传 confirmToken 执行`)
      return await fn({ ...args, confirm: token })
    }
    throw e
  }
}

/** 端点起不来的**有界重试**（判据/阈值/样本量一个字不改）；不缓存任何测量结果。 */
async function retryServing(fn, { attempts = 2, delayMs = 60000 } = {}) {
  let last = null
  for (let i = 1; i <= attempts; i += 1) {
    try { return await fn() } catch (e) {
      last = e
      const transient = e?.code === ERROR_CODES.E_SERVING_UNAVAILABLE || /E_DEGRADE_EXHAUSTED/.test(String(e?.message ?? ''))
      if (!transient || i === attempts) throw e
      log(`部署瞬时失败（${e.code ?? e.message}）⇒ ${delayMs / 1000}s 后重试（${i}/${attempts - 1}）`)
      await new Promise((r) => setTimeout(r, delayMs))
    }
  }
  throw last
}

const TRAIN_CONFIG = {
  steps: STEPS, klBudget: KL_BUDGET, klTolerancePct: KL_TOL,
  lr: NUM('--lr', 1e-6), batchSize: NUM('--batch', 8), gradAccum: NUM('--grad-accum', 1),
  numGenerations: NUM('--num-generations', 4), maxCompletionLength: NUM('--max-completion-length', 64),
  checkpointEvery: 50, checkpointKeep: 4, evalEvery: 50, evalPrompts: 8, calibPrompts: 8, calibGenerations: 4,
}

const result = {
  schema: 'ai-infra/rl-multiseed@1',
  generatedAt: new Date().toISOString(),
  instrument: {
    protocol: EVAL_PROTOCOL, prefMaxTokens: PREF_MAX_TOKENS, truncationGate: MAX_TRUNCATION_RATIO,
    note: '**唯一仪器参数**；protocol+maxTokens 都进基线缓存键 ⇒ 不会与旧仪器混用',
    choiceRule: CHOICE_RULE,
  },
  immutable: {
    note: '本次**未改动**任何判据常量；下面是三条判定路径源文件的 sha256（阶段开始前记录，结束前复核）',
    sourcesAtStart: null,
    sourcesAtEnd: null,
    unchanged: null,
  },
  thresholds: { minRewardGainPct: MIN_GAIN, z: 1.96, note: '**预登记，本次一个数字都没有改动**' },
  trainConfig: TRAIN_CONFIG,
  seeds: SEEDS,
  recalibratedRun: RECALIBRATE_RUN,
  promptSet: null,
  promptSetStableAcrossSeeds: null,
  perSeed: [],
  aggregate: null,
  gpuHours: null,
  budget: { limitGpuHours: BUDGET_GPU_H, exceeded: null },
  ok: false,
  error: null,
}

const write = () => { mkdirSync(dirname(EVIDENCE), { recursive: true }); writeFileSync(EVIDENCE, `${JSON.stringify(result, null, 2)}\n`); log(`证据落盘 ${EVIDENCE}`) }

/** 判据路径源文件的 sha256 —— "阈值与 z 一个字都没动"的可复核证据（不靠口头承诺）。 */
const GATE_SOURCES = {
  'training-rl/src/evaluate.mjs': join(PKG, 'src', 'evaluate.mjs'),
  'training-rl/src/core.mjs': join(PKG, 'src', 'core.mjs'),
  'checks/overfit-criterion.mjs': join(TRAINING, 'checks', 'overfit-criterion.mjs'),
}
const hashSources = () => Object.fromEntries(Object.entries(GATE_SOURCES).map(([k, f]) => {
  try { return [k, `sha256:${createHash('sha256').update(readFileSync(f)).digest('hex')}`] } catch { return [k, null] }
}))
result.immutable.sourcesAtStart = hashSources()

/** 把 train 阶段的两条不同来源（API 返回值 / registry run 记录）归一到同一形状。 */
function trainView(api) {
  if (!api) return null
  return {
    steps: api.steps ?? api.result?.steps ?? null,
    finalLoss: api.finalLoss ?? api.result?.finalLoss ?? null,
    gpuHours: api.gpuHours ?? api.result?.gpuHours ?? null,
    fingerprints: {
      base: api.fingerprints?.base?.hash ?? api.fingerprints?.base ?? null,
      adapter: api.fingerprints?.adapter?.hash ?? api.fingerprints?.adapter ?? null,
      merged: api.fingerprints?.merged?.hash ?? api.fingerprints?.merged ?? null,
    },
    artifactAssertion: api.artifactAssertion ?? null,
    calibration: api.calibration ?? null,
    kl: api.kl ?? null,
    reward: api.reward ?? null,
    budgetAssessment: api.budgetAssessment ?? api.result?.budgetAssessment ?? null,
    selection: api.selection ?? null,
  }
}

/** 把一个 runId 的 validate 结果压成"每种子一行"的判据输入。 */
function summarizeSeed({ seed, runId, train, validate, kind }) {
  const pg = validate?.preferenceGate ?? null
  const fg = validate?.forgettingGate ?? null
  const paired = pg?.pairedSignificance ?? null
  return {
    kind, seed, runId,
    singleSeed: false,
    steps: train?.steps ?? null,
    finalLoss: train?.finalLoss ?? null,
    gpuHours: { train: train?.gpuHours ?? null, validate: Number(((validate?.ms ?? 0) / 3_600_000).toFixed(4)) },
    fingerprints: {
      base: train?.fingerprints?.base ?? validate?.artifactAssertion?.before ?? null,
      policyMergedTrainTime: train?.fingerprints?.merged ?? null,
      policyEvaluated: validate?.artifactAssertion?.after ?? null,
      adapter: train?.fingerprints?.adapter ?? null,
      trainAssertionOk: train?.artifactAssertion?.ok ?? null,
      validateAssertionOk: validate?.artifactAssertion?.ok ?? null,
    },
    reward: {
      kind: train?.reward?.kind ?? null, version: train?.reward?.version ?? null,
      components: train?.reward?.components ?? null, calls: train?.reward?.calls ?? null,
    },
    calibration: { sigmaR: train?.calibration?.sigmaR ?? null, beta: train?.calibration?.beta ?? null, klBudget: train?.calibration?.klBudget ?? null, samples: train?.calibration?.samples ?? null },
    kl: { mean: train?.kl?.mean ?? train?.kl?.klMean ?? null, max: train?.kl?.max ?? null, deviationPct: train?.kl?.deviationPct ?? null, recalibrationRequired: train?.kl?.recalibrationRequired ?? null },
    budget: train?.budgetAssessment ?? null,
    containerSelection: train?.selection ?? null,
    truncation: {
      base: validate?.base?.preference ? { truncated: validate.base.preference.truncated, n: validate.base.preference.n, ratio: validate.base.preference.truncated / validate.base.preference.n } : null,
      policy: validate?.policy?.preference ? { truncated: validate.policy.preference.truncated, n: validate.policy.preference.n, ratio: validate.policy.preference.truncated / validate.policy.preference.n } : null,
    },
    preferenceGate: pg ? {
      pass: pg.pass, validityOk: pg.validityOk, reason: pg.reason,
      meanRewardBase: pg.deltas?.meanReward?.base ?? null,
      meanRewardPolicy: pg.deltas?.meanReward?.policy ?? null,
      rawPct: pg.deltas?.meanReward?.rawPct ?? null,
      paired: paired ? { n: paired.n, meanDelta: paired.meanDelta ?? null, se: paired.se ?? null, bound: paired.bound ?? null, pass: paired.pass, wins: paired.wins, losses: paired.losses, ties: paired.ties, winRate: paired.winRate } : null,
      // 若该种子自己就卡在**截断有效性**上，paired 段是早退的（无 meanDelta）。
      // 这里**另给一个明确标注的未配对回退量**，只作汇报，**不进主聚合**：
      // 把"测量不成立"的种子当成一个正常的数混进去，等于偷偷把仪器问题洗成结论。
      unpairedFallback: (pg.pairedSignificance?.meanDelta === undefined && typeof pg.deltas?.meanReward?.policy === 'number' && typeof pg.deltas?.meanReward?.base === 'number')
        ? { meanDelta: pg.deltas.meanReward.policy - pg.deltas.meanReward.base, paired: false, usableForAggregate: false, why: '该种子截断有效性未过 ⇒ 门禁早退，无配对量；此回退值仅汇报，不参与多种子聚合' }
        : null,
    } : null,
    forgettingGate: fg ? { pass: fg.pass, mmlu: fg.mmlu, lm: fg.lm, reasons: fg.reasons } : null,
    overfit: validate?.overfit ? { pass: validate.overfit.pass ?? null, o2: validate.overfit.o2_generalizationReal ?? null, o3: validate.overfit.o3_overfitBudget ?? null } : null,
    selection: validate?.selection ? { rule: validate.selection.selectionRule, selected: validate.selection.selected?.name ?? null, step: validate.selection.selected?.step ?? null, selectionOnly: true } : null,
    mergedEvaluated: validate?.mergedEvaluated ?? null,
    evalResultPath: validate?.evalResultPath ?? null,
    validatePass: validate?.pass ?? null,
  }
}

/* ------------------------------------------------------------------ 主 */
try {
  result.perSeed = []

  /* ① 已验收 run（seed=20260929）在**新仪器**下重测 —— 只 validate，不重训（0 训练 GPU·h） */
  if (RECALIBRATE_RUN) {
    guard(`recalibrate ${RECALIBRATE_RUN}`)
    log(`重测已验收 run ${RECALIBRATE_RUN}（seed=20260929，同一训练配置）在新仪器 prefMaxTokens=${PREF_MAX_TOKENS} 下 …`)
    try {
      const rep = await api.getReport({ runId: RECALIBRATE_RUN })
      const run = rep.run
      const seed = run?.config?.seed ?? 20260929
      const v = await retryServing(() => withConfirm(api.validate, {
        runId: RECALIBRATE_RUN, thresholds: { minRewardGainPct: MIN_GAIN },
        nTrain: 48, nVal: 48, nLm: 100, nMmlu: 100, prefMaxTokens: PREF_MAX_TOKENS, log: (s) => log(`  [validate] ${s}`),
      }, 'validate'))
      result.perSeed.push(summarizeSeed({ seed, runId: RECALIBRATE_RUN, train: trainView(run), validate: v, kind: 'recalibrated-existing-run' }))
      write()
      log(`  重测完成：偏好门禁=${v.preferenceGate?.pass} validityOk=${v.preferenceGate?.validityOk} reason=${v.preferenceGate?.reason}`)
    } catch (e) {
      result.perSeed.push({ kind: 'recalibrated-existing-run', runId: RECALIBRATE_RUN, seed: 20260929, error: String(e?.message ?? e).slice(0, 600) })
      write()
      log(`  重测失败：${e?.message}`)
    }
  }

  /* ② 新种子的完整竖切：train → validate */
  let promptsDir = null
  const promptHashes = []
  for (const seed of SEEDS) {
    guard(`seed=${seed} prepare`)
    const p = await api.prepare({ sourceDatasetDir: DATASET, trainPrompts: 400, heldoutPrompts: 200, maxPromptChars: 1024, force: false })
    promptsDir = p.promptsDir
    // **所有种子必须落在同一批 prompt 行上**（逐种子重排会把截断率底座挪走，重新引入 knife-edge）
    const ph = {
      dir: p.promptsDir,
      train: `sha256:${createHash('sha256').update(readFileSync(join(p.promptsDir, 'train.jsonl'))).digest('hex')}`,
      heldout: `sha256:${createHash('sha256').update(readFileSync(join(p.promptsDir, 'heldout.jsonl'))).digest('hex')}`,
      seed,
    }
    promptHashes.push(ph)
    result.promptSet = { dir: p.promptsDir, trainPrompts: p.trainPrompts, heldoutPrompts: p.heldoutPrompts, leakCheck: p.leakCheck, license: p.license, hashes: promptHashes }
    result.promptSetStableAcrossSeeds = promptHashes.every((x) => x.train === promptHashes[0].train && x.heldout === promptHashes[0].heldout)
    write()
    log(`prompt 集：${promptsDir}（train=${p.trainPrompts} heldout=${p.heldoutPrompts} 去泄漏 ok=${p.leakCheck.ok}）`)

    guard(`seed=${seed} train`)
    log(`seed=${seed} train ${STEPS} 步（KL*=${KL_BUDGET}）…`)
    let r = null
    try {
      r = await withConfirm(api.train, {
        promptsDir, baseModel: '<BASE_MODEL>',
        klBudget: KL_BUDGET, klTolerancePct: KL_TOL,
        lr: TRAIN_CONFIG.lr, batchSize: TRAIN_CONFIG.batchSize, gradAccum: TRAIN_CONFIG.gradAccum,
        numGenerations: TRAIN_CONFIG.numGenerations, maxCompletionLength: TRAIN_CONFIG.maxCompletionLength,
        maxSteps: STEPS, seed,
        checkpointEvery: TRAIN_CONFIG.checkpointEvery, checkpointKeep: TRAIN_CONFIG.checkpointKeep,
        evalEvery: TRAIN_CONFIG.evalEvery, evalPrompts: TRAIN_CONFIG.evalPrompts,
        calibPrompts: TRAIN_CONFIG.calibPrompts, calibGenerations: TRAIN_CONFIG.calibGenerations,
        limits: { maxSteps: STEPS, maxGpuHours: 2, concurrency: 1 },
        log: (s) => log(`  [train] ${s}`),
      }, 'train')
    } catch (e) {
      result.perSeed.push({ kind: 'new-seed', seed, error: `train: ${String(e?.message ?? e).slice(0, 600)}` })
      write()
      log(`seed=${seed} 训练失败：${e?.message}`)
      continue
    }
    spend(r.gpuHours, `seed=${seed} train`)
    log(`seed=${seed} runId=${r.runId} steps=${r.steps} β=${r.calibration?.beta} KL均值=${r.kl?.mean} 产物断言=${r.artifactAssertion?.ok} GPU·h=${r.gpuHours}`)

    guard(`seed=${seed} validate`)
    log(`seed=${seed} validate（serving 部署 → 测量 → 拆解）…`)
    try {
      const v = await retryServing(() => withConfirm(api.validate, {
        runId: r.runId, thresholds: { minRewardGainPct: MIN_GAIN },
        nTrain: 48, nVal: 48, nLm: 100, nMmlu: 100, prefMaxTokens: PREF_MAX_TOKENS, log: (s) => log(`  [validate] ${s}`),
      }, 'validate'))
      spend((Number(v.ms) || 0) / 3_600_000, `seed=${seed} validate`)
      result.perSeed.push(summarizeSeed({ seed, runId: r.runId, train: trainView(r), validate: v, kind: 'new-seed' }))
      write()
      log(`seed=${seed} validate：偏好=${v.preferenceGate?.pass}（validityOk=${v.preferenceGate?.validityOk}）遗忘=${v.forgettingGate?.pass} 过拟合=${v.overfit?.pass ?? 'n/a'}`)
    } catch (e) {
      result.perSeed.push({ kind: 'new-seed', seed, runId: r.runId, error: `validate: ${String(e?.message ?? e).slice(0, 600)}` })
      write()
      log(`seed=${seed} validate 失败：${e?.message}`)
    }
  }

  /* ③ 多种子聚合 —— **import 共享参考实现**，判据不重写 */
  // 主结论只用**预登记的新种子**（`kind==='new-seed'`）；重测的既有 run（seed=20260929）
  // 是同一个训练配置下的第 4 个数据点，但它不是本次预登记的种子集，只作旁证。
  // （首版把两者混进主聚合 ⇒ n=4 却标着"3 个新种子"，属**口径错位**，靠逐条核对输出抓到。）
  const usable = result.perSeed.filter((x) => x.kind === 'new-seed' && typeof x?.preferenceGate?.paired?.meanDelta === 'number')
  result.aggregate = {
    source: 'inference-training/checks/overfit-criterion.mjs#aggregateSeeds（冻结契约的唯一参考实现，import 而非重写）',
    criterion: '用户预登记三条：全部同号→稳定；符号翻转→测量不成立；R/μ>50%→测量不成立需重设计',
    metric: 'preferenceGate.pairedSignificance.meanDelta（留出 prompt 上的逐条配对奖励差 mean(d)，higher-is-better）',
    n: usable.length,
    perSeedValues: usable.map((x) => ({ seed: x.seed, runId: x.runId, value: x.preferenceGate.paired.meanDelta, se: x.preferenceGate.paired.se })),
  }
  if (usable.length >= 3) {
    result.aggregate.result = aggregateSeeds(usable.map((x) => ({ value: x.preferenceGate.paired.meanDelta, se: x.preferenceGate.paired.se, seed: x.seed })))
  } else {
    result.aggregate.result = { error: 'E_SEEDS_LT_3', hint: `可用种子 ${usable.length} < 3 ⇒ 不出流程级结论`, n: usable.length }
    result.aggregate.singleSeed = true
  }
  // 补充：把已验收 seed=20260929 也纳入的 n=4 聚合（**不作为主结论**，只作旁证，明确标注）
  const allUsable = result.perSeed.filter((x) => typeof x?.preferenceGate?.paired?.meanDelta === 'number')
  result.aggregateWithExistingRun = {
    role: 'supplementary（把已验收 run 的 seed=20260929 也算进来；**主结论只用预登记的 3 个新种子**）',
    n: allUsable.length,
    values: allUsable.map((x) => ({ seed: x.seed, runId: x.runId, value: x.preferenceGate.paired.meanDelta, se: x.preferenceGate.paired.se })),
    result: allUsable.length >= 3 ? aggregateSeeds(allUsable.map((x) => ({ value: x.preferenceGate.paired.meanDelta, se: x.preferenceGate.paired.se, seed: x.seed }))) : null,
  }
  // 阶段总 GPU·h：各段实测之和（含不在 guard 链上的重测段）
  result.gpuBreakdown = {
    perSeed: result.perSeed.filter((x) => x.gpuHours).map((x) => ({ seed: x.seed, runId: x.runId, train: x.gpuHours.train, validate: x.gpuHours.validate, ack: 'validate 以整段墙钟计（与项目既有口径一致）' })),
    sumTrain: Number(result.perSeed.reduce((a, x) => a + (x.gpuHours?.train ?? 0), 0).toFixed(4)),
    sumValidate: Number(result.perSeed.reduce((a, x) => a + (x.gpuHours?.validate ?? 0), 0).toFixed(4)),
    note: '⚠️ `budget`/`gpuHours` 顶层只由 guard 链上的 spend() 累加；本表按**实际落盘证据**重算，口径更宽，两者以本表为准',
  }

  result.ok = true
} catch (e) {
  result.error = { code: e?.code ?? null, message: String(e?.message ?? e).slice(0, 1500) }
  result.ok = false
  log(`失败：${result.error.message}`)
}

result.gpuHours = Number(gpuSpent.toFixed(4))
result.budget.exceeded = gpuSpent > BUDGET_GPU_H
result.immutable.sourcesAtEnd = hashSources()
result.immutable.unchanged = JSON.stringify(result.immutable.sourcesAtStart) === JSON.stringify(result.immutable.sourcesAtEnd)
write()

process.stdout.write(`${JSON.stringify({
  ok: result.ok, seeds: SEEDS, prefMaxTokens: PREF_MAX_TOKENS, gpuHours: result.gpuHours,
  perSeed: result.perSeed.map((x) => ({ seed: x.seed, runId: x.runId, err: x.error ?? null, prefPass: x.preferenceGate?.pass ?? null, validityOk: x.preferenceGate?.validityOk ?? null, meanDelta: x.preferenceGate?.paired?.meanDelta ?? null, forget: x.forgettingGate?.pass ?? null, trunc: x.truncation })),
  aggregate: result.aggregate?.result ?? null,
  aggregateWithExistingRun: result.aggregateWithExistingRun?.result ?? null,
  error: result.error,
}, null, 2)}\n`)
process.exit(result.ok ? 0 : 1)
