/**
 * reward-kl.spec —— 规则奖励 / β 可推导 / KL 记账 / 契约一致性。
 *
 * 覆盖三件事：
 *  ① **规则奖励的跨语言一致性**：规格（tests/fixtures/reward-vectors.json）由 JS 权威实现生成；
 *     JS 实现与**容器侧 Python 实现**都必须逐条相符。这里直接跑 `python3 python/rule_reward.py`
 *     —— 纯 CPU、无 GPU、无网络，所以"两份实现不能漂移"这件事进了 CI 而不是只写在注释里。
 *  ② **β 可推导**：β = σ_r / KL*；缺 KL* ⇒ E_KL_BUDGET_REQUIRED；σ_r ≤ 0 ⇒ E_REWARD_NO_VARIANCE；
 *     容器回传的 β 与 Node 重算不一致 ⇒ E_BETA_DERIVATION_MISMATCH。
 *  ③ **O3 过拟合判据必须 import 冻结参考实现**（不重写一份）。
 */
import { readFileSync, existsSync } from 'node:fs'
import { execFileSync } from 'node:child_process'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

import { ruleReward, rewardOf, rewardDescriptor, RULE_REWARD_VERSION, RULE_BRANCHES, wordCount, contentWords, maxTriRepeat } from '../../src/reward.mjs'
import { deriveBeta, buildKlAccounting, recommendBeta, KL_FORMULA } from '../../src/kl.mjs'
import { ERROR_CODES } from '../../src/version.mjs'
import { evaluateOverfitRL, overfitPerText } from '../../src/core.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const FIXTURE = join(PKG, 'tests', 'fixtures', 'reward-vectors.json')
export const NAME = 'reward-kl（规则奖励跨语言一致 / β 可推导 / KL 记账 / O3 唯一实现）'

const attempt = (fn) => { try { return { ok: true, value: fn() } } catch (e) { return { ok: false, code: e.code, message: String(e.message) } } }

export async function run(t) {
  t.assert(existsSync(FIXTURE), `规格文件存在：${FIXTURE}`)
  const spec = JSON.parse(readFileSync(FIXTURE, 'utf8'))
  t.assert(spec.rewardVersion === RULE_REWARD_VERSION, `规格的奖励版本 = ${RULE_REWARD_VERSION}`)
  t.assert(spec.vectors.length >= 20, `规格向量数 ${spec.vectors.length} ≥ 20`)

  /* ① JS 实现 == 规格 */
  let maxJs = 0
  for (const v of spec.vectors) {
    const r = ruleReward(v.completion, v.prompt)
    maxJs = Math.max(maxJs, Math.abs(r.value - v.expected))
    for (const [k, want] of Object.entries(v.parts)) {
      t.assert(Math.abs(r.parts[k] - want) <= 1e-12, `分量 ${v.branch}.${k} 与规格一致（${r.parts[k]} vs ${want}）`)
    }
  }
  t.assert(maxJs <= 1e-12, `JS 实现与规格逐条一致（最大偏差 ${maxJs}）`)

  /* ①b 规格覆盖每一条分支 */
  const branches = new Set(spec.vectors.map((v) => v.branch))
  const missing = RULE_BRANCHES.filter((b) => !branches.has(b))
  t.assert(missing.length === 0, `规格覆盖全部分支（缺：${missing.join(', ') || '无'}）`)
  t.assert(spec.branches.join(',') === RULE_BRANCHES.join(','), '规格声明的分支表与实现一致')

  /* ①c **容器侧 Python 实现** == JS 实现（跨语言闸门，纯 CPU） */
  let pyOut = null
  try {
    pyOut = execFileSync('python3', [join(PKG, 'python', 'rule_reward.py'), '--vectors', FIXTURE, '--json'], { encoding: 'utf8', timeout: 60000 })
  } catch (e) {
    pyOut = null
    t.assert(false, `容器侧规则实现（python/rule_reward.py）必须能在纯 CPU 下跑通：${String(e.message).slice(0, 200)}`)
  }
  if (pyOut) {
    const py = JSON.parse(pyOut)
    t.assert(py.version === RULE_REWARD_VERSION, `Python 侧版本号一致（${py.version}）`)
    t.assert(py.values.length === spec.vectors.length, `Python 侧向量数一致（${py.values.length}）`)
    let maxPy = 0
    for (let i = 0; i < spec.vectors.length; i += 1) {
      const js = ruleReward(spec.vectors[i].completion, spec.vectors[i].prompt).value
      maxPy = Math.max(maxPy, Math.abs(py.values[i].value - js))
    }
    t.assert(maxPy <= 1e-12, `JS 与容器侧 Python 的规则奖励逐条一致（最大绝对偏差 ${maxPy}）`)
    t.assert(Number.isFinite(maxPy), '偏差是可测量的数（不是"看起来一样"）')
  }

  /* ①d 确定性：同输入同参数 ⇒ 逐位相同（契约冻结条件 2） */
  const det1 = ruleReward('Gravity is a fundamental interaction.', 'Explain gravity.')
  const det2 = ruleReward('Gravity is a fundamental interaction.', 'Explain gravity.')
  t.assert(det1.value === det2.value && det1.value.toFixed(17) === det2.value.toFixed(17), '规则奖励是确定性纯函数（逐位相同）')
  t.assert(rewardOf('x', 'y') === rewardOf('x', 'y'), 'rewardOf 同样确定')

  /* ①e 分支的结构性判据（标签必须名副其实，不是只有"名字"对了） */
  const byBranch = Object.fromEntries(spec.vectors.map((v) => [v.branch, v]))
  t.assert(byBranch['readable-empty'].parts.readable === 0, 'readable-empty 的 readable 分量确为 0')
  t.assert(byBranch['readable-ok'].parts.readable === 1, 'readable-ok 的 readable 分量确为 1')
  t.assert(byBranch['length-short'].parts.length < 1 && byBranch['length-short'].parts.length > 0, 'length-short 落在(0,1)开区间')
  t.assert(byBranch['length-too-long'].parts.length === 0, 'length-too-long 的 length 分量确为 0')
  t.assert(byBranch['repetition-degenerate'].parts.repetition < 1, 'repetition-degenerate 的 repetition 分量 < 1')
  t.assert(byBranch['relevance-zero'].parts.relevance === 0, 'relevance-zero 的 relevance 分量确为 0')
  t.assert(byBranch['relevance-no-content-words'].parts.relevance === 0.5, '提示无内容词 ⇒ 中性 0.5（不冒充 0）')
  for (const b of ['constraint-one-word-ok', 'constraint-number-ok', 'constraint-json-ok']) {
    t.assert(byBranch[b].parts.constraint === 1, `${b} 的 constraint 分量确为 1`)
  }
  for (const b of ['constraint-one-word-violated', 'constraint-number-violated', 'constraint-json-violated']) {
    t.assert(byBranch[b].parts.constraint === 0, `${b} 的 constraint 分量确为 0`)
  }
  t.assert(byBranch['constraint-absent-neutral'].parts.constraint === 0.5, '提示无可判约束 ⇒ 中性 0.5')

  /* ①f 值域 + 描述符 */
  for (const v of spec.vectors) t.assert(v.expected >= 0 && v.expected <= 1, `${v.branch} 落在 [0,1]（实际 ${v.expected}）`)
  const d = rewardDescriptor()
  t.assert(d.kind === 'rule' && d.deterministic === true, '奖励描述符：kind=rule 且确定性')
  t.assert(d.range[0] === 0 && d.range[1] === 1, '奖励描述符声明值域 [0,1]')
  t.assert(d.version === RULE_REWARD_VERSION, '奖励描述符带显式版本号')
  t.assert(d.components.length === 6, `奖励有 6 个分量（实际 ${d.components.length}）`)
  t.assert(wordCount('a b c') === 3 && contentWords('the gravity of the moon').has('gravity'), '词/内容词切分符合预期')
  t.assert(maxTriRepeat('a b c a b c a b c a b c') === 4, '3-gram 重复计数正确（4 次）')

  /* ② β 可推导 */
  const b1 = deriveBeta({ sigmaR: 0.2, klBudget: 0.1 })
  t.assert(Math.abs(b1.beta - 2) < 1e-15, `β = σ_r/KL* = 0.2/0.1 = 2（实际 ${b1.beta}）`)
  t.assert(b1.formula === KL_FORMULA, '公式字符串进结果（可追溯）')
  const b2 = deriveBeta({ sigmaR: 0.035, klBudget: 0.1 })
  t.assert(Math.abs(b2.beta - 0.35) < 1e-15, `β 随 σ_r 变化：0.035/0.1 = 0.35（实际 ${b2.beta}）`)
  // **不给 KL\* 就拒绝**（不内置默认值）
  for (const [label, args] of [
    ['未给出', {}], ['为 0', { sigmaR: 0.2, klBudget: 0 }], ['为负', { sigmaR: 0.2, klBudget: -1 }],
    ['非数', { sigmaR: 0.2, klBudget: 'x' }], ['null', { sigmaR: 0.2, klBudget: null }],
  ]) {
    const r = attempt(() => deriveBeta(args))
    t.assert(r.ok === false && r.code === ERROR_CODES.E_KL_BUDGET_REQUIRED, `KL* ${label} ⇒ E_KL_BUDGET_REQUIRED（实际 ${r.code}）`)
  }
  // σ_r = 0 / 未测出 ⇒ 拒绝开始（GRPO 空转）
  for (const [label, s] of [['0', 0], ['null', null], ['NaN', NaN], ['负', -0.1]]) {
    const r = attempt(() => deriveBeta({ sigmaR: s, klBudget: 0.1 }))
    t.assert(r.ok === false && r.code === ERROR_CODES.E_REWARD_NO_VARIANCE, `σ_r=${label} ⇒ E_REWARD_NO_VARIANCE（实际 ${r.code}）`)
  }

  /* ②b KL 记账 + β 交叉核对 */
  const acct = buildKlAccounting({ sigmaR: 0.2, klBudget: 0.1, betaFromContainer: 2, kls: [0.001, 0.002, 0.003], tolerancePct: 50 })
  t.assert(acct.betaDeclared === 2 && acct.klSteps === 3, '记账：声明 β 与步数')
  t.assert(Math.abs(acct.klMean - 0.002) < 1e-15, `klMean = 0.002（实际 ${acct.klMean}）`)
  t.assert(acct.withinTolerance === false && acct.recalibrationRequired === true, '实测 KL 远低于预算 ⇒ 超容差 ⇒ 要求重新记账')
  t.assert(acct.note.includes('不静默调参'), '记账明确写出"不静默调参"')

  /* ②c **β 重算的方向**（这一节是一个真机缺陷的产物：第一版写成 σ_r/klMean，方向反了）
   * 语义：KL 是 β 的**单调下降**函数；实测 KL 远低于预算 ⇒ 下一次应**减小** β 让它敢动。
   * 一阶模型 KL ≈ c/β ⇒ c ≈ β_declared × klMean ⇒ β_next = c/KL* = β_declared × (klMean/KL*)。
   */
  t.assert(Math.abs(acct.recommendedBeta - 2 * (0.002 / 0.1)) < 1e-15, `recommendedBeta = β_declared×(klMean/KL*) = 0.04（实际 ${acct.recommendedBeta}）`)
  t.assert(acct.recommendation.direction === 'decrease' && acct.recommendedBeta < acct.betaDeclared, '实测 KL 远低于预算 ⇒ 方向是**减小** β（不是放大）')
  t.assert(acct.betaAtBudget === acct.betaDeclared, 'betaAtBudget = σ_r/KL* = 声明的 β（"口径"与"下一步"是两个量，不能混）')
  // **反向回归守卫**：第一版的 σ_r/klMean 在本例给出 100（比声明值大 50 倍，方向相反）
  t.assert(Math.abs(acct.recommendedBeta - (0.2 / 0.002)) > 1, `recommendedBeta 不是 σ_r/klMean（后者为 100，方向相反；实际 ${acct.recommendedBeta}）`)
  const recUp = recommendBeta({ betaDeclared: 2, klMean: 0.4, klBudget: 0.1 })
  t.assert(recUp.direction === 'increase' && Math.abs(recUp.betaNext - 8) < 1e-12, '实测 KL 超过预算 ⇒ 方向是**增大** β（β×4）')
  const recHold = recommendBeta({ betaDeclared: 2, klMean: 0.1, klBudget: 0.1 })
  t.assert(recHold.direction === 'hold' && recHold.betaNext === 2, '实测 KL 恰好在预算上 ⇒ β 不动（自洽性检查）')
  t.assert(recommendBeta({ betaDeclared: 2, klMean: 0, klBudget: 0.1 }).betaNext === null, '实测 KL 为 0 ⇒ 不给建议（不猜），如实返回 null')
  t.assert(/KL ≈ c\/β/.test(recHold.model), '建议里写明所用的一阶模型（可复核，不是魔数）')
  // 真机那一次的实际数字（回归守卫：绝不能再给出 70.34）
  const recReal = recommendBeta({ betaDeclared: 0.8258130286540805, klMean: 0.0011740167962852866, klBudget: 0.1 })
  t.assert(Math.abs(recReal.betaNext - 0.009695183662311129) < 1e-15, `真机数字：β_next = 0.009695（实际 ${recReal.betaNext}）`)
  t.assert(recReal.betaNext < recReal.betaDeclared, '真机数字：β_next 小于声明值（方向已纠正）')
  t.assert(Math.abs(recReal.betaNext - 0.8258130286540805 / 0.0011740167962852866) > 1, '真机数字：不再是 703.4 或 70.34 那一族（方向已纠正）')
  t.assert(recReal.betaDeclared === 0.8258130286540805 && recReal.klMean === 0.0011740167962852866 && recReal.klBudget === 0.1, '建议里回显了它的输入（可复核、可追溯）')
  const acctOk = buildKlAccounting({ sigmaR: 0.2, klBudget: 0.1, betaFromContainer: 2, kls: [0.08, 0.1, 0.12], tolerancePct: 50 })
  t.assert(acctOk.withinTolerance === true && acctOk.recalibrationRequired === false, '实测 KL 落在容差内 ⇒ 不要求重新记账')
  const acctNoKl = buildKlAccounting({ sigmaR: 0.2, klBudget: 0.1, betaFromContainer: 2, kls: [], tolerancePct: 50 })
  t.assert(acctNoKl.klSteps === 0 && acctNoKl.recalibrationRequired === true && acctNoKl.klMean === null, '没有 KL 样本 ⇒ 照实记为 null 并标"需重新记账"（不伪造 0）')
  // 容器回传的 β 与重算不一致 ⇒ 直接失败（"可推导"必须真的成立）
  const mismatch = attempt(() => buildKlAccounting({ sigmaR: 0.2, klBudget: 0.1, betaFromContainer: 0.04, kls: [0.002] }))
  t.assert(mismatch.ok === false && mismatch.code === ERROR_CODES.E_BETA_DERIVATION_MISMATCH, `容器 β 与重算不一致 ⇒ E_BETA_DERIVATION_MISMATCH（实际 ${mismatch.code}）`)
  // β 的**量纲**：σ_r 归一化到 [0,1] + KL* 由调用方给 ⇒ β 随两者变化（不是常数）
  const betas = [[0.05, 0.1], [0.2, 0.1], [0.2, 1.0], [0.5, 0.125]].map(([s, k]) => deriveBeta({ sigmaR: s, klBudget: k }).beta)
  t.assert(new Set(betas).size === betas.length, `β 不是常数：${betas.join(', ')}`)

  /* ③ O3 过拟合判据：必须 import 冻结参考实现（不重写一份） */
  const src = readFileSync(join(PKG, 'src', 'core.mjs'), 'utf8')
  t.assert(/overfit-criterion\.mjs/.test(src), 'core.mjs 引用了 overfit 判据的**参考实现**文件')
  t.assert(!/^\s*const\s+sigmaBase\s*=/m.test(src), 'core.mjs 没有自己实现 sigmaBase（判据只有一份实现）')
  t.assert(/evaluateOverfit\(/.test(src), 'core.mjs 调用 evaluateOverfit（不是自己算 ratio）')
  // **键名映射**：`measureAt` 返回 `prefTrain`（训练切分）/ `preference`（**留出**切分）。
  // 曾经把留出侧写成不存在的 `prefVal` ⇒ val 样本静默变 0 ⇒ O3 退化成 E_SAMPLE_TOO_SMALL。
  // 这条判据就是为了让那次真机失败**再也不可能静默发生**。
  const m = overfitPerText({ prefTrain: { rewards: [0.5, 0.6] }, preference: { rewards: [0.4, 0.7] } })
  t.assert(m.train.perText.length === 2 && m.val.perText.length === 2, `overfitPerText 的 train/val 两侧都非空（实际 ${m.train.perText.length}/${m.val.perText.length}）`)
  t.assert(m.train.perText[0] === -0.5 && m.val.perText[1] === -0.7, 'overfitPerText 做的是保序变换（取负）且键映射正确')
  t.assert(overfitPerText({}).val.perText.length === 0, '缺测量时如实给空数组（不伪造样本）')
  const ov = await evaluateOverfitRL({
    base: { prefTrain: { rewards: Array.from({ length: 40 }, (_, i) => 0.4 + (i % 5) * 0.01) }, preference: { rewards: Array.from({ length: 40 }, (_, i) => 0.4 + (i % 3) * 0.01) } },
    model: { prefTrain: { rewards: Array.from({ length: 40 }, (_, i) => 0.5 + (i % 5) * 0.01) }, preference: { rewards: Array.from({ length: 40 }, (_, i) => 0.45 + (i % 3) * 0.01) } },
  })
  t.assert(typeof ov.improvementTrain === 'number' && typeof ov.improvementVal === 'number', 'O3 返回 train/val 的 improvement')
  t.assert(ov.metric === 'negative_rule_reward', 'O3 的度量标签如实标注（保序变换后的 loss 类度量）')
  t.assert(typeof ov.o3_overfitBudget?.ratio === 'number' && typeof ov.o3_overfitBudget?.theta === 'number', 'O3 给出 ratio 与 theta（阈值可推导）')
  t.assert(ov.improvementTrain > ov.improvementVal, 'O3：训练侧提升大于留出侧（构造数据如预期）')
  // 本适配层**总是**把形状补齐（`neg(undefined) -> []`），故参考实现的 E_NO_BASELINE 分支
  // 不会经由本适配层到达 —— 到达的是 E_SAMPLE_TOO_SMALL。两者都是 fail-closed（不出判定），
  // 但必须**如实**区分，不能断言一个到不了的码。
  const ovTooSmall = await evaluateOverfitRL({ base: { prefTrain: { rewards: [] }, preference: { rewards: [] } }, model: { prefTrain: { rewards: [] }, preference: { rewards: [] } } })
  t.assert(ovTooSmall.error === 'E_SAMPLE_TOO_SMALL', `O3：缺测量/样本为 0 ⇒ fail-closed 返回 E_SAMPLE_TOO_SMALL（实际 ${ovTooSmall.error}）`)
  t.assert(ovTooSmall.pass === undefined, 'O3：fail-closed 时**不出判定**（没有 pass 字段）')
  t.assert(JSON.stringify(ovTooSmall).includes('E_SAMPLE_TOO_SMALL') && ovTooSmall.error !== null, 'O3 的失败是可辨识的错误码，不是被吞掉的空对象')
  // 参考实现自身的 E_NO_BASELINE 分支确实存在（直连 import 验它，而不是猜）
  const crit = await import(`file://${join(PKG, '..', 'checks', 'overfit-criterion.mjs')}`)
  t.assert(crit.evaluateOverfit({ base: {}, model: {} }).error === 'E_NO_BASELINE', '参考实现的 E_NO_BASELINE 分支确实存在（直连调用验证）')
  t.assert(typeof crit.aggregateSeeds === 'function' && crit.aggregateSeeds([], {}).error === 'E_SEEDS_LT_3', '参考实现的多种子聚合在 n<3 时拒绝出流程级结论（O5）')
}
