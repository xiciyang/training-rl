/**
 * evaluate —— training-rl 的**评估适配层**（任务 A：合并四份 evaluate.mjs 之后）。
 *
 * ## 边界（**不绕开共享库契约**）
 * 合并前本文件是一份 375 行的**实现**，与 training-sft / training-pretrain / training-qat
 * 的同名文件是**同一算法、四个所有者**。现在：
 *
 *   - 端点原语与通用评估（`post` / `get` / `evalLM` / `evalMMLU` / `evalUniform` /
 *     `measurementCompleteness` / `attachIntegrity` / `forgettingGate`）
 *     **只有一份实现**：`@xiciyang/testing-core`（`src/evaluate.mjs`）。
 *     本文件只做**再导出**（`export … from`）。
 *   - 本包**特有**的东西留在这里：偏好评测量 `evalPreference`（规则奖励打分）、
 *     偏好门禁 `preferenceGate`、截断率闸门常量、协议版本。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 两类门禁
 * ══════════════════════════════════════════════════════════════════════════════
 * **① 偏好对齐（本包特有）**：RL 到底有没有让策略更符合奖励？
 *   - 用**规则奖励**在同一批**留出 prompt** 上给 base 与 policy 的输出打分；
 *   - 逐 prompt **配对**（同一 prompt 的 base/policy 输出配对），
 *     `d_i = r_policy(i) − r_base(i)`，`SE = std(d)/√n`，判据 `mean(d) > z·SE`（z=1.96）；
 *   - 另有一条**调用方显式给出**的相对提升阈值 `minRewardGainPct`（**本插件不内置默认值** ——
 *     "自己定阈值、自己签收"在本项目里是明确禁止的）。
 *   两条都要过：只有"显著"不够，还要"幅度达标"。
 *
 * **② 遗忘门禁（与 training-sft / training-pretrain 同形，**同一份实现**）**：
 *   - MMLU 准确率：Δ = acc_base − acc_rl；SE = sqrt(p1(1−p1)/n1 + p2(1−p2)/n2)；通过 ⟺ Δ ≤ z·SE
 *   - wikitext 逐文本 NLL：**配对**比较；Δ = mean(nll_rl − nll_base)；SE = std(差)/√n；通过 ⟺ Δ ≤ z·SE
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 仪器边界（training-eval-contract §5.1，**冻结**）
 * ══════════════════════════════════════════════════════════════════════════════
 * 本文件的所有数字都来自 **serving 端点**。容器内评估（如果有）只能作**选点信号**，
 * 且必须标 `selectionOnly: true`。容器内评估**不得**进入任何门禁判定。
 */
import { post, mean, stddev, dualGate, relativeChange } from '@xiciyang/testing-core'
import { ruleReward } from './reward.mjs'

/* ------------------------------------------------------- 通用评估（唯一实现在 testing-core） */

export {
  get, evalLM, evalMMLU, evalUniform,
  measurementCompleteness, attachIntegrity, forgettingGate,
} from '@xiciyang/testing-core'

// 判据本体也一并再导出（与合并前一致：rl 版本原本就有这一行）
export { dualGate, relativeChange, perplexity, mean, stddev, uniformSignature, meanNllFromLogprobs } from '@xiciyang/testing-core'
// `post` 在本文件内部被 `evalPreference` 使用（import 绑定），按原导出面再导出
export { post, ruleReward }

/* ------------------------------------------------------------------ 本包特有 */

/**
 * 评估协议版本 —— **进基线缓存键**：协议一变，缓存自动失效（跨协议比较是最隐蔽的一类错判）。
 *
 * `@2` 的由来（**真机实测，不是调宽容度**）：`@1`（`max_tokens=64`）下 **47/48 条生成被截断**
 * （`finish_reason='length'`），基座平均规则奖励已达 0.8536，48 条配对里 **27 条并列** ——
 * 判据几乎不具区分度，属于**测量不成立**（不是"没达标"）。
 * 处置与 training-sft 的 `task-eval@2` 同族：**让判据可区分**（给足生成长度让模型自然终止），
 * 并把它写进协议版本 ⇒ 基线缓存自动失效、必须重测基线。
 * 阈值（`minRewardGainPct`）与显著性（z=1.96）**一个字都没动**。
 */
export const EVAL_PROTOCOL = 'rl-pref-eval@2'
/**
 * ⚠️ **这不是"验收档"**（缺陷 N2）：256 只是**低层原语** `evalPreference` 的兜底默认值。
 * 判决路径（`core.mjs#validate`）**不得**依赖它 —— 它经 `measurement-identity.mjs#
 * resolvePrefMaxTokens` 解析，调用方省略时只回落到**已标定档 768**（`CALIBRATED_PREF_MAX_TOKENS`），
 * 并把 `prefMaxTokensSource` 记进审计/注册表/返回值。标定证据：
 * `inference-training/out/rl-instrument-calibration.json`（256：两侧截断 72.9%/79.2% ⇒ 必然过不了闸门）。
 */
export const DEFAULT_MAX_TOKENS = 256
/** 截断率闸门：超过这个比例就判"测量不成立"（判据被截断主导 ⇒ 无区分度）。 */
export const MAX_TRUNCATION_RATIO = 0.5

/**
 * 偏好对齐测量：对每个留出 prompt 生成一次（temperature=0，确定性），用**规则奖励**打分。
 * 返回逐 prompt 的奖励序列（配对比较要用同一批 prompt 的**逐条**值）。
 */
export async function evalPreference({ endpoint, model, rows, limit = 100, maxTokens = DEFAULT_MAX_TOKENS, log = () => {} }) {
  const per = []
  for (const r of rows.slice(0, limit)) {
    const prompt = String(r.prompt ?? '')
    if (!prompt.trim()) continue
    // eslint-disable-next-line no-await-in-loop
    const res = await post(endpoint, '/v1/completions', { model, prompt, max_tokens: maxTokens, temperature: 0, seed: 0 }, { timeoutMs: 180000 })
    const text = String(res.json?.choices?.[0]?.text ?? '')
    const finish = res.json?.choices?.[0]?.finish_reason ?? null
    const rew = ruleReward(text, prompt)
    per.push({ promptHash: r.h ?? null, reward: rew, finish, status: res.status, len: text.length })
    if (per.length % 25 === 0) log(`  preference ${per.length}/${Math.min(limit, rows.length)}`)
  }
  const ok = per.filter((x) => x.status === 200)
  return {
    protocol: EVAL_PROTOCOL,
    maxTokens,
    n: per.length,
    nOk: ok.length,
    perPrompt: ok.map((x) => x.reward),
    rewards: ok.map((x) => x.reward.value),
    meanReward: ok.length ? mean(ok.map((x) => x.reward.value)) : null,
    meanComponents: ok.length ? Object.fromEntries(Object.keys(ok[0].reward.parts).map((k) => [k, mean(ok.map((x) => x.reward.parts[k]))])) : null,
    truncated: ok.filter((x) => x.finish === 'length').length,
    stopped: ok.filter((x) => x.finish === 'stop').length,
    // **完整性证据（先修生产者，再移分类）**：此前只给 n/nOk，缺 attempted/failed/partial
    // ⇒ `measurementCompleteness` 无法把它当"逐条网络测量"看，于是 `preference`/`prefTrain`
    // 只能挂在 KNOWN_OTHER 里 —— 后果是**端点中途失效在这条度量上不可见**。
    attempted: per.length,
    succeeded: ok.length,
    failed: per.length - ok.length,
    skipped: Math.max(0, rows.slice(0, limit).length - per.length),
    partial: per.length - ok.length > 0,
  }
}

/* ------------------------------------------------------------------ 门禁 */

/**
 * 偏好对齐门禁。
 * - validity：测量是否成立（端点数、均匀签名、样本量）
 * - 数值①：相对提升 ≥ 调用方给的 `minRewardGainPct`（dualGate 的 perf 项）
 * - 数值②：**配对显著性** mean(d) > z·SE（z=1.96，沿用 eval-contract 已冻结的常量）
 */
export function preferenceGate({ base, policy, thresholds, validity, z = 1.96 }) {
  // **测量有效性**：生成被大量截断 ⇒ 判据被"截断"主导，分辨不出策略好坏。
  // 这一类必须与"没达标"分开报（否则会让人去调参，而真正该做的是修仪器）。
  const trunc = []
  for (const [label, m] of [['base', base], ['policy', policy]]) {
    const n = m?.n ?? 0
    const ratio = n > 0 ? (m.truncated ?? 0) / n : null
    const ok = ratio !== null && ratio <= MAX_TRUNCATION_RATIO
    trunc.push({ kind: 'validity', name: `not-truncated-dominated(${label})`, ok, detail: `截断 ${m?.truncated ?? '?'}/${n}（上限 ${MAX_TRUNCATION_RATIO * 100}%）` })
  }
  validity = [...(validity ?? []), ...trunc]
  const truncOk = trunc.every((x) => x.ok)
  if (!truncOk) {
    return {
      pass: false,
      validityOk: false,
      reason: '测量不成立：生成被截断主导（判据无区分度）—— 换仪器（提高生成长度上限）而不是调阈值',
      checks: [...validity.map((v) => ({ kind: 'validity', name: v.name, ok: v.ok === true, detail: v.detail ?? null }))],
      deltas: { meanReward: { base: base?.meanReward ?? null, policy: policy?.meanReward ?? null, rawPct: null, worseningPct: null } },
      pairedSignificance: { n: 0, pass: false, reason: '测量不成立，配对比较不做' },
      thresholds,
    }
  }
  const baseOk = Number.isFinite(base?.meanReward) && base.meanReward !== 0
  const policyOk = Number.isFinite(policy?.meanReward)
  if (!baseOk || !policyOk) {
    // **测量不成立**（不是"没达标"）：基线的奖励没测出来 / 为 0（相对变化无定义）⇒
    // 直接判不通过并说明原因，**不得**让它抛异常把整条评估崩掉
    // （把两类失败混在一起会让人去调参，而真正该做的是修测量）。
    return {
      pass: false,
      validityOk: false,
      reason: '测量不成立：偏好奖励未能测出（baseline/policy 的 meanReward 非有限数或基线为 0，相对变化无定义）',
      checks: [...(validity ?? []).map((v) => ({ kind: 'validity', name: v.name, ok: v.ok === true, detail: v.detail ?? null })),
        { kind: 'validity', name: 'preference-measured', ok: false, detail: `base=${base?.meanReward ?? null} policy=${policy?.meanReward ?? null}` }],
      deltas: { meanReward: { base: base?.meanReward ?? null, policy: policy?.meanReward ?? null, rawPct: null, worseningPct: null } },
      pairedSignificance: { n: 0, pass: false, reason: '测量不成立，配对比较不做' },
      thresholds,
    }
  }
  const gain = relativeChange({ baseline: base.meanReward, candidate: policy.meanReward, direction: 'higher-is-better' })
  const g = dualGate({
    validity,
    perf: { gainPct: -gain.worseningPct }, // 符号：higher-is-better ⇒ 提升量 = −劣化量
    minPerfGainPct: thresholds.minRewardGainPct,
  })
  // 配对显著性（O2 家族：留出集上必须**真的**动了，而不是被噪声解释掉）
  const n = Math.min(base.rewards?.length ?? 0, policy.rewards?.length ?? 0)
  let paired = null
  if (n >= 2) {
    const d = []
    for (let i = 0; i < n; i += 1) d.push(policy.rewards[i] - base.rewards[i])
    const dm = mean(d)
    const se = stddev(d) / Math.sqrt(n)
    const bound = z * se
    const wins = d.filter((x) => x > 0).length
    const ties = d.filter((x) => x === 0).length
    paired = { n, meanDelta: dm, se, bound, pass: dm > bound, wins, losses: n - wins - ties, ties, winRate: wins / n, z }
  } else {
    paired = { n, pass: false, reason: `配对样本不足（n=${n} < 2）—— **未通过**，不是"没问题"` }
  }
  return {
    pass: g.pass && paired.pass === true,
    validityOk: g.validityOk,
    reason: !g.validityOk ? g.reason : (paired.pass !== true ? '配对显著性未过（提升无法与噪声区分）' : (g.pass ? '达标' : '数值未达标')),
    checks: g.checks,
    deltas: { meanReward: { base: base.meanReward, policy: policy.meanReward, rawPct: gain.rawPct, worseningPct: gain.worseningPct } },
    pairedSignificance: paired,
    thresholds,
  }
}
