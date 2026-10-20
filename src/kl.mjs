/**
 * kl —— **KL 系数 β 的推导与记账**（唯一实现；Node 与容器两侧都按这里的口径）。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * β = σ_r / KL*
 * ══════════════════════════════════════════════════════════════════════════════
 * - `σ_r` = **实测**规则奖励标准差（在**当前策略**上采样打分得到，见 rl_trainer.py 的 _calibrate）；
 * - `KL*` = **调用方显式给定**的 KL 预算（平均每 token KL，nats）—— 本包**不内置默认值**；
 * - 语义：**当实测 KL 达到 KL\* 时，KL 惩罚的代价恰好等于一个 σ_r 的奖励**。
 *   于是 β 是从两个可观测/可解释量**推出来的**，不是魔数。
 *
 * 为什么 σ_r 必须是**实测**的：奖励的绝对尺度随策略与奖励版本变化；
 * 拍一个 σ_r 等于把"奖励尺子"写死，换策略/换奖励版本后 β 就不再是同一个意思。
 *
 * 为什么必须**拒绝** σ_r = 0：GRPO 的组内优势是 `(r − mean)/std`；σ_r = 0 ⇒ 优势恒为 0 ⇒
 * 梯度信号为零 ⇒ 训练是**空转**（步数、loss、checkpoint 全都"正常"）。
 * 这是 GRPO 最隐蔽的失效形态，故在**开跑之前**就 fail-closed（`E_REWARD_NO_VARIANCE`）。
 *
 * 为什么必须**拒绝**缺 KL\*：与 training-pretrain 的 `minDomainGainPct` 同一条纪律 ——
 * "自己定阈值、自己签收"在本项目里是明确禁止的。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 记账（契约 §4.4 的"偏离超容差则重算 β 并重新记账"）
 * ══════════════════════════════════════════════════════════════════════════════
 * 每步的实测 KL 由容器回传，Node 侧落进审计链。run 结束后：
 *   - `klMean` 与 `KL*` 的偏离超过 `tolerancePct` ⇒ `recalibrationRequired: true`，
 *     并给出 `recommendedBeta = σ_r / klMean`（"若要让实测 KL 恰好用满预算"应采用的系数）。
 *   - **本次 run 用的仍是声明的 β** —— 不静默调参；推荐的 β 只进账本，供下一次 run 使用。
 * 这条区分很重要：中途改 β 会让"这一次 run"的语义变得不可复述。
 */
import { ToolError, ERROR_CODES } from './version.mjs'

export const KL_FORMULA = 'beta = sigmaR / klBudget'

/**
 * 「偏离超容差 ⇒ 重算 β」到底该算成什么？（**两个不同的量，第一版把它们混成了一个**）
 *
 * ## 两个量，两种读法
 * 1. **`betaAtBudget` = σ_r / KL\*** —— 字面口径：让"实测 KL 达到预算时，KL 惩罚的代价等于 σ_r"。
 *    这正是**声明的 β 本身** ⇒ 它**不是**一个"重算结果"，而是口径的定义。
 *    （第一版把这里的 `KL*` 换成了实测 `klMean`，于是得到一个既不是口径、也不是步长的数。）
 * 2. **`betaNext`（recommendedBeta）** —— **下一步该用多少**：把实测偏离拉回预算的**一阶控制量**。
 *
 * ## 为什么下一步是 β × klMean / KL*（而不是 σ_r / klMean）
 * KL 正则化目标里，实测 KL 是 β 的**单调下降**函数（惩罚越大，策略越贴参考）。
 * 在参考点附近取一阶模型 `KL ≈ c / β`（等价的常用写法是 KL ∝ 1/β），
 * 则同一 run 的一次观测给出 `c ≈ β_declared × klMean`。要让下一次的 KL 恰好落在预算上：
 *
 *     β_next = c / KL* = β_declared × (klMean / KL*)
 *
 * 自洽性检查：`klMean == KL*` ⇒ `β_next == β_declared`（不动）；
 * `klMean < KL*`（**实测远低于预算，正是本 run 的情形**）⇒ `β_next < β_declared` ⇒ **减小 β**，
 * 让策略敢动 —— 方向与"KL 随 β 下降"一致。
 *
 * 而 `σ_r / klMean ≡ β_declared × (KL* / klMean)`，即"乘以短缺比的**倒数**"：
 * 在本 run 里它给出 70.34（= 0.826 × 85.2），比声明值大 **85 倍** ——
 * **放大了它本该修正的那个偏离方向**。第一版就是这么写的（真机上被实测数据暴露），
 * 现在把它拆成上面两个各自有明确语义的字段，并让**文档与公式一致**。
 *
 * ⚠️ 一阶模型只是**记账用的近似**，不是定律；它从不用于中途改参，只作为"下一次 run 的建议"。
 */
export function recommendBeta({ betaDeclared, klMean, klBudget } = {}) {
  const b = Number(betaDeclared)
  const m = Number(klMean)
  const k = Number(klBudget)
  const model = 'KL ≈ c/β（一阶）；c 由同一 run 的一次观测估计：c ≈ β_declared × 实测 KL 均值'
  if (!Number.isFinite(m) || m <= 0) {
    return { betaNext: null, betaDeclared: b, klMean: m, klBudget: k, model, direction: null, shortfallRatio: null, reason: '实测 KL 未测出或为 0 ⇒ 不给建议（不猜）' }
  }
  const ratio = m / k
  const betaNext = b * ratio
  return {
    betaNext,
    betaDeclared: b,
    klMean: m,
    klBudget: k,
    model,
    shortfallRatio: ratio,
    direction: ratio < 1 ? 'decrease' : (ratio > 1 ? 'increase' : 'hold'),
    note: ratio < 1
      ? `实测 KL 是预算的 ${(ratio * 100).toFixed(1)}% ⇒ 下一次应**减小** β 到 ${betaNext}`
      : `实测 KL 超过预算 ${(ratio * 100).toFixed(1)}% ⇒ 下一次应**增大** β 到 ${betaNext}`,
  }
}

/**
 * 推导 β。三个失败模式都在**开跑之前**响亮失败。
 * @returns {{beta:number, sigmaR:number, klBudget:number, formula:string}}
 */
export function deriveBeta({ sigmaR, klBudget } = {}) {
  const k = Number(klBudget)
  if (!Number.isFinite(k) || k <= 0) {
    throw new ToolError(ERROR_CODES.E_KL_BUDGET_REQUIRED, {
      hint: `KL*（klBudget）必须由调用方显式给出且 > 0，实际 ${klBudget === undefined || klBudget === null ? '未给出' : String(klBudget)}。β = σ_r / KL*；本插件不内置 KL* 的默认值（"自己定阈值、自己签收"是明确禁止的）。`,
      details: { got: klBudget ?? null, formula: KL_FORMULA },
      recoverable: true,
    })
  }
  const s = Number(sigmaR)
  if (!Number.isFinite(s) || s <= 0) {
    throw new ToolError(ERROR_CODES.E_REWARD_NO_VARIANCE, {
      hint: `实测 σ_r = ${sigmaR === undefined || sigmaR === null ? '未测出' : String(sigmaR)} ⇒ GRPO 组内优势恒为 0 ⇒ 训练是空转。**拒绝开始**（不是"跑完再报成功率"）。`,
      details: { sigmaR: sigmaR ?? null, klBudget: k, formula: KL_FORMULA },
      recoverable: false,
    })
  }
  return { beta: s / k, sigmaR: s, klBudget: k, formula: KL_FORMULA }
}

const meanOf = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null)
const stdOf = (xs) => {
  if (xs.length < 2) return null
  const m = meanOf(xs)
  return Math.sqrt(xs.reduce((a, b) => a + (b - m) ** 2, 0) / (xs.length - 1))
}

/**
 * 组装 KL 记账块。
 *
 * **交叉核对**：容器回传的 β 必须等于 Node 侧用**同一个公式**重算的结果
 * （1e-12 容差）。若不等，说明"β 是推导出来的"这件事在容器侧没成立
 * （例如被人塞了一个常数）—— 这属于不可恢复的语义漂移，直接失败而不是记账。
 */
export function buildKlAccounting({ sigmaR, klBudget, betaFromContainer = null, kls = [], tolerancePct = 50, eps = 1e-12 } = {}) {
  const { beta, sigmaR: s, klBudget: k } = deriveBeta({ sigmaR, klBudget })
  if (betaFromContainer !== null && betaFromContainer !== undefined) {
    const got = Number(betaFromContainer)
    if (!Number.isFinite(got) || Math.abs(got - beta) > Math.max(eps, Math.abs(beta) * 1e-12)) {
      throw new ToolError(ERROR_CODES.E_BETA_DERIVATION_MISMATCH, {
        hint: `容器回传的 beta=${betaFromContainer} 与 Node 侧按 ${KL_FORMULA} 重算的 ${beta} 不一致 —— "β 可推导"未成立（不得用一个常数冒充推导结果）`,
        details: { betaFromContainer: betaFromContainer ?? null, recomputed: beta, sigmaR: s, klBudget: k },
        recoverable: false,
      })
    }
  }
  const vals = (kls ?? []).map(Number).filter((x) => Number.isFinite(x))
  const klMean = meanOf(vals)
  const deviationPct = (klMean !== null && k > 0) ? (Math.abs(klMean - k) / k) * 100 : null
  const withinTolerance = deviationPct !== null && Number.isFinite(Number(tolerancePct)) ? deviationPct <= Number(tolerancePct) : false
  const rec = recommendBeta({ betaDeclared: beta, klMean, klBudget: k })
  return {
    formula: KL_FORMULA,
    klBudget: k,
    sigmaR: s,
    betaDeclared: beta,
    klSteps: vals.length,
    klMean,
    klMin: vals.length ? Math.min(...vals) : null,
    klMax: vals.length ? Math.max(...vals) : null,
    klStd: stdOf(vals),
    deviationPct,
    tolerancePct: Number(tolerancePct),
    withinTolerance,
    recommendedBeta: rec.betaNext,
    // 「让惩罚在**预算处**等于 σ_r」的那个 β —— 它就是声明的 β（σ_r/KL*）。
    // 与 recommendedBeta 的区别见 recommendBeta 的注释：一个是"口径"，一个是"下一步该用多少"。
    betaAtBudget: beta,
    recommendation: rec,
    // 偏离超容差 ⇒ 重算 β 并**重新记账**（不静默调参）
    recalibrationRequired: deviationPct === null ? true : !withinTolerance,
    note: '本次 run 用的仍是声明的 beta（**不静默调参**）。recommendedBeta 是"下一次 run 该用的系数"：'
      + '在 KL ≈ c/β 的一阶模型下 β_next = β_declared × (实测 KL 均值 / KL*)，方向随实测 KL 与预算的偏离而定'
      + '（实测 KL 远低于预算 ⇒ **应减小** β 让策略敢动）。只记账，不中途改参。',
  }
}
