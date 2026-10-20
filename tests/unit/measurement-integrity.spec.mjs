/**
 * measurement-integrity.spec —— **缺陷 14 回归测试**：测量完整性必须可判定。
 *
 * 背景（长任务 5 阶段 ⑥ 第 1 轮实测）：base 侧 `evalLM` 只拿到 5/100 条，
 * 而产物上 `texts=5` **与"评估集本来只有 5 条"不可区分** ⇒ 端点中途失效可以伪装成小样本。
 *
 * 本 spec 断言（rl 门禁）（全部不依赖 GPU / 容器 / 网络）：
 *   ① **端点失效**（failed 高）必须被判为"测量不完整"，而不是"样本不足"；
 *   ② **评估集小**（attempted 本身就少、failed=0）必须被判为"样本不足"；
 *   ③ 两者**理由不同** —— 即产物能区分病因；
 *   ④ 配对必须按**成功行原始下标的交集**，不是按数组位置（否则一侧跳行会让其后每对都错位）。
 */
import { forgettingGate } from '../../src/evaluate.mjs'

export const NAME = '测量完整性（缺陷 14 回归·rl）'

const lmOk = (n, { offset = 0, failed = 0, attempted = null, val = 3 } = {}) => ({
  attempted: attempted ?? n + failed,
  succeeded: n, failed, skipped: 0,
  partial: failed > 0,
  okIndices: Array.from({ length: n }, (_, i) => offset + i),
  perText: Array.from({ length: n }, () => val),
  meanNll: val, ppl: Math.exp(val), texts: n,
})
const mmluOk = (n, acc = 0.4) => ({ attempted: n, succeeded: n, failed: 0, n, accuracy: acc, correct: Math.round(n * acc), okIndices: [] })

export async function run(t) {
  // ── ① 端点失效：base 只成功 5/100 ──
  const a = forgettingGate({
    baseMmlu: mmluOk(100), rlMmlu: mmluOk(100),
    baseLm: lmOk(5, { failed: 95 }), rlLm: lmOk(100),
  })
  t.assert(a.pass === false, '端点失效时门禁必须**不通过**')
  const aText = a.reasons.join(' | ')
  t.assert(/测量不完整/.test(aText), `端点失效必须被报成"测量不完整"，实得：${aText}`)
  t.assert(/failed=95/.test(aText), '必须报出 failed 的具体数量（否则无法定位）')
  t.assert(!/配对样本不足/.test(aText), '端点失效**不得**被误报成"配对样本不足"（那是评估集小的说法）')

  // ── ② 评估集小：两侧都只有 10 条，failed=0 ──
  const b = forgettingGate({
    baseMmlu: mmluOk(100), rlMmlu: mmluOk(100),
    baseLm: lmOk(10), rlLm: lmOk(10),
  })
  t.assert(b.pass === false, '评估集小时门禁必须**不通过**')
  const bText = b.reasons.join(' | ')
  t.assert(/配对样本不足/.test(bText), `评估集小必须报"配对样本不足"，实得：${bText}`)
  t.assert(!/测量不完整/.test(bText), '评估集小**不得**被误报成"测量不完整"')

  // ── ③ 可区分：同一批数据下两种病因给出**不同的理由** ──
  t.assert(aText !== bText, '两种病因的理由必须不同（否则产物无法区分病因）')

  // ── ④ 配对按成功行**下标交集**，不是按位置 ──
  // base 成功于下标 0..59，sft 成功于下标 10..69 ⇒ 交集 = 10..59 = **50** 条。
  // 旧的按位置配对会得到 min(60,60)=60 —— 且其后每一对都错位。
  const c = forgettingGate({
    baseMmlu: mmluOk(100), rlMmlu: mmluOk(100),
    baseLm: lmOk(60, { offset: 0 }), rlLm: lmOk(60, { offset: 10 }),
  })
  t.assert(c.lm !== null, '交集 50 ≥ 30，LM 判据应当成立')
  t.assert(c.lm?.n === 50, `配对样本数必须等于下标交集 50，实得 ${c.lm?.n}（按位置配对会得到 60）`)
  t.assert(c.lm?.pairedOn === 'okIndices-intersection', '必须标明配对依据是下标交集')

  // ── ⑤ 完全无交集 ⇒ 不得判定为通过（配对为 0 必须 fail-closed） ──
  const d = forgettingGate({
    baseMmlu: mmluOk(100), rlMmlu: mmluOk(100),
    baseLm: lmOk(60, { offset: 0 }), rlLm: lmOk(60, { offset: 1000 }),
  })
  t.assert(d.pass === false, '两侧成功行完全不相交时必须**不通过**')
  t.assert(d.lm === null, '无交集时 lm 必须为 null（不得用 0 样本算出"通过"）')

  // ── ⑥ 向后兼容：旧产物没有 okIndices 时退化为按位置并**标明**，不得静默 ──
  const legacyBase = { perText: Array.from({ length: 40 }, () => 3), meanNll: 3, ppl: Math.exp(3) }
  const legacySft = { perText: Array.from({ length: 40 }, () => 3), meanNll: 3, ppl: Math.exp(3) }
  const e = forgettingGate({ baseMmlu: mmluOk(100), rlMmlu: mmluOk(100), baseLm: legacyBase, rlLm: legacySft })
  t.assert(e.lm?.pairedOn === 'positional-legacy', '无 okIndices 时必须标明退化为 positional-legacy')

  // ── ⑦ 完整性信息必须进入判定产物（不只是返回值） ──
  t.assert(c.lm?.attempted && typeof c.lm.attempted.base === 'number', 'lm 判据必须携带 attempted（门禁基于它判）')
  t.assert(c.lm?.failed && typeof c.lm.failed.base === 'number', 'lm 判据必须携带 failed')
  t.assert(c.mmlu?.attempted && typeof c.mmlu.attempted.base === 'number', 'mmlu 判据必须携带 attempted')
}
