/**
 * persisted-integrity.spec —— **缺陷 15 的同类残留**（rl 侧）。
 *
 * 与 pretrain / sft 同源，但 rl 侧**多一个独立缺陷**：
 *
 * ① **剥掉完整性字段**：`summarize()` 丢掉 `lm`/`mmlu` 的 `attempted/succeeded/failed/partial`
 *    ⇒ 落盘后 `measurementCompleteness` 恒为 false。
 * ② **部件名与冻结词表不一致**：`summarize()` 写出 `preferenceTrain`，
 *    而 `measurementCompleteness` 的已知部件词表里是 **`prefTrain`**（另有 `prefVal`）。
 *    于是即使把 ① 修好，摘要仍会因 `unclassified-part:preferenceTrain` 判 `complete=false`。
 *
 * **实测（修复前）**：
 * ```
 * measurementCompleteness(rl.summarize(m))
 * ⇒ {"complete":false,
 *    "reason":"unclassified-part:preferenceTrain, lm:no-integrity-evidence, mmlu:no-integrity-evidence"}
 * ```
 *
 * ## 为什么②必须修在**插件侧**而不是扩词表
 * `measurementCompleteness` 的 `KNOWN_OTHER` 是**声明好的部件词表**（消费方契约）。
 * 生产者写了一个不在词表里的名字 ⇒ **是生产者的错**，不是词表漏了。
 * 扩词表会同时留下 `prefTrain`（无人用）与 `preferenceTrain`（新加）两个同名概念，
 * 并把"词表可以随便跟着生产者改"变成惯例 —— 那会让 fail-closed 的"未归类即报出"失去意义。
 *
 * 注：插件**内部**本来就用 `prefTrain` 这个名字（`measureAt` 返回 `{preference, prefTrain}`），
 * 只有 `summarize()` 的输出把它改名成 `preferenceTrain`，且**没有任何消费方**读那个持久化键。
 *
 * 不依赖 GPU / 容器 / 网络。
 */
import { summarize } from '../../src/core.mjs'
import { measurementCompleteness } from '../../src/evaluate.mjs'

export const NAME = '持久化测量的完整性证据（缺陷 15 同类残留·rl）'

const fullMeasurement = () => ({
  uniform: { ok: true, n: 12, epsilon: 0.01, delta: 0.02, bitIdenticalAcrossRuns: true },
  // 2026-10-02：`preference`/`prefTrain` 已移入 `INSTRUMENTED`（生产者 `evalPreference`
  // 已补 expected 证据字段）⇒ "完备测量"的夹具必须带上这些字段，否则它代表的是
  // "偏好侧没有证据"的旧形态，而不是"完备测量"。
  preference: { attempted: 48, succeeded: 48, failed: 0, skipped: 0, partial: false, n: 48, nOk: 48, meanReward: 0.5, truncated: 0, stopped: 0 },
  prefTrain: { attempted: 32, succeeded: 32, failed: 0, skipped: 0, partial: false, n: 32, nOk: 32, meanReward: 0.42 },
  lm: {
    attempted: 200, succeeded: 200, failed: 0, skipped: 0, partial: false,
    texts: 200, tokens: 5000, meanNll: 3.1, ppl: 22.2, perText: [3.0, 3.2],
  },
  mmlu: {
    attempted: 300, succeeded: 300, failed: 0, skipped: 0, partial: false,
    n: 300, correct: 120, accuracy: 0.4, asked: 300, okIndices: [],
  },
})

const partialMeasurement = () => {
  const m = fullMeasurement()
  m.lm = { ...m.lm, attempted: 260, succeeded: 200, failed: 60, partial: true }
  return m
}

export async function run(t) {
  t.assert(typeof summarize === 'function', 'summarize 必须可被 import 到 —— 拿不到就是在测空气')
  const s = summarize(fullMeasurement())
  t.assert(s !== null && typeof s === 'object', 'summarize 必须返回对象')

  // ── ① 持久化摘要必须**自证**完整 ──
  const chk = measurementCompleteness(s)
  t.assert(
    chk.complete === true,
    `完备测量经 summarize() 后必须 complete（实得 ${JSON.stringify(chk)}）`,
  )
  t.assert(chk.reason === null, `complete 时 reason 必须为 null（实得 ${JSON.stringify(chk.reason)}）`)

  // ── ② 逐仪器、逐字段 ──
  for (const part of ['lm', 'mmlu']) {
    const p = s[part]
    const src = fullMeasurement()[part]
    t.assert(p !== undefined && p !== null, `summarize 产物必须保留仪器 ${part}`)
    for (const f of ['attempted', 'succeeded', 'failed']) {
      t.assert(typeof p[f] === 'number', `${part}.${f} 必须是数（实得 ${typeof p[f]}）`)
      t.assert(p[f] === src[f], `${part}.${f} 必须**原样保留**（期望 ${src[f]}，实得 ${p[f]}）`)
    }
    t.assert(typeof p.partial === 'boolean', `${part}.partial 必须是布尔（实得 ${typeof p.partial}）`)
  }

  // ── ③ 部件名必须落在冻结词表内（`prefTrain`，不是 `preferenceTrain`） ──
  t.assert(
    !String(chk.reason ?? '').includes('unclassified-part'),
    `产物不得含未归类部件（实得 ${JSON.stringify(chk.reason)}）`
    + ' —— `preferenceTrain` 不在 measurementCompleteness 的已知部件词表里，应写成 `prefTrain`',
  )
  t.assert(
    Object.prototype.hasOwnProperty.call(s, 'prefTrain'),
    '**训练切分**的偏好测量必须以词表名 `prefTrain` 落盘',
  )
  t.assert(
    !Object.prototype.hasOwnProperty.call(s, 'preferenceTrain'),
    '不得同时保留旧名 `preferenceTrain`（否则同一概念两个键，词表失去意义）',
  )

  // ── ③b 旧记录仍不可认证（**如实**：这是 fail-closed 的正确行为，不是回归） ──
  const legacy = { uniform: { ok: true, n: 1 }, preference: { n: 48 }, preferenceTrain: { n: 32 }, lm: { texts: 200 }, mmlu: { n: 300 } }
  const chkLegacy = measurementCompleteness(legacy)
  t.assert(chkLegacy.complete === false, '修复前的旧记录（含 preferenceTrain、无完整性证据）必须**仍**不可认证')

  // ── ④ 反向对照：不完整测量不得被判为完整 ──
  const chkBad = measurementCompleteness(summarize(partialMeasurement()))
  t.assert(
    chkBad.complete === false,
    `不完整测量经 summarize() 后必须仍 complete=false（实得 ${JSON.stringify(chkBad)}）`,
  )
  t.assert(
    typeof chkBad.reason === 'string' && /failed=60\/260/.test(chkBad.reason),
    `理由必须**点名** failed=60/260（实得 ${JSON.stringify(chkBad.reason)}）`,
  )

  // ── ⑤ 不自证 ──
  t.assert(s.integrity !== undefined && s.integrity !== null, 'summarize 产物必须带 integrity 块')
  t.assert(
    typeof s.integrity?.producedBy === 'string' && s.integrity.producedBy !== '',
    `integrity.producedBy 必须是非空字符串（实得 ${JSON.stringify(s.integrity?.producedBy)}）`,
  )
  t.assert(s.integrity.producedBy !== 'effectJudgment', 'integrity.producedBy 不得是 effectJudgment（不自证）')
}
