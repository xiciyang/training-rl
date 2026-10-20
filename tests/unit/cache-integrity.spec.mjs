/**
 * cache-integrity.spec —— **缺陷 15 回归测试**：部分测量不得进入/复用于基线缓存。
 *
 * 背景（长任务 5 阶段 ⑥ 第 1 轮实测）：一次 base 侧只成功 5/100、MMLU 0 条的测量，
 * 被按 `task100/lm100/mmlu100` 写进了 `baselineCache`。**此后任何同键 run 都会复用它** ——
 * 一次仪器故障被**固化**成永久基线，且无人能看出它是坏的。
 *
 * 本 spec 断言的核心纪律：**可复用的前提是"有证据证明它完整"，不是"看起来像完整"**
 * （`未知 ≠ 完整`，fail-closed）。全部不依赖 GPU / 容器 / 网络。
 */
import { measurementCompleteness, attachIntegrity } from '../../src/evaluate.mjs'

export const NAME = '基线缓存完整性（缺陷 15 回归）'

const good = () => ({
  lm: { attempted: 200, succeeded: 200, failed: 0, partial: false, texts: 200, perText: [] },
  mmlu: { attempted: 300, succeeded: 300, failed: 0, partial: false, n: 300 },
  // `task` 自 v0.3.0 起属于**逐条网络测量**（`evalTask` 每行发一次请求）⇒ 必须有完整性证据。
  // 这里的夹具随之改成 `evalTask` **真实的**新形状（不是放宽断言，是让夹具与现实一致）。
  task: { attempted: 100, succeeded: 100, failed: 0, partial: false, n: 100, nOk: 100 },
  uniform: { ok: true },
})

export async function run(t) {
  // ① 完整（带证据）⇒ 可复用
  const g = measurementCompleteness(good())
  t.assert(g.complete === true, `带完整性证据的完整测量应可复用，实得 ${JSON.stringify(g)}`)

  // ② **第 1 轮的真实污染形态**：lm 只成功 5、mmlu 0 条 ⇒ 必须判为不可复用
  const poisoned = { lm: { attempted: 100, succeeded: 5, failed: 95, partial: true, texts: 5 }, mmlu: { attempted: 100, succeeded: 0, failed: 100, partial: true, n: 0 }, task: { n: 100 } }
  const p = measurementCompleteness(poisoned)
  t.assert(p.complete === false, '被污染的部分测量必须判为**不可复用**')
  t.assert(/lm:failed=95\/100/.test(p.reason), `理由必须点名 lm 的失败数，实得：${p.reason}`)
  t.assert(/mmlu:failed=100\/100/.test(p.reason), `理由必须点名 mmlu 的失败数，实得：${p.reason}`)

  // ③ **旧条目（无完整性证据）⇒ 也不可复用**（"未知 ≠ 完整"）
  const legacy = { lm: { texts: 200, perText: [] }, mmlu: { n: 300 }, task: { n: 100 } }
  const l = measurementCompleteness(legacy)
  t.assert(l.complete === false, '没有 attempted/failed 的旧条目必须判为**不可复用**（未知不等于完整）')
  t.assert(/no-integrity-evidence/.test(l.reason), `理由应说明缺证据，实得：${l.reason}`)

  // ④ 已被显式标记无效的 ⇒ 一律不可复用（即使数字看起来完整）
  const marked = { ...good(), invalid: true, invalidReason: 'lm:failed=95/100' }
  t.assert(measurementCompleteness(marked).complete === false, '被显式标记 invalid 的条目必须不可复用')
  t.assert(/marked-invalid/.test(measurementCompleteness(marked).reason), '理由应说明是被显式标记的')

  // ⑤ `attachIntegrity` 必须**修好**证据缺失：写入时附证据 ⇒ 之后可复用（往返）
  const attached = attachIntegrity(good())
  t.assert(measurementCompleteness(attached).complete === true, 'attachIntegrity 之后必须可复用（否则缓存永远失效）')
  t.assert(attached.integrity?.lm?.attempted === 200, 'integrity 必须记录 lm.attempted')

  // ⑥ 只有探针（uniform）、没有逐条测量 ⇒ **不得**声称完整
  // 只有探针 ⇒ 必须**一个 `INSTRUMENTED` 部件都没有**。
  // 注意：`task` 已不再是"非逐条测量"，把它放进来就不再是"只有探针"的场景了 —— 夹具随之收紧。
  const probeOnly = { uniform: { ok: true } }
  const po = measurementCompleteness(probeOnly)
  t.assert(po.complete === false, '只有有效性探针、没有逐条测量时不得判为完整')
  t.assert(/no-instrumented-parts/.test(po.reason), `理由应为 no-instrumented-parts，实得：${po.reason}`)

  // ⑦ 出现**未知部件**时必须显式报出（不得静默跳过）
  const unknownPart = { ...good(), mystery: { foo: 1 } }
  const u = measurementCompleteness(unknownPart)
  t.assert(u.complete === false, '出现未归类部件时必须 fail-closed')
  t.assert(/unclassified-part:mystery/.test(u.reason), `理由必须点名未知部件，实得：${u.reason}`)

  // ⑧ 空对象 / 非对象 ⇒ 不可复用（fail-closed）
  t.assert(measurementCompleteness({}).complete === false, '空对象必须不可复用')
  t.assert(measurementCompleteness(null).complete === false, 'null 必须不可复用')

  // ⑨ 只坏一侧也必须拦（不能因为 mmlu 好就放过 lm）
  const oneBad = { ...good(), lm: { attempted: 100, succeeded: 60, failed: 40, partial: true } }
  const ob = measurementCompleteness(oneBad)
  t.assert(ob.complete === false, '一侧不完整即判不可复用')
  t.assert(/lm:failed=40\/100/.test(ob.reason), '理由必须点名是哪一侧坏了')
}
