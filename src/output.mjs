/**
 * output —— DSH 工具契约的 `output { schema, render }` 规格 + **无损 JSON 闸门**。
 *
 * ## 两个踩过的坑，都在这里设防
 *
 * **坑 1（registration）**：`ctx.tools.register(tool)` 要求 output **同时**给出 `schema` 与 `render`；
 *   只给 schema 或不给 output 会让**整条 entry 被拒**（inference-tuning 曾因此完全挂不上）。
 *   ⇒ 所有工具共用本文件的构造器，不手写。
 *
 * **坑 2（output contract，长任务 2 缺陷 3）**：**注册成功 ≠ 输出契约成立**。
 *   真实 DSH 用"无损 JSON"校验器拒掉含 `undefined` 键的返回
 *   （`JSON.stringify` 静默丢掉 `undefined`，校验器却拒绝 ⇒ 静默不一致）。
 *   长任务 3 阶段 2 的 `health()` 就曾因为一个 `undefined` 键被 CI 红掉。
 *   ⇒ 本文件提供 `assertLossless()`：**在工具边界**递归检查返回值，
 *      发现 `undefined` / `function` / `symbol` / `bigint` / 非有限数 / 循环引用 **立即响亮失败**
 *      （抛 `E_OUTPUT_NOT_LOSSLESS`），而不是等 DSH 用一个看不懂的错误拒掉。
 */
import { ToolError, ERROR_CODES } from './version.mjs'

export const ROOT_PARAM = { type: 'string', description: '共享状态根目录。默认 ~/.ai-infra，可用 TRAINING_RL_ROOT 覆盖。' }

export const CONFIRM_PARAM = { type: 'string', description: '确认令牌。首次调用若需确认会返回 E_CONFIRMATION_REQUIRED 并在 details.confirmToken 给出该值；把原值回传即可继续。' }

/**
 * 递归校验"这个值能不能无损穿过 JSON"。只读遍历，**不修改**输入
 * （不在校验器里"顺手修好"—— 修好会掩盖生产端的缺陷）。
 */
export function findLosslessProblems(value, { path = '$', seen = new Set(), problems = [], depth = 0 } = {}) {
  if (depth > 64) { problems.push({ path, reason: '嵌套过深（>64），疑似循环或异常结构' }); return { ok: false, problems } }
  if (value === undefined) { problems.push({ path, reason: 'undefined 不是 JSON 值（JSON.stringify 会静默丢弃它）' }); return { ok: false, problems } }
  const t = typeof value
  if (t === 'function') { problems.push({ path, reason: 'function 不是 JSON 值' }); return { ok: false, problems } }
  if (t === 'symbol') { problems.push({ path, reason: 'symbol 不是 JSON 值' }); return { ok: false, problems } }
  if (t === 'bigint') { problems.push({ path, reason: 'bigint 不是 JSON 值（需显式转 string/number）' }); return { ok: false, problems } }
  if (t === 'number' && !Number.isFinite(value)) { problems.push({ path, reason: `非有限数 ${String(value)}（会被 JSON 序列化成 null）` }); return { ok: false, problems } }
  if (value === null || t !== 'object') return { ok: problems.length === 0, problems }
  if (seen.has(value)) { problems.push({ path, reason: '循环引用' }); return { ok: false, problems } }
  seen.add(value)
  if (Array.isArray(value)) {
    value.forEach((v, i) => findLosslessProblems(v, { path: `${path}[${i}]`, seen, problems, depth: depth + 1 }))
  } else {
    for (const [k, v] of Object.entries(value)) findLosslessProblems(v, { path: `${path}.${k}`, seen, problems, depth: depth + 1 })
  }
  seen.delete(value)
  return { ok: problems.length === 0, problems }
}

export function assertLossless(value, { tool = null } = {}) {
  const r = findLosslessProblems(value)
  if (!r.ok) {
    throw new ToolError(ERROR_CODES.E_OUTPUT_NOT_LOSSLESS, {
      hint: `工具${tool ? ` ${tool}` : ''} 的返回值不是无损 JSON：${r.problems.slice(0, 6).map((p) => `${p.path} → ${p.reason}`).join('；')}`,
      details: { tool, problems: r.problems.slice(0, 20), count: r.problems.length },
      recoverable: false,
    })
  }
  return value
}

export function passthroughOutput() {
  return {
    schema: { type: 'object', additionalProperties: true },
    render(_args, value) {
      return [{ type: 'text', text: JSON.stringify(value ?? null, null, 2) }]
    },
  }
}
