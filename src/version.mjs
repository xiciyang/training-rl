/**
 * version / 错误码闭集 —— training-rl。
 *
 * 错误码**闭集**（与 training-sft / training-pretrain / inference-diagnose 同一纪律：
 * 新增码 = schema 版本变更，须回文档修订）。不设"其他"桶：
 * 每个码对应一个**可独立失效、且失效语义不同**的依赖源或闸门。
 *
 * 文档修订记录（本文件即错误码的文档）：
 *   - 追加 `E_INSTRUMENT_LOSSY`：**additive**，未改任何既有码的语义、未改 SCHEMA_VERSION。
 *     语义 = "测量仪器经过 serving 的有损降级档位，故本次测量不得被签收"（fail-closed 闸门）。
 */
export const PACKAGE_VERSION = '0.1.0'
export const SCHEMA_VERSION = '1'
export const MCP_INTERFACE_VERSION = '1.0.0'
export const AUDIT_SCHEMA = 'training-rl/audit@1'
export const REGISTRY_SCHEMA = 'training-rl/registry@1'

/** 规则奖励的显式版本号（改规则 = 改实验，必须换版本号并记进 run.json）。 */
export const RULE_REWARD_VERSION = 'rule-reward@1'

/** RL 训练工具链镜像（= training-sft/toolchain:0.1.0 + trl==1.14.0，已构建并真机验证）。 */
export const RL_IMAGE = 'training-rl/smoke-toolchain:0.1.0'

export const ERROR_CODES = Object.freeze({
  // 参数与输入
  E_MISSING_PARAM: 'E_MISSING_PARAM',
  E_PROMPTS_INVALID: 'E_PROMPTS_INVALID',
  E_PROMPT_LEAK: 'E_PROMPT_LEAK',                   // 训练 prompt 与评估 prompt 内容哈希相交
  // 奖励与 KL 推导（本包特有的两条可推导性闸门）
  E_KL_BUDGET_REQUIRED: 'E_KL_BUDGET_REQUIRED',     // KL* 必须由调用方显式给出（不内置默认值）
  E_REWARD_NO_VARIANCE: 'E_REWARD_NO_VARIANCE',     // σ_r = 0 ⇒ GRPO 优势恒为 0，训练是空转 ⇒ 拒绝
  E_REWARD_IMPL_DRIFT: 'E_REWARD_IMPL_DRIFT',       // JS 与容器内 Python 的规则实现不一致
  E_BETA_DERIVATION_MISMATCH: 'E_BETA_DERIVATION_MISMATCH', // 容器回传的 β ≠ Node 侧按公式重算的 β（"可推导"未成立）
  // 依赖源
  E_BASE_MODEL_NOT_FOUND: 'E_BASE_MODEL_NOT_FOUND',
  E_IMAGE_UNAVAILABLE: 'E_IMAGE_UNAVAILABLE',
  E_SANDBOX_UNAVAILABLE: 'E_SANDBOX_UNAVAILABLE',
  E_SERVING_UNAVAILABLE: 'E_SERVING_UNAVAILABLE',
  // 预算
  E_BUDGET_EXCEEDED: 'E_BUDGET_EXCEEDED',           // 超限 → 暂停并报告（不降级）
  // 状态
  E_STATE_CONFLICT: 'E_STATE_CONFLICT',
  E_RUN_NOT_FOUND: 'E_RUN_NOT_FOUND',
  E_CONFIRMATION_REQUIRED: 'E_CONFIRMATION_REQUIRED',
  // 门禁
  E_GATE_FAILED: 'E_GATE_FAILED',
  E_INSTRUMENT_LOSSY: 'E_INSTRUMENT_LOSSY',         // 测量仪器走了 serving 的有损降级档位（fail-closed）
  // 产物级断言（training-core 的 E_NO_OP_TRAINING / E_ARTIFACT_FINGERPRINT_MISSING 的包内码）
  E_NO_OP_TRAINING: 'E_NO_OP_TRAINING',
  E_ARTIFACT_FINGERPRINT_MISSING: 'E_ARTIFACT_FINGERPRINT_MISSING',
  // 输出契约
  E_OUTPUT_NOT_LOSSLESS: 'E_OUTPUT_NOT_LOSSLESS',
  E_INTERNAL: 'E_INTERNAL',
})

export class ToolError extends Error {
  constructor(code, body = {}) {
    super(`${code}: ${body.hint || ''}`)
    this.name = 'ToolError'
    this.code = code
    this.body = body
  }
}

export const ALL_CODES = Object.freeze(Object.values(ERROR_CODES))
