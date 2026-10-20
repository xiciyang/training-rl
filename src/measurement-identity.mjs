/**
 * measurement-identity —— **判定仪器的身份**（缺陷 N2）与**评价集的身份**（缺陷 N1）的唯一归属地。
 *
 * 为什么把这两件事放在一个文件里：它们回答的是**同一个问题** ——
 * 「这次判决，是在**哪一个仪器**、**哪一个评估集**上测出来的？」
 * 基线缓存把它们一起编码进键；任何一项变了，旧缓存就**不可比**。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * N2：`prefMaxTokens`（偏好测量的生成长度上限）
 * ══════════════════════════════════════════════════════════════════════════
 * ## 缺陷（真机实测，不是假想）
 * `training_rl_validate` 的 schema 声明了 `additionalProperties: false`，而
 * `prefMaxTokens` **不在 `properties` 里** ⇒ **合规的 MCP 客户端会静默丢掉这个参数**
 * ⇒ 服务端回落到未标定档 256 ⇒ 该档的截断率闸门**必然失败** ⇒ 标准客户端**只能拿到
 * 一个假的"测量不成立"**。（上一轮它"能过"，只是因为 MCP server 不做 schema 校验 ——
 * 那是运气，不是设计。）
 *
 * ## 标定结论（本机证据：`inference-training/out/rl-instrument-calibration.json`）
 * | maxTokens | 策略截断 | 基座截断 | 过闸？ |
 * |---|---|---|---|
 * | 256 | 35/48 = 72.9% | 38/48 = 79.2% | ❌ 必失败 |
 * | 512 | 24/48 = **恰 50.00%** | 24/48 = **恰 50.00%** | ⚠️ knife-edge（等于闸门阈值本身） |
 * | **768** | 20/48 = 41.7% | 20/48 = 41.7% | ✅ **带真实余量** ⇒ 取 768 |
 * 截断率闸门阈值（0.5）与任何判据阈值**一个字都没动**，改的只是仪器参数。
 *
 * ## fail-closed（本模块的承诺）
 * 调用方省略时**只**回落到**已标定**的 768，且把 `source: 'calibrated-default'` 与
 * `calibrated: true` **记进结果** —— 回落是**可见**的，不是静默换仪器。
 * 显式传入时 `source: 'caller'`，并如实标注它是否是标定档（`calibrated`）。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * N1：提示集（评估集）身份
 * ══════════════════════════════════════════════════════════════════════════
 * ## 缺陷
 * 基线缓存键只含 `基座@revision@协议@采样规模@生成长度`，**不含提示集身份**。
 * 于是「**同样大小、内容不同**的另一个留出 prompt 集」会**命中同一条缓存基线** ——
 * 判定用的评估集换了，比较却照做。这**直接违反** `eval-contract` §7.1：
 * 规则 1「判定所用评估集必须钉死，与基线比对时两侧 `datasetHash` 必须相同」、
 * 规则 2「不同评估集的 Δ 不可跨集比较」、规则 3「评估集身份必须有哈希背书」。
 *
 * ## 身份口径（与既有约定对齐，不新造一套）
 * - **对文件字节取 sha256**，与 LT5 的 `lineage.evalFileHashes`（CI 不变量
 *   `inference-training/checks/backfill-eval-hashes.mjs --check`）**同口径**；
 * - **组合哈希的构造**与 `inference-contracts/checks/eval-set-identity.mjs#combinedOf` 同形状：
 *   `sha256(sorted("file:sha256(bytes)").join("\n"))`。
 * - 实际用于比较的是**磁盘上文件的真实字节**（那是"将要被评估的东西"的真相）；
 *   `lineage.evalFileHashes` 只作**交叉核对**，不一致就如实报出（`declaredMatches: false`），
 *   不做静默通过。
 *
 * ## fail-closed
 * 拿不到身份（文件缺失/不可读）⇒ `id === null`；`baselineCacheKey()` 对空身份**抛错**，
 * 调用方**不得**在无身份的键上复用或写入基线缓存。
 */
import { createHash } from 'node:crypto'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'

/* ══════════════════════════════════ N2 ══════════════════════════════════ */

/** 已标定的 `prefMaxTokens`（LT5 标定扫描的结论；`256`/`512` 的否决理由见文件头）。 */
export const CALIBRATED_PREF_MAX_TOKENS = 768

/** 标定的完整记录（可被审计/报告直接引用；**不只是一句注释**）。 */
export const PREF_MAX_TOKENS_CALIBRATION = Object.freeze({
  value: CALIBRATED_PREF_MAX_TOKENS,
  protocol: 'rl-pref-eval@2',
  truncationGate: 0.5,
  evidence: 'inference-training/out/rl-instrument-calibration.json',
  choiceRule: '取标定扫描中**两侧截断率都 ≤ MAX_TRUNCATION_RATIO(50%) 且带真实余量**的最小生成长度上限：'
    + '512 恰为 50.00%（knife-edge，换一个种子/子集就会翻闸），768 两侧均 41.7% ⇒ 选 768。'
    + '截断率闸门阈值(0.5)本身未改，改的只是仪器参数 maxTokens。',
  rejected: Object.freeze({
    256: '本机标定：策略截断 35/48=72.9%、基座 38/48=79.2%，均 > 50% ⇒ 截断率闸门必失败（"测量不成立"）',
    512: '本机标定：两侧各 24/48 = **恰 50.00%**，正好等于闸门阈值 ⇒ knife-edge，换种子/子集即翻闸，不取',
  }),
})

/**
 * 解析调用方给的 `prefMaxTokens` ⇒ `{ value, source, calibrated }`。
 *
 * **fail-closed**：省略时**只**回落到已标定档，且 `source='calibrated-default'`
 * ⇒ 回落**留在结果里**，不是静默换一个未标定的仪器。
 *
 * 抛错时用**纯 Error + 可读 code**（本模块不 import `version.mjs`，
 * 以免与 `core.mjs` 形成环）；调用方负责翻译成 `ToolError`。
 */
export function resolvePrefMaxTokens(value) {
  if (value === undefined || value === null) {
    return { value: CALIBRATED_PREF_MAX_TOKENS, source: 'calibrated-default', calibrated: true }
  }
  const n = Number(value)
  if (!Number.isInteger(n) || n <= 0) {
    throw new Error(`E_PREF_MAX_TOKENS_INVALID: prefMaxTokens 必须是正整数（实得 ${JSON.stringify(value)}）；`
      + `已标定档为 ${CALIBRATED_PREF_MAX_TOKENS}`)
  }
  return { value: n, source: 'caller', calibrated: n === CALIBRATED_PREF_MAX_TOKENS }
}

/* ══════════════════════════════════ N1 ══════════════════════════════════ */

/** 缓存键里标记"这一串是提示集身份"的分隔符（检测器按它判旧键）。 */
export const CACHE_KEY_IDENTITY_MARKER = '@ps='

/** 构成"提示集身份"的文件（训练 prompt 与留出 prompt —— 两者都进判定口径）。 */
export const PROMPT_SET_FILES = Object.freeze(['train.jsonl', 'heldout.jsonl'])

/** 组合哈希的构造与 `eval-set-identity.mjs#combinedOf` **同形状**（不新造一套口径）。 */
export function combinedFilesHash(hashes) {
  const lines = Object.keys(hashes).sort().map((f) => `${f}:${hashes[f]}`)
  return `sha256:${createHash('sha256').update(lines.join('\n')).digest('hex')}`
}

const sha256OfFile = (p) => `sha256:${createHash('sha256').update(readFileSync(p)).digest('hex')}`

/**
 * 计算一个提示集目录的**身份**。返回（全部字段都是纯 JSON，可进注册表/审计）：
 *   `{ id, hashBacked, files, datasetDir, declaredAt, declaredMatches, mismatched, reason }`
 * `id === null` ⇒ **身份不可得** ⇒ 调用方必须 fail-closed（不得复用/写入基线缓存）。
 */
export function promptSetIdentity(datasetDir, { files = PROMPT_SET_FILES } = {}) {
  const lineage = (() => {
    try { return JSON.parse(readFileSync(join(datasetDir, 'lineage.json'), 'utf8')) } catch { return null }
  })()
  const declared = lineage && lineage.evalFileHashes && typeof lineage.evalFileHashes === 'object'
    ? lineage.evalFileHashes
    : null

  const hashes = {}
  const missing = []
  for (const f of files) {
    try { hashes[f] = sha256OfFile(join(datasetDir, f)) } catch { missing.push(f) }
  }
  if (missing.length) {
    return {
      id: null, hashBacked: false, files: hashes, datasetDir,
      declaredAt: declared ? 'lineage.evalFileHashes' : null, declaredMatches: null, mismatched: [],
      reason: `prompt-set-files-unreadable:${missing.join(',')}`,
    }
  }
  const declaredFiles = declared ? files.filter((f) => typeof declared[f] === 'string') : []
  const mismatched = declaredFiles.filter((f) => declared[f] !== hashes[f])
  return {
    id: combinedFilesHash(hashes),
    hashBacked: true,
    files: hashes,
    datasetDir,
    // 声明来自哪里、是否覆盖全部文件、是否与磁盘逐条一致 —— 全部如实记录（缺失就是缺失）
    declaredAt: declared && declaredFiles.length === files.length ? 'lineage.evalFileHashes' : null,
    declaredMatches: declared && declaredFiles.length === files.length ? mismatched.length === 0 : null,
    mismatched,
  }
}

/**
 * 基线缓存键 —— **唯一构造处**。
 *
 * 组成：`基座@revision@评估协议@采样规模(t/v/lm/mmlu)@生成长度上限@提示集身份`。
 * 每加一个"判决可比性"的维度就必须进键；**缺提示集身份直接抛错**（fail-closed）：
 * 没有身份的键既不可查也不可写，否则"换个集比一下"会静默命中旧基线。
 */
export function baselineCacheKey({
  baseModel, revision = null, protocol, nTrain, nVal, nLm, nMmlu, prefMaxTokens, promptSetId,
}) {
  if (typeof protocol !== 'string' || protocol.length === 0) {
    throw new Error('E_EVAL_PROTOCOL_MISSING: 基线缓存键必须含评估协议版本（跨协议比较会静默错判）')
  }
  if (typeof prefMaxTokens !== 'number' || !Number.isFinite(prefMaxTokens)) {
    throw new Error('E_PREF_MAX_TOKENS_MISSING: 基线缓存键必须含生成长度上限（换仪器必须重测基线）')
  }
  if (typeof promptSetId !== 'string' || promptSetId.length === 0) {
    throw new Error('E_PROMPT_SET_IDENTITY_MISSING: 基线缓存键必须含提示集身份（eval-contract §7.1 规则 1/3）—— '
      + '否则"同样大小、内容不同"的另一个留出集会命中同一条缓存基线，两个不可比的 Δ 会被当成可比')
  }
  return `${baseModel}@${revision ?? 'norev'}@${protocol}`
    + `@t${nTrain}/v${nVal}/lm${nLm}/mmlu${nMmlu}/mt${prefMaxTokens}`
    + `${CACHE_KEY_IDENTITY_MARKER}${promptSetId}`
}

/** 旧键（**不含身份**）—— 只用于"存量条目必须被显式标记无效"的检测器识别，不用于写入。 */
export function cacheKeyHasPromptSetIdentity(key) {
  return typeof key === 'string' && key.includes(CACHE_KEY_IDENTITY_MARKER)
}
