/**
 * reward —— **规则奖励**（kind = `rule`）的唯一权威定义（JS 侧）。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 为什么是规则函数，而不是偏好数据 / RM（**不是省事，是 license 决定的**）
 * ══════════════════════════════════════════════════════════════════════════════
 * Step 0 只读探测实测（`[verified]`，见 training-rl-contract §4.5）：
 *   - `Anthropic/hh-rlhf` license 成立（MIT，三处一致），但**卡片正文限定用途**
 *     （"not meant for supervised training of dialogue agents"；
 *      red-team 子集"not meant for fine-tuning or preference modeling"）⇒ v0 不用；
 *   - `Dahoas/rm-static` **完全未声明 license**（无 `license` 字段、无 `license:*` tag）
 *     ⇒ **无据不可用**（"可下载 ≠ 有授权"）；
 *   - 本地模型根目录下**无任何 reward model**。
 * ⇒ v0 的奖励来源 = **规则函数**：0 GPU 小时、逐位确定性、无 license 风险。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 为什么必须有 `version`（`RULE_REWARD_VERSION`）
 * ══════════════════════════════════════════════════════════════════════════════
 * 奖励变了就是**另一个实验**：同一份权重在 reward@1 与 reward@2 下得到的分数不可比。
 * 故版本号进 `run.json`、进 registry、进审计链。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * ⚠️ 两处实现、一份规格（**本包唯一一处刻意的双实现**，附机器可验的防漂移闸门）
 * ══════════════════════════════════════════════════════════════════════════════
 * 训练时 GRPOTrainer 在**容器内 Python**里逐批调用奖励函数（跨越 IPC 逐条打分不可行），
 * 而评估时（serving 路径）打分发生在 **Node 侧**。于是同一规则必然有两处实现。
 * 与其假装"只有一份"，不如把"两份必须一致"变成**可执行的判据**：
 *   - `tests/fixtures/reward-vectors.json` 是**规格**（固定向量 + 期望值，由本文件生成）；
 *   - `tests/unit/reward-kl.spec.mjs` 断言本实现与规格逐条相同；
 *   - 每次真机训练时容器内的 Python 实现也会对**同一组向量**打分并回传，
 *     与 JS 实现逐条比对，不一致即 `E_REWARD_IMPL_DRIFT`（fail-closed）。
 * 浮点上两边都是 IEEE754 double 且运算顺序逐条对齐 ⇒ 期望**逐位相同**；
 * 判据用 1e-12 容差并同时回报实测最大偏差（不把"接近"说成"相同"）。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 规则本体（6 个分量；总分 = 算术平均 ⇒ r ∈ [0,1]）
 * ══════════════════════════════════════════════════════════════════════════════
 * 为什么**归一化到 [0,1]**：奖励的绝对尺度会直接进入 β 的推导（β = σ_r / KL*）。
 * 把 r 限制在 [0,1] 让 σ_r 有可解释的量纲（"质量分的标准差"），从而 β 也可解释。
 * 归一化**不改变** GRPO 的梯度（组内优势被标准化），只改变 β 的数值口径。
 *
 *   ① readable     : 非空且含字母/数字/CJK —— "它确实输出了内容"（0/1）
 *   ② no_restart   : 没把提示模板重新起头、长度未爆掉 —— "它在回答，而不是复读"（0/1）
 *   ③ length       : 3–48 词得满分；过短按 w/3 线性衰减；过长线性衰减 —— "简洁且完整"
 *   ④ repetition   : 3-gram 重复次数超过 3 起线性惩罚 —— "没有陷入复读循环"
 *   ⑤ relevance    : 提示的内容词在回答里出现的比例（≥0.6 即满分）—— **切题**
 *   ⑥ constraint   : 由**提示本身**用规则推导出的可判约束：满足 1 / 违反 0 / 无可判约束 0.5
 *
 * 六条都是**纯函数**：无采样、无随机、无外部状态 ⇒ 同输入同参数**逐位相同**
 * （training-rl-contract 冻结条件 2 的确定性要求）。
 *
 * 关于 ③④⑤ 为什么是**连续**而不是阈值：阈值型奖励在模型"已经答得不错"时会饱和
 * （所有样本同分 ⇒ σ_r = 0 ⇒ GRPO 优势恒为 0 ⇒ 训练是空转 —— 这是 GRPO 最隐蔽的失效形态）。
 * 连续分量把"质量差异"保留下来，从根上降低 σ_r=0 的概率；即便如此，
 * 真实 σ_r 仍然是**实测**的，且 σ_r=0 时训练**拒绝开始**（`E_REWARD_NO_VARIANCE`）。
 */

export const RULE_REWARD_VERSION = 'rule-reward@1'

/** 分量权重（全 1 ⇒ 总分 = 平均；显式写出，避免"隐式等权"成为隐藏约定）。 */
export const RULE_WEIGHTS = Object.freeze({
  readable: 1, no_restart: 1, length: 1, repetition: 1, relevance: 1, constraint: 1,
})

export const LENGTH_BAND = Object.freeze({ minWords: 3, maxWords: 48 })
export const MAX_CHARS = 2048
/** 相关性满分的门槛：提示内容词被覆盖的比例。 */
export const RELEVANCE_FULL_AT = 0.6
/** 3-gram 允许的最大重复次数；超过后线性惩罚。 */
export const MAX_TRI_REPEAT = 3

/**
 * 停用词表（**同时存在于本文件与容器内 Python**，属规格的一部分）。
 * 只滤掉"指令性/功能性"词，保留真正的内容词 —— 否则相关性分量会被 "write/answer" 这类词主导。
 */
export const STOPWORDS = Object.freeze([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'of', 'to', 'in', 'on', 'for', 'and', 'or',
  'but', 'with', 'as', 'at', 'by', 'from', 'into', 'that', 'this', 'these', 'those', 'it', 'its', 'your', 'you',
  'i', 'we', 'they', 'he', 'she', 'them', 'their', 'our', 'my', 'me', 'us', 'do', 'does', 'did', 'doing',
  'how', 'what', 'why', 'when', 'where', 'which', 'who', 'whom', 'can', 'could', 'should', 'would', 'will',
  'shall', 'may', 'might', 'must', 'not', 'no', 'yes', 'if', 'then', 'than', 'so', 'such', 'all', 'any', 'each',
  'more', 'most', 'some', 'other', 'only', 'also', 'very', 'just', 'about', 'after', 'before', 'between', 'out',
  'up', 'down', 'over', 'under', 'again', 'here', 'there', 'please', 'write', 'create', 'give', 'make', 'using',
  'use', 'used', 'answer', 'output', 'return', 'example', 'examples', 'following', 'list', 'explain',
  'provide', 'suggest', 'generate', 'describe', 'tell', 'help', 'need', 'want', 'like', 'get', 'got', 'one', 'two',
  'three', 'word', 'words', 'sentence', 'sentences', 'paragraph', 'text', 'number', 'numbers', 'json', 'code',
  'program', 'script', 'function', 'python', 'java', 'javascript', 'based', 'new', 'same',
])

const STOP = new Set(STOPWORDS)
const WORD_RE = /[A-Za-z0-9\u4e00-\u9fff]+/g
const ALNUM_RE = /[0-9A-Za-z\u4e00-\u9fff]/
const DIGIT_RE = /[0-9]/

/** 词序列（与 Python 侧 `re.findall(r"[A-Za-z0-9\u4e00-\u9fff]+", text.lower())` 同口径）。 */
export function words(text) {
  return String(text ?? '').toLowerCase().match(WORD_RE) ?? []
}

export function wordCount(text) {
  return words(text).length
}

/** 内容词（去停用词、去重）—— 相关性与"可判约束"都基于它。 */
export function contentWords(text) {
  const out = new Set()
  for (const w of words(text)) if (w.length >= 3 && !STOP.has(w)) out.add(w)
  return out
}

/** 3-gram 的最大重复次数（词数 < 6 时不可能有意义地重复 ⇒ 1）。 */
export function maxTriRepeat(text) {
  const ws = String(text ?? '').trim().split(/\s+/).filter(Boolean)
  if (ws.length < 6) return 1
  const counts = new Map()
  let best = 1
  for (let i = 0; i + 2 < ws.length; i += 1) {
    const g = `${ws[i]} ${ws[i + 1]} ${ws[i + 2]}`
    const c = (counts.get(g) ?? 0) + 1
    counts.set(g, c)
    if (c > best) best = c
  }
  return best
}

/**
 * 由**提示文本**用规则推导可判约束（不读标注、不看参考答案 —— 否则就不是规则奖励了）。
 * @returns {{id:string, kind:string, satisfied:(boolean|null)}} satisfied=null ⇒ 提示无可判约束
 */
export function deriveConstraint(prompt) {
  const p = String(prompt ?? '').toLowerCase()
  if (p.includes('one word') || p.includes('一个词') || p.includes('一个单词')) return { id: 'one-word', kind: 'one-word', satisfied: null }
  if (p.includes('json')) return { id: 'json', kind: 'json', satisfied: null }
  if (p.includes('number') || p.includes('数字')) return { id: 'number', kind: 'number', satisfied: null }
  return { id: 'none', kind: 'none', satisfied: null }
}

function constraintScore(prompt, text) {
  const { kind } = deriveConstraint(prompt)
  const t = String(text ?? '')
  if (kind === 'one-word') return words(t).length === 1 ? 1.0 : 0.0
  if (kind === 'json') { const s = t.trim(); return (s.startsWith('{') && s.endsWith('}')) ? 1.0 : 0.0 }
  if (kind === 'number') return DIGIT_RE.test(t) ? 1.0 : 0.0
  return 0.5 // 无可判约束 ⇒ 中性分（不奖励也不惩罚，避免把"自由回答"当成违规）
}

/**
 * 规则奖励本体。**确定性纯函数**：同 (completion, prompt) ⇒ 同分数（逐位相同）。
 * @param {string} completion 策略的输出文本
 * @param {string} prompt     对应的提示文本（只用于推导可判约束与相关性）
 * @returns {{value:number, parts:object}}
 */
export function ruleReward(completion, prompt = '') {
  const trimmed = String(completion ?? '').trim()

  // ① readable
  const readable = (trimmed.length > 0 && ALNUM_RE.test(trimmed)) ? 1.0 : 0.0
  // ② no_restart（含长度爆掉：超长几乎必然是复读或没在回答）
  const restarted = trimmed.includes('### Instruction:') || trimmed.includes('### Input:')
  const noRestart = (!restarted && trimmed.length > 0 && trimmed.length <= MAX_CHARS) ? 1.0 : 0.0
  // ③ length（连续）
  const w = wordCount(trimmed)
  let length
  if (w === 0) length = 0.0
  else if (w < LENGTH_BAND.minWords) length = w / LENGTH_BAND.minWords
  else if (w <= LENGTH_BAND.maxWords) length = 1.0
  else length = Math.max(0.0, 1.0 - (w - LENGTH_BAND.maxWords) / LENGTH_BAND.maxWords)
  // ④ repetition（连续）
  const mt = maxTriRepeat(trimmed)
  const repetition = mt <= MAX_TRI_REPEAT ? 1.0 : Math.max(0.0, 1.0 - (mt - MAX_TRI_REPEAT) / MAX_TRI_REPEAT)
  // ⑤ relevance（连续）
  const pw = contentWords(prompt)
  let relevance
  if (pw.size === 0) relevance = 0.5 // 提示无内容词（极短/全停用词）⇒ 中性，不冒充 0
  else {
    const cw = contentWords(trimmed)
    let hit = 0
    for (const x of pw) if (cw.has(x)) hit += 1
    relevance = Math.min(1.0, (hit / pw.size) / RELEVANCE_FULL_AT)
  }
  // ⑥ constraint
  const constraint = readable === 0.0 ? 0.0 : constraintScore(prompt, trimmed)

  const parts = { readable, no_restart: noRestart, length, repetition, relevance, constraint }
  const keys = Object.keys(RULE_WEIGHTS)
  const totalW = keys.reduce((a, k) => a + RULE_WEIGHTS[k], 0)
  const value = keys.reduce((a, k) => a + RULE_WEIGHTS[k] * parts[k], 0) / totalW
  return { value, parts }
}

/** 只要数值的便捷形态（批量打分的默认用法）。 */
export function rewardOf(completion, prompt = '') {
  return ruleReward(completion, prompt).value
}

/** 一批完成的打分（保持顺序）。 */
export function rewardBatch(completions, prompts) {
  return completions.map((c, i) => rewardOf(c, Array.isArray(prompts) ? prompts[i] : prompts))
}

/** 奖励来源描述符 —— 进 registry / 审计链 / run.json（可追溯"这一次用的是哪个奖励"）。 */
export function rewardDescriptor() {
  return {
    kind: 'rule',
    ref: RULE_REWARD_VERSION,
    version: RULE_REWARD_VERSION,
    deterministic: true,
    range: [0, 1],
    components: Object.keys(RULE_WEIGHTS),
    weights: { ...RULE_WEIGHTS },
    lengthBand: { ...LENGTH_BAND },
    relevanceFullAt: RELEVANCE_FULL_AT,
    maxTriRepeat: MAX_TRI_REPEAT,
    maxChars: MAX_CHARS,
    stopwordCount: STOPWORDS.length,
    note: '纯函数；无采样、无随机、无外部状态 ⇒ 同输入同参数逐位相同（契约冻结条件 2）',
  }
}

/** 覆盖判定分支的向量标签（规格与真机一致性检查都按它对齐）。 */
export const RULE_BRANCHES = Object.freeze([
  'readable-empty', 'readable-whitespace', 'readable-symbols-only', 'readable-ok',
  'restart-instruction', 'restart-input', 'restart-too-long', 'restart-ok',
  'repetition-degenerate', 'repetition-ok',
  'length-zero', 'length-short', 'length-band-low', 'length-band-mid', 'length-band-high', 'length-too-long',
  'relevance-full', 'relevance-partial', 'relevance-zero', 'relevance-no-content-words',
  'constraint-one-word-ok', 'constraint-one-word-violated',
  'constraint-number-ok', 'constraint-number-violated',
  'constraint-json-ok', 'constraint-json-violated',
  'constraint-absent-neutral',
])
