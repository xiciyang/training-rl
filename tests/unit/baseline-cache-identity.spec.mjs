/**
 * baseline-cache-identity.spec —— **N2 / N1 两个缺陷的回归测试**（无 GPU / 无容器 / 无网络）。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * N2：`prefMaxTokens` 必须真的出现在 tool schema 里，且缺省**只**回落到已标定档
 * ══════════════════════════════════════════════════════════════════════════
 * 缺陷形态（真机）：schema 声明了 `additionalProperties: false`，而 `prefMaxTokens`
 * **不在 `properties`** ⇒ **合规的 MCP 客户端把它静默丢掉** ⇒ 服务端回落到未标定档 256
 * ⇒ 该档的截断率闸门必失败（本机标定 72.9%/79.2%）⇒ 标准客户端**只能拿到假的"测量不成立"**。
 *
 * 本 spec **模拟一个严格客户端**（按 schema 丢弃未声明字段），断言它仍拿到**已标定档**；
 * 断言**对着 schema 本身**（不是对着注释）—— 谁把 `properties.prefMaxTokens` 删掉，
 * 严格客户端就会丢掉该字段，`source` 由 `caller` 退化成 `calibrated-default` ⇒ **变红**。
 *
 * ══════════════════════════════════════════════════════════════════════════
 * N1：基线缓存键必须含**提示集身份**
 * ══════════════════════════════════════════════════════════════════════════
 * 缺陷形态：键只含 `基座@revision@协议@采样规模@生成长度`，**不含提示集身份**
 * ⇒ 「同样大小、内容不同」的另一个留出集会命中同一条缓存基线 ⇒ 违反
 * `eval-contract` §7.1（规则 1 钉死 / 规则 2 不可跨集比较 / 规则 3 哈希背书）。
 * 本 spec 造两个**等长但内容不同**的提示集，断言两者的缓存键**不相等**；
 * 并断言缺身份时构造**抛错**（fail-closed，不得退化成"更短的键"）。
 */
import { mkdtempSync, mkdirSync, writeFileSync, readFileSync, rmSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { ALL_TOOL_SPECS, findToolSpec } from '../../src/tools.mjs'
import {
  CALIBRATED_PREF_MAX_TOKENS, PREF_MAX_TOKENS_CALIBRATION, resolvePrefMaxTokens,
  promptSetIdentity, baselineCacheKey, cacheKeyHasPromptSetIdentity, CACHE_KEY_IDENTITY_MARKER,
} from '../../src/measurement-identity.mjs'

export const NAME = '基线缓存身份（N2 仪器参数 schema / N1 提示集身份）'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')

/**
 * 模拟一个**严格客户端**：`additionalProperties: false` 的语义就是"未声明的字段不传"。
 * 注意：这里按**被检对象自己的 schema** 过滤，而不是按一张写死的字段表。
 */
function strictClientFilter(args, schema) {
  const declared = schema?.properties ?? {}
  const out = {}
  for (const k of Object.keys(args)) if (Object.prototype.hasOwnProperty.call(declared, k)) out[k] = args[k]
  return out
}

const lineageOf = (dir, extra = {}) => {
  writeFileSync(join(dir, 'lineage.json'), `${JSON.stringify({ schema: 'ai-infra/prompt-set-lineage@1', allClear: true, ...extra }, null, 2)}\n`)
}

/** 建一个提示集：`train.jsonl` / `heldout.jsonl` **长度相同、内容不同**。 */
function makeSet(root, name, { mark }) {
  const dir = join(root, name)
  mkdirSync(dir, { recursive: true })
  // 行内字符数刻意保持**逐字相同**：这样"文件大小相同"也成立 ⇒ 大小/条数无法区分两份集
  const line = (i, m) => JSON.stringify({ instruction: `${m}-${i}`.padEnd(24, 'x'), h: `h${i}` })
  writeFileSync(join(dir, 'train.jsonl'), `${Array.from({ length: 4 }, (_, i) => line(i, `${mark}T`)).join('\n')}\n`)
  writeFileSync(join(dir, 'heldout.jsonl'), `${Array.from({ length: 3 }, (_, i) => line(i, `${mark}V`)).join('\n')}\n`)
  lineageOf(dir)
  return dir
}

export async function run(t) {
  /* ══════════════════ N2 ══════════════════ */

  const spec = findToolSpec('training_rl_validate')
  t.assert(Boolean(spec), '找到 training_rl_validate 的 spec')
  const schema = spec.parameters
  t.assert(schema?.additionalProperties === false,
    '本 spec 的前提成立：training_rl_validate 的 schema 是 additionalProperties:false（合格客户端会丢未声明字段）')

  // ① **对着 schema 本身**断言：prefMaxTokens 必须被声明（删掉即红 —— 不依赖任何注释）
  const prop = schema?.properties?.prefMaxTokens
  t.assert(Boolean(prop), 'schema.properties.prefMaxTokens **已声明**（未声明 ⇒ 合规客户端会静默丢掉它）')
  t.assert(prop?.type === 'integer', `prefMaxTokens 类型为 integer（实得 ${JSON.stringify(prop?.type)}）`)

  // ② 严格客户端：带上已标定值 ⇒ 该字段必须**活下来**
  const rawArgs = { runId: 'rl_x', thresholds: { minRewardGainPct: 2 }, prefMaxTokens: CALIBRATED_PREF_MAX_TOKENS }
  const accepted = strictClientFilter(rawArgs, schema)
  t.assert(accepted.prefMaxTokens === CALIBRATED_PREF_MAX_TOKENS,
    `严格客户端仍把 prefMaxTokens 送到服务端（实得 ${JSON.stringify(accepted.prefMaxTokens)}）—— `
    + '若 schema 删掉该属性，这里会变成 undefined ⇒ 断言变红')
  t.assert(Object.keys(accepted).length === 3, `严格客户端保留了全部 3 个已声明参数（实得 ${Object.keys(accepted).join(',')}）`)

  // ③ 服务端解析：来自调用方 ⇒ source=caller；且必须是**已标定档**
  const fromCaller = resolvePrefMaxTokens(accepted.prefMaxTokens)
  t.assert(fromCaller.value === CALIBRATED_PREF_MAX_TOKENS, `解析出的档位 = 已标定档 ${CALIBRATED_PREF_MAX_TOKENS}（实得 ${fromCaller.value}）`)
  t.assert(fromCaller.source === 'caller', `schema 送达时 source=caller（实得 ${fromCaller.source}）—— 用它区分"调用方给的"与"我们回落的"`)
  t.assert(fromCaller.calibrated === true, '已标定档 calibrated=true')

  // ④ **对照（红条件）**：若 schema 未声明该属性，严格客户端会丢掉它。
  //    服务端此时**只**回落到已标定档（fail-closed），但 `source` 必须变成 calibrated-default
  //    ⇒ 上面第 ②/③ 条断言会红。这就是"删掉 schema 属性 ⇒ 测试变红"的机制。
  const afterDrop = strictClientFilter(rawArgs, { properties: Object.fromEntries(Object.entries(schema.properties).filter(([k]) => k !== 'prefMaxTokens')) })
  t.assert(afterDrop.prefMaxTokens === undefined, '对照：schema 里没有该属性时，严格客户端**确实**会丢掉它（证明本测试的模拟是有效的）')
  const fallback = resolvePrefMaxTokens(afterDrop.prefMaxTokens)
  t.assert(fallback.source === 'calibrated-default' && fromCaller.source !== fallback.source,
    '对照：丢掉后 `source` 与"送达"路径不同（caller ≠ calibrated-default）⇒ 删属性必然让断言红')

  // ⑤ fail-closed：省略**不得**回落到未标定档 256；回落必须是**已标定档**且标记来源
  const omitted = resolvePrefMaxTokens(undefined)
  t.assert(omitted.value === CALIBRATED_PREF_MAX_TOKENS, `省略时回落到已标定档 ${CALIBRATED_PREF_MAX_TOKENS}（实得 ${omitted.value}）`)
  t.assert(omitted.value !== 256, '省略时**不得**回落到未标定档 256（该档截断率必然过不了闸门）')
  t.assert(omitted.source === 'calibrated-default' && omitted.calibrated === true,
    '省略时的回落**可见**：source=calibrated-default、calibrated=true（不静默换仪器）')
  t.assert(resolvePrefMaxTokens(null).source === 'calibrated-default', 'null 与 undefined 同处理（都是"没给"）')

  // ⑥ 非法值必须响亮失败（不得静默取整/取默认）
  for (const bad of [0, -1, 1.5, 'abc', {}]) {
    let threw = false
    try { resolvePrefMaxTokens(bad) } catch { threw = true }
    t.assert(threw, `非法 prefMaxTokens=${JSON.stringify(bad)} 必须抛错（E_PREF_MAX_TOKENS_INVALID）`)
  }

  // ⑦ 标定记录本身可审计（值/证据/被否决的档位与理由）
  t.assert(PREF_MAX_TOKENS_CALIBRATION.value === 768 && CALIBRATED_PREF_MAX_TOKENS === 768,
    '标定值 = 768（LT5 标定扫描的结论）')
  t.assert(typeof PREF_MAX_TOKENS_CALIBRATION.evidence === 'string' && /rl-instrument-calibration\.json$/.test(PREF_MAX_TOKENS_CALIBRATION.evidence),
    '标定记录带**可复核的证据路径**（不是一句注释）')
  t.assert(Boolean(PREF_MAX_TOKENS_CALIBRATION.rejected[512]) && Boolean(PREF_MAX_TOKENS_CALIBRATION.rejected[256]),
    '被否决的两档（512 knife-edge / 256 必失败）各有理由')
  t.assert(/50\.00%/.test(PREF_MAX_TOKENS_CALIBRATION.rejected[512]), '512 的否决理由写明"恰 50.00%"')
  t.assert(ALL_TOOL_SPECS.filter((x) => x.name === 'training_rl_validate').length === 1, 'training_rl_validate 只有一份 spec（无分叉）')

  // ⑧ **判决路径本身不得绕开解析器**（否则 schema 修好了、服务端仍可能用未标定档）
  //    对着**源码**断言"没有把 256 写回默认值 / 真的走唯一构造处" —— 这是对"回退修复"的防线。
  const coreSrc = readFileSync(join(PKG, 'src', 'core.mjs'), 'utf8')
  t.assert(!/prefMaxTokens\s*=\s*DEFAULT_MAX_TOKENS/.test(coreSrc),
    'core.mjs **不得**再把未标定档 (`DEFAULT_MAX_TOKENS`=256) 写成 prefMaxTokens 的默认值（N2 的根因）')
  t.assert(/resolvePrefMaxTokens\(prefMaxTokens\)/.test(coreSrc),
    'validate 真的经 `resolvePrefMaxTokens` 解析（解析逻辑只有一份，不各写一套）')
  t.assert(/baselineCacheKey\(\{/.test(coreSrc) && !/cacheKey = `\$\{run\.baseModel\.path\}/.test(coreSrc),
    '缓存键由 `baselineCacheKey` 唯一构造（不得在 core 里手拼字符串 —— 手拼就会漏维度）')

  /* ══════════════════ N1 ══════════════════ */

  const tmp = mkdtempSync(join(tmpdir(), 'rl-identity-'))
  try {
    const setA = makeSet(tmp, 'setA', { mark: 'AAAA' })
    const setB = makeSet(tmp, 'setB', { mark: 'BBBB' })
    const idA = promptSetIdentity(setA)
    const idB = promptSetIdentity(setB)

    // ① 身份是**哈希背书**的，且两份等长集的身份不同
    t.assert(typeof idA.id === 'string' && idA.id.startsWith('sha256:'), `身份是 sha256 形态（实得 ${String(idA.id).slice(0, 20)}…）`)
    t.assert(idA.id !== idB.id, '**内容不同（长度相同）**的两个提示集必须得到**不同**的身份')
    t.assert(idA.hashBacked === true && idB.hashBacked === true, '两份集的身份都有哈希背书（hashBacked=true）')
    t.assert(idA.files['train.jsonl'] !== idB.files['train.jsonl'] && idA.files['heldout.jsonl'] !== idB.files['heldout.jsonl'],
      '两个文件（train/heldout）都逐文件记了字节 sha256')

    // ② 同一份集重复计算必须稳定（否则缓存永远不命中）
    t.assert(JSON.stringify(promptSetIdentity(setA)) === JSON.stringify(idA), '同一份集的身份可复现（幂等）')

    // ③ **核心断言**：只有提示集身份不同的两次 run **不得**共享缓存条目
    const common = {
      baseModel: '/models/Qwen2.5-0.5B-Instruct', revision: 'rev0', protocol: 'rl-pref-eval@2',
      nTrain: 48, nVal: 48, nLm: 100, nMmlu: 100, prefMaxTokens: CALIBRATED_PREF_MAX_TOKENS,
    }
    const keyA = baselineCacheKey({ ...common, promptSetId: idA.id })
    const keyB = baselineCacheKey({ ...common, promptSetId: idB.id })
    t.assert(keyA !== keyB, '**只差提示集身份**的两次 run → 缓存键不同（⇒ 不会共享缓存条目）')
    t.assert(keyA.includes(idA.id) && keyB.includes(idB.id), '缓存键里确实嵌入了提示集身份')
    t.assert(cacheKeyHasPromptSetIdentity(keyA) && cacheKeyHasPromptSetIdentity(keyB),
      `缓存键带身份标记 ${CACHE_KEY_IDENTITY_MARKER}（检测器据此判"旧键"）`)

    // ④ 键的其余维度仍逐项生效（不许"多了一位就把旧维度吃掉"）
    t.assert(baselineCacheKey({ ...common, promptSetId: idA.id }) === keyA, '同输入 → 同键（稳定）')
    t.assert(baselineCacheKey({ ...common, protocol: 'rl-pref-eval@1', promptSetId: idA.id }) !== keyA, '协议不同 → 键不同（跨协议比较被隔离）')
    t.assert(baselineCacheKey({ ...common, prefMaxTokens: 256, promptSetId: idA.id }) !== keyA, '生成长度上限不同 → 键不同（换仪器必须重测基线）')
    t.assert(baselineCacheKey({ ...common, nVal: 96, promptSetId: idA.id }) !== keyA, '留出样本量不同 → 键不同')

    // ⑤ fail-closed：缺身份 ⇒ **抛错**，绝不退化成"更短的键"（那是缺陷 N1 本身）
    const missing = [
      { why: '身份缺失（无提示集）', args: { ...common } },
      { why: '身份为 null（不可能得到身份）', args: { ...common, promptSetId: null } },
      { why: '身份为空串', args: { ...common, promptSetId: '' } },
    ]
    for (const m of missing) {
      let threw = null
      try { baselineCacheKey(m.args) } catch (e) { threw = String(e.message) }
      t.assert(threw !== null && /E_PROMPT_SET_IDENTITY_MISSING/.test(threw), `缺提示集身份 ⇒ 抛 E_PROMPT_SET_IDENTITY_MISSING（${m.why}）`)
    }
    let protoThrew = null
    try { baselineCacheKey({ baseModel: 'm', nTrain: 1, nVal: 1, nLm: 1, nMmlu: 1, prefMaxTokens: 768, promptSetId: 'sha256:x' }) } catch (e) { protoThrew = String(e.message) }
    t.assert(protoThrew !== null && /E_EVAL_PROTOCOL_MISSING/.test(protoThrew), '缺协议版本 ⇒ 抛 E_EVAL_PROTOCOL_MISSING')
    let mtThrew = null
    try { baselineCacheKey({ baseModel: 'm', protocol: 'p', nTrain: 1, nVal: 1, nLm: 1, nMmlu: 1, promptSetId: 'sha256:x' }) } catch (e) { mtThrew = String(e.message) }
    t.assert(mtThrew !== null && /E_PREF_MAX_TOKENS_MISSING/.test(mtThrew), '缺生成长度上限 ⇒ 抛 E_PREF_MAX_TOKENS_MISSING')

    // ⑥ 无法读文件 ⇒ 身份为 null（调用方据此 fail-closed，而不是拿一个"看起来像身份"的东西）
    const gone = promptSetIdentity(join(tmp, 'nope'))
    t.assert(gone.id === null && gone.hashBacked === false, '提示集文件不可读 ⇒ id=null、hashBacked=false（fail-closed）')
    t.assert(/prompt-set-files-unreadable/.test(gone.reason), `理由点名缺哪些文件（实得 ${gone.reason}）`)

    // ⑦ 与 lineage 的声明**交叉核对**（声明与磁盘不一致 ⇒ 如实报出，不静默通过）
    const setC = makeSet(tmp, 'setC', { mark: 'CCCC' })
    const real = promptSetIdentity(setC)
    lineageOf(setC, { evalFileHashes: { 'train.jsonl': 'sha256:deadbeef', 'heldout.jsonl': real.files['heldout.jsonl'] } })
    const stale = promptSetIdentity(setC)
    t.assert(stale.declaredMatches === false && stale.mismatched.includes('train.jsonl'),
      'lineage 声明与磁盘不一致 ⇒ declaredMatches=false 且点名 train.jsonl')
    t.assert(stale.id === real.id, '不一致时身份仍取**磁盘字节**（那才是将要被评估的东西），不伪造')
    lineageOf(setC, { evalFileHashes: { 'train.jsonl': real.files['train.jsonl'], 'heldout.jsonl': real.files['heldout.jsonl'] } })
    const matched = promptSetIdentity(setC)
    t.assert(matched.declaredAt === 'lineage.evalFileHashes' && matched.declaredMatches === true,
      '声明覆盖且一致时 declaredAt=lineage.evalFileHashes、declaredMatches=true')

    // ⑧ 组合哈希的口径与 eval-set-identity.mjs 的 combinedOf 同形状（改口径必须显式）
    t.assert(idA.id === 'sha256:' + (await import('node:crypto')).createHash('sha256')
      .update(['heldout.jsonl', 'train.jsonl'].sort().map((f) => `${f}:${idA.files[f]}`).join('\n')).digest('hex'),
    '组合哈希 = sha256(sorted("file:sha256").join("\\n"))（与 eval-set-identity 同口径）')

    /* ══════ ⑨ 存量处置的检测器**有牙齿**（CI_COVERAGE_RULES R4：新断言必须看过它红） ══════
       合成一个只含"旧键（无提示集身份）"的 RL 注册表，跑真实的处置脚本：
       未标记 ⇒ `--check` **必须红**；`--apply` 后 ⇒ 绿；且**只加字段不改数字**。
       这一段不依赖本机状态（全部在 tmp 里），因此它在任何机器上都是可复现的。 */
    const { spawnSync } = await import('node:child_process')
    const { stringify, parseFile } = await import('@xiciyang/state-core')
    const { readFile } = await import('node:fs/promises')
    const CHECK_SCRIPT = join(PKG, '..', 'checks', 'mark-invalid-rl-baselines.mjs')
    const fakeRoot = join(tmp, 'fakeroot')
    mkdirSync(join(fakeRoot, 'state'), { recursive: true })
    const legacyKey = '/models/m@rev@rl-pref-eval@2@t48/v48/lm100/mmlu100/mt768'
    const fakeDoc = {
      version: 1, pkg: 'training-rl', updatedAt: '2026-01-01T00:00:00.000Z', runs: [],
      baselineCache: {
        [legacyKey]: { uniform: { ok: true }, lm: { attempted: 100, succeeded: 100, failed: 0, texts: 100, perText: [3] }, mmlu: { attempted: 100, succeeded: 100, failed: 0, n: 100 }, task: { n: 48, nOk: 48 } },
      },
    }
    const regPath = join(fakeRoot, 'state', 'rl-registry.yaml')
    writeFileSync(regPath, stringify(fakeDoc))

    const run = (args) => spawnSync(process.execPath, [CHECK_SCRIPT, '--root', fakeRoot, ...args], { encoding: 'utf8', timeout: 120000 })
    const red = run(['--check'])
    t.assert(red.status === 1, `未标记的旧键（不含提示集身份）⇒ --check 必须**红**（实得 exit=${red.status}）`)
    t.assert(/n1:cache-key-without-prompt-set-identity/.test(`${red.stdout}${red.stderr}`), '红的时候必须点名 N1 的原因（可定位）')
    const applied = run(['--apply'])
    t.assert(applied.status === 0, `--apply 成功（实得 exit=${applied.status}：${applied.stderr?.slice(0, 120)}）`)
    const green = run(['--check'])
    t.assert(green.status === 0, `标记后 --check 必须绿（实得 exit=${green.status}）`)

    const after = await parseFile({ readFile }, regPath)
    const entry = after.baselineCache[legacyKey]
    t.assert(entry.invalid === true && /^n1:/.test(entry.invalidReason ?? ''), '标记为 invalid:true 且理由属 N1')
    t.assert(entry.lm.succeeded === 100 && entry.lm.attempted === 100 && entry.mmlu.n === 100 && entry.task.nOk === 48,
      '**只加字段、不改数字**（原来的测量数字逐项保留）')
    t.assert(String(entry.invalidNote ?? '').includes('eval-contract §7.1'), '标记里写明依据（eval-contract §7.1）与人工恢复路径')

    // 对照（R4 的另一半）：条目**带**身份时不得被误标 —— 否则检测器会"见谁都红"
    const goodKey = `${legacyKey}@ps=${idA.id}`
    const okDoc = { ...fakeDoc, baselineCache: { [goodKey]: { ...fakeDoc.baselineCache[legacyKey], promptSetId: idA.id } } }
    const okRoot = join(tmp, 'okroot')
    mkdirSync(join(okRoot, 'state'), { recursive: true })
    writeFileSync(join(okRoot, 'state', 'rl-registry.yaml'), stringify(okDoc))
    const okRun = spawnSync(process.execPath, [CHECK_SCRIPT, '--root', okRoot, '--check'], { encoding: 'utf8', timeout: 120000 })
    t.assert(okRun.status === 0, `带身份的条目 ⇒ --check 绿（阴性对照；实得 exit=${okRun.status}）`)
  } finally {
    rmSync(tmp, { recursive: true, force: true })
  }
}
