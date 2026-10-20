/**
 * prepare —— RL 的 **prompt 集**构造（薄适配层，不碰原始语料）。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 为什么 RL 的 prompt 不从"新数据集"来
 * ══════════════════════════════════════════════════════════════════════════════
 * 1. **偏好数据一律不引入**（license 决定，见 reward.mjs 顶部）。
 * 2. **不自行合成数据**（本包的硬性 not_provided）。
 * ⇒ RL 的 prompt 只能来自**已经被 data-core 校验过去泄漏**的既有切分。
 * 本文件因此**不重新切分、不重新扫 PII**，而是：
 *   - 取既有数据集的 `train.jsonl`（→ RL 训练 prompt）
 *   - 取同一数据集的 `heldout.jsonl`（→ 偏好对齐评估 prompt）
 *   - **在 RL 这一层再复核一次去泄漏**（整行内容哈希 + 行内子串双轨），
 *     因为"上游保证过"与"我这一层成立"是两件事（上游换切分就会破）。
 * 上游血缘（含 license 证据链）**原样继承**到 RL 的 lineage.json —— 不许凭空断言 license。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * prompt 形态（与评估侧必须逐字一致）
 * ══════════════════════════════════════════════════════════════════════════════
 * `prompt = instruction` + （有 input 时）`"\n\n" + input`。
 * 训练侧（容器内）与评估侧（serving）用的是**同一个函数** `promptOf`，
 * 否则"训练时的分布"与"评估时的分布"就不是同一个东西。
 */
import { readFileSync, writeFileSync, existsSync, mkdirSync, readdirSync } from 'node:fs'
import { join } from 'node:path'

import { contentHash, dedupByContent, assertNoLeakage, lineage as makeLineage, recordHash } from '@xiciyang/data-core'
import { ToolError, ERROR_CODES } from '../version.mjs'

/** 训练/评估共用的 prompt 渲染（**唯一实现**）。 */
export function promptOf(row) {
  const ins = String(row?.instruction ?? '').trim()
  const inp = String(row?.input ?? '').trim()
  return inp ? `${ins}\n\n${inp}` : ins
}

export function loadJsonl(p) {
  return readFileSync(p, 'utf8').split('\n').filter((l) => l.trim()).map((l) => JSON.parse(l))
}

/** 双轨去泄漏复核：① 整行内容哈希交集；② heldout 行（≥64 字符）不得作为子串出现在任何 train 行内。 */
export function leakCheck(trainPrompts, heldoutPrompts, { keyOf = (r) => r.prompt } = {}) {
  const hash = assertNoLeakage(trainPrompts, heldoutPrompts, { keyOf })
  const trainTexts = trainPrompts.map((r) => String(keyOf(r) ?? ''))
  const contained = []
  for (const r of heldoutPrompts) {
    const t = String(keyOf(r) ?? '')
    if (t.length < 64) continue
    if (trainTexts.some((x) => x.includes(t))) contained.push(contentHash(t))
  }
  return {
    hashOverlap: hash.intersection.length,
    hashOk: hash.ok,
    substringContained: contained.length,
    substringOk: contained.length === 0,
    ok: hash.ok && contained.length === 0,
    calibOnly: hash.calibOnly,
    testOnly: hash.testOnly,
  }
}

/**
 * 构造 RL prompt 集。
 * @param {object} p
 * @param {string} p.sourceDatasetDir 既有（已去泄漏）数据集目录：需含 train.jsonl 与 heldout.jsonl
 * @param {string} p.datasetsRoot    RL prompt 集的根（默认 <root>/state/rl-prompts）
 */
export async function preparePrompts({
  sourceDatasetDir, seed = 20260929, trainPrompts = 400, heldoutPrompts = 200,
  maxPromptChars = 1024, datasetsRoot, force = false,
}) {
  if (!sourceDatasetDir) throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: 'prepare 需要 sourceDatasetDir（既有数据集目录）', recoverable: true })
  const trainFile = join(sourceDatasetDir, 'train.jsonl')
  const heldoutFile = join(sourceDatasetDir, 'heldout.jsonl')
  if (!existsSync(trainFile) || !existsSync(heldoutFile)) {
    throw new ToolError(ERROR_CODES.E_PROMPTS_INVALID, {
      hint: `源数据集目录必须同时含 train.jsonl 与 heldout.jsonl：${sourceDatasetDir}`,
      details: { train: existsSync(trainFile), heldout: existsSync(heldoutFile) },
      recoverable: true,
    })
  }

  const rawTrain = loadJsonl(trainFile)
  const rawHeldout = loadJsonl(heldoutFile)

  const toPrompt = (r) => {
    const p = promptOf(r)
    return { prompt: p.length > maxPromptChars ? p.slice(0, maxPromptChars) : p, source: sourceDatasetDir }
  }
  // 内容哈希去重（保序）—— 去重后仍然要复核去泄漏
  const trainDedup = dedupByContent(rawTrain.map(toPrompt).filter((r) => r.prompt.trim().length >= 8), { keyOf: (r) => r.prompt })
  const heldoutDedup = dedupByContent(rawHeldout.map(toPrompt).filter((r) => r.prompt.trim().length >= 8), { keyOf: (r) => r.prompt })

  const train = trainDedup.kept.slice(0, trainPrompts)
  const heldout = heldoutDedup.kept.slice(0, heldoutPrompts)
  if (train.length === 0 || heldout.length === 0) {
    throw new ToolError(ERROR_CODES.E_PROMPTS_INVALID, {
      hint: `prompt 集为空（train=${train.length} heldout=${heldout.length}）—— 拒绝用一个空集开始 RL`,
      recoverable: true,
    })
  }

  const leak = leakCheck(train, heldout)
  if (!leak.ok) {
    throw new ToolError(ERROR_CODES.E_PROMPT_LEAK, {
      hint: `RL prompt 集去泄漏复核未通过（契约 §4.1 O1）：哈希交集=${leak.hashOverlap}，子串包含=${leak.substringContained} —— 拒绝出结论`,
      details: leak, recoverable: false,
    })
  }

  const name = `rl-prompts__s${seed}__n${train.length}x${heldout.length}`
  const out = join(datasetsRoot, name)
  if (!existsSync(out) || force) {
    mkdirSync(out, { recursive: true, mode: 0o700 })
    writeFileSync(join(out, 'train.jsonl'), `${train.map((r) => JSON.stringify(r)).join('\n')}\n`)
    writeFileSync(join(out, 'heldout.jsonl'), `${heldout.map((r) => JSON.stringify(r)).join('\n')}\n`)
  }

  // 上游血缘（含 license 证据链）原样继承 —— 不许凭空断言 license
  let upstream = null
  const upLineage = join(sourceDatasetDir, 'lineage.json')
  if (existsSync(upLineage)) { try { upstream = JSON.parse(readFileSync(upLineage, 'utf8')) } catch { upstream = null } }
  if (!upstream?.license) {
    throw new ToolError(ERROR_CODES.E_PROMPTS_INVALID, {
      hint: `源数据集血缘里没有 license（及其证据链）：${upLineage} —— RL 不允许在"许可来源不明"的 prompt 上训练`,
      recoverable: false,
    })
  }

  const lin = makeLineage({
    source: `${upstream.source}#rl-prompts`,
    revision: upstream.revision ?? null,
    seed,
    calib: train,
    test: heldout,
    extra: {
      license: upstream.license,
      sourceDatasetDir,
      sourceDatasetLineage: `sha256:${contentHash(readFileSync(upLineage, 'utf8'))}`,
      promptOf: 'instruction + (input ? "\\n\\n"+input : "")',
      maxPromptChars,
      leak,
      dedup: { trainDropped: trainDedup.dropped.length, heldoutDropped: heldoutDedup.dropped.length },
      allClear: true,
      note: 'RL 的 prompt 集来自既有（已去泄漏）数据集的两侧切分；本层只做去重 + 去泄漏复核 + 渲染，不重新切分、不合成、不脱敏',
    },
  })
  writeFileSync(join(out, 'lineage.json'), `${JSON.stringify(lin, null, 2)}\n`)
  writeFileSync(join(out, 'train.prompts.jsonl'), `${train.map((r) => JSON.stringify({ prompt: r.prompt, h: recordHash(r, (x) => x.prompt) })).join('\n')}\n`)

  return {
    promptsDir: out,
    trainPrompts: train.length,
    heldoutPrompts: heldout.length,
    dedup: { trainInput: rawTrain.length, heldoutInput: rawHeldout.length, trainDropped: trainDedup.dropped.length, heldoutDropped: heldoutDedup.dropped.length },
    leak,
    lineage: lin,
    license: upstream.license,
  }
}

/** 列出已有 prompt 集（幂等复用；也让 get_report 能显示"有哪些可用 prompt 集"）。 */
export function listPromptSets(datasetsRoot) {
  if (!existsSync(datasetsRoot)) return []
  const out = []
  for (const d of readdirSync(datasetsRoot)) {
    const lg = join(datasetsRoot, d, 'lineage.json')
    if (!existsSync(lg)) continue
    try {
      const j = JSON.parse(readFileSync(lg, 'utf8'))
      out.push({ name: d, dir: join(datasetsRoot, d), train: j.calib?.count ?? null, heldout: j.test?.count ?? null, allClear: j.allClear === true, license: j.license?.spdx ?? null })
    } catch { /* 忽略坏血缘 */ }
  }
  return out
}
