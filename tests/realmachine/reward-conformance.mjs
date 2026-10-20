#!/usr/bin/env node
/**
 * reward-conformance —— **容器侧规则奖励 vs JS 权威实现**（CPU-only，不需要 GPU）。
 *
 * 训练时这条闸门是**自动**的（`training_rl_train` 把规格向量发给容器、逐条比对，
 * 不一致即 `E_REWARD_IMPL_DRIFT`）。但那条闸门只有在真机训练时才跑得起来；
 * 本脚本把它**提前**成一条 20 秒、零 GPU 的可执行检查：起容器（`CUDA_VISIBLE_DEVICES=` 空，
 * 不碰 GPU）→ 发 `reward_vectors` → 与 JS 逐条比对。
 *
 * 判据：**最大绝对偏差 ≤ 1e-12**（两边都是 IEEE754 double、运算顺序逐条对齐 ⇒ 期望逐位相同；
 * 实测有 1 ULP 的差，故不把"接近"说成"相同"：同时回报 maxAbsDiff 与 bitIdentical）。
 *
 * 用法：node tests/realmachine/reward-conformance.mjs [--out <json>]
 */
import { execFileSync } from 'node:child_process'
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const OUTDIR = join(PKG, '..', 'out')

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }

const { ruleReward, RULE_REWARD_VERSION } = await import(join(PKG, 'src', 'reward.mjs'))
const spec = JSON.parse(readFileSync(join(PKG, 'tests', 'fixtures', 'reward-vectors.json'), 'utf8'))
const vectors = spec.vectors.map((v) => ({ prompt: v.prompt, completion: v.completion }))
const inputs = [JSON.stringify({ id: 1, cmd: 'reward_vectors', vectors }), JSON.stringify({ id: 2, cmd: 'shutdown' }), ''].join('\n')

const USER = process.env.USER || process.env.LOGNAME || 'user'
let stdout = ''
let stderr = ''
let containerError = null
try {
  stdout = execFileSync('docker', [
    'run', '--rm', '-i',
    '--user', `${process.getuid()}:${process.getgid()}`,
    '-e', 'HOME=/tmp', '-e', `USER=${USER}`, '-e', `LOGNAME=${USER}`,
    '-e', 'CUDA_VISIBLE_DEVICES=', // **空**：本检查不需要 GPU，也就不占用 GPU 串行窗口
    '-v', `${join(PKG, 'python')}:/work:ro`,
    '--entrypoint', 'python3', 'training-rl/smoke-toolchain:0.1.0', '/work/rl_trainer.py',
  ], { input: inputs, encoding: 'utf8', timeout: 600000, stdio: ['pipe', 'pipe', 'pipe'] })
} catch (e) {
  containerError = String(e?.message ?? e).slice(0, 500)
  stdout = String(e?.stdout ?? '')
  stderr = String(e?.stderr ?? '')
}

const lines = stdout.trim() ? stdout.trim().split('\n').map((l) => { try { return JSON.parse(l) } catch { return null } }).filter(Boolean) : []
const resp = lines.find((l) => l.id === 1) ?? null
const js = spec.vectors.map((v) => ruleReward(v.completion, v.prompt).value)

const result = {
  schema: 'training-rl/reward-conformance@1',
  generatedAt: new Date().toISOString(),
  instrument: 'container (training-rl/smoke-toolchain:0.1.0, CUDA_VISIBLE_DEVICES= ⇒ 不占 GPU)',
  jsVersion: RULE_REWARD_VERSION,
  containerVersion: resp?.version ?? null,
  n: resp?.values?.length ?? 0,
  maxAbsDiff: null,
  bitIdentical: null,
  withinTolerance: false,
  tolerance: 1e-12,
  worstBranch: null,
  containerError,
  stderrTail: stderr.trim().split('\n').slice(-3).join(' | ').slice(0, 400),
  passed: false,
}

if (resp && Array.isArray(resp.values) && resp.values.length === js.length) {
  let max = 0
  let worst = null
  for (let i = 0; i < js.length; i += 1) {
    const d = Math.abs(Number(resp.values[i]) - js[i])
    if (d > max) { max = d; worst = spec.vectors[i].branch }
  }
  result.maxAbsDiff = max
  result.worstBranch = worst
  result.bitIdentical = resp.values.every((x, i) => x === js[i])
  result.withinTolerance = max <= result.tolerance
  result.passed = result.containerVersion === RULE_REWARD_VERSION && result.withinTolerance
}

mkdirSync(OUTDIR, { recursive: true })
const outPath = argOf('--out', join(OUTDIR, 'rl-reward-conformance.json'))
writeFileSync(outPath, `${JSON.stringify(result, null, 2)}\n`)
process.stdout.write(`${JSON.stringify(result, null, 2)}\n`)
process.exit(result.passed ? 0 : 1)
