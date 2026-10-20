#!/usr/bin/env node
/**
 * worker-protocol.spec —— **真容器**上的流式协议测试（CPU-only，不需要 GPU）。
 *
 * ## 为什么值得一条独立测试
 * 容器侧协议里有一个**用 id 校验发现不了**的陷阱：所有流式事件共用**同一个 id**。
 * 于是"多回一条 ack"或"少回一条 ack"都会让 Python 的 `read_ack` **错位读到上一条**，
 * 而 id 完全匹配 —— 看起来一切正常，实际把上一步的 `stop` / `save` 当成了这一步的决定。
 * 这类缺陷在单测里用"假 worker"是抓不到的（假 worker 直接用返回值，不走写回路径），
 * 必须让**真的 ContainerWorker** 面对**真的容器**。
 *
 * ## 怎么做到既真又可判
 * 用一个**脚本化容器**（`node:20-alpine` + 一个 Node 脚本）扮演训练器：
 *   - 它按协议 emit `log` / `step(ack:true)` / `eval` / `done`；
 *   - 它**只在 step 之后读一行 ack**，并断言那一行**带 `step` 字段**。
 *     若宿主对 `log`/`eval` 多回了一条 ack，读到的就是那条**没有 step 字段**的 ack
 *     ⇒ 立刻记为 `ackMismatches` 并让整条 run 失败。
 * 于是"应答策略必须对称"这件事变成了**容器侧可判定的**判据。
 *
 * 用法：node tests/realmachine/worker-protocol.spec.mjs
 */
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const { ContainerWorker } = await import(join(PKG, 'src', 'worker.mjs'))

const CONTAINER_SCRIPT = String.raw`
// 脚本化"训练器"：只做协议，不做数学。
import readline from 'node:readline'
const out = (o) => process.stdout.write(JSON.stringify(o) + '\n')
const rl = readline.createInterface({ input: process.stdin })
const queue = []
const waiters = []
rl.on('line', (l) => { const w = waiters.shift(); if (w) w(l.trim()); else queue.push(l.trim()) })
const readLine = () => new Promise((res) => { const q = queue.shift(); if (q !== undefined) res(q); else waiters.push(res) })

let ackMismatches = 0
let acksSeen = []
const stepsN = Number(process.env.FAKE_STEPS || '5')
for await (const line of (async function* () { for (;;) { yield await readLine() } })()) {
  const req = JSON.parse(line)
  if (req.cmd === 'shutdown') { out({ id: req.id, ok: true, bye: true }); process.exit(0) }
  if (req.cmd === 'ping') { out({ id: req.id, ok: true, pong: true }); continue }
  if (req.cmd !== 'run') { out({ id: req.id, ok: false, error: 'unknown ' + req.cmd }); continue }
  const id = req.id
  // 单向通知：宿主**不应**回 ack
  out({ id, ev: 'log', text: 'scripted container up' })
  for (let s = 1; s <= stepsN; s += 1) {
    out({ id, ev: 'step', ack: true, step: s, loss: 1 / s, gpuHours: s * 0.001, wallMs: s * 10, metrics: { kl: 0.001 * s, reward: 0.5, rewardStd: 0.1 } })
    const ackLine = await readLine()
    const ack = JSON.parse(ackLine)
    acksSeen.push(ack)
    if (ack.id !== id || ack.ack !== true) ackMismatches += 1
    // **关键判据**：step 的 ack 必须携带 step 字段。若宿主对前面的 log 多回了一条 ack，
    // 这里读到的就是那条没有 step 的 ack —— id 校验抓不到，只有这个字段能抓到。
    if (ack.step !== s) ackMismatches += 1
    if (s % 2 === 0) out({ id, ev: 'eval', step: s, split: 'val', metric: 'meanRuleReward', value: 0.5, unit: 'rule-reward', selectionOnly: true })
  }
  out({ id, ev: 'done', ok: true, result: { steps: stepsN, ackMismatches, acksSeen: acksSeen.length }, calibration: { sigmaR: 0.1, klBudget: 1, beta: 0.1 } })
}
`

let failed = 0
const t = { assert(c, m) { if (!c) { failed += 1; console.log(`  - ${m}`) } } }

const tmp = mkdtempSync(join(tmpdir(), 'rl-proto-'))
writeFileSync(join(tmp, 'fake_container.mjs'), CONTAINER_SCRIPT)

const w = new ContainerWorker({
  image: 'node:20-alpine',
  python: 'node',
  entry: '/work/fake_container.mjs',
  gpus: false, // 协议测试**不需要 GPU**（也不该占用 GPU 串行窗口）
  mounts: [`${tmp}:/work:ro`],
  env: { FAKE_STEPS: '5' },
  log: () => {},
}).start()

try {
  const ping = await w.call('ping', {}, { timeoutMs: 60000 })
  t.assert(ping.ok === true && ping.pong === true, `真容器 ping 应答（实际 ${JSON.stringify(ping)}）`)

  const seen = []
  const done = await w.stream('run', { config: { x: 1 } }, {
    timeoutMs: 120000,
    onEvent: async (msg) => {
      seen.push(msg.ev)
      // 宿主侧应答策略：只有 step 事件返回带 step 的 ack，其余返回空
      if (msg.ev === 'step' && msg.ack === true) return { step: msg.step, stop: false, save: null }
      return {}
    },
  })
  t.assert(done.ev === 'done' && done.ok === true, '流以 ev:done 结清')
  t.assert(done.result?.ackMismatches === 0, `容器侧 ack 匹配数 = 0 次错位（实际 ${done.result?.ackMismatches}）`)
  t.assert(done.result?.acksSeen === 5, `容器收到 5 条 step ack（实际 ${done.result?.acksSeen}）`)
  t.assert(seen.filter((x) => x === 'step').length === 5, `宿主看到 5 个 step 事件（实际 ${seen.filter((x) => x === 'step').length}）`)
  t.assert(seen.filter((x) => x === 'eval').length === 2, `宿主看到 2 个 eval 事件（s=2,4）`)
  t.assert(seen.filter((x) => x === 'log').length === 1, '宿主看到 1 个 log 事件')
  t.assert(seen[seen.length - 1] !== 'done', 'done 不经 onEvent（由 stream 结清）')
  t.assert(JSON.stringify(done.result).length > 0, 'done 载荷可 JSON 化')
} catch (e) {
  t.assert(false, `真容器流式协议测试抛出：${String(e.message).slice(0, 300)}`)
} finally {
  await w.stop()
  rmSync(tmp, { recursive: true, force: true })
}

console.log(`\n${failed === 0 ? 'PASS' : 'FAIL'}: worker-protocol（真容器 node:20-alpine，CPU-only）${failed === 0 ? '' : `，${failed} 项未通过`}`)
process.exit(failed === 0 ? 0 : 1)
