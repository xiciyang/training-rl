/**
 * monolithic.spec —— **单体档位**的接线、逐 step 握手、checkpoint 编排、产物级断言。
 *
 * 为什么必须有这一条：单体档位（`{monolithic:true, run(ctx)}`）把循环交给框架，
 * 于是"插件到底有没有守限额"不能靠约定 —— training-core 的纪律是**不信任插件、事后核对**。
 * 这条测试用**假容器 worker**（它模拟真容器的流式协议语义：emit 事件 → 读 ack → 照 ack 行事）
 * 把整条握手跑通，零 GPU、零容器、零框架：
 *
 *   容器 ev:step ─▶ ctx.onStep(info)  （core 记账 + 限额判定 + 可能的 checkpoint 编排）
 *                ◀─ ack{stop, save}   （stop 由 core 决定，本包只搬运）
 *
 * 另外三条**失败路径**同样是判据（阳性对照）：
 *   - 合并产物与基座逐位相同 ⇒ E_NO_OP_TRAINING
 *   - 权重一个字节都没写 ⇒ E_ARTIFACT_FINGERPRINT_MISSING
 *   - 容器内奖励实现与 JS 权威实现不一致 ⇒ E_REWARD_IMPL_DRIFT
 */
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, existsSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'

import { runTraining, makeRlPlugin } from '../../src/train.mjs'
import { shouldAckStreamEvent, trainingMounts } from '../../src/worker.mjs'
import { pluginMode, MODES } from '@xiciyang/training-core'
import { ERROR_CODES } from '../../src/version.mjs'
import { ruleReward } from '../../src/reward.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
export const NAME = 'monolithic（逐 step 握手 / checkpoint 编排 / 产物级断言）'

const attempt = async (fn) => {
  try { return { ok: true, value: await fn() } } catch (e) { return { ok: false, code: e.code, message: String(e.message) } }
}

const rewardVectors = () => JSON.parse(readFileSync(join(PKG, 'tests', 'fixtures', 'reward-vectors.json'), 'utf8'))
  .vectors.map((v) => ({ branch: v.branch, prompt: v.prompt, completion: v.completion }))

/**
 * 假容器 worker：**模拟真容器的流式协议语义**（不是只返回值）。
 * 它真的会读 ack、真的按 ack 写权重、真的在 stop 时停下（或故意不停下）。
 */
function fakeWorker({ runDir, nSteps = 5, merged = 'MERGED', writeAdapter = true, writeMerged = true, writeCheckpointAdapters = true, ignoreStop = false, driftReward = false, betaOverride = null, vectors = [] }) {
  const saved = []
  const evals = []
  const stepRecs = []
  return {
    saved, evals,
    async stream(_cmd, _payload, { onEvent }) {
      let stoppedAt = null
      for (let i = 1; i <= nSteps; i += 1) {
        const rec = { step: i, epoch: 0, loss: 1 / i, gpuHours: i * 0.001, wallMs: i * 10, metrics: { kl: 0.001 * i, reward: 0.4 + i * 0.01, rewardStd: 0.05 } }
        stepRecs.push(rec)
        const reply = await onEvent({ id: 1, ev: 'step', ...rec })
        if (reply?.save?.dir && writeCheckpointAdapters) {
          const adir = join(runDir, reply.save.dir, 'adapter')
          mkdirSync(adir, { recursive: true })
          writeFileSync(join(adir, 'adapter_model.safetensors'), `W${i}`)
          saved.push({ dir: reply.save.dir, step: i, bytes: 2, files: 1 })
        }
        if (reply?.stop === true && !ignoreStop) { stoppedAt = i; break }
      }
      await onEvent({ id: 1, ev: 'eval', step: nSteps, split: 'val', metric: 'meanRuleReward', value: 0.55, unit: 'rule-reward', selectionOnly: true })
      evals.push({ step: nSteps, split: 'val', metric: 'meanRuleReward', value: 0.55, unit: 'rule-reward', selectionOnly: true })

      if (writeAdapter) {
        const adir = join(runDir, 'adapter')
        mkdirSync(adir, { recursive: true })
        writeFileSync(join(adir, 'adapter_model.safetensors'), 'FINAL')
      }
      if (writeMerged) {
        mkdirSync(join(runDir, 'merged'), { recursive: true })
        writeFileSync(join(runDir, 'merged', 'model.safetensors'), merged)
      }
      const sigmaR = 0.2
      const klBudget = 0.1
      return {
        id: 1, ev: 'done', ok: true,
        result: {
          steps: stoppedAt ?? nSteps, finalLoss: 0.2, stopReason: stoppedAt ? 'host-budget' : null,
          saved, finalAdapter: { dir: 'adapter', bytes: 6, files: 1 },
          merged: { dir: 'merged', weightsSha256: 'sha256:deadbeef', params: 1, bytes: 6, files: 1, vocabSize: 100 },
          rewardCalls: nSteps, rewardSamplesFirst: [0.5], klMean: 0.003, klMin: 0.001, klMax: 0.005,
          rewardMean: 0.43, fracRewardZeroStdMean: 0, logTail: [],
        },
        calibration: { sigmaR, klBudget, beta: betaOverride ?? (sigmaR / klBudget), samples: 16, distinctValues: 5, measuredAt: 'policy@step0', rewardMean: 0.5 },
        rewardConformance: {
          version: 'rule-reward@1',
          values: vectors.map((v) => (driftReward ? 0.123 : ruleReward(v.completion, v.prompt).value)),
        },
        steps: stepRecs,
        evaluations: evals,
        memory: { torchPeakGiB: 1.0, processRssGiB: 1.0 },
        timing: { loadMs: 1, calibrationMs: 1, trainMs: 1, mergeMs: 1, totalMs: 4 },
        versions: { torch: 'x', trl: 'y', transformers: 'z', peft: 'p', cuda: true, deviceCapability: [12, 1] },
      }
    },
    async stop() { /* 幂等 */ },
  }
}

function env(tmp, name, { base = 'BASE' } = {}) {
  const modelDir = join(tmp, name, 'model')
  mkdirSync(modelDir, { recursive: true })
  writeFileSync(join(modelDir, 'model.safetensors'), base)
  const runDir = join(tmp, name, 'run')
  mkdirSync(runDir, { recursive: true })
  return { modelDir, runDir, promptsDir: join(tmp, name, 'data') }
}

const baseArgs = (paths, vectors, over = {}) => ({
  root: paths.root ?? '/tmp',
  runId: 'rl_test', runDir: paths.runDir, modelDir: paths.modelDir, promptsDir: paths.promptsDir,
  config: { klBudget: 0.1, klTolerancePct: 50, maxSteps: 5, rewardVectors: vectors, runId: 'rl_test' },
  limits: { maxSteps: 5, maxGpuHours: 2, concurrency: 1 },
  checkpoint: { dir: join(paths.runDir, 'checkpoints'), every: 2, keep: 4 },
  log: () => {},
  ...over,
})

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tr-mono-'))
  const vectors = rewardVectors()

  /* ⓪ 档位判定：本插件确实是**单体**插件（而不是伪装成逐步） */
  const probePlugin = makeRlPlugin({ worker: { stream: async () => ({}) }, runDir: tmp, config: {}, shared: {} })
  t.assert(probePlugin.monolithic === true, '插件声明 monolithic: true')
  t.assert(typeof probePlugin.run === 'function' && probePlugin.step === undefined, '插件只实现 run(ctx)，**不**实现 step（两档不得同时声明）')
  t.assert(pluginMode(probePlugin) === MODES.MONOLITHIC, `training-core 的 pluginMode 判为 monolithic（实际 ${pluginMode(probePlugin)}）`)

  /* ⓪b 协议：**只有**声明 ack:true 的事件才回应答（两侧步调必须严格一致）
   *
   * 为什么这条是硬判据：所有流式事件共用**同一个 id**（= 请求 id），
   * 多写或少写一条 ack 都会让 Python 的 read_ack **错位读到上一条**，而 id 校验发现不了。
   * 这里同时验 JS 侧的策略函数与**容器侧源码**是否一致（跨语言协议契约）。
   */
  t.assert(shouldAckStreamEvent({ ev: 'step', ack: true }) === true, 'step 事件（ack:true）需要应答')
  t.assert(shouldAckStreamEvent({ ev: 'eval', selectionOnly: true }) === false, 'eval 事件是**单向通知**，不回应答')
  t.assert(shouldAckStreamEvent({ ev: 'log', text: 'x' }) === false, 'log 事件是单向通知，不回应答')
  t.assert(shouldAckStreamEvent({ ev: 'done' }) === false, 'done 是流结束，不回应答')
  t.assert(shouldAckStreamEvent(null) === false && shouldAckStreamEvent({}) === false, '缺字段时保守不回 ack')
  {
    const wSrc = readFileSync(join(PKG, 'src', 'worker.mjs'), 'utf8')
    const pySrc = readFileSync(join(PKG, 'python', 'rl_trainer.py'), 'utf8')
    t.assert(/shouldAckStreamEvent\(msg\) && reply/.test(wSrc), 'ContainerWorker 真的用该策略决定是否回 ack')
    t.assert(/emit\(\{"ev": "step", "ack": True/.test(pySrc), '容器侧**只**给 step 事件标 ack:True')
    const ackMarks = (pySrc.match(/"ack": True/g) ?? []).length
    t.assert(ackMarks === 1, `容器侧只有一处声明 ack（实际 ${ackMarks} 处）—— 多于一处就会出现"读了不该读的 ack"`)
    t.assert((pySrc.match(/read_ack\(/g) ?? []).length === 2, '容器侧只有 read_ack 的定义与那一处调用')
  }

  /* ① 快乐路径：逐 step 握手 + checkpoint 编排 + 产物断言 */
  const p1 = env(tmp, 'ok'); p1.root = tmp
  const w1 = fakeWorker({ runDir: p1.runDir, nSteps: 5, vectors })
  const out1 = await runTraining({ ...baseArgs(p1, vectors), workerFactory: () => w1 })
  t.assert(out1.result.state.step === 5, `跑完 5 个优化器步（实际 ${out1.result.state.step}）`)
  t.assert(out1.result.state.mode === MODES.MONOLITHIC, 'state.mode = monolithic（core 如实标注档位）')
  t.assert(out1.stepRecords.length === 5, `逐 step 记账 5 条（实际 ${out1.stepRecords.length}）`)
  t.assert(out1.stepRecords.every((s) => Number.isFinite(s.kl)), '每条记录都带实测 KL（契约冻结条件 4）')
  t.assert(w1.saved.length === 2 && w1.saved.map((x) => x.step).join(',') === '2,4', `core 在 step 2/4 编排了 checkpoint（实际 ${w1.saved.map((x) => x.step).join(',')}）`)
  t.assert(existsSync(join(p1.runDir, 'checkpoints', 'step-000002', 'payload.json')), 'checkpoint payload.json 由 training-core 落盘')
  t.assert(existsSync(join(p1.runDir, 'checkpoints', 'step-000002', 'adapter', 'adapter_model.safetensors')), 'LoRA 权重写进**同名同目录**的 adapter/ 子目录')
  t.assert(existsSync(join(p1.runDir, 'checkpoints', 'step-000002', 'adapter-fingerprint.json')), '每个 checkpoint 有 adapter 指纹侧车')
  const side = JSON.parse(readFileSync(join(p1.runDir, 'checkpoints', 'step-000002', 'adapter-fingerprint.json'), 'utf8'))
  t.assert(typeof side.hash === 'string' && side.hash.startsWith('sha256:'), '侧车指纹是 sha256')
  t.assert(existsSync(join(p1.runDir, 'metrics.jsonl')), '逐 step 记账落盘 metrics.jsonl')
  t.assert(readFileSync(join(p1.runDir, 'metrics.jsonl'), 'utf8').trim().split('\n').length === 5, 'metrics.jsonl 有 5 行')
  t.assert(out1.fingerprints.checkpoints.length === 2, '返回两个 checkpoint 的 adapter 指纹')
  t.assert(out1.fingerprints.adapter.hash !== null, '最终 adapter 指纹非空（权重确实写了）')
  t.assert(out1.artifactAssertion.ok === true, '产物级断言通过')
  t.assert(out1.artifactAssertion.before !== out1.artifactAssertion.after, '基座指纹 ≠ 合并后指纹（权重确实变了）')
  t.assert(out1.conformance.withinTolerance === true && out1.conformance.n === vectors.length, `跨语言奖励一致性通过（n=${out1.conformance.n}, maxAbsDiff=${out1.conformance.maxAbsDiff}）`)
  t.assert(out1.klAccounting.betaDeclared === 2, `β 由 σ_r/KL* 推导 = 0.2/0.1 = 2（实际 ${out1.klAccounting.betaDeclared}）`)
  t.assert(out1.klAccounting.klSteps === 5 && out1.klAccounting.klMean === 0.003, 'KL 记账：5 步、均值 0.003')
  // β 重算的**方向**：实测 KL（0.003）远低于预算（0.1）⇒ 下一次应**减小** β（KL 随 β 单调下降）。
  // β_next = β_declared × (klMean/KL*) = 2 × 0.03 = 0.06（**不是** σ_r/klMean = 66.7 —— 那方向是反的）。
  t.assert(out1.klAccounting.recalibrationRequired === true, '实测 KL 远低于预算 ⇒ 超容差 ⇒ 要求重新记账')
  t.assert(Math.abs(out1.klAccounting.recommendedBeta - 0.06) < 1e-9, `recommendedBeta = β_declared×(klMean/KL*) = 0.06（实际 ${out1.klAccounting.recommendedBeta}）`)
  t.assert(out1.klAccounting.recommendation.direction === 'decrease' && out1.klAccounting.recommendedBeta < out1.klAccounting.betaDeclared, '方向是"减小 β"（β_next < β_declared）')
  t.assert(out1.klAccounting.betaAtBudget === out1.klAccounting.betaDeclared, 'betaAtBudget = 声明的 β（口径定义，不是重算结果）')
  // **如实暴露**：training-core v0.2.0 单体档位的事后核对用 `>=`，故"恰好用满预算"也会被标记。
  // 本包不改共享库、也不掩盖，而是额外给出 exceeded（真的超过上限才算超）。
  t.assert(out1.result.state.budgetViolation === true, '恰好用满 5/5 步 ⇒ training-core 事后核对标 budgetViolation=true（共享库语义，如实传递）')
  t.assert(out1.result.violations.every((v) => v.phase === 'post-run'), 'violations 全部标 phase=post-run（不是运行中真实越限）')
  t.assert(out1.budgetAssessment.exceeded === false, 'budgetAssessment.exceeded=false（实际步数未超过声明上限）')
  t.assert(out1.budgetAssessment.corePostCheckOperator === '>=' && out1.budgetAssessment.note.includes('恰好用满'), 'budgetAssessment 说明这是 >= 语义所致（可读的边界说明）')

  /* ② 阳性对照：插件**无视 stop** 继续跑 ⇒ 事后核对记为越限（不信任插件） */
  const p2 = env(tmp, 'ignore-stop'); p2.root = tmp
  const w2 = fakeWorker({ runDir: p2.runDir, nSteps: 5, vectors, ignoreStop: true })
  const out2 = await runTraining({ ...baseArgs(p2, vectors, { limits: { maxSteps: 2, maxGpuHours: 2, concurrency: 1 }, config: { klBudget: 0.1, klTolerancePct: 50, maxSteps: 5, rewardVectors: vectors } }), workerFactory: () => w2 })
  t.assert(out2.result.state.step === 5, '插件无视 stop ⇒ core 仍如实记账 5 步（而不是假装 2 步）')
  t.assert(out2.result.state.budgetViolation === true, 'core 的**事后核对**把越限记为 budgetViolation=true')
  t.assert(out2.result.violations.some((v) => v.limit === 'maxSteps' && v.phase === 'post-run'), 'violations 里有 maxSteps 且标 phase=post-run')

  /* ③ 0 步 = 空操作 ⇒ 违规（不得当成功） */
  const p3 = env(tmp, 'no-steps'); p3.root = tmp
  const w3 = fakeWorker({ runDir: p3.runDir, nSteps: 0, vectors })
  const out3 = await runTraining({ ...baseArgs(p3, vectors), workerFactory: () => w3 })
  t.assert(out3.result.state.step === 0, '0 步被如实记账')
  t.assert(out3.result.violations.some((v) => v.limit === 'no-steps'), '0 步 ⇒ violations 含 no-steps（空操作不得当成功）')

  /* ④ 失败路径 A：合并产物与基座**逐位相同** ⇒ E_NO_OP_TRAINING */
  const p4 = env(tmp, 'noop', { base: 'SAME' }); p4.root = tmp
  const w4 = fakeWorker({ runDir: p4.runDir, nSteps: 3, merged: 'SAME', vectors })
  const r4 = await attempt(() => runTraining({ ...baseArgs(p4, vectors), workerFactory: () => w4 }))
  t.assert(r4.ok === false && r4.code === ERROR_CODES.E_NO_OP_TRAINING, `权重逐位不变 ⇒ E_NO_OP_TRAINING（实际 ${r4.code}）`)
  t.assert(/逐位相同|E_NO_OP_TRAINING/.test(String(r4.message)), '错误信息点名"逐位相同"（不是含糊的失败）')

  /* ⑤ 失败路径 B：权重一个字节都没写 ⇒ E_ARTIFACT_FINGERPRINT_MISSING */
  const p5 = env(tmp, 'nothing'); p5.root = tmp
  const w5 = fakeWorker({ runDir: p5.runDir, nSteps: 2, writeAdapter: false, writeCheckpointAdapters: false, vectors })
  const r5 = await attempt(() => runTraining({ ...baseArgs(p5, vectors, { checkpoint: { dir: join(p5.runDir, 'checkpoints'), every: 0, keep: 0 } }), workerFactory: () => w5 }))
  t.assert(r5.ok === false && r5.code === ERROR_CODES.E_ARTIFACT_FINGERPRINT_MISSING, `什么都没写 ⇒ E_ARTIFACT_FINGERPRINT_MISSING（实际 ${r5.code}）`)

  /* ⑥ 失败路径 C：容器内奖励实现与 JS 权威实现不一致 ⇒ E_REWARD_IMPL_DRIFT */
  const p6 = env(tmp, 'drift'); p6.root = tmp
  const w6 = fakeWorker({ runDir: p6.runDir, nSteps: 2, driftReward: true, vectors })
  const r6 = await attempt(() => runTraining({ ...baseArgs(p6, vectors), workerFactory: () => w6 }))
  t.assert(r6.ok === false && r6.code === ERROR_CODES.E_REWARD_IMPL_DRIFT, `奖励实现漂移 ⇒ E_REWARD_IMPL_DRIFT（实际 ${r6.code}）`)

  /* ⑦ 失败路径 D：容器回传的 β 与 Node 重算不一致 ⇒ E_BETA_DERIVATION_MISMATCH */
  const p7 = env(tmp, 'beta'); p7.root = tmp
  const w7 = fakeWorker({ runDir: p7.runDir, nSteps: 2, betaOverride: 0.04, vectors })
  const r7 = await attempt(() => runTraining({ ...baseArgs(p7, vectors), workerFactory: () => w7 }))
  t.assert(r7.ok === false && r7.code === ERROR_CODES.E_BETA_DERIVATION_MISMATCH, `β 不是推导出来的 ⇒ E_BETA_DERIVATION_MISMATCH（实际 ${r7.code}）`)

  /* ⑧ 预算命中时容器收到 stop=true（stop 由 core 决定，插件只搬运） */
  const p8 = env(tmp, 'budget-stop'); p8.root = tmp
  const w8 = fakeWorker({ runDir: p8.runDir, nSteps: 10, vectors })
  const out8 = await runTraining({
    ...baseArgs(p8, vectors, { limits: { maxSteps: 3, maxGpuHours: 2, concurrency: 1 }, config: { klBudget: 0.1, klTolerancePct: 50, maxSteps: 10, rewardVectors: vectors } }),
    workerFactory: () => w8,
  })
  t.assert(out8.result.state.step === 3, `预算 maxSteps=3 命中 ⇒ 恰好在第 3 步停下（实际 ${out8.result.state.step}）`)
  t.assert(out8.result.stop?.stop === true && String(out8.result.stop.reason).includes('maxSteps'), `stop 原因是 maxSteps（实际 ${out8.result.stop?.reason}）`)
  t.assert(out8.budgetAssessment.exceeded === false, '正面对照：停在步数上限 ⇒ exceeded=false')
  // 真·越限（插件无视 stop）⇒ exceeded=true（与"恰好用满"可区分）
  const p9 = env(tmp, 'overrun'); p9.root = tmp
  const w9 = fakeWorker({ runDir: p9.runDir, nSteps: 6, vectors, ignoreStop: true })
  const out9 = await runTraining({
    ...baseArgs(p9, vectors, { limits: { maxSteps: 3, maxGpuHours: 2, concurrency: 1 }, config: { klBudget: 0.1, klTolerancePct: 50, maxSteps: 6, rewardVectors: vectors } }),
    workerFactory: () => w9,
  })
  t.assert(out9.result.state.step === 6 && out9.budgetAssessment.exceeded === true, '真·越限（6 步 > 上限 3）⇒ exceeded=true（与"恰好用满"可区分）')

  rmSync(tmp, { recursive: true, force: true })
}
