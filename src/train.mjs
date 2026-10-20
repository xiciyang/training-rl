/**
 * train —— RL 训练执行（**公共循环走 training-core 的「单体档位」**，本文件只做适配与编排）。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 为什么是单体档位（**这是用户已裁决的选项 B，不是本包自己的选择**）
 * ══════════════════════════════════════════════════════════════════════════════
 * `@xiciyang/training-core` 的**逐步档位**要求 `step(batch, state)` —— 即循环必须由 core 拥有。
 * 而 TRL 1.14 的 `GRPOTrainer.train()` 自带 dataloader / 生成 / 优势估计 / KL / 优化器步 /
 * checkpoint，**不暴露单步 API**（TRL 1.x 甚至已经没有 PPOTrainer）。
 * 只有两条路：把 GRPO 逐步重写（等于重实现框架、引入数值漂移），或给 core 加**单体档位**。
 * ⇒ 用户裁决 **选项 B**，该档位已落在 `training-core` contract **v0.2.0**
 *   （`createMonolithicLoop` / `pluginMode` / `MODES.MONOLITHIC`）。
 * 本文件消费该档位，**不改共享库一行**。
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 本包负责什么 / 不负责什么（边界）
 * ══════════════════════════════════════════════════════════════════════════════
 *   training-core 负责：限额判定（每步 ctx.onStep 内）、墙钟/GPU 小时记账、checkpoint 编排、
 *                       产物级断言原语、审计钩子接口、**事后核对**（不信任插件）
 *   training-rl 负责：RL 配置、规则奖励、β 推导、容器内 GRPO 驱动、逐 step 握手、
 *                     KL/奖励记账落链、产物指纹
 *   Python（容器内）负责：只做张量数学（加载/LoRA/生成/反向/优化器步/存权重）
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 产物级断言：**为什么是「基座 vs 合并后的策略」而不是「adapter 训练前后」**
 * ══════════════════════════════════════════════════════════════════════════════
 * LoRA 的 adapter 输出目录**训练前是空的** ⇒ `fingerprintArtifacts(dir).hash === null`，
 * 而 `assertArtifactChanged` 对 `null` 指纹一律抛 `E_ARTIFACT_FINGERPRINT_MISSING`
 * （"无法断言"按不通过处理）。于是"训练前为空"与"训练什么都没写"**无法区分** ——
 * 直接把空目录当 before 会得到一个恒失败的断言，那等于没有断言。
 *
 * 本包采用的表述**同时**把两件事区分开：
 *   ① `fingerprintArtifacts(baseModelDir)`  → 基线权重指纹（非空；= reference 的产物身份）
 *   ② `fingerprintArtifacts(adapterDir)`    → 训练后 LoRA adapter 指纹
 *        - **null ⇒ 什么都没写** ⇒ 立即 `E_ARTIFACT_FINGERPRINT_MISSING`（"无法断言"按不通过）
 *   ③ `fingerprintArtifacts(mergedDir)`     → 合并后的策略权重指纹（非空；真正的产物）
 *   ④ `assertArtifactChanged({before: ①, after: ③})`
 *        - 逐位相同 ⇒ `E_NO_OP_TRAINING`（**权重一步没动**，例如 lr=0、LoRA 全零增量）
 *        - 不同   ⇒ 权重确实变了
 * LoRA 的 B 矩阵初始化为 0 ⇒ 合并结果**恰好等于基座**当且仅当 adapter 未被训练推动；
 * 故 ①↔③ 的比较正是"权重变了没有"的**充分判据**，而 ② 负责"写没写"。
 * （只断言 loss 下降是不够的：loss 是过程量，什么都不写盘的 run 也能打印出下降的 loss。）
 */
import { writeFileSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { createMonolithicLoop, fingerprintArtifacts, assertArtifactChanged, pluginMode, MODES, BUDGET_DEFAULTS, saveCheckpoint } from '@xiciyang/training-core'

import { ToolError, ERROR_CODES, RULE_REWARD_VERSION, RL_IMAGE } from './version.mjs'
import { ContainerWorker, trainingMounts } from './worker.mjs'
import { rewardOf } from './reward.mjs'
import { buildKlAccounting } from './kl.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const FIXTURE = join(HERE, '..', 'tests', 'fixtures', 'reward-vectors.json')

function loadRewardVectors() {
  try {
    const doc = JSON.parse(readFileSync(FIXTURE, 'utf8'))
    return doc.vectors.map((v) => ({ branch: v.branch, prompt: v.prompt, completion: v.completion }))
  } catch (e) {
    throw new ToolError(ERROR_CODES.E_INTERNAL, { hint: `规则奖励规格（reward-vectors.json）不可读：${e.message}`, recoverable: false })
  }
}

const relOut = (dir, runDir) => (String(dir).startsWith(`${runDir}/`) ? String(dir).slice(runDir.length + 1) : String(dir))

/**
 * 构造 training-core 的**单体**插件对象。
 *
 * `run(ctx)` 里每一步都做一次**双向握手**（见 worker.mjs 顶部时序图）：
 *   容器 ev:step ─▶ ctx.onStep(info)（core 侧记账 + 限额判定 + 可能的 checkpoint 编排）
 *                ◀─ ack{stop, reason, save} （stop 由 core 决定，本包不自行判断限额）
 */
export function makeRlPlugin({ worker, runDir, config, shared, log = () => {} }) {
  const plugin = {
    id: 'rl-grpo',
    algorithm: 'rl',
    monolithic: true,
    async run(ctx) {
      const done = await worker.stream('run', { config }, {
        onEvent: async (msg) => {
          if (msg.ev === 'step') {
            // ① 交给 core：记账 + 限额判定（stop 是 core 的决定，本包只搬运）
            const gate = await ctx.onStep({
              step: msg.step,
              epoch: msg.epoch,
              loss: msg.loss,
              metrics: msg.metrics,
              wallMs: msg.wallMs,
              gpuHours: msg.gpuHours,
            })
            // ② checkpoint：core 的 doCheckpoint 已经把 payload/meta 落盘（hooks.onCheckpoint 记下目录），
            //    真正的 **LoRA 权重**由容器写进同一个目录的 adapter/ 子目录
            const saveDir = shared.pendingSaveDir
            shared.pendingSaveDir = null
            return {
              stop: gate?.stop === true,
              reason: gate?.reason ?? null,
              step: msg.step,
              save: saveDir ? { dir: relOut(saveDir, runDir) } : null,
            }
          }
          if (msg.ev === 'eval') { await ctx.onEval({ ...msg }); return {} }
          if (msg.ev === 'log') { log(`  [rl] ${msg.text}`); return {} }
          return {}
        },
      })
      return done
    },
  }
  return plugin
}

/**
 * 跑一次 RL 训练。返回 { result(训练核结果), done(容器结算), fingerprints, adapterWritten, ... }。
 *
 * `workerFactory` 是**测试接缝**：CI 里用一个模拟容器协议语义的假 worker 驱动单体握手，
 * 从而在不碰 GPU/容器/框架的前提下验证"流式握手 + 限额 + checkpoint 编排 + 产物指纹"。
 */
export async function runTraining({
  root, runId, runDir, image = RL_IMAGE, modelDir, promptsDir,
  config, limits, checkpoint, onEvent = () => {}, log = () => {},
  workerFactory = null, auditHooks = null, extraMounts = [],
}) {
  const worker = workerFactory
    ? workerFactory({ runDir, config })
    : new ContainerWorker({
      image,
      mounts: trainingMounts({ modelDir, promptsDir, outDir: runDir, extra: extraMounts }),
      log,
    }).start()

  const shared = { pendingSaveDir: null }
  const plugin = makeRlPlugin({ worker, runDir, config, shared, log })

  const stepRecords = []
  const gpuClock = { acc: 0 }

  const hooks = {
    onStep: async ({ state, info }) => {
      const gpu = Number(info?.gpuHours)
      if (Number.isFinite(gpu)) gpuClock.acc = Math.max(gpuClock.acc, gpu)
      else state.gpuHours = gpuClock.acc
      // **每步 KL 与奖励记账**（契约冻结条件 4）—— 落一份 run 侧 JSONL，并进审计链
      const rec = {
        step: state.step,
        loss: Number.isFinite(info?.loss) ? info.loss : null,
        kl: Number.isFinite(info?.metrics?.kl) ? info.metrics.kl : null,
        reward: Number.isFinite(info?.metrics?.reward) ? info.metrics.reward : null,
        rewardStd: Number.isFinite(info?.metrics?.rewardStd) ? info.metrics.rewardStd : null,
        learnerKl: Number.isFinite(info?.metrics?.klLearner) ? info.metrics.klLearner : null,
        gpuHours: state.gpuHours,
        wallMs: state.wallMs,
      }
      stepRecords.push(rec)
      onEvent({ type: 'step', ...rec })
      if (auditHooks?.onStep) await auditHooks.onStep(rec, state)
    },
    onCheckpoint: async ({ checkpoint: cp }) => {
      shared.pendingSaveDir = cp?.dir ?? null
      if (auditHooks?.onCheckpoint) await auditHooks.onCheckpoint(cp)
    },
    onEval: async ({ eval: ev }) => { if (auditHooks?.onEval) await auditHooks.onEval(ev) },
    onBudgetExceeded: async ({ gate }) => {
      log(`  预算命中：${gate?.reason} —— 由 training-core 暂停（policy=${gate?.policy}）`)
      if (auditHooks?.onBudgetExceeded) await auditHooks.onBudgetExceeded(gate)
    },
    onStop: async (ctx2) => { if (auditHooks?.onStop) await auditHooks.onStop(ctx2) },
  }

  let out
  try {
    const loop = createMonolithicLoop({ plugin, limits, checkpoint, onEvent, hooks })
    const result = await loop.run()
    out = { result, stepRecords }
  } finally {
    try { await worker.stop?.() } catch (e) { log(`worker 停止异常（忽略）：${e.message}`) }
  }

  const { result, stepRecords: steps } = out
  const done = result.outcome ?? {}
  if (!done || done.ev !== 'done') {
    throw new ToolError(ERROR_CODES.E_INTERNAL, { hint: '容器没有返回结算消息（ev:done）—— 训练未真正完成', details: { stop: result.stop }, recoverable: false })
  }
  if (done.ok === false) {
    // **先把容器内的真实错误抛出去**：否则后面"什么都没写"的产物闸门会把它掩盖成
    // E_ARTIFACT_FINGERPRINT_MISSING，让排查从"框架报的那一行"变成"一个症状"。
    throw new ToolError(ERROR_CODES.E_INTERNAL, {
      hint: `容器内训练失败：${done.error}`,
      details: { containerError: String(done.error ?? '').slice(0, 600), trace: String(done.trace ?? '').slice(-1500) },
      recoverable: false,
    })
  }

  // ── ① 规则奖励的**跨语言一致性**闸门（两份实现必须逐位相同）
  const cfgVectors = (done.rewardConformance?.values ?? [])
  const jsValues = (config.rewardVectors ?? []).map((v) => rewardOf(v.completion, v.prompt))
  const diffs = cfgVectors.map((v, i) => Math.abs(Number(v) - Number(jsValues[i])))
  const maxAbsDiff = diffs.length ? Math.max(...diffs) : null
  const conformance = {
    version: done.rewardConformance?.version ?? null,
    n: cfgVectors.length,
    maxAbsDiff,
    bitIdentical: diffs.length > 0 && diffs.every((d) => d === 0),
    withinTolerance: diffs.length > 0 && diffs.every((d) => d <= 1e-12),
    expectedVersion: RULE_REWARD_VERSION,
  }
  if (config.rewardVectors?.length) {
    if (cfgVectors.length !== config.rewardVectors.length) {
      throw new ToolError(ERROR_CODES.E_REWARD_IMPL_DRIFT, { hint: `容器只对 ${cfgVectors.length}/${config.rewardVectors.length} 条奖励向量打分`, details: { conformance }, recoverable: false })
    }
    if (conformance.version !== RULE_REWARD_VERSION) {
      throw new ToolError(ERROR_CODES.E_REWARD_IMPL_DRIFT, { hint: `容器内的奖励版本 ${conformance.version} ≠ 本包权威版本 ${RULE_REWARD_VERSION}`, details: { conformance }, recoverable: false })
    }
    if (!conformance.withinTolerance) {
      throw new ToolError(ERROR_CODES.E_REWARD_IMPL_DRIFT, {
        hint: `JS 与容器内 Python 的规则奖励实现不一致（最大绝对偏差 ${maxAbsDiff}）：奖励漂移会让"加权后的分数"与"评估时的分数"不是同一个东西`,
        details: { conformance }, recoverable: false,
      })
    }
  }

  // ── ② 产物指纹（见文件头：为什么是这三个指纹 + 这条断言）
  const baseFp = await fingerprintArtifacts(modelDir)
  const adapterDir = join(runDir, 'adapter')
  const adapterFp = await fingerprintArtifacts(adapterDir)
  const savedFps = []
  // 注意：容器把 `saved` 放在 `result` 里面（协议形状见 rl_trainer.py 的 emit({"ev":"done","result":{...}})）
  for (const s of (done.result?.saved ?? [])) {
    const dir = join(runDir, s.dir, 'adapter')
    const fp = await fingerprintArtifacts(dir)
    savedFps.push({ step: s.step, dir: join(runDir, s.dir, 'adapter'), hash: fp.hash, files: fp.files.length, bytes: s.bytes ?? null })
    // 侧车：把 adapter 指纹写进对应 checkpoint 目录（payload 里那一份是 null —— 见文件头说明）
    if (fp.hash) writeFileSync(join(runDir, s.dir, 'adapter-fingerprint.json'), `${JSON.stringify(fp, null, 2)}\n`)
  }
  const adapterWritten = Boolean(adapterFp.hash) || savedFps.some((x) => Boolean(x.hash))
  if (!adapterWritten) {
    throw new ToolError(ERROR_CODES.E_ARTIFACT_FINGERPRINT_MISSING, {
      hint: `训练结束但 LoRA 权重一个字节都没写：${adapterDir}（与所有 checkpoint 的 adapter/ 都是空）—— "无法断言"按不通过处理，不得记为训练成功`,
      details: { adapterDir, savedFps }, recoverable: false,
    })
  }

  // ── ②b **产物级断言**：基座权重 vs 合并后的策略权重（见文件头：为什么是这一对）
  const mergedDir = join(runDir, done.result?.merged?.dir ?? 'merged')
  const mergedFp = await fingerprintArtifacts(mergedDir)
  let artifactAssertion
  try {
    artifactAssertion = {
      ok: true,
      ...assertArtifactChanged({ before: baseFp, after: mergedFp, what: 'policy weights（基座 vs 合并后的 RL 策略）' }),
      baseDir: modelDir,
      mergedDir,
      adapterDir,
      adapterFiles: adapterFp.files.length,
      checkpointAdapterHashes: savedFps.map((x) => ({ step: x.step, hash: x.hash })),
    }
  } catch (e) {
    throw new ToolError(e.message.startsWith('E_NO_OP_TRAINING') ? ERROR_CODES.E_NO_OP_TRAINING : ERROR_CODES.E_ARTIFACT_FINGERPRINT_MISSING, {
      hint: `${e.message}（产物级断言：只断言 loss 下降抓不到空操作，必须断言权重）`,
      details: { base: baseFp.hash, merged: mergedFp.hash, adapter: adapterFp.hash },
      recoverable: false,
    })
  }

  // ── ②c 预算语义（如实暴露 training-core v0.2.0 单体档位事后核对的一个真实性质）
  //     `createMonolithicLoop` 的事后核对用 `enforceLimits`，而它对 maxSteps 的判据是
  //     **`state.step >= l.maxSteps`**（=）；逐步档位则是在**步进之前**判定（故"恰好用满"合法）。
  //     两个档位对"恰好用满"的解释因此不同：单体档位下 200/200 步会被记为 post-run 违规。
  //     本包**不改共享库、也不掩盖**这个标记，而是额外给出一个把"用满"与"超出"分开的判定，
  //     两者一并进报告/注册表/契约的已知边界。
  const actualSteps = result.state.step
  const declaredMaxSteps = Number(limits.maxSteps)
  const budgetAssessment = {
    declaredMaxSteps: Number.isFinite(declaredMaxSteps) ? declaredMaxSteps : null,
    actualSteps,
    exceeded: Number.isFinite(declaredMaxSteps) ? actualSteps > declaredMaxSteps : null,
    flaggedByCore: (result.violations ?? []).length > 0,
    corePostCheckOperator: '>=',
    note: 'training-core v0.2.0 的单体档位事后核对用 `state.step >= maxSteps` ⇒ **恰好用满预算也会被标记**为 post-run 违规（逐步档位在步进前判定，故不会）。本包不改共享库、也不隐藏该标记：`exceeded` 才代表"真的超过上限"。',
  }

  // ── ③ 逐 step 记账落盘（供审计链之外的人读）
  writeFileSync(join(runDir, 'metrics.jsonl'), steps.map((s) => JSON.stringify(s)).join('\n') + (steps.length ? '\n' : ''))

  // ── ④ KL 记账与 β 复核（契约 §4.4：偏离超容差则重算 β 并**重新记账**，不是偷偷调参）
  //     并在 Node 侧按同一公式**重算** β 与容器回传的 β 交叉核对（"β 可推导"必须真的成立）
  const klAccounting = buildKlAccounting({
    sigmaR: done.calibration?.sigmaR,
    klBudget: config.klBudget,
    betaFromContainer: done.calibration?.beta ?? null,
    kls: steps.map((s) => s.kl),
    tolerancePct: config.klTolerancePct,
  })
  if (auditHooks?.onKlAccounting) await auditHooks.onKlAccounting(klAccounting)

  return {
    result,
    done,
    fingerprints: { base: baseFp, adapter: adapterFp, merged: mergedFp, checkpoints: savedFps },
    artifactAssertion,
    stepRecords: steps,
    conformance,
    klAccounting,
    budgetAssessment,
  }
}

export { assertArtifactChanged, fingerprintArtifacts, pluginMode, MODES, BUDGET_DEFAULTS, saveCheckpoint }
