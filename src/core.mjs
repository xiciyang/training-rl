/**
 * core —— training-rl 的 6 个 capability 实现（薄适配层）。
 *
 * 能力→源：
 *   prepare   → @xiciyang/data-core（内容哈希 / 去重 / 去泄漏双轨 / 血缘）
 *   train     → @xiciyang/training-core（**单体档位** + 限额 + checkpoint + 产物级断言）+ 容器内 TRL GRPO
 *   validate  → inference-serving（部署/判定仪器）+ @xiciyang/testing-core（判定）+ overfit 判据参考实现
 *   rollback / get_report / health → 状态与审计（state-core / audit-core / sandbox-core）
 *
 * ══════════════════════════════════════════════════════════════════════════════
 * 两条本包特有的**可推导性闸门**（都在 train 里，都会 fail-closed）
 * ══════════════════════════════════════════════════════════════════════════════
 *  ① **KL\* 必须由调用方显式给出**：β = σ_r / KL*。本包**不内置 KL\* 的默认值**
 *     （与 training-pretrain 的 `minDomainGainPct` 同一条纪律：不许"自己定阈值、自己签收"）。
 *  ② **σ_r 必须实测且 > 0**：σ_r = 0 ⇒ GRPO 组内优势恒为 0 ⇒ 训练是空转 ⇒ 拒绝开始
 *     （`E_REWARD_NO_VARIANCE`）。这是 GRPO 最隐蔽的失效形态：200 步全跑完、loss 也"正常"，
 *     但梯度信号根本不存在。
 */
import { readFileSync, writeFileSync, existsSync, readdirSync, promises as fs } from 'node:fs'
import { createHash } from 'node:crypto'
import { hostname } from 'node:os'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { execFileSync } from 'node:child_process'

import { CONTRACT_VERSION as TESTING_CORE_VERSION } from '@xiciyang/testing-core'
import { BUDGET_DEFAULTS, listCheckpoints, loadCheckpoint, fingerprintArtifacts, assertArtifactChanged, instrumentFromDeployReport, assertInstrumentUsable } from '@xiciyang/training-core'

import { ToolError, ERROR_CODES, PACKAGE_VERSION, SCHEMA_VERSION, RULE_REWARD_VERSION, RL_IMAGE } from './version.mjs'
import { paths, audit, auditVerify, readRegistry, updateRegistry, sandboxProbe, requireContainerSandbox, digestArgs, mkdirp, nowId } from './runtime.mjs'
import { runTraining } from './train.mjs'
import { rewardDescriptor } from './reward.mjs'
import { preparePrompts, listPromptSets, loadJsonl, leakCheck } from './data/prepare.mjs'
import {
  evalPreference, evalLM, evalMMLU, evalUniform, preferenceGate, forgettingGate, measurementCompleteness, attachIntegrity,
  EVAL_PROTOCOL, MAX_TRUNCATION_RATIO,
} from './evaluate.mjs'
// N2（仪器参数标定）/ N1（提示集身份 + 基线缓存键）—— 两件事的唯一归属地
import {
  CALIBRATED_PREF_MAX_TOKENS, PREF_MAX_TOKENS_CALIBRATION, resolvePrefMaxTokens,
  promptSetIdentity, baselineCacheKey,
} from './measurement-identity.mjs'
import { resolveServingPath, ServingMcpClient, callWithConfirm } from './serving-client.mjs'
import { ContainerWorker, trainingMounts } from './worker.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG_ROOT = join(HERE, '..')
// 契约层位置从包位置推导（training-rl 与 inference-contracts 是兄弟目录），可用环境变量覆盖
const WORKSPACE = process.env.DSH_WORKSPACE || join(PKG_ROOT, '..', '..')
const SCHEMA_DIR = join(process.env.INFERENCE_CONTRACTS_DIR || join(WORKSPACE, 'inference-contracts'), 'state-schema')
const OVERFIT_CRITERION = join(WORKSPACE, 'inference-training', 'checks', 'overfit-criterion.mjs')

export const TRAINING_IMAGE = RL_IMAGE
/**
 * 端点部署声明的显存占用比例。
 * 为什么从 0.18 降到 0.12：真机实测 ——— 与本机既有 vLLM 服务（qwen3-embedding，4.5GiB）
 * **以及并行 workstream 的部署**同时存在时，0.18（≈23GiB 声明）会让部署降级链耗尽
 * （`E_DEGRADE_EXHAUSTED`，两次实测）。0.5B 的 bf16 权重 + KV 只要 ~1.4GiB，
 * 声明 0.12（≈14GiB）仍有约 10× 余量，同时给整机留出可用内存。
 */
export const DEPLOY_GPU_MEM_FRACTION = 0.12
export const DEFAULT_LIMITS = Object.freeze({ ...BUDGET_DEFAULTS, maxSteps: 500, maxGpuHours: 2, concurrency: 1, maxParams: 0.5e9 })
const SCHEMA_FILE = join(SCHEMA_DIR, 'rl-registry.schema.json')

function schemaPath() { return existsSync(SCHEMA_FILE) ? SCHEMA_FILE : null }
const P = (root) => paths(root)

async function withRun(root, runId, mutator) {
  return updateRegistry(root, (doc) => {
    const i = (doc.runs || []).findIndex((r) => r.id === runId)
    if (i < 0) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
    const runs = [...(doc.runs || [])]
    runs[i] = mutator({ ...runs[i] })
    return { ...doc, runs }
  }, { schemaPath: schemaPath() })
}

/* ═══════════════════════════════════════════════════ 0. reconcile（幽灵 running） */

/**
 * **幽灵 `running` 的检测与回收**。
 *
 * ## 为什么需要它（这是一个真机缺陷的产物，不是"防御性编程"）
 * `train` 的 `catch` 能把**抛出的异常**记成 `failed`。但**外部杀死**（SIGKILL / OOM /
 * 容器被 kill / 父进程被杀 / 我在彩排时用 `job_kill` 掐掉它）**任何 in-process 处理器都抓不到**，
 * 于是注册表里留下一条永远 `running` 的幽灵：`rl_20260929151030_yota`（第一次彩排，
 * 被我用 `job_kill` 掐掉；它的 run 目录只有 `trl/`、创建后再没被写过）。
 * 后果不只是"不好看"：任何"现在有没有在跑？"的判断都会得到**错误答案** ——
 * 与本项目已经固化的一条同族：**没走到的那条路径不会报错**。
 *
 * 处置形状与 `inference_serving_reconcile`（"SIGKILL/掉电后的残留"）一致：
 * **对账 + 显式登记证据**，而不是相信注册表。
 *
 * ## 判据（可解释、可复核、不误伤）
 *   陈旧 ⟺ **归属进程不可用** 且（run 目录缺失 **或** run 目录最新写入已超过 `staleAfterMs`）
 *   - 归属进程：`owner.{host,pid}`（本包从本次起在 train 里登记）。
 *     同一主机且 `kill(pid,0)` 成功 ⇒ **活着** ⇒ 不判陈旧（再慢也不误杀）。
 *     没有 owner 信息（老记录）或不同主机 ⇒ 只能靠"有没有在写"。
 *   - `ageMs` 只作**参考证据**，判据落在"run 目录最近有没有被写"上（进度而不是时长）。
 *     本包的 train 每步都写审计链、结束时写 metrics.jsonl，checkpoint 还会落 adapter/
 *     ⇒ "有进度 ⟹ 目录 mtime 新"这条蕴含关系在实践中成立。
 *
 * 只读模式（`dryRun`）供 `health()` / `get_report()` 使用 —— 它们**承诺只读**，
 * 绝不因为"顺手"而写状态。
 */
export function isPidAlive(pid) {
  const p = Number(pid)
  if (!Number.isInteger(p) || p <= 0) return null // 无法判定
  try { process.kill(p, 0); return true } catch (e) { return e && e.code === 'EPERM' ? true : false }
}

async function newestMtimeMs(dir) {
  try {
    const entries = await fs.readdir(dir, { recursive: true, withFileTypes: true })
    let newest = null
    let count = 0
    for (const e of entries) {
      if (!e.isFile()) continue
      const abs = join(e.parentPath ?? e.path ?? dir, e.name)
      // eslint-disable-next-line no-await-in-loop
      const st = await fs.stat(abs)
      count += 1
      if (newest === null || st.mtimeMs > newest) newest = st.mtimeMs
    }
    return { newest, count }
  } catch { return { newest: null, count: 0 } }
}

/**
 * 探测（并可回收）幽灵 `running`。
 * @returns {{checked:number, stale:Array, interrupted:Array, alive:Array, host:string, staleAfterMs:number, dryRun:boolean}}
 */
export async function reconcile({ root = paths().root, staleAfterMs = 15 * 60 * 1000, dryRun = false, now = Date.now(), log = () => {} } = {}) {
  const reg = await readRegistry(root, schemaPath())
  const host = hostname()
  const stale = []
  const alive = []
  const interrupted = []
  for (const r of (reg.runs || [])) {
    if (r.status !== 'running') continue
    const owner = r.owner ?? null
    const ownHost = owner?.host ?? null
    const pidState = (ownHost === null || ownHost === host) ? isPidAlive(owner?.pid) : null
    if (pidState === true) { alive.push({ id: r.id, pid: owner.pid, rule: 'pid-alive' }); continue }
    const dir = r.runDir ?? join(paths(root).runsDir, r.id)
    // eslint-disable-next-line no-await-in-loop
    const { newest, count } = await newestMtimeMs(dir)
    const runDirAgeMs = newest === null ? null : (now - newest)
    const startedMs = Date.parse(r.startedAt ?? '')
    const ageMs = Number.isFinite(startedMs) ? (now - startedMs) : null
    const noProgress = newest === null || runDirAgeMs > staleAfterMs
    const ownerEvidence = owner ? `owner=${ownHost}:${owner.pid}(${pidState === false ? 'dead' : 'unknown'})` : 'owner=absent(旧记录)'
    if (!noProgress) { alive.push({ id: r.id, rule: 'progress-fresh', runDirAgeMs, files: count, owner: ownerEvidence }); continue }
    const reason = newest === null
      ? `stale-running: run 目录无任何产物（${dir}）且归属进程不可用（${ownerEvidence}）—— 外部杀死/OOM 的典型痕迹`
      : `stale-running: run 目录最新写入 ${Math.round(runDirAgeMs / 1000)}s 前（>阈值 ${Math.round(staleAfterMs / 1000)}s）且归属进程不可用（${ownerEvidence}）`
    const rec = { id: r.id, startedAt: r.startedAt ?? null, ageMs, runDirAgeMs, files: count, owner: ownerEvidence, reason }
    stale.push(rec)
    if (!dryRun) {
      const ar = await audit(root, { capability: 'reconcile', action: 'reconcile', phase: 'end', runId: r.id, from: 'running', to: 'interrupted', evidence: rec })
      await withRun(root, r.id, (x) => ({ ...x, status: 'interrupted', interruptedAt: new Date().toISOString(), interruptedReason: reason, reconciledAt: new Date().toISOString(), auditSeqs: [...(x.auditSeqs || []), ar.seq] }))
      interrupted.push({ ...rec, auditSeq: ar.seq })
      log(`reconcile：${r.id} running → **interrupted**（${reason}）`)
    }
  }
  return { checked: (reg.runs || []).filter((r) => r.status === 'running').length, stale, alive, interrupted, host, staleAfterMs, dryRun }
}

/* ══════════════════════════════════════════════════════════ 1. prepare */

export async function prepare({ root = paths().root, sourceDatasetDir = null, seed = 20260929, trainPrompts = 400, heldoutPrompts = 200, maxPromptChars = 1024, force = false, log = () => {} }) {
  const t0 = Date.now()
  await audit(root, { capability: 'prepare', action: 'prepare', phase: 'begin', ...digestArgs({ sourceDatasetDir, seed, trainPrompts, heldoutPrompts }) })
  const out = await preparePrompts({ sourceDatasetDir, seed, trainPrompts, heldoutPrompts, maxPromptChars, datasetsRoot: P(root).promptsDir, force })
  const r = await audit(root, { capability: 'prepare', action: 'prepare', phase: 'end', outcome: 'ok', train: out.trainPrompts, heldout: out.heldoutPrompts, leakOk: out.leak.ok, reward: RULE_REWARD_VERSION, ms: Date.now() - t0 })
  log(`RL prompt 集就绪：train=${out.trainPrompts} heldout=${out.heldoutPrompts} → ${out.promptsDir}（去泄漏双轨均空）`)
  return {
    promptsDir: out.promptsDir,
    trainPrompts: out.trainPrompts,
    heldoutPrompts: out.heldoutPrompts,
    dedup: out.dedup,
    leakCheck: out.leak,
    license: out.license,
    reward: rewardDescriptor(),
    lineage: out.lineage,
    audit: { seq: r.seq, hash: r.hash },
  }
}

/* ══════════════════════════════════════════════════════════ 2. train */

export async function train({
  root = paths().root, promptsDir = null, baseModel = process.env.DSH_BASE_MODEL,
  klBudget = null, klTolerancePct = 50,
  lora = {}, lr = 1e-6, batchSize = 8, gradAccum = 1, numGenerations = 4, maxCompletionLength = 64,
  maxSteps = 200, seed = 20260929, limits = {}, checkpointEvery = 50, checkpointKeep = 4,
  evalEvery = 50, evalPrompts = 8, calibPrompts = 8, calibGenerations = 4,
  confirm = null, log = () => {},
}) {
  const p = P(root)
  await mkdirp(p.runsDir)

  // ── 输入闸门（全部 fail-closed，且**在起容器之前**）
  if (!promptsDir || !existsSync(join(promptsDir, 'train.jsonl'))) {
    throw new ToolError(ERROR_CODES.E_PROMPTS_INVALID, { hint: `prompt 集未就绪：${promptsDir}（先调 training_rl_prepare）`, recoverable: true })
  }
  const lineage = JSON.parse(readFileSync(join(promptsDir, 'lineage.json'), 'utf8'))
  if (lineage.allClear !== true) throw new ToolError(ERROR_CODES.E_PROMPT_LEAK, { hint: 'prompt 集血缘里 allClear !== true —— 拒绝训练', recoverable: false })
  // **KL\* 必须显式给出**（本包不内置默认值）
  const klb = Number(klBudget)
  if (!Number.isFinite(klb) || klb <= 0) {
    throw new ToolError(ERROR_CODES.E_KL_BUDGET_REQUIRED, {
      hint: 'klBudget（KL*）必须由调用方显式给出且 > 0。β = σ_r / KL* 里的 KL* 是**政策选择**（允许策略偏离参考多远），本插件不内置默认值 —— "自己定阈值、自己签收"是明确禁止的',
      details: { got: klBudget ?? null, formula: 'beta = sigmaR / klBudget' },
      recoverable: true,
    })
  }
  if (!existsSync(join(baseModel, 'config.json'))) throw new ToolError(ERROR_CODES.E_BASE_MODEL_NOT_FOUND, { hint: `基座模型目录不合法：${baseModel}`, recoverable: true })

  const limitsResolved = { ...DEFAULT_LIMITS, ...limits, maxSteps }
  if (limitsResolved.maxSteps > 500) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `maxSteps=${limitsResolved.maxSteps} 超过单次上限 500`, recoverable: true })
  if (limitsResolved.maxGpuHours > 2) throw new ToolError(ERROR_CODES.E_BUDGET_EXCEEDED, { hint: `maxGpuHours=${limitsResolved.maxGpuHours} 超过单次上限 2`, recoverable: true })

  // 沙箱：训练属 write 档 → 必须容器层（策略来自长任务 1 冻结的 sandbox-core）
  const probe = await sandboxProbe()
  requireContainerSandbox(probe)

  // **开跑前先对账**：把"外部杀死留下的幽灵 running"回收成 interrupted。
  // 放在这里（而不是只在 health 里只读报告）是因为"新 run 开始"正是最需要干净账本的时刻，
  // 而且此时没有并发写（本包并发恒为 1）。health/get_report 只做**只读**探测。
  const rec = await reconcile({ root, log })
  if (rec.interrupted.length) log(`开跑前回收了 ${rec.interrupted.length} 条幽灵 running：${rec.interrupted.map((x) => x.id).join(', ')}`)

  const runId = nowId('rl')
  const runDir = join(p.runsDir, runId)
  await mkdirp(runDir)
  const started = new Date().toISOString()
  const baseModelDirName = baseModel.split('/').filter(Boolean).pop()
  // 规则奖励的**规格向量**（跨语言一致性闸门用）：容器侧必须对同一组向量给出同样分数
  const rewardVectors = JSON.parse(readFileSync(join(PKG_ROOT, 'tests', 'fixtures', 'reward-vectors.json'), 'utf8'))
    .vectors.map((v) => ({ branch: v.branch, prompt: v.prompt, completion: v.completion }))

  const config = {
    runId,
    baseModel: `/models/${baseModelDirName}`,
    promptsFile: '/data/train.jsonl',
    valPromptsFile: '/data/heldout.jsonl',
    lora: { r: 8, alpha: 16, dropout: 0.0, targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'], ...lora },
    lr, batchSize, gradAccum, numGenerations, maxCompletionLength, maxSteps, seed,
    klBudget: klb, klTolerancePct,
    calibPrompts, calibGenerations,
    evalEvery, evalPrompts,
    rewardVectors,
    // ⚠️ 权重产物的 license = **基座模型**的 license，**不是语料**的 license。
    //
    // 本行曾经写成 `lineage.license?.spdx ?? 'apache-2.0'` —— 即把 RL 语料的 `cc-by-4.0`
    // 写进了合并产物的 README frontmatter。而同一份产物里 `_merge_and_export` 又从基座目录
    // 复制了 `LICENSE`（Apache-2.0）⇒ **frontmatter 与随包 LICENSE 互相矛盾**。
    // 后果（长任务 5 阶段 3 真机实测）：`cc-by-4.0` 不在 serving 的 `license.permissive` 表里
    // （`inference-serving/src/policy-defaults.mjs:109`）⇒ 判 unknown ⇒ 部署被
    // `E_LICENSE_UNRESOLVED` 拒绝。**该缺陷只在"选点落在最后一步"时暴露** —— 因为那种情况
    // validate 用 `merged`（本行生成的 frontmatter），而选到中间 checkpoint 时用
    // `mergedSelected`（`mergeSelected` 里硬编码 `apache-2.0`，可部署）。同一份产物两条路径
    // 写两个 license，是"没走到的那条路径不会报错"的又一实例。
    // 统一为基座 license（与同包的 `mergeSelected`、以及 `training-sft` 的 merged 产物一致）。
    // **语料 license 不丢**：它仍在 promptsDir 的 `lineage.json` 里，并随 `datasetLicense` 进
    // 产物的 `training-rl-provenance.json`（署名链保留，只是不再冒称权重产物的 license）。
    licenseSpdx: 'apache-2.0',
    datasetLicense: lineage.license
      ? { spdx: lineage.license.spdx ?? null, class: lineage.license.class ?? null, source: lineage.license.source ?? null, role: 'attribution-only：RL 语料的 license，**不是**权重产物的 license' }
      : null,
  }

  await audit(root, {
    capability: 'train', action: 'train', phase: 'begin', runId, baseModel, promptsDir,
    owner: { host: hostname(), pid: process.pid }, reconciledBeforeStart: rec.interrupted.length,
    reward: rewardDescriptor(), klBudget: klb, klFormula: 'beta = sigmaR / klBudget',
    config: digestArgs(config).argsDigest, sandboxTier: 'container', image: TRAINING_IMAGE,
    mode: 'monolithic', plugin: 'rl-grpo',
  })

  await updateRegistry(root, (doc) => ({
    ...doc,
    runs: [...(doc.runs || []).filter((r) => r.id !== runId), {
      id: runId, status: 'running', startedAt: started, mode: 'monolithic',
      // 归属进程：reconcile 用它判断"这条 running 还有没有主人"
      owner: { host: hostname(), pid: process.pid, startedAt: started },
      baseModel: { path: baseModel, revision: readRevision(baseModel) },
      reward: rewardDescriptor(),
      kl: { budget: klb, formula: 'beta = sigmaR / klBudget', tolerancePct: klTolerancePct },
      dataset: { dir: promptsDir, trainRows: lineage.calib?.count ?? null, heldoutRows: lineage.test?.count ?? null, lineageHash: `sha256:${contentHashOfFile(join(promptsDir, 'lineage.json'))}` },
      config: { lora: config.lora, lr, batchSize, gradAccum, numGenerations, maxCompletionLength, maxSteps, seed, evalEvery },
      budget: limitsResolved,
      auditSeqs: [],
    }],
  }), { schemaPath: schemaPath() })

  try {
    const out = await runTraining({
      root, runId, runDir, image: TRAINING_IMAGE, modelDir: baseModel, promptsDir,
      config, limits: limitsResolved,
      checkpoint: { dir: join(runDir, 'checkpoints'), every: checkpointEvery, keep: checkpointKeep },
      log,
      auditHooks: {
        // **每步 KL 与奖励记账**（契约冻结条件 4）—— 逐步进哈希链，链可复算
        onStep: async (rec) => {
          await audit(root, {
            capability: 'train', action: 'step', phase: 'progress', runId, step: rec.step,
            kl: rec.kl, reward: rec.reward, rewardStd: rec.rewardStd, loss: rec.loss, gpuHours: rec.gpuHours,
          })
        },
        onKlAccounting: async (acct) => {
          await audit(root, { capability: 'train', action: 'kl-accounting', phase: 'end', runId, ...acct })
        },
      },
    })

    // ── 选点（O4）：容器内周期评估是**选点信号**（selectionOnly），不作判定
    const selection = await selectCheckpoint({ runDir, done: out.done, checkpointEvery, maxSteps })

    const ar = await audit(root, {
      capability: 'train', action: 'train', phase: 'end', runId,
      outcome: out.result.stop?.reason ?? 'ok', steps: out.result.state.step,
      budgetViolation: out.result.state.budgetViolation === true, violations: out.result.violations.length,
      budgetExceeded: out.budgetAssessment?.exceeded ?? null,
      wallMs: out.result.state.wallMs, gpuHours: Number(out.result.state.gpuHours.toFixed(4)),
      finalLoss: out.stepRecords.slice(-1)[0]?.loss ?? null, checkpoints: out.fingerprints.checkpoints.length,
      sigmaR: out.klAccounting.sigmaR, beta: out.klAccounting.betaDeclared, klMean: out.klAccounting.klMean,
      artifactAssertion: out.artifactAssertion.ok, rewardConformanceMaxAbsDiff: out.conformance.maxAbsDiff,
      selectedStep: selection.selected.step, selectionRule: selection.selectionRule,
    })

    await withRun(root, runId, (r) => ({
      ...r, status: 'trained', finishedAt: new Date().toISOString(),
      result: {
        steps: out.result.state.step, wallMs: out.result.state.wallMs,
        gpuHours: Number(out.result.state.gpuHours.toFixed(4)),
        stopReason: out.result.stop?.reason ?? null, stopChecks: out.result.stop?.checks ?? [],
        budgetViolation: out.result.state.budgetViolation === true, violations: out.result.violations,
        budgetAssessment: out.budgetAssessment,
        lossesTail: out.stepRecords.slice(-10).map((s) => s.loss),
        finalLoss: out.stepRecords.slice(-1)[0]?.loss ?? null,
        checkpoints: out.fingerprints.checkpoints,
        timing: out.done.timing ?? null,
        memory: out.done.memory ?? null,
        versions: out.done.versions ?? null,
      },
      reward: { ...rewardDescriptor(), calls: out.done.result.rewardCalls ?? null },
      calibration: out.done.calibration ?? null,
      beta: out.klAccounting.betaDeclared,
      kl: { ...out.klAccounting },
      rewardConformance: out.conformance,
      artifactAssertion: out.artifactAssertion,
      fingerprints: out.fingerprints,
      selection,
      runDir,
      heartbeatAt: new Date().toISOString(),
      auditSeqs: [...(r.auditSeqs || []), ar.seq],
    }))

    return {
      runId, runDir, status: 'trained',
      steps: out.result.state.step,
      finalLoss: out.stepRecords.slice(-1)[0]?.loss ?? null,
      gpuHours: Number(out.result.state.gpuHours.toFixed(4)),
      stopReason: out.result.stop?.reason ?? null,
      budgetViolation: out.result.state.budgetViolation === true,
      violations: out.result.violations,
      budgetAssessment: out.budgetAssessment,
      checkpoints: out.fingerprints.checkpoints.length,
      reward: rewardDescriptor(),
      calibration: {
        sigmaR: out.klAccounting.sigmaR, klBudget: out.klAccounting.klBudget, beta: out.klAccounting.betaDeclared,
        formula: out.klAccounting.formula, samples: out.done.calibration?.samples ?? null,
        distinctValues: out.done.calibration?.distinctValues ?? null,
      },
      kl: {
        mean: out.klAccounting.klMean, min: out.klAccounting.klMin, max: out.klAccounting.klMax,
        std: out.klAccounting.klStd, deviationPct: out.klAccounting.deviationPct,
        tolerancePct: out.klAccounting.tolerancePct, withinTolerance: out.klAccounting.withinTolerance,
        recommendedBeta: out.klAccounting.recommendedBeta, recalibrationRequired: out.klAccounting.recalibrationRequired,
      },
      rewardConformance: out.conformance,
      artifactAssertion: out.artifactAssertion,
      fingerprints: { base: out.fingerprints.base.hash, adapter: out.fingerprints.adapter.hash, merged: out.fingerprints.merged.hash },
      selection,
      audit: { seq: ar.seq, hash: ar.hash },
    }
  } catch (e) {
    const ar = await audit(root, { capability: 'train', action: 'train', phase: 'failed', runId, error: String(e.message).slice(0, 400) })
    // ⚠️ 第二版把这里的 `.catch(() => {})` 去掉了：**静默吞掉**"连失败都记不上"这件事，
    // 会让 run 永远停在 running，而且没有任何痕迹（正是本项目那条同族纪律：没走到的那条路径不会报错）。
    // 现在的处置：把"标记失败也失败了"写进审计链，留给 `reconcile` 与人事后追溯。
    await withRun(root, runId, (r) => ({ ...r, status: 'failed', error: String(e.message).slice(0, 600), auditSeqs: [...(r.auditSeqs || []), ar.seq] }))
      .catch(async (e2) => {
        log(`标记 run ${runId} 为 failed 失败（**不静默吞掉**）：${String(e2?.message ?? e2).slice(0, 200)} —— 该 run 可能停在 running，由 reconcile 兜底`)
        await audit(root, { capability: 'train', action: 'train', phase: 'failed', runId, markFailedError: String(e2?.message ?? e2).slice(0, 300) }).catch((e3) => log(`连记账也失败了：${String(e3?.message ?? e3).slice(0, 120)}`))
      })
    throw e
  }
}

function contentHashOfFile(f) {
  try { return createHash('sha256').update(readFileSync(f)).digest('hex') } catch { return null }
}
function readRevision(modelDir) {
  try {
    const m = join(modelDir, '.cache', 'huggingface', 'download', 'config.json.metadata')
    return existsSync(m) ? readFileSync(m, 'utf8').split('\n')[0].trim() : null
  } catch { return null }
}

/**
 * O4 选点（training-eval-contract §4.4）：**val loss 的全局最小者**，并列时取 payloadHash 字典序小者。
 *
 * 度量朝向：契约里的 `L` 是 loss 类（越低越好）。RL 的规则奖励是"越高越好"，
 * 故 `L_val = −meanRuleReward_val` —— 这是一次**保序变换**，argmin 不变，
 * 但把语义对齐到契约（否则"最小"会选到最差的 checkpoint）。
 * `N` 不是魔数：候选点由 `checkpointEvery` 与数据/配置自然给出（与 §4.4 同一条理由）。
 */
async function selectCheckpoint({ runDir, done, checkpointEvery, maxSteps }) {
  const cps = await listCheckpoints({ dir: join(runDir, 'checkpoints') })
  const byStep = new Map(cps.map((c) => [c.step, c]))
  const evals = done.evaluations ?? []
  const valByStep = new Map()
  const trainByStep = new Map()
  for (const e of evals) {
    if (e.selectionOnly !== true) continue
    if (e.metric !== 'meanRuleReward') continue
    if (e.split === 'val') valByStep.set(e.step, e.value)
    if (e.split === 'train') trainByStep.set(e.step, e.value)
  }
  const points = [...valByStep.keys()].sort((a, b) => a - b)
  const curve = {
    schema: 'training-rl/selection-curve@1',
    metric: 'meanRuleReward',
    selectionOnly: true,
    note: '容器内周期评估 ⇒ 只能用于**选择** checkpoint，不得进入任何门禁判定（training-eval-contract §5.2）',
    steps: points,
    train: points.map((s) => trainByStep.get(s) ?? null),
    val: points.map((s) => valByStep.get(s) ?? null),
    valLoss: points.map((s) => (Number.isFinite(valByStep.get(s)) ? -valByStep.get(s) : null)),
    checkpointEvery,
    maxSteps,
  }
  if (points.length === 0) {
    return {
      selectionRule: 'final-no-val-signal',
      selected: { name: 'final', step: done.result?.steps ?? maxSteps, payloadHash: null, selectionOnly: false },
      candidates: [], curve,
      note: '没有 val 选点信号（evalEvery=0 或步数不足以触发周期评估）⇒ 直接用最终权重；**不做任何选点声明**',
    }
  }
  let best = null
  for (const s of points) {
    const c = byStep.get(s)
    const cand = { step: s, valLoss: -valByStep.get(s), valReward: valByStep.get(s), trainReward: trainByStep.get(s) ?? null, name: c?.name ?? null, payloadHash: c?.hash ?? null }
    if (best === null) { best = cand; continue }
    if (cand.valLoss < best.valLoss) best = cand
    else if (cand.valLoss === best.valLoss) {
      // 并列 ⇒ payloadHash 字典序小者（确定性，契约 §4.4）
      const a = String(cand.payloadHash ?? '')
      const b = String(best.payloadHash ?? '')
      if (a && b && a < b) best = cand
    }
  }
  const isFinal = best.step >= (done.result?.steps ?? -1)
  return {
    selectionRule: 'val-loss-argmin',
    selected: {
      name: best.name ?? `step-${String(best.step).padStart(6, '0')}`,
      step: best.step,
      payloadHash: best.payloadHash,
      valLoss: best.valLoss,
      valReward: best.valReward,
      selectionOnly: true,
      isFinalStep: isFinal,
    },
    candidates: points.map((s) => ({ step: s, valLoss: -valByStep.get(s), payloadHash: byStep.get(s)?.hash ?? null })),
    curve,
    note: '选点规则 = val loss 的全局最小者（并列取 payloadHash 字典序小者）；val loss = −meanRuleReward（保序变换，argmin 不变）。判定数字一律来自 serving（§5.1）。',
  }
}

/* ══════════════════════════════════════════════════════════ 2b. merge（按需） */

/**
 * 从指定 checkpoint 的 adapter 重新合并（选点选到早期 checkpoint 时用）。
 * 合并走**容器内 Python**（与训练同一镜像），保证宿主/容器版本一致。
 */
export async function mergeSelected({ root = paths().root, runId, adapterRelDir = null, outDir = null, log = () => {} }) {
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  const target = outDir || join(run.runDir, 'merged-selected')
  if (adapterRelDir && existsSync(target)) return { runId, mergedDir: target, cached: true }
  const worker = new ContainerWorker({
    image: TRAINING_IMAGE,
    mounts: trainingMounts({ modelDir: run.baseModel.path, promptsDir: run.dataset.dir, outDir: run.runDir }),
    log,
  }).start()
  try {
    const r = await worker.call('merge_export', {
      adapterDir: adapterRelDir ? `/out/${adapterRelDir}` : '/out/adapter',
      outDir: `/out/${target.slice(run.runDir.length + 1)}`,
      config: { baseModel: `/models/${run.baseModel.path.split('/').filter(Boolean).pop()}`, licenseSpdx: 'apache-2.0' },
      provenance: { runId, baseModel: run.baseModel, adapterRelDir: adapterRelDir ?? 'adapter', reason: 'selected-checkpoint re-merge' },
    }, { timeoutMs: 30 * 60 * 1000 })
    const ar = await audit(root, { capability: 'merge', action: 'merge', phase: 'end', runId, mergedDir: target, weightsSha256: r.merged?.weightsSha256 ?? null, adapterRelDir: adapterRelDir ?? 'adapter' })
    await withRun(root, runId, (x) => ({ ...x, mergedSelected: { dir: target, adapterRelDir: adapterRelDir ?? 'adapter', weightsSha256: r.merged?.weightsSha256 ?? null }, auditSeqs: [...(x.auditSeqs || []), ar.seq] }))
    return { runId, mergedDir: target, cached: false, weightsSha256: r.merged?.weightsSha256 ?? null, params: r.merged?.params ?? null }
  } finally { await worker.stop() }
}

/* ══════════════════════════════════════════════════════════ 3. validate */

function loadEvalSets() {
  const sets = []
  const root = join(process.env.HOME || '/root', '.ai-infra/state/datasets')
  if (existsSync(root)) {
    for (const d of readdirSync(root)) {
      const p = join(root, d, 'test.jsonl')
      if (!existsSync(p)) continue
      sets.push({ name: `wikitext-test(${d.includes('c512') ? 'c512' : 'c128'})`, kind: 'lm', file: p, rows: loadJsonl(p) })
    }
  }
  const mmlu = process.env.DSH_MMLU_PATH
  if (existsSync(mmlu)) sets.push({ name: 'mmlu-test', kind: 'qa', file: mmlu, rows: loadJsonl(mmlu) })
  return sets
}

/** 持久化测量摘要里"是谁产出了这份测量"——供下游（如 `effectJudgment`）判定**不自证**。 */
export const MEASUREMENT_PRODUCER = 'training-rl.validate'

/**
 * 从仪器测量里**原样搬运**完整性证据（缺陷 14/15 的持久化面）。
 *
 * `attempted` / `failed` 一旦被摘要丢掉就**不可恢复** ⇒ "端点中途失效"与"评估集本来就小"
 * 在**落盘后**重新变得不可区分（缺陷 14 被判定为缺陷的那个理由）。
 * **只在源字段存在时搬运**（源里没有就不写，不补 `null`）。
 * 回归测试：`tests/unit/persisted-integrity.spec.mjs`。
 */
const INTEGRITY_FIELDS = ['attempted', 'succeeded', 'failed', 'skipped', 'partial']
const integrityOf = (x) => {
  const out = {}
  if (!x || typeof x !== 'object') return out
  for (const k of INTEGRITY_FIELDS) if (x[k] !== undefined) out[k] = x[k]
  return out
}

const summarize = (m) => ({
  uniform: { ok: m.uniform.ok, n: m.uniform.n, epsilon: m.uniform.epsilon, delta: m.uniform.delta, bitIdenticalAcrossRuns: m.uniform.bitIdenticalAcrossRuns },
  // ⚠️ 手列字段的老毛病（与 sft 的 `checkpointPayload` 同类）：**漏掉完整性证据字段**
  // ⇒ `measurementCompleteness` 判 `preference:no-integrity-evidence`，
  // 于是"端点中途失效"在偏好这条度量上不可见（缺陷 14 的同款形态）。
  // 生产者必须先给全，分类才敢把它当逐条网络测量看。
  preference: { ...integrityOf(m.preference), n: m.preference.n, nOk: m.preference.nOk, meanReward: m.preference.meanReward, truncated: m.preference.truncated, stopped: m.preference.stopped },
  // ⚠️ 键名必须是 `prefTrain`（`measurementCompleteness` 的已知部件词表用这个名字）。
  // 旧代码写成 `preferenceTrain` ⇒ 落盘摘要恒被判 `unclassified-part:preferenceTrain` ⇒ 永不可认证。
  // 插件**内部**本来就用 `prefTrain`（见 `measureAt`），只有这里改过名；且无任何消费方读这个持久化键。
  prefTrain: { ...integrityOf(m.prefTrain), n: m.prefTrain?.n ?? null, nOk: m.prefTrain?.nOk ?? null, meanReward: m.prefTrain?.meanReward ?? null },
  lm: { ...integrityOf(m.lm), texts: m.lm.texts, tokens: m.lm.tokens, meanNll: m.lm.meanNll, ppl: m.lm.ppl, perText: m.lm.perText },
  mmlu: { ...integrityOf(m.mmlu), n: m.mmlu.n, correct: m.mmlu.correct, accuracy: m.mmlu.accuracy, asked: m.mmlu.asked },
  integrity: { producedBy: MEASUREMENT_PRODUCER },
})

// ⚠️ `prefMaxTokens` **没有默认值**：它是**已标定**的仪器参数，任何"悄悄回落到未标定档"
// 都会让截断率闸门必然失败、产出一个假的"测量不成立"。调用方（`validate`）先经
// `resolvePrefMaxTokens` 解析并把 source 记进结果，再传进来。
async function measureAt({ endpoint, model, trainPrompts, valPrompts, wikitext, mmlu, nTrain, nVal, nLm, nMmlu, vocabSize, prefMaxTokens, log }) {
  const uniform = await evalUniform({ endpoint, model, prompt: 'The quick brown fox jumps over the lazy dog. In a distant country, the population of', vocabSize })
  const prefTrain = await evalPreference({ endpoint, model, rows: trainPrompts, limit: nTrain, maxTokens: prefMaxTokens, log })
  const prefVal = await evalPreference({ endpoint, model, rows: valPrompts, limit: nVal, maxTokens: prefMaxTokens, log })
  const lm = await evalLM({ endpoint, model, rows: wikitext, limit: nLm, log })
  const mmluRes = await evalMMLU({ endpoint, model, rows: mmlu, limit: nMmlu, log })
  // 注：门禁用 `preference`（= **留出**切分）；`prefTrain` 只用于 O3 的 train↔val 间隙计算。
  return { uniform, preference: prefVal, prefTrain, lm, mmlu: mmluRes }
}

/**
 * fail-closed：**有损仪器不得用于验收**（`E_INSTRUMENT_LOSSY`）。
 *
 * 本包是这条闸门的**直接触发者**：实测那次判决翻转（同一份策略权重 25/48 vs 23/48）
 * 就发生在 RL 的测量里，当时的部署落进了 serving 的有损档位 `precision`，
 * 而产物里没有任何字段说明被测策略已被量化。判据在 `@xiciyang/training-core`（单一实现），
 * 这里只做错误码转译 —— 否则 `toEnvelope` 会把非 ToolError 一律归成 `E_INTERNAL`。
 */
function guardInstrument(instrument, runId) {
  try {
    assertInstrumentUsable(instrument, { runId })
  } catch (e) {
    if (e?.code !== ERROR_CODES.E_INSTRUMENT_LOSSY) throw e
    throw new ToolError(ERROR_CODES.E_INSTRUMENT_LOSSY, { hint: e.message, details: e.details ?? null, recoverable: false })
  }
}

export async function validate({
  root = paths().root, runId, deploy = true, nTrain = 48, nVal = 48, nLm = 100, nMmlu = 100,
  thresholds = null, teardown = true, forceMerge = false, prefMaxTokens, log = () => {},
}) {
  const t0 = Date.now()
  const p = P(root)
  // **阈值必须显式给出**（本包不内置默认值）
  const minGain = Number(thresholds?.minRewardGainPct)
  if (!thresholds || !Number.isFinite(minGain)) {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, {
      hint: 'thresholds.minRewardGainPct 必须由调用方显式给出（本插件不内置默认值）—— 判据要能拒掉"提升不显著"的 run',
      details: { got: thresholds ?? null }, recoverable: true,
    })
  }
  // ── N2：仪器参数解析（**fail-closed + 可见**）────────────────────────────────
  // 省略时**只**回落到已标定档 768，并把 `source='calibrated-default'` 记进
  // 审计 / 注册表 / 返回值 —— 回落**留在证据里**，不是静默换仪器。
  // 绝不回落到未标定的 256：那一档的截断率闸门必然失败（本机标定 72.9%/79.2%）。
  let instrumentParam
  try { instrumentParam = resolvePrefMaxTokens(prefMaxTokens) } catch (e) {
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, {
      hint: e.message, details: { got: prefMaxTokens ?? null, calibration: PREF_MAX_TOKENS_CALIBRATION }, recoverable: true,
    })
  }
  prefMaxTokens = instrumentParam.value
  if (instrumentParam.source === 'calibrated-default') {
    log(`prefMaxTokens 未给出 ⇒ 用**已标定档** ${CALIBRATED_PREF_MAX_TOKENS}（source=calibrated-default；`
      + `未标定档 256 的截断率必然过不了闸门）`)
  } else if (!instrumentParam.calibrated) {
    log(`⚠️ prefMaxTokens=${prefMaxTokens} 由调用方显式给出，但**不是已标定档** `
      + `${CALIBRATED_PREF_MAX_TOKENS}（calibrated=false）—— 该值会如实记进结果`)
  }
  const zForgetting = Number.isFinite(Number(thresholds?.zForgetting)) ? Number(thresholds.zForgetting) : 1.96
  const thresholdsResolved = { minRewardGainPct: minGain, zForgetting }

  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  if (!['trained', 'validated', 'gate-failed'].includes(run.status)) {
    throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run ${runId} 状态为 ${run.status}，不能直接 validate（需要先 train）`, recoverable: true })
  }

  // ── 评估用 prompt：**留出切分**（不得混入训练 prompt）
  const trainPrompts = loadJsonl(join(run.dataset.dir, 'train.jsonl'))
  const valPrompts = loadJsonl(join(run.dataset.dir, 'heldout.jsonl'))
  const lk = leakCheck(trainPrompts, valPrompts)
  if (!lk.ok) {
    await audit(root, { capability: 'validate', action: 'validate', phase: 'failed', runId, outcome: 'leak', leak: lk })
    throw new ToolError(ERROR_CODES.E_PROMPT_LEAK, {
      hint: `评估前复核发现训练/留出 prompt 仍有交集（O1 泄漏闸）：哈希交集=${lk.hashOverlap}，子串包含=${lk.substringContained} —— **不出结论**（fail-closed）`,
      details: lk, recoverable: false,
    })
  }

  // ── N1：**提示集身份**（eval-contract §7）────────────────────────────────────
  // 判定用的评估集是判决的一部分：身份必须由**内容哈希**背书，且必须进基线缓存键。
  // 拿不到身份 ⇒ 既不查缓存也不写缓存（fail-closed），**不出一份"跨集比较"的结论**。
  const promptSet = promptSetIdentity(run.dataset.dir)
  if (promptSet.id === null) {
    await audit(root, { capability: 'validate', action: 'validate', phase: 'failed', runId, outcome: 'prompt-set-identity-missing', promptSet })
    throw new ToolError(ERROR_CODES.E_MISSING_PARAM, {
      hint: `提示集身份不可得（${promptSet.reason}）：${run.dataset.dir} —— `
        + 'eval-contract §7.1 规则 3 要求评估集身份必须有哈希背书；没有身份就无法证明"与基线比对用的是同一个集" ⇒ 拒绝判定（fail-closed）',
      details: promptSet, recoverable: true,
    })
  }
  if (promptSet.declaredMatches === false) {
    // lineage 的声明与磁盘**不一致** ⇒ 如实报出（哪个文件对不上），但不因此伪造一个身份：
    // 用于比较的仍是**磁盘字节**的哈希（那才是将要被测量的东西）。
    log(`⚠️ lineage.evalFileHashes 与磁盘不一致（${promptSet.mismatched.join(', ')}）—— 身份取磁盘字节，不一致如实记进结果`)
  }
  log(`提示集身份 ${promptSet.id.slice(0, 26)}…（${promptSet.datasetDir}；声明来源=${promptSet.declaredAt ?? 'lineage 未记 evalFileHashes，按磁盘字节现算'}）`)

  // ── 判定产物的选择（O4）：选到早期 checkpoint 时另起一次合并
  const selection = run.selection ?? { selectionRule: 'final-no-val-signal', selected: { name: 'final', step: run.result?.steps ?? null, payloadHash: null } }
  let mergedDir = join(run.runDir, 'merged')
  let mergeInfo = { reused: true, dir: mergedDir, by: 'train-time merge' }
  if (selection.selectionRule === 'val-loss-argmin' && selection.selected?.isFinalStep !== true && selection.selected?.name) {
    const adapterRel = `checkpoints/${selection.selected.name}/adapter`
    if (forceMerge || !existsSync(join(run.runDir, 'merged-selected'))) {
      log(`选点选中 ${selection.selected.name}（val loss ${selection.selected.valLoss}）⇒ 从该 checkpoint 重新合并 …`)
      const m = await mergeSelected({ root, runId, adapterRelDir: adapterRel, log })
      mergedDir = m.mergedDir
      mergeInfo = { reused: m.cached === true, dir: mergedDir, by: 'selected-checkpoint re-merge', adapterRelDir: adapterRel }
    } else {
      mergedDir = join(run.runDir, 'merged-selected')
      mergeInfo = { reused: true, dir: mergedDir, by: 'selected-checkpoint re-merge（已存在）', adapterRelDir: adapterRel }
    }
  }
  if (!existsSync(join(mergedDir, 'config.json'))) {
    throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `合并产物不合法（缺 config.json）：${mergedDir}`, recoverable: true })
  }

  // ── 产物级断言（对**被评估的那个产物**再断言一次 —— 选点可能换了产物）
  const baseFp = await fingerprintArtifacts(run.baseModel.path)
  const mergedFp = await fingerprintArtifacts(mergedDir)
  let artifactAssertion
  try {
    artifactAssertion = { ok: true, ...assertArtifactChanged({ before: baseFp, after: mergedFp, what: 'policy weights（基座 vs 被评估的 RL 策略）' }) }
  } catch (e) {
    throw new ToolError(e.message.startsWith('E_NO_OP_TRAINING') ? ERROR_CODES.E_NO_OP_TRAINING : ERROR_CODES.E_ARTIFACT_FINGERPRINT_MISSING, {
      hint: `${e.message}（validate 侧复核对被评估产物再断言一次）`, details: { base: baseFp.hash, merged: mergedFp.hash }, recoverable: false,
    })
  }

  // 审计 begin 记录**仪器 + 评估集身份**：判决是在哪一个仪器、哪一个集上做的，链上可查。
  // （N2：prefMaxTokens 的**值与来源**都要留痕；"回落"必须可见。）
  const instrument2 = {
    protocol: EVAL_PROTOCOL, prefMaxTokens, prefMaxTokensSource: instrumentParam.source,
    calibrated: instrumentParam.calibrated, calibratedValue: CALIBRATED_PREF_MAX_TOKENS,
    truncationGate: MAX_TRUNCATION_RATIO,
    promptSet: { id: promptSet.id, dir: promptSet.datasetDir, declaredAt: promptSet.declaredAt, declaredMatches: promptSet.declaredMatches },
  }
  await audit(root, {
    capability: 'validate', action: 'validate', phase: 'begin', runId, mergedDir, nTrain, nVal, nLm, nMmlu,
    thresholds: thresholdsResolved, selectionRule: selection.selectionRule, selectedStep: selection.selected?.step ?? null,
    prefMaxTokens, prefMaxTokensSource: instrumentParam.source, prefMaxTokensCalibrated: instrumentParam.calibrated,
    promptSetId: promptSet.id, promptSetDeclaredAt: promptSet.declaredAt,
  })

  const sets = loadEvalSets()
  const wikitext = (sets.find((s) => s.name.startsWith('wikitext-test(c128)')) || sets.find((s) => s.kind === 'lm'))?.rows ?? []
  const mmlu = (sets.find((s) => s.kind === 'qa'))?.rows ?? []
  const vocabSize = JSON.parse(readFileSync(join(mergedDir, 'config.json'), 'utf8')).vocab_size

  const sp = await resolveServingPath({ root })
  const client = await new ServingMcpClient(sp.serverPath).start()
  const results = {}
  let ownerToken = null
  let rlEndpoint = null
  let baseEndpoint = null
  let baseOwnerToken = null
  // ── 仪器来源：每次用于测量的部署各留一份归一化记录（见 training-sft 同处说明）
  let candidateInstrument = null
  let baseInstrument = null
  try {
    if (deploy !== true) throw new ToolError(ERROR_CODES.E_MISSING_PARAM, { hint: 'validate 需要 deploy=true（本实现的闭环要求经 serving 加载器验证）', recoverable: true })

    // ① 部署 RL 策略（inference-serving 的加载器 + 它自己的可用性探针）
    log(`部署 RL 策略 ${mergedDir} …`)
    const d = await callWithConfirm(client, 'inference_serving_deploy', {
      path: mergedDir, owner: 'training-rl', purpose: `长任务3 阶段3 RL 垂直闭环 validate ${runId}`,
      servedModelName: `rl-${runId}`, gpuMemoryUtilization: DEPLOY_GPU_MEM_FRACTION, maxNumSeqs: 4,
    }, { log })
    if (d.ok === false) throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `RL 策略部署失败：${d.code} ${d.message ?? ''}`, details: d })
    ownerToken = d.ownerToken
    rlEndpoint = { url: d.endpoint.endpoint, model: d.endpoint.servedModelName, id: d.endpoint.id, container: d.endpoint.containerName, imageDigest: d.endpoint.imageDigest, usability: d.report?.attempts?.[0]?.modelUsability ?? null }
    candidateInstrument = instrumentFromDeployReport(d.report)
    guardInstrument(candidateInstrument, runId) // 有损 ⇒ 立刻停：这次判决翻转就是这么发生的
    log(`  → ${rlEndpoint.url} (usability.ok=${rlEndpoint.usability?.ok}, lossy=${candidateInstrument?.lossy ?? null})`)

    // ② 策略侧测量
    log('测量 RL 策略 …')
    results.rl = await measureAt({ endpoint: rlEndpoint.url, model: rlEndpoint.model, trainPrompts, valPrompts, wikitext, mmlu, nTrain, nVal, nLm, nMmlu, vocabSize, prefMaxTokens, log })

    // ③ 基线测量（缓存到 registry：同一基座 + 同一协议 + 同一采样规模 + **同一提示集**只测一次）
    // 缓存键：基座 + revision + **评估协议** + 采样规模 + 生成长度上限 + **提示集身份**。
    // 协议版本、maxTokens、提示集身份**都**要进键：
    //   · 缺协议 ⇒ 跨协议比较（最隐蔽的一类错判）；
    //   · 缺 maxTokens ⇒ 两个不同仪器被混用；
    //   · **缺提示集身份（N1）⇒ "同样大小、内容不同"的另一个留出集会命中同一条缓存基线**，
    //     两个不可比的 Δ 被当成可比 —— 直接违反 eval-contract §7.1 规则 1/2/3。
    // 构造只此一处（`baselineCacheKey`）：缺任一维度**抛错**，不退化成"更短的键"。
    const cacheKey = baselineCacheKey({
      baseModel: run.baseModel.path, revision: run.baseModel.revision ?? null,
      protocol: EVAL_PROTOCOL, nTrain, nVal, nLm, nMmlu, prefMaxTokens, promptSetId: promptSet.id,
    })
    const cachedRaw = reg.baselineCache?.[cacheKey]
    // **缺陷 15**：只有**能证明完整**的缓存才可复用；"未知"不等于"完整"（fail-closed）。
    const cachedCheck = cachedRaw ? measurementCompleteness(cachedRaw) : null
    if (cachedRaw && !cachedCheck.complete) log(`  基线缓存**不可复用**（${cachedCheck.reason}）⇒ 本次重新测量基线`)
    // **N1 第二道闸**：即使键命中了，也复核条目**自带**的提示集身份与本次逐字一致。
    // 旧条目（N1 之前写的）没有这个字段 ⇒ 一律不可复用（unknown ≠ verified）。
    const identityOk = Boolean(cachedRaw) && cachedRaw.promptSetId === promptSet.id
    if (cachedRaw && !identityOk) {
      log(`  基线缓存**不可复用**（条目未记录提示集身份或与本次不符：`
        + `条目=${cachedRaw.promptSetId ?? 'null'} 本次=${promptSet.id}）⇒ 本次重新测量基线`)
    }
    const cached = (cachedRaw && cachedCheck.complete && identityOk) ? cachedRaw : null
    if (cached) {
      // 基线用缓存 ⇒ 本次没有部署、没有 deploy-report ⇒ `instrument.base = null`（**不编造**）。
      results.base = cached
      baseInstrument = null
      log('基线用缓存（本次无基线部署 ⇒ instrument.base = null，不编造）')
    } else {
      log(`部署基座 ${run.baseModel.path} …`)
      const b = await callWithConfirm(client, 'inference_serving_deploy', {
        path: run.baseModel.path, owner: 'training-rl', purpose: `长任务3 阶段3 RL 基线测量 ${runId}`,
        servedModelName: `base-${runId}`, gpuMemoryUtilization: DEPLOY_GPU_MEM_FRACTION, maxNumSeqs: 4,
      }, { log })
      if (b.ok === false) throw new ToolError(ERROR_CODES.E_SERVING_UNAVAILABLE, { hint: `基座部署失败：${b.code}`, details: b })
      baseEndpoint = { url: b.endpoint.endpoint, model: b.endpoint.servedModelName, id: b.endpoint.id, container: b.endpoint.containerName }
      baseOwnerToken = b.ownerToken
      baseInstrument = instrumentFromDeployReport(b.report)
      guardInstrument(baseInstrument, runId) // 基线仪器同样不得有损：基线侧仪器不同，比较就不成立
      log(`测量基座 … (lossy=${baseInstrument?.lossy ?? null})`)
      results.base = await measureAt({ endpoint: baseEndpoint.url, model: baseEndpoint.model, trainPrompts, valPrompts, wikitext, mmlu, nTrain, nVal, nLm, nMmlu, vocabSize, prefMaxTokens, log })
    }

    // ④ 门禁
    const validity = [
      { name: 'rl-forward', ok: results.rl.uniform.n > 0, detail: `策略侧拿到 ${results.rl.uniform.n} 个 logprob` },
      { name: 'rl-not-uniform', ok: results.rl.uniform.ok === true, detail: results.rl.uniform.criterion ?? results.rl.uniform.reason ?? '' },
      { name: 'preference-samples', ok: results.rl.preference.nOk >= Math.min(20, nVal), detail: `策略侧有效生成 ${results.rl.preference.nOk}/${results.rl.preference.n}` },
      { name: 'base-preference-samples', ok: results.base.preference.nOk >= Math.min(20, nVal), detail: `基座侧有效生成 ${results.base.preference.nOk}/${results.base.preference.n}` },
    ]
    const pg = preferenceGate({ base: results.base.preference, policy: results.rl.preference, thresholds: thresholdsResolved, validity })
    const fg = forgettingGate({ baseMmlu: results.base.mmlu, rlMmlu: results.rl.mmlu, baseLm: results.base.lm, rlLm: results.rl.lm, z: zForgetting })

    // ── O3 过拟合预算：**用冻结契约的唯一参考实现**（import，不重写）
    const overfit = await evaluateOverfitRL({ base: results.base, model: results.rl })

    const pass = pg.pass && fg.pass && overfit.pass === true

    const evalResult = {
      schema: 'ai-infra/eval-result@1',
      subject: { kind: 'checkpoint', ref: mergedDir, revision: null },
      measurement: {
        n: results.rl.preference.nOk + results.rl.lm.texts + results.rl.mmlu.n,
        unit: 'samples',
        datasetRef: `${run.dataset.dir}#heldout+train+wikitext-test+mmlu-test`,
        datasetHash: run.dataset.lineageHash ? String(run.dataset.lineageHash).replace(/^(sha256:)+/, 'sha256:') : null,
      },
      metrics: [
        { name: 'preference_reward', value: results.rl.preference.meanReward, unit: 'rule-reward', direction: 'higher-is-better', baseline: results.base.preference.meanReward, relativePct: pg.deltas.meanReward.rawPct },
        { name: 'mmlu_accuracy', value: results.rl.mmlu.accuracy, unit: 'ratio', direction: 'higher-is-better', baseline: results.base.mmlu.accuracy, relativePct: null },
        { name: 'wikitext_mean_nll', value: results.rl.lm.meanNll, unit: 'nats', direction: 'lower-is-better', baseline: results.base.lm.meanNll, relativePct: null },
      ],
      validity: pg.checks.filter((c) => c.kind === 'validity').map((c) => ({ name: c.name, ok: c.ok, detail: c.detail ?? null })),
      gate: { pass, validityOk: pg.validityOk, reason: !pg.validityOk ? pg.reason : (pass ? '达标' : `未达标：preference=${pg.pass} forgetting=${fg.pass} overfit=${overfit.pass}`), checks: pg.checks },
      versions: { testingCore: TESTING_CORE_VERSION, assets: EVAL_PROTOCOL },
      // 仪器来源（**additive**，schema 里是可选属性）：判决是在哪两次部署上测出来的、有没有有损降级。
      // 注意与下面 `trainSpecific.instrument: 'serving'`（一个**字符串**，指"判定仪器是 serving"）
      // 不是同一个东西：这里记录的是"那次 serving 部署的**具体配置**"。
      instrument: { base: baseInstrument, candidate: candidateInstrument },
      // training-eval-contract §3 要求训练侧 eval.json 追加 trainSpecific
      trainSpecific: {
        lossCurve: {
          step: selection.curve?.steps ?? [],
          train: selection.curve?.train ?? [],
          val: selection.curve?.val ?? [],
          valAt: 'container-eval',
          selectionOnly: true,
          note: '容器内周期评估 ⇒ **只作选点信号**，不得进入任何门禁判定（§5.2）',
        },
        overfit: { ...overfit, metric: 'negative_rule_reward（保序变换后的 loss 类度量）', instrument: 'serving' },
        selectedCheckpoint: {
          name: selection.selected?.name ?? null, step: selection.selected?.step ?? null,
          payloadHash: selection.selected?.payloadHash ?? null, selectionRule: selection.selectionRule,
          selectionOnly: true, isFinalStep: selection.selected?.isFinalStep !== false,
        },
        reward: { ...rewardDescriptor(), calls: run.reward?.calls ?? null },
        instrument2,
        klAccounting: run.kl ?? null,
        beta: run.beta ?? null,
        artifactAssertion,
        mergeInfo,
        leakGate: { hashOverlap: lk.hashOverlap, substringContained: lk.substringContained },
      },
    }
    // 只在**必需数字都能测出**时才写 eval-result.json：schema 要求 metrics[].value 是 number，
    // 而"测不出来"必须表现为"没有结论"，不能写成 null/0 冒充一个数（那正是本契约禁止的那类伪造）。
    const requiredNums = [results.rl.preference.meanReward, results.rl.mmlu.accuracy, results.rl.lm.meanNll]
    const measurementComplete = requiredNums.every((x) => typeof x === 'number' && Number.isFinite(x))
    let evalResultPath = null
    if (measurementComplete) {
      writeFileSync(join(run.runDir, 'eval-result.json'), `${JSON.stringify(evalResult, null, 2)}\n`)
      evalResultPath = join(run.runDir, 'eval-result.json')
    } else {
      log(`测量不完整（preference=${results.rl.preference.meanReward} mmlu=${results.rl.mmlu.accuracy} lmNll=${results.rl.lm.meanNll}）⇒ **不写 eval-result.json**（不伪造数字）`)
    }

    const ar = await audit(root, {
      capability: 'validate', action: 'validate', phase: 'end', runId, outcome: pass ? 'pass' : 'fail', measurementComplete,
      preferenceGate: pg.pass, pairedSignificance: pg.pairedSignificance?.pass === true, pairedMeanDelta: pg.pairedSignificance?.meanDelta ?? null,
      validityOk: pg.validityOk, forgettingGate: fg.pass, overfitPass: overfit.pass, overfitRatio: overfit.o3_overfitBudget?.ratio ?? null,
      rewardBase: results.base.preference.meanReward, rewardPolicy: results.rl.preference.meanReward,
      mmluBase: results.base.mmlu.accuracy, mmluPolicy: results.rl.mmlu.accuracy,
      lmPplBase: results.base.lm.ppl, lmPplPolicy: results.rl.lm.ppl, ms: Date.now() - t0,
    })
    // **缺陷 15**：部分测量**不得**入缓存（fail-closed）。只有能证明完整的测量才写入。
    // **N1**：写入的条目**自带**提示集身份（`promptSetId`）—— 读取侧据此复核（第二道闸），
    //        "键命中"与"身份一致"是两件事，都要成立才可复用。
    const baseCacheNext = (() => {
      if (cached) return null                       // 本次复用了完整缓存 ⇒ 保持原样
      const chk = measurementCompleteness(results.base)
      if (!chk.complete) { log(`  基线测量不完整（${chk.reason}）⇒ **不写入缓存**（fail-closed）`); return null }
      return {
        ...attachIntegrity(results.base),
        // additive：不改任何测量数字，只记录"这条基线是在哪个集上测的"
        promptSetId: promptSet.id,
        promptSetDir: promptSet.datasetDir,
        promptSetDeclaredAt: promptSet.declaredAt,
        prefMaxTokens, prefMaxTokensSource: instrumentParam.source,
      }
    })()
    await updateRegistry(root, (doc) => ({
      ...doc,
      baselineCache: baseCacheNext ? { ...(doc.baselineCache ?? {}), [cacheKey]: baseCacheNext } : (doc.baselineCache ?? {}),
      runs: (doc.runs || []).map((r) => (r.id === runId ? {
        ...r, status: pass ? 'validated' : 'gate-failed', validatedAt: new Date().toISOString(),
        endpoint: rlEndpoint,
        evaluation: {
          thresholds: thresholdsResolved, preferenceGate: pg, forgettingGate: fg, overfit,
          base: summarize(results.base), policy: summarize(results.rl),
          // N2/N1：本次判决用的**仪器参数（含来源）**与**评估集身份**（additive；判决可复核）
          instrument2,
          cacheKey,
          baselineFromCache: Boolean(cached),
        },
        decidedBudget: { gpuHoursTrain: r.result?.gpuHours ?? null, gpuHoursValidate: Number(((Date.now() - t0) / 3_600_000).toFixed(4)) },
        evalResult: evalResultPath ? { schema: evalResult.schema, path: evalResultPath } : { schema: evalResult.schema, path: null, reason: 'measurement-incomplete（不写产物，避免伪造数字）' },
        mergedEvaluated: mergedDir,
        artifactAssertion,
        auditSeqs: [...(r.auditSeqs || []), ar.seq],
      } : r)),
    }), { schemaPath: schemaPath() })

    return {
      runId, pass,
      preferenceGate: pg, forgettingGate: fg, overfit,
      endpoint: rlEndpoint,
      base: summarize(results.base), policy: summarize(results.rl),
      mergedEvaluated: mergedDir, selection, artifactAssertion,
      measurementComplete,
      evalResultPath,
      // N2：仪器参数的**值与来源**（calibrated-default = 调用方省略、回落到已标定档）
      instrument2,
      baseline: { fromCache: Boolean(cached), cacheKey },
      audit: { seq: ar.seq, hash: ar.hash }, ms: Date.now() - t0,
    }
  } finally {
    if (teardown) {
      for (const [label, ep, tok] of [['rl', rlEndpoint, ownerToken], ['base', baseEndpoint, baseOwnerToken]]) {
        if (!ep) continue
        try {
          const r = await callWithConfirm(client, 'inference_serving_teardown', { selector: ep.id, ownerToken: tok }, { log })
          log(`  teardown ${label}: ok=${r.ok} ${r.code ?? ''}`)
        } catch (e) { log(`  teardown ${label} 异常：${e.message}`) }
      }
    }
    await client?.close?.()
  }
}

/**
 * O3 过拟合预算 —— **直接 import 冻结契约的唯一参考实现**
 * （`inference-training/checks/overfit-criterion.mjs` 的 `evaluateOverfit`），不重写一份。
 *
 * 度量朝向：参考实现把输入当 loss 类（越低越好）。RL 的规则奖励越高越好，
 * 故喂 `−reward`（**保序变换**）：improvement 的含义因此仍是"变好了多少"，
 * 而 σ_base 也随之变成"基线在该数据上奖励的离散度"。返回时把 metric 标签改回如实描述。
 */
/**
 * `measureAt` 的测量结果 → O3 判据的输入（**键名映射的唯一归属地**）。
 *
 * ⚠️ 这里曾经是一个**静默的键名 bug**：留出侧被写成了 `prefVal`，而 `measureAt` 返回的键是
 * `preference`（留出）与 `prefTrain`（训练）。于是 `neg(undefined) → []` ⇒ val 配对样本为 0
 * ⇒ O3 **静默**退化成 `E_SAMPLE_TOO_SMALL`（`pass` 变成 undefined），
 * 只有真机第一次 validate 才把它暴露出来。
 * 所以把映射抽成这个函数并**单测它**：映射错了就是红的，而不是悄悄少算一半。
 */
export function overfitPerText(m) {
  const neg = (xs) => (Array.isArray(xs) ? xs.map((x) => -x) : [])
  return { train: { perText: neg(m?.prefTrain?.rewards) }, val: { perText: neg(m?.preference?.rewards) } }
}

async function evaluateOverfitRL({ base, model }) {
  if (!existsSync(OVERFIT_CRITERION)) {
    return { error: 'E_CRITERION_MISSING', hint: `找不到 overfit 判据参考实现：${OVERFIT_CRITERION}`, pass: null }
  }
  const mod = await import(`file://${OVERFIT_CRITERION}`)
  const r = mod.evaluateOverfit({ base: overfitPerText(base), model: overfitPerText(model) })
  return { ...r, metric: 'negative_rule_reward', source: 'inference-training/checks/overfit-criterion.mjs#evaluateOverfit（冻结契约的唯一参考实现，import 而非重写）' }
}

/* ══════════════════════════════════════════════════════════ 4. rollback */

export async function rollback({ root = paths().root, runId, purge = false, log = () => {} }) {
  const reg = await readRegistry(root, schemaPath())
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  await audit(root, { capability: 'rollback', action: 'rollback', phase: 'begin', runId, purge })
  let purged = []
  if (purge) {
    // **只允许删本包的 run 目录**（越界删 /etc 那类事故的正面防线）
    for (const d of [run.runDir, join(paths(root).runsDir, runId, 'merged'), join(paths(root).runsDir, runId, 'merged-selected')]) {
      if (d && existsSync(d) && d.startsWith(paths(root).runsDir)) { await fs.rm(d, { recursive: true, force: true }); purged.push(d) }
    }
  }
  const ar = await audit(root, { capability: 'rollback', action: 'rollback', phase: 'end', runId, purged: purged.length })
  await withRun(root, runId, (r) => ({ ...r, status: 'rolled-back', rolledBackAt: new Date().toISOString(), purged, auditSeqs: [...(r.auditSeqs || []), ar.seq] }))
  log(`run ${runId} → rolled-back（purge=${purge}, 删除 ${purged.length} 个目录）`)
  return { runId, status: 'rolled-back', purged, audit: { seq: ar.seq, hash: ar.hash } }
}

/* ══════════════════════════════════════════════════════════ 5. get_report */

export async function getReport({ root = paths().root, runId = null, limit = 10 }) {
  const reg = await readRegistry(root, schemaPath())
  const chain = await auditVerify(root)
  const ghost = await reconcile({ root, dryRun: true })
  if (!runId) {
    return {
      staleRunning: ghost.stale.map((x) => ({ id: x.id, startedAt: x.startedAt, ageMs: x.ageMs, reason: x.reason })),
      runs: (reg.runs || []).slice(-limit).map((r) => ({
        id: r.id, status: r.status, startedAt: r.startedAt, mode: r.mode ?? null,
        steps: r.result?.steps ?? null, gpuHours: r.result?.gpuHours ?? null,
        sigmaR: r.calibration?.sigmaR ?? null, beta: r.beta ?? null, klMean: r.kl?.klMean ?? r.kl?.mean ?? null,
        preferenceReward: r.evaluation?.policy?.preference?.meanReward ?? null,
        baselineReward: r.evaluation?.base?.preference?.meanReward ?? null,
        pass: r.evaluation ? (r.evaluation.preferenceGate?.pass && r.evaluation.forgettingGate?.pass && r.evaluation.overfit?.pass) : null,
      })),
      promptSets: listPromptSets(P(root).promptsDir),
      audit: { ok: chain.ok, shards: chain.shards },
    }
  }
  const run = (reg.runs || []).find((r) => r.id === runId)
  if (!run) throw new ToolError(ERROR_CODES.E_RUN_NOT_FOUND, { hint: `run 不存在：${runId}`, recoverable: true })
  return {
    run,
    reward: rewardDescriptor(),
    audit: { ok: chain.ok, shards: chain.shards, results: chain.results.map((x) => ({ day: x.day, checked: x.checked, ok: x.ok })) },
  }
}

/* ══════════════════════════════════════════════════════════ 6. health */

export async function health({ root = paths().root, deep = false }) {
  const p = P(root)
  const probe = await sandboxProbe()
  const chain = await auditVerify(root)
  let gpu = null
  try { gpu = execFileSync('nvidia-smi', ['--query-gpu=name,driver_version', '--format=csv,noheader'], { encoding: 'utf8', timeout: 20000 }).trim() } catch { gpu = null }
  let imageOk = false
  let imageId = null
  try {
    imageId = execFileSync('docker', ['images', '-q', TRAINING_IMAGE], { encoding: 'utf8', timeout: 20000 }).trim() || null
    imageOk = Boolean(imageId)
  } catch { imageOk = false }
  const baseOk = process.env.DSH_BASE_MODEL ? existsSync(join(process.env.DSH_BASE_MODEL, 'config.json')) : false
  let servingOk = false
  let servingPath = null
  try { const sp = await resolveServingPath({ root }); servingPath = sp.serverPath; servingOk = true } catch { servingOk = false }
  const reg = await readRegistry(root, schemaPath())
  // **只读**探测幽灵 running（dryRun ⇒ 不写状态；health 承诺只读）
  const ghost = await reconcile({ root, dryRun: true })
  const criterionOk = existsSync(OVERFIT_CRITERION)
  return {
    ok: probe.container.available && chain.ok && imageOk && baseOk && servingOk,
    sandboxTiers: { process: probe.process.available, container: probe.container.available, vm: probe.vm.available },
    sandboxEvidence: probe.container.evidence,
    image: { name: TRAINING_IMAGE, available: imageOk, id: imageId },
    gpu,
    baseModel: { path: process.env.DSH_BASE_MODEL ?? null, available: baseOk },
    serving: { available: servingOk, serverPath: servingPath },
    audit: { ok: chain.ok, shards: chain.shards, detail: chain.results.map((x) => ({ day: x.day, checked: x.checked, ok: x.ok })) },
    registry: {
      runs: (reg.runs || []).length, file: p.registryFile,
      running: ghost.checked,
      // "在跑"与"还有主人在跑"是两件事：这里把后者单列出来（幽灵 running 必须可见）
      staleRunning: ghost.stale.map((x) => ({ id: x.id, startedAt: x.startedAt, ageMs: x.ageMs, runDirAgeMs: x.runDirAgeMs, reason: x.reason })),
      staleRunningCount: ghost.stale.length,
      reconcileRule: 'stale ⟺ 归属进程不可用（本机 pid 已死 / 无 owner 记录 / 异机）且 run 目录最新写入超过 15 分钟（或目录不存在）',
    },
    promptSets: listPromptSets(p.promptsDir).length,
    reward: rewardDescriptor(),
    trainer: { mode: 'monolithic', plugin: 'rl-grpo', framework: 'trl/GRPOTrainer', klFormula: 'beta = sigmaR / klBudget' },
    criterion: { overfitReference: OVERFIT_CRITERION, available: criterionOk },
    limits: DEFAULT_LIMITS,
    // 幽灵 running 的**顶层**可见性：不要求调用方去翻 registry 子对象才看见
    // （否则"有没有在跑？"这个判断仍旧容易被忽略 —— 那正是 yota 那次失败的形态）。
    staleRunning: ghost.stale.map((x) => ({ id: x.id, startedAt: x.startedAt, runDirAgeMs: x.runDirAgeMs, reason: x.reason })),
    staleRunningCount: ghost.stale.length,
    warnings: ghost.stale.length
      ? ghost.stale.map((x) => `stale-running: ${x.id}（${x.runDirAgeMs === null ? 'run 目录无产物' : `目录 ${Math.round(x.runDirAgeMs / 1000)}s 未写入`}）—— 跑 \`training-rl reconcile --apply\` 回收`)
      : [],
    version: { pkg: PACKAGE_VERSION, schema: SCHEMA_VERSION, reward: RULE_REWARD_VERSION },
    // `deep` 为 false 时**不要留一个 undefined 属性**：DSH 要求无损 JSON，
    // 而 `undefined` 不是 JSON 值（JSON.stringify 静默丢掉它，校验器却拒绝）。
    ...(deep ? { deep: { paths: p } } : {}),
  }
}

// 供单测直接调用（判据的**唯一实现**在 core 里，测试不得另写一份）
// 注：`reconcile` / `isPidAlive` 已在上面 `export function` 声明处导出。
export { PKG_ROOT, loadCheckpoint, selectCheckpoint, evaluateOverfitRL, summarize, loadEvalSets, reconcile as _reconcile }
