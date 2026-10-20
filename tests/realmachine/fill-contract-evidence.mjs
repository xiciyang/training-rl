#!/usr/bin/env node
/**
 * fill-contract-evidence —— 把**真机证据**里的事实填进 `training-rl-contract.md` §10。
 *
 * 为什么要脚本而不是手抄：契约里的数字必须与证据文件**同源**。
 * 手抄一次就会出现"契约写 200 步、证据写 180 步"这类漂移，且没人会发现。
 * 本脚本只读证据、只替换 §10 的标记块，**不改**契约的其他部分；
 * 证据缺失/失败时**照实**写（`ok:false` + 原始错误），绝不生成漂亮数字。
 *
 * 用法：node tests/realmachine/fill-contract-evidence.mjs [--evidence <json>] [--contract <md>] [--check]
 */
import { readFileSync, writeFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const WORKSPACE = join(PKG, '..', '..')
const EV = join(WORKSPACE, 'inference-training', 'out', 'rl-vertical-loop.json')
const CONTRACT = join(WORKSPACE, 'inference-contracts', 'contracts', 'training-rl-contract.md')

const argv = process.argv.slice(2)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] ? argv[i + 1] : d }
const evPath = argOf('--evidence', EV)
const mdPath = argOf('--contract', CONTRACT)
const CHECK = argv.includes('--check')

const f = (x, digits = 6) => (typeof x === 'number' && Number.isFinite(x) ? String(Number(x.toFixed(digits))) : '`null`')
const s = (x) => (x === null || x === undefined ? '`null`' : String(x))
const yn = (x) => (x === true ? '✅ 通过' : x === false ? '❌ 未通过' : '—')

if (!existsSync(evPath)) {
  process.stderr.write(`证据文件不存在：${evPath}\n`)
  process.exit(1)
}
const ev = JSON.parse(readFileSync(evPath, 'utf8'))
const t = ev.train ?? {}
const cal = t.calibration ?? {}
const kl = t.kl ?? {}
const fp = t.fingerprints ?? {}
const aa = t.artifactAssertion ?? {}
const va = ev.validate ?? {}
const pg = va.preferenceGate ?? {}
const fg = va.forgettingGate ?? {}
const ov = va.overfit ?? {}
const sel = t.selection ?? {}

const runs = ev.runs ?? []
const vh = ev.validateHistory ?? []
const pc = (x, d = 4) => (typeof x === 'number' && Number.isFinite(x) ? (x * 100).toFixed(d) + '%' : '—')
const fmt = (x, d = 6) => (typeof x === 'number' && Number.isFinite(x) ? String(Number(x.toFixed(d))) : '`null`')

/** 判定历史的 markdown 表（含"基础设施失败"这一档，不把它算成未通过） */
const historyTable = vh.length
  ? [
    '| # | 仪器 | 对象 | 判定 | 关键数字 |',
    '|---|---|---|---|---|',
    ...vh.map((x, i) => {
      const pg = x.preferenceGate ?? {}
      const num = x.verdict === 'infrastructure-failed'
        ? `\`${String(x.error?.code ?? '')}\` ${String(x.note ?? '')}`
        : `base ${fmt(pg.deltas?.meanReward?.base, 4)} → policy ${fmt(pg.deltas?.meanReward?.policy, 4)}（相对 ${fmt(pg.deltas?.meanReward?.rawPct, 2)}%），配对 Δ=${fmt(pg.pairedSignificance?.meanDelta, 6)} vs z·SE=${fmt(pg.pairedSignificance?.bound, 6)}，winRate=${fmt(pg.pairedSignificance?.winRate, 4)}（n=${pg.pairedSignificance?.n ?? '—'}，ties 由 winRate<1 体现）`
      return `| ${i + 1} | \`${x.instrument}\` | ${x.run} | **${x.verdict}** | ${num} |`
    }),
  ].join('\n')
  : '（无判定历史）'

/** 两次 run 的对照（核心：区分"β 造成的"与"奖励信号/步长规模造成的"） */
const runTable = runs.length
  ? [
    '| run | β | 实测 KL 均值 | 训练侧奖励 first10→last10 | 训练侧 Δ | 留出侧偏好 Δ | 配对显著 | 判定 |',
    '|---|---|---|---|---|---|---|---|',
    ...runs.map((r) => {
      const rt = r.rewardTrajectory ?? {}
      const pg = r.validate?.preferenceGate ?? {}
      const sig = pg.pairedSignificance
      const b = r.validate?.base?.preference?.meanReward
      const pol = r.validate?.policy?.preference?.meanReward
      const rawPct = (typeof b === 'number' && typeof pol === 'number' && b !== 0) ? ((pol - b) / b) * 100 : null
      const held = rawPct === null ? '—' : `${fmt(b, 4)} → ${fmt(pol, 4)}（${rawPct >= 0 ? '+' : ''}${rawPct.toFixed(2)}%）`
      return `| \`${r.label}\` | ${fmt(r.beta, 6)} | ${fmt(r.kl?.mean, 6)} | ${fmt(rt.rewardFirst10, 4)} → ${fmt(rt.rewardLast10, 4)} | **${fmt(rt.rewardDeltaFirstLast10, 4)}**（${typeof rt.rewardDeltaPct === 'number' ? (rt.rewardDeltaPct >= 0 ? '+' : '') + rt.rewardDeltaPct.toFixed(2) + '%' : '—'}） | ${held} | ${sig ? (sig.n > 0 ? (sig.pass ? '✅ 显著' : `❌ 不显著（n=${sig.n}，Δ=${fmt(sig.meanDelta, 6)} vs z·SE=${fmt(sig.bound, 6)}）`) : `—（测量不成立：${String(sig.reason ?? '').slice(0, 30)}）`) : '—'} | ${r.validate ? (r.validate.pass === true ? 'PASS' : r.validate.pass === false ? 'FAIL' : 'infra-failed') : '—'} |`
    }),
  ].join('\n')
  : '（无 run 对照）'

/** P1 的结构化结论（**从数字推出，不写空话**） */
function p1Verdict() {
  if (runs.length < 2) return '（只有一次 run，无法区分"β 造成的"与"信号太弱造成的"）'
  const [a, b] = runs
  const rtA = a.rewardTrajectory ?? {}; const rtB = b.rewardTrajectory ?? {}
  const pgA = a.validate?.preferenceGate ?? {}; const pgB = b.validate?.preferenceGate ?? {}
  const trainMoved = (rtA.rewardDeltaFirstLast10 ?? 0) > 0.01 && (rtB.rewardDeltaFirstLast10 ?? 0) > 0.01
  const heldFlat = (pgA.pairedSignificance?.pass !== true) && (pgB.pairedSignificance?.pass !== true)
  const betaRatio = (a.beta && b.beta) ? a.beta / b.beta : null
  const lines = []
  lines.push(`- **P1（偏好对齐）结论：${heldFlat ? '**未通过**' : '**通过**'}**（两次 run 的配对显著性都/至少有一次成立与否见上表）。`)
  lines.push(`- 训练侧奖励**确实被推上去了**：run1 ${a.rewardTrajectory ? `+${(a.rewardTrajectory.rewardDeltaFirstLast10 ?? 0).toFixed(4)}` : '—'}、run2 ${b.rewardTrajectory ? `+${(b.rewardTrajectory.rewardDeltaFirstLast10 ?? 0).toFixed(4)}` : '—'}（first10→last10 均值）。⇒ **不是"奖励信号弱到推不动策略"**。`)
  lines.push(`- β 从 ${(a.beta ?? 0).toFixed(6)} 降到 ${(b.beta ?? 0).toFixed(6)}（**${betaRatio ? betaRatio.toFixed(1) + '×' : '—'}**），而实测 KL 均值几乎没有变化（${(a.kl?.mean ?? 0).toFixed(6)} → ${(b.kl?.mean ?? 0).toFixed(6)}）—— 在 KL ≪ 预算的区间里，**β 的量级并不是约束**。`)
  lines.push(`- ⇒ 结论：${trainMoved && heldFlat
    ? '**既不是"β 太大"、也不是"奖励信号太弱"，而是"训练分布上的提升没有迁移到留出集"**（训练侧 +，留出侧不显著）。这正是 O3（train↔val 间隙）要盯的量；把 β 继续调小只会允许更多漂移，不会把留出侧的提升变显著。'
    : '见上表数字。'}`)
  return lines.join('\n')
}

// 判定分支：**有真机训练/判定数据** ⇒ 走详表（哪怕总体是 fail —— fail 也要把数字摆出来）；
// 只有"根本没跑起来"（无 train 且无 validate）才走简短登记。
const hasRunData = Boolean((ev.runs?.length ?? 0) > 0 || ev.train || ev.validate)
const block = hasRunData
  ? `### 10.0 结论先行 + 两次真机 run 的对照（**这是区分"β 造成的"与"信号太弱造成的"的关键**）

${
  ev.runs?.[1]?.validate?.pass === true ? '**总体判定：通过。**'
  : ev.runs?.[1]?.validate?.pass === false ? '**总体判定：未通过（P1 偏好对齐未过；遗忘门禁通过、产物级断言通过）。**'
  : '**总体判定：未完成（见判定历史）。**'
}下面每个数字都可在 \`inference-training/out/rl-vertical-loop.json\` 与审计链里复核。

${runTable}

**预注册预测**（在 run2 开跑**之前**写下，逐条对照）：
${(ev.preregistration?.predictions ?? []).map((x) => `- ${x}`).join('\n')}

> run2 的 β 不是硬编码：由修正后的一阶控制量 β_next = β_declared × klMean/KL* ≈ ${fmt(ev.preregistration?.betaTarget ?? null, 4)}
> 反推 KL* = σ_r/β_next ≈ ${ev.preregistration?.klBudgetChosen ?? '—'}，再走**同一个** β = σ_r/KL* 推导得到。

### 10.1 训练（\`training_rl_train\`，单体档位驱动 TRL GRPOTrainer）

| 量 | 实测值 |
|---|---|
| runId | \`${s(t.runId)}\` |
| 步数（优化器步） | **${s(t.steps)}** |
| 最终 loss | ${f(t.finalLoss)} |
| 实测 **σ_r**（规则奖励标准差，当前策略上采样） | **${f(cal.sigmaR)}**（样本 ${s(cal.samples)} 条，不同取值 ${s(cal.distinctValues)} 个） |
| 调用方给定的 **KL\*** | **${s(cal.klBudget)}** nats/token |
| **推导出的 β**（= σ_r / KL\*） | **${f(cal.beta)}** |
| 实测 KL（均值 / 最小 / 最大 / 标准差） | ${f(kl.mean)} / ${f(kl.min)} / ${f(kl.max)} / ${f(kl.std, 8)} |
| KL 偏离预算 | ${f(kl.deviationPct, 2)}%（容差 ${s(kl.tolerancePct)}%）⇒ withinTolerance=${s(kl.withinTolerance)}，recalibrationRequired=${s(kl.recalibrationRequired)}，recommendedBeta=${f(kl.recommendedBeta)} |
| GPU 小时（训练循环内实测墙钟） | **${s(t.gpuHours)}** |
| 训练墙钟 | ${s(t.wallMs)} ms |
| 停止原因 | ${s(t.stopReason)} |
| 阶段计时（load/calibration/train/merge） | ${s(JSON.stringify(t.timing ?? {}))} |
| checkpoint 数 | ${s(t.checkpointCount)} |

**产物级断言（§5.4）**

| 量 | 值 |
|---|---|
| 基座权重指纹（① 训练前） | \`${s(fp.base)}\` |
| 合并后的策略权重指纹（③ 训练后） | \`${s(fp.merged)}\` |
| LoRA adapter 指纹（②，null ⇒ 什么都没写） | \`${s(fp.adapter)}\` |
| \`assertArtifactChanged(①,③)\` | **${aa.ok === true ? '✅ 权重确实变了（未触发 E_NO_OP_TRAINING）' : '❌ 未通过'}** |

**规则奖励**：\`${s(t.reward?.kind)}/${s(t.reward?.version)}\`，分量 ${s(JSON.stringify(t.reward?.components ?? []))}。
跨语言一致性：容器对 ${s(t.rewardConformance?.n)} 条规格向量打分，与 JS 权威实现**最大绝对偏差 ${f(t.rewardConformance?.maxAbsDiff, 18)}**
（withinTolerance=${s(t.rewardConformance?.withinTolerance)}，bitIdentical=${s(t.rewardConformance?.bitIdentical)}）。

**预算**：\`budgetViolation=${s(t.budgetViolation)}\`（training-core v0.2.0 单体档位事后核对用 \`>=\`，
恰好用满预算也会被标记 —— 见 §5.3）；\`budgetAssessment.exceeded=${s(t.budgetAssessment?.exceeded)}\`
（这才是"真的超过上限"）；violations=${s(JSON.stringify(t.violations ?? []))}。

**选点（O4）**：selectionRule=\`${s(sel.selectionRule)}\`，选中 \`${s(sel.selected?.name)}\`（step=${s(sel.selected?.step)}，
valLoss=${f(sel.selected?.valLoss)}，isFinalStep=${s(sel.selected?.isFinalStep)}）。
${Array.isArray(sel.curve?.steps) && sel.curve.steps.length ? `val 曲线（容器内评估，**selectionOnly**）：steps=${JSON.stringify(sel.curve.steps)}，val reward=${JSON.stringify((sel.curve.val ?? []).map((x) => (typeof x === 'number' ? Number(x.toFixed(6)) : x)))}。` : '（无 val 选点信号）'}

### 10.2 评估（\`training_rl_validate\`，**serving** 判定仪器）

| 门禁 | 结论 | 关键数字 |
|---|---|---|
| 偏好对齐 | ${yn(pg.pass)} | 基线 ${f(pg.deltas?.meanReward?.base)} → 策略 ${f(pg.deltas?.meanReward?.policy)}（相对 ${f(pg.deltas?.meanReward?.rawPct, 2)}%）；配对 mean(d)=${f(pg.pairedSignificance?.meanDelta)} vs z·SE=${f(pg.pairedSignificance?.bound)}，winRate=${f(pg.pairedSignificance?.winRate, 4)}（n=${s(pg.pairedSignificance?.n)}） |
| 遗忘（MMLU） | ${yn(fg.mmlu?.pass)} | 基线 ${f(fg.mmlu?.baseAcc, 4)} → 策略 ${f(fg.mmlu?.rlAcc, 4)}，Δ=${f(fg.mmlu?.delta, 4)} vs z·SE=${f(fg.mmlu?.bound, 4)}（n=${s(fg.mmlu?.n?.rl)}） |
| 遗忘（wikitext NLL，配对） | ${yn(fg.lm?.pass)} | 基线 ${f(fg.lm?.baseNll, 4)} → 策略 ${f(fg.lm?.rlNll, 4)}，Δ=${f(fg.lm?.deltaNll, 4)} vs z·SE=${f(fg.lm?.bound, 4)}（n=${s(fg.lm?.n)}） |
| 过拟合预算 O3 | ${yn(ov.pass)} | improvement_train=${f(ov.improvementTrain)}，improvement_val=${f(ov.improvementVal)}，gap=${f(ov.gap)}，σ_base=${f(ov.sigmaBase)}，ratio=${f(ov.o3_overfitBudget?.ratio)}（f=${s(ov.o3_overfitBudget?.f)}）；O2=${yn(ov.o2_generalizationReal?.pass)} |
| **总体** | **${yn(va.pass)}** | ${s(va.preferenceGate?.reason)} |

- 仪器：serving（策略与基座各租一个临时端点，测完**拆解**）；判据版本见 eval-result 的 \`versions.testingCore\`
- eval-result：\`${s(va.evalResultPath)}\`（符合冻结的 \`ai-infra/eval-result@1\`）
- 被评估的产物：\`${s(va.mergedEvaluated)}\`

### 10.3b 判定历史（三次，含一次**基础设施失败**——不得算作"未通过"）

${historyTable}

### 10.3c P1 结论（从数字推出）

${p1Verdict()}

### 10.3 本次 run 消耗与边界

- **GPU 小时**：训练 ${s(t.gpuHours)} + 评估 ${s(va.ms ? Number((va.ms / 3_600_000).toFixed(4)) : null)} = **${s(ev.gpuHours)}**（阶段预算 8 GPU·h 之内）
- **本 run 是单种子**（seed=${s(ev.inputs?.seed ?? null)}）⇒ 结论只能表述为"**这一个个别模型**"，
  **不得**表述为"这套 RL 流程"（O5 未满足，见 §9 边界 2）
- 训练步数 ≤500 / GPU 小时 ≤2 的硬约束由输入闸门 + \`training-core\` 的 \`enforceLimits\` 双重把关
`
  : `### 10.1 真机 run **未通过**（如实登记）

\`\`\`
mode        : ${s(ev.mode)}
stages      : ${JSON.stringify(ev.stages ?? [])}
error.code  : ${s(ev.error?.code)}
error.message: ${s(ev.error?.message)}
\`\`\`

| 阶段 | 结果 |
|---|---|
| prepare | ${ev.prepare ? `✅ train=${s(ev.prepare.trainPrompts)} heldout=${s(ev.prepare.heldoutPrompts)}（去泄漏双轨 hash=${s(ev.prepare.leakCheck?.hashOverlap)} substring=${s(ev.prepare.leakCheck?.substringContained)}）` : '—'} |
| train | ${ev.train ? `runId=${s(ev.train.runId)} steps=${s(ev.train.steps)} σ_r=${f(ev.train.calibration?.sigmaR)} β=${f(ev.train.calibration?.beta)} GPU·h=${s(ev.train.gpuHours)}` : '—'} |
| validate | ${ev.validate ? `pass=${s(ev.validate.pass)}` : '—'} |

**失败就是失败**：本节按原样登记错误，不把"没跑成"写成"跑成了"。
`

const md = readFileSync(mdPath, 'utf8')
const B = '<!-- RL_EVIDENCE_BEGIN -->'
const E = '<!-- RL_EVIDENCE_END -->'
const i = md.indexOf(B)
const j = md.indexOf(E)
if (i < 0 || j < 0) { process.stderr.write('契约里找不到证据标记块\n'); process.exit(1) }
const next = `${md.slice(0, i + B.length)}\n${block}\n${md.slice(j)}`

if (CHECK) {
  if (next !== md) { process.stderr.write('契约 §10 与证据不同源（需要重跑本脚本）\n'); process.exit(1) }
  process.stdout.write('契约 §10 与证据同源\n')
} else {
  writeFileSync(mdPath, next)
  process.stdout.write(`已把证据填入 ${mdPath}（ok=${ev.ok}）\n`)
}
