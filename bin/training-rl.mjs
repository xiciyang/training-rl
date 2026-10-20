#!/usr/bin/env node
/**
 * training-rl CLI（极薄）。
 *
 * ## 为什么 CLI 只有这几个子命令
 * 写能力一律走 MCP/DSH 工具（带确认令牌与治理信封）；CLI **故意不提供 train/validate 的旁路**
 * —— 旁路 = 没有审计、没有确认、没有风险声明。
 *
 * ## 唯一的例外：`reconcile`
 * 它是**可调用的回收入口**：把"外部杀死（SIGKILL/OOM/父进程被杀）留下的幽灵 running"
 * 对账成 `interrupted`。为什么需要 CLI 而不是第 7 个工具：本包的工具面被交付要求**钉死为 6 个**
 * （`training_rl_{prepare,train,validate,rollback,get_report,health}`）。
 * 于是入口有三个，且都不破坏"6 工具"约束：
 *   ① `train` 开跑前自动对账（真回收）；
 *   ② `health` / `get_report` **只读**探测并列出 `staleRunning`（可见性，默认就能看到）；
 *   ③ 本 CLI 的 `reconcile`（**默认 dry-run 只报告**，`--apply` 才写）。
 *
 * 用法：
 *   node bin/training-rl.mjs health|report|tools [--json]
 *   node bin/training-rl.mjs reconcile [--apply] [--stale-after-min 15] [--json]
 */
import { health, getReport, reconcile } from '../src/core.mjs'
import { ALL_TOOL_SPECS } from '../src/tools.mjs'

const argv = process.argv.slice(2)
const cmd = argv[0] || 'health'
const has = (k) => argv.includes(k)
const argOf = (k, d) => { const i = argv.indexOf(k); return i >= 0 && argv[i + 1] !== undefined ? argv[i + 1] : d }
const rootArg = argOf('--root', undefined)
const json = has('--json')

const out = async () => {
  if (cmd === 'health') return health({ root: rootArg })
  if (cmd === 'report') return getReport({ root: rootArg, runId: argOf('--run', null) ?? null })
  if (cmd === 'tools') return { tools: ALL_TOOL_SPECS.map((t) => ({ name: t.name, description: t.description.slice(0, 120) })) }
  if (cmd === 'reconcile') {
    const apply = has('--apply')
    const staleAfterMs = Number(argOf('--stale-after-min', '15')) * 60_000
    const r = await reconcile({ root: rootArg, staleAfterMs, dryRun: !apply, log: (m) => process.stderr.write(`${m}\n`) })
    return {
      mode: apply ? 'apply' : 'dry-run',
      checked: r.checked, stale: r.stale, alive: r.alive, interrupted: r.interrupted,
      staleAfterMs: r.staleAfterMs, host: r.host,
      hint: apply ? '已写 interrupted（含审计）' : '**只报告**：加 --apply 才写状态（默认 dry-run，与 inference_serving_reconcile 同口径）',
    }
  }
  throw new Error(`未知子命令：${cmd}（可用：health / report / tools / reconcile）`)
}

try {
  const r = await out()
  process.stdout.write(`${JSON.stringify(r, null, 2)}\n`)
} catch (e) {
  process.stderr.write(`training-rl ${cmd} 失败：${String(e?.message ?? e)}\n`)
  process.exitCode = 1
}
