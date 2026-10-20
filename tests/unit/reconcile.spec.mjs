/**
 * reconcile.spec —— **幽灵 `running` 的检测与回收**。
 *
 * ## 这条判据的由来（真机缺陷，不是想象）
 * `train` 的 catch 能记 `failed`，但**外部杀死**（SIGKILL/OOM/容器被 kill/父进程被杀）
 * 任何 in-process 处理器都抓不到。真机留下了一条永远 `running` 的幽灵
 * （`rl_20260929151030_yota` —— 第一次彩排被 `job_kill` 掐掉，run 目录只有 `trl/`、
 * 创建后再没被写过）。后果不是"不好看"：任何"现在有没有在跑？"的判断都会得到**错误答案**。
 *
 * 这条 spec 钉住三件事：
 *   ① 判据**不误伤**：归属进程还活着 ⇒ 再慢也不判陈旧；
 *   ② 判据**抓得住**：进程已死/无 owner 记录 + 目录长时间没被写 ⇒ 判陈旧；
 *   ③ 只读与写入**分离**：`dryRun` 绝不改状态（`health()`/`get_report()` 承诺只读），
 *      真回收必须走 `dryRun:false` 并留下审计。
 */
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, readFileSync, utimesSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir, hostname } from 'node:os'
import { fileURLToPath } from 'node:url'

import { reconcile, isPidAlive, health, getReport } from '../../src/core.mjs'
import { paths, updateRegistry, readRegistry, auditVerify } from '../../src/runtime.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
export const NAME = 'reconcile（幽灵 running：检测 / 回收 / 只读分离）'

const OLD = new Date('2020-01-01T00:00:00Z')
const mkRun = (id, extra) => ({
  id, status: 'running', startedAt: '2020-01-01T00:00:00.000Z',
  baseModel: { path: '/b' }, dataset: { dir: '/d' }, config: {}, budget: { maxSteps: 1, maxGpuHours: 1 },
  runDir: null, ...extra,
})

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tr-rec-'))
  const root = join(tmp, 'ai-infra')
  const p = paths(root)
  mkdirSync(p.stateDir, { recursive: true })

  // 三个场景：① 归属进程活着（不能误伤）② 进程已死 + 目录陈旧（应回收）③ 无 owner + 目录不存在（应回收）
  const aliveDir = join(tmp, 'alive'); mkdirSync(aliveDir, { recursive: true }); writeFileSync(join(aliveDir, 'metrics.jsonl'), 'x\n')
  const deadDir = join(tmp, 'dead'); mkdirSync(deadDir, { recursive: true }); writeFileSync(join(deadDir, 'trl.json'), 'y\n')
  utimesSync(join(deadDir, 'trl.json'), OLD, OLD)
  utimesSync(deadDir, OLD, OLD)

  await updateRegistry(root, (doc) => ({
    ...doc,
    runs: [
      mkRun('rl_alive', { runDir: aliveDir, owner: { host: hostname(), pid: process.pid, startedAt: 'x' } }),
      mkRun('rl_dead', { runDir: deadDir, owner: { host: hostname(), pid: 2147483646, startedAt: 'x' } }),
      mkRun('rl_nodir', { runDir: join(tmp, 'never-existed'), owner: null }),
    ],
  }), { schemaPath: null })

  /* ① pid 存活判定 */
  t.assert(isPidAlive(process.pid) === true, 'isPidAlive(本进程) === true')
  t.assert(isPidAlive(2147483646) === false, 'isPidAlive(不存在的 pid) === false')
  t.assert(isPidAlive('nonsense') === null && isPidAlive(NaN) === null, '无法判定的 pid ⇒ 返回 null（不冒充 false）')

  /* ② dryRun：只报告、**不写状态** */
  const before = readFileSync(p.registryFile, 'utf8')
  const dry = await reconcile({ root, dryRun: true })
  const after = readFileSync(p.registryFile, 'utf8')
  t.assert(before === after, 'dryRun 探测**不修改**注册表（health/get_report 承诺只读）')
  t.assert(dry.checked === 3, `探测了 3 条 running（实际 ${dry.checked}）`)
  t.assert(dry.stale.length === 2, `判陈旧 2 条（实际 ${dry.stale.length}：${dry.stale.map((x) => x.id).join(',')}）`)
  t.assert(dry.alive.length === 1 && dry.alive[0].id === 'rl_alive', '归属进程活着的那条**不被误伤**（规则 pid-alive）')
  t.assert(dry.stale.some((x) => x.id === 'rl_dead' && x.owner.includes('dead')), '已死进程 + 目录陈旧 ⇒ 陈旧，且证据里点名 owner 已死')
  t.assert(dry.stale.some((x) => x.id === 'rl_nodir' && x.owner.includes('absent')), '无 owner 记录 + 目录不存在 ⇒ 陈旧，且证据里点名 owner 缺失')
  t.assert(dry.stale.every((x) => /^stale-running: /.test(x.reason)), '每条都带可读的 reason（不是只有一个布尔）')

  /* ③ 真回收：写 interrupted + 审计 */
  const rep = await reconcile({ root, dryRun: false })
  t.assert(rep.interrupted.length === 2, '真回收了 2 条')
  const reg = await readRegistry(root, null)
  const byId = Object.fromEntries(reg.runs.map((r) => [r.id, r]))
  t.assert(byId.rl_dead.status === 'interrupted' && byId.rl_nodir.status === 'interrupted', '幽灵 running → **interrupted**（不是 failed，也不是悄悄留着 running）')
  t.assert(typeof byId.rl_dead.interruptedReason === 'string' && byId.rl_dead.interruptedReason.includes('stale-running'), '回收留下 interruptedReason')
  t.assert(typeof byId.rl_dead.reconciledAt === 'string' && typeof byId.rl_dead.interruptedAt === 'string', '回收留下时间戳（可追溯）')
  t.assert(byId.rl_alive.status === 'running', '活着的那条状态**不变**')
  t.assert((byId.rl_dead.auditSeqs ?? []).length === 1, '回收写进了审计链（auditSeq 已记账）')
  const chain = await auditVerify(root)
  t.assert(chain.ok === true, '回收后审计链仍然完整（可复算）')
  t.assert(rep.interrupted.every((x) => Number.isInteger(x.auditSeq)), '回收结果里回显审计序号')

  /* ④ 幂等：再跑一次不应重复回收 */
  const rep2 = await reconcile({ root, dryRun: false })
  t.assert(rep2.checked === 1 && rep2.interrupted.length === 0, '再跑不重复回收（只有 1 条 running，且它是活的）')

  /* ⑤ health / get_report 必须把幽灵**可见**（而不是只在 YAML 里） */
  await updateRegistry(root, (doc) => ({ ...doc, runs: doc.runs.map((r) => (r.id === 'rl_alive' ? r : r)) }), { schemaPath: null })
  const h = await health({ root })
  t.assert(h.registry?.staleRunningCount === 0, `health 报告 staleRunningCount=0（实际 ${h.registry?.staleRunningCount}）`)
  t.assert(typeof h.registry?.reconcileRule === 'string' && h.registry.reconcileRule.includes('stale ⟺'), 'health 报告判据本身（可复核，不是黑盒）')
  t.assert(h.registry?.running === 1, 'health 报告 running 计数')
  // 再制造一条幽灵，确认 health 能看见（只读）
  const ghostDir = join(tmp, 'ghost2'); mkdirSync(ghostDir, { recursive: true }); writeFileSync(join(ghostDir, 'a'), 'z')
  utimesSync(ghostDir, OLD, OLD); utimesSync(join(ghostDir, 'a'), OLD, OLD)
  await updateRegistry(root, (doc) => ({ ...doc, runs: [...doc.runs, mkRun('rl_ghost2', { runDir: ghostDir, owner: null })] }), { schemaPath: null })
  const h2 = await health({ root })
  t.assert(h2.registry?.staleRunningCount === 1 && h2.registry.staleRunning[0].id === 'rl_ghost2', 'health 能看见新幽灵（只读探测）')
  t.assert(h2.registry.staleRunning[0].runDirAgeMs > 1000, '幽灵报告里带"目录多久没被写"的证据')
  const hreg = await readRegistry(root, null)
  t.assert(hreg.runs.find((r) => r.id === 'rl_ghost2').status === 'running', 'health **没有**顺手改状态（只读承诺）')
  const gr = await getReport({ root })
  t.assert(Array.isArray(gr.staleRunning) && gr.staleRunning.length === 1, 'get_report 也列出幽灵 running（列表视图可见）')

  /* ⑥ **可调用入口**：不能只在 train 里隐式对账（否则没人开新 run 时幽灵永远留着）
   *    · health/get_report：只读**可见**（顶层 staleRunning + warnings，不必翻子对象）
   *    · CLI `reconcile`：默认 dry-run 只报告，--apply 才写
   */
  t.assert(Array.isArray(h2.warnings) && h2.warnings.length === 1 && /stale-running/.test(h2.warnings[0]), 'health 顶层给 warnings（不必翻 registry 子对象就能看见）')
  t.assert(Array.isArray(h2.staleRunning) && h2.staleRunning[0].id === 'rl_ghost2', 'health 顶层给 staleRunning 明细')
  t.assert(/training-rl reconcile --apply/.test(h2.warnings[0]), 'warnings 里直接给出可执行的回收命令（可操作）')
  {
    const { execFileSync } = await import('node:child_process')
    const cli = join(PKG, 'bin', 'training-rl.mjs')
    // 默认 dry-run：只报告，不改状态
    const dry = JSON.parse(execFileSync(process.execPath, [cli, 'reconcile', '--root', root, '--json'], { encoding: 'utf8' }))
    t.assert(dry.mode === 'dry-run' && dry.stale.length === 1 && dry.interrupted.length === 0, 'CLI `reconcile` 默认 dry-run：只报告不写状态')
    const stillRunning = (await readRegistry(root, null)).runs.find((r) => r.id === 'rl_ghost2')
    t.assert(stillRunning.status === 'running', 'CLI dry-run 之后状态**未变**（默认只报告）')
    // --apply：真回收
    const applied = JSON.parse(execFileSync(process.execPath, [cli, 'reconcile', '--root', root, '--apply', '--json'], { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }))
    t.assert(applied.mode === 'apply' && applied.interrupted.length === 1 && applied.interrupted[0].id === 'rl_ghost2', 'CLI `--apply` 真回收并回显回收结果')
    t.assert((await readRegistry(root, null)).runs.find((r) => r.id === 'rl_ghost2').status === 'interrupted', 'CLI --apply 之后状态变为 interrupted')
    // 未知子命令要响亮失败（不静默当 health）
    let unknownFailed = false
    try { execFileSync(process.execPath, [cli, 'nope'], { encoding: 'utf8', stdio: ['pipe', 'pipe', 'pipe'] }) } catch { unknownFailed = true }
    t.assert(unknownFailed, '未知子命令 ⇒ 非零退出（不静默降级成 health）')
  }

  /* ⑦ 源码级：train 必须在开跑前对账；health 必须用 dryRun */
  const src = readFileSync(join(PKG, 'src', 'core.mjs'), 'utf8')
  const trainBlock = src.slice(src.indexOf('export async function train('), src.indexOf('async function selectCheckpoint'))
  t.assert(/await reconcile\(\{ root, log \}\)/.test(trainBlock), 'train 在创建新 run **之前**调用 reconcile（真回收）')
  t.assert(trainBlock.indexOf('await reconcile(') < trainBlock.indexOf("nowId('rl')"), '对账发生在 runId 生成之前（顺序可判）')
  const healthBlock = src.slice(src.indexOf('export async function health('))
  t.assert(/reconcile\(\{ root, dryRun: true \}\)/.test(healthBlock), 'health 只用 dryRun（只读承诺有代码依据）')
  t.assert(/owner: \{ host: hostname\(\), pid: process\.pid/.test(trainBlock), 'train 登记归属进程（reconcile 的判据依赖它）')
  const toolsSrc = readFileSync(join(PKG, 'src', 'tools.mjs'), 'utf8')
  t.assert((toolsSrc.match(/name: 'training_rl_/g) ?? []).length === 6, '工具面**恰好 6 个**（交付要求钉死；reconcile 走 CLI + health 可见性，不新增第 7 个工具）')

  rmSync(tmp, { recursive: true, force: true })
}
