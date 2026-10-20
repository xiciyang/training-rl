/**
 * registration.spec —— 注册路径 / 工具契约 / 命名空间 / 无损 JSON / manifest 一致性。
 *
 * 判据来自本机实测教训：**注入返回 host ✓ ≠ 工具注册成功**。
 * mock ctx 的 `register` 必须**像 DSH 一样挑刺**（缺 output{schema,render} 就抛），
 * 然后**真调每一个只读工具**，并对**每个工具**的返回跑无损 JSON 检查 ——
 * 长任务 2 缺陷 3（`health()` 里的一个 `undefined` 键让 CI 红）正是这一条要防的。
 */
import { mkdtempSync, rmSync, readFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { apply, TOOL_NAMES, TOOL_RISK, existingNamespaces, CANDIDATE_PKGS } from '../../src/index.mjs'
import { findLosslessProblems } from '../../src/output.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
const WORKSPACE = '/home/xiciyang/deepseek_harness'
export const NAME = 'registration（注册路径 / 契约 / 命名空间 / 无损 JSON）'

function strictCtx(sink) {
  const routes = []
  return {
    routes,
    // cordis 的 ctx.logger 是**可调用服务**（见 run-all.mjs 的 makeLoggerMock 注释）
    logger: (() => { const noop = () => {}; const mk = () => ({ info: noop, warn: noop, error: noop, debug: noop, success: noop, fail: noop, child: () => mk() }); const fn = function logger() { return mk() }; Object.assign(fn, mk()); return fn })(),
    effect(fn) { return fn() },
    get(n) { return n === 'webServer' ? { route(m, p, h) { routes.push({ m, p, h }); return () => {} } } : undefined },
    on() { return () => {} },
    tools: {
      register(tool) {
        if (!tool.output || !tool.output.schema || typeof tool.output.render !== 'function') {
          throw new Error(`tool "${tool.name}" must declare output { schema, render, presentationMeta? }`)
        }
        if (!tool.parameters || tool.parameters.type !== 'object') throw new Error(`tool "${tool.name}" 缺 parameters`)
        if (typeof tool.description !== 'string' || tool.description.length < 40) throw new Error(`tool "${tool.name}" description 太短`)
        sink.push(tool)
        return () => {}
      },
    },
  }
}

export async function run(t) {
  const tmp = mkdtempSync(join(tmpdir(), 'tr-reg-'))
  const prev = process.env.TRAINING_RL_ROOT
  process.env.TRAINING_RL_ROOT = tmp
  const registered = []
  const ctx = strictCtx(registered)
  let threw = null
  try { apply(ctx) } catch (e) { threw = String(e.message) }
  t.assert(threw === null, `apply() 在严格 mock ctx 下不抛错（实际：${threw}）`)
  t.assert(registered.length === 6, `注册了 6 个工具（实际 ${registered.length}）`)
  t.assert(JSON.stringify(registered.map((x) => x.name)) === JSON.stringify([...TOOL_NAMES]), '注册顺序与 TOOL_NAMES 一致')

  for (const tool of registered) {
    t.assert(tool.name.startsWith('training_rl_'), `${tool.name} 带 training_rl_ 前缀`)
    t.assert(typeof tool.execute === 'function', `${tool.name} 有 execute`)
    t.assert(tool.output && tool.output.schema && typeof tool.output.render === 'function', `${tool.name} 声明 output{schema,render}`)
    // render 必须能真的把返回渲染成 text content（否则工具卡片是空的）
    const rendered = tool.output.render({}, { ok: true })
    t.assert(Array.isArray(rendered) && rendered[0]?.type === 'text' && typeof rendered[0].text === 'string', `${tool.name} 的 render 产出 text content`)
  }

  // ── 真调每一个**只读**工具（真调 ≠ 注册成功），并对**每个工具**的输出跑无损 JSON 检查
  const healthTool = registered.find((x) => x.name === 'training_rl_health')
  const out = await healthTool.execute({})
  t.assert(out && typeof out === 'object', '真调 training_rl_health 返回对象')
  t.assert(out.ok === true || out.ok === false, '返回值是标准信封（ok 字段）')
  t.assert(out.versions?.pkg === '0.1.0', `信封带 versions（实际 ${JSON.stringify(out.versions)}）`)
  t.assert(out.data?.governance === undefined, 'governance 在信封顶层（不在 data 里）')
  t.assert(out.governance?.audit?.ok === true, '返回信封带 governance.audit（只读调用也留痕）')
  t.assert(out.data?.reward?.kind === 'rule' && out.data?.reward?.version === 'rule-reward@1', 'health 回报规则奖励的来源与版本号')
  t.assert(out.data?.trainer?.mode === 'monolithic', 'health 回报训练档位 = monolithic')
  t.assert(out.data?.trainer?.klFormula === 'beta = sigmaR / klBudget', 'health 回报 β 的推导公式（不是数值）')

  const reportTool = registered.find((x) => x.name === 'training_rl_get_report')
  const rep = await reportTool.execute({})
  t.assert(rep.ok === true && Array.isArray(rep.data.runs), '真调 training_rl_get_report 返回 runs 数组')
  t.assert(rep.data.audit && typeof rep.data.audit.ok === 'boolean', 'get_report 附审计链校验结果')
  t.assert(Array.isArray(rep.data.promptSets), 'get_report 列出可用 prompt 集')

  // **无损 JSON**：对两个真调结果逐个路径检查（undefined/非有限数/函数/循环引用都拒）
  for (const [label, val] of [['health', out], ['get_report', rep]]) {
    const r = findLosslessProblems(val)
    t.assert(r.ok, `${label} 的返回值是无损 JSON（问题：${r.problems.slice(0, 4).map((p) => `${p.path}→${p.reason}`).join('；') || '无'}）`)
    // JSON.stringify 之后不得丢键（undefined 会被静默丢掉 —— 这正是"静默不一致"的来源）
    const round = JSON.parse(JSON.stringify(val))
    const countKeys = (o) => (o && typeof o === 'object' ? Object.keys(o).length + Object.values(o).reduce((a, v) => a + countKeys(Array.isArray(v) ? v[0] ?? null : v), 0) : 0)
    t.assert(countKeys(round) === countKeys(val), `${label} 序列化往返后键数不变（无 undefined 被静默丢弃）`)
  }

  // deep=true 时也必须无损（曾经 deep 分支是最容易漏 undefined 的地方）
  const outDeep = await healthTool.execute({ deep: true })
  const rd = findLosslessProblems(outDeep)
  t.assert(rd.ok, `health(deep=true) 也是无损 JSON（问题：${rd.problems.slice(0, 4).map((p) => `${p.path}→${p.reason}`).join('；') || '无'}）`)
  t.assert(outDeep.data?.deep?.paths?.registryFile?.includes('rl-registry.yaml'), 'deep=true 回报完整路径')

  // 健康路由
  t.assert(ctx.routes.some((r) => r.p === '/__training_rl/health'), '注册了 /__training_rl/health 路由')

  // ── **6 个工具全部真调一次**（工具边界），并对每一个的返回跑无损 JSON 检查
  //    参数刻意设计成"能过确认闸门但立刻因输入不合法而失败"，从而不碰容器/GPU：
  //    · prepare  无需确认 → 缺 sourceDatasetDir → E_MISSING_PARAM
  //    · train/validate/rollback 需确认 → 带正确令牌进入 handler → 输入闸门失败
  //    这样 6 条路径都真的走完了"确认闸门 → handler → 信封 → 无损 JSON 闸门 → 审计"。
  const { tokenFor } = await import('../../src/confirm.mjs')
  const probe = [
    { name: 'training_rl_prepare', args: {}, expect: 'E_MISSING_PARAM' },
    { name: 'training_rl_train', args: { promptsDir: join(tmp, 'nope'), klBudget: 0.1 }, scope: 'train', expect: 'E_PROMPTS_INVALID' },
    { name: 'training_rl_validate', args: { runId: 'rl_nope', thresholds: { minRewardGainPct: 5 } }, scope: 'validate', expect: 'E_RUN_NOT_FOUND' },
    { name: 'training_rl_rollback', args: { runId: 'rl_nope' }, scope: 'rollback', expect: 'E_RUN_NOT_FOUND' },
    { name: 'training_rl_get_report', args: {}, expect: 'OK' },
    { name: 'training_rl_health', args: {}, expect: 'OK' },
  ]
  const seen = new Set()
  for (const c of probe) {
    const tool = registered.find((x) => x.name === c.name)
    t.assert(Boolean(tool), `找到工具 ${c.name}`)
    const args = { root: tmp, ...c.args }
    if (c.scope) args.confirm = tokenFor(c.scope, { ...c.args, root: undefined })
    let env = null
    try { env = await tool.execute(args) } catch (e) { env = { thrown: String(e.message) } }
    seen.add(c.name)
    t.assert(env && env.thrown === undefined, `${c.name} 真调未抛异常（实际 ${env?.thrown ?? ''}）`)
    t.assert(typeof env.ok === 'boolean' && typeof env.code === 'string', `${c.name} 返回标准信封（ok/code）`)
    t.assert(env.code === c.expect, `${c.name} → ${c.expect}（实际 ${env.code}）`)
    t.assert(env.versions?.pkg === '0.1.0' && env.versions?.schema === '1', `${c.name} 信封带 versions`)
    t.assert(env.governance?.schema === 'training-rl/governance@1', `${c.name} 信封带 governance（成功或失败都留痕）`)
    const r = findLosslessProblems(env)
    t.assert(r.ok, `${c.name} 的返回是**无损 JSON**（问题：${r.problems.slice(0, 4).map((x) => `${x.path}→${x.reason}`).join('；') || '无'}）`)
    const str = JSON.stringify(env)
    t.assert(str !== undefined && !str.includes('undefined'), `${c.name} 序列化后不含 undefined 字面量`)
    if (env.ok === false) t.assert(typeof env.error?.message === 'string' && env.error.message.length > 0, `${c.name} 的失败带可诊断的 message`)
  }
  t.assert(seen.size === 6, `6 个工具都被真调过（实际 ${seen.size}）`)

  // 命名空间：去兄弟包源码实时抓名，交集必须为空
  const siblings = existingNamespaces(WORKSPACE)
  t.assert(siblings.length >= 30, `抓到兄弟包工具名（${siblings.length} 个）`)
  const overlap = TOOL_NAMES.filter((n) => siblings.includes(n))
  t.assert(overlap.length === 0, `与既有命名空间无交集（实际重叠：${overlap.join(', ') || 'none'}）`)
  t.assert(siblings.every((n) => n.startsWith('inference_') || n.startsWith('training_')), '兄弟包一律 inference_/training_ 前缀')
  t.assert(!CANDIDATE_PKGS.includes('inference-training/training-rl'), '候选目录**不含本包**（含自己会让"交集为空"自相矛盾）')
  t.assert(siblings.includes('training_sft_train') && siblings.includes('training_pretrain_train'), '确实抓到了两个训练兄弟包的工具名')

  // 风险声明与工具面双向一致
  t.assert(Object.keys(TOOL_RISK).length === TOOL_NAMES.length, '风险声明数与工具数一致')
  for (const n of TOOL_NAMES) t.assert(Boolean(TOOL_RISK[n]), `${n} 有风险声明`)
  t.assert(TOOL_RISK.training_rl_train.risk === 'write' && TOOL_RISK.training_rl_train.sandbox === 'container', 'train 是 write + container 沙箱')
  t.assert(TOOL_RISK.training_rl_train.confirmation === 'required', 'train 需要确认令牌')
  t.assert(TOOL_RISK.training_rl_health.risk === 'read', 'health 是 read 档')

  // manifest.yaml 与代码逐条一致（声明漂移是这类机制唯一真正的失败模式）
  const mf = readFileSync(join(PKG, 'manifest.yaml'), 'utf8')
  for (const n of TOOL_NAMES) t.assert(mf.includes(`tool: ${n}`), `manifest.yaml 声明了 ${n}`)
  for (const [n, d] of Object.entries(TOOL_RISK)) {
    const block = mf.split(`tool: ${n}`)[1]?.split('- id:')[0] ?? ''
    t.assert(block.includes(`risk: ${d.risk}`), `manifest ${n} risk 一致`)
    t.assert(block.includes(`sandbox: ${d.sandbox}`), `manifest ${n} sandbox 一致`)
    t.assert(block.includes(`confirmation: ${d.confirmation}`), `manifest ${n} confirmation 一致`)
  }
  t.assert(mf.includes('preference_datasets') && mf.includes('reward_model'), 'manifest 显式声明不做偏好数据/RM（不是"忘了做"）')
  t.assert(mf.includes('reference_model_writes'), 'manifest 显式声明 reference 不被写入（整轮哈希不变）')

  // 包元数据：RL 框架必须**显式**声明为可选依赖组
  const pkg = JSON.parse(readFileSync(join(PKG, 'package.json'), 'utf8'))
  t.assert(pkg.name === '@xiciyang/training-rl', 'package.json name')
  t.assert(pkg.dsh?.bundle?.patch, '声明 dsh.bundle.patch（否则 DSH 不会装载）')
  for (const f of ['torch', 'transformers', 'trl', 'accelerate', 'peft']) {
    t.assert(Boolean(pkg.optionalDependencies?.[f]), `optionalDependencies 显式声明 ${f}`)
  }
  for (const l of ['testing-core', 'data-core', 'state-core', 'audit-core', 'sandbox-core', 'training-core']) {
    t.assert(Boolean(pkg.dependencies?.[`@xiciyang/${l}`]), `声明共享库依赖 @xiciyang/${l}`)
  }
  t.assert(String(pkg.optionalDependencies.trl).includes('<2'), 'trl 钉在 1.x（TRL 1.x 才是已现场核对的 API 面）')

  // cordis.patch.yml 的身份同步面
  const patch = readFileSync(join(PKG, 'cordis.patch.yml'), 'utf8')
  t.assert(/^\s*-\s*id:\s*training-rl\s*$/m.test(patch), 'patch 的 entry id = training-rl')
  t.assert(patch.includes("name: '@xiciyang/training-rl'"), "patch 的 name 是 npm 包名（不是相对路径）")
  t.assert(/inject:\s*\n\s*-\s*tools/.test(patch), 'patch 声明 inject: [tools]')

  if (prev === undefined) delete process.env.TRAINING_RL_ROOT; else process.env.TRAINING_RL_ROOT = prev
  rmSync(tmp, { recursive: true, force: true })
}
