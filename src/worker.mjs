/**
 * worker —— 容器内 Python 训练执行体的**持久 worker 客户端**（training-rl 版）。
 *
 * 与 training-sft / training-pretrain 的 worker 同形（同一条"为什么不是 Python 自己跑完"的论证），
 * **再加一条流式通道** `stream()`：RL 用的是 training-core 的**单体档位**
 * （`{monolithic:true, run(ctx)}` —— 循环归 TRL 的 `GRPOTrainer`），
 * 于是 optimizer 步发生在**容器内**，而限额判定/审计/checkpoint 编排仍在 **Node 侧**。
 * 二者必须逐**步**握手：
 *
 *     容器(GRPOTrainer)                 Node(training-core 单体档位)
 *     TrainerCallback.on_log   ──ev:step──▶   ctx.onStep(info) → 限额判定
 *                              ◀──ack{stop,save}──
 *     继续/停止训练；stop=true 时置 control.should_training_stop
 *     结束时                  ──ev:done──▶   结算 / 产物断言
 *
 * 这条握手是**可执行的**：`ctx.onStep` 的返回值（与 `enforceLimits` 同形）原样回传，
 * 容器据此决定是否继续；Node 侧另有事后核对（单体档位的核心纪律）。
 *
 * 安全：容器以 `--rm`、只读挂载模型与数据、可写挂载输出目录；镜像固定（见 version.mjs）。
 */
import { spawn } from 'node:child_process'
import { createInterface } from 'node:readline'
import { join, dirname } from 'node:path'
import { userInfo } from 'node:os'
import { fileURLToPath } from 'node:url'
import { ToolError, ERROR_CODES } from './version.mjs'

export class ContainerWorker {
  constructor({ image, mounts = [], gpus = true, entry = '/work/rl_trainer.py', python = 'python3', env = {}, log = () => {}, user = null }) {
    this.image = image
    this.mounts = mounts
    this.gpus = gpus
    this.entry = entry
    this.python = python
    // **以宿主 uid:gid 运行容器**：否则容器（root）写出的文件宿主改不动
    // （实测踩到：容器建了 checkpoints/step-000050/adapter/ 后，宿主写 payload.json 得 EACCES）。
    this.user = user === null ? (typeof process.getuid === 'function' ? `${process.getuid()}:${process.getgid()}` : null) : user
    // **非 root uid 必须显式给 USER/LOGNAME**（长任务 3 阶段 3 冒烟闸实测抓到的缺陷）：
    //   该 uid 在容器 /etc/passwd 里没有条目 ⇒ `getpass.getuser()` 抛 KeyError；
    //   而 **torch 的缓存目录解析会走 getpass**（`torch._inductor.runtime.cache_dir_utils`）。
    //   只给 TORCHINDUCTOR_CACHE_DIR 会把报错换成 `Artifact of type=inductor already registered`。
    //   RL 路径必然触发（TRL 的生成/编译），故这里必须有。
    const username = (() => {
      try { return userInfo().username || 'user' } catch { return process.env.USER || process.env.LOGNAME || 'user' }
    })()
    this.env = {
      HOME: '/tmp', USER: username, LOGNAME: username,
      XDG_CACHE_HOME: '/tmp/.cache', HF_HOME: '/tmp/.cache/huggingface',
      TRITON_CACHE_DIR: '/tmp/.triton', TORCHINDUCTOR_CACHE_DIR: '/tmp/.torchinductor',
      ...env,
    }
    this.log = log
    this.child = null
    this.seq = 0
    this.pending = new Map()
    this.stderrTail = []
    // 进行中的**流式**请求（用于检测"流式事件不带 id"这类协议违规；见上面 readline 的处理）
    this.activeStreamId = null
    this.activeStreamCmd = null
  }

  args() {
    const a = ['run', '-i', '--rm']
    if (this.gpus) a.push('--gpus', 'all')
    if (this.user) a.push('--user', this.user)
    for (const m of this.mounts) a.push('-v', m)
    for (const [k, v] of Object.entries(this.env)) a.push('-e', `${k}=${v}`)
    a.push('--entrypoint', this.python, this.image, this.entry)
    return a
  }

  start() {
    if (this.child) return this
    this.child = spawn('docker', this.args(), { stdio: ['pipe', 'pipe', 'pipe'] })
    const rl = createInterface({ input: this.child.stdout })
    rl.on('line', (line) => {
      let msg
      try { msg = JSON.parse(line) } catch { this.log(`[worker] 非协议行（忽略）：${line.slice(0, 160)}`); return }
      const p = this.pending.get(msg.id)
      if (!p) {
        // **协议违规检测**（真机彩排抓到的真缺陷）：流式事件必须带**当前流式请求的 id**。
        // 若不带，事件找不到归属 ⇒ 被丢弃 ⇒ 容器在 `read_ack` 上永久阻塞，
        // 而症状只是"训练卡住不动"。这里把它变成一条**响亮失败**，而不是静默忽略。
        if (msg.ev !== undefined && this.activeStreamId !== null) {
          const err = new Error(
            `流式协议违规：收到 ev=${msg.ev} 但 id=${JSON.stringify(msg.id ?? null)} 不匹配进行中的流式请求（id=${this.activeStreamId}，cmd=${this.activeStreamCmd}）。`
            + '事件会被丢弃、容器将永久阻塞在 read_ack 上 —— 容器侧必须给**每个**流式事件带上请求 id（见 rl_trainer.py 的 emit/CUR_ID）。')
          const ap = this.pending.get(this.activeStreamId)
          this.pending.delete(this.activeStreamId)
          if (ap) { clearTimeout(ap.timer); ap.reject(err) }
          this.activeStreamId = null
          this.log(`[worker] ${err.message}`)
          return
        }
        // 真正的"外来行"（库直接 print 到 stdout 的产物）：忽略，但把内容打出来以便诊断
        this.log(`[worker] 未知 id 的应答（忽略）：${line.slice(0, 200)}`)
        return
      }
      // 流式：同一 id 可能有多条消息；只有 `ev === 'done'` 才结清
      if (typeof p.onEvent === 'function') {
        // `ev:done` 是**流的结清**，不是事件：直接 resolve，**不进 onEvent**
        // （否则 onEvent 会收到一个没有语义的"事件"，而它还可能被误当成需要应答的消息）。
        if (msg.ev === 'done' || msg.done === true) {
          this.pending.delete(msg.id)
          clearTimeout(p.timer)
          p.resolve(msg)
          return
        }
        Promise.resolve()
          .then(() => p.onEvent(msg))
          .then((reply) => {
            // **只有**声明 ack:true 的事件才回 ack —— 否则 Python 的 read_ack 会错位读到上一条
            // （同一个 id，错位无法被 id 校验发现）。见 shouldAckStreamEvent 的说明。
            if (shouldAckStreamEvent(msg) && reply && this.child) this.child.stdin.write(`${JSON.stringify({ id: msg.id, ack: true, ...reply })}\n`)
          })
          .catch((e) => {
            this.pending.delete(msg.id)
            clearTimeout(p.timer)
            p.reject(e)
          })
        return
      }
      this.pending.delete(msg.id)
      p.resolve(msg)
    })
    const rlErr = createInterface({ input: this.child.stderr })
    rlErr.on('line', (line) => {
      this.stderrTail.push(line)
      if (this.stderrTail.length > 60) this.stderrTail.shift()
      const noisy = /^(Loading weights|Writing model shards|Downloading|Fetching)/.test(line)
      if (!noisy && line.trim() !== '') this.log(`[worker:err] ${line.slice(0, 300)}`)
    })
    this.child.on('exit', (code, sig) => {
      for (const [, p] of this.pending) {
        clearTimeout(p.timer)
        p.reject(new Error(`训练 worker 退出（code=${code} sig=${sig}）；stderr 尾部：${this.stderrTail.slice(-6).join(' | ').slice(0, 400)}`))
      }
      this.pending.clear()
      this.child = null
    })
    return this
  }

  call(cmd, payload = {}, { timeoutMs = 30 * 60 * 1000 } = {}) {
    if (!this.child) throw new Error('worker 未启动')
    const id = ++this.seq
    const req = JSON.stringify({ id, cmd, ...payload })
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        reject(new Error(`worker 命令 ${cmd} 超时（${timeoutMs}ms）`))
      }, timeoutMs)
      this.pending.set(id, {
        timer,
        resolve: (m) => {
          clearTimeout(timer)
          if (m.ok === false) {
            const e = new ToolError(ERROR_CODES.E_INTERNAL, { hint: `训练 worker 内部错误（${cmd}）：${m.error}`, details: { cmd, trace: String(m.trace || '').slice(-800) } })
            reject(e)
          } else resolve(m)
        },
        reject: (e) => { clearTimeout(timer); reject(e) },
      })
      this.child.stdin.write(`${req}\n`)
    })
  }

  /**
   * 流式命令：容器按 `ev` 逐步上报，`onEvent(msg) -> reply` 的返回值被原样回传（ack）。
   * 直到 `ev === 'done'` 才结清（resolve 该条消息）。
   *
   * @param {string} cmd
   * @param {object} payload
   * @param {{onEvent:(msg:object)=>object|void, timeoutMs?:number}} opts
   */
  stream(cmd, payload = {}, { onEvent = null, timeoutMs = 2 * 3600 * 1000 } = {}) {
    if (!this.child) throw new Error('worker 未启动')
    if (typeof onEvent !== 'function') throw new Error('stream 需要 onEvent')
    const id = ++this.seq
    const req = JSON.stringify({ id, cmd, ...payload })
    this.activeStreamId = id
    this.activeStreamCmd = cmd
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id)
        this.activeStreamId = null
        reject(new Error(`worker 流式命令 ${cmd} 超时（${timeoutMs}ms，最后 stderr：${this.stderrTail.slice(-3).join(' | ').slice(0, 300)}）`))
      }, timeoutMs)
      this.pending.set(id, {
        timer,
        onEvent,
        resolve: (m) => { this.activeStreamId = null; resolve(m) },
        reject: (e) => { this.activeStreamId = null; reject(e) },
      })
      this.child.stdin.write(`${req}\n`)
    })
  }

  async stop() {
    if (!this.child) return
    try { await this.call('shutdown', {}, { timeoutMs: 20000 }) } catch { /* 已退出 */ }
    // **竞态（长任务 3 阶段 1 实测抓到）**：`shutdown` 应答之后 Python 才退出，
    // 容器 `exit` 事件可能已触发并把 `this.child` 置为 null。
    // 若此处还去读 `this.child.stdin` / `c.on` 就会 TypeError。两处都要防。
    try { this.child?.stdin?.end() } catch { /* ignore */ }
    const c = this.child
    if (!c) return
    await new Promise((r) => {
      if (c.exitCode !== null || c.signalCode !== null) { r(); return }
      const t = setTimeout(() => { try { c.kill('SIGKILL') } catch { /* ignore */ } ; r() }, 8000)
      c.on('exit', () => { clearTimeout(t); r() })
    })
    this.child = null
  }
}

/**
 * 流式事件的**应答策略**（显式，且可单测）。
 *
 * ⚠️ 这里是一个真实的协议陷阱：所有流式事件共用**同一个 id**（= 请求 id），
 * 因此"多写一条 ack"与"少写一条 ack"都会让 Python 侧的 `read_ack` **错位读到上一条**，
 * 而 id 校验**发现不了**（id 相同）。所以：
 *   - **只有**显式声明 `ack: true` 的事件才回 ack（目前只有 step：它需要 core 的 stop/save 决定）；
 *   - `log` / `eval` 是**单向通知**，绝不回 ack；
 *   - Python 侧每个 step 事件恰好读一条 ack ⇒ 两侧步调严格一致。
 */
export function shouldAckStreamEvent(msg) {
  return Boolean(msg && msg.ack === true)
}

/** 训练容器的标准挂载（模型只读 / 数据只读 / 输出可写 / 训练脚本只读）。 */
export function trainingMounts({ modelDir, promptsDir, outDir, pythonDir = null, extra = [] }) {
  // 关键：模型目录必须挂成 `/models/<basename>`（挂成 /models 会让容器内看到的是**目录内容**，
  // transformers 找不到 config.json 就把它当成 HF repo id，报 "Repo id must be in the form ..."。
  // 这是实测踩到的第二个容器路径坑。）
  const base = String(modelDir).split('/').filter(Boolean).pop()
  const m = [`${modelDir}:/models/${base}:ro`, `${promptsDir}:/data:ro`, `${outDir}:/out`]
  // 训练脚本必须**挂进容器**（否则 /work/rl_trainer.py 不存在）
  const py = pythonDir || join(dirname(fileURLToPath(import.meta.url)), '..', 'python')
  m.push(`${py}:/work:ro`)
  return [...m, ...extra]
}
