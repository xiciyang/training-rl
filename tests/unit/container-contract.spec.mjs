/**
 * container-contract.spec —— **容器侧执行体与宿主侧 worker 的契约**（源码级，跨语言）。
 *
 * 这一条 spec 的每项判据都对应一次**真机彩排抓到的真缺陷**（不是想象出来的规范）：
 *
 * | 彩排症状 | 根因 | 这里的判据 |
 * |---|---|---|
 * | 日志里刷 `未知 id 的应答 undefined`，训练**卡在第 1 步不动** | 容器 emit 的流式事件（log/step/eval/done）**没带请求 id** ⇒ 宿主按 id 找不到进行中的请求 ⇒ 事件被丢弃 ⇒ 容器永久阻塞在 `read_ack` | ① `emit` 必须把当前流式 id 补进缺 id 的消息；② worker 必须把"流式事件不带 id"变成**响亮失败**（而不是静默忽略） |
 * | `trainer.model` 是 `Qwen2ForCausalLM`（不是 PeftModel） | 构造 `GRPOTrainer` 时**漏传 `peft_config`** ⇒ 变成全参微调，且保存/合并路径全错 | ③ 必须传 `peft_config=`；④ 必须有 `isinstance(S.model, PeftModel)` 的**开训前**断言 |
 * | stdout 里混进库的 `print()` | transformers 的 `PrinterCallback.on_log` 会 `print(logs)` | ⑤ 启动时把 `sys.stdout` 指到 stderr（协议通道单独留出来）；⑥ 去掉 `PrinterCallback` |
 *
 * 这些都是**跨语言契约**：单靠 JS 侧单测或单靠 Python 侧跑都发现不了，
 * 必须在源码层把两侧的对齐关系钉住（真机 run 再验行为）。
 */
import { readFileSync, existsSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const PKG = join(HERE, '..', '..')
export const NAME = 'container-contract（容器执行体 ⇄ 宿主 worker 的跨语言契约）'

export async function run(t) {
  const py = readFileSync(join(PKG, 'python', 'rl_trainer.py'), 'utf8')
  const rule = readFileSync(join(PKG, 'python', 'rule_reward.py'), 'utf8')
  const wSrc = readFileSync(join(PKG, 'src', 'worker.mjs'), 'utf8')
  const trainSrc = readFileSync(join(PKG, 'src', 'train.mjs'), 'utf8')

  /* ① 流式事件必须带 id（emit 单点注入） */
  t.assert(/if "id" not in obj and CUR_ID is not None:/.test(py), '① emit 会把当前流式 id 补进缺 id 的消息')
  t.assert(/CUR_ID = rid/.test(py) && /global CUR_ID/.test(py), '① cmd_run 设置 CUR_ID（流式请求 id）')
  t.assert(/PROTO_OUT = sys\.stdout/.test(py), '① 协议通道来自**原始** stdout（在重定向之前保存）')

  /* ② worker 必须把"流式事件不带 id"变成响亮失败 */
  t.assert(/this\.activeStreamId = id/.test(wSrc), '② stream() 记录进行中的流式请求 id')
  t.assert(/流式协议违规：收到 ev=/.test(wSrc), '② worker 检测到"流式事件不带 id"即报协议违规')
  t.assert(/ap\.reject\(err\)/.test(wSrc) && /this\.activeStreamId = null/.test(wSrc), '② 协议违规则**拒绝该流**（不是静默忽略后挂起）')
  t.assert(/未知 id 的应答（忽略）/.test(wSrc), '② 真正的外来行仍然忽略，但把内容打出来便于诊断')

  /* ③④ LoRA 口径：必须传 peft_config，且开训前断言策略是 PeftModel */
  t.assert(/peft_config=peft_config/.test(py), '③ 构造 GRPOTrainer 时**传了** peft_config（漏了会变成全参微调）')
  t.assert(/isinstance\(S\.model, PeftModel\)/.test(py) && /E_NOT_A_PEFT_POLICY/.test(py), '④ 开训前断言策略是 PeftModel（fail-closed）')
  t.assert(/merge_and_unload/.test(py), '④ 断言策略具备 merge_and_unload（合并阶段依赖它）')
  // 自己先 get_peft_model 再传给 trainer 会被 TRL 明确拒绝（已现场核对源码）
  const loadBlock = py.slice(py.indexOf('def _load_policy'), py.indexOf('def _calibrate'))
  t.assert(!/get_peft_model\(/.test(loadBlock), '③ _load_policy **不**自己 get_peft_model（否则与 peft_config 冲突，TRL 直接 raise）')
  t.assert(/由 TRL 自己包 LoRA|TRL 用 peft_config 包装/.test(py) === true || /peft_config/.test(loadBlock), '③ 注释里写明了"由 TRL 包装"的口径（可追溯）')

  /* ⑤⑥ stdout 卫生 */
  t.assert(/sys\.stdout = sys\.stderr/.test(py), '⑤ 启动时把 sys.stdout 指到 stderr（库的 print 不再污染协议通道）')
  t.assert(/PrinterCallback/.test(py) && /remove_callback\(PrinterCallback\)/.test(py), '⑥ 去掉 transformers 的 PrinterCallback（它会 print(logs)）')
  t.assert(/import sys/.test(py), '⑤ sys 已导入')

  /* 训练配置口径 */
  t.assert(/save_strategy="no"/.test(py), 'checkpoint 由 host（training-core）编排 ⇒ TRL 自己不许存')
  t.assert(/logging_steps=1/.test(py), '每步都要记账（logging_steps=1，否则 KL/奖励会缺步）')
  t.assert(/use_vllm=False/.test(py), '本 run 不启用 vLLM（GB10 共驻未验证，登记在契约 §9 边界 5）')
  t.assert(/beta=float\(beta\)/.test(py) && !/beta=0\.0?\d/.test(py), 'β 只能来自推导（源码里不得出现 β 的数值字面量）')
  t.assert(/beta = sigma_r \/ kl_budget/.test(py), '容器侧 β 的推导式与 Node 侧一致（node 会重算交叉核对）')
  t.assert(/E_REWARD_NO_VARIANCE/.test(py), 'σ_r ≤ 0 ⇒ 容器侧也 fail-closed（双重防线）')
  t.assert(/max_steps=int\(cfg\.get\("maxSteps"/.test(py), '容器侧步数上限来自 host 的配置（不是自定义常量）')

  /* 奖励：函数名即日志键；版本号两侧一致 */
  t.assert(/def rule_reward\(completions, prompts=None/.test(py), 'reward_funcs 入口签名兼容 TRL 的调用约定（prompts=/completions= 关键字）')
  t.assert(/from rule_reward import/.test(py) && /S\.reward_calls \+= 1/.test(py), '奖励实现在独立文件里被 import，且真的被计数（P4 类判据可复核）')
  t.assert(/RULE_REWARD_VERSION = "rule-reward@1"/.test(rule), '容器侧奖励版本号与 JS 权威实现一致（rule-reward@1）')
  t.assert(!/hh-rlhf|rm-static/.test(rule.replace(/"""/g, '')), '容器侧没有引入偏好数据集（license 决定）')

  /* 产物路径口径：容器 /out 与宿主 runDir 的映射必须一致 */
  t.assert(/os\.path\.join\("\/out", d\)/.test(py), 'ack 里的相对目录按 /out 前缀解析（与宿主 runDir 挂载一致）')
  t.assert(/trainingMounts\(\{ modelDir, promptsDir, outDir: runDir/.test(trainSrc), '宿主侧挂载把 runDir 映射到 /out')
  t.assert(/promptsDir\}:\/data:ro/.test(readFileSync(join(PKG, 'src', 'worker.mjs'), 'utf8')), 'prompt 集挂到 /data（容器读 /data/train.jsonl）')
  t.assert(existsSync(join(PKG, 'tests', 'realmachine', 'worker-protocol.spec.mjs')), '真容器协议测试存在（真机验证，不依赖 GPU）')
  t.assert(existsSync(join(PKG, 'tests', 'realmachine', 'reward-conformance.mjs')), '容器侧奖励跨语言一致性脚本存在（CPU-only）')
}
