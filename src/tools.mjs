/**
 * tools —— 6 个工具的规格（**MCP 与 DSH 两条通道共用同一份**，不存在行为分叉）。
 *
 * 命名空间纪律：`training_rl_*`，与既有插件（`inference_serving_*` / `inference_quantization_*` /
 * `inference_tuning_*` / `training_sft_*` / `training_pretrain_*` / `training_qat_*`）**均无交集**
 * —— 由 `tests/unit/registration.spec.mjs` 去兄弟包源码实时抓名并断言交集为空。
 *
 * DSH 工具契约（**踩过的坑**）：`ctx.tools.register` 要求 `output { schema, render }`
 * **同时**给出；缺 render 会让**整条 entry 被拒**（inference-tuning 就这样挂不上）。
 * 故这里每个工具都带 `output`，且 `src/index.mjs` 里用"像 DSH 一样挑刺"的 mock ctx 跑过，
 * 并在工具边界跑**无损 JSON 闸门**（长任务 2 缺陷 3：注册成功 ≠ 输出契约成立）。
 */
import { CONFIRM_PARAM, ROOT_PARAM, passthroughOutput } from './output.mjs'
import * as api from './core.mjs'

export const ALL_TOOL_SPECS = [
  {
    name: 'training_rl_prepare',
    description: 'Build the RL prompt sets from an ALREADY de-leaked dataset split (never synthesizes, never resplits, never re-sanitizes): train.jsonl -> RL training prompts, heldout.jsonl -> preference-evaluation prompts. Re-runs the double-track leak check at the RL layer (whole-line content hash intersection AND substring containment) and carries the upstream license evidence chain forward. Fails closed (E_PROMPT_LEAK) when either track is non-empty. Preference datasets (hh-rlhf / rm-static) are deliberately NOT used: their cards restrict or omit the license.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['sourceDatasetDir'],
      properties: {
        sourceDatasetDir: { type: 'string', description: '既有（已去泄漏）数据集目录：需含 train.jsonl 与 heldout.jsonl（例如 ~/.ai-infra/state/datasets/<name>）。' },
        seed: { type: 'integer', description: '切分/去重种子。默认 20260929。' },
        trainPrompts: { type: 'integer', description: 'RL 训练 prompt 条数上限。默认 400。' },
        heldoutPrompts: { type: 'integer', description: '偏好评估留出 prompt 条数上限。默认 200。' },
        maxPromptChars: { type: 'integer', description: '单条 prompt 最大字符数（超出截断）。默认 1024。' },
        force: { type: 'boolean', description: '已存在同名 prompt 集时是否重建。默认 false（幂等复用）。' },
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.prepare({ ...a }),
  },
  {
    name: 'training_rl_train',
    description: 'Run GRPO (TRL 1.14 GRPOTrainer) through the shared training loop in its MONOLITHIC tier (@xiciyang/training-core createMonolithicLoop; the user pre-authorized option B and the tier exists in contract v0.2.0), inside a container sandbox with hard budget enforcement (maxSteps<=500, maxGpuHours<=2, concurrency 1, params<=0.5B). The KL coefficient beta is DERIVED, never hardcoded: beta = sigmaR / KL*, where sigmaR is the rule reward standard deviation MEASURED on the current policy and KL* is a caller-supplied KL budget (no built-in default; E_KL_BUDGET_REQUIRED otherwise). Refuses to start when sigmaR = 0 (E_REWARD_NO_VARIANCE: GRPO advantages would be identically zero - a silent no-op). Asserts the product changed: fingerprints the base weights and the merged policy and calls assertArtifactChanged, so a run whose weights are byte-identical FAILS.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['promptsDir', 'klBudget'],
      properties: {
        promptsDir: { type: 'string', description: 'training_rl_prepare 产出的 prompt 集目录（需含 train.jsonl、heldout.jsonl 与 allClear=true 的 lineage.json）。' },
        klBudget: { type: 'number', description: '【必填，无默认值】KL*：允许策略偏离参考多远（**平均每 token KL**，nats）。β = σ_r / KL*，其中 σ_r 由本插件在容器内**实测**。语义：当实测 KL 达到 KL* 时，KL 惩罚的代价恰好等于一个 σ_r 的奖励。' },
        klTolerancePct: { type: 'number', description: '实测 KL 与 KL* 的容差（%）。超出则**重新记账**（recommendedBeta）而**不**悄悄调参。默认 50。' },
        baseModel: { type: 'string', description: '基座模型本地目录（同时是 KL 锚点 reference，整轮哈希不变）。默认 <BASE_MODEL>（默认取环境变量 DSH_BASE_MODEL）。' },
        lora: { type: 'object', description: '{r, alpha, dropout, targetModules}。默认 r=8 alpha=16 dropout=0 target=[q,k,v,o]_proj。' },
        lr: { type: 'number', description: '学习率。默认 1e-6。' },
        batchSize: { type: 'integer', description: '每步采样完成数（= 有效批，gradAccum 默认 1）。默认 8。' },
        gradAccum: { type: 'integer', description: '梯度累积步数。默认 1。' },
        numGenerations: { type: 'integer', description: 'GRPO 组内生成数（每步 prompt 数 = batchSize/numGenerations）。默认 4。' },
        maxCompletionLength: { type: 'integer', description: '最大生成长度（token）。默认 64。' },
        maxSteps: { type: 'integer', description: '优化器步数上限（≤500）。默认 200。' },
        seed: { type: 'integer', description: '训练随机种子。默认 20260929。' },
        checkpointEvery: { type: 'integer', description: '每 N 步存一个 checkpoint（含 LoRA 权重）。默认 50。' },
        checkpointKeep: { type: 'integer', description: '最多保留 N 个 checkpoint。默认 4。' },
        evalEvery: { type: 'integer', description: '容器内周期评估间隔（**只作选点信号**，selectionOnly）。0 = 关闭。默认 50。' },
        evalPrompts: { type: 'integer', description: '每次周期评估使用的 prompt 数（train/val 各这么多）。默认 8。' },
        calibPrompts: { type: 'integer', description: 'σ_r 实测使用的 prompt 数。默认 8。' },
        calibGenerations: { type: 'integer', description: '每个 prompt 的采样数（σ_r 实测）。默认 4。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.train({ ...a }),
  },
  {
    name: 'training_rl_validate',
    description: 'Close the vertical loop through the SERVING path (the only judging instrument, training-eval-contract 5.1): deploy the merged policy and the frozen baseline, then measure both with the same protocol and judge with @xiciyang/testing-core. Three gates: (1) preference alignment - paired rule-reward gain on HELD-OUT prompts, requiring both a caller-supplied relative-gain threshold AND paired significance mean(d) > z*SE; (2) catastrophic forgetting - MMLU accuracy delta and paired wikitext NLL delta bounded by z*SE; (3) the overfit budget O3 imported from the frozen reference implementation checks/overfit-criterion.mjs (never re-implemented). Re-checks the O1 leak gate before drawing any conclusion (fail-closed). Deploys only its own leased endpoints and tears them down.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId', 'thresholds'],
      properties: {
        runId: { type: 'string', description: 'training_rl_train 返回的 run id。' },
        thresholds: { type: 'object', description: '【必填，无默认值】{minRewardGainPct: number(必填), zForgetting?: number}。minRewardGainPct = 偏好奖励相对基线的**最小提升百分比**。' },
        nTrain: { type: 'integer', description: 'O3 过拟合预算用的**训练** prompt 条数。默认 48。' },
        nVal: { type: 'integer', description: '偏好对齐判定用的**留出** prompt 条数。默认 48。' },
        nLm: { type: 'integer', description: 'wikitext 评估文本数。默认 100。' },
        nMmlu: { type: 'integer', description: 'MMLU 题目数。默认 100。' },
        // **N2 缺陷的修复点**：本 schema 声明 additionalProperties:false ⇒ 合规客户端会**丢掉**
        // 未声明的参数。prefMaxTokens 此前不在 properties 里 ⇒ 标准客户端只能拿到未标定档（256）
        // ⇒ 截断率闸门必失败 ⇒ 一个假的"测量不成立"。标定值 768 及其由来见
        // src/measurement-identity.mjs#PREF_MAX_TOKENS_CALIBRATION（证据：
        // inference-training/out/rl-instrument-calibration.json）。
        // 省略时的回落**只**到已标定档 768，并在结果里记 source=calibrated-default（可见，不静默）。
        prefMaxTokens: {
          type: 'integer',
          description: '偏好测量的生成长度上限（**唯一仪器参数**）。**已标定值 = 768**'
            + '（本机标定扫描，证据 inference-training/out/rl-instrument-calibration.json）：'
            + '256 两侧截断 35/48=72.9%、38/48=79.2% > 闸门 50% ⇒ 必失败；'
            + '512 两侧**恰为 50.00%（24/48）**，等于闸门阈值本身 ⇒ knife-edge，换种子/子集即翻闸；'
            + '768 两侧均 20/48=41.7% ⇒ 带真实余量。截断率闸门阈值(0.5)未改，改的只是这个仪器参数。'
            + '省略时回落到**已标定档 768**（不是未标定的 256），并在结果/注册表里记下 '
            + 'prefMaxTokensSource=calibrated-default —— 回落可见，绝不静默换仪器。',
        },
        forceMerge: { type: 'boolean', description: '选点选中早期 checkpoint 时强制重新合并。默认 false（复用已有 merged-selected）。' },
        confirm: CONFIRM_PARAM,
        root: ROOT_PARAM,
      },
    },
    output: passthroughOutput(),
    handler: (a) => api.validate({ ...a }),
  },
  {
    name: 'training_rl_rollback',
    description: 'Roll a run back to a non-promoted state. Marks it rolled-back in the shared registry and (with purge=true) deletes ONLY that run directory under the training-rl run root - never a prompt set, never the base model, never another run.',
    parameters: {
      type: 'object', additionalProperties: false, required: ['runId'],
      properties: { runId: { type: 'string' }, purge: { type: 'boolean', description: '是否删除该 run 的产出目录。默认 false。' }, confirm: CONFIRM_PARAM, root: ROOT_PARAM },
    },
    output: passthroughOutput(),
    handler: (a) => api.rollback({ root: a?.root, runId: a?.runId, purge: a?.purge === true }),
  },
  {
    name: 'training_rl_get_report',
    description: 'Read the RL run registry and the audit hash chain. Without runId it lists recent runs (steps, GPU hours, measured sigmaR, derived beta, measured KL mean, preference reward vs baseline, pass) plus the available prompt sets; with runId it returns the full run record (reward descriptor, calibration, beta derivation, KL accounting, fingerprints, artifact assertion, selection rule and gate results) plus the chain verification result.',
    parameters: { type: 'object', additionalProperties: false, required: [], properties: { runId: { type: 'string' }, limit: { type: 'integer' }, root: ROOT_PARAM } },
    output: passthroughOutput(),
    handler: (a) => api.getReport({ root: a?.root, runId: a?.runId ?? null, limit: a?.limit ?? 10 }),
  },
  {
    name: 'training_rl_health',
    description: 'Self-check for the RL side: behavioural sandbox tier probe (process/container/vm), RL toolchain image presence, GPU, base model availability, inference-serving MCP resolution, the overfit-criterion reference implementation, audit chain integrity, the shared registry path, the rule-reward descriptor and the beta derivation formula. Read-only.',
    parameters: { type: 'object', additionalProperties: false, required: [], properties: { deep: { type: 'boolean' }, root: ROOT_PARAM } },
    output: passthroughOutput(),
    handler: (a) => api.health({ root: a?.root, deep: a?.deep === true }),
  },
]

export function findToolSpec(name) {
  return ALL_TOOL_SPECS.find((t) => t.name === name) || null
}
