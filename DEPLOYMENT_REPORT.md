# DEPLOYMENT_REPORT — `training-rl`（长任务 3 · 阶段 3）

> 本文件是**装机与运行手册** + **本次真机运行报告**。所有数字都可在
> `inference-training/out/rl-vertical-loop.json` 与审计链 `~/.ai-infra/logs/audit/training-rl/2026-09-29.jsonl`
> 里复核（`[verified]` 一律指本机实测）。判据与边界见 `inference-contracts/contracts/training-rl-contract.md`
> 与 `ACCEPTANCE_MATRIX.md`。

---

## 1. 交付形态

| 项 | 值 |
|---|---|
| 包名 / 路径 | `training-rl` / `inference-training/training-rl/` |
| 入口（DSH） | `src/index.mjs` → `apply(ctx)`；`export const name = 'training-rl'`，`inject = ['tools']` |
| 入口（MCP） | `mcp/server.mjs`（stdio，行分隔 JSON-RPC 2.0；与 DSH 共用 `src/tools.mjs` 的同一份规格） |
| 入口（CLI） | `bin/training-rl.mjs`（`health` / `report` / `tools` / **`reconcile`**） |
| 装配声明 | `package.json` 的 `dsh.bundle.patch: ./cordis.patch.yml`；patch 里 `id: training-rl`、`name: 'training-rl'`（npm 包名）、`inject: [tools]` |
| 能力清单 | `manifest.yaml`（白名单；`not_provided` 逐条说明"主动不做"） |
| 训练镜像 | `training-rl/smoke-toolchain:0.1.0`（= `training-sft/toolchain:0.1.0` + `trl==1.14.0`，`[verified]` 已存在） |
| 共享库依赖 | 六个 `@ai-infra/*` 经 `file:../shared/libs/<name>` 声明，本包 `node_modules/@ai-infra/*` 为 junction |
| 可选依赖组 | `torch` / `transformers` / `trl (>1,<2)` / `accelerate` / `peft`（`optionalDependencies` 显式声明） |
| 共享状态 | `<root>/state/rl-registry.yaml`（schema: `inference-contracts/state-schema/rl-registry.schema.json`） |
| 共享状态根 | `TRAINING_RL_ROOT`，缺省 `~/.ai-infra` |
| 审计链 | `<root>/logs/audit/training-rl/<date>.jsonl`（`@ai-infra/audit-core` 的哈希链） |
| 产出目录 | `<root>/run/training-rl/<runId>/{adapter,merged,merged-selected,checkpoints/,metrics.jsonl,eval-result.json}` |

---

## 2. 安装 / 挂载

```bash
# ① 依赖（六个共享库走 file: 声明；本机已建好 junction，重装也安全）
cd /home/xiciyang/deepseek_harness/inference-training/training-rl
npm install --no-audit --no-fund        # 只需 file: 依赖，无网络需求

# ② 常驻挂载（与本项目训练侧既有做法一致：运行时注入，不改 profile）
#    dev_inject_plugin  dir=/home/xiciyang/deepseek_harness/inference-training/training-rl
# ⚠️ 本机实测：本会话内注入会失败（见 §6 已知问题 1），需先重启 DSH 让长驻进程重新加载升级后的共享库。

# ③ 只读自检（随时可跑，不写共享状态）
node bin/training-rl.mjs health --json | head -40
```

**装配期 fail-closed 的四道闸门**（`src/index.mjs`）：风险声明双向一致、工具名与既有命名空间无交集、
每个工具声明 `output{schema, render}`、返回值过**无损 JSON** 检查。

---

## 3. 日常操作（六个工具 + 一个维护入口）

| 目的 | 命令 / 工具 | 备注 |
|---|---|---|
| 构造 RL prompt 集 | `training_rl_prepare` | 从**既有已去泄漏**数据集的 `train.jsonl`/`heldout.jsonl` 取；本层再复核双轨去泄漏；继承上游 license 证据链 |
| 训练 | `training_rl_train` | 需 `confirm`（首次会返回 `E_CONFIRMATION_REQUIRED` + `confirmToken`）；**`klBudget` 必填**（无默认值） |
| 判定 | `training_rl_validate` | 需 `thresholds.minRewardGainPct`（无默认值）；经 serving 部署/测/拆 |
| 回退 | `training_rl_rollback` | 只删**本包 run 目录**；需 `confirm` |
| 查报告 | `training_rl_get_report` | 注册表 + 审计链校验 + 幽灵 running 列表 |
| 自检 | `training_rl_health` | 沙箱层级 / 镜像 / GPU / 基座 / serving / 判据参考实现 / 审计链 / **stale-running** |
| 回收幽灵 running | `node bin/training-rl.mjs reconcile [--apply]` | **默认 dry-run 只报告**（与 `inference_serving_reconcile` 同口径）；`--apply` 才写 |

### 3.1 为什么 `reconcile` **没有**做成第 7 个 MCP 工具

交付要求把本包的工具面**钉死为 6 个**（`training_rl_{prepare,train,validate,rollback,get_report,health}`）。
因此"可调用入口"由三处共同承担，且都不破坏该约束：

1. **`training_rl_train` 开跑前自动对账**（真回收，`reconciledBeforeStart` 进审计）；
2. **`training_rl_health` / `training_rl_get_report` 只读探测**并把
   `staleRunning` / `staleRunningCount` / `warnings` 放在**顶层**（不必翻 `registry` 子对象就能看见，
   `warnings` 里直接给出可执行的回收命令）—— 没有新 run 也能"看见"；
3. **CLI `reconcile`**（`--apply` 才写）。

> 若后续允许扩工具面，把 (3) 直接包成 `training_rl_reconcile`（`dryRun` 默认 `true`）是一次改动量极小的升级。

---

## 4. 本次真机运行报告

### 4.1 环境

| 项 | 值 |
|---|---|
| 主机 / GPU | `thinkstationpgx-3269` / NVIDIA GB10，`deviceCapability=[12,1]`（sm_121） |
| 驱动 / 镜像 | 580.126.09 / `training-rl/smoke-toolchain:0.1.0`（torch 2.11.0+cu130、transformers 5.17.0、trl 1.14.0、peft 0.21.0） |
| 基座 | `/data/models/Qwen2.5-0.5B-Instruct`（只读挂载，全程哈希不变） |
| prompt 集 | 400 训练 / 200 留出，来自 `yahma-alpaca-cleaned`（CC BY 4.0）经 data-core 去泄漏的既有切分；本层双轨复核 `hashOverlap=0`、`substringContained=0` |
| GPU 串行纪律 | 先等 `out/lt4-seed-retrospect-raw.json` 出现（23:09）再开跑；两次 main run 之间不并发 |

### 4.2 三次真机运行（含一次彩排）

| 运行 | 目的 | KL\* | β | 步数 | 训练 GPU·h | 实测 KL 均值 | 结果 |
|---|---|---|---|---|---|---|---|
| 彩排 `rl_20260929151528_a474` | 走通协议（2 步） | 0.1 | 0.757257 | 2 | 0.0022 | 0.000640 | ✅ 协议全通（adapter 落盘、合并、产物断言、跨语言一致性） |
| **run1 `rl_20260929151620_wxnz`** | 主 run（声明的 KL\*） | 0.1 | **0.8258130286540805** | 200 | 0.0723 | **0.0011740167962852866** | ✅ 训练通过 / ❌ P1 未通过（仪器无区分度） |
| **run2 `rl_20260929155357_gr86`** | **β 对照实验** | 8.5 | **0.009715447395930358** | 200 | 0.0642 | **0.001172277771984227** | ✅ 训练通过 / ⚠️ P1 判**测量不成立** |

**总 GPU 消耗 ≈0.37 GPU·h**（含彩排、两次 main 与跨语言一致性检查），远低于单次 2 GPU·h 的上限。

### 4.3 关键结论

**(1) β 不是杠杆（对照实验，`[verified]`）**

```
β:  0.8258130286540805  →  KL 均值 0.0011740167962852866
β:  0.009715447395930358 →  KL 均值 0.001172277771984227     （β ↓85.0×，KL 差 1.7e-6）
```

⇒ 在 `KL ≪ KL*` 的区间里，β 的量级不构成约束。**下一个杠杆是奖励设计 / 步数规模 / prompt 形态**，不是继续调 β。

**(2) 训练侧奖励确实被推上去了，但没迁移到留出集**

| | 训练侧 first10→last10 | 留出侧（偏好） |
|---|---|---|
| run1 | 0.8315 → 0.8574（**+3.11%**） | 0.8536 → 0.8511（−0.29%，n.s.） |
| run2 | 0.8351 → 0.8590（**+2.87%**） | 0.7243 → 0.7250（+0.10%，**测量不成立**） |

run2 的 O3/O2 复算（serving 仪器、train/val 两侧）：`improvement_train=+0.012710`、`improvement_val=+0.000703`
vs `z·SE_val=0.011187` ⇒ **O2 未通过：留出集没有被显著推动**。

**(3) SFT 能力回归：通过**（MMLU 0.30→0.32；wikitext 配对 NLL Δ=+1.36e-4 vs 界 9.42e-4）

**(4) P1 判「测量不成立」——这一点必须与"未达标"分开读**

- `@1`（`max_tokens=64`）：**47/48 条生成被截断**，48 条配对里 **27 条并列** ⇒ 判据几乎无动态范围；
- `@2`（`max_tokens=256`）：基座 38/48、策略 35/48 仍被截断 ⇒ 截断率闸门判 `validityOk=false`；
- 处置：**换仪器**（提高生成长度上限 / 要求自然终止 / 改用可判约束类 prompt），
  **不动阈值**（`minRewardGainPct=2%`、`z=1.96` 全程未改）。
- ⇒ **不得**把本次结果读作"RL 没让模型变好"：那是一个更强的断言，现有证据不支持。

### 4.4 产物指纹（产物级断言的原始证据）

| 指纹 | 值 |
|---|---|
| 基座（训练前，只读） | `sha256:9f55e03fcc9ee87ff0397961c53912923f86da186b6c08ed9d890fb2fbfa25a9` |
| run1 合并后策略 | `sha256:ebe8f51984175550d…`（≠ 基座 ⇒ `assertArtifactChanged` 通过） |
| run2 合并后策略 | `sha256:c702542c0e6ca3b2f…`（≠ 基座 ⇒ 通过） |
| run1 / run2 adapter | `sha256:6f875ce4060081818…` / `sha256:4698d11794ba95719…`（非空 ⇒ 权重确实写了） |

---

## 5. 复核流程（照抄即可）

```bash
# 1) 单测（零 GPU / 零容器 / 零网络）：621 断言
cd inference-training/training-rl && npm test

# 2) 容器侧规则奖励 == JS 权威实现（CPU-only，不占 GPU）
node tests/realmachine/reward-conformance.mjs
#   → passed:true  maxAbsDiff:2.22e-16（1 ULP）  containerVersion:rule-reward@1

# 3) 真容器流式协议（CPU-only）：ack 必须对称
node tests/realmachine/worker-protocol.spec.mjs        # → PASS

# 4) 契约一致性（含 C13d frozen 证据路径、C22 training-rl 组）
cd /home/xiciyang/deepseek_harness && node inference-contracts/checks/run-all.mjs
#   → 合计 117/117，未过 0

# 5) 契约 §10 与证据同源
node inference-training/training-rl/tests/realmachine/fill-contract-evidence.mjs --check

# 6) 运行时状态（只读）
node inference-training/training-rl/bin/training-rl.mjs report --json | head -40
node inference-training/training-rl/bin/training-rl.mjs reconcile            # dry-run：只报告幽灵 running
```

真机重跑（**先确认 GPU 空闲**）：

```bash
cd inference-training/training-rl
node tests/realmachine/rl-vertical-loop.mjs --rehearse --only train    # 先 2 步彩排，~0.002 GPU·h
node tests/realmachine/rl-vertical-loop.mjs --steps 200 --kl-budget 8.5 --min-gain-pct 2 --out /tmp/rl.json
node tests/realmachine/assemble-evidence.mjs && node tests/realmachine/fill-contract-evidence.mjs
```

---

## 6. 已知问题与排障

| # | 现象 | 根因（`[verified]`） | 处置 |
|---|---|---|---|
| 1 | `dev_inject_plugin` 失败：`'@ai-infra/training-core' does not provide an export named 'MODES'` | 长驻 DSH host（启动 15:09:42）缓存了 **21:58:18 升级前**的 `training-core` 模块实例；Node 的 ESM 缓存按解析 URL 命中，文件变了也不重载 | **重启 DSH** 即可（但本会话内 GPU 串行任务 `lt4-seed-retrospect` 是该 host 的**直接子进程**，重启会杀死它 ⇒ 延后到 GPU 任务结束）。mock ctx 下的注册/真调已覆盖（156 断言） |
| 2 | `E_DEGRADE_EXHAUSTED`（部署降级链耗尽） | 与本机既有 vLLM 服务（`qwen3-embedding`，4.5GiB）**及并行 workstream 的部署**抢统一内存：实测 `availableBytes` 从 90GiB 掉到 45GiB 时必失败 | ① 端点声明从 `gpuMemoryUtilization=0.18` 降到 **0.12**（`DEPLOY_GPU_MEM_FRACTION`，0.5B 仍有 ~10× 余量）；② runner 对本类**瞬时**失败做 2 次有界重试（判据/阈值不变） |
| 3 | 训练日志刷 `未知 id 的应答 undefined`，训练卡在第 1 步 | 容器 emit 的流式事件**没带请求 id** ⇒ host 按 id 找不到归属 ⇒ 事件被丢弃 ⇒ 容器永久阻塞在 `read_ack` | 已修：`emit` 单点注入当前流式 id；worker 侧把"流式事件不带 id"变成**响亮失败**（协议违规即拒绝该流）。判据在 `tests/unit/container-contract.spec.mjs` + 真容器协议测试 |
| 4 | `trainer.model` 是 `Qwen2ForCausalLM`（不是 PeftModel） | 构造 `GRPOTrainer` 时漏传 `peft_config` ⇒ 变全参微调，且保存/合并路径全错 | 已修：传 `peft_config=peft_config`，并加**开训前**断言 `isinstance(S.model, PeftModel)`（fail-closed） |
| 5 | 某条 run 永远停在 `running` | **外部杀死**（SIGKILL/OOM/父进程被杀）——`catch` 抓不到；审计链里只有 `train.begin`、没有 `failed`/`end` | 已修：`reconcile`（pid 存活 + 目录进度双判据，先写审计再改状态）+ health/get_report 顶层可见 + CLI `reconcile --apply`；`train` 开跑前自动对账 |
| 6 | O3 判定与契约字面不一致 | 冻结点参考实现 `pass = ratio <= theta`（`theta=f·σ_base`）单位不一致；`σ_base` 远离 1 时两种读法结论相反 | **未修改**（改判据口径属 major，须暂停报告）。已登记契约 §9 边界 8；本次总体判定不依赖它（O2 独立失败） |
| 7 | 偏好判据无区分度 | 生成被截断主导（`@1` 47/48；`@2` 38/48、35/48） | 已按项目既有纪律处置：**换仪器（提高生成长度上限）而不是调阈值**；`@2` 新增截断率闸门把它报成"测量不成立" |

---

## 7. 回滚

```bash
# 标记回退（不动磁盘）
node -e "import('./inference-training/training-rl/src/core.mjs').then(m=>m.rollback({runId:'rl_...'}))"
# 连同产出目录一起删（只删本包 run 目录，越界删除会在代码里被拒）
#   → 见 tests/unit/guards.spec.mjs 的 S4（purge 拒绝删除 runs 根以外目录）
```

审计链与注册表**不删**：回退只改 `status` 并追加审计，历史可复算（`get_report` 会校验整链）。
