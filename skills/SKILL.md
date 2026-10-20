---
name: training-rl
description: RL（GRPO）训练侧插件——规则奖励 + β 可推导（β=σ_r/KL*）+ training-core 单体档位驱动 TRL GRPOTrainer + 产物级断言 + serving 判定（偏好对齐 / 遗忘 / 过拟合预算）
---

# training-rl

> 六个工具（`training_rl_*`）+ 一条只读健康路由 `/__training_rl/health`。
> 依赖长任务 1 冻结的六个 `@ai-infra/*` 共享库（含 training-core **v0.2.0 的单体档位**）。

## 一句话

把「策略 + 参考 + **规则奖励**」三者耦合关系做成可判定的形状：
**β 从实测 σ_r 与调用方给的 KL\* 推出来**、**每步 KL/奖励进审计链**、
**权重变了没有由产物级断言说了算**、**判定一律走 serving**。

## 六个能力

| 工具 | 做什么 | 风险 |
|---|---|---|
| `training_rl_prepare` | 从**既有（已去泄漏）** 数据集取 `train.jsonl`（训练 prompt）+ `heldout.jsonl`（评估 prompt），本层再复核双轨去泄漏，继承上游 license 证据链 | write / 无沙箱 |
| `training_rl_train` | 容器内 TRL 1.14 `GRPOTrainer`（**单体档位**驱动）+ σ_r 实测 + β 推导 + 每步 KL/奖励记账 + checkpoint + 合并 + 产物级断言 | write / **container** |
| `training_rl_validate` | 合并 → serving 部署（策略 + 基座）→ 偏好对齐 / 遗忘 / 过拟合预算三门禁 → 拆解 | write / **container** |
| `training_rl_rollback` | 标记 rolled-back；（可选）删**本包 run 目录** | write / 无沙箱 |
| `training_rl_get_report` | 注册表 + 审计链校验 | read |
| `training_rl_health` | 沙箱层级 / 镜像 / GPU / 基座 / serving / 判据参考实现 / 审计链 | read |

## 三条不可协商的纪律（都在代码里有判据）

### ① 奖励是**规则函数**，不是偏好数据

`Anthropic/hh-rlhf` 卡片正文**限定用途**；`Dahoas/rm-static` **完全未声明 license**；
本地模型根目录下无 reward model ⇒ 偏好对/RM 路径**整体关闭**。
规则奖励带**显式版本号**（`rule-reward@1`）：改规则 = 改实验。

规则实现有**两处**（容器内 Python 逐批打分 / Node 侧评估打分），这是刻意接受的：
与之配套的是一条**机器判据** —— 每次训练时容器对 `tests/fixtures/reward-vectors.json`
的同一组向量打分并回传，与 JS 权威实现**逐条比对**，不一致即 `E_REWARD_IMPL_DRIFT`。
纯 CPU 下也能跑同一份 Python 实现：`python3 python/rule_reward.py --vectors <fixture>`。

### ② β **可推导**，不是魔数

```
β = σ_r / KL*
```
- `σ_r` = **实测**（当前策略上采样 + 规则奖励打分，样本标准差 n−1 分母）；
- `KL*` = **调用方显式给出**的 KL 预算（平均每 token nats）—— **本插件不内置默认值**
  （缺即 `E_KL_BUDGET_REQUIRED`：不许"自己定阈值、自己签收"）；
- 语义：**实测 KL 达到 KL\* 时，KL 惩罚的代价恰好等于一个 σ_r 的奖励**。

`σ_r = 0` ⇒ GRPO 组内优势恒为 0 ⇒ 训练是**空转**（步数/loss/checkpoint 全都"正常"）
⇒ **开跑之前**就 `E_REWARD_NO_VARIANCE` 拒绝。

跑完后：实测 KL 均值与 KL\* 的偏离超容差 ⇒ `recalibrationRequired` + `recommendedBeta`
（**重新记账，不静默调参**）；Node 还会按同一公式重算 β 与容器回传值交叉核对
（不一致 ⇒ `E_BETA_DERIVATION_MISMATCH`）。

### ③ 必须断言**产物**

`fingerprintArtifacts` 三个指纹：
① 基座权重（非空）② 训练后的 LoRA adapter（**null ⇒ 什么都没写** ⇒ `E_ARTIFACT_FINGERPRINT_MISSING`）
③ 合并后的策略权重；然后 `assertArtifactChanged({before:①, after:③})`
—— 逐位相同 ⇒ `E_NO_OP_TRAINING`。
为什么不是"adapter 训练前后"：LoRA 的 adapter 目录训练前是**空的**（hash=null），
`assertArtifactChanged` 对 null 一律不通过 ⇒ "训练前为空"与"什么都没写"无法区分。
LoRA 的 B 矩阵初始化为 0 ⇒ 合并结果恰好等于基座 ⟺ adapter 未被推动，故 ①↔③ 是充分判据。

## 训练档位：**单体**（用户已裁决的选项 B）

TRL 的 `GRPOTrainer.train()` 自带 dataloader/生成/优势/KL/优化器步，**不暴露单步 API**
（TRL 1.x 已无 PPOTrainer）⇒ 用 `training-core` **v0.2.0 新增的单体档位**
（`createMonolithicLoop` / `{monolithic:true, run(ctx)}`）。每一步都握手：

```
容器 TrainerCallback.on_log ──emit(ev=step)──▶ Node ctx.onStep(info)（core 记账 + 限额判定）
                            ◀──ack{stop, save}──
若 save：容器把 LoRA 权重写进 ack 指定的**与 core checkpoint 同名同目录**的 adapter/
若 stop：control.should_training_stop = True（不抛异常，正常收尾）
```
等价 hooks：`onStep` / `onCheckpoint` / `onEval`（**只能作选点信号**，`selectionOnly:true`）/ `onBudgetExceeded`。

**已知边界（如实登记）**：`createMonolithicLoop` 的事后核对用 `state.step >= maxSteps`，
故**恰好用满预算**也会被标 `budgetViolation=true`（逐步档位在步进前判定，不会）。
本包不改共享库、也不隐藏该标记，而是额外给 `budgetAssessment.exceeded`
（"真的超过上限"才算越限）。

## 评估：判定一律走 serving

| 门禁 | 判据 | 来源 |
|---|---|---|
| 偏好对齐 | 留出 prompt 上**配对**奖励提升 `mean(d) > z·SE`（z=1.96）**且**相对提升 ≥ 调用方给的 `minRewardGainPct` | testing-core `dualGate`/`relativeChange` |
| 遗忘 | MMLU 准确率 Δ ≤ z·SE；wikitext 逐文本配对 NLL Δ ≤ z·SE | testing-core `mean`/`stddev` |
| 过拟合预算 O3 | `ratio = max(gap,0)/σ_base ≤ f=1`，`σ_base` 实测 | **import** `inference-training/checks/overfit-criterion.mjs`（唯一参考实现，不重写） |
| 泄漏闸 O1 | 训练/留出 prompt 内容哈希交集 + 行内子串**双轨**必须为空 | data-core + 本包 `leakCheck` |

O3 的度量朝向：参考实现把输入当 loss 类（越低越好），RL 的奖励越高越好
⇒ 喂 `−reward`（**保序变换**，argmin 不变），并在产物里如实标注度量名。

选点（O4）：容器内周期评估给 val 曲线（`selectionOnly`）⇒ val loss argmin
（并列取 payloadHash 字典序小者）；选中早期 checkpoint 时 `validate` 会**重新合并**该 checkpoint。

## 阈值：**必须由调用方给出**

- `training_rl_train` 的 `klBudget`（KL\*）—— 无默认值
- `training_rl_validate` 的 `thresholds.minRewardGainPct` —— 无默认值
（`z` 沿用 eval-contract 已冻结的 1.96，不新引入常量。）

## 命令

```bash
npm test                                  # 单测（零 GPU / 零容器 / 零网络）
npm run test:contract                     # 契约一致性检查（inference-contracts/checks/run-all.mjs）
node tests/gen-reward-vectors.mjs --check # 奖励规格是否与实现一致（CI 用）
python3 python/rule_reward.py --vectors tests/fixtures/reward-vectors.json   # 容器侧规则（纯 CPU）
node bin/training-rl.mjs health --json    # 只读自检（CLI 故意不提供 train/validate 旁路）
```

## 真机主 run 怎么跑（手动，带确认令牌）

```js
import * as api from '<repo>/src/core.mjs'
const p = await api.prepare({ sourceDatasetDir: '~/.ai-infra/state/datasets/yahma-alpaca-cleaned__sft__s20260929__n1000' })
const r = await api.train({ promptsDir: p.promptsDir, klBudget: 0.1, maxSteps: 200, /* 缺 confirm 会返回 E_CONFIRMATION_REQUIRED + confirmToken */ })
const v = await api.validate({ runId: r.runId, thresholds: { minRewardGainPct: 5 } })
```

`train`/`validate`/`rollback` 都要**确认令牌**（`per-class-once`，24h 内同类不再询问）；
回传 `confirmToken` 即可执行。
