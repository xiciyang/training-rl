# @xiciyang/training-rl

强化学习微调（RL）：在**真实 GPU** 上用可替换的奖励适配器训练，并对「收益是真的」「KL 没失控」「同一套判据只有一份实现」分别取证。

DSH（DeepSeek Harness）插件包，同时可作为普通 Node 库使用。

## 它解决的问题

RL 阶段最容易自欺的三件事：

1. **奖励涨了但没意义**——奖励是实现里的常数、或者前后对比没有显著性。本包把奖励规则固定成可复算的向量，验收时先判差异是否成立。
2. **KL 漂移**——策略跑飞而 loss 好看。本包按预算记账，越界的 run 直接判失败。
3. **判据多份实现**——训练侧、验收侧各写一份门禁逻辑，改一处忘另一处。本包的判据来自共享库（`@xiciyang/testing-core`），不在插件内重写。

## 安装

```bash
dsh plugin add github:xiciyang/training-rl
```

要求：

- Node.js >= 20，`engines.dsh >= 0.2.0 < 0.3.0`
- 真实训练需要 NVIDIA GPU + 容器运行时 + 训练镜像；`training_rl_health` 会逐项报告
- 奖励适配器是可插拔的：内置一份契约（`contract.json`），也可以指向自己的实现

## 工具面（DSH）

| 工具 | 作用 |
|---|---|
| `training_rl_prepare` | 数据/提示集准备，奖励向量与契约落盘 |
| `training_rl_train` | 训练：预算闸门、KL 记账、权重指纹、checkpoint 保留、审计链 |
| `training_rl_validate` | 验收：收益显著性 + KL 在预算内，先判测量有效性再判数值 |
| `training_rl_rollback` | 回退到指定 run；purge 只允许删本插件 run 目录 |
| `training_rl_get_report` | 取人类可读报告 |
| `training_rl_health` | 环境自检：GPU / 容器 / 镜像 / 基座 / 奖励适配器 |

## 配置（不把本机布局写死）

| 环境变量 | 用途 |
|---|---|
| `DSH_BASE_MODEL` | 基座模型本地目录（同时是 KL 锚点的 reference） |
| `DSH_MMLU_PATH` | 评测数据路径（可选门禁） |
| `TRAINING_RL_ROOT` | 运行目录根（默认 `~/.ai-infra`） |

奖励模型/偏好数据缺失时，工具会说明缺哪一层并拒绝开跑，而不是用「没有奖励模型」跑出一轮看似成功的训练。

## 测试

```bash
node tests/run.mjs
```

纯逻辑与契约测试应全绿；依赖本机历史产物的判据自检在产物缺失时显式 `SKIP` 并写明原因。

## 安全边界

- 训练需显式确认令牌；预算超限直接拒绝
- 回退只作用于本插件自己的 run 目录
- 审计链可第三方复算

## 许可

BSD-3-Clause。见 `LICENSE`。
