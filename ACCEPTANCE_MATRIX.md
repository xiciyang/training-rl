# ACCEPTANCE_MATRIX — `training-rl`（长任务 3 · 阶段 3）

> **图例**：`[verified=本机实测]` 表示有本机命令/产物可复核（附路径）；`[claimed]` 表示二手口径；
> `[inferred]` 表示推论。**没有实测证据的项目一律不写"通过"。**
>
> 真机证据：`inference-training/out/rl-vertical-loop.json`（两次 main run + 一次 2 步彩排）、
> `inference-training/out/rl-vertical-loop.maxtok64.json`（run1 的 `@1` 仪器判定，保留原样）、
> `inference-training/out/rl-vertical-loop.v2.json`（一次**基础设施失败**，如实登记）、
> `inference-training/out/rl-rehearsal.json`、`inference-training/out/rl-reward-conformance.json`。
> 契约：`inference-contracts/contracts/training-rl-contract.md`（frozen 1.0.1，§10 与证据**同源**，由脚本生成）。

---

## 0. 一句话结论

**工程交付通过；P1（偏好对齐）判为「测量不成立」——不是「未达标」。**
训练本身走通了（两次 200 步 GRPO，产物级断言、遗忘门禁、审计链全过），
但**判偏好对齐的仪器没有区分度**（生成被截断主导），所以**不能**得出"RL 没让模型变好"这个结论 ——
那是另一个更强的断言，现有证据不支持。

---

## 1. P0 验收项（必须真机）

| # | 验收项 | 判据 | 结果 | 证据 |
|---|---|---|---|---|
| P0-1 | 真机 GRPO 训练走**本包自己的** `training_rl_train` 路径 | 在 `<BASE_MODEL>` 上跑通，≤500 步、≤2 GPU·h | ✅ `[verified]` **200 步 / 0.0642 GPU·h**（run2 `rl_20260929155357_gr86`）+ **200 步 / 0.0723 GPU·h**（run1 `rl_20260929151620_wxnz`）+ 2 步彩排 0.0022 | `out/rl-vertical-loop.json#train`、`out/rl-rehearsal.json`；审计链 `~/.ai-infra/logs/audit/training-rl/2026-09-29.jsonl` |
| P0-2 | 端到端 `validate`（serving 部署 → 评估 → 拆解） | 策略 + 基座各租一个临时端点，测完拆解 | ✅ `[verified]` run2 完整走完（`teardown rl: ok=true` / `teardown base: ok=true`）；端点 `http://127.0.0.1:8100` | `out/rl-vertical-loop.v3.json#validate` |
| P0-3 | 契约冻结，且 **§5.3 的裁决结果（选项 B）**落文档 | 头 `status: frozen`；单体档位/hooks/产物断言/β 口径/已知边界齐备 | ✅ `[verified]` `version: 1.0.1`、`status: frozen`、`frozenAt: 2026-09-29` | `inference-contracts/contracts/training-rl-contract.md` |
| P0-4 | `run-all.mjs` 0 失败 | 全部检查通过 | ✅ `[verified]` **合计 117/117，未过 0** | 命令与输出见 §5 |
| P0-5 | 6 个工具且每个声明 `output{schema,render}`、输出**无损 JSON** | mock ctx 真注册 + **6 个全部真调一次** + 逐个无损检查 | ✅ `[verified]` 156 项断言（含 6 工具逐个执行、`undefined`/非有限数/循环引用全拒） | `training-rl/tests/unit/registration.spec.mjs` |
| P0-6 | 产物级断言（"只断言过程抓不到空操作"） | `fingerprintArtifacts` 基座 vs 合并后策略 + `assertArtifactChanged` | ✅ `[verified]` 两次 run 都通过：基座 `sha256:9f55e03f…` ≠ run1 `sha256:ebe8f519…` / run2 `sha256:c702542c…`；adapter 指纹非空（`6f875ce4…` / `4698d117…`） | `out/rl-vertical-loop.json#train.artifactAssertion` |
| P0-7 | β **可推导**、σ_r **实测**、容器 β 与 Node 重算交叉核对 | 缺 KL\* ⇒ `E_KL_BUDGET_REQUIRED`；σ_r≤0 ⇒ `E_REWARD_NO_VARIANCE`；不符 ⇒ `E_BETA_DERIVATION_MISMATCH` | ✅ `[verified]` run2：σ_r=0.08258130286540805、KL\*=8.5 ⇒ β=0.009715447395930358；Node 侧按同一公式重算一致 | `out/rl-vertical-loop.v3.json#train.calibration`；`src/kl.mjs` |
| P0-8 | 规则奖励跨语言一致（容器 vs JS 权威） | 27 条规格向量逐条比对 ≤1e-12 | ✅ `[verified]` **maxAbsDiff = 2.22e-16**（1 ULP，在 `length-short`），版本号一致 | `out/rl-reward-conformance.json`（容器内跑，`CUDA_VISIBLE_DEVICES=` 空、不占 GPU） |
| P0-9 | 单体档位接线（不是"偷偷另起一套循环"） | `createMonolithicLoop` + `{monolithic:true, run}` + 逐 step 握手 + 等价 hooks | ✅ `[verified]` 真机日志显示逐 step `emit(ev:step, ack:true)` ↔ `ctx.onStep` ↔ `ack{stop,save}`；预算命中时容器收到 stop 并停止；`state.mode=monolithic` | `out/rl-vertical-loop.json`；`tests/unit/monolithic.spec.mjs`（49 断言） |
| P0-10 | 幽灵 `running` 的对账与回收（真机缺陷的正面防线） | 外部杀死留下的 `running` 能被检测并回收，且**可见** | ✅ `[verified]` `rl_20260929151030_yota` 已 `running → interrupted`（含证据与审计 seq 224）；`health()` 顶层报 `staleRunningCount`；CLI `reconcile` 默认 dry-run | `src/core.mjs#reconcile`；`tests/unit/reconcile.spec.mjs`（39 断言） |

---

## 2. P1 验收项

| # | 验收项 | 判据 | 结果 | 证据 / 说明 |
|---|---|---|---|---|
| P1-1 | **偏好对齐提升** | 留出 prompt 上配对提升 `mean(d) > z·SE` 且相对提升 ≥ 调用方给的 2% | ❌ **测量不成立**（不是"未达标"） | run1（`@1` 仪器，64 token）：配对 Δ=−0.002459 vs z·SE=0.008494，winRate=0.1458，48 条里 **27 条并列**、**47/48 条生成被截断**。run2（`@2`，256 token）：基座 38/48、策略 35/48 仍被截断 ⇒ `validityOk=false`，判据**无区分度**。**处置：换仪器（提高生成长度上限），不调阈值** —— 阈值（2%）与 z（1.96）一个字都没动 |
| P1-2 | SFT 能力回归（遗忘门禁） | MMLU 准确率 Δ ≤ z·SE；wikitext 配对 NLL Δ ≤ z·SE | ✅ `[verified]` **通过**：MMLU 0.30 → 0.32（Δ=−0.02）；wikitext NLL 3.296031 → 3.295896（Δ=+0.000136 vs 界 0.000942） | `out/rl-vertical-loop.v3.json#validate.forgettingGate` |
| P1-3 | 过拟合预算 O3 | `ratio = max(gap,0)/σ_base ≤ f=1`（import 冻结参考实现） | ⚠️ **判据本身有单位不一致（见 §4 未验证/缺陷）** | run2：gap=0.012007、σ_base=0.052122、**ratio=0.230** ⇒ 按契约字面（`ratio ≤ f=1`）**PASS**，按参考实现的 `ratio ≤ f·σ_base` **FAIL**。**本次总体判定不依赖它**：O2 独立失败 |
| P1-4 | 泛化真实性 O2 | `improvement_val > z·SE_val` | ❌ **未通过** `[verified]` | run2：improvement_val=+0.000703 < z·SE=0.011187 ⇒ **留出集没有被显著推动**（train 侧 +0.012710） |
| P1-5 | 单测（无 GPU / 无容器 / 无网络） | `npm test` 全过 | ✅ `[verified]` **6 spec / 621 断言** | 见 §5 |
| P1-6 | RL 训练子空间之外的硬约束 | ≤500 步、≤2 GPU·h、并发 1、≤0.5B、基座只读 | ✅ `[verified]` 200 步；`budgetAssessment.exceeded=false`；LoRA 可训练参数 1,081,344/495,114,112（0.2184%）；基座目录只读挂载 | `out/rl-vertical-loop.v3.json#train.budgetAssessment` |
| P1-7 | 审计链可复算（每步 KL/奖励记账） | 每步进链；整链可验证 | ✅ `[verified]` 224 条记录、链 `ok=true` | `src/core.mjs#train` 的 `onStep` 钩子；`tests/unit/guards.spec.mjs` 的篡改检出 |

---

## 3. 真机数字（可直接复核）

| 量 | run1 `rl_20260929151620_wxnz` | run2 `rl_20260929155357_gr86` |
|---|---|---|
| KL 预算 KL\*（调用方给，**无默认值**） | 0.1 | 8.5 |
| 实测 σ_r（当前策略上采样 32 条） | 0.08258130286540805 | 0.08258130286540805 |
| 推导出的 **β = σ_r/KL\*** | **0.8258130286540805** | **0.009715447395930358** |
| 实测 KL 均值 | **0.0011740167962852866** | **0.001172277771984227** |
| 步数 / 最终 loss | 200 / 9.7134e-4 | 200 / 1.0875e-5 |
| 训练 GPU 小时（实测墙钟） | 0.0723 | 0.0642 |
| 训练侧规则奖励 first10→last10 | 0.8315 → 0.8574（**+0.0258 / +3.11%**） | 0.8351 → 0.8590（**+0.0240 / +2.87%**） |
| 基座权重指纹（只读） | `sha256:9f55e03fcc9ee87ff0397961c53912923f86da186b6c08ed9d890fb2fbfa25a9` | 同左（同一份只读基座） |
| 合并后策略指纹 | `sha256:ebe8f51984175550d…` | `sha256:c702542c0e6ca3b2f…` |
| adapter 指纹 | `sha256:6f875ce4060081818…` | `sha256:4698d11794ba95719…` |
| 遗忘门禁 | — | **PASS**（MMLU 0.30→0.32；wikitext NLL Δ=+1.36e-4） |
| 偏好对齐 | FAIL（仪器无区分度） | **测量不成立**（截断主导） |
| 选点（O4） | step-150（val argmin） | step-150（val loss −0.8249，与 step-100/200 并列，取 payloadHash 字典序小者） |
| 总 GPU 小时（含彩排与跨语言一致性检查） | — | **≈0.37 GPU·h**（run1 0.1595 + run2 0.2071 + 彩排 0.0022） |

### 3.1 对照实验：**β 不是杠杆**（这是本阶段最有价值的一条实测结论）

| β | 实测 KL 均值 | 结论 |
|---|---|---|
| 0.825813 | 0.0011740167962852866 | — |
| 0.009715447395930358（**β 降 85.0×**） | 0.001172277771984227 | **KL 几乎不动**（差 1.7e-6） |

⇒ 在 `KL ≪ KL*` 的区间里，**β 的量级不是约束**：把 β 再调小只会允许更多漂移，不会把留出侧的提升变显著。
**下一个杠杆应是奖励设计 / 步数规模 / prompt 形态**（例如可判约束类 prompt），不是继续迭代 β。

---

## 4. **未验证 / 未通过 / 已知缺陷**（如实登记）

| # | 项 | 状态 | 说明 |
|---|---|---|---|
| 1 | **P1-1 偏好对齐** | ❌ **测量不成立** | 生成截断主导 ⇒ 判据无区分度。**不得**读作"RL 没让模型变好"；现有证据不支持那个更强的断言。补救：提高生成长度上限 / 要求自然终止 / 换可判约束类 prompt（**不是**调阈值） |
| 2 | **O3 参考实现的单位不一致**（新发现，**未修改**） | ⚠️ 缺陷 | `inference-training/checks/overfit-criterion.mjs` 返回 `pass = (ratio <= theta)`，而 `theta = f·σ_base` —— 无量纲的 ratio 与以 σ_base 为量纲的界相比；契约 §4.3 写的是 `ratio ≤ f`。`σ_base=0.0521`（≪1）时两种读法结论相反。冻结证据里 `σ_base≈0.6` ⇒ 该缺陷在 LT4 未被暴露。改它属 **major（须暂停报告）**，故只登记不改 |
| 3 | O5 多种子流程级结论 | ❌ 不做 | 本阶段是**单种子**（seed=20260929）；结论只能指"这一个个别模型"，不得表述为"这套流程"。`aggregateSeeds` 在 n<3 时返回 `E_SEEDS_LT_3` |
| 4 | O4 的**早停** | ❌ 未实现 | 只做了"选点"（val loss argmin）；契约 §4.4 的显著性早停条件未落地 |
| 5 | `use_vllm=True`（rollout 共驻） | ❌ 未验证 | 两次 run 都是 `use_vllm=False`；GB10 上 vLLM colocate 的内存行为仍是 `UNVERIFIED_PATHS`（`TRAINING_RL_PLAN.md` §5 V1） |
| 6 | 奖励只有 `rule` 一种 kind | ⚠️ | 契约里"奖励为模型"的分支**未被任何实现验证**（偏好数据因 license 关闭） |
| 7 | 长跑热冻结 / 利用率 | ❌ 未观测 | 单次 ≤200 步（≤0.1 GPU·h），未覆盖长时间运行的退化形态 |
| 8 | DSH 内**运行时装载** | ❌ 未验证（有明确原因） | `dev_inject_plugin` 实测失败：`The requested module '@ai-infra/training-core' does not provide an export named 'MODES'`。根因 **`[verified]`**：`training-core/src/index.mjs` 的 mtime 是 **21:58:18**，而 DSH host 进程启动于 **15:09:42** ⇒ 长驻进程里缓存的是**升级前**的模块实例（模块缓存按解析 URL 命中，文件改了也不重载）。**修复需重启 DSH**，而 GPU 串行任务（`lt4-seed-retrospect`，pid 1771750）是该 host 的**直接子进程** ⇒ 重启会杀死它，故未重启。mock ctx 下的 6 工具真调已验证（156 断言），但**"能挂进真实 DSH"未验证** |
| 9 | 偏好仪器在 256 token 下仍被截断 | ⚠️ 已量化 | 基座 38/48、策略 35/48（`@1` 的 64 token 下是 47/48）⇒ 仪器即使放到 256 也不够 |
| 10 | RL prompt 只有一种形态 | ⚠️ | 全部来自既有 SFT 数据集的 instruction（去泄漏复核通过）；未验证可判约束类 prompt（如 `one word`/JSON）上的行为 |

---

## 5. 可复核命令与决定性输出

```bash
cd inference-training/training-rl

# ① 单测（无 GPU / 无容器 / 无网络）
npm test
# → ALL SPECS PASSED: 6 passed / 0 failed；断言合计 621

# ② 容器侧规则奖励 vs JS 权威实现（CPU-only，20s；CUDA_VISIBLE_DEVICES= 空 ⇒ 不占 GPU）
node tests/realmachine/reward-conformance.mjs
# → passed: true, n: 27, maxAbsDiff: 2.220446049250313e-16, bitIdentical: false（1 ULP）

# ③ 真容器流式协议（node:20-alpine，CPU-only）：ack 策略必须对称
node tests/realmachine/worker-protocol.spec.mjs
# → PASS: worker-protocol（真容器 node:20-alpine，CPU-only）

# ④ 契约一致性（含 C13d 对 frozen 契约证据路径的存在性校验、C22 training-rl 组）
cd <workspace>
node inference-contracts/checks/run-all.mjs
# → 合计 117/117，未过 0

# ⑤ 契约 §10 与证据同源校验
node inference-training/training-rl/tests/realmachine/fill-contract-evidence.mjs --check
# → 契约 §10 与证据同源
```

真机（需要 GPU，且**必须先确认 GPU 空闲**：等 `out/lt4-seed-retrospect-raw.json` 出现）：

```bash
cd inference-training/training-rl
node tests/realmachine/rl-vertical-loop.mjs --rehearse --only train        # 2 步彩排，~0.002 GPU·h
node tests/realmachine/rl-vertical-loop.mjs --steps 200 --kl-budget 0.1  --min-gain-pct 2 --out .../rl-vertical-loop.maxtok64.json
node tests/realmachine/rl-vertical-loop.mjs --steps 200 --kl-budget 8.5  --min-gain-pct 2 --out .../rl-vertical-loop.v3.json
node tests/realmachine/assemble-evidence.mjs                              # 合成最终证据
node tests/realmachine/fill-contract-evidence.mjs                         # 把证据填进契约 §10
```

> `thresholds.minRewardGainPct`（=2%）与 `klBudget` 都必须由调用方给出，**本插件不内置默认值**。
> 2% 的取法有理由、且**在开跑前定下**：σ_r≈0.083、基线均值≈0.854 ⇒ 2% ≈ 0.017 ≈ 0.21σ_r
> （要求提升与奖励自身离散度同量级），不是"挑一个刚好能过的数"。

---

## 6. 交付物清单

| 路径 | 内容 |
|---|---|
| `inference-training/training-rl/` | 插件包：6 工具 + 单体档位驱动 + 容器内执行体 + 单测 |
| `inference-training/training-rl/ACCEPTANCE_MATRIX.md` | 本文件 |
| `inference-training/training-rl/DEPLOYMENT_REPORT.md` | 装机/运行/复核手册 |
| `inference-contracts/contracts/training-rl-contract.md` | 契约 **1.0.1 frozen**（§10 由证据脚本生成） |
| `inference-contracts/state-schema/rl-registry.schema.json` | `rl-registry` 状态方言 schema |
| `inference-training/out/rl-vertical-loop.json` 等 4 份 | 真机证据（main×2 / `@1` 判定 / 基础设施失败 / 彩排 / 跨语言一致性） |
