#!/usr/bin/env python3
"""
rl_trainer.py —— training-rl 的**计算侧**执行体（容器内持久 worker）。

## 与 training-sft / training-pretrain 的分工
  - **Node 侧（@ai-infra/training-core 单体档位）**：限额判定、墙钟/GPU 小时记账、
    checkpoint 编排、审计链、**产物级断言**
  - **Python 侧（本文件）**：只做张量数学（加载/LoRA/生成/反向/优化器步/存权重/合并）

## 为什么这里是"单体"而不是"逐步"（契约 v0.2.0 的单体档位）
TRL 1.14 的 `GRPOTrainer.train()` 自带 dataloader / 生成 / 优势估计 / KL / 优化器步 / checkpoint，
**不暴露单步 API**（TRL 1.x 甚至没有 PPOTrainer）。用户已裁决走 training-core 的**单体档位**，
于是 optimizer 步留在本文件里，而**每一步都要与 Node 握手一次**：

    TrainerCallback.on_log ── emit(ev=step) ──▶ Node: ctx.onStep(info)（core 记账 + 限额判定）
                            ◀── ack{stop, save} ──
    若 save：把 LoRA 权重写进 ack 指定的目录（与 Node 侧 checkpoint 目录同名同目录）
    若 stop：control.should_training_stop = True（**不抛异常**，让 Trainer 正常收尾）

## 判据口径（与 training-rl-contract 一致）
  - **KL 系数 β 可推导**：β = σ_r / KL*；σ_r 由**本文件实测**（当前策略上采样 + 规则奖励），
    KL* 由调用方**显式给定**。本文件**绝不**内置 β 的默认值。
  - **规则奖励的版本号必须一致**：容器侧的 `RULE_REWARD_VERSION` 与 Node 侧必须相同；
    且对同一组向量打分必须**逐位相同**（不一致 ⇒ Node 侧抛 E_REWARD_IMPL_DRIFT）。
  - **KL 与奖励逐步记账**：每步的 `kl` / `reward` / `reward_std` / `frac_reward_zero_std` 都随
    step 事件回传，由 Node 落进审计链（契约冻结条件 4）。

## 协议（行分隔 JSON；stdout **只允许**协议行，人类可读信息一律走 stderr）
请求 cmd：ping | run（流式） | merge_export | reward_vectors | shutdown
流式事件 ev：log | step | eval | done
"""
import json
import os
import re
import sys
import time
import math
import hashlib
import random
import shutil
import traceback
from typing import Any, Dict, List, Optional

import numpy as np
import torch

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402
from trl import GRPOTrainer, GRPOConfig  # noqa: E402

# ── 协议通道（stdout 只允许协议行）─────────────────────────────────────────────
# 先把**真正的 stdout** 存下来当协议通道，然后让 `sys.stdout` 指向 stderr：
# 之后任何库的 `print()`（transformers 的 PrinterCallback 就会 `print(logs)`）都进 stderr，
# 不会再往协议通道里塞非协议行。真机彩排就是在这里抓到"stdout 被污染"的。
PROTO_OUT = sys.stdout
sys.stdout = sys.stderr

# 流式命令的当前请求 id：`emit` 会把它补进任何**没有 id** 的消息里。
# 为什么要有这一步（真机彩排抓到的真缺陷）：流式事件（log/step/eval/done）都必须带请求 id，
# 否则 host 侧按 id 找不到进行中的请求 ⇒ 事件被丢弃 ⇒ 容器在 read_ack 上**永久阻塞**，
# 而症状只是"训练卡住不动"。把 id 的注入收敛到 emit 一处，就不可能再漏。
CUR_ID = None

# ──────────────────────────────────────────────────────────────────────────────
# 规则奖励（**必须与 src/reward.mjs 逐位一致**）
# ──────────────────────────────────────────────────────────────────────────────
# 实现放在**独立文件** `rule_reward.py`：这样"容器侧规则"可以在**无 GPU 的 CI** 里
# 直接 `python3 python/rule_reward.py --vectors <fixture>` 跑，与 JS 权威实现逐条比对；
# 而容器内也是同一个文件（/work = 本目录）⇒ CI 验的就是真机跑的那份代码。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rule_reward import (  # noqa: E402
    RULE_REWARD_VERSION,
    rule_reward_detail,
    rule_reward_value,
    stdev,
)


def rule_reward(completions, prompts=None, **kwargs) -> List[float]:
    """GRPOTrainer 的 reward_funcs 入口。

    **名字进日志键**（`rewards/rule_reward/mean|std`），故必须叫 rule_reward。
    TRL 1.14 的调用约定（已现场核对源码）：`reward_func(prompts=…, completions=…,
    completion_ids=…, trainer_state=…, log_extra=…, log_metric=…)`。
    """
    n = len(completions)
    if prompts is None:
        prompts = [""] * n
    out = []
    for i, c in enumerate(completions):
        text = c if isinstance(c, str) else " ".join(str(m.get("content", "")) for m in c)
        p = prompts[i] if i < len(prompts) else ""
        out.append(rule_reward_value(text, p))
    S.reward_calls += 1
    S.reward_values.extend(out)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 工具
# ──────────────────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    sys.stderr.write(f"[rl] {msg}\n")
    sys.stderr.flush()


def emit(obj: Dict[str, Any]) -> None:
    """写出**一条**协议行。缺 id 时自动补当前流式请求的 id（见 CUR_ID 的说明）。"""
    if "id" not in obj and CUR_ID is not None:
        obj = {"id": CUR_ID, **obj}
    PROTO_OUT.write(json.dumps(obj, ensure_ascii=False) + "\n")
    PROTO_OUT.flush()


def _num(x: Any) -> Optional[float]:
    """把任意值转成**可 JSON 化**的数：非有限值一律 None（NaN/Inf 不是合法 JSON）。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _sanitize(v: Any, depth: int = 0) -> Any:
    """递归清洗：非有限 float → None；非 JSON 类型 → str。保证 stdout 上永远是合法 JSON。"""
    if depth > 16:
        return None
    if v is None or isinstance(v, (str, bool, int)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {str(k): _sanitize(x, depth + 1) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize(x, depth + 1) for x in v]
    try:
        return _sanitize(v.item(), depth + 1)  # numpy 标量
    except Exception:
        return str(v)


def read_ack(rid: int) -> Dict[str, Any]:
    """等待 host 对某个流式事件的应答。**严格**：id 不匹配即失败，不静默错位。"""
    line = sys.stdin.readline()
    if not line:
        raise RuntimeError("host 断开（stdin EOF）—— 训练中止，避免写出'看起来跑完了'的产物")
    msg = json.loads(line)
    if int(msg.get("id", -1)) != int(rid):
        raise RuntimeError(f"协议错位：收到 id={msg.get('id')} 的应答，期望 id={rid}")
    return msg


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def sha256_file(p: str) -> Optional[str]:
    if not os.path.exists(p):
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def list_files(d: str) -> List[Dict[str, Any]]:
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        if os.path.isfile(p):
            out.append({"name": name, "bytes": os.path.getsize(p), "sha256": sha256_file(p)})
    return out


# ──────────────────────────────────────────────────────────────────────────────
# 状态
# ──────────────────────────────────────────────────────────────────────────────


class State:
    def __init__(self) -> None:
        self.model = None
        self.tok = None
        self.cfg: Dict[str, Any] = {}
        self.base_dir: Optional[str] = None
        self.reward_calls = 0
        self.reward_values: List[float] = []
        # 周期评估（选点信号）用：train 切分与 **留出** 切分的 prompt
        self.train_prompts: List[str] = []
        self.val_prompts: List[str] = []


S = State()


# ──────────────────────────────────────────────────────────────────────────────
# 命令：ping / reward_vectors
# ──────────────────────────────────────────────────────────────────────────────


def cmd_ping(_req: Dict[str, Any]) -> Dict[str, Any]:
    return {"pong": True, "torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "rewardVersion": RULE_REWARD_VERSION, "trl": __import__("trl").__version__}


def cmd_reward_vectors(req: Dict[str, Any]) -> Dict[str, Any]:
    """对**同一组**向量打分（宿主会与 JS 权威实现逐条比对）—— 跨语言防漂移闸门。"""
    vectors = req.get("vectors") or []
    values = [rule_reward_value(v.get("completion", ""), v.get("prompt", "")) for v in vectors]
    return {"version": RULE_REWARD_VERSION, "values": values, "n": len(values)}


# ──────────────────────────────────────────────────────────────────────────────
# 命令：run（流式）
# ──────────────────────────────────────────────────────────────────────────────


def _load_policy(cfg: Dict[str, Any]):
    """加载**原始基座** + 构造 LoRA 配置。

    ⚠️ **不要**在这里 `get_peft_model`：TRL 1.14 的 GRPOTrainer 明确拒绝
    "PeftModel + peft_config 同时传入"（源码里就是一句 `raise ValueError("You passed a PeftModel
    instance together with a peft_config to the trainer...")`，已现场核对）。
    正确口径是：**传原始模型 + peft_config，由 TRL 自己包 LoRA**，包好后的模型在 `trainer.model`。
    于是"当前策略"在 step 0 与基座**数值等价**（LoRA 的 B 初始化为 0 ⇒ ΔW ≡ 0），
    所以 σ_r 校准放在构造 trainer 之前、直接在原始基座上做，语义一致。
    """
    base = cfg["baseModel"]
    S.base_dir = base
    tok = AutoTokenizer.from_pretrained(base, trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # 生成必须左填充（decoder-only）
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map={"": 0})
    model.config.use_cache = False
    lora_cfg = cfg.get("lora") or {}
    peft_config = LoraConfig(
        r=int(lora_cfg.get("r", 8)),
        lora_alpha=int(lora_cfg.get("alpha", 16)),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=list(lora_cfg.get("targetModules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
    )
    S.tok = tok
    S.model = model  # 校准期用原始基座（= 策略@step0）
    return model, peft_config


def _calibrate(model, tok, prompts: List[str], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """**实测** σ_r：在当前策略上采样、用规则奖励打分、取样本标准差。

    为什么必须实测（不能拍数字）：β = σ_r / KL* 里 σ_r 是**策略相关的量**；
    换了策略/换了奖励版本，σ_r 就变了。σ_r = 0 意味着 GRPO 的组内优势恒为 0（空转），
    此时必须**拒绝开始训练**（E_REWARD_NO_VARIANCE），而不是跑完 200 步再报"成功率 100%"。
    """
    n_prompts = min(int(cfg.get("calibPrompts", 8)), len(prompts))
    gens = int(cfg.get("calibGenerations", 4))
    max_new = int(cfg.get("maxCompletionLength", 64))
    t0 = time.time()
    values: List[float] = []
    was_training = model.training
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for p in prompts[:n_prompts]:
            enc = tok([p], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                 max_new_tokens=max_new, num_return_sequences=gens, pad_token_id=tok.pad_token_id)
            for seq in out:
                text = tok.decode(seq[enc["input_ids"].shape[1]:], skip_special_tokens=True)
                values.append(rule_reward_value(text, p))
    if was_training:
        model.train()
    sigma = stdev(values)
    distinct = len({round(v, 9) for v in values})
    reward_mean = (sum(values) / len(values)) if values else None
    return {
        "promptsUsed": n_prompts,
        "generationsPerPrompt": gens,
        "samples": len(values),
        "rewardMean": _num(reward_mean),
        "sigmaR": _num(sigma),
        "rewardStd": _num(sigma),
        "rewardMin": _num(min(values)) if values else None,
        "rewardMax": _num(max(values)) if values else None,
        "distinctValues": distinct,
        "sampleValues": [_num(v) for v in values[:24]],
        "measuredAt": "policy@step0",
        "method": "model.generate(temperature=1.0) 采样 + 规则奖励打分，样本标准差（n−1 分母）",
        "ms": int((time.time() - t0) * 1000),
    }


class HostBridge(TrainerCallback):
    """把每个 optimizer 步交给 host 判定，并执行 host 的回传指令（save / stop）。"""

    def __init__(self, rid: int, cfg: Dict[str, Any]) -> None:
        self.rid = rid
        self.cfg = cfg
        self.last_step = 0
        self.t0 = time.time()
        self.steps: List[Dict[str, Any]] = []
        self.saved: List[Dict[str, Any]] = []
        self.log_tail: List[Dict[str, Any]] = []
        self.stop_reason: Optional[str] = None
        self.max_steps = int(cfg.get("maxSteps", 0) or 0)
        self.evals: List[Dict[str, Any]] = []

    def on_log(self, args, state, control, **kwargs):
        logs = kwargs.get("logs") or {}
        gs = int(getattr(state, "global_step", 0) or 0)
        if gs <= self.last_step:  # 收尾/epoch 的重复日志：不重复记账
            return control
        self.last_step = gs
        self.log_tail.append(_sanitize(logs))
        if len(self.log_tail) > 6:
            self.log_tail.pop(0)
        elapsed = time.time() - self.t0
        metrics = {
            "kl": _num(logs.get("kl")),
            "reward": _num(logs.get("reward", logs.get("rewards/rule_reward/mean"))),
            "rewardStd": _num(logs.get("reward_std", logs.get("rewards/rule_reward/std"))),
            "lr": _num(logs.get("learning_rate")),
            "gradNorm": _num(logs.get("grad_norm")),
            "entropy": _num(logs.get("entropy")),
            "fracRewardZeroStd": _num(logs.get("frac_reward_zero_std")),
            "completionMeanLength": _num(logs.get("completions/mean_length")),
            "clipRatioRegionMean": _num(logs.get("clip_ratio/region_mean")),
        }
        rec = {"step": gs, "loss": _num(logs.get("loss")), "epoch": _num(getattr(state, "epoch", None)),
               "gpuHours": elapsed / 3600.0, "wallMs": int(elapsed * 1000), "metrics": metrics}
        self.steps.append(rec)
        # `ack: True` 是**协议契约**：只有声明要应答的事件，host 才回一条 ack；
        # log/eval 是单向通知。两侧步调必须严格一致（所有事件共用同一个 id，
        # 少读/多读一条都会错位，且 id 校验发现不了）。
        emit({"ev": "step", "ack": True, **rec})
        ack = read_ack(self.rid)
        save = ack.get("save")
        if save and save.get("dir"):
            d = str(save["dir"])
            if not d.startswith("/"):
                d = os.path.join("/out", d)
            info = _save_adapter(d, gs)
            self.saved.append({"dir": os.path.relpath(d, "/out"), "step": gs, "bytes": info["bytes"],
                               "files": len(info["files"])})
            emit({"ev": "log", "text": f"saved adapter step={gs} → {d} ({info['bytes']} bytes)"})
            # 周期评估与 checkpoint 同频（**选点信号**，见 _quick_eval 的说明）
            self._maybe_eval(gs)
        if ack.get("stop"):
            self.stop_reason = ack.get("reason") or "host-budget"
            control.should_training_stop = True
            log(f"host 要求停止：{self.stop_reason}")
        return control

    def _maybe_eval(self, gs: int) -> None:
        """容器内周期评估（`selectionOnly: true`）—— 只用于 O4 的选点，不作判定。"""
        every = int(self.cfg.get("evalEvery", 0) or 0)
        if every <= 0 or gs % every != 0:
            return
        for split, prompts in (("train", S.train_prompts), ("val", S.val_prompts)):
            if not prompts:
                continue
            try:
                v = _quick_eval(S.model, S.tok, prompts, self.cfg)
            except Exception as e:  # noqa: BLE001 —— 选点信号失败不应毁掉训练
                emit({"ev": "log", "text": f"周期评估({split})失败：{type(e).__name__}: {e}"})
                continue
            if v is None:
                continue
            rec = {"step": gs, "split": split, "metric": "meanRuleReward", "value": _num(v),
                   "unit": "rule-reward", "selectionOnly": True}
            self.evals.append(rec)
            emit({"ev": "eval", **rec})


def _quick_eval(model, tok, prompts: List[str], cfg: Dict[str, Any]) -> Optional[float]:
    """**容器内周期评估**（只作选点信号，见 training-eval-contract §5.2）。

    为什么允许它存在：O4 的选点规则需要一条 val 曲线；把它放到 serving 路径上做会造成
    每次评估都起/停端点（成本不成比例）。契约因此明确规定：容器内评估**只能用于选择
    checkpoint**，必须标 `selectionOnly: true`，且**不得**进入任何门禁判定 ——
    所有判定数字一律来自 serving（§5.1）。
    """
    n = min(int(cfg.get("evalPrompts", 8)), len(prompts))
    if n == 0:
        return None
    max_new = int(cfg.get("evalMaxNew", 48))
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    vals: List[float] = []
    with torch.no_grad():
        for p in prompts[:n]:
            enc = tok([p], return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            out = model.generate(**enc, do_sample=False, max_new_tokens=max_new,
                                 num_return_sequences=1, pad_token_id=tok.pad_token_id)
            text = tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
            vals.append(rule_reward_value(text, p))
    if was_training:
        model.train()
    return (sum(vals) / len(vals)) if vals else None


def _save_adapter(d: str, step: int) -> Dict[str, Any]:
    adir = os.path.join(d, "adapter")
    os.makedirs(adir, exist_ok=True)
    S.model.save_pretrained(adir, safe_serialization=True)
    files = list_files(adir)
    return {"adapterDir": adir, "files": files, "bytes": sum(f["bytes"] for f in files), "step": step}


def _merge_and_export(out: str, cfg: Dict[str, Any], provenance: Dict[str, Any]) -> Dict[str, Any]:
    os.makedirs(out, exist_ok=True)
    merged = S.model.merge_and_unload()
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    S.tok.save_pretrained(out)
    # serving 的 license 闸门读 README frontmatter 或 LICENSE 文件
    base = S.base_dir or cfg.get("baseModel") or ""
    src_license = os.path.join(base, "LICENSE")
    if os.path.exists(src_license):
        shutil.copyfile(src_license, os.path.join(out, "LICENSE"))
    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"license: {cfg.get('licenseSpdx', 'apache-2.0')}\n")
        f.write("base_model: Qwen/Qwen2.5-0.5B-Instruct\n")
        f.write("library_name: transformers\n")
        f.write("---\n\n")
        f.write("# training-rl GRPO LoRA-merged checkpoint\n\n")
        f.write(f"reward: {RULE_REWARD_VERSION} (rule); beta derived as sigmaR / KL*\n\n")
    with open(os.path.join(out, "training-rl-provenance.json"), "w", encoding="utf-8") as f:
        json.dump(_sanitize(provenance), f, ensure_ascii=False, indent=2)
    files = list_files(out)
    weights = [f for f in files if f["name"].endswith(".safetensors")]
    wsum = hashlib.sha256(("\n".join(f["sha256"] for f in weights)).encode()).hexdigest()
    return {"outDir": out, "files": files, "weightsSha256": "sha256:" + wsum,
            "bytes": sum(f["bytes"] for f in files),
            "params": int(sum(p.numel() for p in merged.parameters())),
            "vocabSize": int(merged.config.vocab_size)}


def cmd_run(req: Dict[str, Any]) -> None:
    """流式命令：**自己负责 emit done**（main 对它不发普通应答）。"""
    global CUR_ID
    rid = int(req["id"])
    CUR_ID = rid  # 之后所有流式事件自动带上这个 id
    cfg = req.get("config") or {}
    S.cfg = cfg
    started = time.time()
    try:
        seed = int(cfg.get("seed", 20260929))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        t_load = time.time()
        model, peft_config = _load_policy(cfg)
        model.train()
        load_ms = int((time.time() - t_load) * 1000)
        emit({"ev": "log", "text": f"基座加载完成 {load_ms}ms（LoRA 由 TRL 用 peft_config 包装）"})

        rows = load_jsonl(cfg["promptsFile"])
        prompts = [str(r.get("prompt", "")) for r in rows if str(r.get("prompt", "")).strip()]
        if not prompts:
            raise RuntimeError("prompts 为空 —— 拒绝开始训练")
        emit({"ev": "log", "text": f"训练 prompt {len(prompts)} 条"})
        # 周期评估（选点信号）用的两套 prompt：训练切分 + **留出**切分
        S.train_prompts = prompts
        S.val_prompts = []
        if cfg.get("valPromptsFile") and os.path.exists(str(cfg.get("valPromptsFile"))):
            vrows = load_jsonl(str(cfg["valPromptsFile"]))
            S.val_prompts = [str(r.get("prompt", "")) for r in vrows if str(r.get("prompt", "")).strip()]
            emit({"ev": "log", "text": f"留出 prompt {len(S.val_prompts)} 条（仅用于选点信号）"})

        # ── 规则奖励的跨语言一致性（向量由宿主给，容器打分回传）
        vecs = cfg.get("rewardVectors") or []
        conformance = cmd_reward_vectors({"vectors": vecs})

        # ── σ_r 实测 → β 推导（**绝不给 β 默认值**）
        t_cal = time.time()
        calib = _calibrate(model, S.tok, prompts, cfg)
        calib_ms = int((time.time() - t_cal) * 1000)
        sigma_r = calib.get("sigmaR")
        kl_budget = _num(cfg.get("klBudget"))
        if kl_budget is None or kl_budget <= 0:
            raise RuntimeError("klBudget（KL*）必须由调用方显式给出且 > 0 —— 本文件不内置默认值")
        if sigma_r is None or sigma_r <= 0:
            raise RuntimeError(
                f"E_REWARD_NO_VARIANCE: 实测 σ_r={sigma_r}（样本 {calib.get('samples')} 条，"
                f"不同取值 {calib.get('distinctValues')} 个）—— GRPO 的组内优势恒为 0，训练是空转，拒绝开始")
        beta = sigma_r / kl_budget
        calib["beta"] = _num(beta)
        calib["klBudget"] = _num(kl_budget)
        calib["betaFormula"] = "beta = sigmaR / klBudget"
        calib["ms"] = calib_ms
        emit({"ev": "log", "text": f"σ_r={sigma_r:.6f} KL*={kl_budget} ⇒ β={beta:.6f}"})

        # ── 训练
        dataset = Dataset.from_dict({"prompt": prompts})
        args = GRPOConfig(
            output_dir="/out/trl",
            max_steps=int(cfg.get("maxSteps", 200)),
            per_device_train_batch_size=int(cfg.get("batchSize", 4)),
            gradient_accumulation_steps=int(cfg.get("gradAccum", 1)),
            num_generations=int(cfg.get("numGenerations", 4)),
            max_completion_length=int(cfg.get("maxCompletionLength", 64)),
            learning_rate=float(cfg.get("lr", 1e-6)),
            beta=float(beta),
            temperature=float(cfg.get("temperature", 1.0)),
            bf16=True,
            report_to=[],
            logging_steps=1,
            save_strategy="no",            # checkpoint 由 host（training-core）编排，TRL 自己不许存
            gradient_checkpointing=False,
            use_vllm=False,
            seed=seed,
            remove_unused_columns=False,
        )
        bridge = HostBridge(rid, cfg)
        # ⚠️ **必须**把 `peft_config` 交给 GRPOTrainer（而不是自己先 get_peft_model 再传模型）：
        #    TRL 1.14 明确拒绝"PeftModel + peft_config 同时传入"，而"只传 PeftModel 不传 peft_config"
        #    会走去另一条分支（为已有 adapter 造 ref adapter），语义与"从零开训"不同。
        #    正确口径：**传原始模型 + peft_config**，TRL 自己包 LoRA。
        #    （第一版漏了 peft_config ⇒ 真机彩排里 `trainer.model` 是 Qwen2ForCausalLM，
        #      等于在全参微调，且合并阶段会因没有 merge_and_unload 而失败。这条断言就是那次彩排的产物。）
        trainer = GRPOTrainer(model=model, reward_funcs=[rule_reward], args=args,
                              train_dataset=dataset, processing_class=S.tok, callbacks=[bridge],
                              peft_config=peft_config)
        S.model = trainer.model
        # **配置不变量**（fail-closed，且必须在开训之前）：策略必须是 PeftModel。
        # 否则后面 "只保存 adapter" 与 "merge_and_unload" 两处都会静默走错路径。
        if not isinstance(S.model, PeftModel):
            raise RuntimeError(
                f"E_NOT_A_PEFT_POLICY: trainer.model 是 {type(S.model).__name__}，不是 PeftModel —— "
                "说明 peft_config 没被 TRL 用上（会变成全参微调，且保存/合并路径全错）。拒绝开训。")
        if not hasattr(S.model, "merge_and_unload"):
            raise RuntimeError("E_PEFT_NO_MERGE: 策略缺少 merge_and_unload（合并阶段会失败）—— 拒绝开训")
        # 去掉 transformers 默认的 PrinterCallback（它会 `print(logs)` 打 200 行 dict）：
        # 逐 step 的指标已经由 HostBridge 走协议回传，不需要再往 stderr 灌一份。
        try:
            from transformers.trainer_callback import PrinterCallback
            trainer.remove_callback(PrinterCallback)
        except Exception as e:  # noqa: BLE001 —— 拿不到就算了，不该因它失败
            log(f"remove PrinterCallback 失败（忽略）：{type(e).__name__}: {e}")
        n_train = sum(p.numel() for p in S.model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in S.model.parameters())
        emit({"ev": "log", "text": f"GRPOTrainer 构造完成（策略类型 {type(S.model).__name__}，可训练参数 {n_train}/{n_all} = {n_train / n_all * 100:.4f}%），开始训练"})
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t_train = time.time()
        bridge.t0 = t_train
        outcome = trainer.train()
        train_ms = int((time.time() - t_train) * 1000)

        # ── 最终 adapter（**无论是否命中 checkpoint 边界都要有产物**）
        final = _save_adapter("/out", bridge.last_step)
        emit({"ev": "log", "text": f"最终 adapter 落盘：{final['bytes']} bytes"})

        # ── 合并（产物级断言要的是"策略权重"；同时让 validate 直接用 merged 目录）
        t_merge = time.time()
        provenance = {"runId": cfg.get("runId"), "baseModel": cfg.get("baseModel"),
                      "reward": {"kind": "rule", "version": RULE_REWARD_VERSION},
                      "beta": _num(beta), "sigmaR": _num(sigma_r), "klBudget": _num(kl_budget),
                      "steps": bridge.last_step, "seed": seed,
                      # 权重产物的 license = 基座（apache-2.0）；语料 license 只作**署名**记录，
                      # 不冒称产物的 license（见 core.mjs 中 licenseSpdx 处的说明）
                      "artifactLicense": cfg.get("licenseSpdx"),
                      "datasetLicense": cfg.get("datasetLicense")}
        merged = _merge_and_export("/out/merged", cfg, provenance)
        merge_ms = int((time.time() - t_merge) * 1000)

        kls = [s["metrics"]["kl"] for s in bridge.steps if s["metrics"]["kl"] is not None]
        rewards = [s["metrics"]["reward"] for s in bridge.steps if s["metrics"]["reward"] is not None]
        zero_std = [s["metrics"]["fracRewardZeroStd"] for s in bridge.steps if s["metrics"]["fracRewardZeroStd"] is not None]
        peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        rss = None
        try:
            with open("/proc/self/status") as f:
                rss = int(f.read().split("VmRSS:")[1].split()[0]) / 2**20
        except Exception:
            rss = None

        emit({
            "ev": "done", "ok": True,
            "result": {
                "steps": bridge.last_step,
                "finalLoss": _num(getattr(outcome, "training_loss", None)) if outcome is not None else None,
                "stopReason": bridge.stop_reason,
                "saved": bridge.saved,
                "finalAdapter": {"dir": "adapter", "bytes": final["bytes"], "files": len(final["files"])},
                "merged": {"dir": "merged", "weightsSha256": merged["weightsSha256"], "params": merged["params"],
                           "bytes": merged["bytes"], "files": len(merged["files"]), "vocabSize": merged["vocabSize"]},
                "rewardCalls": S.reward_calls,
                "rewardSamplesFirst": [_num(v) for v in S.reward_values[:12]],
                "klMean": _num(sum(kls) / len(kls)) if kls else None,
                "klMin": _num(min(kls)) if kls else None,
                "klMax": _num(max(kls)) if kls else None,
                "rewardMean": _num(sum(rewards) / len(rewards)) if rewards else None,
                "fracRewardZeroStdMean": _num(sum(zero_std) / len(zero_std)) if zero_std else None,
                "logTail": bridge.log_tail[-3:],
            },
            "calibration": calib,
            "rewardConformance": conformance,
            "steps": bridge.steps,
            "evaluations": bridge.evals,
            "memory": {"torchPeakGiB": _num(peak), "processRssGiB": _num(rss)},
            "timing": {"loadMs": load_ms, "calibrationMs": calib_ms, "trainMs": train_ms, "mergeMs": merge_ms,
                       "totalMs": int((time.time() - started) * 1000)},
            "versions": {
                "torch": torch.__version__, "trl": __import__("trl").__version__,
                "transformers": __import__("transformers").__version__,
                "peft": __import__("peft").__version__,
                "cuda": torch.cuda.is_available(),
                "deviceCapability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
            },
        })
    except Exception as e:  # noqa: BLE001 —— 必须走协议回传，不能污染 stdout 也不能静默
        emit({"ev": "done", "ok": False, "error": f"{type(e).__name__}: {e}",
              "trace": traceback.format_exc()[-2000:]})


def cmd_merge_export(req: Dict[str, Any]) -> Dict[str, Any]:
    """独立合并命令（从已落盘的 adapter 重新合并；rollback / 复核用）。"""
    cfg = req.get("config") or {}
    S.base_dir = cfg["baseModel"]
    tok = AutoTokenizer.from_pretrained(cfg["baseModel"], trust_remote_code=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg["baseModel"], dtype=torch.bfloat16, device_map={"": 0})
    S.tok = tok
    S.model = PeftModel.from_pretrained(model, req["adapterDir"], is_trainable=False)
    merged = _merge_and_export(req["outDir"], cfg, req.get("provenance") or {})
    return {"merged": {k: v for k, v in merged.items() if k != "files"}, "fileCount": len(merged["files"])}


# ──────────────────────────────────────────────────────────────────────────────


STREAM_CMDS = {"run"}


def main() -> None:
    log(f"worker up pid={os.getpid()} torch={torch.__version__} cuda={torch.cuda.is_available()} "
        f"trl={__import__('trl').__version__} reward={RULE_REWARD_VERSION}")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        cmd = req.get("cmd")
        rid = req.get("id")
        if cmd == "shutdown":
            emit({"id": rid, "ok": True, "bye": True})
            break
        if cmd in STREAM_CMDS:
            # 流式命令**自己负责** emit done；这里绝不能额外回一条普通应答
            # （那会让 host 的流式解析看到一个没有 ev 的消息）
            try:
                cmd_run(req)
            except Exception as e:  # noqa: BLE001
                emit({"ev": "done", "ok": False, "error": f"{type(e).__name__}: {e}",
                      "trace": traceback.format_exc()[-2000:]})
            continue
        t0 = time.time()
        try:
            fn = {"ping": cmd_ping, "reward_vectors": cmd_reward_vectors, "merge_export": cmd_merge_export}.get(cmd)
            if fn is None:
                raise ValueError(f"未知命令 {cmd}")
            res = fn(req)
            res.update({"id": rid, "ok": True, "handlerMs": int((time.time() - t0) * 1000)})
            emit(res)
        except Exception as e:  # noqa: BLE001
            emit({"id": rid, "ok": False, "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    main()
