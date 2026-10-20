#!/usr/bin/env python3
"""
rule_reward.py —— 规则奖励的**容器侧实现**（与 `src/reward.mjs` 逐条对齐）。

## 为什么单独成文件（而不是塞在 rl_trainer.py 里）
两个理由，都是"让判据可执行"：

1. **跨语言一致性必须能在无 GPU 的 CI 里验证**。`rl_trainer.py` 在模块级 import torch/trl，
   于是"Python 侧规则"这件事就无法在无 GPU 环境里被检查；拆出来后
   `python3 python/rule_reward.py --vectors <fixture>` 可以在**纯 CPU**下跑，
   与 JS 权威实现逐条比对（`tests/unit/reward-kl.spec.mjs` 就是这么做的）。
2. **容器内也是同一个文件**（`/work` = 本目录，`rl_trainer.py` 直接 import），
   所以 CI 验的就是真机跑的那份代码 —— 不存在"测的是一份、跑的是另一份"。

## 与 JS 的逐位一致
两边都是 IEEE754 double，且**运算顺序逐条对齐**（同样的分量顺序、同样的求和顺序）。
规格 `tests/fixtures/reward-vectors.json` 由 JS 实现生成并覆盖每一条判定分支。

## 约定（改规则必须同时改三处，缺一即 CI 红）
  - `src/reward.mjs`（权威 · JS）
  - `python/rule_reward.py`（本文件 · 容器侧）
  - `tests/fixtures/reward-vectors.json`（规格；用 `node tests/gen-reward-vectors.mjs` 重新生成）
"""
import argparse
import json
import math
import re
import sys
from typing import Any, Dict, List

RULE_REWARD_VERSION = "rule-reward@1"

RULE_WEIGHTS = {"readable": 1, "no_restart": 1, "length": 1, "repetition": 1, "relevance": 1, "constraint": 1}
LENGTH_BAND_MIN_WORDS = 3
LENGTH_BAND_MAX_WORDS = 48
MAX_CHARS = 2048
RELEVANCE_FULL_AT = 0.6
MAX_TRI_REPEAT = 3

STOPWORDS = [
    'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'of', 'to', 'in', 'on', 'for', 'and', 'or',
    'but', 'with', 'as', 'at', 'by', 'from', 'into', 'that', 'this', 'these', 'those', 'it', 'its', 'your', 'you',
    'i', 'we', 'they', 'he', 'she', 'them', 'their', 'our', 'my', 'me', 'us', 'do', 'does', 'did', 'doing',
    'how', 'what', 'why', 'when', 'where', 'which', 'who', 'whom', 'can', 'could', 'should', 'would', 'will',
    'shall', 'may', 'might', 'must', 'not', 'no', 'yes', 'if', 'then', 'than', 'so', 'such', 'all', 'any', 'each',
    'more', 'most', 'some', 'other', 'only', 'also', 'very', 'just', 'about', 'after', 'before', 'between', 'out',
    'up', 'down', 'over', 'under', 'again', 'here', 'there', 'please', 'write', 'create', 'give', 'make', 'using',
    'use', 'used', 'answer', 'output', 'return', 'example', 'examples', 'following', 'list', 'explain',
    'provide', 'suggest', 'generate', 'describe', 'tell', 'help', 'need', 'want', 'like', 'get', 'got', 'one', 'two',
    'three', 'word', 'words', 'sentence', 'sentences', 'paragraph', 'text', 'number', 'numbers', 'json', 'code',
    'program', 'script', 'function', 'python', 'java', 'javascript', 'based', 'new', 'same',
]
_STOP = set(STOPWORDS)
_WORD_RE = re.compile(r"[A-Za-z0-9\u4e00-\u9fff]+")
_ALNUM_RE = re.compile(r"[0-9A-Za-z\u4e00-\u9fff]")
_DIGIT_RE = re.compile(r"[0-9]")


def r_words(text: str) -> List[str]:
    return _WORD_RE.findall(str(text if text is not None else "").lower())


def r_word_count(text: str) -> int:
    return len(r_words(text))


def r_content_words(text: str) -> set:
    return {w for w in r_words(text) if len(w) >= 3 and w not in _STOP}


def r_max_tri_repeat(text: str) -> int:
    ws = [w for w in str(text if text is not None else "").strip().split() if w]
    if len(ws) < 6:
        return 1
    counts: Dict[str, int] = {}
    best = 1
    for i in range(len(ws) - 2):
        g = f"{ws[i]} {ws[i + 1]} {ws[i + 2]}"
        c = counts.get(g, 0) + 1
        counts[g] = c
        if c > best:
            best = c
    return best


def r_derive_constraint(prompt: str) -> Dict[str, Any]:
    p = str(prompt if prompt is not None else "").lower()
    if "one word" in p or "一个词" in p or "一个单词" in p:
        return {"id": "one-word", "kind": "one-word"}
    if "json" in p:
        return {"id": "json", "kind": "json"}
    if "number" in p or "数字" in p:
        return {"id": "number", "kind": "number"}
    return {"id": "none", "kind": "none"}


def _constraint_score(prompt: str, text: str) -> float:
    kind = r_derive_constraint(prompt)["kind"]
    t = str(text if text is not None else "")
    if kind == "one-word":
        return 1.0 if r_word_count(t) == 1 else 0.0
    if kind == "json":
        s = t.strip()
        return 1.0 if (s.startswith("{") and s.endswith("}")) else 0.0
    if kind == "number":
        return 1.0 if _DIGIT_RE.search(t) else 0.0
    return 0.5


def rule_reward_detail(completion: str, prompt: str = "") -> Dict[str, Any]:
    """**与 src/reward.mjs 的 ruleReward 逐条对齐**（运算顺序也一致，以便逐位相同）。"""
    trimmed = str(completion if completion is not None else "").strip()
    readable = 1.0 if (len(trimmed) > 0 and _ALNUM_RE.search(trimmed)) else 0.0
    restarted = ("### Instruction:" in trimmed) or ("### Input:" in trimmed)
    no_restart = 1.0 if ((not restarted) and len(trimmed) > 0 and len(trimmed) <= MAX_CHARS) else 0.0
    w = r_word_count(trimmed)
    if w == 0:
        length = 0.0
    elif w < LENGTH_BAND_MIN_WORDS:
        length = w / LENGTH_BAND_MIN_WORDS
    elif w <= LENGTH_BAND_MAX_WORDS:
        length = 1.0
    else:
        length = max(0.0, 1.0 - (w - LENGTH_BAND_MAX_WORDS) / LENGTH_BAND_MAX_WORDS)
    mt = r_max_tri_repeat(trimmed)
    repetition = 1.0 if mt <= MAX_TRI_REPEAT else max(0.0, 1.0 - (mt - MAX_TRI_REPEAT) / MAX_TRI_REPEAT)
    pw = r_content_words(prompt)
    if len(pw) == 0:
        relevance = 0.5
    else:
        cw = r_content_words(trimmed)
        hit = 0
        for x in pw:
            if x in cw:
                hit += 1
        relevance = min(1.0, (hit / len(pw)) / RELEVANCE_FULL_AT)
    constraint = 0.0 if readable == 0.0 else _constraint_score(prompt, trimmed)
    parts = {"readable": readable, "no_restart": no_restart, "length": length,
             "repetition": repetition, "relevance": relevance, "constraint": constraint}
    keys = list(RULE_WEIGHTS.keys())
    total_w = sum(RULE_WEIGHTS[k] for k in keys)
    value = sum(RULE_WEIGHTS[k] * parts[k] for k in keys) / total_w
    return {"value": value, "parts": parts}


def rule_reward_value(completion: str, prompt: str = "") -> float:
    return rule_reward_detail(completion, prompt)["value"]


def stdev(xs: List[float]):
    """样本标准差（n−1 分母）—— 与 testing-core 的 stddev 同口径。σ_r 用的就是它。"""
    n = len(xs)
    if n < 2:
        return None
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _main() -> int:
    ap = argparse.ArgumentParser(description="容器侧规则奖励：对规格向量打分（跨语言一致性检查）")
    ap.add_argument("--vectors", required=True, help="reward-vectors.json 路径")
    ap.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    a = ap.parse_args()
    doc = json.load(open(a.vectors, "r", encoding="utf-8"))
    out = []
    for v in doc.get("vectors", []):
        d = rule_reward_detail(v.get("completion", ""), v.get("prompt", ""))
        out.append({"branch": v.get("branch"), "value": d["value"], "parts": d["parts"]})
    if a.json:
        print(json.dumps({"version": RULE_REWARD_VERSION, "values": out}, ensure_ascii=False))
    else:
        for x in out:
            print(f"{str(x['branch']):34s} {x['value']:.6f}")
    # 与规格自带的期望值自检（规格过时/实现漂移都会红）
    bad = []
    for got, want in zip(out, doc.get("vectors", [])):
        if abs(float(got["value"]) - float(want["expected"])) > 1e-12:
            bad.append(f"{got['branch']}: {got['value']!r} != {want['expected']!r}")
    if bad:
        sys.stderr.write("与规格不符：\n  " + "\n  ".join(bad) + "\n")
        return 1
    sys.stderr.write(f"OK {len(out)} 条向量全部与规格一致（版本 {RULE_REWARD_VERSION}）\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
