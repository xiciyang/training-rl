#!/usr/bin/env node
/**
 * gen-reward-vectors —— 生成**规则奖励的跨语言规格**（tests/fixtures/reward-vectors.json）。
 *
 * 为什么要一个生成器而不是手写：手写的期望值会随规则演进悄悄漂移，
 * 而"规格由实现生成、随后被两份实现各自校验"是这里唯一可执行的防漂移形态：
 *   规格 = 固定的 (prompt, completion) 输入 + **由 JS 权威实现算出的**期望值；
 *   - JS 侧：`tests/unit/reward-kl.spec.mjs` 断言实现 == 规格；
 *   - Python 侧：真机训练时容器对**同一组向量**打分并回传，Node 逐条比对（E_REWARD_IMPL_DRIFT）。
 * 于是"两处实现"这件事有了机器判据，而不是靠人记得同步。
 *
 * 向量必须覆盖 `RULE_BRANCHES` 的每一条分支 —— 生成器自检覆盖率（漏了即非零退出）。
 *
 * 用法：node tests/gen-reward-vectors.mjs [--check]
 *   --check：只比对、不写盘（规格过时则非零退出）—— CI 用。
 */
import { writeFileSync, readFileSync, existsSync, mkdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { ruleReward, RULE_REWARD_VERSION, RULE_BRANCHES } from '../src/reward.mjs'

const HERE = dirname(fileURLToPath(import.meta.url))
const OUT = join(HERE, 'fixtures', 'reward-vectors.json')

const GRAV = 'Explain gravity.'
const alpha = (n) => Array.from({ length: n }, (_, i) => String.fromCharCode(97 + (i % 26))).join('')
const words = (n, w = 'alpha') => Array.from({ length: n }, () => w).join(' ')
const TOO_LONG_CHARS = 'x'.repeat(3000)
const REPEAT = 'the same three words the same three words the same three words the same three words'
const LONG_96 = Array.from({ length: 96 }, (_, i) => `w${i}`).join(' ')
const TOO_LONG_96_WORDS = `${Array.from({ length: 60 }, () => 'alpha').join(' ')} ${Array.from({ length: 36 }, (_, i) => `z${i}`).join(' ')}`
void alpha; void words; void TOO_LONG_96_WORDS

/** 每条向量标注它想覆盖的**分支**（生成器自检覆盖率；结构判据在 spec 里）。 */
export const VECTORS = [
  // ── ① readable
  { branch: 'readable-empty', prompt: GRAV, completion: '' },
  { branch: 'readable-whitespace', prompt: GRAV, completion: '   \n\t  ' },
  { branch: 'readable-symbols-only', prompt: GRAV, completion: '... --- ***' },
  { branch: 'readable-ok', prompt: GRAV, completion: 'Gravity pulls masses toward each other.' },

  // ── ② no_restart
  { branch: 'restart-instruction', prompt: GRAV, completion: '### Instruction:\nExplain gravity.' },
  { branch: 'restart-input', prompt: GRAV, completion: '### Input:\nsomething about gravity' },
  { branch: 'restart-too-long', prompt: GRAV, completion: TOO_LONG_CHARS },
  { branch: 'restart-ok', prompt: GRAV, completion: 'Gravity is a force of attraction between masses.' },

  // ── ④ repetition（③ 的长度在这里都落在带内，隔离出重复分量）
  { branch: 'repetition-degenerate', prompt: GRAV, completion: REPEAT },
  { branch: 'repetition-ok', prompt: GRAV, completion: 'Gravity keeps the planets in orbit around the sun.' },

  // ── ③ length
  { branch: 'length-zero', prompt: GRAV, completion: '###' },
  { branch: 'length-short', prompt: GRAV, completion: 'Gravity.' },
  { branch: 'length-band-low', prompt: GRAV, completion: 'Gravity attracts two masses.' },
  { branch: 'length-band-mid', prompt: GRAV, completion: 'Gravity is a fundamental interaction that attracts any two masses toward each other.' },
  { branch: 'length-band-high', prompt: GRAV, completion: LONG_96.split(' ').slice(0, 47).join(' ') + ' end' },
  { branch: 'length-too-long', prompt: GRAV, completion: LONG_96 },

  // ── ⑤ relevance
  { branch: 'relevance-full', prompt: 'Explain gravity and orbital mechanics.', completion: 'Gravity and orbital mechanics explain why planets stay in orbit.' },
  { branch: 'relevance-partial', prompt: 'Explain gravity and orbital mechanics.', completion: 'Gravity is a fundamental interaction between masses.' },
  { branch: 'relevance-zero', prompt: 'Explain gravity and orbital mechanics.', completion: 'Bananas are yellow and ripe in summer.' },
  { branch: 'relevance-no-content-words', prompt: 'to be or not to be', completion: 'That is the question asked by many.' },

  // ── ⑥ constraint
  { branch: 'constraint-one-word-ok', prompt: 'Answer in one word: what color is the sky?', completion: 'Blue' },
  { branch: 'constraint-one-word-violated', prompt: 'Answer in one word: what color is the sky?', completion: 'The sky is blue today' },
  { branch: 'constraint-number-ok', prompt: 'What is 2+2? Answer with a number.', completion: 'The answer is 4.' },
  { branch: 'constraint-number-violated', prompt: 'What is 2+2? Answer with a number.', completion: 'The answer is four.' },
  { branch: 'constraint-json-ok', prompt: 'Return the result as json.', completion: '{"answer": 4}' },
  { branch: 'constraint-json-violated', prompt: 'Return the result as json.', completion: 'The result is four.' },
  { branch: 'constraint-absent-neutral', prompt: 'Describe the water cycle in a few sentences.', completion: 'Water evaporates, condenses into clouds and falls back as rain.' },
]

export function build() {
  return {
    schema: 'training-rl/reward-vectors@1',
    rewardVersion: RULE_REWARD_VERSION,
    note: '由 tests/gen-reward-vectors.mjs 从 JS 权威实现生成；JS 单测与容器内 Python 实现都必须逐条相符。',
    branches: [...RULE_BRANCHES],
    vectors: VECTORS.map((v) => {
      const r = ruleReward(v.completion, v.prompt)
      return { branch: v.branch, prompt: v.prompt, completion: v.completion, expected: r.value, parts: r.parts }
    }),
  }
}

const doc = build()

const covered = new Set(doc.vectors.map((v) => v.branch))
const missing = RULE_BRANCHES.filter((b) => !covered.has(b))
if (missing.length) {
  process.stderr.write(`分支未覆盖：${missing.join(', ')}\n`)
  process.exit(2)
}

if (process.argv.includes('--check')) {
  if (!existsSync(OUT)) { process.stderr.write(`规格文件不存在：${OUT}\n`); process.exit(1) }
  const cur = readFileSync(OUT, 'utf8')
  const next = `${JSON.stringify(doc, null, 2)}\n`
  if (cur !== next) { process.stderr.write('规格已过时（规则改了但规格未重新生成）\n'); process.exit(1) }
  process.stdout.write(`reward-vectors 规格与实现一致（${doc.vectors.length} 条向量 / ${RULE_BRANCHES.length} 条分支）\n`)
} else {
  mkdirSync(dirname(OUT), { recursive: true })
  writeFileSync(OUT, `${JSON.stringify(doc, null, 2)}\n`)
  process.stdout.write(`写出 ${OUT}（${doc.vectors.length} 条向量）\n`)
}
