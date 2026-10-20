#!/usr/bin/env node
/**
 * aggregate-rl-seeds —— **只做聚合、不碰 GPU**：把 `out/rl-multiseed.json` 里逐种子的配对量
 * 交给冻结契约的唯一参考实现 `inference-training/checks/overfit-criterion.mjs#aggregateSeeds`。
 *
 * 存在的理由（诚实登记）：`rl-multiseed.mjs` 的首版把"重测的既有 run（seed=20260929）"
 * 与新种子混进了同一个 `aggregate`，于是 n=4 却被描述成"预登记的 3 个新种子"——**口径错位**。
 * 该口径已在 `rl-multiseed.mjs` 里修正，但重跑种子要再烧 GPU；故本脚本以**同样输入**重算
 * 两份口径（主=n 个新种子 / 旁证=+既有 run），**聚合公式一个字都不改**（import，不重写）。
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

const HERE = dirname(fileURLToPath(import.meta.url))
const TRAINING = join(HERE, '..', '..', '..')
const EVIDENCE = process.argv[2] || join(TRAINING, 'out', 'rl-multiseed.json')
const OUT = process.argv[3] || join(TRAINING, 'out', 'rl-seed-aggregate.json')

const { aggregateSeeds } = await import(join(TRAINING, 'checks', 'overfit-criterion.mjs'))

const j = JSON.parse(readFileSync(EVIDENCE, 'utf8'))
const rows = (j.perSeed || [])
  .filter((x) => typeof x?.preferenceGate?.paired?.meanDelta === 'number')
  .map((x) => ({
    kind: x.kind, seed: x.seed, runId: x.runId,
    value: x.preferenceGate.paired.meanDelta,
    se: x.preferenceGate.paired.se,
    validityOk: x.preferenceGate.validityOk === true,
    truncationBase: x.truncation?.base ?? null,
    truncationPolicy: x.truncation?.policy ?? null,
  }))

const prim = rows.filter((r) => r.kind === 'new-seed')
const all = rows

const out = {
  schema: 'ai-infra/rl-seed-aggregate@1',
  generatedAt: new Date().toISOString(),
  source: EVIDENCE,
  aggregator: 'inference-training/checks/overfit-criterion.mjs#aggregateSeeds（import，未重写）',
  metric: 'preferenceGate.pairedSignificance.meanDelta（留出 prompt 逐条配对奖励差 mean(d)；higher-is-better）',
  criterion: '用户预登记三条：①全部同号→稳定 ②符号翻转→测量不成立 ③R/μ>50%→测量不成立需重设计',
  primary: {
    role: '主结论：**预登记的新种子**',
    seeds: prim.map((r) => r.seed),
    rows: prim,
    result: prim.length >= 3 ? aggregateSeeds(prim) : { error: 'E_SEEDS_LT_3', n: prim.length },
  },
  supplementary: {
    role: '旁证：新种子 + 已验收 run 的重测（seed=20260929，同一训练配置）；**不作为主结论**',
    seeds: all.map((r) => r.seed),
    rows: all,
    result: all.length >= 3 ? aggregateSeeds(all) : null,
  },
  notes: [
    'z 与阈值未参与聚合：aggregateSeeds 用 z=1.96（默认，未传参覆盖）；判据三条逐字沿用用户预登记。',
    'distinguishableFromZero 只作补充信息，**不参与 verdict**（参考实现文件头已写明理由）。',
  ],
}
writeFileSync(OUT, `${JSON.stringify(out, null, 2)}\n`)
process.stdout.write(`${JSON.stringify(out.primary.result, null, 2)}\n`)
process.stdout.write(`\n[supplementary] ${JSON.stringify({ n: all.length, mu: out.supplementary.result?.mu, R: out.supplementary.result?.R, rOverMu: out.supplementary.result?.rOverMu, verdict: out.supplementary.result?.verdict })}\n`)
process.stdout.write(`\n落盘 ${OUT}\n`)
