/**
 * 冻结判据 `aggregateSeeds` 的 O5 色散估计量回归（`o5-dispersion@2`，G1）。
 *
 * 保护四件事：
 *   ① `c4(k)` 的数值正确（与闭式/BigInt 精确值一致）；
 *   ② `tau = 0.5/√π`，且它是**使 k=3 上「旧判失败 ⟹ 新判失败」成立的最大阈值**（实现级支配）；
 *   ③ 旧口径（`o5-dispersion@1`）的字段与判决**逐字保留**（可审计、可回退）；
 *   ④ `n < 3` 仍然 fail-closed。
 *
 * 只 import 冻结参考实现，不重写判据。
 */
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'

export const NAME = 'o5-dispersion（G1 估计量变更：极差 → 正态无偏标准差）'

const HERE = dirname(fileURLToPath(import.meta.url))
const CHECKS = join(HERE, '..', '..', '..', 'checks', 'overfit-criterion.mjs')

export async function run(t) {
  const m = await import(CHECKS)

  // ① c4
  t.assert(Math.abs(m.c4(2) - 0.7978845608028654) < 1e-15, `c4(2)=0.7978845608028654（实得 ${m.c4(2)}）`)
  t.assert(Math.abs(m.c4(3) - Math.sqrt(Math.PI) / 2) < 1e-15, `c4(3)=√π/2（实得 ${m.c4(3)}）`)
  t.assert(Math.abs(m.c4(4) - 0.9213177319235613) < 1e-15, `c4(4)=0.9213177319235613（实得 ${m.c4(4)}）`)
  t.assert(Math.abs(m.c4(10) - 0.972659274121588) < 1e-14, `c4(10)=0.972659274121588（实得 ${m.c4(10)}）`)
  t.assert(m.c4(1) === null && m.c4(2.5) === null, 'c4(k) 对 k<2 / 非整数返回 null（fail-closed）')

  // ② tau
  t.assert(Math.abs(m.DEFAULT_TAU_ROBUST - 0.5 / Math.sqrt(Math.PI)) < 1e-16, `tau = 0.5/√π（实得 ${m.DEFAULT_TAU_ROBUST}）`)
  t.assert(Math.abs(m.D3_RANGE - 3 / Math.sqrt(Math.PI)) < 1e-15, 'd_3 = 3/√π（闭式）')
  t.assert(Math.abs(m.DEFAULT_TAU_ROBUST - 0.5 * m.SD_OVER_RANGE_FLOOR_K3) < 1e-16, 'tau = 0.5 ×（k=3 的 S/R 下确界）')

  // ② 实现级支配：k=3 上「旧判失败 ⟹ 新判失败」，随机 50k 组
  let s = 20261001
  const rnd = () => { s = (s * 1103515245 + 12345) % 2147483648; return s / 2147483648 }
  let failToPass = 0, passToFail = 0
  for (let i = 0; i < 50000; i += 1) {
    const c = (rnd() - 0.5) * 10
    const v = [c + rnd(), c + rnd(), c + rnd()]
    const mu = (v[0] + v[1] + v[2]) / 3
    if (Math.abs(mu) < 1e-9) continue
    const R = Math.max(...v) - Math.min(...v)
    const sdv = Math.sqrt(((v[0] - mu) ** 2 + (v[1] - mu) ** 2 + (v[2] - mu) ** 2) / 2)
    const legacyFail = R / Math.abs(mu) > m.DEFAULT_R_OVER_MU_MAX
    const newFail = (sdv / m.c4(3)) / Math.abs(mu) > m.DEFAULT_TAU_ROBUST
    if (legacyFail && !newFail) failToPass += 1
    if (!legacyFail && newFail) passToFail += 1
  }
  t.assert(failToPass === 0, `k=3 实现级支配：旧判失败 ⇒ 新判失败（50k 随机组里 fail→pass 数 = ${failToPass}）`)
  t.assert(passToFail > 0, `k=3 上确实存在"新更严"的样本（pass→fail 数 = ${passToFail} > 0）`)

  // ③ legacy 字段逐字保留 + 新字段存在（用真实预登记值）
  const vals = [0.044068433174335946, 0.025602524083426864, 0.04838616405456683]
  const r = m.aggregateSeeds(vals.map((value, i) => ({ value, se: 0, seed: i })))
  t.assert(r.R === 0.02278363997113997, `legacy R 逐位保留（实得 ${r.R}）`)
  t.assert(Math.abs(r.rOverMu - 0.578964819348695) < 1e-15, `legacy rOverMu 逐位保留（实得 ${r.rOverMu}）`)
  t.assert(r.rOverMuMax === 0.5 && r.legacy.rOverMuMax === 0.5, 'legacy 阈值仍是 0.5（未改）')
  t.assert(r.legacy.criterionId === m.LEGACY_CRITERION_ID && r.legacy.stable === false, 'legacy 判决被完整保留')
  t.assert(r.criterionId === m.CRITERION_ID && r.estimator.id === m.ESTIMATOR_ID, '输出带显式的 criterion/estimator 标识')
  t.assert(Math.abs(r.sigmaSeedUnbiased - r.sigmaSeed / m.c4(3)) < 1e-18, 'sigmaSeedUnbiased = sigmaSeed/c4(k)')
  t.assert(r.rOverMuRobust > r.tau && r.stable === false, '本资产主臂：新判据仍判「测量不成立」（未把失败的案例翻绿）')
  t.assert(r.verdictAgreesWithLegacy === false && r.legacy.verdict !== r.verdict, '两个口径的判决各自可读（可审计）')

  // ④ fail-closed 未退化
  t.assert(m.aggregateSeeds([], {}).error === 'E_SEEDS_LT_3', 'n<3 仍拒绝出流程级结论')
  t.assert(m.aggregateSeeds([{ value: 1, se: 0 }, { value: 1, se: 0 }], {}).error === 'E_SEEDS_LT_3', 'n=2 也拒绝')
  // 符号翻转两条判据逐字同款
  const flip = m.aggregateSeeds([{ value: -0.1, se: 0 }, { value: 0.2, se: 0 }, { value: 0.3, se: 0 }])
  t.assert(flip.signFlip === true && flip.verdict === '测量不成立（符号翻转）' && flip.legacy.verdict === flip.verdict,
    '符号翻转分支未被改动（新旧判决相同）')
}
