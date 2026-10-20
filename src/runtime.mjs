/**
 * runtime —— 共享状态根 / 依赖封装（薄）。
 *
 * 纪律：**不绕开共享库契约直接实现共享逻辑**。
 * 因此这里**只做装配与路径解析**，所有实质逻辑一律转发给冻结的共享库：
 *   状态方言与写入      → @xiciyang/state-core
 *   审计哈希链          → @xiciyang/audit-core
 *   沙箱能力与降级      → @xiciyang/sandbox-core
 *   测量与判据          → @xiciyang/testing-core
 *   公共训练循环/限额/产物断言 → @xiciyang/training-core
 *   prompt 集哈希与去泄漏 → @xiciyang/data-core
 */
import { join } from 'node:path'
import { homedir } from 'node:os'
import { promises as fs } from 'node:fs'

import { appendAudit as coreAppend, verifyAllChains, digestArgs, KNOWN_PACKAGES, CHAIN_VARIANTS } from '@xiciyang/audit-core'
import { parse, stringify, readText, withOptimisticWrite, validateAgainstSchema, assertSchemaSupported, contentHash } from '@xiciyang/state-core'
import { probeTiers, requireTier, DEFAULT_DEGRADATION } from '@xiciyang/sandbox-core'
import { PACKAGE_VERSION, REGISTRY_SCHEMA, ToolError, ERROR_CODES } from './version.mjs'

export const PKG = 'training-rl'

export function rootFromEnv(env = process.env) {
  return env.TRAINING_RL_ROOT || join(homedir(), '.ai-infra')
}

export function paths(root = rootFromEnv()) {
  return {
    root,
    stateDir: join(root, 'state'),
    registryFile: join(root, 'state', 'rl-registry.yaml'),
    promptsDir: join(root, 'state', 'rl-prompts'),
    runsDir: join(root, 'run', 'training-rl'),
    reportsDir: join(root, 'reports', 'training-rl'),
    auditDir: join(root, 'logs', 'audit', PKG),
    cacheDir: join(root, 'cache', 'training-rl'),
    confirmFile: join(root, 'state', 'training-rl-confirmations.json'),
  }
}

/* ------------------------------------------------------------------ 审计 */

/**
 * 追加审计事件。**事件清单**：
 *   prepare.begin / prepare.end
 *   train.begin / train.progress / train.checkpoint / train.kl-accounting / train.end / train.failed
 *   validate.begin / validate.end / validate.failed
 *   rollback.begin / rollback.end
 * 每条都只记**摘要**（参数摘要 + 结果码 + 每步 KL/奖励的聚合），不记样本内容。
 */
export async function audit(root, entry, { pkgVersion = PACKAGE_VERSION, date = new Date() } = {}) {
  return coreAppend({ root, pkg: PKG, pkgVersion, entry, date })
}

export async function auditVerify(root) {
  return verifyAllChains({ root, pkg: PKG })
}

export { digestArgs, CHAIN_VARIANTS, KNOWN_PACKAGES }

/* ------------------------------------------------------------------ 状态 */

export const REGISTRY_SCHEMA_ID = REGISTRY_SCHEMA

export async function readRegistry(root, schemaPath = null) {
  const p = paths(root).registryFile
  const text = await readText(p, null)
  if (text === null) return { version: 1, updatedAt: null, pkg: PACKAGE_VERSION, runs: [], _hash: null }
  const doc = parse(text)
  if (schemaPath) {
    const schema = JSON.parse(await fs.readFile(schemaPath, 'utf8'))
    assertSchemaSupported(schema)
    const errs = validateAgainstSchema(doc, schema)
    if (errs.length > 0) throw new ToolError(ERROR_CODES.E_STATE_CONFLICT, { hint: `rl-registry 不符合 schema：${errs.slice(0, 4).join('；')}`, details: errs })
  }
  return { ...doc, _hash: contentHash(text) }
}

/** 乐观并发写（CAS）—— 走 state-core，不自己写文件。 */
export async function updateRegistry(root, mutator, { schemaPath = null, expectedHash = undefined } = {}) {
  const p = paths(root).registryFile
  const outcome = await withOptimisticWrite(p, async (doc) => {
    const base = doc ?? { version: 1, updatedAt: null, pkg: PACKAGE_VERSION, runs: [] }
    const next = await mutator(base)
    next.updatedAt = new Date().toISOString()
    next.pkg = PACKAGE_VERSION
    next.version = 1
    if (schemaPath) {
      const schema = JSON.parse(await fs.readFile(schemaPath, 'utf8'))
      assertSchemaSupported(schema)
      const errs = validateAgainstSchema(next, schema)
      if (errs.length > 0) throw new ToolError(ERROR_CODES.E_STATE_CONFLICT, { hint: `写前 schema 校验失败：${errs.slice(0, 4).join('；')}`, details: errs })
    }
    return stringify(next)
  }, expectedHash === undefined ? {} : { expectedHash }).catch((e) => {
    // 把共享库的错误码翻译成本包闭集里的码（调用方只看一套码）
    if (e && e.code === 'ERR_STATE_CONFLICT') {
      throw new ToolError(ERROR_CODES.E_STATE_CONFLICT, { hint: e.message, remediation: e.remediation, details: { expectedHash } })
    }
    throw e
  })
  return { ...outcome, doc: parse(await readText(p, '')) }
}

/* ------------------------------------------------------------------ 沙箱 */

export async function sandboxProbe(opts = {}) {
  return probeTiers({ image: opts.image || 'alpine:3.20', timeoutMs: opts.timeoutMs || 60000 })
}

export async function requireContainerSandbox(probe) {
  try {
    return requireTier('container', probe)
  } catch (e) {
    throw new ToolError(ERROR_CODES.E_SANDBOX_UNAVAILABLE, {
      hint: `训练需要容器沙箱但本机不可用：${e.message}。补救：${e.remediation}`,
      details: { tier: 'container', decision: e.decision, evidence: e.evidence },
      recoverable: false,
    })
  }
}

export { DEFAULT_DEGRADATION, contentHash }

/** 目录/文件存在性（不信"应该存在"）。 */
export async function exists(p) {
  try { await fs.access(p); return true } catch { return false }
}

export async function mkdirp(d) {
  await fs.mkdir(d, { recursive: true, mode: 0o700 })
  return d
}

export function nowId(prefix = 'rl') {
  const d = new Date()
  const p = (n) => String(n).padStart(2, '0')
  return `${prefix}_${d.getUTCFullYear()}${p(d.getUTCMonth() + 1)}${p(d.getUTCDate())}${p(d.getUTCHours())}${p(d.getUTCMinutes())}${p(d.getUTCSeconds())}_${Math.random().toString(36).slice(2, 6)}`
}
