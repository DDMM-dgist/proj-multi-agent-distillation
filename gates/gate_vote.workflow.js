export const meta = {
  name: 'gate-vote',
  description: 'Convene three separate-context, mutually blind judges and tally PASS/REVISE/FAIL',
  phases: [
    { title: 'Judge', detail: 'Three mutually blind judge instances read the same artifact' },
  ],
}

// ---------------------------------------------------------------------------
// Separate-context validation gate. Already model-independent — do not add
// teacher/student-specific logic here. All instance-specific detail (which
// criteria apply) is passed in by the caller (the Director), typically pulled
// from that run's configs/*.yaml.
//
// Spawns exactly three judge subagents IN PARALLEL. Each runs in its own context
// and mutually blind, reads the SAME artifact, and returns a structured vote.
// The Director tallies the result and records EVERY individual vote (so split
// votes are visible in the audit trail, not just unanimous tallies).
//
// args = {
//   gate:     string   e.g. "dft-label-judge-gate"
//   target:   string   e.g. "cell_023"  (what is being judged)
//   artifact: string   free text: the paths to read + what the artifact is
//   artifact_sha256: object  exact path -> SHA-256 map from controller gate-context
//   criteria: string[] the explicit gate criteria + thresholds, one per line
//   n:        number    must be 3 (matches the persistent controller)
//   rule:     string    must be "unanimous"
// }
//
// Returns { gate, target, criteria, artifact_sha256, decision, tally, votes[] }.
// The Director stores the returned bundle under the run's gates/ directory
// and records it through workflow.controller gate --votes.
// ---------------------------------------------------------------------------

const gate     = args?.gate     || 'unnamed-gate'
const target   = args?.target   || 'unnamed-target'
const artifact = args?.artifact || '(no artifact description provided)'
const artifactSha256 = args?.artifact_sha256
const criteria = Array.isArray(args?.criteria) ? args.criteria : []
const N        = args?.n ?? 3
const rule     = args?.rule || 'unanimous'

if (N !== 3) throw new Error(`the persistent controller requires exactly 3 judges; got ${N}`)
if (rule !== 'unanimous') throw new Error(`the persistent controller requires unanimous rule; got ${rule}`)
if (criteria.length === 0) throw new Error('gate criteria must not be empty')
if (!artifactSha256 || typeof artifactSha256 !== 'object' || Array.isArray(artifactSha256) ||
    Object.keys(artifactSha256).length === 0) {
  throw new Error('artifact_sha256 must be the non-empty map from controller gate-context')
}

const VERDICT_SCHEMA = {
  type: 'object',
  additionalProperties: false,
  properties: {
    verdict: { type: 'string', enum: ['PASS', 'REVISE', 'FAIL'] },
    criteria_checked: {
      type: 'array',
      items: {
        type: 'object',
        additionalProperties: false,
        properties: {
          criterion: { type: 'string' },
          value_read: { type: 'string' },
          ok: { type: 'boolean' },
        },
        required: ['criterion', 'ok'],
      },
    },
    rationale: { type: 'string' },
    required_fix: { type: 'string' },
  },
  required: ['verdict', 'criteria_checked', 'rationale', 'required_fix'],
}

const criteriaBlock = criteria.map((c, i) => `  ${i + 1}. ${c}`).join('\n')

function judgePrompt(idx) {
  return [
    `You are judge #${idx + 1} of ${N} on an INDEPENDENT validation committee.`,
    `You are blind to the other judges. Vote alone, from the evidence.`,
    ``,
    `## Gate`,
    gate,
    ``,
    `## Target (artifact under review)`,
    target,
    ``,
    `## Where the artifact is / what it contains`,
    artifact,
    ``,
    `## Criteria you MUST apply (vote PASS only if ALL are demonstrably met)`,
    criteriaBlock,
    ``,
    `## Instructions`,
    `- Actually OPEN and READ the artifact files above (Read/Grep/Bash). Quote the`,
    `  real numbers you read for each criterion.`,
    `- Conservative default: if a criterion is not demonstrably met, vote REVISE`,
    `  (fixable) or FAIL (invalid/unphysical). "Probably fine" is REVISE.`,
    `- Return your vote via StructuredOutput: verdict, criteria_checked[], rationale,`,
    `  required_fix (only if REVISE/FAIL).`,
  ].join('\n')
}

function invocationFailureVote(idx, error) {
  return {
    id: idx + 1,
    verdict: 'REVISE',
    criteria_checked: criteria.map(criterion => ({
      criterion,
      value_read: 'judge invocation failed',
      ok: false,
    })),
    rationale: `Judge invocation failed: ${String(error?.message || error)}`,
    required_fix: 'Re-run the failed judge invocation before recording PASS.',
  }
}

log(`Convening ${N} mutually blind judge instances on gate "${gate}" / target "${target}" (rule=${rule})`)

const raw = await parallel(
  Array.from({ length: N }, (_, i) => () =>
    agent(judgePrompt(i), {
      label: `judge#${i + 1}:${target}`,
      phase: 'Judge',
      agentType: 'judge',
      schema: VERDICT_SCHEMA,
    }).then(v => ({ id: i + 1, ...v }))
      .catch(error => invocationFailureVote(i, error))
  )
)

const votes = raw.filter(Boolean)
const tally = { PASS: 0, REVISE: 0, FAIL: 0 }
for (const v of votes) tally[v.verdict] = (tally[v.verdict] || 0) + 1

// Decision rule. Conservative escalation in both modes:
//   any FAIL          -> FAIL  (an invalid/unphysical artifact blocks regardless)
//   unanimous: all PASS -> PASS else REVISE
let decision
if (votes.length !== N) {
  // Fail closed: a missing/failed judge is not evidence of consensus.
  decision = 'REVISE'
} else if (tally.FAIL > 0) {
  decision = 'FAIL'
} else {
  decision = tally.PASS === votes.length && votes.length > 0 ? 'PASS' : 'REVISE'
}

log(`Tally PASS=${tally.PASS} REVISE=${tally.REVISE} FAIL=${tally.FAIL} -> ${decision}`)

return {
  gate, target, criteria, artifact_sha256: artifactSha256,
  requested_n: N, received_n: votes.length, rule, decision, tally, votes,
}
