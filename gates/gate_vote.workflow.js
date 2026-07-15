export const meta = {
  name: 'gate-vote',
  description: 'Convene N independent blind judges on one artifact and tally PASS/REVISE/FAIL',
  phases: [
    { title: 'Judge', detail: 'N independent judges read the artifact and vote, blind to each other' },
  ],
}

// ---------------------------------------------------------------------------
// Independent validation gate. Already model-independent — do not add
// teacher/student-specific logic here. All instance-specific detail (which
// criteria apply) is passed in by the caller (the Director), typically pulled
// from that run's configs/*.yaml.
//
// Spawns N judge subagents (default 3) IN PARALLEL. Each runs in its own context
// (genuinely independent), reads the SAME artifact, and returns a structured vote.
// The Director tallies the result and records EVERY individual vote (so split
// votes are visible in the audit trail, not just unanimous tallies).
//
// args = {
//   gate:     string   e.g. "dft-label-judge-gate"
//   target:   string   e.g. "cell_023"  (what is being judged)
//   artifact: string   free text: the paths to read + what the artifact is
//   criteria: string[] the explicit gate criteria + thresholds, one per line
//   n:        number    judges (default 3)
//   rule:     string    "unanimous" (default) | "majority"
// }
//
// Returns { gate, target, n, rule, decision, tally, votes[] }.
// The Director appends the result to gates/coordination_votes.csv (per-judge)
// and coordination_log.csv (aggregate) after this workflow completes.
// ---------------------------------------------------------------------------

const gate     = args?.gate     || 'unnamed-gate'
const target   = args?.target   || 'unnamed-target'
const artifact = args?.artifact || '(no artifact description provided)'
const criteria = Array.isArray(args?.criteria) ? args.criteria : []
const N        = args?.n ?? 3
const rule     = args?.rule || 'unanimous'

if (!Number.isInteger(N) || N < 1) throw new Error(`n must be a positive integer; got ${N}`)
if (!['unanimous', 'majority'].includes(rule)) throw new Error(`unknown decision rule: ${rule}`)
if (criteria.length === 0) throw new Error('gate criteria must not be empty')

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

log(`Convening ${N} independent judges on gate "${gate}" / target "${target}" (rule=${rule})`)

const raw = await parallel(
  Array.from({ length: N }, (_, i) => () =>
    agent(judgePrompt(i), {
      label: `judge#${i + 1}:${target}`,
      phase: 'Judge',
      agentType: 'judge',
      schema: VERDICT_SCHEMA,
    }).then(v => ({ id: i + 1, ...v }))
  )
)

const votes = raw.filter(Boolean)
const tally = { PASS: 0, REVISE: 0, FAIL: 0 }
for (const v of votes) tally[v.verdict] = (tally[v.verdict] || 0) + 1

// Decision rule. Conservative escalation in both modes:
//   any FAIL          -> FAIL  (an invalid/unphysical artifact blocks regardless)
//   unanimous: all PASS -> PASS else REVISE
//   majority:  PASS is the strict majority (> n/2) and no FAIL -> PASS else REVISE
let decision
if (votes.length !== N) {
  // Fail closed: a missing/failed judge is not evidence of consensus.
  decision = 'REVISE'
} else if (tally.FAIL > 0) {
  decision = 'FAIL'
} else if (rule === 'majority') {
  decision = tally.PASS > N / 2 ? 'PASS' : 'REVISE'
} else {
  decision = tally.PASS === votes.length && votes.length > 0 ? 'PASS' : 'REVISE'
}

log(`Tally PASS=${tally.PASS} REVISE=${tally.REVISE} FAIL=${tally.FAIL} -> ${decision}`)

return { gate, target, criteria, requested_n: N, received_n: votes.length, rule, decision, tally, votes }
