# Hands-Off Claude Audit Handoff: HMC Tuning Interface Documentation Plan

Date: 2026-08-27

Status: `READY_FOR_READ_ONLY_BOUNDED_REVIEW`

Plan under review:
`/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-hmc-tuning-interface-documentation-and-verification-plan-2026-08-27.md`

Baseline commit: `553208502e2e43e6883ad9467381eb5c3e82867a`

## Audit Goal

Determine whether the plan can produce a technically correct, falsifiable, and
agent-usable description of BayesFilter HMC tuning. Audit the plan; do not
implement it. The review must distinguish route registry authority, actual
runner mechanics, final admission behavior, and scientific claims.

The required standard is not whether the proposal sounds coherent. The plan
must cause a false statement about mass tuning, trajectory selection, fixed
transport, neural-force mechanics, R-hat/ESS admission, or artifact authority
to fail an executable check.

## Exact Initial Prompt

Send this prompt without a path bundle or pasted source:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
/home/ubuntu/python/BayesFilter/docs/plans/bayesfilter-hmc-tuning-interface-documentation-and-verification-plan-2026-08-27.md
Do not edit, run commands, launch agents, or review the whole repo. Question: Is
this plan technically and scientifically sufficient to produce a correct,
falsifiable, agent-usable account of BayesFilter's HMC tuning interfaces,
including the ordinary, fixed-transport, and neural-force boundaries? Report
findings in severity order with exact plan line anchors, identify unsupported
defaults or claims, and state the smallest required correction. End with
VERDICT: AGREE or VERDICT: REVISE.
```

This one-path prompt is mandatory for the first attempt. Do not send this memo,
an artifact packet, multiple source files, or a repo-wide request with it.

## Staged Evidence Protocol

The plan explicitly cites exact evidence paths and baseline line anchors. Claude
may inspect a cited line only when needed to decide a plan claim. It must not
expand into a whole-repository review.

If one cited anchor is insufficient, Claude must name one next exact path and
the question that path will answer. The supervisor then sends a new bounded
read-only prompt for only that path. Repeat one path at a time. Do not infer
agreement from a timeout, partial response, or request for more context.

Likely evidence anchors, to be supplied only on request, are:

- `/home/ubuntu/python/BayesFilter/bayesfilter/inference/tuning_contract.py` for
  route role and artifact authority;
- `/home/ubuntu/python/BayesFilter/bayesfilter/inference/hmc_kernel_tuning.py`
  for the ordinary public signature and internal runner hook;
- `/home/ubuntu/python/BayesFilter/bayesfilter/inference/fixed_transport_hmc_tuning_tf.py`
  for fixed-transport prerequisites and runner binding;
- `/home/ubuntu/python/BayesFilter/bayesfilter/inference/neural_force_hmc.py`
  for fixed-mass/fixed-`L` mechanics and coordinate fallback;
- `/home/ubuntu/python/BayesFilter/docs/chapters/ch21_hmc_for_state_space.tex`
  for existing monograph claims;
- `/home/ubuntu/python/BayesFilter/docs/audits/bayesfilter-hmc-tuning-full-audit-2026-08-22.md`
  for the reported R-hat and ESS findings; and
- one exact downstream policy, lock, or test path only if a finding depends on
  consumer compatibility.

These paths are a routing index, not authorization to read all of them at once.

## Required Audit Questions

Claude must test the plan against each question below.

1. Does the baseline evidence support exactly two active artifact-authority
   tuners, without treating registry membership as proof of behavior?
2. Does the plan correctly distinguish an ordinary tuner, a fixed nonlinear
   transport tuner, an internal stage helper, and a low-level chain runner?
3. Is `tune_fixed_transport_hmc_kernel` prevented from becoming a generic
   arbitrary-force injection escape hatch?
4. Is the current neural-force limitation stated plainly: direct execution
   fixes mass coordinates and `L`, optionally adapts epsilon, and may use the
   `direct_fixed_transport_z` identity fallback?
5. Can the proposed typed runner binding prove that the same algorithm,
   coordinates, force, exact endpoint target, mass, `L`, epsilon, telemetry,
   and identity pass through every canonical stage?
6. Are there adequate rejection conditions for abandoning the two-route
   compatibility design and writing a separate API decision?
7. Do tests independently verify route identity, stage mechanics, fresh
   verification, and final artifact admission?
8. Will a forced verifier failure, including the reported R-hat-cap case,
   prevent a final handoff?
9. Does the ESS work characterize current behavior without inventing an
   unsupported default threshold?
10. Can generated documentation drift or a deliberately false prose claim be
    detected automatically?
11. Are numeric thresholds, retry counts, build allowances, and execution
    budgets classified by provenance rather than repetition?
12. Are target correctness, posterior convergence, sampler performance, and
    default readiness kept outside the documentation pass claim?
13. Does the downstream phase respect repository pins and owner decision
    boundaries?
14. Do the proposed commands answer separate questions, run in a valid
    CPU-hidden environment, and preserve the no-GPU/non-scientific scope?
15. Are tracked claim-supporting generated files separated from ignored build
    debris, with a zero-visible-untracked-files gate?

## Findings Standard

Use these classifications directly:

- `BLOCKER`: the plan could publish a materially false interface or admission
  claim, change a mathematical target, or cross a required owner boundary.
- `MAJOR`: a required falsification test, provenance field, stop condition, or
  consumer boundary is absent or ineffective.
- `MINOR`: clarity or maintainability issue that cannot change the substantive
  verdict.

For each finding provide:

```text
[SEVERITY] Short title
Plan anchor: <exact line or section>
Claimed target: <what the plan says it will establish>
Actual support: <correct / wrong relative to target / unsupported / not checked / heuristic only>
Evidence: <cited inspected path and line, or state that evidence was not inspected>
Consequence: <how the documentation or procedure could become wrong>
Smallest required correction: <bounded edit or test>
```

Do not soften a mismatch as merely "potentially confusing" when the documented
quantity differs from the implementation. Call it wrong relative to the stated
target. Do not approve an unexamined inherited default because it appears in an
older plan or audit.

## Verdict Rule

`VERDICT: AGREE` is allowed only when:

- no blocker or major finding remains;
- every material interface and admission claim can be falsified;
- the neural-force design is labeled as a hypothesis until its tests pass;
- R-hat and ESS intent is not confused with current behavior;
- every material numeric choice has provenance or is explicitly unresolved;
- commands and artifacts answer the stated engineering question; and
- downstream locks and default-policy decisions remain outside implicit
  authorization.

Otherwise return `VERDICT: REVISE`. If evidence is missing, classify the claim
`not checked`, name the one next exact path needed, and still end with a verdict.

## Required Response Shape

```text
FINDINGS
<severity-ordered findings, or "No material findings">

OPEN QUESTIONS OR UNCHECKED CLAIMS
<items, each with one exact next path if inspection is required>

AUDIT COVERAGE
<which of the 15 required questions were resolved>

VERDICT: AGREE|REVISE
```

No edits, commands, implementation, experiments, agents, or external messages
are authorized by this handoff. The supervisor owns finding adjudication and
any later implementation decision.
