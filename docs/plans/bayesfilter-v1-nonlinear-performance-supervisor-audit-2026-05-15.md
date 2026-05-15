# Supervisor Audit: BayesFilter V1 Nonlinear Performance Master Program

## Date

2026-05-15

## Scope

Codex is supervising the critical-review loop requested by the user for:

```text
docs/plans/bayesfilter-v1-nonlinear-performance-master-program-2026-05-15.md
```

Claude Code is used only as a bounded critical reviewer through:

```bash
bash scripts/claude_worker.sh --cwd /home/chakwong/BayesFilter --name <name> --model sonnet --effort high "<worker prompt>"
```

Claude is instructed not to edit files, not to run tests, not to run
benchmarks, not to run GPU/CUDA/NVIDIA probes, and not to run HMC diagnostics.

The loop limit is 5 reviews.

## Codex Pre-Review Audit

Codex inspected:

- current nonlinear implementation files;
- TensorFlow value and score filter structure;
- NumPy reference sigma-point and particle filters;
- current nonlinear CPU and GPU/XLA benchmark harnesses;
- existing benchmark and GPU gate plans;
- prior Model B/C Claude-supervised plan-review precedent;
- source-map conventions;
- local Claude worker wrapper behavior.

Pre-review risks identified:

- treating NumPy reference filters as XLA/GPU candidates;
- mixing correctness, branch, compile, and timing ledgers;
- letting dense one-step projection diagnostics become exact nonlinear
  likelihood evidence;
- omitting trusted GPU escalation from CPU/GPU benchmark rules;
- accepting Claude review findings without Codex audit;
- creating a plan too vague to execute.

Pre-review status:

- The draft master program explicitly separates production TensorFlow paths
  from NumPy reference paths.
- It defines NP0-NP7 phases, evidence contracts, veto diagnostics, artifacts,
  XLA gates, CPU/GPU trusted execution, and default-policy stop rules.

## Review Log

### Claude Round 1

Verdict: `REJECT`.

Material blockers:

- NP1-NP5 were long research-engineering phases, but the draft did not require
  phase-specific plan/result artifacts or template-equivalent run manifests
  before long ladders, sweeps, GPU benchmarks, or optimization-promotion runs.
- NP2 listed value-path algebraic optimizations without requiring a local
  derivation or executable preconditions for semantic equivalence.
- NP3 listed analytic score transformations without requiring a derivative
  proof obligation or naming the reference derivation artifact.
- NP4 did not require an authoritative XLA support matrix with static-shape,
  dynamic-shape, retracing, and unsupported-cell claim boundaries.
- NP5 required trusted GPU pre-probes but did not explicitly require trusted or
  escalated execution for the GPU benchmark commands themselves.

Codex audit:

- Accepted all five blockers as material.
- They align with the repo's scientific coding, mathematical-claim, and
  GPU/CUDA policies.

Revisions made:

- Added a required run manifest and phase-specific plan/result-note rule for
  NP1-NP5 and other meaningful runs.
- Added a mathematical pre-gate for NP2 value-path algebraic rewrites.
- Added a mathematical pre-gate for NP3 score-path transformations.
- Added an NP4 XLA support-matrix requirement.
- Tightened NP5 so GPU benchmark commands themselves require trusted or
  escalated execution.
- Added separate engineering correctness, numerical validity, and performance
  ledgers to NP7.

### Claude Round 2

Verdict: `ACCEPT`.

Material blockers:

- None.

Non-blocking suggestions:

- Record derivation or proof-obligation artifact paths in NP2/NP3 run manifests
  so mathematical pre-gates remain easy to audit.
- Clarify trusted versus escalated wording for NP5 artifacts.

Codex audit:

- Accepted both suggestions as useful auditability refinements, not blockers.

Revisions made:

- Added derivation/proof-obligation artifact path fields to the master run
  manifest and NP2/NP3 result-note requirements.
- Defined `escalated_sandbox`, `trusted_external`, and `nontrusted_sandbox`
  labels for NP5 GPU-visible commands.

### Claude Round 3

Verdict: `ACCEPT`.

Material blockers:

- None.

Non-blocking suggestions:

- None.

Codex audit:

- Accepted the convergence result.  The review loop stopped at round 3, within
  the user-requested maximum of 5 loops.

## Final Decision

Decision: `ACCEPT`.

The nonlinear performance master program is ready as a planning artifact.  It
does not execute benchmarks or authorize performance claims by itself; each
future phase still needs its phase-specific evidence contract, skeptical
audit, run manifest, result artifact, and continuation decision.

## Verification

No numerical tests, benchmarks, GPU probes, or HMC diagnostics were run during
this planning review.

Document checks:

```bash
python -c "import yaml; yaml.safe_load(open('docs/source_map.yml', encoding='utf-8')); print('source_map ok')"
```

Observed before final review:

```text
source_map ok
```
