# IE6 learned-OT residual result

## Outcome

- Master-program exit label: `ie_phase_deferred_with_recorded_reason`
- Local exit label: `ie6_learned_ot_residual_deferred_no_artifact`
- Phase status: `deferred`

## Skeptical audit before execution

- Artifact gate audit: no approved pre-existing teacher/student artifact with provenance was identified.
- Substitution audit: IE6 did not invent an analytic teacher/student substitute.
- Training boundary audit: no optimizer steps, checkpoint mutation, or network training occurred.
- Environment audit: the runner fixes `CUDA_VISIBLE_DEVICES=-1` before local imports and records a CPU-only manifest.

## Research-intent ledger

- Main question: can IE6 evaluate learned/amortized OT residuals against an approved teacher/student artifact?
- Result: no; the required artifact provenance is absent, so residual execution is deferred.
- Promotion criterion: not triggered because no approved artifact exists.
- Continuation veto: triggered for residual execution; future execution requires approved provenance-bearing artifacts.
- Repair trigger: missing artifact provenance.
- Not concluded: no learned-OT, neural-OT, surrogate-quality, posterior, production, banking, model-risk, or readiness claim follows.

## Evidence contract

| Diagnostic | Comparator | Status | Source support |
| --- | --- | --- | --- |
| `learned_map_residual` | `no_approved_teacher_student_artifact` | `deferred` | `bibliography_spine_only` |

## Decision table

| Diagnostic | Status | Primary criterion | Promotion veto | Continuation veto | Repair trigger |
| --- | --- | --- | --- | --- | --- |
| `learned_map_residual` | `deferred` | `not_triggered` | `not_triggered` | `fail` | `missing artifact provenance` |

## Coverage semantics

- IE3 through IE5 diagnostics are carried forward as `passed`.
- `learned_map_residual` is `deferred`.
- IE7 and IE8 remain `missing`.

## Artifact list

- `experiments/dpf_monograph_evidence/reports/outputs/learned_ot_residual.json`
- `experiments/dpf_monograph_evidence/reports/learned-ot-residual-result.md`
- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie6-learned-ot-result-2026-05-16.md`

## Run manifest

- command: `python -m experiments.dpf_monograph_evidence.runners.run_learned_ot_residual`
- branch: `main`
- commit: `0684d6fe4350664c34826e83df59b2413d7be89c`
- python: `3.13.13`
- cpu_only: `True`
- pre-import CUDA_VISIBLE_DEVICES: `-1`
- seed policy: `deterministic_no_rng_deferred_no_artifact`

## Required non-implication text

IE6 learned-OT diagnostics were deferred because no approved pre-existing teacher/student artifact with provenance was available. This deferral does not validate or invalidate learned OT, neural OT, surrogate-map quality, posterior quality, production bayesfilter code, banking use, model-risk use, or production readiness.

## Post-run red-team note

The strongest reason not to over-read IE6 is that a clean deferral records only the absence of an approved teacher/student artifact. It says nothing about whether learned OT would pass or fail under a provenance-bearing artifact.

## Next-phase justification

IE6 deferral does not block IE7 because IE7 uses a fixed scalar HMC-facing target fixture and does not depend on learned-OT residual execution. IE7 must not interpret IE6 as learned-OT validation.
