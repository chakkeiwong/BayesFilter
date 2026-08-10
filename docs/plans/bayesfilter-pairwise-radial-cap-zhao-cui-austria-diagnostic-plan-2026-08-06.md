# Pairwise Radial-Cap Zhao-Cui/Austria Diagnostic Plan

Date: 2026-08-06

## Research intent ledger

- Main question: does a smooth per-particle RMS cap on the existing pairwise
  third/fourth-moment correction reduce tail dominance without destroying its
  Austria value/score stabilization?
- Candidate mechanism: after affine projection and global RMS normalization,
  replace each particle direction `v_n` by
  `v_n / sqrt(1 + mean(v_n^2) / c^2)`.
- Exact baseline: the current unbounded pairwise correction with the same
  source cloud, targets, correction steps, strength, seeds, backend, and
  arithmetic.
- Expected failure mode: a few particles have disproportionately large
  normalized correction directions or JVPs; a cap may suppress them but can
  also under-correct the pairwise moments or shift the finite likelihood.
- Promotion criterion: none. This is a mechanism diagnostic, not a default or
  HMC promotion run.
- Promotion veto: any manual-JVP parity failure, loss of mean/covariance
  restoration, non-finite row, or failure to recover the existing route when
  the cap is disabled.
- Continuation veto: the implementation computes a different cap from the
  declared formula, the reference fixture is invalid, or the Austria baseline
  cannot reproduce a finite working pairwise run.
- Repair trigger: localized shape/JVP/XLA plumbing failure that leaves the
  formula, target, controls, data, and budget unchanged.
- Explanatory diagnostics: maximum pre/post-cap particle RMS, cap activation,
  pairwise residual, value, score, per-seed displacement, and runtime.
- Nonclaims: this run cannot repair or validate the recursively fitted Austria
  Zhao-Cui TT teacher, establish an exact Austria score, establish posterior
  correctness, or select a repository default.

## Evidence contract

The first rung uses a deterministic Zhao-Cui squared-TT shape-target fixture
and checks the executed reset map, derivative, and affine restoration. The
second rung uses the working empirical-target Austria pairwise route at
`T=20,N=1008`; it answers whether the same reset-side tail mechanism helps the
actual Austria value/score computation. The Austria Zhao-Cui recursive teacher
is not used because its existing TT-fit validity veto fires before a reset can
be evaluated.

The primary pass/fail criterion is engineering correctness: exact disabled-cap
parity, manual JVP agreement with an independent TensorFlow forward
accumulator, and mean/covariance restoration. Austria statistics are
descriptive only. A cap is mechanically promising if it reduces maximum
per-particle direction RMS while retaining finite values/scores and not
recreating diagonal-route score explosions. No three-seed ranking will be
claimed.

Artifacts will be written under the unique root
`docs/benchmarks/artifacts/pairwise_radial_cap_diagnostic_20260806/`.

## Default and assumption audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Caps `inf,8,4,2` | hypothesis grid | spans no, light, medium, and strong intervention in globally normalized direction units | all finite caps may be inactive or overly strong | record pre/post RMS and activation |
| Existing pairwise `steps=4,strength=0.02` | frozen historical baseline | July Austria arm produced 16/16 finite rows and large score-SD reduction | stale relative to other later controls | this run makes only a within-route cap comparison |
| Three common Austria seeds | diagnostic convenience | bounded compute and includes existing common seeds | insufficient ranking uncertainty | report each row; forbid superiority claims |
| Empirical pairwise targets for Austria | working baseline, not Zhao-Cui teacher | isolates reset tail behavior on the model where it matters | cannot establish recursive TT feasibility | state target owner in artifact/result |
| FP32/TF32/GPU/XLA | historical Austria execution scope | matches the prior successful pairwise evidence | arithmetic may interact with cap | all arms share arithmetic; no arithmetic attribution |

## Skeptical pre-execution audit

- Wrong baseline: avoided by comparing against the exact current unbounded
  pairwise route, not diagonal-only or SGQF.
- Proxy promotion: tail RMS and residuals are explanatory; value and score are
  reported but three seeds cannot rank stochastic arms.
- Missing stop conditions: parity/restoration/non-finite failures stop the
  ladder; ordinary candidate underperformance does not invalidate the harness.
- Unfair comparison: every cap shares particles, seeds, targets, reset counts,
  strength, observations, and backend.
- Hidden target change: the cap changes the executed reset map but not the raw
  moment targets. The resulting finite likelihood is therefore a different
  finite approximation and will be labeled as such.
- Stale context: the recursive Austria Zhao-Cui TT teacher remains blocked;
  this plan does not claim otherwise.
- Environment mismatch: focused correctness runs are deliberate CPU-hidden
  references; the Austria diagnostic is escalated GPU/XLA with memory growth.
- Artifact adequacy: per-arm/per-seed value, score, validity, shape residual,
  displacement and cap diagnostics answer the bounded mechanism question.

Audit verdict: pass for the bounded diagnostic after separating the Zhao-Cui
fixture from the empirical-target Austria run.

## Execution budget

- One focused CPU-hidden test command, up to two localized repair retries.
- One escalated GPU/XLA smoke and one three-seed cap ladder, up to one localized
  infrastructure retry.
- Expected total GPU time: below five minutes.
- Stop if any run would require changing the target, data, particle count,
  method formula, or environment.
