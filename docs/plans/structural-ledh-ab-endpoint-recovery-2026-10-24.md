# Model B endpoint recovery, 24 October 2026

The master stopped after Model A completed 2,000 warmup and 3,000 retained
draws per chain. Model B completed training and nomination, then archived its
first 250 warmup transitions before JSON serialization failed on positive
infinity. The archived states are finite; four proposed endpoints have numerical
failure flags and were rejected. There are no divergence flags. This is not
evidence of convergence or permission to disable endpoint validity checks.

## Immediate question and evidence contract

Which filter invariant failed at the four actual proposed endpoints, and is the
failure in the computed value, derivative, or numerical diagnostic? Reconstruct
the proposals with the original saved states, stateless seeds, frozen map and
nomination. Require replay agreement with the saved transitions before using the
diagnostic. Record physical parameters, completed steps and failed check indices.
The exact original filter is the comparator. No tuning or default change is
authorized by this diagnostic. All prior targets, data, N=4032, FP32, TF32 off,
scientific criteria and 48 GPU-hour total remain fixed.

The diagnostic is capped at 600 seconds on GPU0 and is charged to the campaign.
Save results and logs in a fresh endpoint-diagnostic attempt. Stop the diagnostic
on replay mismatch or invalid evidence; a candidate failure triggers a localized
repair, not abandonment. Numerical endpoint failure remains a sampling veto.
Energy magnitude and ESS remain explanatory only.

## Skeptical audit

Passed for this bounded diagnostic: inspecting retained states alone would miss
the rejected proposals, so reconstruct the actual proposal positions. A JSON fix
alone cannot establish endpoint validity. Four rejected endpoints are not proof
that the posterior was sampled correctly. Model A and learned-map evidence must
be preserved; numerical source changes need dependency-aware invalidation.
The frozen map and seeds are original receipts, not transferred assumptions.
No heuristic comparison is being made and no method ranking will be claimed.

## Repair scope

First make nonfinite energy diagnostics JSON-safe with explicit nonfinite counts,
preserving all health decisions and finite diagnostic values. Then use the
reconstructed filter evidence to choose the smallest repair. Do not relabel a
numerical failure as declared support or simply continue the failed checkpoint.

### Replayed cause and bounded precision experiment

The four transitions reproduce with identical rejection/status and maximum
retained-state error 4.44e-16. All fail the reset factorization: relative
eigenvalue margins 1.32–1.92e-7 are below the FP32 two-dimensional threshold
2.38e-7. The factorization substitutes an identity matrix and flags invalidity;
the resulting 99.7–99.9% covariance errors describe that flagged substitute,
not an accepted reset. The JSON repair passes 14 CPU controller regressions.

Test a localized precision candidate, capped at 900 GPU seconds. Keep FP32
particles, proposal maps and pairwise transport. Only for rows whose original
reset factorization fails, recompute the weighted moments and the transported
cloud moments, recoloring and analytical tangents in FP64, then cast back to
FP32. This changes finite arithmetic and therefore needs a new target identity
if adopted; it is not a tolerance relaxation or diagonal ridge. The FP32 guard
remains, and the FP64 fallback must pass its own guard and the existing 1%
restoration check. Healthy rows retain the original output exactly.

Comparators are the original FP32 implementation and full-FP64 execution of
the same algorithm and design. Evaluate the four actual failed positions and
the four existing final-data probes, separately. Success requires all candidate
evaluations to finish, finite scores, unchanged healthy branches, and likelihood/
score agreement within 1% of full FP64 (absolute scale floor 1 for likelihood,
score-vector norm for score). Report component errors too. This precision check
does not certify the finite-particle posterior or rank stochastic methods.
If it fails, investigate the reset rather than weakening health checks.

Default audit: FP32 remains the owner-selected default; conditional FP64 is a
repair hypothesis motivated by measured ill-conditioning, not an inherited
setting. The original epsilon1/reset5/flow8 controls remain fixed to isolate
arithmetic. This comparison is deterministic on a frozen design; no stochastic
ranking, heuristic dominance or posterior-convergence claim follows. Skeptical
audit passes for diagnosis: compare actual endpoint positions and untouched
healthy anchors, preserve target identity distinctions and charge all time.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Diagnose and repair B | Exact saved-transition replay | Replay mismatch or invalid evidence | Cause of four endpoint failures | Local replay and filter trace | B convergence or exact-posterior accuracy |

| Inference status | Finding |
|---|---|
| Hard veto screen | B numerical endpoint failures; A complete and reference agreement passes |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime and individual diagnostics |
| Default readiness | Unchanged |
| Next evidence needed | Reproduced cause, regression, valid B sampling |

## Precision result and executable continuation

The isolated arithmetic experiment completed all eight evaluations. At the four
previously failed proposals the maximum likelihood relative error against full
FP64 is 6.2341e-7 and the maximum score-vector relative error is 4.0850e-5
(0.004085%). The maximum covariance restoration error is 0.00015112 (0.015112%).
The four healthy probes retain identical likelihood values and scores. Evidence:

docs/plans/artifacts/structural-ledh-ab-20261023-01/attempts/model_b-reset-precision-01/

Adopt this as an opt-in Model B numerical repair. The shared implementation reuses
the existing transported cloud, avoiding a duplicate quadratic transport. It
emits a per-row, per-time fallback indicator. FP32 remains the default; FP64 is
used only for moment formation, factorization, recolouring and analytical Jets
when the original FP32 reset factorization fails. Its own factorization guard
and the existing 1% covariance restoration check remain active.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Proceed to production wiring check | All eight probes within 1% of full FP64 | No nonfinite/incomplete candidate evaluation | Finite-design posterior approximation error remains | Test production path, retune map, restart B HMC | Exact nonlinear posterior or general default readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Prototype passes at all eight inspected positions |
| Statistically supported ranking | None; deterministic arithmetic check only |
| Descriptive differences | Errors above concern this fixed design and these positions |
| Default readiness | Opt-in B repair only; global FP32 policy unchanged |
| Next evidence | Production endpoint and posterior wiring, fresh adaptive HMC |

### Dependency-aware restart

The master has a scoped migration mode. It rejects changes outside the shared
reset, structural target, HMC diagnostic serializer and master. It preserves
prepared data, references, all original receipts and completed Model A inference.
It records old and new source identities, creates a new Model B selection receipt,
and assigns a distinct target identity to conditional reset precision. Old B
sampling is never appended to the new target's draws.

1. Production GPU check (maximum 900 seconds): same four failed endpoints and
   four healthy probes versus archived full FP64. Require valid complete filters,
   <=1% likelihood/score error, identical healthy values/scores, and posterior
   value/score wiring including the prior. CPU checks separately exercise mixed
   healthy/ill-conditioned/singular rows, tangent parity and migration accounting.
2. Warm-start the selected canonical IAF parameters only. Verify the checkpoint
   checksum and architecture; reset Adam. Train 256 accepted updates on the new
   target with fresh training draws, capped at 5,400 seconds. This is a local
   numerical repair to the same dataset and model, not transfer from another
   problem. The previous width/rate/seed search remains warm-start evidence.
   Require finite heldout evaluations and inverse/logdet checks before freezing
   a newly identified map. Training loss is nomination evidence only.
3. Fresh B HMC nomination and adaptive warmup, then cumulative retained sampling
   until all required R-hat values meet 1.01 or the 20,000-draw maximum. Target
   acceptance remains 70%; ESS and finite energy errors remain diagnostic only.
   No extra validation-sampling phase. Allocate at most 36,000 seconds initially
   and at most 36,000 more for a compatible checkpoint continuation, all charged
   to the original 48 GPU-hour campaign. Every worker allowance is clipped to
   the remaining total budget (40.290645 hours before this continuation).
4. Update the final assessment. Keep historical uniform-precision reference
   comparisons explicitly labelled; they do not silently become repaired-target
   comparisons. Exact long-horizon Model B posterior accuracy remains unresolved.

Skeptical audit: original A evidence remains valid because the default arithmetic
is unchanged. A bitwise CPU test of all downstream particle tangents was too
strong: separate graphs can reorder the later moment correction (observed maximum
absolute difference 2.47e-5), even without fallback. The test now requires exact
healthy reset rows, exact short-filter likelihood/score, and numerical particle
and tangent parity; production likelihood/score checks remain strict. Reusing
optimizer state or old B warmup would be wrong after the arithmetic identity
change; neither is allowed. The conditional precision boundary does not establish
global differentiability or posterior approximation accuracy; HMC uses frozen
position-only forces with MH endpoints, and downstream diagnostics remain
necessary. A new numerical failure triggers localized investigation, not a
claim against the scientific direction. Budget exhaustion, corrupted evidence,
invalid endpoints or broken invariants remain continuation vetoes.

Exact command from the current checkout:

    /home/chakwong/anaconda3/envs/tf-gpu/bin/python scripts/run_structural_ledh_ab.py resume --output docs/plans/artifacts/structural-ledh-ab-20261023-01 --migration-note docs/plans/structural-ledh-ab-endpoint-recovery-2026-10-24.md --repair-model-b-reset

Subsequent resumes use the ordinary resume command: the state records the repair
route and does not repeat migration or completed stages. Output attempts are
versioned; existing evidence is retained. No stronger security ceremony applies.

### Production diagnostic attempt 01: harness repair

The XLA filter and posterior evaluator compiled successfully. A subsequent eager
GPU subtraction in the diagnostic's independent prior calculation failed with
CUDA_ERROR_INVALID_HANDLE on the Blackwell device. This is a harness exception,
not evidence of numerical filter failure. Move that independent prior diagnostic
to CPU, and use NumPy reductions for reporting already-returned arrays. No
numerical kernel, target identity, guard, criterion, or training setting changes.
Charge the 30.039 seconds and retry in a fresh attempt. The scoped migration
allows a reviewed master-only harness repair before new-target training starts;
all existing evidence and budget accounting remain. No approval is needed for
this localized retry under the same scientific contract and hardware budget.

### Production attempt 02 passed; training resumed

Production result B-reset-production-check-02/result.json is RESET_PRECISION_VALID.
All eight filters completed; prior/posterior value and score wiring passed.
The four formerly failed positions used the fallback at 6, 3, 1 and 1 time steps,
respectively. The four healthy probes used it zero times, and retained bitwise
identical likelihoods and scores. Maximum score-vector error against full FP64
is 0.004085%, matching the isolated experiment. This is verified call-chain
numerical evidence; posterior convergence and nonlinear posterior accuracy remain
separate questions. The master advanced automatically to B-reset-train-01.
