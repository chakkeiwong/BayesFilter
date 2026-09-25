# q20 batched HMC repair and training-cost clarification

Date: 2026-09-16. Status: implemented, tested and integrated into the main worktree.

The owner requests parallel execution of the four scalar chains and an account
of Python loops and the large training/validation schedule. The repeated HMC
and likelihood kernels already use stable-signature XLA. Their Python chain
scheduler is outside that graph and serializes independent work. Connect the
existing batched TFP runner to q20 qualification, pricing, public candidate
tuning and retained sampling. Four independent states become a `[4,D]` tensor;
TFP draws separate momentum/acceptance variates for the rows. Markov time and
the filter's observation time remain sequential graph loops.

## Evidence and implementation contract

Baseline: the current `chain_mode="serial"` public execution binding, with
unchanged target, geometry, L, epsilon, accepted/proposed health checks and
posterior assessment. Add an explicit `batched` execution mode using the
existing `ReusableFullChainHMCRunner`; preserve compatibility modes for other
consumers. q20 explicitly selects batched mode. Bind the mode into transition
identity and replay, and record accurate retained-controller metadata. Issue
a new q20 qualification schema that requires actual batched public-runner
evidence. Reject historical serial receipts as authority for this new topology.

Engineering acceptance: batched public tuning, export/reload, continued draws,
trace shapes, per-row independence, deterministic seed replay and numerical
health work in focused CPU-hidden reference tests. Include one tiny XLA graph
check through q20 qualification. Existing q20 GPU evidence from the preceding
performance audit demonstrated the unchanged reusable batched kernel; do not
rerun a full q20 campaign for this wiring repair. New q20 campaign execution
still needs qualification for the revised source identity.

Primary criterion is executable public consumer correctness. Wrong shapes,
shared randomness across chains, dropped proposal health, lost continuation
state, callbacks in the compiled graph or accepting serial-only receipts veto
the repair. Compilation and small timing checks cannot establish posterior
accuracy, whitening or method superiority. Preserve all former run artifacts.

No training count, loss threshold, model, precision or backend is changed in
this repair. Explain which training checks address learning, variability,
capacity and downstream validity. Distinguish necessary questions from the
uncalibrated numbers chosen to answer them, and document repeated heldout work
and the overly restrictive fine-precision expansion condition.

## Review and budget

Skeptical audit: equal root seed integers do not give scalar-versus-batched
pathwise parity because TFP random shapes differ. Test reproducibility within
the new topology and row independence, not false equality across topologies.
Candidate health must retain per-chain axes. The shared exact-score runner
does not confer authority on the separate position-field algorithm. Its
existing execution rules remain unchanged. A cached/new graph must consume
tensor step size and seed so repeated candidates do not capture stale values.
No new training/validation defaults are inferred from runtime pressure.
Verdict: proceed with this bounded wiring repair and focused reference tests.

Work in `/tmp/BayesFilter-q20-batched-hmc-20260916` at base main `d31e1b76`, then
apply only the reviewed changed files to the shared main worktree. Preserve its
unrelated edits. No package/environment mutation or new backend.

Reserve at most 900 seconds from the previous performance-audit settlement
(campaign 123223.99030228473 seconds, including diagnostic
42247.537327354854 seconds). Two test invocations may consume at most 600 and
300 seconds, including timeout cleanup; a second requires a concrete failure
or additional uncovered concern. These are convenience engineering limits,
not scientific counts. Use `CUDA_VISIBLE_DEVICES=-1` before import and record
the intentional CPU reference exception. No GPU workload is launched here.
Record commands, measured time, failures, result and settlement below.

## Execution record

Attempt 001 failed during test collection: the fresh Git worktree did not
contain the ignored compiled `_symmetric_sylvester_ops.so` required at package
import. No test or scientific calculation ran. Copy the already-built library
from the preserved previous execution checkout; no package install, rebuild,
backend substitution or environment change. Preserve the failed attempt and
rerun the same focused tests within the remaining phase reservation.

Attempt 002 passed all 70 focused execution, q20 qualification and master
integration tests in 277.588088494 seconds of supervised wall time. Attempt
001 used 3.874637348 seconds. Tests included CPU XLA compilation, row independence,
seed replay, dynamic step sizes, export/reload, continued sampling, proposal
health and rejection of historical serial qualification receipts.

Integration audit: another task added retained-chunk runtime accounting and
support for transformed model dimensions in the shared checkout. Preserve those
changes and its adaptive verification-count test. Copy the two concurrent source
files into this isolated checkout, then merge the batched topology flag and let
runtime accounting recognize the batched runner's invocation-count field. A
third focused invocation is justified by this newly discovered interaction:
check q20 member sampling, checkpoint reuse, and first/repeated call metadata
for identity, classical and NeuTra consumers, plus the changed evidence-count
test. Reserve at most 300 seconds (295 plus five for cleanup) from the unchanged
900-second total. This replaces the original two-invocation limit without
expanding compute or scientific scope. No inference about speed or posterior
quality is drawn from these CPU fixtures. The integration audit passes because
the checks exercise the affected public handoff and preserve source identity,
randomness, numerical policy and earlier results.

Attempt 003 passed all five integration checks in 60.220974507 seconds of
supervised wall time: three actual q20 consumer variants and the verification
evidence-count check in both serial and batched modes. The identity consumer
also reloaded its durable chunks and reproduced the retained draws exactly.
The changed source and test files in main were compared byte-for-byte with
the tested checkout after integration; every comparison passed. Concurrent
retained timing, transformed-dimension support and adaptive evidence-count
assertions were preserved. `git diff --check` passed for the changed files.
No new commit or serious campaign was launched by this repair.

Artifacts are under
`docs/plans/artifacts/ssl-lstm-q20-batched-execution-2026-09-16/`.
Each attempt has its command, source identity, result and log. Attempt 003 also
preserves the tested source patch, including the concurrent edits it exercised.
The total supervised test time is 341.683700349 seconds including the failed
import attempt. Conservatively retain the whole 900-second phase allocation
as a debit, not a claim of measured compute. The separate settlement leaves
122323.990302285 campaign seconds and 41347.537327355 diagnostic seconds;
diagnostics remain included in campaign time.

## Result and continuation

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept q20 batched execution wiring | 70 focused tests plus five integration checks passed | No dropped chain axis, callbacks, shared row randomness, stale seed/step capture or lost replay observed | Full q20 GPU qualification of the revised source has not run | Fresh qualification before serious sampling; reprice changed HMC topology | Posterior accuracy, whitening or production readiness |
| Reject the claim that 449 hours are mathematically necessary | Quote decomposes into chosen optimizer floor, validation reserve and safety factor | No measured q20 calibration supports the loss-resolution or all-scope counts | Actual learning curves and downstream useful training budget are unmeasured | Repair caching and decision-specific validation under a revised training plan, then price a staged search | A smaller arbitrary count is sufficient |

| Inference status | Result |
| --- | --- |
| Hard veto screen | CPU mechanics and XLA checks passed; no new q20 scientific run |
| Statistically supported ranking | None |
| Descriptive-only differences | Earlier GPU diagnostic observed about 53.5 seconds serial versus 16.5 seconds batched for a short four-chain chunk |
| Default-readiness | q20 selects batched execution; posterior/default-quality evidence remains outstanding |
| Next evidence needed | Revised-source GPU qualification, calibrated training protocol and downstream posterior/reference checks |

Post-run skeptical review: the synthetic tests cannot expose every interaction
between batch arithmetic and q20's strict eigensolver. The preceding real q20
GPU probe supports feasibility but is not a new-source qualification receipt.
The code therefore rejects serial v1 receipts and preserves per-chain target
and proposal checks. Different batched random streams can produce different
tuning outcomes. Compare target values and gradients with independent scalar
references where required; do not require identical scalar/batched chain draws.
The expensive strict target, repeated target telemetry and training-validation
costs remain separate limitations. Batched execution alone does not make the
full current campaign affordable. These findings reject an execution and
budget design choice, not NeuTra or HMC as research directions.

## What the training and validation are for

Training estimates a nonlinear map T from Gaussian z into the beta-temperature
target using the batched reverse-KL objective
`E[-log pi_beta(T(z)) - log|det J_T(z)|]`, up to a constant. Architecture and
learning-rate comparisons examine capacity and optimization; independent roots
examine variability; direct beta-one training supplies plain NeuTra, while
beta-half followed by beta-one training supplies the continuation comparison.
These are useful experimental questions, not reasons that every Cartesian
combination must be trained to an arbitrary count before anything is learned.

The current grid contains two widths, two learning rates and three roots for
each of two schedules: 24 histories. Twelve direct histories have one positive
beta, and twelve continuation histories have two, giving 36 training scopes.
The 128/512/2048/8192 update counts are cumulative observation points and caps,
not four additive training allocations. The initial all-history floor is 512.
It costs 18,432 optimizer updates or 589,824 target rows at batch 32. Neither
512 nor 8192 follows from a convergence theorem or measured q20 stopping curve.
Adam's moment memory likewise does not determine the required training count.

Validation compares paired losses at the same latent rows for the beta-start,
previous and current maps. It asks whether learning improved on its baseline,
whether additional optimization still helps, and whether the apparent change
is Monte Carlo noise. This is training-selection evidence. A low loss or a
plateau does not prove whitening, mode coverage or posterior convergence.

For a fixed comparison with IID row differences of standard deviation s,
the approximate interval half-width is `h = 1.96*s/sqrt(M)`. Resolving h=0.02
requires approximately 9,604 rows when s=1. This explains why the chosen
768/3072/12288 ladder becomes large; it does not justify the chosen precision.
The mathematical audit explicitly states that delta=0.04 and h=0.02 have no
established q20 adequacy basis. Reused adaptive selection banks do not provide
nominal final 95% inference, however many rows are evaluated.

Four separate excess-cost mechanisms are confirmed:

1. The all-scope reservation funds the largest validation bank at all four
   rungs for three maps, before learning whether each expansion is useful.
   `(768+3072+12288)/32 * 3 * 4 * 36 = 217728` target batches, or 6,967,296 rows,
   are reserved. Actual adaptive execution can stop earlier. This validation
   count is about 11.8 times the optimizer-floor target-row count.
2. Growing from 768 to 3072 re-evaluates the first 768 rows, and growing to
   12288 re-evaluates the earlier prefix again. Immutable cached losses and
   prefix-only extension preserve the same paired sample and remove this work.
3. Baseline and previous map losses are recomputed across rungs. A cache bound
   to target, beta, frozen map, bank seed and row range can preserve the exact
   comparison. Mutating-map or cross-beta cache reuse would be invalid.
4. `_evaluate_rung` expands while either interval half-width exceeds 0.02.
   This is stricter than the documented rule allowing clear progress to justify
   continued learning without that fine precision. For example, an improvement
   interval [0.5,1.5] is already wholly above delta=0.04 although h=0.5. More
   validation is unnecessary for that particular continue-training decision.
   A plateau or small-difference ranking requires different evidence.
   This example uses loss decrease; the code stores the opposite sign,
   `after - before`, so the corresponding stored interval is [-1.5,-0.5].

Checked source anchors: `q20_production_config.py:51` defines the grid and
thresholds; `q20_production_training.py:74` performs the repeated three-map
validation; `neutra_training_protocol.py:249` regenerates and evaluates the
same bank prefixes; its lines 265 and 283 define paired differences and the
plateau/export policy. These are chosen training controls, not HMC invariants.

The saved 449.37-hour quote consisted of about 17.57 optimizer-floor hours and
207.11 heldout-reservation hours, multiplied by the engineering factor two.
It is neither an observed HMC runtime nor a lower bound on necessary training.

For an invertible differentiable T and the declared finite-program target,
the transformed HMC density is
`pi_z(z) = pi_theta(T(z)) * |det J_T(z)|`. Applying the correct HMC/Metropolis
transition to this density targets the same distribution for any such fixed
map, assuming a correct value/gradient program and valid chain operation.
Good training can improve exploration and efficiency. It is not required for
that change-of-variables identity and does not replace convergence, effective
sample size, Monte Carlo error or reference checks. A badly trained map can
still be computationally unusable; a low-loss map can still miss important
regions. The current two-plateau export rule is a chosen training-assessment
policy, not a mathematical prerequisite for running diagnostic HMC.

The next training repair should cache unchanged evaluations, stop validation
when it resolves the actual declared decision, and calibrate costly precision
against downstream use. Stage the architecture/LR/seed search within a priced
budget and extend it for specific unresolved questions. Preserve fair matched
comparisons and all failed histories. A reduced exploration cohort supports
narrower exploratory claims, not the former full-cohort robustness claim.
Do not substitute another untested count or revert to treating a tiny canary
as a trained production map. These changes require a revised training contract;
this execution-only repair leaves the numerical schedule unchanged.
