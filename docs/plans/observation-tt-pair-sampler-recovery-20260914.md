# Pair sampler recovery and terminal writeup

## Active question and skeptical audit

Does the saved d4 pair sampler fail through non-finite arithmetic, an invalid
CDF bracket, or inverse-CDF residual? Replay campaign-01's saved cores, charts,
fixture, particle seeds and consumer call chain before changing any numerics.
All five launch source hashes match on resumption; HEAD is now
`5836f0344293f1c4af85abba23689ba83d34d9af`, versus the recorded launch HEAD
`14a292098f35b6de450ffca29131ea34f4a4d8d7`. Relevant uncommitted sources match;
unrelated changes are preserved. This resolves the stale-checkout concern.

Skeptical audit passes for localization: comparing a replay with the original
guard answers the failure question without refitting or selecting on exposed
observations. Sampling inputs and raw diagnostics must be saved before raising.
The exact baseline is the unchanged failed sampler and frozen path. Success is
deterministic localization, not filtering promotion. Non-finite values, invalid
positive mass or brackets, and unexplained CDF residual remain continuation
vetoes for downstream execution. The replay may reproduce a veto for diagnosis.
Runtime, fit residuals and ESS are explanatory only. No HMC, superiority,
default-readiness or Zhao–Cui source-faithfulness claim follows.

## Bounds and repair decisions

Charge every new numerical process by elapsed wall time, including startup and
tests, against the conservative 2,020.475387 seconds remaining in the reset
memo. That ledger includes all six recorded research runs plus three documented
mechanics/test checks. No other metered charges are present in the checkpoint;
unmetered historical routine checks remain a ledger limitation. The first replay
has an external 120-second timeout and stops at the first failure. Use the
manifest's tftwogpu Python, float64, RTX5080 GPU/XLA, threads 2/1, verified memory
growth, and escalated GPU permission. No environment/package mutation.

Preserve runs under
`docs/benchmarks/artifacts/observation_tt_pair_block_remedy_20260914/replay-01/`
and subsequent unique names. Record command, hashes, device policy, seeds,
wall time, diagnostics and inputs. Class A diagnostic retention is authorized;
the existing numerical guard and its 1e-8 threshold stay unchanged.

A localized repair needs an identified cause, derivation, focused independent
regression, and a no-fire check on healthy inputs. Numerics-altering protections
need a dedicated non-harm comparison before adoption. A true unresolved validity
veto stops downstream work, not documentation. Once resolved, one repair retry
can complete the existing comparison under the inherited third-launch ceiling,
without new data, settings, fitting selection, or promotion criteria. Replay
the saved fits rather than spending a launch on refitting. Budget is checked
again before that retry; maximum total remaining compute is unchanged.

## Terminal deliverables

Record diagnosis, any repair, completed comparisons and explicit decision and
inference-status tables. Finish attempt05's pair mathematics/result section;
preserve its protected baseline, audit limits, equations and citations. Build
and inspect the rendered PDF. Mathematical theorem checks remain inconclusive
unless further checking establishes a stronger verdict. Update the active
checkpoint at each material boundary.

## Localized cause and repair contract

Replay-01 reproduced the guard at t=2, seed 210102 in 14.598617 seconds.
The bracket passed and CDF residual was 5.551115e-16; conditional mass was
-0.00869398. The forward Gram recurrence currently uses
`nab,nakc,nald,kl->ncd`: index b is summed without entering the second core,
while a enters both cores. The required recurrence is
`E_next[c,d] = sum_{a,b,k} E[a,b] C[a,k,c] C[b,k,d]`, hence the second
core must use `nbld`. This is a wrong contraction relative to the stated
squared-polynomial mass, not an ill-conditioned CDF or a failed statistical
candidate. Correcting the index restores the claimed formula; no clipping,
ridge, threshold adjustment, or other Class C protection is introduced.

The focused regression independently expands the conditional polynomial's
coefficient tensor and sums its squares. This is the exact Gaussian integral
in the orthonormal Hermite basis and is independent of the Gram recurrence.
Use signed rank-three coefficients, multiple particles, and both eager and
compiled execution. First preserve the failing regression, then require
agreement after correction. Healthy one-core and zero-polynomial cases remain
covered by the existing suite. CPU tests explicitly hide GPUs; they are
independent reference checks. Bound the failed and corrected tests to 60 seconds
each. Then replay all four d4 seeds with unchanged saved fits under 120 seconds.
No downstream result generated with the incorrect density normalizer is valid,
including completed d1 pair runs; those must be recomputed in the terminal retry.

Regression outcome: the independent rank-three coefficient-norm test failed
before repair (pytest 4.83 s); all 13 tests in the pair and observation-guided
suites passed after repair (pytest 7.79 s), including CPU XLA. Charge the full
two 60-second test ceilings conservatively because process-startup wall times
were not separately metered. Replay-02 completed all four GPU/XLA d4 trajectories
in 12.479128 s: all conditional masses positive, all brackets valid, maximum
CDF residual 9.992007e-16. Remaining conservative allowance: 1,873.397641 s.
The one-letter repair changes only sampling/density normalization; inspection
finds no call to that function in regression or retained-density construction.
MathDevMCP independently verified the two-dimensional quadratic-form identity;
this is a scoped algebraic check, not a full theorem certificate.

Terminal retry audit passes: use `--reuse-fits-from campaign-01` in the existing
master. Validate all frozen scope fields and fixture hash before TF import;
check saved chart parity against rebuilt SGQF charts. Reload all scalar and pair
fits and copy their diagnostics, preserving hashes. Recompute all reference and
particle arms on the original observations. Maximum 600 seconds externally,
590 seconds internally. This consumes the third and final full launch. Settings,
holdout exposure, methods, heuristic regimes and scientific criteria are unchanged.
Recomputed d4 pair runs must match replay-02 as a loader/consumer wiring check.
Existing analytical-score checks concern the scalar guided proposal only; they
cannot establish a pair-path or retraining derivative claim.

## Terminal completion

Campaign-02 completed the third/final launch in 48.464704 seconds. All six
saved proposal-file hashes match campaign-01; no refitting occurred. All d4
pair runs exactly match replay-02. Recomputed d1 pair runs exactly match
campaign-01: the one-core case has no nontrivial internal bond, so the earlier
blanket invalidation was conservative and d1 was unaffected. The corrected
consumer and independent integral regression now supply the relevant evidence.

All six methods pass the declared mean and log-evidence reference screens in
both dimensions. Pair promotion remains rejected by the heuristic comparison
in ordinary/large d1 observations and large d4 observations. Four particle
seeds on one sequence do not establish a ranking. No sampler continuation veto
remains in this scope; the tested configuration failed promotion, not the
research direction. The allowed solver repair was already exercised and did
not establish convergence. A terminal default audit also records that the
inherited 0.05 defensive mixture mass lacks calibration evidence; it remains
a hypothesis rather than a justified numerical default.

The conservative numerical allowance is now 1824.932937 seconds; all three
full-launch slots are consumed. Numerical execution is complete. The detailed
[result](observation-tt-pair-block-remedy-20260914-result.md) contains decision
and inference-status tables and links to exact commands, manifests and results.
Attempt05 now contains the Gram derivation, counterexample, completed reference
and conditional-heuristic tables, solver limitations and mathematical-audit
limits. The 48-page PDF builds with pdflatex and has been inspected in rendered
form. Protected baseline mathematics and citations are retained; the one
reformatted objective has identical algebraic terms. Human prose assessment
and full theorem certification remain unestablished; neither is represented
as completed by the build.
