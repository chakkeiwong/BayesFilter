# Corrected Parameter-Authority Phase 50 Result

Date: 2026-08-26  
Version: `v3.2-defensive-proposal-support`  
Subplan: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase50-subplan-2026-08-26.md`  
Status: `PASS_V3_2_DEFENSIVE_SUPPORT_REPORT_REPAIR_TRIGGERED`

## Question and scope

Phase 50 tested whether a broader independent-proposal law repairs the finite
between-bank variability left by Phase 49's eight-step independent MH arm. The
declared target remained the batch-native q=20 SSL-LSTM log target
`pi(theta) proportional to exp(V(theta))` in `theta in R^4`. The internal UKF
state is 60-dimensional and remained inside `V`; it was never a particle
coordinate. The base annealing law, initial clouds, resampling schedule, eight
mutation steps, seeds, and target signature were unchanged.

The candidate law was

`r(theta) = (1-rho) q(theta) + rho s(theta)`,

where `q` is the frozen defensive-mixture proposal,
`s(theta)=Normal(center, tau^2 I)`, `rho=0.50`, and `tau=4.0`. These values are
hypotheses for this diagnostic, not defaults. The tempered bridge stayed

`bridge_q(theta) = (1-beta) log q(theta) + beta V(theta)`.

Because `r` is not the bridge base, the independent-MH log ratio used by the
runner was exactly

`bridge_q(theta') - bridge_q(theta) + log r(theta) - log r(theta')`.

Thus the run tested proposal support while retaining the declared target and
measure. It did not test whitening, posterior correctness, HMC, or canonical
LEDH.

## Hard-gate evidence

| Gate | Result | Artifact/evidence |
|---|---|---|
| finite q-base/r-proposal algebra fixture | passed | `PASS_V3_2_DEFENSIVE_SUPPORT_FIXTURE`; beta-zero and beta-one ratios, movement, finite states, and eight steps |
| target and theta measure | passed | target signature `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d727`; measure `theta_R4`; retained rows `[256,4]` |
| q versus r separation | passed | q used in annealing weights; r evaluated at current and candidate rows |
| candidate validity | passed | no invalid candidate was accepted in any active stage |
| paired replay | passed | all Phase 47 initial clouds and identity endpoint hashes reproduced for three replicates |
| finite artifacts and unique root | passed | boundary and report receipts are finite and were written below a fresh Phase 50 root |
| GPU/XLA policy | passed | two RTX 4080 SUPER devices, memory growth before logical-device use, XLA and TF32 enabled |
| report validation | passed | `PASS_V3_2_DEFENSIVE_SUPPORT_REPORT` |

The fixture, GPU boundary, and CPU-hidden report wall times were respectively
`0.6551598490332253 s`, `5566.864650205011 s`, and `0.33393983298446983 s`.
The boundary source and artifact hashes are in its `run_manifest`; the report
records the fixture, boundary, comparator, and source hashes.

## Descriptive result

The support arm generated the intended broad component at approximately one
half of candidate draws and moved particles at every nonterminal stage. Mean
acceptance by stage was approximately `0.292`, `0.243`, `0.202`, and `0.148`
for replicate 1, with comparable values in replicates 2 and 3. No invalid
candidate was accepted. These are implementation diagnostics, not convergence
evidence.

The report's three paired spread summaries are:

| Metric (spread across three banks) | Support arm | Frozen Phase 49 depth 8 | Support <= Phase 49? |
|---|---:|---:|---|
| theta mean[0] | `0.8011797849159568` | `0.4301091997991443` | no |
| maximum covariance off-diagonal | `4.253474535052258` | `1.214406340218456` | no |
| negative-mode mass | `0.06533211325460875` | `0.06927841802549417` | yes |
| retained root count | `11.0` | `10.0` | no |
| weighted ESS fraction | `0.04410268334541745` | `0.03022295068852543` | no |

Only the negative-mode spread decreased. The predeclared primary condition
required all three primary metrics (theta mean, negative-mode mass, and
covariance off-diagonal) to be no larger than the frozen Phase 49 arm. The
resulting branch is therefore
`support_broadened_does_not_reduce_variability`.

This is a candidate-method failure under the declared finite diagnostic. It is
not evidence that the non-symmetric MH identity is mathematically invalid, and
it is not a failure of the target, theta measure, replay harness, or GPU
boundary. With three replicates and no uncertainty model, no ranking is
statistically supported.

## Decision table

| Decision | Primary criterion | Status | Veto/limitation | Next action | Not concluded |
|---|---|---|---|---|---|
| retain theta target authority | target/status/measure/pairing/replay gates | pass | none | retain the R4 target boundary | posterior correctness |
| promote IID Gaussian whitening | finite mutation clouds | veto | finite clouds do not identify a Gaussian law; primary spread condition failed | keep whitening closed | IID Gaussian law |
| promote broad support as default | paired spread against frozen Phase 49 | defer/reject candidate | covariance, theta-mean, root, and ESS spreads were not reduced; only three replicates | keep this law role-limited and test geometry | superiority/default readiness |
| admit HMC or canonical LEDH | downstream density and posterior gates | veto | whitening and posterior gates remain closed | keep HMC and canonical LEDH closed | HMC/LEDH readiness |

## Inference status

| Evidence class | Status |
|---|---|
| hard veto screen | passed |
| statistically supported ranking | none; three replicates and no uncertainty analysis |
| descriptive-only differences | broad support changed movement and mode spread, but did not reduce the declared primary variability vector |
| default readiness | not ready |
| next evidence needed | a mode-aware proposal geometry test with exact density correction, then uncertainty-aware downstream validation if it passes |

## Mathematical and audit limits

The fixture directly evaluates the substituted formula for the q-base/r-proposal
ratio and checks that the implemented bridge and correction agree numerically.
No new external MathDevMCP invocation was made for Phase 50, so this result
does not claim a new MathDevMCP proof. The earlier Phase 48 audit remains
historical: its unconstrained symbolic form was inadmissible, while direct
substitution was certified by the available SymPy backend. The Phase 50
fixture is an implementation check, not an invariance theorem or a population
limit.

## Research-direction classification and repair

Engineering correctness passed. Numerical validity of the declared finite
kernel passed. Scientific interpretation is a negative result for isotropic
support broadening under this scope. The strongest alternative explanation is
that an isotropic tail changes proposal coverage without aligning candidate
mass with the two target-mode neighborhoods; the result does not distinguish
that explanation from finite-bank variability.

The next smallest discriminating artifact is a mode-aware candidate component
constructed from the existing stationary representatives and their stable
local curvature matrices. It will keep `q` in the bridge, evaluate the exact
mode-aware mixture density in `theta`, and compare against the frozen Phase 49
and Phase 50 receipts without pooling or training on their rows.

## Red-team note

| Item | Statement |
|---|---|
| strongest alternative explanation | the isotropic `tau=4` component is poorly aligned with narrow, separated mode neighborhoods; the apparent spread change may be proposal geometry rather than target behavior |
| evidence that would overturn the repair choice | a fresh mode-aware proposal with exact correction fails under the same paired target and still shows the same variability, or an independent uncertainty-aware downstream check shows no geometry effect |
| weakest part of current evidence | three finite paired replicates, descriptive spread ranges, and no Monte Carlo interval or independent population check |

## Artifacts and provenance

- Fixture: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase50-defensive-proposal-support/fixture/`
- Boundary: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase50-defensive-proposal-support/q20-paired/result.json`
- Report: `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase50-defensive-proposal-support/report/result.json`
- Runner: `docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase50_2026_08_26.py`
- Fixture runner: `docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase50_fixture_2026_08_26.py`
- Reporter: `docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase50_2026_08_26.py`

The report records boundary SHA-256
`8637066a40ee2f85ac63ef7f5a00f67566216d52c3694e8788650dc9fe4e1f96`, fixture
SHA-256 `9ec404077f0ebd8b5f20301d98fab38e0430391b4a74066597226dda0f058efe`,
and frozen Phase 49 report SHA-256
`9afb3c79423a3293a5cc7040059fd7adb8f8a543cef25fca102a7f1ca18e021d`.

No rows from this diagnostic were used to train NeuTra, select an objective,
admit HMC, or promote a default.
