# SSL-LSTM NeuTra Phase 5 Exact Transformed-Target Preflight Result

Date: 2026-07-16

Status: `PHASE5_PASSED_R2_PHASE6_PLAN_READY_EXECUTION_NOT_AUTHORIZED`

## Decision

Both immutable trial-0 transports passed the exact transformed-target
preflight on CPU-hidden reference and trusted GPU/XLA routes. Phase 5 closes
with decision `PHASE5_EXACT_TRANSFORMED_TARGET_PASSED`.

This establishes that the reloaded G/H transports implement the intended
locked SSL-LSTM posterior in `z` coordinates at the prospectively fixed probe
bank, including the Jacobian value and score terms. It does not establish HMC
readiness, posterior correctness, support completeness, predictive validity,
superiority, or default readiness.

No HMC transition, tuning, retained sampling, candidate search, or transport
training was run in Phase 5.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 5 for G and H | Passed exact binding, value, full score, finite difference, scalar/batch/permutation, serialization, original-start roundtrip, and trusted GPU/XLA gates | No target, payload, sign, score, finiteness, mutable-state, serialization, device, or XLA veto | Finite probes cannot exclude shared reverse-KL mode omission | Prepare a separately authorized Phase 6 identity-mass transformed-HMC mechanics/scale pilot, with repair only on a declared trigger | HMC readiness, convergence, posterior correctness, support/mode coverage, prediction, ranking, or defaults |

## Exact Evidence

| Check | Fresh G GPU/XLA | Fresh H GPU/XLA | Gate |
| --- | ---: | ---: | ---: |
| Payload SHA-256 | `6e147d5b...dedc354` | `ed0e4260...c9120fb` | Exact frozen identity |
| Value identity maximum absolute residual | `2.55e-14` | `1.35e-14` | `<=1e-10` |
| Score-formula maximum absolute residual | `1.29e-14` | `7.77e-15` | `<=1e-6` |
| Full finite-difference maximum absolute residual | `6.78e-8` | `6.94e-8` | Per component: absolute `<=1e-6` or relative `<=1e-5` |
| Probe roundtrip maximum absolute residual | `5.00e-16` | `9.44e-16` | `<=1e-9` |
| Original-start-only roundtrip maximum absolute residual | `2.78e-16` | `9.44e-16` | `<=1e-9` |
| Scalar/batch value/score residual | `0 / 0` | `7.11e-15 / 0` | `<=1e-10 / <=1e-10` |
| Batch permutation value/score residual | `0 / 0` | `0 / 0` | `<=1e-10 / <=1e-10` |
| Original-start inverse radii | `0.068, 1.601, 1.232, 1.076` | `0.275, 2.011, 1.602, 1.110` | Explanatory only |
| Pre-freeze/reload forward/logdet residual | `0 / 0` | `0 / 0` | `<=1e-10` |
| Mutable trainer/optimizer state reachable | None | None | Must be none |
| CUDA XLA HLO SHA-256 | `b9804855...d7124ce` | `0e618921...c25b88` | Nonempty, source-bound |

The CPU-hidden reference maxima were `6.78e-8` for G and `6.87e-8` for H.
CPU/GPU differences were numerical roundoff; the largest difference among the
recorded residual summaries was `7.11e-10`.

## Negative Controls

Both payloads rejected a modified tensor and the wrong target signature. For G,
wrong Jacobian sign changed the value identity by `3.54`, omitting the
Jacobian score missed the finite-difference score by as much as `0.236`, and
using its negative missed by `0.473`. The corresponding H margins were `3.33`,
`0.344`, and `0.688`. Thus the exact pass cannot be reproduced by the three
principal sign/omission mistakes.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for both G and H; all required values/scores finite, identities bound, hashes valid, and negative controls discriminating |
| Statistically supported ranking | None; no ranking was attempted and deterministic residual size is not a transport-quality ranking criterion |
| Descriptive-only differences | Original-start radii, residual magnitudes below tolerance, HLO size/identity, and runtime |
| Candidates remaining viable | G and H remain eligible for separately planned Phase 6 mechanics/tuning |
| Default-readiness | Not assessed and not supported |
| Next evidence needed | Prospective GPU/XLA identity-mass transformed-HMC mechanics and target-specific scale pilot, followed by longer tuning confirmation if no hard veto fires |

## Evidence Ledgers

| Ledger | Status | Evidence |
| --- | --- | --- |
| Engineering correctness | `PASSED` | `47` focused tests; compilation; `git diff --check`; corruption, target, direction, sign, omitted-score, immutability, and batch-dispatch tests |
| Numerical correctness | `PASSED_AT_FIXED_PROBES` | Direct value identity, complete analytic score finite differences, roundtrip, parity, exact serialization, CPU/GPU agreement |
| Sampler validity | `NOT_ASSESSED` | No HMC transition or tuning ran |
| Posterior correctness | `NOT_ASSESSED` | No oracle exists; no retained chains or predictive-moment validation ran |
| Scientific interpretation | `EXACT_BINDING_ONLY` | The frozen chart is correctly implemented at declared probes; global coverage remains open |

## Implementation Repairs Before Evidence

The first focused test run had two assertion-specification failures: an exact
zero-tolerance repeated forward comparison differed at floating-point roundoff,
and a corruption test expected a component-level message even though the
loader failed earlier at the aggregate tensor hash. These were repaired to
`1e-15` and `tensor_hash mismatch`; the numerical implementation was unchanged.

The first CPU runner invocation failed before writing a receipt because the
batch bridge looked for parameter names on the target config instead of the
target's public name/order contract. The bridge was changed to bind the locked
`FREE_PARAMETER_NAMES` constant, focused tests were rerun, and only then were
the source-bound CPU and GPU receipts generated. None of these failed attempts
is scientific evidence or included in the passing runtime comparison.

Final contract audit then found that the first executable score-parity gate
reused the `1e-6` finite-difference tolerance instead of the prospective
`1e-10` parity threshold. Although the observed residuals already passed
`1e-10`, the first receipts were superseded. The gate was corrected, an
original-start-only roundtrip field was added, focused tests passed again, and
fresh CPU/GPU `r2` receipts were generated. Only `r2` is authoritative.

## Run Manifest

| Field | CPU reference | Trusted GPU/XLA |
| --- | --- | --- |
| Git commit/worktree | `20835ecf...94e63ce`, dirty | `56b1f3c8...574f534c`, dirty |
| Environment | `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0` | Same |
| Device | GPU hidden by `CUDA_VISIBLE_DEVICES=-1`; CPU reference | Physical GPU 1 selected by `CUDA_VISIBLE_DEVICES=1`; logical `GPU:0` |
| Dtype/JIT/TF32 | `float64`; outer transformed program non-JIT; locked target retains default XLA; TF32 enabled but irrelevant to CPU | `float64`; whole transformed program CUDA XLA; TF32 enabled |
| Seeds | N/A, deterministic fixed bank | N/A, deterministic fixed bank |
| Wall time/cap | `246.1920 / 1800` seconds | `417.3270 / 1800` seconds |
| Command | `CUDA_VISIBLE_DEVICES=-1 ... --mode cpu-reference .../cpu-reference-r2.json --wall-cap-seconds 1800` | `CUDA_VISIBLE_DEVICES=1 ... --mode gpu-xla .../gpu-xla-r2.json --wall-cap-seconds 1800` |
| Trust basis | `cpu_hidden_reference_only` | `owner_designated_managed_session_visible_gpu_trusted` |

The wall cap is checked before each candidate. It is a sequential launch stop,
not asynchronous termination inside one target call. Neither run approached
the cap.

Another lane advanced `HEAD` between the CPU and GPU runs. The complete Phase
5 source-binding mappings in both receipts are byte-identical, including the
runner, plan, target, adapter, and loader hashes. The immutable payload and
best-state hashes also match. The commit difference therefore does not imply a
change to this lane's computation.

## Artifacts

| Artifact | SHA-256 |
| --- | --- |
| Authoritative CPU-hidden `r2` receipt | `59f2dad72afd5753810569d1931568591c1574c391d7363a84df8d5450299a4e` |
| Authoritative trusted GPU/XLA `r2` receipt | `f855fd3fc83260867582a79b024087efbc5ecd463321ddd98f5fae7c9056f55b` |
| Pre-`r2` runner source | `af6d5e22f52b422d08f720472b3059b791aff6142985afce1885a37fda6d9ad3` |
| Pre-`r2` plan | `be1d32b4dc99f1b8dd2821ae9e38027414a4f48e70ef61abba7e8dfc27710102` |
| Focused tests after parity repair | `1a470e3b6c2496ca27d8e862bc55102c1b75b9a8db30e6a872d630a84b419821` |
| Superseded first CPU/GPU receipts | `a66f698c...e9243a1` / `6ebfdc52...47fa4d1` |

The plan is updated after execution for closeout, so its current byte hash is
expected to differ from the pre-run hash bound in both receipts.

## Post-Run Red Team

The strongest alternative explanation is that G and H share reverse-KL mode
seeking and the fixed 21-point bank does not enter a missed posterior region.
This would leave every exact identity intact while making a future sampler
scientifically incomplete. Phase 6 must therefore preserve all four original
starts and per-chain telemetry; later predictive moment validation remains
necessary even if transformed HMC tunes successfully.

Evidence that would overturn this closeout is payload/source drift, failure to
reproduce the source-bound receipts, a demonstrated analytic score error away
from the fixed bank, or failure of the first exact HMC mechanics invariant.
