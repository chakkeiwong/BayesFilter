# Corrected Parameter-Authority Phase 52 Subplan

Parent: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Version: `v3.4-fresh-paired-uncertainty-replication`  
Entry gate: Phase 51 report branch `mode_aware_geometry_reduces_between_bank_variability_descriptive`  
Status: `COMPLETE_CANDIDATE_REJECTED_NEXT_VALID_PHASE_UNDER_BUDGET`  
Long-boundary attempt cap: `24000 s`; successful boundary wall time: `23316.68617718201 s`  
Final conservative campaign use: `60316.343009740929 s`; remainder: `4483.656990259071 s`

## Question and scope

Does the Phase 51 mode-aware proposal geometry retain its descriptive
advantage when evaluated on six newly generated q=20 particle banks, with the
identity, isotropic-support, and mode-aware arms paired on exactly the same
initial cloud and resampling stream? The declared target remains the
batch-native SSL-LSTM target in `theta in R^4`; the 60-dimensional UKF state
remains internal to target evaluation.

This phase addresses the principal limitation of Phase 51: three paired banks,
inherited pilot seeds, and no uncertainty diagnostic. It does not retune the
geometry, change the target, or open NeuTra/HMC/LEDH.

## Frozen target and arm laws

Each fresh pilot supplies an M0 seed and a passing q=20 target/protocol
receipt. The runner regenerates the initial cloud with the frozen defensive
base law `q` and creates three arms from that same cloud:

1. `identity`: no mutation; q remains the annealing base.
2. `isotropic_support_mh`: candidates from
   `r_support=(1-rho)q+rho*N(center,4^2 I)` with the exact current/candidate
   `log r_support` correction.
3. `mode_aware_geometry_mh`: candidates from
   `r_geom=(1-rho)q+rho*s_geom`, where
   `s_geom=0.5*N(m_minus,4*C_minus)+0.5*N(m_plus,4*C_plus)` and the exact
   current/candidate `log r_geom` correction is used.

The q-based bridge, schedule, eight independent proposals at each
nonterminal stage, particle count `N=256`, target signature, and theta measure
are unchanged. The Phase 51 geometry and Phase 50 support receipts are frozen
comparators and provenance references; no old rows are pooled into the fresh
arms.

## Research-intent ledger

| Field | Statement |
|---|---|
| Main question | Does mode-aware geometry reproduce a paired finite-bank advantage on fresh banks? |
| Mechanism | Three-arm paired independent-MH comparison under the same initial clouds and resampling streams. |
| Comparator | Fresh isotropic-support arm; identity is a within-bank reference; Phase 50/51 reports are historical context only. |
| Expected failure | The Phase 51 spread pattern is seed-specific, ESS variability remains worse, or local geometry does not generalize. |
| Hard promotion criteria | Fresh pilot status, target/protocol, q/r separation, valid candidates, finite tensors, pairing, GPU/XLA, and artifact provenance. |
| Promotion veto | Any target/measure mismatch, stale pilot, arm-specific initial cloud, wrong MH correction, invalid acceptance, nonfinite artifact, or overwritten root. |
| Descriptive uncertainty criterion | For each primary spread metric, the 95% paired-bootstrap upper bound for `(geometry spread - support spread)` is `<= 0`. This only nominates the result for further validation; it is not a superiority claim. |
| Continuation veto | Six fresh pilots cannot be produced, the target/common support is unavailable, an exact fixture contradicts the arm law, three focused infrastructure repairs fail, or the campaign budget is exhausted. |
| Repair trigger | A candidate failure or uncertainty-incompatible result; it does not close the parameter-space direction. |
| Nonclaims | No posterior correctness, IID Gaussian law, exhaustive mode discovery, HMC readiness, canonical LEDH status, superiority, or default promotion. |

## Evidence contract

The CPU-hidden pilot receipts must be fresh, distinct, target-signature-bound,
and use the reviewed M0 protocol. The boundary must regenerate each initial
q cloud from the receipt seed, prove all three arms share its initial tensor and
resampling seeds, keep q in the bridge, evaluate each arm's proposal density at
both current and candidate rows, and reject invalid candidates. The report
must retain all six paired rows and compute deterministic paired bootstrap
intervals for arm-spread differences.

The primary estimand is the finite six-bank spread difference

`D_m = range({G_i,m}) - range({S_i,m})`,

for `m` in `{theta_mean_0, negative_mode_fraction,
covariance_offdiag_max_abs}`. The report also records paired per-bank
differences, ESS, root count, exact sign counts, and the Phase 51/50 frozen
descriptive summaries. Bootstrap intervals quantify uncertainty in this finite
replication diagnostic; they do not establish a population ranking.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| six fresh banks | minimum expansion beyond Phase 51's three; budget-bound hypothesis | still too few for a stable population claim | raw rows, bootstrap interval, no default promotion | reviewed diagnostic design |
| fresh M0-only pilot receipts | existing repository pilot protocol, with new seeds | stale or incomplete seed/protocol metadata | pilot schema/status/hash/target checks | required input |
| `rho=0.50`, support scale `4.0` | frozen Phase 50 arm | comparator itself may be poor | direct fresh support arm; no claim that it is optimal | frozen comparator hypothesis |
| `rho=0.50`, geometry scale `2.0` | Phase 51 nomination | effect may be seed-specific or ESS-costly | fresh arm and bootstrap uncertainty | frozen candidate hypothesis |
| eight MH steps | Phase 49-51 protocol | residual finite mixing | per-stage movement/acceptance/ESS | frozen diagnostic setting |
| bootstrap reps `20000` | predeclared computational convenience | interval Monte Carlo error | deterministic seed and manifest | uncertainty diagnostic, not proof |
| per-pilot supervisor guard `900 s` | derived from measured Phase 47 pilot maximum `516.3418564099702 s` with recovery margin | a regression could consume the boundary reserve | stop the process and classify as infrastructure/timing failure | reviewed execution guard |
| GPU/XLA and memory growth | repository owner policy | allocator or compilation failure | pre-import device receipt | required execution |

## Skeptical pre-execution audit

| Audit question | Finding | Control |
|---|---|---|
| Is this a new target or objective? | No; target signature, q bridge, theta measure, and schedule are frozen. | Validate signature and q/r fields in every arm. |
| Are the three arms paired fairly? | They can share the same initial cloud and resampling seeds while using arm-specific proposal streams. | Hash initial tensors and record seed offsets per replicate. |
| Is the uncertainty criterion a hidden promotion gate? | It is explicitly a nomination signal only. | Default/HMC/whitening decisions remain vetoed. |
| Are Phase 50/51 rows being pooled? | No; they are frozen provenance comparators only. | Fresh six-row report and separate source hashes. |
| Can six banks support a population claim? | No. | Exact statement that intervals are finite-replication diagnostics. |
| Is the budget sufficient? | Measured Phase 51 boundary was `5558.9085 s`; six pilots plus a six-bank three-arm boundary are estimated below the remaining campaign pool, with a `24000 s` local cap. | Stop before another launch if the manifest shows exhaustion. |

The audit passes for this bounded replication. It authorizes fresh pilot
generation and, if all pilots pass, the GPU boundary only.

### Measured pre-boundary audit

All six attempt-02 pilots passed the corrected theta-measure contract with
root seeds `[20260826, 5101]` through `[20260826, 5106]` and M0 seeds
`[20260826, 5201]` through `[20260826, 5206]`. Their six receipt hashes are
distinct, all 72 arm tensor receipts match their recorded SHA-256 digests, and
the combined valid-pilot wall time is `2118.4382774129044 s`. Including prior
campaign work, the valid fixture, and the preserved invalid attempt-01 cost,
the measured campaign use is `36993.01111694591 s`, leaving
`27806.988883054088 s`. This exceeds the single-boundary supervisor cap of
`24000 s`. The boundary output root is absent, the three Phase 52 programs
still pass `py_compile` and `git diff --check`, and the scientific target,
criteria, vetoes, hardware class, and nonclaims are unchanged. The skeptical
audit therefore passes for the trusted GPU boundary launch.

### GPU launch infrastructure blocker

The first trusted-launch request after the measured audit was rejected before
process creation because the automatic permission reviewer returned HTTP 503.
No GPU process started and no boundary output root was created. This is neither
a scientific veto nor a harness failure. Execution remains paused at the same
reviewed command and budget boundary until the user explicitly approves a retry
after being informed of the permission-review failure.

A second trusted-launch request, made after the user reported switching the
API, was also rejected before process creation with HTTP 503 from the same
permission-review service. The boundary output root remains absent and no GPU
or TensorFlow process was launched. The API change therefore did not resolve
the execution-boundary blocker; a non-escalated GPU launch would violate the
repository's trusted GPU policy and is not an acceptable workaround.

A third trusted-launch request, made after the user reported repairing the
gateway, was again rejected before process creation with HTTP 503. The output
root and process checks remain clean. This is a repeated external execution
boundary failure; it does not invalidate the six-bank evidence or the Phase 52
scientific design, but it prevents the GPU boundary and all dependent reporting
from running in this session.

On 2026-08-27, after the user reported another gateway reconfiguration, a
fourth request for the unchanged trusted boundary was rejected before process
creation with HTTP 503. The boundary output root is still absent and no
TensorFlow/GPU process exists. The failure is therefore in the permission
review service rather than in the Phase 52 command or its inputs.

The follow-up one-line elevated probe `true` completed successfully on
2026-08-27. This confirms that the permission gateway is currently accepting a
minimal trusted request. The probe did not import TensorFlow, initialize a GPU,
create the boundary output root, or consume the Phase 52 boundary budget, so
the unchanged GPU boundary is ready for an explicitly requested retry.

The subsequent explicitly requested retry of the unchanged GPU boundary was
rejected before process creation with HTTP 503. Thus the minimal probe is
accepted while the long TensorFlow/XLA boundary admission still fails. No
process, GPU initialization, output root, or boundary wall time was created by
this attempt. A shorter wrapper, indirect launch, or non-escalated execution
would be a policy workaround and is not permitted; the reviewed boundary must
wait for the permission-review service to accept it.

The bounded admission ladder in
`docs/plans/bayesfilter-gateway-admission-debug-reproducer-2026-08-27.md`
subsequently passed the full 1,245-byte Phase 52 argument shape with both
`60s` and `24000s` timeout values, CPU-hidden TensorFlow import, a tiny trusted
GPU/XLA operation with verified memory growth, and the real Phase 52 runner
under `--help`. These results do not run or validate the experiment, but they
rule out each isolated command feature as a sufficient cause of the 503. The
remaining diagnosis is an intermittent permission-review backend failure on
the combined scientific launch request. The boundary is ready for one clean
retry of the unchanged command; no wrapper or policy bypass is needed.

That clean retry was submitted after the user switched to another gateway. The
tool initially yielded a running-cell identifier while approval was pending,
but its completed result was another pre-process HTTP 503 with request ID
`ca16e829-1eaf-4dd9-8767-c6713c752e7d`. A subsequent process and output-root
check confirmed that no Python/TensorFlow process started and no boundary
artifact was created. The gateway problem therefore remains unresolved; the
transient running-cell report is not launch or scientific evidence.

The independent diagnostic at `/tmp/gateway_admission_repro_standalone.sh`
then narrowed the blocker further. Its non-GPU modes pass, but the bounded
`tensorflow-gpu` invocation was rejected before process creation with HTTP 503
and request ID `3fe18c6e-53ab-414a-b923-2d260650df26`. This reproduces the
gateway failure without repository imports, Phase 52 inputs, scientific code,
or artifact writes. It confirms an external GPU-request admission failure and
does not consume the Phase 52 campaign budget.

The same standalone script was extended with a tool-free `codex-model` mode
that calls ephemeral, read-only `codex exec`. The managed attempt to start that
mode was itself rejected by the outer permission reviewer before Bash started,
with HTTP 503 and request ID `658312a2-13e2-4759-99dc-a492e90dd2c9`. This shows
that the current blocker is not specific to TensorFlow or GPU initialization;
the outer review backend also fails while admitting an independent Codex
gateway-health command.

On 2026-08-28, after the user changed the gateway configuration and approval
process, the same tool-free managed probe was retried. It was rejected before
Bash started with `Error running remote compact task`, HTTP 503, and request ID
`eef0ca4b-1470-4d4f-b0a2-04a6993becae`. No nested Codex process, probe state,
GPU process, or Phase 52 output exists. The approval/gateway blocker therefore
remains active, and the scientific boundary was not retried or charged against
the campaign budget.

The replacement governance now designates a visible managed-session
TensorFlow/TFP GPU/XLA run as trusted when the structured result records
`owner_designated_managed_session_visible_gpu_trusted` and the other listed
device/provenance conditions hold. Phase 52 already records physical and
logical devices, TF32, XLA, memory growth, source hashes, command, seeds, and
versioned output, and it launches no HMC, package mutation, network fetch, API
change, or scientific/default promotion. The boundary runner was therefore
repaired only to add the exact trust-basis field to its device and run-manifest
records. No scientific or numerical setting changed. The generic remote
reviewer remains unavailable, but it is no longer execution authority for this
owner-designated managed-session route.

The first managed-session attempt on 2026-08-28 was admitted without a remote
review, then failed closed in `2.24441798 s` because the ordinary sandbox hid
all physical GPUs. The trusted retry was admitted and initialized both RTX 4080
SUPER devices with memory growth, proving that the approval and GPU boundary
now work. It stopped in `4.020615832 s` before creating the output root because
the six pilot receipts recorded corrected Phase 28 runner SHA-256
`c0b793ab10bd8d69cec22347c3beba00b5dd15e77e129f61b25d8dc585b9b703`
while the current file hashed to
`e06845ee3f16773f181380c35297beaa2c4a489561c4b7d642c89853bb8ace1b`.
Recovery of the exact recorded Git blob and a direct byte comparison showed
that the sole difference was one missing trailing blank line after the module
entry point. The boundary now binds both hashes and the audited difference: it
requires `c0b793...` in every pilot receipt, requires `e06845...` for the
current executable, and records equivalence identifier
`one_trailing_blank_line_only_verified_2026_08_28`. This is a localized
provenance repair under the unchanged evidence contract; the unused output
root permits one trusted retry.

### Implementation-readiness repair audit

The first Phase 52 scaffold was copied from Phase 51 and was wrong relative to
this phase's question even though it parsed. It still required Phase 47/49
three-bank endpoint replay, exposed only identity and geometry arms, reused a
single `proposal_r` field, and retained a geometry-only fixture. A successful
command from that scaffold could not have answered the fresh three-arm
question. Before any experimental command, the scaffold was repaired to:

- accept exactly six fresh Phase 52 pilot roots and the predeclared M0 seeds;
- remove old endpoint-hash replay and use Phase 50/51 only as frozen context;
- store `proposal_q`, `proposal_support`, and `proposal_geometry` separately;
- use `proposal_q` only for the annealing bridge and the selected arm density
  at both MH endpoints;
- run identity, support, and geometry from one serialized initial tensor and
  common resampling-seed ledger; and
- check both non-symmetric proposal corrections in the finite fixture at
  `beta=0` and `beta=1`.

All three repaired runner/fixture/report sources pass `py_compile`,
`git diff --check`, and a stale-symbol
audit. This closes an implementation/harness defect only; it is not evidence
for the candidate or target. The skeptical audit now passes for fixture and
pilot execution. The GPU boundary remains conditional on all six pilots and
the fixture passing.

Attempt 01 then exposed a second harness defect: the exact pilot commands named
the historical particle-authority runner rather than the corrected-theta Phase
28 runner. Its first receipt had schema `authority_pilot.v1`, status
`PASS_GATE`, no corrected calibration block, and M0 seed `5301`; it is wrong
relative to the Phase 52 input contract. The receipt is preserved but excluded.
The repair binds attempt 02 to corrected runner SHA-256
`c0b793ab10bd8d69cec22347c3beba00b5dd15e77e129f61b25d8dc585b9b703`,
which exactly matches the runner hash in the valid Phase 47 receipts. The full
failure record is
`docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase52-attempt01-harness-failure-2026-08-26.md`.

## Pre-mortem

| Misleading outcome | Distinguishing check | Response |
|---|---|---|
| fresh arms accidentally use different initial clouds | per-arm tensor hashes | hard veto and fresh root repair |
| support/geometry proposal density is omitted from MH correction | fixture, source audit, stored current/candidate logs | hard veto; no scientific interpretation |
| bootstrap interval looks favorable due to six-bank noise | raw paired differences, exact sign count, finite-replication nonclaim | retain as descriptive or trigger another repair |
| geometry improves spread but worsens ESS again | report all primary and secondary metrics separately | do not promote; treat as mixed candidate |
| pilot or boundary runtime exhausts budget | run manifest and campaign ledger | stop before the next launch; do not relax gates |

## Commands and artifacts

Six valid pilot roots are fresh and unique below
`phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-01` through
`pilot-06`. The invalid attempt-01 `pilot-01` is preserved outside that retry
root and is never pooled. The pilot lane is CPU-hidden. The boundary uses trusted GPU/XLA with
`TF_FORCE_GPU_ALLOW_GROWTH=true`; the report is CPU-hidden. All writes are
versioned and refuse overwrite.

No NeuTra optimizer update, HMC chain, LEDH route, training-data selection, or
default-policy change is launched.

The fresh seed ledger is fixed before generation:

| Pilot | Corrected-generator root seed | M0 seed |
|---|---|---|
| `attempt-02/pilot-01` | `[20260826, 5101]` | `[20260826, 5201]` |
| `attempt-02/pilot-02` | `[20260826, 5102]` | `[20260826, 5202]` |
| `attempt-02/pilot-03` | `[20260826, 5103]` | `[20260826, 5203]` |
| `attempt-02/pilot-04` | `[20260826, 5104]` | `[20260826, 5204]` |
| `attempt-02/pilot-05` | `[20260826, 5105]` | `[20260826, 5205]` |
| `attempt-02/pilot-06` | `[20260826, 5106]` | `[20260826, 5206]` |

The recorded lower bound through Phase 50 was `28960.22324898499 s`.
Adding the Phase 51 fixture, boundary, and report (`1.3551 + 5558.9085 +
0.3726 s`) gives `34520.85944898499 s` used and
`30279.140551015007 s` remaining before Phase 52. The valid fixture used
`1.7999030420323834 s`; invalid attempt 01 used `351.91348750598263 s`.
The resulting lower-bound remainder is `29925.427160466992 s`. The `24000 s` ceiling governs the single
long GPU boundary attempt. Each pilot records its own wall time; the combined
pilot-plus-boundary projection must remain below the global remainder before
the GPU launch.

Exact fixture command:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase52_fixture_2026_08_26.py --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/fixture --seed 20260826 5200
```

Exact attempt-02 commands use the hash-verified corrected-theta runner. It adds
`100` to the root seed for M0, so the command and M0 seed are deliberately
different:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-01 --particles 256 --calibration-particles 64 --seed 20260826 5101 --arms both
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-02 --particles 256 --calibration-particles 64 --seed 20260826 5102 --arms both
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-03 --particles 256 --calibration-particles 64 --seed 20260826 5103 --arms both
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-04 --particles 256 --calibration-particles 64 --seed 20260826 5104 --arms both
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-05 --particles 256 --calibration-particles 64 --seed 20260826 5105 --arms both
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-06 --particles 256 --calibration-particles 64 --seed 20260826 5106 --arms both
```

Exact trusted GPU boundary command:

```bash
TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase52_2026_08_26.py --pilot-root-1 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-01 --pilot-root-2 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-02 --pilot-root-3 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-03 --pilot-root-4 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-04 --pilot-root-5 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-05 --pilot-root-6 docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/pilot-06 --fixture-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/fixture --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/q20-paired
```

Exact CPU-hidden report command:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/report_ssl_lstm_q20_parameter_authority_corrected_phase52_2026_08_26.py --fixture-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/fixture --boundary-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/q20-paired --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/report
```

## Interpretation branches

| Branch | Meaning | Next action |
|---|---|---|
| `fresh_geometry_uncertainty_compatible` | all primary spread-difference bootstrap upper bounds are nonpositive | retain geometry as a role-limited nominee; require larger replication/downstream validation |
| `fresh_geometry_uncertainty_incompatible` | at least one primary upper bound is positive | classify the Phase 51 pattern as unresolved/seed-sensitive; keep whitening/HMC/LEDH closed and write the next repair |
| `phase52_hard_veto` | pilot, target, measure, pairing, density, device, or artifact contract failed | repair the harness under a fresh root; make no scientific interpretation |

No branch establishes a population limit, posterior law, IID Gaussian law,
mode-discovery theorem, method superiority, or default readiness.

## Terminal execution and decision

The trusted managed-session boundary completed with status
`PASS_V3_4_FRESH_PAIRED_BOUNDARY`. Both RTX 4080 SUPER devices were visible,
memory growth was verified before logical-device initialization, and the
manifest records TF32, XLA, the exact command, seeds, source hashes, wall time,
and trust basis `owner_designated_managed_session_visible_gpu_trusted`. All six
replicates and all 18 arm receipts passed the engineering and numerical gates.

The deterministic CPU-hidden report completed with status
`PASS_V3_4_FRESH_PAIRED_REPORT` and selected branch
`fresh_geometry_uncertainty_incompatible`. The 95% upper endpoints for the
geometry-minus-support range differences were nonpositive for
`covariance_offdiag_max_abs` and `negative_mode_fraction`, but the endpoint for
`theta_mean_0` was `0.1823817845 > 0`. The predeclared conjunctive criterion
therefore failed. The frozen equal-weight geometry is rejected as a
spread-reduction nominee; the target, harness, and broader mode-aware proposal
idea are not invalidated.

The terminal records are:

- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase52-result-2026-08-28.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase52-repair-refresh-2026-08-28.md`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/q20-paired/result.json`
- `docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase52-fresh-paired-uncertainty-replication/attempt-02/report/result.json`

The conservative campaign remainder is `4483.656990259071 s`, below the
observed `5558.9085 s` cost of the prior three-bank boundary alone. It cannot
fund a scientifically valid phase with disjoint calibration, reference
construction, and untouched validation. The campaign stops at this genuine
budget blocker. No Phase 53 is active, and the six Phase 52 banks must not be
reused for tuning.
