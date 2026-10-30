# Shared score-study input execution result

The Gaussian and nonlinear adapters now prepare initial/process/reset normals
and resampling uniforms in the cached `make_score_inputs` XLA factory. Gaussian
twist initial ancestors and uniform concatenation execute there too. Numerical
filter and analytical derivative authorities are unchanged. The factory has a
fixed int32 seed-table signature and defaults to JIT. Seed labels remain host
metadata. Existing Philox word/conversion helpers preserve the prior streams;
no RNG migration is performed.

The owner-deferred adaptive iAPF, KDM variants and Gaussian rows requesting KDM
pilot collection keep their existing eager preparation branches. AST comparison
checks their unchanged draw assignments without executing those algorithms.
This is an explicit scope deferral, not compliance or scientific admission.

Baseline: 0025cdf97. Runs 04897--04913 contain 17 serialized workers, all passing,
using 248.490935 CPU and 365.218921 GPU process-seconds. Final readback/policy
04913 passes 164 checks. Static coverage is 315 sources with the same 1457 exact
exceptions; the new factory needs none. GPU2 UUID
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba` was used with trusted execution and
verified memory growth. CPU runs are explicit references. Commands, environment,
source hashes, timing and placement are in each run manifest and process log.

Validation includes dimensions 1/3, odd/even shapes, changed seeds, ordinary
and twist uniforms, FP64/FP32, CPU/GPU, exact replay, one trace, unchanged HLO
across seeds and no pfor/host callbacks. Raw Philox words and uniform values
match exactly. FP64 normal draws match exactly. Maximum FP32 normal error is
5.96046e-8 CPU / 2.38419e-7 GPU, below the existing 2e-6 absolute bound.

There are 112 live complete-endpoint cases across the two backends: Gaussian
Kalman/UKF, bootstrap, prior SIS, adapted/resampling and adapted SIS, twist;
nonlinear EKF/UKF, bootstrap, prior SIS and local-linear proposals; and Gaussian
and nonlinear LEDH/SGQF analytical-direction callers. Every FP64 record is exact.
FP32 maximum complete-record differences are 4.76837e-7 CPU / 3.33787e-6 GPU,
within the declared 5e-5 bound. All frozen-input controls and replays are exact.
Invalid numerical inputs preserve public refusal. Forty ancestor-history
comparisons match exactly; each diagnostic reproduces its ordinary program's
value/score/ESS bytes before serving as a label witness.

Previously qualified healthy frozen LEDH/SGQF cases, original invalid-scope
regressions and actual fixed fitted-APF Gaussian/nonlinear siblings pass. The
nonlinear FP64 CPU physical dataset generator is unchanged. Eight/eight reset
counts reuse the independently nominated mechanics scopes through04821; no
canonical tuning/admission artifact or scientific claim follows. The older
frozen-stream direction tests now inject arrays at the new factory boundary;
the separate live-stream tests exercise the actual factory.

| Descriptive full Gaussian twist endpoint | CPU before / after | GPU before / after |
|---|---:|---:|
| Cold seconds | 3.96404 / 4.34445 | 7.10865 / 9.29102 |
| Warm median milliseconds | 3.79712 / 2.37027 | 9.43573 / 6.99997 |
| Extra sampled warm host RSS | +24.718750 MiB | +23.835938 MiB |
| GPU allocator peak bytes | N/A | 48640 / 47104 |

Both arms include data, input generation, oracle, filtering and reporting under
the same preconfigured runtime descriptor, excluding TensorFlow import and
runtime initialization. Each arm uses one fresh process, two untimed warm-ups
and 30 timed calls. Both GPU arms pass uncontended preflight. Cost records are
numerically exact. Repaired sampled RSS rises only 0.019531 MiB CPU / 0 GPU
between the cold and warm observations, but this does not establish general
native-memory bounds or executable eviction. The host-residency increase and
GPU cold ratio 1.307 remain explicit acceptance findings.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain input execution repair | Streams, live/frozen consumers, labels and execution checks pass | No observed changed decision or new exception | Finite tested scopes | Commit scoped repair and retain all evidence | Universal branch stability |
| Record cost tradeoff | Matched complete-caller records pass | Added host RSS and GPU cold time remain open | Single fresh process per arm | Include in repeated costs/native-capacity work | Terminal memory/cost acceptance |
| Continue master | Other findings remain open | Main merge withheld | Geometry/status/angle, locator and source applicability | Next bounded geometry unit | Whole-program completion |

| Inference status | Result |
|---|---|
| Hard veto screen | All declared numerical/status/label checks pass |
| Statistically supported ranking | None; no replicated process cohort |
| Descriptive-only differences | Lower warm time; higher host RSS and cold time |
| Default-readiness | Scoped XLA input execution qualifies; broader admission open |
| Next evidence needed | Current-source dispositions, repeated costs and capacity |

Evidence: `run-04913/score-inputs-readback.json` binds source, result, manifest,
HLO and runtime-policy receipts. Fixture hashes are bound during readback to
unchanged tracked baseline bytes; worker manifests did not directly hash these
JSON fixtures. That provenance limit is retained explicitly. Archive
`score-inputs-04913-evidence.tar.gz` has 174 verified members, 5035107 bytes,
SHA-256 `9d82da9ae4dc8aedb1fb3e93d44cc5ea230eb59cbba25044cd6edc223387b2b5`.
All raw evidence is under the existing filter-gradient-repair-20260917 root.

Primary-agent review: the strongest numerical alternative is that wrapper
agreement hides changed discrete paths. The exact ordinary-output label
witnesses address that risk for the tested resampling cases; they do not prove
all seeds. Unchanged filter-source bytes and exact frozen controls separate
input conversion from equation changes. A different seed can expose a decision
boundary and remains a repair trigger. Single-process RSS/cold observations are
the weakest cost evidence and remain open. No independent agent was used.
