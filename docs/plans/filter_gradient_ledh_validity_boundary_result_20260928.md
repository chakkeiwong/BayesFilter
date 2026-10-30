# LEDH native value rejection-boundary result

The supplied-input native value owner now rejects failed resets explicitly.
It returns NaN `value`, false `program_valid`, a numerical failure code and the
first executed rejected-reset index. `raw_value` and `finite_program_valid`
preserve the old finite-only diagnostics and must not substitute for an
accepted value. Codes are0(usable),1(nonfinite lifecycle) and2(failed or
incomplete reset); index-1 means no executed reset was recorded as rejected.

The registered legacy wrapper remains unmigrated and still has NumPy/host
numerical execution. This component repair therefore does not close F07 or
admit the complete public consumer. It changes no reset formula, precision,
tolerance, seeded realization, flow setting, derivative or canonical method.

Current-source characterization04662 reproduces the actual lost diagnostic:

| Fixture | Prior finite-only status | Reset validity | New public status/code | CPU raw eager/XLA value difference |
|---|---|---|---|---:|
| Composed, three stages |true|true,true,true|usable /0|3.649284e-8|
| Dual-trust |true|false,true,true|rejected /2, index0|3.528351e-6|
| One-step |true|false|rejected /2, index0|0|

The GPU dual-trust raw difference remains1.682840e-5. These rejected raw
programs are not declared numerically equivalent. The old complete-value veto
is preserved as evidence; the repaired boundary makes its unusable outcome
explicit. The earlier condition proxy201 does not establish severe
ill-conditioning and is not used to excuse the mismatch.

All reported raw fields are bitwise unchanged between the three CPU before/
after fixtures04662/04663, with identical frozen input hashes, controls and
reference source identities. The only changed repository runtime dependency
in those manifests is `ledh_canonical_value_program_tf.py`. The healthy
composed case retains the unchanged1e-6 absolute/relative comparison against
the frozen9d8202b77 value/flow authorities with the shared reset. Tests also
check changed operands, exact replay, one trace, enclosing HLO and no host
callbacks. The baseline and its NumPy code are diagnostic authorities only.

CPU04663 and GPU04666 each pass three boundary fixtures. CPU04664 and
GPU04667 each pass nine full-value/resampling regressions, including annealed
flow and nonfinite initial/predicted/observed states. Numerical bounds still
apply to healthy complete records. Rejected records require the correct
invalid public result, validity/history/NaN-mask agreement and explicit raw
difference reporting; a passing rejection test is not a raw-equivalence pass.
Readback/policy04665 passes161 checks; final04668 passes162 checks. Ruff and
whitespace checks pass. No policy allowance was added.

Both GPU workers ran onGPU2, UUID
`GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`, selected by the existing
non-desktop availability policy. TensorFlow2.19.1, TF32 enabled, CUDA visibility,
trusted-session basis and memory growth on the sole visible physical GPU are
preserved in the process logs and final structured readback. Growth was verified
before logical-device initialization; whole-device preallocation was disabled.
This is correctness evidence, not a before/after GPU cost or memory benchmark.
Bitwise raw invariance was measured on CPU, not inferred for GPU.

Artifacts are raw04662--04668 under the existing campaign root.04662 also
preserves the pre-guard characterization test source.04665/04668 bind complete
raw comparisons to source, input and manifest hashes.04668 separately binds
GPU reports/HLO, log hashes, selected hardware and growth-policy evidence.
Archive/receipt `ledh-validity-04668-evidence.tar.gz` and
`ledh-validity-04668-verification.json` preserve the cohort and source checkpoint.

The CPU unit closes at5/8 workers,83.297397/2400 process-seconds; the GPU unit
closes at2/4 workers,72.800756/1200 process-seconds. Global charges are
108237.060357 CPU and96144.148720 GPU seconds;25.934150 CPU and25.293292 GPU
hours remain inside the unchanged56/52-hour caps. No worker remains active.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Accept native rejection boundary | Healthy reference and rejected-status tests pass CPU/GPU | No healthy numerical regression; CPU raw fields unchanged | Limited supplied-input fixtures | Migrate the registered wrapper using this owner | Complete consumer or canonical-method admission |
| Preserve rejected raw discrepancies | CPU/GPU dual-trust raw mismatch is reported | Invalid reset blocks usable value | Exact FP32 reset sensitivity remains | Diagnose only if needed for a valid reset scope | Eager/XLA raw equivalence or severe ill-conditioning |
| Keep full repair program open | Registered consumer still uses legacy execution | Public/default and cost gates remain open | Callback ownership, seeded orchestration, actual consumers and costs | Bounded seeded-owner/public-wrapper migration | Whole-program completion or main merge |

Skeptical review: masking an invalid value could hide a changed raw algorithm.
The separate raw fields, unchanged CPU records and preserved residuals reduce
that risk. This repair deliberately strengthens public usability at the native
owner boundary; it does not make all downstream consumers use that boundary.
Any direct use of `raw_value` as an admitted likelihood would be a new defect.
GPU cost attribution and actual public call-chain evidence remain necessary.
The primary agent reviewed this unit; no independent reviewer was used.
