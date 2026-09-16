# iAPF objective-underflow guard and consumer verification

This is the next engineering repair inside the already-authorized score master.
The preceding calibration allocation is closed at 64 charges and two launches;
it is not extended by this plan.

## Question and evidence contract

Does a fail-closed check reject a profiled density fit when its represented
loss is zero but its normalized shape residual is positive, while leaving
healthy fits unchanged? The baseline is frozen source `f5a4d411`, with the
existing underflow diagnostic but no validity guard. The candidate changes
only validity/convergence status for that diagnosed condition. It must not
alter the objective, optimizer, fitted coefficients, healthy scores, default
controls, target, or analytical derivative.

The pass criterion is exact healthy-path parity and rejection of a reproduced
underflow fixture through the fitting and consuming adapter call chain.
Unexpected healthy-path changes, lost diagnostics, source drift, or inability
to reproduce the original condition veto integration. Score MSE, residual
size, speed and fit-boundary rates are explanatory, not tuning targets. No
score improvement, iAPF ranking, scientific promotion, or default readiness
follows from passing this check.

## Audit and assumptions

The saved calibration contains eight underflowing iAPF rows, including every
selected evaluation. The existing test fixture in
`tests/fixtures/iapf_density_underflow.json` provides a small reproducer.
The loss is amplitude-squared times a scaled residual norm. Zero represented
loss with positive represented relative residual identifies loss of the
objective's amplitude, rather than an exact residual match. Rejecting it is
a Class B numerical guard: it changes admission, not accepted numerical values.

An absolute-gradient tolerance can legitimately be met at very small density
amplitudes. This check therefore makes no assertion that the preceding
stopping rule was mathematically inconsistent with its absolute tolerance.
It prevents an underflowing objective from certifying a usable fitted result.
Boundary activity alone is diagnostic and is not silently turned into a
continuation veto. The exact-zero/positive-residual test already exists in
the implementation; no new threshold is introduced.

The skeptical audit identifies no need for a new optimization objective or
comparison campaign to evaluate this guard. Retain negative fixtures and
verify the actual adapter invokes the guarded recursive fitter. A candidate
rejection must remain distinct from infrastructure failure and from rejection
of the whole research direction in subsequent calibration design.

## Execution and budget

Create an isolated checkout from `f5a4d411`, patch only the fit guard and
necessary tests, and run CPU/XLA reference tests with CUDA explicitly hidden.
Use `tftwogpu`, one intra/inter-op thread and one OMP thread. Preserve stdout,
exit status and timings under the active score artifact root. Maximum 1,800
CPU process-seconds, three focused test attempts and one 180-second GPU smoke
launch. The GPU smoke is optional if no runtime/device arithmetic changes;
if used, it requires trusted access and verified memory growth.

Run the existing objective/analytical-gradient reference, a healthy exact-fit
case, the saved underflow fixture and the consumer rejection regression.
Update positive-path fixtures only when their prior inputs are independently
shown to violate the new condition; preserve the original as a negative case.
Do not loosen tolerances or remove assertions to make the suite pass.

Before integration, compare relevant main files to the frozen predecessors
and preserve concurrent edits. Record the exact patch and source revision.
Close with a result, decision and refreshed next-phase plan. The subsequent
research phase must handle rejected calibration candidates explicitly, include
the frozen-control claim baseline and both conditional heuristic tables, use
fresh data, and budget worst-case fits before execution. Full numerical-control
calibration and fitted-moment integration into LEDH remain open master work.

## Execution and repair decision

The isolated repair checkout was created from
`f5a4d411a4a0197fe9c1d25c24573337a633c96a` and changed only the fit validity
predicate and its preserved negative fixture. The main checkout received the
same two-file patch after the isolated comparison. The focused command was:

```
env CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_PRELOAD_CUSTOM_OP=0 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 OMP_NUM_THREADS=1 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q \
tests/highdim/test_younis_score_master_iapf_tf.py \
tests/highdim/test_younis_score_master_nonlinear_iapf.py
```

It passed with 19 tests in the isolated checkout (41.62 seconds) and 19
tests in the main checkout (43.17 seconds). Both runs emitted only the known
TensorFlow Probability deprecation warnings. Bytecode compilation and
`git diff --check` also passed; complete logs, hashes, revisions and the
decision are in
`docs/plans/artifacts/younis-kdm-score-master-20260914/run-20260914-140520-01/iapf-fit-underflow-guard-repair-20260916/`.

The repair is admitted as a Class B fail-closed guard. It sets
`objective_underflow = (loss == 0) and (normalized_shape_residual > 0)` and
requires `not objective_underflow` in the existing `valid` flag. The recursive
fitter propagates that flag and the iAPF adapter rejects the fit before score
execution. Finite diagnostics are retained. Healthy fitted values, gradients,
and objective arithmetic are unchanged. A GPU smoke was not run because this
is an admission-only change and the preceding calibration allocation is
already closed at 64 charged attempts and two launches.

The phase is complete as an engineering repair, not as a scientific
calibration or promotion phase. All prior selected calibration rows remain
partial-phase evidence and are not restamped as valid. The next research phase
must have a new output root and budget, use fresh partitions, reject invalid
fits before comparison, include the frozen-control claim baseline, complete
both weak and curved conditional heuristic tables, and obtain independent
replications. No old calibration row can substitute for those requirements.
