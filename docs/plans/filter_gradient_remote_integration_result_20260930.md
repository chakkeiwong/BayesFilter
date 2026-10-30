# Current-main integration and XLA execution repairs

The repair branch integrates main023e10610 into cf76d2cca. The four overlapping
runtime files merged automatically. Incoming KSC observation-offset derivatives,
corrected mixture interpretation, right-sided inverse-CDF search, linear-memory
ancestor counts and the native annealed analytical-score branch are retained.
This checkpoint does not merge the repair branch into main.

The incoming SQMC input preparation had numerical Python loops outside its
compiled score owner. It now has a retained default-XLA owner with dynamic seeds
and TensorFlow control flow. The shared Halton helper preserves TFP0.25's salted
split, digit permutations, trailing-zero correction and ordering, including the
different ordinary and enclosing-XLA uniform conversions. The fixed prime table
replaces dependency NumPy shape preparation; it does not define a new random
method. LGSSM/full-LGSSM defaults and matrix constants use tensor operations
under compiled preparation. Schema/name formatting remains host configuration.
The expanded benchmark trace summary now reduces rows/columns across stacked
time tensors inside its compiled owner, preserving nomination decisions.
An explicit selected-index shape restores enclosing-XLA tracing for annealed
resampling without changing the selected indices or score composition.

The plan is filter_gradient_remote_integration_20260930.md. Exact commands,
source hashes, seeds, TF2.19.1 environment, wall times, CPU/GPU placement and
verified GPU memory growth are in run manifests05170 onward beneath
/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917.
GPU checks use the available compute GPU3, UUID
GPU-b8045e28-4433-ec7a-77a5-db0636748322. CPU runs hide GPUs explicitly.

Evidence includes96 live input comparisons,32 live endpoint theta/seed cases,
actual annealed CPU/GPU XLA with independent derivative checks and refusal
contracts, original full/diagonal parameter preparation, callback directions,
independent KSC references, ordering, trace nomination and source-policy checks.
Final boundary validation covers N3/9/27/81/1024/1008/1020 with D2/3/10, both
dtypes and ordinary/enclosing streams:56 backend/dtype/count/context cases.
The exact source-bound terminal readback records applicable runs and owners.
The guard covers326 sources with1505 exact configuration, orchestration or
reference allowances; no numerical filtering/score recurrence is allowed.
This is scoped enforcement, not proof about every repository file.

All failed workers are retained with their original verdicts:

| Runs | Finding and disposition |
| --- | --- |
|05176|Eight endpoints passed; annealed tracing lost static N. tf.ensure_shape repairs that metadata;05180/05181 qualify CPU/GPU.|
|05177--05179|Numerical comparisons passed; new refusal fixtures omitted required arguments or hit an earlier guard. Correct fixtures preserve the original API/guard order.|
|05183|21 passed; seven diagonal defaults hit a new FP64 constant-coercion bug and one incoming test had stale error wording. Explicit FP64 constant and corrected test expectation pass05186/05187.|
|05184|Nine passed; static trace HLO introspection omitted its literal epsilon=None argument. Correct query passes05185.|
|05192|17 passed; three assertions hard-coded CPU despite successful GPU execution. Output placement now follows the actual input; all three pass05193.|
|05198/05199|FP32 passes; two ordinary eager GPU FP64 Halton coordinates fail at N1020,D10. Independent attribution05200 proves an original GPU pow defect; see below.|

In the original eager GPU Halton implementation,19**2 evaluates as
361.00000000000006. Floor division then gives wrong radix digits at one-based
sample indices361 and722. CPU and XLA both evaluate361 exactly, and their digits
match independent integer division. The repaired ordinary stream agrees with
the original CPU authority within2.22e-16 and the original enclosing GPU/XLA
stream exactly. The failed eager-GPU comparison differs by8.74763e-4. Run05201
renews the larger GPU boundary cases against those independently checked
references at the original tolerances. No runtime change, RNG substitution or
tolerance relaxation was needed. The known wrong eager-GPU digits are not an
equivalence requirement; raw failures and their attribution remain archived.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
|Accept scoped incoming execution repairs|Live inputs, values/scores, errors and decisions retain checked authorities|No unexplained failure in qualified scopes; original GPU reference defect recorded|Only declared dimensions/counts/callers are tested|Finish resource acceptance and terminal affected-caller dispositions|Whole program completion, unrestricted capacity or canonical LEDH admission|
|Retain integration on repair branch|Incoming corrections preserved with source witnesses|Main merge withheld while resource obligations remain|Unmeasured changed preparation costs|Matched cost and lifetime checks|Universal speed improvement or zero native-memory growth|

This is an implementation comparison, not a stochastic-method ranking or tuning
campaign. No trained-map, posterior or HMC claim follows. Adaptive iAPF/KDM stay
deferred; canonical LEDH rebuilding remains outside this campaign and unsupported
claims remain blocked. Earlier core witnesses remain applicable only where their
executed dependency bytes did not change; do not rerun preintegration hash-bound
readback as though it described the merged source tree.

Primary-agent terminal review: the strongest misleading-pass risk was tiny
Halton fixtures hiding digit boundaries. The extended test exposed a real
reference defect, which was resolved by independent integer arithmetic rather
than loosened bounds. Exact random words/ordering and both invocation contexts
are retained. A mismatch in a changed active caller or a new unexplained stream
case would reopen this conclusion. Performance, native allocation lifetime and
remaining affected-use coverage still require the planned resource phase.

Terminal05202 passes161 readback/policy checks. The verified evidence archive is
docs/plans/artifacts/filter-gradient-repair-20260917/remote-integration-05170-05202-evidence.tar.gz;
the adjacent verification JSON records its SHA-256 and verifies every member.
