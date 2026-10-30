# TensorFlow public-tuner repair for surrogate HMC

The owner asked why execution stopped and instructed continuation. The earlier
stop applies to retained sampling, not to implementation of its prerequisites.
Continue within the authorized all-model campaign. Its remaining budget before
this repair is 62.9676 GPU hours and 14.9366 CPU-worker hours; LG1 inference
attempts 4 and 5 remain unused. Initially reserve at most 1 GPU hour and 1 CPU
worker hour for this repair's numerical checks, from the shared repair reserve.
Each GPU job has a 900-second timeout. Preserve every attempt and charge it.

Question: can the existing public ordinary tuner own a TensorFlow-only,
XLA-compiled adaptation/search/fresh-verification/retained handoff for typed
endpoint-corrected position fields? This is an engineering prerequisite to
testing surrogate HMC parameter inference, not a posterior result itself.

Implement an explicit verified configuration/result through `tune_hmc_kernel`.
Reuse its TensorFlow adaptation and kernel; old diagnostic results remain
non-authoritative. After nomination, freeze mass/epsilon/L, start four separately
declared chains with a fresh stateless seed, archive discarded warmup and fresh
verification, and require finite health, all-chain movement, valid endpoints,
the declared energy bound, and maximum split rank/folded R-hat <= 1.01. Archive
and bind the exact target/force/source identities before retained handoff.
Do not use NumPy in runtime decisions or call a private alternative tuner.

Evidence contract: independent Gaussian moments and rank/R-hat reference
calculations, existing kernel invariants, graph/XLA parity on identical fixed
states, source/binding mutation rejection, and an actual public-call-to-retained
archive test qualify implementation. Stuck chains, nonfinite states, endpoint
failure, corrupted artifacts, scope drift, or missing fresh verification veto
handoff. Acceptance, fallback counts and runtime explain behavior; they cannot
replace R-hat or establish posterior accuracy. The full LG1 DCTR consumer then
receives a bounded wiring/compile/cost check before any inference launch.

Defaults and assumptions: retain the existing nomination's four-chain FP64
affine mass ladder, powers-of-two trajectories and TFP dual averaging as explicit
starting hypotheses, with caller-supplied counts, shrinkage/floors and scales.
Their failure mode is bad geometry or excessive regularization; record measured
covariances, chosen steps/L, movement and failed nomination. Fresh chains must
be dispersed in every coordinate, independently supplied in raw coordinates.
At least 1,000 even verification draws prevent tiny diagnostic runs gaining
authority; this minimum is a conservative engineering requirement, not a
convergence theorem. Retained posterior gates remain separate. XLA defaults on
for this new route only after qualification; diagnostics for rank sorting use a
stable non-XLA TF graph because tied-rank segment sizes are data dependent.

Pre-mortem: identical initialization can conceal bad mixing; reused selection
draws can falsely certify a selected candidate; unchecked XLA assertions can
silently accept invalid numerics; stale sources can misbind an artifact. Test
each directly and perform host-side finite/domain guards before compilation.
Skeptical audit: source scope is fixed, the baseline is an independent exact
target, no proxy grants inference promotion, budgets and stop conditions are
explicit. Proceed with implementation and bounded checks. A failed candidate
is preserved for bounded repair; implementation failure blocks only dependent
sampling, not continued repair.

Artifacts: new directories under
`docs/plans/artifacts/surrogate-hmc-all-models-20260919-tuner-repair-*` with command,
environment, CPU/GPU and memory-growth status, source hashes, timings, tests,
decision and accounting. Final result must state whether the tuner prerequisite
is removed and which posterior comparisons remain unrun.
