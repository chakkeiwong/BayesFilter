# Corrected-source surrogate-HMC computation results

Status: **COMPUTATION_EVIDENCE_COMPLETE_POSTERIOR_NOT_RUN**.
The goal remains testing surrogate HMC. Computation checks do not complete the posterior experiment.
FP64 is the same finite program in higher precision, not a model oracle. Errors below are
100 times abs(candidate-reference)/max(1,abs(reference)); the threshold is 1%.

| Model | T / state / parameter dimension | Precision | Value error % | Score error % | Force error % | Four-chain force s | Candidate/baseline time |
|---|---|---|---:|---:|---:|---:|---:|
| lgssm | 50 / 3 / 4 | uniform | 1.99562e-05 | 0.000542334 | 0.000429546 | 25.936 | 0.7933 |
| ksc | 50 / 1 / 2 | uniform | 1.88581e-05 | 0.000129974 | 0.000123848 | 9.373 | 0.9858 |
| predator_prey | 50 / 2 / 6 | uniform | 4.90278e-05 | 0.000836394 | 0.000860868 | 16.257 | 0.8753 |
| sir_d18 | 50 / 18 / 3 | fp64_recursion_fp32_pairs | 3.10251e-06 | 0.196062 | 0.194316 | 26.720 | 0.6798 |
| range_bearing | 20 / 4 / 2 | uniform | 0.00126627 | 0.209931 | 0.145513 | 6.383 | 0.8188 |

N4032/K2016, TF32 off, GPU/XLA and analytical all-parameter recursion apply throughout.
Uniform precision means FP32; SIR uses FP64 recursion with FP32 pair kernels.
Runtime ratios are descriptive paired timings with few repeats. No statistically supported ranking is established.
Mechanics runs contain two transitions per chain; acceptance does not establish convergence.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| COMPUTATION_EVIDENCE_COMPLETE_POSTERIOR_NOT_RUN | Saved all-model checks | See per-model issues in result.json | Four local positions | Resolve practical posterior execution | Convergence or canonical LEDH admission |

| Inference status | Finding |
|---|---|
| Hard veto screen | Source, scope, finite outputs, precision, binding and mechanics checked separately. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Runtime ratios, fixed-position costs, tiny-pilot acceptance and energy error. |
| Default-readiness | FP32 remains default; mixed SIR is an optional scope. No posterior or canonical LEDH promotion. |
| Next evidence needed | Target-specific posterior tuning, cumulative convergence and adequate references. |

The extension used 8.3656 of 24 authorized worker-wall hours; 15.6344 hours remain.
Unused time is retained. The next posterior phase belongs to the existing master, which remains unfinished.
It has not started: the benchmark prior and tiny mechanics do not resolve the scientific target/reference
question, and practical long-chain cost is unestablished. L1 timing scenarios are not warmup bounds.

Post-run skeptical review: arithmetic agreement can coexist with model approximation error; local
force agreement can coexist with poor mixing. The tiny local test is the strongest alternative
explanation for apparently good mechanics. Wider posterior checks could overturn that impression.
Evidence paths, exact scopes, source hashes, commands, timings and memory measurements are preserved
in result.json and its linked worker manifests.
