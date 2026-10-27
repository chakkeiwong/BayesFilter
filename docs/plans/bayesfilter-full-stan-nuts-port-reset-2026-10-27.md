# Stan Euclidean NUTS transfer: continuation record

Completed: 23 launches, 5052.046 worker seconds within the 5400-second ceiling;
all workers closed. The final Gaussian MH archive audit passed in attempt 23.
All 216 verified fixed-HMC members were assessed: 13 passed the posterior
screen in 7 of 21 fits. One regression64 search was partial at its per-fit
budget; its six verified members were all assessed. No inferential efficiency
ranking or production-default promotion follows from these counts.

The owner authorized the standalone plan, review and execution on October 27.
The active scope is the exception recorded at the top of AGENTS.md. Retired
public NUTS APIs remain excluded, and repository production defaults are
unchanged. Extensive unrelated dirty work belongs to other tasks and must be
preserved.

The implementation is `docs/benchmarks/stan_nuts.py`; the generic interface,
source coverage and final results are linked below. Native CmdStan 2.37.0 is
installed at `.localresources/cmdstan/cmdstan-2.37.0`. Numerical code uses
TensorFlow/TFP; the native C++ executables are independent comparators.

The direct port implements the positive-dimensional Euclidean multinomial
sampler, unit/diagonal/dense metrics, source-ordered adaptation and step search,
flat unconstrained initialization, thinning and retained continuation. It uses
TensorFlow Philox, so identical integer seeds do not reproduce Stan's native
MIXMAX/Boost stream. Named parameter constraints, the Stan compiler, generated
quantities and Riemannian algorithms are outside the agreed scope. Zero
dimensions use an explicitly local constant-evaluation convention.

The 128-test CPU suite passed without skips in attempt 16. The core and tests
have not changed since that run. GPU common-variate comparisons passed 182
cases, and complete lifecycle kernels compiled for all three metric forms.
Current launchers use distinct per-chain keys. Earlier calibration attempts
06/08 and stopped attempt 09 had overlapping keys and are preserved as invalid
for direct-arm statistical/layout evidence. Fresh calibration is in 11–13.

The main three-seed comparison is attempt 14; separate overdispersed runs are
attempt 17. NUTS passed the declared posterior screens on six targets and failed
on the centered funnel across all engines/seeds. Four parallel scalar CPU
chains were chosen from calibration. Native Stan had lower observed elapsed
costs; this campaign does not establish a time-to-precision ranking. The direct
port's lower total than TFP on regression64 came from warm-up; retained sampling
was slower. GPU banana cost and tree/RNG overhead remain unprofiled.

Fixed HMC uses the actual public standard tuner, with automatic mass
preparation, a 180-second per-fit search allowance and fresh retained assessment
of every verified member. The first seed is in 15/19/20, the second in 21 and
the third in 22. Its schedule and acceptance target belong to the public
protocol, rather than the NUTS settings. All rotated-Gaussian fits retained
identity mass because window metric-evidence checks failed; NUTS used dense
adaptation. This is an observed preparation limitation, not an isolated
comparison of fixed and dynamic trajectory length. Do not replace the tuner
with a grid or select an efficiency winner from retained draws.

The preflight's per-member costs omit export/build time and are marked as lower
bounds. Its aggregate includes that work. Subsequent runs explicitly include
grouped export/reload. All-member aggregate costs and single-member costs must
remain distinct. Public HMC sources are archived exactly; all 67 preflight
source reconstruction hashes matched. No shared HMC implementation was changed
for this study.

The final result note and campaign manifest preserve run counts, total budget
and fixed-HMC outcomes. Preserve all failed/invalid attempts and their charges.
Future optimization needs a new bounded experiment, frozen-kernel
profiles, native semantic regressions and fresh posterior assessment. The
current candidate failures do not reject either HMC or NUTS as a research
direction.

- [Reviewed execution plan](bayesfilter-full-stan-nuts-port-execution-2026-10-27.md)
- [Results and interpretation](bayesfilter-full-stan-nuts-port-result-2026-10-27.md)
- [Reusable sampler and tuning guide](../reference/nuts-direct-stan.md)
- [Source coverage and deliberate differences](../reference/nuts-stan-coverage.md)
- Evidence root: `docs/benchmarks/artifacts/stan-full-port-20261027/`

Review was by the implementing agent; no independent agent review was used.
The native oracles and native model remain independent numerical comparators.
