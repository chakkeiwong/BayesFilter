# Prospective precision audit for surrogate-HMC tests

Question: under an ideal IID posterior sampler, is the existing bounded
two-look schedule long enough to make the median-CDF equivalence margin
plausible? This is a deterministic design calculation, not another look at
any retained chain. It neither revises closed results nor changes active SV20
sampling. Its role is explanatory and prospective budget planning.

For a median indicator with probability one half, its variance is one quarter.
With four independent chains and n draws per chain, the ideal standard error
is sqrt(.25/(4n)). The current interval uses the Student-t critical value with
36 degrees of freedom and upper tail .05/(2*m*2*f), where m is the number of
functionals, two is the number of looks, and f is the number of comparison
families. Compute c*sqrt(.25/(4n)) for the existing six-function/one-family
and thirteen-function/four-family contracts, at n=1000,2000,4000,8000.
The median-CDF equivalence half-width is .02. Set reference error to zero for
this optimistic calculation. Also compute the draw count at which this ideal
radius reaches .02 and .01, the latter leaving room for estimation error.

The expected radius is not a deterministic lower bound on the realized
estimated interval: batch variance estimates fluctuate, antithetic sampling
can reduce variance, and dependence can increase it. Crossing this design
threshold cannot establish power, mixing, coverage or equivalence. It does
expose an inadequately justified claim that a short pilot is adequately
powered. Full power would require the joint probability of all equivalence
criteria passing, with dependence and reference uncertainty included.

Default audit: the critical value, dimensions, two looks, CDF margin and
36 degrees of freedom come from the frozen runtime and model plans. IID
sampling and zero reference error are explicit optimistic reference choices.
Use TensorFlow Probability in float64 for the quantile. This is a small
CPU-only, eager independent-reference calculation; hide GPU devices before
import. It is not a production kernel or GPU performance measurement.

Skeptical audit: the calculation checks the actual multiple-comparison
allocation and median functional; it does not substitute a single-parameter
normal interval, tune a threshold on claim data, or interpret diagnostic
acceptance as posterior accuracy. A finding of poor planned precision calls
for a prospectively funded design, not an extension of the active looks.

Reserve at most15 CPU seconds from P0 scope closure and no GPU work. Preserve
the exact script, command, environment, Git commit, elapsed time, numerical
table and accounting under
`artifacts/surrogate-hmc-all-models-20260922-precision-audit-01`. Stop on an
invalid/nonfinite quantile or a mismatch with the runtime critical value.
