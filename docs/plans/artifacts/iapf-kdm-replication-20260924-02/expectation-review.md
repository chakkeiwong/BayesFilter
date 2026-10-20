# K1b expectation result and protocol disposition

The actual source matches its derived one-step expectation, but that expectation
differs from the ideal derivative after finite-particle weight normalization.
This is a limitation of a blanket unbiasedness claim, not a code-parity failure
or a rejection of MDPF. Both requested replications remain incomplete.

Attempt07 ran the added CPU independent-reference diagnostic under the unchanged
continuation budget. It passed all four overlap/separation by N=2/25 situations:
32,768 independent clouds, eight sampling seeds, 1,024 clouds per seed/situation.
Twelve focused dispatcher tests passed before launch. No optimizer updated.

## Target, computation and derivation

The paper's Equations (14)–(15) hold sampled locations and the proposal density
fixed when differentiating an importance weight. For a linear test statistic
f(z)=z_x and matched sampling density q=m at the evaluation parameter, the
unnormalized estimator N^-1 sum_i f_i s_i has expectation d E_m[f]/d phi,
where s_i=d log m(z_i)/d phi. This is the checked ideal target.

The source instead samples from raw mixture weights, evaluates density scores
with additive mixture-weight epsilon, injects a weight whose base value is
c=1+1e-8, and normalizes those weights in the next measurement step. Let s_e be
that epsilon-modified score. With constant measurement weights, differentiating
the actual normalized weighted average gives

    G = (1 / (N c)) sum_i (f_i - f_bar) s_e,i.

For independent draws from q,

    E[f_bar s_bar] = E[f s_e]/N + (N-1) E[f] E[s_e]/N,
    E[G] = (N-1)/(N c) { E_q[f s_e] - E_q[f] E_q[s_e] }.

In the exact matched-density limit E_q[s]=0, this is (N-1)/(N c) times the
ideal derivative. It is different from the claimed unnormalized unbiased
quantity at finite N. We are not deriving the expectation of the nonlinear
training NLL or its truncated sequence gradient here.

The fixture uses N source components, with repeated identical left/right
Gaussian components representing two groups. Thus the source draws exactly N
samples and applies epsilon N times. The identical second-coordinate and
von-Mises factors cancel in the mixture-weight score. Eight actual source
sampling seeds are replayed; the comparison score and standard-normal
Gaussian quadrature are independently expressed. Their 64/128-node agreement,
exact Gaussian moments and exact mixture-mean derivative controls all pass.

## Evidence

See [contract](stage0/attempt07/contract.json),
[full comparisons](stage0/attempt07/comparisons.json),
[result](stage0/attempt07/result.json), and
[source snapshot](source-repair05/docs/benchmarks/diagnose_replication_expectation.py).

The maximum sampled algebra error is 1.776e-15. The normalized constant
statistic's maximum gradient is 3.556e-16. All source sample means agree with the
independent source-expectation prediction within the predeclared six-MCSE
bounds. Approximate 95% intervals use independent-cloud variance, not just
eight seed means. These are one-statistic diagnostic intervals, not adjusted
simultaneous claims about training or whole-method performance.

| Situation | N | Ideal derivative | Source predicted normalized derivative | Measured normalized derivative (approx. 95% interval) |
| --- | --- | --- | --- | --- |
| overlap | 2 | 0.182000 | 0.091000 | 0.089462 (0.086867, 0.092056) |
| overlap | 25 | 0.182000 | 0.174720 | 0.174871 (0.173882, 0.175860) |
| separated | 2 | 1.365000 | 0.682500 | 0.679681 (0.663401, 0.695960) |
| separated | 25 | 1.365000 | 1.310400 | 1.314440 (1.310495, 1.318386) |

For the quadrature-only rare-weight case, the right group mass is 1e-10 and
N=25. The ideal mean derivative is 6e-10; the source unnormalized expectation
is 3.0023e-10 and its normalized expectation is 4.4274e-13. The latter is about
0.000738 of the ideal derivative. This is a deliberately extreme epsilon
sensitivity fixture; rare-event Monte Carlo was not run, and its prevalence or
effect in trained models has not been measured. Source smoothing cannot be
assumed negligible merely because its numerical epsilon is small.

## Source and protocol disposition

The author source remains unedited at
b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18. Anchors are:

- Paper: local Younis–Sudderth text lines 448–474, Equations (14)–(15).
- Source KDE: kernel_density_estimator.py:146–183 samples raw categorical
  weights; :230–268 smooths and normalizes density weights.
- Source KDEParticleFilter: :675–810 resamples/injects gradients and
  :928–934 normalizes subsequent weights. These actual endpoints execute in
  the diagnostic; sampled algebra and expectation are executable checks.
- Paper Appendix B: local text :929–984 states linear dynamics with bimodal
  observations, distinct bandwidths, final-time NLL and mixture-Kalman reference.
- The located src/datasets/toy_problem_dataset.py:199–244 instead uses the
  nonlinear 0.5*x + 25*x/(1+x^2) + 8*cos(1.2*t) transition and x^2/20
  observation. It is wrong relative to the Appendix B target. A bounded search
  for mixture-Kalman and C1/C2 identifiers found no Python/YAML consumer;
  this is not proof that no separate author source exists.
- Source evaluation_base.py:22–39 ignores the configured random_seed when
  initializing RNGs and uses default seed=0. For the source-fidelity arm,
  freeze and record effective evaluation seed zero rather than silently
  changing the source. A corrected seed setter or extra independent evaluation
  streams would be explicitly separate reconstruction/robustness arms.

Retain source epsilons and normalization for source replication. Do not call
the complete normalized NLL estimator unbiased. Keep the ideal paper estimator
as a separately labeled mathematical comparator. The Appendix B source consumer,
its initialization/learning-rate details, and whole-loss expectation remain
unresolved; bearings fixtures cannot close them.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept this diagnostic | Actual source agrees with independent algebra and expected statistic | No nonfinite, quadrature or call-chain veto | Whole NLL and sequence expectation untested | Recover/specify Appendix B consumer and missing protocol choices | Full IWSG unbiasedness, training quality or paper replication |
| Reject blanket source-unbiasedness claim | Normalized expectation differs from ideal; ordinary-case intervals exclude it | Mathematical target mismatch | Practical effect on trained filters unknown | Preserve source behavior and report its exact target | MDPF research direction rejected |
| Retain rare-weight warning | Independent quadrature exposes non-negligible relative smoothing effect | No quadrature consistency veto | Very small absolute derivative; unmeasured prevalence | Monitor source weight/bandwidth margins in future runs | Typical gradients are suppressed this much |

| Inference item | Status |
| --- | --- |
| Hard veto screen | Mechanics/algebra pass; target identity differs as derived |
| Statistically supported ranking | None; no learned methods compared |
| Descriptive-only differences | Short-run costs and prior undertrained losses remain descriptive |
| Default readiness | Not established; independent R/PyTorch references only |
| Next evidence | Full target-specific training, comparators, heldout uncertainty and actual TF/GPU/XLA consumers |

Post-run red-team: the strongest overinterpretation would extrapolate this
linear-statistic finite-N calculation to every gradient in MDPF. That is
unsupported. A different downstream objective may have a different expectation;
changing the normalizer or applying a correction defines a new estimator and
would require a separate decision. Rare-weight quadrature is not a measurement
of trained-network pathology. Exact source parity remains supported.

Budget after attempt07: seven of twelve launches, 25.488099 CPU seconds and
118.317269 GPU seconds used; five launches, 1774.511901 CPU seconds and
781.682731 GPU seconds remain. Full paper experiments remain NOT_BUDGETED.
No numerical child remains running. The manifest records the exact command,
source/environment/configuration identities and seeds; the ledger records time.
