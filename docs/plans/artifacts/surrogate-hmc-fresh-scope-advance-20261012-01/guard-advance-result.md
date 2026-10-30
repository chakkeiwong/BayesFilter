# Fresh-scope continuation after covariance guard repair

Range bearing completed fresh offset40 calibration/validation and independent
FP64 arithmetic checks. Selected candidate0 uses flow8/reset40/epsilon0.5 and
beta(0.1,0.7338126073431386,0.1661873926568614). Fresh artifacts are under
`../surrogate-hmc-guard-advance-range-20261013-01`.
The exact initial four-chain force is healthy. Both cold and warmed two-step
HMC runs are finite, with zero fallback/divergence. Observed warmed acceptance
is1 for each chain; maximum absolute energy error0.0028364843. These two draws
per chain do not support convergence or acceptance-rate precision.

The warmed L1 transition costs28.0336 seconds for four chains. A straight-line
2000-transition warmup at this rate costs15.57 hours before retained sampling;
tuned trajectory length and adaptation are not measured. The endpoint is the
fixed full-support4x4032 trajectory-importance approximation under the benchmark
normal prior, not an exact model posterior. Precision agreement does not test
that approximation against a model oracle.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Range mechanics passes | Full initial force and actual cold/warm sampler health | None observed | Short fixed-step pilot and approximate endpoint | Continue SIR within budget | Posterior convergence or practical full-HMC cost |
| SIR continuation authorized |3800+1200 seconds fit5247.979 remaining | Budget monitored by master | Full-horizon SIR force/compilation cost | Fresh tuning then automatic mechanics | Success before execution |

| Inference status | Finding |
|---|---|
| Hard veto screen | No range mechanics veto observed |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime, two-step acceptance and energy error |
| Default-readiness | FP32 remains default; mixed recursion is optional |
| Next evidence needed | SIR phase, target/prior review and affordable tuned cumulative inference |

Pre-launch SIR audit: new source-bound tuning is required; historical selection
cannot authorize this phase. The same declared candidate family and new seeds
avoid tuning on the previous claim. Reserve1200 seconds for mechanics; total
phase cap5000 seconds fits the remaining ten-hour authorization. Full posterior
HMC is still incomplete. All source-bound manifests remain in each run directory.

## SIR completed result

Fresh offset40 SIR tuning selected candidate0: flow16/reset40/epsilon0.5,
beta(0.1,0.07527773579726693,0.8247222642027332). All two-dataset/two-row
precision and fullT50 validity checks pass. Maximum scaled likelihood error
1.88142e-8 and score error0.00183521 (0.183521%). Score scale is
max(1,abs(FP64 component)); this is not always a relative percent.
Candidate1 fails with a maximum scaled score error6.9830; candidate2 fails at
0.0377638. The retained precision veto matters despite small likelihood gaps.
No arithmetic candidate ranking is statistically supported.

Both cold and warm SIR mechanics are finite with no divergence or force fallback;
all four initial forces are valid. Warm acceptance over two transitions is
[0,1,0.5,0.5], overall0.5, and maximum absolute energy error3.46104. The first
chain did not accept either of its two proposals, so movement/convergence is not
established. This short health criterion is insufficient for full-chain admission.
Warm four-chain transition63.3403s gives an illustrative2000-transition cost35.19h
at L1, before retained draws or tuning. The fixed4x4032 bootstrap endpoint is an
approximate posterior and can be nonsmooth as resampling indices change. Small
force arithmetic error does not establish accurate model scores or good HMC.
Artifacts: `../surrogate-hmc-guard-advance-sir-20261013-01`.

The selected range candidate maximum scaled score error is0.000235435
(0.0235435%), and maximum scaled likelihood error1.14064e-6. Both new scopes use
optional FP64 recursion/FP32 pairwise kernels; uniform FP32 remains the default.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain SIR candidate0 for exploratory mechanics |1% precision and short numerical health pass | Candidates1/2 rejected on precision; no observed sampler numerical veto for0 | One nonmoving two-step chain; narrow parameter coverage; approximate endpoint | Complete current-source LGSSM phase, then report outstanding inference work | SIR convergence, exact posterior or economical HMC |

| Inference status | Finding |
|---|---|
| Hard veto screen | SIR candidates1/2 fail arithmetic precision; candidate0 short mechanics passes |
| Statistically supported ranking | None |
| Descriptive-only differences | Two-step acceptance/energy and single warmed runtime |
| Default-readiness | No new numerical default or HMC inference admission |
| Next evidence needed | Proper target/prior scope, reference adequacy, tuned adaptive warmup and cumulative retained diagnostics under an adequate budget |

SIR tuning1934.681s and mechanics384.460s were charged; combined2319.141s.
Cumulative33071.163 of36000 seconds used before final LGSSM continuation.
The next authorized master phase is in guard-lgssm-plan.md (2800s cap).

Post-run red team: score precision can pass locally while the force disagrees
with the MH target or performs poorly elsewhere. SIR acceptance and cost are
consistent with this unresolved risk; they cannot identify its cause alone.
The weakest evidence is posterior-region coverage and endpoint approximation,
not the now-resolved exact range replay. A failed wider precision check would
restrict the viable scope further; a longer converged chain is still needed.
