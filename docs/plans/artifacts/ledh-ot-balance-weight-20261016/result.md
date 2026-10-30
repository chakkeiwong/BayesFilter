# Transport mass and conditional proposal diagnosis — 16 October 2026

**Scope correction after owner review:** these runs used the single-flow
canonical/backward-guide consumer. They did not exercise the existing
identity/global/local density-mixture filter, whose local map already uses
conditional Q and whose beta is selected with an independent-pilot variance
criterion. Both implementations are present in this branch and local main.
The measured TV threshold failure does not establish material score harm;
the covariance calculation does not show the mixture is wrong. See
[the checked route correction](mixture-route-correction.md) for source paths
and the revised next action. The measurements below remain unchanged.

**Configuration: UNTUNED FP64 CPU/XLA REFERENCE.** GPU devices were
intentionally hidden with `CUDA_VISIBLE_DEVICES=-1`. This is not TF32
production evidence, scope-specific tuning, or a model-score ranking.
The default remains epsilon1/Sinkhorn5/balance32 and denominator1e-7.
The actual scaling count is5+32=37, not five followed by an exact projection.

## Answer and decisions

Two distinct mechanisms were localized. The additive scaling denominator
materially contributes to the rejected cloud's transport mass error. The
first weighted cloud is already concentrated before transport: its ancestor-
conditional proposal variance differs from the predictive UKF covariance
used by the flow. Removing the mass guard would fix neither problem.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject iteration-only mass repair | Counts32/64/128/256 all fail column-TV<=1e-4 on the same cloud | Mass validity fails | Behavior outside these frozen inputs | Preserve failure and inspect denominator equations | LEDH or backward information is invalid |
| Reject zero-denominator candidate as interchangeable repair | Mass passes, but13/17 formerly valid clouds exceed the predeclared1e-6 state non-harm limit | Non-harm promotion veto fires; no derivative/consumer validity veto | Scope-specific calibration and FP32 behavior remain untested | Keep optional diagnostic setting and unchanged default; any further calibration needs a new prospective contract | A drop-in fix, a tuned default, or superior score accuracy |
| Continue flow-aware proposal design | Independent Gaussian density identities agree; conditional spread calculation explains a plausible concentration mechanism | No harness/derivative continuation veto | A finite-sample variance observation does not prove universal bias | Derive the desired endpoint proposal and its pre-flow law without replacing the UKF lifecycle | Existing guide is universally harmful or nonlinear transfer is established |

## Transport evidence

The reconstructed persistent dataset3/stream1 case uses saved observations
and newly archived CPU noise, not the unavailable original GPU noise tensors.
At terminal counts32,64,128,256, column-TV is respectively
2.70547e-4,2.67881e-4,2.62703e-4,2.52924e-4. The other17 clouds pass.
Increasing iteration count alone is therefore not an accepted repair.

An independent log-domain reference matches the current1e-7 transport to
about1e-15. At the same37 updates on the rejected cloud, denominators
1e-7,1e-8,1e-9,0 give column-TV2.70547e-4,2.74174e-5,2.74546e-6,
4.44847e-11. The exact fixed-point identities are row mass=a-delta*u
and column mass=w-delta*v. Row normalization does not generally restore
the requested column mass; the manuscript now gives a counterexample.

The optional zero setting uses the same shared reset and analytical quotient
derivatives, with fail-closed finite/positive denominator checks. Default
transport parity is exactly zero on the18 frozen clouds. Candidate transport
agrees with the independent reference within5.56e-16. All18 mass checks pass;
the largest scaled finite-difference error is1.3343e-8. Nevertheless,13/17
previously mass-valid clouds fail the1e-6 state non-harm screen; maximum
scaled state change is4.6685e-6. The screen is not relaxed after observing
the result. This rejects interchangeability, not the unperturbed equations
as a separately calibrated numerical method.

Four additional full-consumer fixtures (independent, two persistent and one
large-innovation stress case) use fresh particle noise. All pass validity;
maximum column-TV is1.3703e-10 and maximum tested directional score
finite-difference error is3.2957e-8. These reuse observation regimes and are
**not untouched new-data holdouts**. One direction per fixture was checked,
not every parameter direction. No score-error ranking was calculated.

## First-step proposal evidence

The independent physical-prior, proposal, observation-density and importance-
weight identities agree to1.5e-14. CPU01 exposed an error in the diagnostic
checker: casting a Python constant through TensorFlow rounded it at float32
before conversion to float64. CPU02 fixes only that checker.

For the scalar model, the proposal conditional on an ancestor has varianceQ,
whereas the flow uses predictive varianceP=alpha^2*P0+Q. The continuous
affine flow produces conditional varianceQ*R/(R+H^2*P), not the physical
current-observation conditional varianceQ*R/(R+H^2*Q). Their ratio is
approximately0.296 in the persistent fixture. The exact marginal variance
is0.1956522; the sampled-ancestor conditional mixture gives0.1919881; the
weighted child clouds give0.0452593 at48 steps and0.0437819 at192 steps.
The independent regime does not have this predictive/conditional covariance
gap. These observations support a proposal-coverage explanation, not a
missing Gaussian density factor or a proof of model-score improvement.

A positive-semidefinite quadratic pre-flow guide contractsQ further. It
cannot by itself restore the current-only conditional spread through the
same affine map. Future conditioning can legitimately reduce variance;
the next design must account for composition with LEDH, not discard future
information or silently substituteQ for the canonical UKF covariance.

## Inference status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Iteration-only mass repair fails; zero-denominator drop-in non-harm fails. Tested zero-denominator derivatives/consumer validity pass. |
| Statistically supported ranking | None; this stage is numerical localization, not a stochastic superiority comparison. |
| Descriptive-only differences | Particle variance, ESS, finite-cloud state changes and their maxima. |
| Default-readiness | Not established; default and mass guard remain unchanged. |
| Next evidence needed | Flow-aware proposal derivation, executable consumer parity, then target-specific tuning and untouched paired comparisons against exact/heuristic references. |

## Implementation, attempts and reproducibility

Checkout: `/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922`,
branch `ledh-refactor-with-policy-fix`, HEAD
`ea1018de1da55ed02a29a8d60f08a6827fce8a42` plus preserved working changes.
Per-launch manifests contain executed commands, environment, inputs, seeds,
source hashes and wall times. Interpreter is
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`; both TensorFlow thread
counts are2. Actual tensor inputs, executed scripts and logs are preserved.

| Attempt | Wall seconds | Outcome/artifact |
| --- | --- | --- |
| cpu01 |20.334050| Preserved diagnostic constant-checker failure after transport capture |
| cpu02 |25.145484| Complete capture,72 balance rows, six first-step comparisons |
| floor-reference01 |3.796919| Independent72-row log-domain comparison |
| reset-tests01 |42.68 pytest reported|29 pass; one pre-existing return-arity test failure |
| repair01 |11.726089|18 frozen and four fresh-noise full-consumer evaluations complete |

Subtotal103.683 seconds plus unmeasured short test startup overhead; final
validation/build consumption is recorded in `checkpoint.md`. The stage cap
is300 CPU/400 GPU wall seconds, with zero GPU seconds used. No campaign
budget expansion, model training, commit, push or branch synchronization.

Call-chain checks exercise both the batched shared primitive and the
single-cloud adapters, including full canonical Gaussian consumers under
XLA. They do not establish configuration/admission wiring for every other
consumer lane. Analytical recursion remains the score implementation;
finite differences and independent log-domain calculations are references.

`reset-tests01.log` contains29 passes and one failure in
`test_contract_e_covariance_uses_same_target_by_source_transport`: that test
unpacks six results although the API already returned seven at HEAD. The
same test/source mismatch was confirmed in Git before this change. It is
not repaired as part of this numerical investigation.

## Post-run skeptical review and document audit

The strongest alternative explanation is finite-sample importance-weight
concentration rather than a wrong density calculation. The conditional
variance derivation supports this mechanism but cannot quantify score bias.
A correctly composed endpoint proposal with adequate particles and unchanged
physical target would discriminate the explanations. Mass validity alone
would not overturn the proposal diagnosis; new GPU/TF32 evidence could
overturn transfer of the numerical candidate.

The LaTeX additions separate additive mass error from conditional proposal
spread in proposition/proof form, retaining the preceding manuscript.
`manuscript-before.tex` is the protected pre-edit reference (SHA256
3ec736727898c77f47b5b9f2da03724b1042f82f9c7b1917df7b74d90e5be6a9).
The audience is the research owner: explain which Gaussian is conditional
on an ancestor before comparing variances. MathDev algebra checks cover
the marginal identity, variance ratio and mean-ODE substitution, not the
entire manuscript or nonlinear theory. Final bounded audit/build outcomes
are recorded in `validation.md`. Human readability acceptance remains pending.
