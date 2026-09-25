# M31 design decision and downstream execution inventory

The design audit is complete. Fund the engineering and bounded development
work below; retain stopped-coverage and repeated whole-fit power as underfunded
at measured costs. No confirmation denominator is reduced to manufacture a
passing result. Existing phase ceilings and common reserve are unchanged.

## Mathematical and source audit

Sources are the checked local copies under
`.localresources/papers/hmc_validation_standards_20260915/`.

* Talts et al.: `talts-sbc.txt`, section 4.1, equations (2)--(3), section 5.1
  Algorithm 2, Appendix B (lines 1007--1099). The rank proof factors the
  conditional posterior draws; ordinary correlated draws cannot be inserted
  into its iid rank null. Thinning is approximate and quantity-dependent.
* Modrak et al.: `sbc-test-quantities-2211.02383.txt`, sections 3.1--3.3,
  Theorems 3--6, section 4.6, Appendix A definitions (1350--1415), Theorem 3
  proof (1668 onward) and Theorem 4 proof (1802--1829). Passing a chosen rank
  test is necessary, not sufficient; data-sensitive quantities matter. Its
  ideal conditional law does not certify finite, stopped HMC output.
* Gandy--Scott: `gandy-scott-mcmc-unit-2001.06465.txt`, section 2.2 Algorithm 2
  and Proposition 2.3 (297--326), Algorithm 3/Theorem 3.1 (343--416), appendix
  proofs (1036--1115). Reversibility plus the randomized stationary anchor
  gives the rank construction; independent superuniform look vectors give
  the sequential wrapper bound. Adaptive preparation is not that kernel.
* Official code: `mcunit-expect-invariant.R` lines 129--172 constructs random
  anchors, two directions and randomized ties, then calls `expect_mc_test`.
  The local BayesFilter sequential wrapper retains those required roles.
  `sbc-diagnostics.R` is diagnostic reporting, not the rank-test implementation;
  it cannot justify an independence shortcut. No claim is made to have audited
  an unavailable SBC rank implementation merely by reading this file.

Bibliographic identifiers in the inspected local copies are Talts arXiv 1804.06788,
Gandy--Scott 2001.06465, Modrak 2211.02383. The old misnamed
`sbc-1805.09294` file remains excluded as directed by SOURCE-NOTE.md.
The local paper/code hashes are in `design-arithmetic.json`.

The baseline costs 128 datasets times three fresh fits per dataset, 384 fits
per complete SBC experiment, before independent experiments for power. One
output per independent fit protects the conditional independence construction;
neither posterior sibling pooling nor replay calls multiply its denominator.

Alternative A is accepted for **diagnostic development on the exact normal
target only**. Complete the square to obtain
`s^2=(tau^-2+n/sigma^2)^-1`, `mu=s^2*sum(y)/sigma^2`. For one predetermined
output X from each independent full fit, define Z=(X-mu)/s. Under the ideal
correct-output null, independent Z are N(0,1). The existing translated target
has Z~N(d,1) only if its output actually follows that target. A finite adaptive
stop may violate this premise; the test is intended to expose output errors.
Use two predeclared two-sided tests: sqrt(N)*mean(Z) against N(0,1), and
sum(Z^2) against chi-square(N), each at .05/2. The second detects scale/ignored
data discrepancies missed by a mean-only statistic. Bonferroni does not require
independence between those two statistics. Across fits, independence and a
fixed reporting rule are necessary. Missing/capped/unqualified outputs make
the experiment incomplete; no conditional result supplies unconditional power.

Alternative B, replacing independent fits with several thinned outputs from
one adaptive fit, is **not admitted for confirmatory use**. The inspected
sources supply no exact calibration of this particular adaptive stopping and
thinning law. That is an unresolved null, not a demonstrated implementation
defect. The current independent-fit SBC remains available.

## Operating characteristics and funding

`design-arithmetic.json` reproduces the exact binomial boundaries using an
independent log-binomial sum. It uses M30 outer receipts and normal exit codes.
384 fits cost 36.97/30.45 observed GPU hours for Gaussian/beta respectively.
Their sum exceeds the entire remaining GPU allowance. These are descriptive
prices and not guarantees for another target or a repaired policy.

For Alternative A, under the ideal translated-normal comparator, 199 fits for
d=.25 or 50 for d=.5 give at least .90 power for the mean component alone.
To have at least .95 planning probability of a two-sided .95 CP lower endpoint
above .80 when actual power is .90 requires 167 independent experiments
(at least 144 detections). This means 33,233 or 8,350 fits per defect arm.
The .10-upper null screen at hypothetical .05 rejection similarly needs 363
experiments for .95 planning assurance. These are local planning derivations,
not achieved HMC power or universal minimum sample sizes.

Using the Gaussian price only as an illustrative transfer yields about 3,199
and 804 GPU hours per defect arm. A target-specific price is still required;
the required improvement would be orders of magnitude, not justified by M30.
Four bounded normal-target development fits can test activation/serialization
and reporting, never power. R2/R3 statistical criteria remain open.

## Resolved work package inventory

| Phase | Funded action now | Scientific disposition / prerequisite |
| --- | --- | --- |
| M32 | Run current-source candidate, authority, restart, diagnostic and multi-model suites. Add real static/dynamic parity for rotated Gaussian, constrained Dirichlet and supplied residual funnel. Six GPU fits (one static/dynamic pair per model), at most 800 seconds each plus tests inside 5400 GPU seconds. | Exact numerical/lifecycle parity only. No speed or posterior-reliability claim. |
| M33 | Repair the concrete missing validation wiring for batch size/lugsail controls and readiness ESS/consecutive-check controls. Test fixed/stopped estimator agreement, unavailable batches, cap behavior, policy identity and checkpoint mismatch. | Replicated full-fit stopping confirmation is underfunded; counts and defaults unchanged. Reuse M25/M26 closed development records rather than duplicate pilots. |
| M34 | Add the accepted exact-normal endpoint diagnostic with independent formula tests, missing-fit denominator and immutable output rule; execute baseline, no-op and .25/.5 translated target public fits once each, at most 700 seconds per fit. | Four fits are activation/composition evidence only; repeat-experiment power remains underfunded. No special acceptance bands for these ordinary development fits. |
| M35 | Audit global-indicator wiring and constant/missed-mode counterexamples; run the existing real multi-start public fixture and independent analytic-reference tests. | Base scientific global-exploration cell remains failed unless declared numerical/reference checks pass; learned extension is M37. No new sampler family. |
| M36 | Recheck named consumer paths, new dated handoffs and exact target/reference identity; run matched consumer only if the actual bundle qualifies. | Missing exact inputs leave the affected cell awaiting inputs; no substitute posterior claim. |
| M37 | Inspect the actual selected training call chain, prepare separate banana/mixture protocols and run focused graph/batch/freeze-reload tests. | Serious map quality is not funded by an arbitrary 1.5-hour inherited recipe. Final protocol must price its target-specific search and untouched downstream assessment before training; record unknown price as unresolved rather than inventing one. |
| M38 | Final affected integration/guide build, current requirement table, all receipts and selective Git publication. | Reports open scientific requirements; does not certify them by closing the program. |

The M32 GPU fixtures inherit the small real-pipeline mechanics allocations and
broader mechanics-only acceptance screen from existing multi-model tests.
They compare execution routes and cannot qualify a production numerical policy.
L=(2,3), 64 measurement/verification draws, 128 posterior cap and fixed maps
are engineering probes, not calibrated scientific defaults. All verified
members are compared; missing verification fails the fixture. Seeds are fixed
identifiers before results; one pair per model means no independent replication
claim. Stage any extension only within its explicit phase ceiling.

M34 retains native ordinary L=(3,5,9,13,18,25), preparation options from the M27
normal-scale control, 128 measurement/verification draws, first-verified member
by identity, and normal-conjugate tau=2/sigma=1/n=6. One simulated dataset per
arm and all fit streams are fixed by the resolved design before launch. For
the four activation controls, the frozen inventory uses the same simulated
dataset and streams to make the no-op an exact parity check. These paired
controls are not four independent statistical replications. This refinement
changes no power criterion; no power inference is made from them. A
10000 warmup/retained cap and .05 precision are inherited development choices,
not adequacy assumptions. A cap remains unavailable for endpoint confirmation.

## Review and decision

The design does not close R2/R3. Its engineering work is useful without that
claim and remains within the master. The newly identified validation-policy
wiring gap is a real prerequisite to the requested M33 experiments; an option
that silently never reaches the posterior runner cannot evaluate that policy.
Do not change runtime defaults to compensate. Preserve exact source and seed
identity; no later source can relabel a prior result.

| Decision | Criterion | Veto | Uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept endpoint diagnostic development | Checked exact conditional law and multiplicity | Missing/invalid fits prevent complete test | Finite stopped-output law | Implement and test on full fits | Achieved power or coverage |
| Defer R2/R3 confirmation | Exact design and current costs exceed allowance | No reduced denominator promotion | Future target-specific costs | Execute bounded engineering; retain funding gap | Research direction rejected |
| Proceed M32--M38 | Named invariants and phase caps | Invalid evidence or exhausted allocation stops affected run | Untested model/backend interactions | Concrete repairs and terminal audit | All gaps will close |

Hard veto evidence is limited to invalid comparisons/artifacts. No stochastic
ranking is supported. All observed timings and transferred costs are descriptive.
No default is ready for promotion from this design. Fresh adequate full-fit
inventories remain the additional evidence needed for R2/R3.
