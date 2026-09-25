# HMC repair program: analysis and roadmap review

Date: 2026-09-23. Planning baseline: `e9fee584704a465fc6ab984e3a8cc53980f335d0`.
This is a read-only code/evidence analysis followed by a documentation repair.
No new sampler, training or confirmation run is part of this reset. Unrelated
Q20, training-protocol and governance edits are outside its source scope.

## Question and inspection scope

Can the remaining work be organized into a finite, dependency-aware program
whose completion means something stronger than completing its next experiment?
The comparator is the master and machine progress record after M30. Inspection
covers the public interface and capability registry; candidate execution and
retained sampling; precision and posterior assessment; the validation catalog,
pipeline, SBC, power and reporting engines; relevant integration tests; and the
M22--M30 result/plan records. This is not an exhaustive line-by-line audit of all
20,751 lines of the historical `hmc_kernel_tuning.py` module or of every consumer.

Evidence is the inspected source and existing audited results, not a new test
pass. A catalog entry is not an executed test. An existing result remains bound
to its own source and model. Source-dependent random streams prevent treating
same-config executions on different source snapshots as exact paired replays.

## Findings

1. **The planning defect is real.** The former master contains 3,000-plus lines
   of successive status additions with several historical "next phase" sections.
   M31 has a design, but six residual workstreams have neither a terminal phase
   nor a common completion rule. Successful local repairs keep moving the next
   action without resolving the total cost or required evidence. Archive that
   history and retain the existing master pathname for one active roadmap.

2. **Public tuning unification is implemented in the inspected paths.**
   `tuning_contract.py::HMC_TUNING_INTERFACE_CAPABILITIES` declares two public
   artifact-authority tuners. The public reference and integration tests agree
   on ordinary/fixed-map preparation, shared candidate search, fresh verification
   and retention. R-hat, ESS and MCSE are downstream posterior quantities.
   Existing coverage does not prove that every consumer has migrated, every
   option combination works, or every scientific target will admit a candidate.

3. **Posterior reliability is the main scientific repair, not a missing MCSE
   implementation.** `hmc_precision.py` already implements autocorrelation,
   batch means, lugsail and quantile precision. `hmc_posterior_assessment.py`
   supplies ESS floors, consecutive readiness checks and monitored quantities;
   `neutra_hmc.py` applies these to archived warmup and retained chunks. Finite
   MCSE measures sampling uncertainty under its assumptions, not initialization
   bias. Its presence does not establish sufficient burn-in or coverage after
   repeated stopping. M25 found inadequate Gaussian/beta-binomial stopped
   coverage on the actual public pipeline. M26's larger-count successes are
   development observations. These are open validation/repair requirements.

4. **Statistical design is a cost driver.** `engines/sbc.py` takes one terminal
   output from each independent complete fit, with several fits per dataset.
   `engines/power.py::calibrate_experiment` repeats whole experiments. This
   protects independence but multiplies complete preparation/tuning costs.
   A cheaper statistic must be justified for its exact question, not substituted
   silently. Its null under an ideal posterior does not establish the output law
   of finite adaptive HMC. Failures and missing ranks must remain visible.

5. **Coverage and defect power are distinct objectives.**
   `engines/pipeline.py::stopped_intervals` measures intervals for posterior
   functionals at the actual stop, and its summary retains missing fits in the
   denominator. SBC checks a conditional distribution across datasets. Frozen
   invariance tests check a different property again. M22's null-size screen is
   closed only for its frozen Gaussian experiment. A detected shift in one
   full-fit experiment is not an estimate of repeated-experiment power.

6. **Global exploration and learned geometry cannot be assigned to epsilon/L
   search.** The mixture has a known missed-mode failure; the monitored mode
   indicator now exposes some such failures but cannot prove all modes found
   on an arbitrary target. M28 closes supplied exact/residual-funnel cells.
   Reopening centered-funnel success as a tuner requirement would be a wrong
   baseline. Actual map training needs a target-specific protocol and untouched
   downstream validation. M23 fixed graph/batch prerequisites, not map quality.

7. **Consumer integration has two separate dependencies.** M24's bootstrap and
   diagnostic repairs are tested. Its exact nine-parameter/48-observation fresh
   consumer bundle and the earlier full-joint MIDAS reference are different
   outstanding inputs. The existing result documents their absence; this reset
   has not independently qualified a new matching bundle. Block-factorized
   consumer runs or posteriordb models cannot substitute for either target.
   M36 must make a fresh bounded inventory before declaring the input missing.

8. **Maintenance needs a bounded scope.** The active modules are substantially
   smaller than the historical facade, but preparation still imports its
   `_json_ready` and `_HMCPhaseAttemptState`. These are concrete dependency
   boundaries to audit, not evidence that a whole-facade rewrite will improve
   correctness or cost. M30 provides exact optional-reuse parity on two models;
   it does not prove bounded memory within a fit or performance on other models.

9. **Progress metadata also needs repair.** M31's opening says M30's terminal
   ledger is pending although it exists. Top-level source metadata still names
   older development baselines while M30 has its own correct source record.
   Preserve historical phase records and explicitly mark that no M31 numerical
   source or worker is active. Update the roadmap and machine state together.

## Pre-edit skeptical audit and decision

The proposed repair replaces the active master with M31--M38, a requirement
matrix, dependencies, budget envelopes and explicit claim boundaries. M38 is
terminal integration and reporting; it cannot close unmet scientific criteria.
M31 resolves statistical designs once, M32 establishes the common engineering
baseline, and the other phases execute separately defined work packages. A
local failure loops within its phase and budget; it does not automatically
create another numbered phase.

| Risk checked | Required correction |
| --- | --- |
| Wrong baseline | Preserve closed source-bound cells; distinguish supplied maps, trained maps, prepared HMC and complete automatic fits. |
| Proxy promotion | Test counts, pilot passes, oracle power, local R-hat and runtime never substitute for posterior coverage or full-fit power. |
| Missing stop conditions | Preserve phase and campaign caps, invalidity vetoes and per-cell missing-input states; candidate failure triggers the named repair. |
| Unfair comparisons | Fix member/quantity rules and independent streams before results; include reference uncertainty and all unsuccessful fits. |
| Hidden assumptions | Audit moment conditions, repeated looks, recent-window information, test independence, source effects and target-specific training choices. |
| Stale context | Use the M30 terminal ledger and committed planning baseline; preserve unrelated edits and consumed consumer runs. |
| Environment mismatch | GPU/XLA/growth for serious execution; explicit CPU reference exceptions, no unapproved pfor or algorithmic NumPy. |
| Uninformative commands | Price complete fits; require each future run to name the requirement and evidence it can actually resolve. |

Audit verdict before documentation changes: proceed. This accepts the roadmap
repair, not a new statistic, sampler policy or assurance that the remaining
allowance funds every claim. Confirmation and target-specific training must
first meet their numerical design requirements within the appropriate phase.

For the planning arithmetic only, recompute binomial acceptance probabilities
from elementary binomial sums and multiply M30's recorded outer-fit costs by
predeclared inventory sizes 64, 128, 256 and 384. These sizes are alternatives
for explaining the existing cost/precision tradeoff, not selected new sample
counts. The .90 floor, two-sided .95 interval and hypothetical .95 true success
rate come from M26. Outputs will be saved in the roadmap review directory.
They cannot estimate actual future coverage, guarantee run cost, replace a
fresh study or promote a reduced design. No GPU/framework execution is needed.

## Review record and verification

The active master was replaced by the M31--M38 roadmap, and the former file was
preserved as `bayesfilter-hmc-repair-master-history-through-m30-2026-09-23.md`.
The M31 detail was corrected to say that M30's terminal ledger is audited. The
machine progress record now points at the active roadmap, records M31 as
reviewed/not started, lists M32--M38 as planned/not started, and keeps the
M30 remaining allowance.

The offline check is
`artifacts/hmc-repair-master-2026-09-16/roadmap-review-2026-09-23/roadmap-review.json`.
It passed the history snapshot, eight phase headings and table rows, stale
status check, progress-pointer/status checks, budget sums, remaining-budget
check and all ten active-plan links. The roadmap envelopes total 61,200 CPU
seconds and 63,000 GPU seconds, leaving the documented common reserve within
the M30 remainder. No HMC, training, GPU framework or numerical worker was
launched for this planning repair.

This verification establishes a consistent roadmap and metadata. It does not
establish posterior calibration, complete-fit coverage, defect power, global
mode exploration, exact MacroFinance integration, learned-map quality or
adequacy of the proposed phase envelopes. Those remain the explicit work of
M31--M37 and are reported by M38.
