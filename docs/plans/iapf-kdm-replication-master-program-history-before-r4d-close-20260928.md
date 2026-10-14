# iAPF and MDPF/KDM-IWSG replication master program

R4d execution amendment: the next bounded source-semantic repair is now implemented
and queued under [the pre-run contract](artifacts/iapf-mdpf-whole-repair-20260928-01/r4d-finite-semantics-spec.md).
It preserves all prior failed results and unchanged time budgets; source controls,
small consumer regression and the existing paper-horizon gate are separate phases.


Updated: 28 September 2026, Hong Kong. **Both scientific replications remain
INCOMPLETE. The R4c diagnostic tranche is complete; the bounded R4d solver
repair and regression tranche is executing. Paper-horizon finite-parameter
parity remains FAIL; a complete actual
GPU/XLA filtering run is now available as a diagnostic.**

This is the authoritative entry point. The [whole repair plan](iapf-mdpf-whole-repair-plan-20260928.md)
defines the remaining phases; [run_replication_master.py](../../scripts/run_replication_master.py)
is the executable dispatcher. Its process, numerical and scientific statuses
are separate. The current [R4c result](artifacts/iapf-mdpf-whole-repair-20260928-01/r4c-result.md),
[review](artifacts/iapf-mdpf-whole-repair-20260928-01/r4c-review.md) and
[checkpoint](iapf-kdm-replication-checkpoint.md) supersede stale next-step prose
in the [previous master](iapf-kdm-replication-master-program-history-before-r4c-close-20260928.md).
The [earlier result](artifacts/iapf-mdpf-whole-repair-20260928-01/result.md)
remains evidence for attempts 01–16.

## What we are replicating

**iAPF:** Guarniero, Johansen and Lee, *The Iterated Auxiliary Particle Filter*,
JASA 112(520), 1636–1647, DOI 10.1080/01621459.2016.1222291.
Replicate the twisted filter, finite backward fitting, positive floor,
ESS-based resampling, adaptive iteration/particle controller and fresh final
filter. Initial empirical scope is Section 5.2: d=5,10,20,40,80; T=100;
1,000 complete algorithm repetitions on each fixed generated dataset, with
Kalman likelihoods and the declared BPF/fully adapted comparators. Include
adaptation cost. The [publisher paper](../../.localresources/papers/guarniero-johansen-lee-2017-iapf-publisher-version.pdf)
and [thesis](../../.localresources/papers/guarniero-2017-iapf-thesis.pdf)
support a finite-BFGS reconstruction family, but the original executable
initializer, coordinates, optimizer controls, floor, regularizer and seeds
remain unrecovered. A_raw/B_scaled are explicit reconstruction hypotheses.

**MDPF/KDM-IWSG:** Younis and Sudderth, *Differentiable and Stable Long-Range
Tracking of Multiple Posterior Modes*, NeurIPS 2023.
Replicate kernel-density mixture filtering and importance-weighted sample
gradient training, including the actual Appendix B parameter-learning
experiment and the declared bearings-tracking objectives, refinement,
comparators and independent repetitions. KDM is the mixture representation,
not a third replication. [Author code](../../.localresources/code/younis-mdpf-neurips-2023)
is pinned at b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.
Its nonlinear Cappe toy dataset is wrong relative to an Appendix B replication
claim; Appendix B uses scalar linear-Gaussian dynamics with bimodal observations.

Filtering replication comes first. Parameter scores, HMC, learned extensions
and LEDH integration are later work, not prerequisites for these two
replications. A GPU port and a successful single filter run do not establish
empirical replication.

## Checked progress

| Phase | Status | Evidence and remaining limit |
| --- | --- | --- |
| R0: sources and finite recipes | Reconstruction scope specified | Exact original settings still unrecovered |
| R1: executable dispatcher | Implemented and regression checked | Commands, snapshots, manifests, dependencies, timeouts, recovery, locks and cumulative budgets; future unimplemented phases remain explicit |
| R2: independent iAPF reference | Controls/pilot complete | 20 controls pass; A_raw 8/8 and B_scaled 5/8 complete, three amplitude failures; no stochastic ranking |
| R3a: saved MDPF trial screen | Complete for one checkpoint | 5,000 trajectories; source metric parity and three conditional point baselines pass; normalized-density comparison remains open |
| R4a: frozen-twist iAPF value | PASS in tested scope | 10 cases agree across CPU/GPU graph/GPU XLA; maximum discrepancy about 6.8e-12 |
| R4b: legacy graph failure | Localized, not uniquely root-caused | Fresh-process TensorFlow optimizer interaction; new kernel works with default optimization and XLA |
| R4c: finite fitter/controller | Implemented; small parity PASS | 129 checks per CPU/GPU graph/GPU XLA lane, one trace per signature; source snapshots identify tested versions |
| R4c: paper-horizon parameter parity | FAIL | Exact-roundtrip fixture still differs in backward call 2; failures are retained |
| R4c: actual paper-horizon filter impact | Diagnostic COMPLETE | d=5, T=100; eight calls, six backward passes, same stopping iteration 6, fresh final N=2000; final logZ differs from R by 0.0022013 |
| R3b/R3c/R4d/R4e/R5/R6 | Open | Tasks and closure conditions below; no pretend runnable implementations |

The shared fitter/backward/controller is
[iapf_finite_tf.py](../../bayesfilter/score_study/iapf_finite_tf.py); it calls
the shared [paper value kernel](../../bayesfilter/score_study/iapf_paper_tf.py).
All repeated numerical particle/time/optimizer work is in TensorFlow with
stable signatures and native loops. Python orchestrates adaptive passes and
artifacts. The tested route has no NumPy numerical path, pfor, per-particle
Python loop or observed repeated tracing.

The new sensitivity test holds backend and inputs fixed except for about
1e-14 perturbations of log targets. Fitted parameters change by roughly 0.003.
Earlier gradient replay agreed within 4.1e-17. GPU graph also showed divergence;
XLA alone is not the explanation. A CSV precision defect was repaired and
3,142 exact matrix roundtrips pass, but that repair did not close parity.
The actual filter completes; this is not evidence that the numerical issue
is harmless over new datasets or that the paper is fundamentally wrong.

## Remaining gaps and whether closure is known

| Gap | What is known | Required next evidence |
| --- | --- | --- |
| iAPF finite solver semantics and sensitivity | A discriminating repair path is known; a successful numerical remedy is not yet validated | R4d: isolate source-semantic edges and finite trajectory amplification, then fresh downstream/distributional checks; preserve failed 1e-5 parameter criterion |
| Exact original iAPF specification | Not recoverable from the sources inspected so far | Recover original implementation/settings or explicitly retain reconstruction status; more GPU hours cannot identify undocumented settings |
| Full iAPF empirical experiment | Protocol known, total runtime not yet established | R5a: steady-state cost ladder, frozen recipe, all declared dimensions/repetitions, uncertainty and conditional comparators |
| MDPF normalized density baselines | Implementation work with a defined route | R3b: matched-information persistence/motion/UKF/BPF densities, circular observations, source initial cloud, conditional NLL and error |
| MDPF Appendix B learning | Model/objective known; true coefficients and several training details unrecovered | R3c: recover them or freeze labeled hypotheses, then exact mixture-Kalman and gradient/learning comparisons |
| MDPF TF/TFP GPU implementation | Engineering route known; not implemented | R4e: batch-native learned consumer and actual-call-chain value/gradient parity |
| MDPF refinement and independent repetitions | Required work known; full scope exceeds current resources at measured source cost | R5b: freeze source protocol, measure eligible runtime and secure sufficient total resources before promising all repetitions |

An exact numerical reproduction cannot be promised from an unrecovered
executable recipe. A labeled protocol reconstruction remains a valid research
task. No observed failure presently establishes a fundamental defect in
either paper.

## Executable operation and budget

Run from /home/chakwong/BayesFilter:

~~~text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py status
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py validate
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_replication_master.py run-next
~~~

run-ready executes eligible implemented phases until completion or a failed
child; recover reconciles interrupted manifests without inventing successful
exits. R4d now has executable source controls and regressions. The completed R4c
tranche used 28 recorded launch slots, including prior phases and failed
attempts; the R4d amendment raises the ceiling to 36 inside unchanged time limits. This is not a request for a new approval token: a subsequent bounded
repair tranche may be implemented/reviewed under existing authorization and
the remaining total compute, with its own explicit attempt ceiling.

After attempt 28, campaign charge is **1,017.654854 CPU seconds** and
**42,530.694701 GPU seconds**. Remaining: **782.345146 CPU seconds** and
**36.435918 GPU hours**. [Live status](artifacts/iapf-mdpf-whole-repair-20260928-01/status-after-r4c.json)
is the saved close; the command reports current authority. Ordinary document
editing and short dispatcher unit tests are not numerical campaign attempts.

One previous full MDPF NLL training took 8.589 hours. Ten more at that cost
would require 85.89 GPU hours for that arm alone. Refinement, comparators,
Appendix B and port validation add cost. Do not shrink the repetition count
and keep the full-replication claim.

Use the existing direct interpreter/script command forms. GPU probes/jobs
require escalated access and verified memory growth before initialization.
CPU diagnostics hide devices before framework import. No broad shell approval,
allow-list edit, environment mutation or repeated scientific approval is
needed for authorized local repairs.

## Review and stop rules

The [R4c review](artifacts/iapf-mdpf-whole-repair-20260928-01/r4c-review.md)
accepts the bounded diagnostics and blocks full replication/default promotion.
It is self-review, not independent terminal acceptance. The updated status
preserves failed paper parity even when a later diagnostic completes.

Wrong targets/data/comparators, corrupted evidence, invalid accepted
probabilities, missing required diagnostics and exhausted budgets stop an
affected experiment. A failed fit or parameter-comparison candidate normally
triggers its next discriminating repair; it does not reject the research
direction. Stationarity, descriptive likelihood closeness and elapsed time
are not substitutes for the stated scientific evidence.
