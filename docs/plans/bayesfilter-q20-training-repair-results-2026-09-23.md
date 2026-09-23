# q20 training repair execution

The repair plan is implemented and all four bounded training runs completed.
Terminal review found no numerical rejections, corrupt checkpoints or source
mismatches. Clipping now acts occasionally in the repair arms, but the old
scale saturation remains and no new HMC posterior estimate has been produced.
The controlling plan is
[the September23 repair plan](bayesfilter-q20-training-repair-plan-2026-09-23.md).

## Completed training and terminal review

Each arm processed 32,768 training rows. The control, clipping and depth arms
added 1,024 updates, reaching 2,048 lifetime updates; batch 128 added 256,
reaching 1,280. Their final comparisons are between genuinely different maps.
All updates passed finite-state and target-status checks; all exported maps
passed forward, inverse, log determinant and score parity. GPU/XLA training
traced once per worker, with verified memory growth and explicit GPU assignment.

| Arm | Added updates | Clipped updates | Separate final-bank loss change |
| --- | ---: | ---: | ---: |
| Existing cap 10 control | 1,024 | 1,014/1,024 (99.02%) | -1.3888 ±0.2581 |
| Cap 159.39995, depth 2, batch 32 | 1,024 | 7/1,024 (0.68%) | -1.8243 ±0.2967 |
| Same repaired cap, depth 4, batch 32 | 1,024 | 0/1,024 | -2.4598 ±0.4610 |
| Same repaired cap, depth 2, batch 128 | 256 | 0/256 | -1.0585 ±0.1863 |

The loss is the reverse-KL training objective up to its fixed additive terms,
evaluated on 768 paired base draws. These are descriptive normal-approximation
intervals for before/after changes within each arm. They are not a predeclared
comparison between arms and provide no training-seed replication. No ranking
or optimal setting has been established. All four maps have lower heldout loss
than their starting checkpoint; the endpoint assessment records continued
improvement and no observed plateau, not proof of convergence or an endpoint
learning rate.

The clipping-role contradiction is repaired in these observed trajectories:
the proposed safeguard no longer modifies most ordinary updates. This does
not prove its tail behavior, its suitability at initialization, or that
clipping caused every earlier training failure. The unchanged control also
learned, supporting the finding that stopping training early was material.

The scale issue remains. The original second-stage observation-bias log scale
averages approximately −1.9873, -1.9936, -1.9713 and-1.9859 in the four arms,
against its −2 bound. Every point in the 32-point geometry bank has local slope
below 0.1 for that output. The depth arm's new layers have no such slope alert
on this bank and one new scale averages −0.7767, but this does not establish
posterior whitening. Residual transformed scores and downstream HMC behavior
have not been rechecked for these new maps.

The seven attempts (three canaries and four continuations) all completed with
no recovery launch. Total elapsed campaign wall time was 97.35 minutes across
two GPUs; the sum of worker time was 11,108.765 seconds (3.086 hours), including
286.490 diagnostic seconds. The remaining recorded balances are 144,452.404
campaign seconds (**40.126 hours**) and 484.829 diagnostic seconds
(**8.080 minutes**). The watcher finished normally.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close the bounded repair run | All requested updates and final evaluations completed; state and export checks passed | No numerical or artifact veto | One parent map and one stream per arm | Preserve all four checkpoints and carry viable repairs forward | All scientific gaps closed |
| Retain repaired clipping as a viable local hypothesis | Intervention 0–0.68%, compared with 99.02% for control | Control fails its occasional-guard role; all maps remain numerically viable | Other training states and unseen gradient tails | Monitor clipping on further training, including any fresh initialization | Optimal threshold or causal attribution of the loss differences |
| Keep capacity and whitening questions open | Added stages execute and preserve the initial map; original scale remains saturated | Scale pressure is a repair signal, not method invalidity | Residual curvature, mode coverage and sustained learning | Assess repaired-map geometry, then continue training or run fresh fixed-transport tuning under a bounded next phase | Successful whitening, HMC convergence or stable posterior estimation |

| Inference status | Terminal finding |
| --- | --- |
| Hard veto screen | All arms pass numerical/export checks; cap 10 still fails its intended occasional-guard role |
| Viable candidates | Clipping, depth and batch repairs pass the declared local screens; control remains numerical evidence with its clipping defect |
| Statistically supported ranking | None |
| Descriptive-only differences | Loss changes, intervention fractions, scale summaries and runtimes |
| Default-readiness | Not established; the candidate settings are not global defaults |
| Next evidence needed | Sustained training and seed evidence, residual geometry, fresh HMC tuning, posterior precision/coverage/reference checks |

Post-run review: a map can lower reverse KL while learning only a local region
or leaving severe transformed curvature. That is the strongest alternative
explanation to useful downstream geometry. Failed fresh HMC checks or inadequate
mode coverage would overturn an inference of HMC usefulness. The weakest
evidence remains one parent and one training stream per arm. Existing
workspace posterior-checker differences also require resolution before its
results can inherit the frozen source's successful integration checks.

Terminal evidence: [structured review](artifacts/q20-training-repair-2026-09-23/campaign-01/terminal-review.json),
[full results](artifacts/q20-training-repair-2026-09-23/campaign-01/result.json),
and [automatic tranche table](artifacts/q20-training-repair-2026-09-23/campaign-01/tranche-summary.md).
The following launch notes remain as historical context.

## Engineering evidence and source boundary

The initial affected-suite run passed72 tests. Its final integration attempt
correctly rejected a source change made while its supervised stages were
running. A rerun against fixed workspace sources exposed an unrelated current
posterior-checker behavior: tail ESS is unavailable for the binary
`positive_theta_2` indicator, so posterior information checks fail even when
R-hat and precision screens pass. The current
`hmc_posterior_assessment.py` differs from the previously executed source. We
have not weakened the check or claimed an estimate.

The exact frozen execution source passed11 tests, including the complete
process-supervised Gaussian estimation and replay test and the new repair
worker, map-equivalence, Adam-prefix, queue-overlap and accounting tests.
The separate workspace migration fixture also exhausted its allowance before
training: its entire7200-second allowance was already the unchanged7200-second
repair hold. That result tests a budget stop, not a numerical migration failure.
These failures and source differences are preserved in XML reports; they are
not training-quality observations.

The final affected workspace suite passed **64 tests** in64.32 seconds. A
further11-test focused run passed after adding the comparison-replay regression;
the last complete64-test run includes that regression. These are deliberately
CPU-only engineering checks. They do not spend the numerical diagnostic
allocation or establish q20 training quality.

After freezing the numerical worker, further workspace review repaired two
coordinator edge cases: an empty eligible cohort now reaches the next training
rung instead of immediately abandoning the method; and a requested ensemble
rung covers all its temperatures. A source import clears unsupported legacy
plateau counts while retaining genuine distinct-map comparisons. These changes
do not modify the running frozen worker's target, optimizer or repair map.
Repeated reissuance of the same genuine comparison also preserves its plateau
count rather than adding another observation. Reaching a training cap is
reported as a cap, not as evidence of training completion.

## Sanity phase

All three planned canaries completed. Each used eight disposable updates from
the original saved checkpoint; the continuation starts afresh from that saved
checkpoint. Clipping159.39995043193295 is the observed maximum of the declared
saved128 update norms and31 diagnostic batch norms, not a tail guarantee.

| Arm | Updates | Clipped updates | Supervisor seconds | Interpretation |
| --- | ---: | ---: | ---: | --- |
| clip, depth 2/batch 32 | 8 | 0 | 81.1 | No local numerical or occasional-guard contradiction detected |
| depth 4/batch 32 | 8 | 0 | 84.1 | Initial map equivalence passed; short updates viable |
| batch 128/depth 2 | 8 | 0 | 121.3 | Larger batch executes; no training-efficiency selection |

The diagnostic cost is286.49 aggregate worker seconds, charged within the
existing allowance. GPU0 was occupied; GPUs1 and2 were available RTX4080 SUPER
devices. Workers record explicit assignment and verified memory growth before
TensorFlow initialization. The queue charges each worker separately, including
overlapping wall time.

At an interim inspection, the control had completed358 added updates, with355
clipped, and the clipping repair had completed342, with5 clipped. All were
accepted finite updates; each had saved its128- and256-update checkpoints.
These are interim observations, not end-of-tranche learning results. The
depth and batch arms remain in the queue. Estimated remaining wall time was
75–85 minutes with two available GPUs; runtime and future numerical failures
remain uncertain.

## Decision at continuation launch

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute funded continuation | Engineering checks and three local sanity checks passed | No numerical veto in canaries | Long-run learning, residual geometry, seed and mode coverage | Complete control/clip/depth/batch arms with full checkpoints and common validation | Calibration, optimal settings, HMC readiness or posterior validity |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No training-canary veto; current workspace posterior checker still refuses the binary-tail-ESS case |
| Statistically supported ranking | None |
| Descriptive-only differences | Short losses, clipping counts and per-update time |
| Default-readiness | Not established |
| Next evidence needed | Full continuation results, sustained geometry checks, fresh HMC tuning and posterior assessment |

Reset: campaign05 is marked superseded by this repair campaign, with no new
compute allocation. Active artifacts are under
`docs/plans/artifacts/q20-training-repair-2026-09-23/campaign-01`.
Source manifest, exact request and test reports are in its parent directory.
The deadline remains September25 at18:00 Asia/Shanghai. Never run the old
campaign concurrently using its superseded balance.

Persistent service: `q20-training-repair-watch-20260923.service`, now completed.
Its `watcher.json` records the script/request hashes and zero
recovery attempt. Completion writes `result.json` and an automatic
`tranche-summary.md` inside the active campaign. The exact command is the
frozen production benchmark driver's `repair-training` mode with the parent
directory's `request.json` and `campaign-01` output root. No compute allowance
was added, no existing evidence was overwritten, and no commit or push was
performed for this repair.
