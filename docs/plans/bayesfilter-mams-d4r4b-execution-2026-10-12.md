# D4R4b preconfirmation execution record

Plan: [D4R4b preconfirmation](bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md).
Date: 12 October 2026. Status: reference compatibility and first
discarded-window rung complete; structural rung terminal with partial evidence. No
posterior or tuning-membership authority follows from this cost diagnostic.

Current checkpoint: the 1024/256 structural rung exhausted 7200.434712
supervisor seconds. Its sixteen complete pilot calls and 384 tensor receipts
pass the read-only `window-1024/terminal-audit.json` audit. First-duration source
IACTs are 19.4152 and 20.3532, with all thirteen observables usable. The second
duration remains incomplete; no proposal cohort or short price was finalized.
The stale archived running manifest is reconciled explicitly against the
terminal supervisor/result, without rewriting the old files. Heavy GPU
contention was observed and is an explanatory cost diagnostic, not a candidate
or implementation veto. The [consumer repair and settlement](bayesfilter-modern-hmc-adaptive-consumer-repair-2026-10-12.md#terminal-review-and-attainable-evidence-repair)
records the source-bound deployment checks, remaining budget, terminal decision
tables and next executable repair. Earlier running-status entries below are
dated history. No worker from these two window rungs remains active.

## Prelaunch review and source

The skeptical plan audit checked the archived-reference baseline, the
independence of reference compatibility from cost pricing, the unproven
64/16-to-256/64 window hypothesis, complete four-target denominator, explicit
stop conditions, GPU contention, and the risk that a healthy local path might
still miss posterior mass. It found one code defect: caller-supplied subset
indices could be distinct yet identify the wrong targets. The consumer now
requires exact original target positions. The four-model isolated CPU-hidden-
GPU suite passed seven tests in 453.26 seconds from staged tree
`59cb47ad1bb47288b33afcec81e17525d3a525cd`; the focused index check
passed after correction in 2.03 seconds in the working tree and 2.77 seconds
from the committed archive. These are mechanics checks, not sampler evidence.

The first-rung source is immutable archive `N5/generic-adaptation/source-163d24b34/`
from commit `163d24b34b1350f3db0681867ea4d03c580f87c2`, tree
`e18a3988442bff81d406c8bf6ef0985ca6481513`, limited to committed
`bayesfilter`, `scripts` and `tests`. Its compiled Sylvester operator has
SHA-256 `2feb93135f2359e60aca446f52f8468b104199b4d398d4fe6057030df20f23d6`.
The archive's `.validation-source.json` binds these identities. The source
contains both the consumer and the tested SIR status repair.

GPU 1 was occupied by a separate NeuTra process (PID 1990789), using about
454 MiB at the capacity check. The trusted GPU probe selected otherwise idle
host GPU 0, also an RTX 4080 SUPER, and verified TensorFlow 2.20.0 device
placement and memory growth before initialization at 2026-10-11 18:33:46 UTC.
This cost-only fallback preserves the hardware class, source, settings and
allocation. It cannot support a cross-device speed ranking.

## First rung command and bounds

Run from `N5/generic-adaptation/source-163d24b34/`:

```bash
env CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/price_mams_adaptation_full_models.py \
  --geometry-root /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/window-256 \
  --targets NL-PP-SGQF NL-PP-UKF NL-SIR NL-STR --seed 5400 \
  --fixed-window-transitions 256 --max-autocorrelation-lag 64 \
  --plan docs/plans/bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md \
  --max-wall-seconds 7200
```

The output path is fresh. Its sibling `window-256.log` preserves console
output. The supervisor cap is 7,200 seconds; the full phase separately reserves
at most 3,600 local-check seconds and a conditional second 7,200-second GPU
rung. Charge elapsed supervisor time once, preserve failures and unstarted
targets, and release unused reserve at terminal. No R-hat, ESS or MCSE is used
to choose a rung or a tuning candidate. The separate reference-compatibility
bridge must pass before any posterior comparison.

The first rung started at 2026-10-11 23:58:25 UTC after tool approval, parent
PID 2058439 and worker PID 2058441, session `86753`. Its manifest confirms
GPU 0, XLA, verified memory growth and the intended commit. At approximately
14 minutes, PP-SGQF remained in discarded adaptation with advancing completed
work; no target failure had been reported. The earlier readiness probe and
actual start timestamps are distinct and preserved.

## B0 reference audit preparation

The [source audit](bayesfilter-mams-reference-source-audit-2026-10-12.md)
records the exact nine source-difference arguments and the distinction between
an independent scalar reference and the old native-parity binary. The new
diagnostic reader reaggregates all eight scramble rows, binds current and
archived source, checks data and quantity charts, and keeps old references
unchanged. Thirty-one corruption/identity tests passed in 0.21 seconds; import
and compile checks passed. These tests use synthetic evidence and establish
reader mechanics only. The real four-archive summary/source precheck passed.

The CPU parity command is bounded to 2700 wall seconds, charged within the
3600-second local allowance (453.26 seconds already used by the isolated
four-model suite). This ceiling is an allocation choice, not a predicted
runtime. Two CPU threads inherit the original reference-driver diagnostic
setting and are not used for performance comparisons. Execute the new
diagnostic driver from its next scoped immutable commit, while passing
`source-163d24b34` as the exact current target source. Preserve four original
reference points and four starts using the first-rung seeds for every target.
The output root is `N5/generic-adaptation/preconfirmation-01/reference-01/`;
the sibling log preserves failures and progress. No reference compatibility
is claimed until the current native numerical checks finish.

## B0 terminal result and continuation review

The CPU reference diagnostic finished at 2026-10-12 00:22:58 UTC in
165.772147 seconds. The driver was the immutable `reference-driver-b1cb4e51c`
archive; the evaluated target remained `source-163d24b34`. The exact command,
source/operator identities, environment, seeds, quantities and four reports
are preserved in
[the terminal manifest](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/reference-01/manifest.json).
GPUs were deliberately hidden. This is independent reference evidence, not a
GPU qualification run.

| Target | Maximum absolute value difference | Maximum scaled score difference, FD steps `1e-4` / `1e-5` | Result |
| --- | ---: | ---: | --- |
| PP-SGQF | 2.70e-13 | 1.59e-7 / 4.26e-9 | Compatible |
| PP-UKF | 1.49e-12 | 1.59e-7 / 3.49e-9 | Compatible |
| SIR | 6.82e-13 | 1.94e-5 / 1.87e-7 | Compatible |
| Structural UKF | 2.26e-11 | 1.20e-6 / 1.10e-8 | Compatible |

All four source audits, eight-point native parity checks, quantity-chart
checks and final-two-rung reference reaggregations passed. The SIR coarse-step
score error is close to its inherited `2e-5` bound; its much smaller error at
the finer step is consistent with finite-difference truncation. No tolerance
was changed. These checks supplement the inspected dependency argument; they
do not prove global numerical equivalence from eight points alone. The bridge
applies to this exact current source/data/quantity identity. Later numerical
source changes require another compatibility check.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete B0 | Audited integrand/source identity, quantity definitions, parity and reference refinement all pass | No reference continuation veto remains for the audited source | Empirical integration refinement has no rigorous global error bound | Finish B1, then freeze a priced qualification and posterior comparison using the compatible references | Sampler convergence, robust adaptation, exact nonlinear likelihood truth |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No source, data, chart, quantity, native-status or refinement veto in the four checked references. |
| Statistically supported ranking | None; this is a reference compatibility check. |
| Descriptive-only differences | Finite-point errors in the table. |
| Default-readiness | Unchanged; MAMS remains experimental. |
| Next evidence needed | Complete longer-window costs; remaining analytic controls; fresh qualification and paired posterior comparisons. |

The reference run consumes 165.772147 seconds of the existing 3,600-second
local allowance. Keep the full allowance reserved until phase settlement;
do not charge this time again outside it. The running GPU rung remains inside
its existing 7,200-second supervisor allocation. PP-SGQF has completed with
both length estimates usable, four frozen proposals and four healthy short
prices. This is availability and cost evidence only. PP-UKF is running; later
targets retain their place in the declared order.

The continuation audit found no reason to interrupt that run. It also checked
the master's remaining D4R3 requirements: antithetic/oscillation controls,
calibrated tail quantities and funnel quantities in original model coordinates
are still missing. Those checks must be implemented under an explicit bounded
design before robustness interpretation. The strongest alternative explanation
for the positive pilot is adequate local exploration within an incomplete
region; a compatible reference makes that explanation testable, but does not
itself disprove it.

## Analytic controls while B1 runs

The [analytic-control plan](bayesfilter-mams-analytic-controls-2026-10-12.md)
and [result](bayesfilter-mams-analytic-controls-result-2026-10-12.md) record
the repaired partial-funnel map, a matching exact-map model, declared model/tail
quantities and oscillation controls. Forty-two distinct regressions passed.
The 23.603492-second CPU/XLA calibration completed 384 positive and 192 negative
reference replications; all negative controls were detected. Strong antithetic
dependence often left lugsail MCSE unavailable, so general calibration and
posterior-policy promotion remain open. This is reference diagnostic work,
not a GPU sampler comparison. Known local usage is now 699.295639 seconds
inside the unchanged 3600-second reserve.

PP-UKF has now completed with both pilot estimates usable, four proposals and
four healthy short prices. The all-model first rung is still active on SIR;
structural remains in order. A trusted automatic capacity check at
2026-10-12 00:51:33 UTC reported `no_idle_policy_permitted_gpu`. That is current
contention, not a hardware or method failure. No extra GPU job was launched.

The ensuing 900-second CPU reference/debug pipeline allocation completed in
93.538730 supervisor seconds. All six Student-t/exact-funnel/partial-funnel
members verified, exported/reloaded and passed their short posterior checks;
one Student-t member's second-moment reference screen raised an alarm.
It remains a verified tuning member. This is scoped wiring and reference
diagnostic evidence, not broad calibration or GPU qualification. The analytic
result records the exact discrepancy, alternatives and all member outcomes.
Known local usage is 792.834369 seconds inside the unchanged 3600-second
reserve. SIR has now also completed with both estimates usable and four
healthy short prices; structural is the final running target of the first rung.

## First-rung terminal audit and conditional continuation

The first rung completed with exit code zero in 4144.141505 supervisor seconds.
The [terminal audit](artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/window-256/terminal-audit.json)
checked all 177 source hashes, the compiled operator, eight geometry inputs,
durable price bundles, work accounting and 140 distinct seed pairs. All
fourteen proposals received healthy 32-transition prices. No proposal is a
verified tuning member from this evidence.

| Target | Source length estimate | Slow-observable length estimate | Healthy prices | Planned disposition |
| --- | --- | --- | ---: | --- |
| PP-SGQF | Usable | Usable | 4/4 | Keep first-rung proposals |
| PP-UKF | Usable | Usable | 4/4 | Keep first-rung proposals |
| SIR | Usable | Usable | 4/4 | Keep first-rung proposals |
| Structural UKF | Unavailable | Unavailable | 2/2 | Run the predeclared 1024/256 rung |

The continuation audit passes: the structural outcome is the precise
information-insufficiency trigger declared before the run. No reference,
numerical-source or artifact veto occurred. Increasing this target's window
does not use posterior convergence or acceptance to choose membership. Preserve
both rungs, including the first rung's fallback proposals; a second unavailable
estimate is a bounded diagnostic result, not a reason for an unplanned third
rung or rejection of MAMS.

Settle the first rung at 4144.141505 seconds, releasing 3055.858495 seconds of
its 7200-second reserve. With the full 3600-second local reserve (known usage
792.834369 seconds) and conditional 7200-second second rung still reserved,
the unallocated campaign balance is **136075.474013 seconds**. Do not charge
the local diagnostic costs again outside their reserve.

A trusted NVIDIA capacity probe at 2026-10-12 01:09:40 UTC selected idle host
GPU 1, UUID `GPU-3eb0894d-1bb7-c79f-73a7-ac5b5c1dc79c`, the same RTX 4080
SUPER hardware class. The second rung uses the unchanged immutable
`source-163d24b34` and original structural target index 3. Its actual manifest
must verify TensorFlow memory growth and device/backend identity before any
result is accepted. Run from that source with the fresh output below:

```bash
env CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/price_mams_adaptation_full_models.py \
  --geometry-root /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/preconfirmation-01/window-1024 \
  --targets NL-STR --target-indices 3 --seed 5600 \
  --fixed-window-transitions 1024 --max-autocorrelation-lag 256 \
  --plan docs/plans/bayesfilter-mams-d4r4b-preconfirmation-2026-10-12.md \
  --max-wall-seconds 7200
```

The worker actually started at 2026-10-12 05:12:58 UTC (tool session `52617`).
The manifest verifies host GPU 1, XLA and growth on every visible physical
device before initialization; the structural seed is 35600 as specified by
the preserved original index. The earlier probe timestamp is not the launch
timestamp. Initial progress is in discarded adaptation.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete first rung; continue structural only | Full proposal/cost inventory and terminal audit pass | No source, numerical-health or artifact veto | Local estimates may miss slow global exploration; structural estimates remain unavailable | Execute the declared longer rung, then price the qualification design | Tuning membership, posterior validity, method ranking or a universal window default |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | All fourteen short prices pass numerical health; this is a short-path screen only. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Proposal counts, length availability and per-model costs. |
| Default-readiness | Unchanged; all new method paths remain experimental. |
| Next evidence needed | Structural rung, current consumer compatibility, adequately searched baselines and fresh posterior/reference evidence. |

The confirmation-harness review found that the older D4b method-fit driver
assumes the old proposal schema, epsilon domain `(.001,1)`, no repairs, strict
historical reference loaders and exactly the original candidate count. It is
not suitable for these adaptive proposals, some of which have epsilon above
one and may create repair descendants. Repair the consumer under the shared
public authorities and the audited reference bridge before launching a new
full-model comparison. Do not silently reuse or weaken the older driver.
