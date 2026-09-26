# FAB+IAF training execution checkpoint

Active checkout: `/tmp/BayesFilter-neutra-fab-20260925`, branch `main`.
The shared `/home/ubuntu/python/BayesFilter` checkout is preserved on
`preserve/shared-main-before-fab-20260926` with unrelated dirty changes.
Do not reset, clean or restore that checkout.

Owner request: commit/merge/push and test whether FAB training of the canonical
IAF improves whitening while covering the modes; training only, no HMC.

Current commits on origin/main: FAB integration `10e28f32a`, initial plan
`15e57c9d0`, runner `07d8084db`, serialization repair `d7194ce40`, corrected
coverage/canonical width and reviewed plan `07167c3a4`, bounded queue `695498f57`.

The governing plan is `bayesfilter-fab-iaf-training-campaign-2026-09-26.md`,
including its recovery audit. Repairs were needed for JSON scalar conversion,
wrong independent-bank importance denominator, and accidental width-4 IAF.
The active arms use width 16, FP32 map, FP64 target, TF32 off, XLA, verified
memory growth, native batch 32, and no global gradient clipping. FAB uses
source-form Adam and RKL uses existing Keras Adam; this is an explicit
comparison of trainers, not an exactly isolated objective comparison.

Validation: 16 focused FAB tests and 2 campaign diagnostic tests pass. Both
one-update GPU preflights completed the full artifact/export path. Original
failed preflight cost 15.960 seconds; repaired preflights cost 32.094 and
22.046 seconds. They are mechanics evidence only.

The one-seed 240-update analytic control completed, cost 38.474 + 25.898
worker seconds. FAB raw component responsibility masses were
`[.56721,.27294,.15985]` versus exact `[.5,.3,.2]`; reverse KL was essentially
`[1,0,0]`. FAB score residual per-coordinate RMS 27.568, density RMS 22.193;
RKL .2951 and .4132. Thus local whitening alone hides mode collapse. No ranking
or global success is established. Total smoke/control charge: 134.473 seconds
of 600 (the exact individual JSON wall values govern).

The six q20 arms are executing under the stdlib supervisor
`docs/benchmarks/supervise_fab_iaf_campaign_2026_09_26.py`; GPU 1 and 2 are
queued automatically, GPU 0 has unrelated jobs. Run root:
`docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26/q20-r1`.
Read `queue.json`, then only the last progress row and selected result fields.
Supervisor log: `/tmp/fab-iaf-q20-supervisor-r1.log`. Unified exec session
10407 owns the supervisor. Do not launch duplicate arms. Worker caps are
3,600 seconds FAB and 1,800 reverse KL, 16,200 seconds total including final
diagnostics. SIGTERM timeout/kill grace is included in each cap. Engineering
failure pauses pending jobs for repair; finite poor fit continues the plan.

Next: monitor the initial probes and pass pricing, then completed arms. Verify
paired initialization/target equality and artifact hashes, summarize complete
240-update pairs, and write terminal decision/inference tables. If a cap stops
an arm early, do not treat it as a matched-update pair. q20 sign-region mass
is not an enumeration of modes, and independent prior ESS must accompany it.
Preserve every attempt and account its worker time before any retry. Commit
and push the terminal result and relevant reproducibility artifacts.
