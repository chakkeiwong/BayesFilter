# Five-model campaign and range-bearing 16-step outcome

Checked 2026-10-16 00:24 UTC (08:24 Hong Kong). The campaign has stopped;
the five-model task is incomplete. No further GPU work is authorized under
the exhausted allocation. The existing master, LGSSM worker and range-only
supervisor have exited. The ledger records no RUNNING attempts and
86400.5304 seconds charged against 86400 seconds (0.5304 seconds of timeout
termination overhead). The last worker stopped at23:15:28 UTC, 07:15:28 Hong Kong.

| Model | Current outcome | Warmup / retained draws per chain | Maximum modern R-hat | Acceptance | Minimum bulk / tail ESS |
|---|---|---:|---:|---:|---:|
| LGSSM T50 | Adaptive sampling complete | 2000 / 1000 | 1.0070824 | 73.9–74.5% | 567.1 / 386.1 |
| KSC SV T50 | Adaptive sampling complete | 2000 / 1000 | 1.0059979 | 68.5–72.6% | 693.9 / 639.5 |
| Predator–prey T50 | Stage time budget stop | No saved sequential progress | Unavailable | Unavailable | Unavailable |
| SIR T50 | Inference not started: campaign phase limit | None | Unavailable | Unavailable | Unavailable |
| Range bearing T20, 16 steps | Fresh preparation passed; inference time budget stop | No saved sequential progress | Unavailable | Unavailable | Unavailable |

LGSSM and KSC use four chains. Their retained draws are inference draws;
the internal `verification` count does not denote another validation chain.
Both completed with zero divergences and zero force fallbacks. R-hat and ESS
are diagnostics, not proof of complete posterior exploration. Their saved
`posterior_agreement` fields remain `not_checked`. KSC uses its fixed-grid
likelihood endpoint, whose approximation is distinct from sampler convergence.

Range-bearing fresh tuning selected the owner's 16 flow steps with seed offset
400, N4032, FP32, TF32 off and the literature prior. Tuning used78.93 worker
seconds and GPU mechanics81.03 seconds. The two-transition-per-chain mechanics
check passed with finite states, zero divergences and zero force fallbacks.
The inference worker then received only1029.17 seconds, ran1029.70 seconds and
was terminated by its budget. Its last log reports an XLA compilation still
in progress; neither a nomination result, sequential progress nor `hmc.json`
was saved. Thus the repair has not yet been tested over a full warmup. The
earlier eight-step warmup health veto remains separate historical evidence.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve LGSSM/KSC posterior draws | Recorded cumulative R-hat below1.01 | No recorded divergences or force fallbacks | Coverage and endpoint approximation/reference agreement | Analyze saved draws; no redundant validation chain | General superiority or exact nonlinear posterior agreement |
| Keep 16-step range candidate | Full HMC criterion not evaluated | Short mechanics passed; budget continuation veto fired | Warmup robustness and tuned execution cost | After additional compute authorization, reuse unchanged fresh preparation and run range inference in a new output directory | Eight-step failure solved throughout the posterior |
| Five-model campaign incomplete | Three models lack completed posterior sampling | Budget exhausted | PP/SIR and range adaptive runtime/convergence | Continue these existing master phases only with a new allocation | Rejection of surrogate HMC or any model |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Old eight-step range warmup failed its force check. No equivalent new 16-step long-run observation exists. Latest stops are resource limits. |
| Statistically supported ranking | None; this is not a paired method comparison. |
| Descriptive-only differences | Acceptance, ESS and runtime reported for their respective models; no cross-model ranking. |
| Default-readiness | Not established by completed sampling or the short mechanics check. |
| Next evidence needed | Completed 16-step range warmup and retained sampling; completed PP/SIR inference; separate endpoint accuracy evidence where applicable. |

Post-run audit: preparation success cannot substitute for HMC evidence. The
range inference timeout is not evidence that16 steps fail numerically.
The allocation left after earlier phases was too short to complete its XLA
compilation. No warmup state exists to resume for that attempt; the fresh
tuning selection and mechanics results are reusable only while their full
scope/source identities remain unchanged. The 3.89-second warmed mechanics
transition timing uses L=1 and only two transitions, and is not a reliable
full-run estimate. Do not repeat a17-minute inference allocation expecting
posterior evidence.

Evidence: `state.json`, `preparation/advance.json`,
`preparation/range_bearing/filter-tuning/range_bearing/filter-selection.json`,
`preparation/range_bearing/mechanics/range_bearing/pilot.json`,
`hmc/master.json`, `hmc/range_bearing-inference.log`, and
`hmc/range_bearing/inference/manifest.json` under this directory. The original
campaign's `literature-prior-hmc-01/{lgssm,ksc}/inference/hmc.json` contains
completed results; its `budget.json` records all worker charges. Existing
manifests preserve commands, commit, source hashes, environment, GPU/memory
policy, seeds and timing. This note adds no experiment or changed default.
