# Actual SV20 claim endpoint qualification

The claim-data endpoint passed all17 checks. Its values and scores match the
previous endpoint exactly at the checked positions, while excessive quadrature
domain loss now produces a numerical-failure status. The full canonical DCTR
proposal field remains independent of that endpoint. This is preparation for
testing surrogate HMC, not a posterior inference result.

Evidence is in `artifacts/surrogate-hmc-all-models-20260922-sv20-endpoint-01`.
The maximum log-density refinement difference was7.11e-15, maximum relative
score difference1.88e-16 and maximum predicted domain loss2.98e-13 across
the nine declared parameter probes. Replay, row permutation, row independence,
direct canonical call-chain parity and independent typed binding passed.
The checked L1 proposal reversed with zero measured error. All five complete
transitions and five32-row predictive batches were healthy. Peak TensorFlow
allocation was544,101,376 bytes. The prior fifteen focused tests passed;
their CPU cost across two invocations was11.549138 seconds.

The complete L1-only B/R forecast is11,655.452986 CPU-worker seconds and
11,661.125201 wall seconds, including tuning, fresh verification, both retained
looks, cached predictive diagnostics, compilation/analysis allowances and a
1.5 factor. This supersedes the earlier incomplete cost estimate prospectively.
It exceeds the initial SV family allocation and needs recorded redistribution
within the existing campaign caps before inference can launch.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Unsupported conclusion |
|---|---|---|---|---|---|
| Qualified claim endpoint | All17 checks pass | No observed validity or wiring veto | Sparse probes do not prove global reference accuracy or force behavior | Reserve full inference cost and run the public tuner | Posterior recovery or surrogate-HMC validity |

| Inference status | Finding |
|---|---|
| Hard veto screen | No failure in the declared qualification checks |
| Statistically supported ranking | None; no comparative inference run |
| Descriptive differences | Timings and reversal error only |
| Default readiness | Not established; heuristics remain NOT_RUN |
| Next evidence needed | Public-tuned B/R retained inference against qualified R |

Run command: `python scripts/qualify_surrogate_hmc_sv_endpoint.py --output
docs/plans/artifacts/surrogate-hmc-all-models-20260922-sv20-endpoint-01` using
the tftwogpu environment, trusted GPU1, FP32/TF32 DCTR, FP64 endpoint and
integrator, XLA and verified memory growth. The manifest preserves the exact
command, Git commit, source snapshot, seeds, input checksums and hardware.
Exit0;89.839520 CPU-worker seconds and87.842590 GPU/wall seconds charged.
Global usage after qualification and tests is16,668.389978 CPU and
14,933.899787 GPU seconds of the unchanged54,000/226,800 caps.

The strongest alternative explanation for a later failure is inadequate
tuning or uncertainty precision with the restricted L1 proposal. A successful
qualification cannot rule that out. Reference error in unprobed regions remains
the weakest numerical evidence; the existing full-prior refinement, retained
domain-loss guard and posterior comparison remain necessary.
