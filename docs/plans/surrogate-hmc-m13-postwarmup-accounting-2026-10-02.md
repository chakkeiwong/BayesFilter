# M13: count every eligible post-warmup draw

The owner correctly challenged the separate verification/retained accounting.
The scientific question is whether the already frozen M13 surrogate-HMC chains
provide sufficiently mixed, precise posterior estimates. The 9,000 transitions
called verification used `adaptive=False`; the subsequent 3,750 transitions
continue the same states through the same kernel. Their names do not justify
excluding the first 9,000 from inference. The ordinary public tuner remains
the tuning authority; passing its diagnostic does not require discarding its
post-warmup draws or starting posterior diagnostics at zero.

## Evidence contract and skeptical audit

Reconstruct the 9,000-draw prefix and all 12,750 post-warmup draws, separately
from the 9,000 warmup draws. Verify checksums, frozen tuning/binding identity,
health, state handoffs, unique seeds, raw-coordinate transformation, and counts.
Use the existing TensorFlow posterior diagnostics. R-hat <=1.01 is the owner's
mixing stopping criterion. ESS, posterior SD and mean MCSE quantify precision;
report them explicitly rather than treating a sample-phase label as a veto.
The old ESS>=400 campaign screen will be reported without silently changing it.
Checksum, target/kernel mismatch, invalid health, or broken state/seed continuity
veto pooling. Acceptance is descriptive only.

The comparator is the saved verification R-hat for exact prefix reconstruction
and the previously reported suffix-only diagnostics for the accounting error.
This is an accounting/diagnostic correction, not a new sampler comparison;
no heuristic ranking or efficiency/default promotion is made. Existing reference
equivalence calibration has specific sample looks and is not automatically
valid at 9,000 or 12,750; this diagnostic will not relabel it as passed.
The frozen importance endpoint still defines an approximate likelihood target.

The audit passes for this bounded analysis: all already generated post-warmup
draws must be reported, even if later samples worsen R-hat. We will not select
only the favorable 9,000-draw prefix and suppress the later evidence. A passing
R-hat is a mixing diagnostic, not a proof of precision for every functional or
of agreement with the original model's exact posterior.

## Assumptions, budget and artifacts

The hypothesis that verification and retained draws are one chain comes from
`hmc_tensorflow_verified.py::_chunk_graph` (`adaptive=False`) and the retained
runner's `state = verification.final_state`. An executable archive continuity
check must confirm it. The affine transform comes from the saved nomination;
compare it against archived raw samples. Diagnostic definitions come from
`hmc_posterior_diagnostics.py`; compare recomputed prefix R-hat with the saved
values. The threshold 1.01 and maximum 20,000 come from the owner's instructions.

Run no new HMC transitions. This is a CPU-only, graph-mode saved-sample diagnostic
exception, with GPUs intentionally hidden before TensorFlow import. Budget:
at most five minutes per attempt, at most two attempts. No midnight campaign
extension is inferred. Preserve all original results. Write the script, command
log, source/input hashes, environment, git commit, timings, diagnostic report,
and result note under
`docs/plans/artifacts/surrogate-hmc-m13-postwarmup-accounting-20261002-01`.

The next action is part of the existing master: correct sample accounting and
the interpretation of its stopped run. Any future sampling must continue the
same accumulated posterior samples and needs a renewed execution budget.

## Completed diagnostic and runtime repair status

Attempt 01 stopped on an exact CPU/GPU initial-coordinate reconstruction check (difference 2.22e-16). Attempt 02 permits FP64 rounding only for that reconstruction; all actual saved state handoffs still match exactly. Attempt 02 passed in 3.23 seconds wall / 6.96 CPU seconds. Results: docs/plans/artifacts/surrogate-hmc-m13-postwarmup-accounting-20261002-02/result.md.

The old launcher's verification/retained split has not yet been repaired in runtime code. Before another launch, count the public tuner's frozen post-warmup draws as the posterior prefix, continue the same chains, and assess the combined history. Do not rerun the old zero-count posterior phase. This repair is part of the existing master. The execution budget is expired; no new sampling was performed or authorized here.
