# Comparator protocol and short source cost review

Status: source configuration inventory and short DIS-PF execution PASS.
Both scoped paper replications remain INCOMPLETE. No numerical child remains
running. This is independent author PyTorch/NumPy reference work, not an
admitted TensorFlow/GPU/XLA implementation.

## What executed and what failed

Attempt08 resolved the real source configurations for MDPF, TG-MDPF, IRG-MDPF
and DIS-PF, including the 11-trial generator, common parameters, source overrides,
training objectives and checkpoint dependencies. It used no numerical framework.

Attempts09-11 failed before training because newly added job names were omitted
from the reference LD_LIBRARY_PATH dispatcher. Matplotlib therefore loaded the
system C++ library without CXXABI_1.3.15. These are harness failures, not negative
method results. Their 9.278338 GPU seconds remain charged. The repaired dispatcher
covers every command using the reference interpreter. A CPU import regression
passes against the actual author model factory.

Attempt12 executed the DIS-PF source setup_training -> trainer -> dataset ->
optimizer -> validation -> checkpoint chain on the RTX 4080. It used one
pretraining epoch, two MSE epochs and two bandwidth-only KDE_NLL epochs; N=25,
T=17, 5000/1000 trajectories, batches 64/256, labels/BPTT modulo four, one loader
worker and seeds 924011/924012/924013/924014. All 1185 optimizer calls and 19
reloadable best checkpoints were recorded. The process took 155.532327 seconds.
The bandwidth stage consumed the MSE checkpoint and changed only
weighted_bandwidth_predictor.log_bandwidths. Frozen parameter invariance and
dataset reuse passed. Peak Torch allocated memory was 43,251,200 bytes.

The numerical attempt used source-repair08 and manifest-repair08.json. The final
dispatcher-only repair preserves failure classifications on nonzero process
exits, and stops subsequent reference jobs after a shared library failure.
Sixteen focused CPU tests pass, including actual dispatch coverage, failure
classification retention, no PASS from a failed process, and the shared-failure
stop. Ordinary candidate failures still permit independent diagnostics.
Final tests: ../iapf-kdm-replication-continuation-dev-20260924-08/harness-tests.log.

## Source protocol findings

Paper Section 6 and Appendix D.4.1 distinguish direct likelihood training for
MDPF variants from MSE training followed by bandwidth-only likelihood fitting
for discrete comparators. The author files preserve that distinction:
experiment0001/config.yaml selects TG; experiment0002_implicit/config.yaml
selects IRG; importance_sampling_pf_learned_band/config.yaml selects DIS.
The base directory is src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs.

Do not infer evaluation lineage from names. The MSE-named MDPF/TG/IRG evaluation
in configs/our_experiments.yaml:102-108 still loads the NLL-stage checkpoint;
it does not evaluate the MSE-refined model. DIS's active MSE evaluation uses the
pre-bandwidth-fit checkpoint, while its calibrated NLL evaluation is a separate,
disabled source entry. A frozen map from each reported metric/branch to the
intended checkpoint is still required before K3. Preserve these source defaults
as provenance; record any execution override explicitly.

The inspected paper is the local arXiv version. Attempts to download the official
18,857,175-byte NeurIPS PDF timed out; partial downloads are not accepted paper
evidence. The official abstract record was retrieved. The iAPF DOI metadata
provides a version-of-record link, but the publisher record returned HTTP 403.
A bounded GitHub repository search did not identify original-author iAPF solver
code. This does not prove it is unavailable. iAPF solver/floor provenance and
published-version equivalence remain open. See source-search02/search-note.json.

## Descriptive costs

| Source stage | Train + validation seconds/epoch | Trainer-inclusive seconds/epoch | 11 x 800 epoch scenario, GPU hours |
| --- | ---: | ---: | ---: |
| MDPF NLL, prior attempt03 | 33.132 | 34.952 | 80.99–85.44 |
| DIS MSE, attempt12 | 43.106 | 45.157 | 105.37–110.38 |
| DIS bandwidth-only NLL, attempt12 | 24.224 | 25.649 | 59.21–62.70 |

DIS's two stages alone imply 164.58–173.08 hours at these short-run rates.
These columns describe different included operations, not uncertainty bounds.
Pretraining, optional refinement, all comparator evaluations, other methods,
Appendix B, iAPF, late plotting and retries are not included in that sum.
Early stopping may materially change actual work. No full-campaign budget or
method speed ranking follows. Exact arithmetic: cost-projection02.json.

## Decision and uncertainty

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept protocol inventory | Source objectives and dependencies resolved | Named method/class mismatches rejected | Paper/version and final branch map open | Freeze metric-to-checkpoint lineage | Experimental replication |
| Accept short DIS execution | Actual source stages, checkpoint transition and frozen parameters verified | Finite parameters/losses and data reuse pass | Only two epochs per later stage, one seed | Retain as wiring/cost evidence | Learning quality or estimator unbiasedness |
| Repair TG/IRG harness | Import fault localized and CPU regression passes | Prior attempts fail before training | Their numerical path/timing remains unexecuted | Run bounded probes after attempt-budget decision | Candidate rejection |
| Keep full master incomplete | Required paper experiments absent | Source/protocol/tuning requirements remain open | Full resources uncosted | I2a.2, K1b and B, then I3/K2/K3/P/C | Default or HMC readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Attempts09-11 are infrastructure failures; DIS's short finite/update/frozen-parameter screen passes. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Epoch times and short losses; timing ranges are accounting scenarios. |
| Default-readiness | Not established; independent reference only. |
| Next evidence needed | Complete source protocols, independent final training repeats, untouched evaluation and uncertainty, conditional heuristic comparisons, TF consumer conformance. |

Heuristic dominance remains NOT_EVALUATED: this tranche estimates source costs
and checks mechanics, without promoting a learned method. Final conditional
heuristic evaluation remains a K2/K3 requirement.

Post-run red team: passing a short source pipeline cannot exclude later
instability or show that a named evaluation loads the desired trained model.
The strongest alternative to the timing projection is changed late-epoch
plotting/early stopping cost. Longer representative timing and exact checkpoint
mapping would resolve these weaknesses. The unresolved Appendix B consumer
and iAPF fitting policy remain scientific prerequisites.

## Budget and next action

Twelve of twelve launches used. Charged CPU time: 26.154372690s; GPU time: 283.127934446s.
Remaining time: 1773.845627310 CPU s and 616.872065554 GPU s,
but zero launches remain. Do not recycle failed attempts or launch numerical
work outside this ledger to evade its limit. No full campaign has started.

Next: finish TG/IRG cost probes after a bounded attempt-budget decision; resolve
I2a.2 fit/floor policy, Appendix B source protocol and final evaluation lineage.
Preserve both the original -01 and this -02 ledger. Continue routine source,
plan and regression work without pretending it completes a paper experiment.

## Terminal integrity verification

[terminal-evidence-review03.json](terminal-evidence-review03.json) passes:
nine manifest revisions, 119 snapshot-file checks, 1989 checks of unchanged
original-author source files, and all 19 DIS-PF checkpoint hashes. The final
dispatcher-only repair is source-repair09; DIS-PF actually executed
source-repair08. Sixteen harness tests pass. No numerical child is alive.
Older numerical checks retain their original source identities and are not
transferred to the final dispatcher snapshot. The recorded campaign still has
12/12 attempts and unchanged 1800/900 CPU/GPU-second limits.
