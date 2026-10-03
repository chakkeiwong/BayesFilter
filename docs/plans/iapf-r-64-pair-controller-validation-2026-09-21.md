# Frozen controller comparison: two fresh d80 datasets

This is phase 2 of the authorized 24-hour independent R campaign in
`iapf-r-24hour-campaign-2026-09-21.md`. Run 64 paired replicas on each of two
fresh datasets: data seed 89600080, replica IDs 2701--2764; data seed 89700080,
IDs 2801--2864. These seeds/IDs were not used in preceding campaign plans.
Phase 1 completed all 40 method pairs with valid source/data, fit and tail
records before this amendment. Phase 2 is fixed before inspecting its data.

Question: does shortening the QR controller's likelihood-history window from
six to five reduce work without failing the frozen likelihood and conditional
heuristic screens? Both arms keep k=5, tau=.5, N0=1000, T=100, alpha=.42,
positive floor power 8, delayed doubling, common per-replica seeds and a fresh
final APF. QR is a reconstruction with a different fitting objective from
paper Equation 15. Neither arm is currently promoted. No settings are tuned
on these new datasets.

Exact Kalman likelihood and prefixes are the authority. BPF10000, fully adapted
APF5000 and SIS10000 are the simple comparators, respectively testing the need
for a learned guide, the benefit over an optimal one-step proposal, and the
benefit over no resampling. Evaluate squared relative prefix errors separately
for ordinary and large innovations using the existing chi-square 90% cutoff.
These are fixed-count comparisons, not matched-cost efficiency claims.

Primary practical screen, separately on each dataset: the pointwise 95% mean
likelihood-ratio interval lies in [.8,1.2], the upper SD interval is <=.70,
and mean final particle count is <=1713. A conditional observed mean loss to
any heuristic vetoes promotion even if its uncertainty interval spans zero;
that conservative veto is not a population-inferiority finding. Literal paper
pattern is reported separately (SD interval within [.175,.70], mean resampling
within [35.94,143.76]) and does not identify unpublished author choices.
Use 4000 replicate bootstrap draws, seed 9212026, for pointwise intervals of
means, SDs, paired error, cost and particle differences. No optional stopping.

Non-finite log values, failed fit/tail checks, incomplete records or wrong
source/data/seed identity veto an affected cell. Corrupt shared code/oracle
stops dependent work; a failed scientific candidate does not stop independent
cells or the planned descriptive five-dimension study. Preserve finite log
ratios when exponentiation underflows. No superiority, default change, LEDH,
KDM, TensorFlow/GPU or HMC claim follows from this R experiment.

Launch `run_iapf_r_campaign_phase.py` with the campaign's `phase02.json`, using
the existing CPU-only R environment and two single-thread workers. Each replica
is a resumable unit with a 300-second wall cap; localized timeouts retry only
missing method pairs with unchanged seeds. Expected work is about 9500 summed
worker seconds, charged to the existing 172800-second aggregate and elapsed
deadline. All attempts, source snapshots, observations, Kalman output, fits,
tails, prefixes, histories and decisions remain under the new campaign root.

Default audit: sample size 64 is a predeclared uncertainty-reduction rung,
not tail certification; two datasets distinguish replication noise from data
dependence. Window 5 remains an extension hypothesis; window 6 preserves the
current comparator. QR/floor and baseline counts remain frozen hypotheses
with exact-likelihood and conditional checks. Hardware concurrency affects
timing, so elapsed costs are descriptive. Small-sample heavy tails can fool
bootstrap intervals; the subsequent 1000-replica rung addresses this limitation.

Skeptical audit: PASS. The baseline, numerical authority and full learning cost
are explicit; no proxy is substituted for the primary criterion, no paper-table
tuning is performed, data are fresh, and candidate failure is distinguished
from continuation failure. The next evidence is larger fixed replication,
whether or not a controller clears this screen.
