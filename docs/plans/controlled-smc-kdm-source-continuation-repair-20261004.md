# Replication-only program repair and continuation

Authority: the owner's request to repair, review and execute the master program.
Attempt 143 was interrupted after an Octave memory failure; the companion memory-repair plan governs its verified source-preserving retry. This revision
uses the existing CPU allocation; it adds no GPU allocation or training scope.

Owner clarification, 4 October: C2 is the separate paper-statistics analysis. Its full-matrix gate is not a prerequisite for no-worse source-to-port replication. Audited intrinsic author failures are accepted baseline outcomes; no author-method repair is planned.

Current execution disposition: C2 is BLOCKED by attempt145's Lorenz source
covariance failure. The ineligible queue has been retired, and new starts reject
the known failure immediately. The analyzer's complete-matrix requirements
remain unchanged. Independent neuroscience work continues, followed by the
[reviewed frozen-case diagnostic](controlled-smc-lorenz-source-failure-replay-20261004.md).

## Review findings and repairs

The master described C2 analysis but the executable stopped after C1. Its queue
also expired after 11,000 seconds, shorter than the approved 72-hour source run.
Add a bounded C2 phase and a detached, duplicate-safe continuation whose waiting
deadline follows the actual predecessor deadline. Never read partial matrices
as completed evidence or launch overlapping numerical jobs.

The old Lorenz localization is already done, not a new experiment to repeat:
attempts 88/90 captured identical source inputs and found fit discrepancies;
91–94 separated Gram rounding from Octave's conditional least-squares fallback;
95 reproduced invalid precision in the author code on the frozen failing input.
See controlled-smc-lorenz-matrix-crossover-result-20261001.md. Preserve those
findings. A diagnostic PASS means that its measurement completed, not that the
port passed. Attempts 133/136 rejected later numerical candidates. E0 remains
excluded. No existing mismatch is relabeled a successful translation.

## Research intent and evidence contract

Question: what do the complete, unchanged author-source Figure 6 and Figure 4
right runs show under the already frozen C1/C2 statistical specification?
Comparator: original APF and BPF at the source settings, not a replacement
baseline. C1 execution identity requires all 12,000 cases, source input hashes,
native-result/CSV agreement, and the source APF particle-count/refinement link.
C2 computes sample variance(logZ)/mean(logZ)^2 with denominator n-1.

Use the existing 5,000 within-cell bootstrap draws, seed 202610035, 95% pointwise
intervals, and simultaneous max-absolute-log-deviation bands for each experiment
grid. Lorenz resamples paired cSMC/APF repetition indices; neuroscience resamples
independently. Keep the unnormalized Lorenz source convention primary. Also
report the normalized mean and relative variance after adding
-101*6/2*log(2*pi*1e-4); variance is unchanged.

Primary analysis completion: every cell accounted for with a defined statistic
or an explicit veto. Comparative evidence supports a lower relative variance
throughout a grid only if the entire simultaneous upper band is below one.
Nonfinite source output vetoes the unconditional grid inference; never drop it
silently. Conditional finite-only descriptive statistics retain failure counts.
Undefined or nonfinite bootstrap statistics also veto the grid inference.
Wrong scope, missing/duplicate cases, broken source hashes, wrong settings or
corrupt artifacts invalidate analysis and stop continuation.

Runtime, ESS, coefficient differences and refinement counts are explanatory.
The source APF/BPF pair is the replication comparator; additional heuristic
adversaries cannot redefine the owner's replication request. No port parity,
MATLAB/Octave equivalence, exact published-plot equality, superiority beyond the
declared statistic, default readiness, HMC readiness or complete C/K replication
is concluded. The paper lacks original per-repetition draws and uncertainty.

## Assumptions and pre-mortem

All experimental defaults remain those frozen in the C1 plan. The bootstrap
assumes exchangeable repetitions within a fixed grid cell; sequential RNG
streams and unavailable original author seeds are disclosed limitations. The
paired Lorenz rule follows the APF N dependency; pairing neuroscience would be
wrong. Bootstrap percentile interpolation is linear between adjacent order
statistics. Synthetic scale-invariant pairs test both pairing and statistics.

The run could exit successfully while a scientific comparison is undefined.
Therefore execution completion, source numerical failures, statistical support,
and replication completion are separate fields. Missing data fail closed;
observed source failures are retained and explained. A small CPU-only synthetic
check verifies the analyzer and orchestration; it supplies no research result.

## Execution and budget

New phase: c2_source_results, dependent on c1_source_full. One planned attempt,
maximum 600 CPU wall seconds, charged to the existing allocation. The repaired C1 timeout explicitly reserves
600 seconds for analysis; the initial attempt did not leave this full margin. Source data are read-only. Output uses a fresh
numbered attempt directory. No framework/GPU work happens while waiting.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py start --phase c2_source_results
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py progress

The queued supervisor stops if the source attempt fails, its process disappears
without a completed artifact, the waiting deadline expires, or the remaining
CPU allowance cannot cover analysis. It does not automatically launch P1, KDM
training, solver modifications or E0. It writes an explicit terminal result.

## Remaining source-to-TensorFlow work

P0 must inspect actual consumer calls to design construction, solver semantics,
coefficient unpacking, policy accumulation, Gaussian integrals, resampling and
adaptive stopping. Use source-valid inputs and captured draws. Existing checks
have established design ordering and bounded tracing, but not full forward
parity. Exact coefficient tolerances on ill-conditioned normal equations are
diagnostics, not an invented paper-quality criterion. Freeze downstream error
and no-extra-failure criteria before P1; alternative algorithms cannot close P0.

KDM K0 still needs the Appendix B numerical initialization/training schedule or
a fully specified, explicitly identified bearings source experiment. Then use
the actual author training consumer to compare fixed draws, density, IWSG
gradient and optimizer updates, followed by matched independent training. The
nonlinear Cappe toy dataset cannot substitute for Appendix B. Configuration
generation and the existing primitive checks do not complete training.

Table 1 exact datasets and the multimodal missing driver/horizon discrepancy
remain explicit source gaps. J reports each source and port verdict separately.

## Skeptical audit before execution

Pass for this bounded continuation: original baselines and statistics are
retained; source validity and statistical ranking stay separate; incomplete
results cannot enter the analyzer; pairing and normalization are explicit;
budget and queue deadlines are bounded; source failures do not activate E0.
Validate the actual dispatcher-to-analyzer command and negative dependency,
corruption, duplicate-launch and failure cases before queuing. The analyzer
result and its source manifest preserve commands, environment, seeds and hashes.
