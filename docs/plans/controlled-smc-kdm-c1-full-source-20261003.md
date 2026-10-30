# C1 continuation: full author-source Figure 6 and Figure 4 right

Date: 3 October 2026. Parent: controlled-smc-kdm-replication-first-execution-20261003.md.
Executable specification: artifacts/controlled-smc-kdm-replication-20260928/full-source-spec-20261003.json.
Status: 72 additional CPU wall hours approved by the owner on 4 October 2026; review passed and attempt 143 launched at 02:44 Hong Kong time. The run is active.
This does not authorize a new GPU budget.

## Question, comparator and fixed scope

Execute the four original drivers at all their original grid points and
100 repetitions: Lorenz Figure 6 cSMC/APF (4000 each) and neuroscience Figure 4
right cSMC/BPF (2000 each). Baseline: pinned author source commit
2696f6d611aae172d10102cb7bfbfd314bb6b88f and C0 verified data snapshot.
C1 asks for a complete source execution ledger. C2 separately judges how the
source outcomes reproduce the paper. A process exit or finite output alone
does not answer the latter question.

Keep model, data, solver, precision, particle counts, adaptive refinement and
source stopping semantics unchanged. Use explicit Octave independent-reference
execution. Driver edits add timing/output preservation only; full mode does
not reduce grid/repetition counts. Diff round-trip checks must pass.
Use seeds 202610131..202610134, one independent stream per driver; unavailable
original seeds mean this reruns the source design, not the original random draws.

The APF source consumes each corresponding cSMC refinement count and preserves
the author's particle-count lookup. Current CPU costs need not match the
original historical cost ratios; preserve source N and report current cost.
Do not retune N for local hardware and still call it the identical source run.

## Evidence, validity and interpretation

C1 completeness requires every scheduled case and source setting, original
algorithm/data hashes, expected native output matrices, APF iteration matching,
all logs/warnings, failures and exact commands. Missing cases, wrong settings,
changed sources or corrupt output invalidate the execution claim. Finite-real
logZ is a first numerical screen, not proof that every proposal covariance was
valid. Source warnings and failures remain explicit; no clipping, projection,
ridge, precision escalation or alternative solver is allowed in this lane.
If a driver aborts, retain its partial evidence and run independent drivers;
APF cannot run without its required cSMC iteration matrix.

C2 uses the source summaries described in published Sections 6.1/6.2 and
Figures 4/6: sample variance(logZ)/sample mean(logZ)^2, with sample variance
denominator n-1, plus the adaptive refinement distribution. Also report raw
mean(logZ), variance(logZ), failure counts and costs by every grid point.
For Lorenz preserve the source's unnormalized observation convention; report
a separately identified normalized-likelihood view with the derived constant
from the remaining-source audit. Do not use normalized means in a source
relative-variance comparison without making the change explicit.

Predeclared uncertainty analysis: retain100 repetitions per cell; use5000
bootstrap resamples within cells with reporting seed202610035. For Lorenz,
resample matched cSMC/APF repetition indices together because APF N depends on
the corresponding cSMC iteration. For neuroscience the independent drivers
are resampled independently. Report pointwise95% intervals and, for claims
across the entire grid, simultaneous 95% max-deviation bootstrap bands on the
log relative-variance ratio: for each bootstrap draw take the maximum absolute
deviation from the observed log ratio across the full grid, use its 95th
percentile as the common log-scale half-width, then exponentiate both ends. Undefined/nonfinite statistics veto that inference.
No post-hoc removal of failed cases. Conditional summaries of valid results
must be labeled and accompanied by unconditional failure rates.

The paper plots do not supply original per-repetition draws or uncertainty
bounds. Exact numerical equivalence to the published image is not established
by overlap or visual similarity. Report reproduced model/procedure separately
from supported qualitative variance-reduction trends. Require the simultaneous
band to lie below 1 before saying cSMC has lower relative variance throughout
the tested grid. Failure of that test leaves the comparative claim inconclusive;
it does not erase faithful source execution or justify numerical alterations.
Other heuristic adversaries are reserved for claims beyond replication.

No TensorFlow parity, default readiness, robustness, HMC, full Table 1,
multimodal, iAPF or KDM claim follows from these four matrices.

## Budget, commands and stop conditions

Pilot attempts 140/141 account for all 12 selected cases. Mean-time extrapolation
is 30.50 CPU hours, and twice the slowest observed case per driver plus startup
totals 67.90 hours. These are scheduling estimates, not statistical confidence
bounds. Separate Octave/Python clocks disagreed in some cells, so the full
runner additionally records process CPU time. Use single-thread BLAS/OMP/MKL.

Approved additional allocation: 72 CPU wall hours total, one planned full-run
launch. Any localized retry must first repair its execution specification and
fit the same aggregate budget; no new owner approval is needed within it. Per-driver limits in the specification sum 69 hours;
the remainder covers preparation, source I/O, review assembly and any localized
retry. Unused time is not spent merely because it is allocated. Stop at the
first applicable driver/aggregate limit; do not borrow GPU hours.
The owner approved the additional 72 CPU hours and execution on 4 October 2026.
The opening ledger records 259200 additional CPU seconds and no additional GPU
grant; the attempt allowance is 144. Original grant fields and old ledgers are
unchanged. The full-source specification records APPROVED and the authorization
provenance. Localized repairs/retries within this method and total budget do not
need renewed owner approval.

Use the following exact token order with the absolute interpreter and script
paths. Environment setup, logging and detached execution occur inside the
reviewed supervisor; shell wrappers, inline environment assignments, redirection
and command substitution are unnecessary.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py start --phase c1_source_full
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py progress
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B /home/chakwong/BayesFilter/scripts/run_controlled_smc_kdm.py report --phase c1_source_full

The detached supervisor executes the same bounded run command, records live
process identity and writes unique launch logs. The existing campaign lock
prevents overlapping numerical launches. Monitoring reports all four case
counts and does not import TensorFlow or probe a GPU. Rule matches are verified
in the launch record. Existing platform permission controls remain enabled.

No package changes or GPU allocation are involved. Use the installed Octave
and the pinned statistics dependency hashes. Every launch creates a new
attempt directory; preserve all old results.

The prepared full-mode driver copies/diffs live in
artifacts/controlled-smc-kdm-replication-20260928/full-source-preparation-20261003-02/.
They contain no numerical results and cannot be mistaken for an executed run.

## Review and next boundaries

Skeptical audit: correct original baselines and APF budget mapping; no weak
replacement comparator; unchanged numerical settings; sparse pilot costs
treated conservatively; original MATLAB/Octave equivalence left unresolved;
full-data output expected 12000 cases; source ledger and paper comparison
distinct; original RNG not invented; no source-failure concealment.
The unchanged algorithm retains its numerical risks as observed source behavior.
The wrapper passes driver-text round-trip and case-accounting checks.

After completion, assemble C2 with the predeclared uncertainty above, then
freeze P0/P1 tolerances before matched TensorFlow claims. Table 1 needs its own
generative-design/cost specification; multimodal needs missing-driver/horizon
resolution. K0 proceeds independently. None is closed by this continuation.

## Launch review supplement, 4 October 2026

The owner approved the concrete 72-hour reservation. The scientific target,
source/model/data, 12000 cases, seeds, numerical policies, comparisons and
stopping limits remain unchanged. The final operational review found that a
foreground-only supervisor and an unbounded status dump were unsuitable for a
multi-day run. A detached start and bounded progress command now use the same
executor, locks, manifest and timeout handling. Failed full-source runs are
reported as needing repair, not as ready to rerun. These changes add no
numerical route.

Before launch, check approval and budget, source/dependency hashes, generated
driver identity, source settings, negative launch conditions, and the progress
reader on finite/nonfinite/partial records. Verify the detached launch path
reaches the same command and records its manifest. An isolated mechanics
fixture is sufficient for the orchestration changes; no additional pilot
experiment or scientific retuning is needed.
