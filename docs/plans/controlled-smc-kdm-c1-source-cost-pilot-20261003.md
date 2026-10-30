# C1: author-source executability and cost pilot

Date: 3 October 2026. Active parent: controlled-smc-kdm-replication-first-execution-20261003.md.
C0 input: artifacts/controlled-smc-kdm-replication-20260928/attempt139_c0_source_lock.
Result: controlled-smc-kdm-restructure-execution-result-20261003.md.

## Evidence contract and intent

Question: can the pinned author drivers execute at their paper-scale numerical
settings in the installed Octave reference environment, and what CPU reservation
would the complete source matrices require? The comparator is the exact C0 author
source (commit 2696f6d611aae172d10102cb7bfbfd314bb6b88f), not our TensorFlow port.
The candidate under test is the scheduling/reporting wrapper. Author algorithm
files and data are copied byte-for-byte; the wrapper preserves a driver diff.

A PASS means all twelve selected cases are accounted for, with exact settings
and the four drivers completing. It is a harness/executability verdict. Finite
real logZ, ESS where returned, adaptive refinement count, warnings, failures and
cost are recorded separately. Finite output does not certify every covariance or
domain operation. A source numerical failure is evidence, not permission to
add a ridge, change precision, replace the solver or conceal the failure.

Wrong hashes, settings or case coverage veto the harness. A timeout or runtime
error stops that driver and triggers a bounded harness/cost reassessment;
independent drivers continue. Lorenz APF depends on the source cSMC refinement
counts. No failure of this pilot rejects controlled SMC as a method.

Cost projections are descriptive scheduling estimates, not confidence bounds.
No replication, stochastic ranking, port parity, source-runtime equivalence or
default readiness follows from a pilot PASS. Heuristic dominance is not a
promotion criterion for this scheduling question; the author's APF/BPF comparator
drivers are themselves included in the cost measurement.

## Frozen choices and assumption audit

| Choice | Provenance and role | Risk and earliest check |
|---|---|---|
| Lorenz cSMC: N=512, I<=4, T=100; F grid 2.5..8.5, 40 cells | Author cSMC_VaryF.m, Figure 6 source baseline | Adaptive count affects APF cost; preserve returned counts |
| Lorenz APF: N=1382/2867/4505/6144 for 1/2/3/4 refinements, I=0 | Author APF_VaryF.m | Wrong comparator budget; check every case against its cSMC count |
| Neuroscience cSMC N=128,I=3; BPF N=5529,I=0; full data horizon | Author NormConst_cSMC.m / NormConst_BPF.m, Figure 4 right | Wrong source settings/data; assert settings and data hash |
| One repetition at Lorenz indices 1,20,40 and neuroscience 1,10,20 | Scheduling convenience, not a statistical sample of the whole matrix | May understate cost; retain per-cell times and reserve 2x maximum case cost plus startup |
| Seeds 202610031..202610034; separate full-run seeds +100 | Local reproducibility choice; author seeds unavailable | Cannot recover exact published draws; label a rerun of the source design |
| Octave with pinned local statistics dependency files | Installed independent CPU reference, not original MATLAB runtime | Runtime/solver differences; preserve warnings and keep MATLAB equivalence unresolved |
| Original arithmetic and absent additional safeguards | Required comparator identity; numerical validity is observed, not assumed | Invalid original output is retained as source failure; it does not demand E0 repair |

Table 1 is a separate generative-design experiment and is not included in this
pilot. Original Table 1 datasets were unseeded and not saved. Multimodal Table 2
remains blocked by the absent Repeat.m driver and unresolved 100/101-observation
convention. Neither gap is closed by Figure 6 or Figure 4 results.

## Command, budget and artifacts

Use the existing environment /home/chakwong/anaconda3/envs/tftwogpu/bin/python
and installed Octave. CUDA_VISIBLE_DEVICES=-1 and BLAS/OMP/MKL threads=1.
No GPU work, installation, network or source-code numerical modification.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_source_cost_pilot

Each driver gets at most 45 wall seconds; the supervisor caps an attempt at
205 seconds, with 210 seconds total across at most two attempts. This is within
the existing unspent CPU allocation, not a transfer of the GPU grant.
Retries must fix an identified harness issue and fit the remaining cap.

The supervisor creates a fresh attempt directory, snapshots code/plans and
records commands, source/data hashes, Git state, seeds, environment, wall time,
device and budget charge. The runner preserves per-driver logs, diffs, CSV
cases, progress MAT, native result MAT and summary.json. It refuses to reuse a
reference-work directory. Read-only preparation for a larger source matrix
does not authorize or consume its numerical run.

## Skeptical audit, before execution

The audit found and fixed four consequential wrapper risks: treating a reduced
case count as full replication, changing APF cost matching, accepting duplicate
or missing grid cells, and deriving repository paths from a copied script.
Checks now bind to C0 hashes, validate exact case keys/settings and source cSMC
iteration-dependent APF counts, preserve reporting-only driver diffs and take
an explicit repository root. No solver, loss, precision, model, data or
particle-count setting changes. The preflight must check all four driver diffs
and block historical dispatcher phases.

Premortem: a successful command can still hide incomplete matrix coverage,
Octave/MATLAB differences, invalid internal covariances, and underpredicted
full-run costs. Exact case accounting detects the first; explicit nonclaims
and retained source warnings prevent the other three from being promoted.
Once focused preflight passes, this bounded cost pilot is appropriate to run.

Focused preflight passed before launch: Python AST checks, all eight pilot/full
driver instrumentation round trips (removing reporting and restoring pilot
scheduling yields the original source), historical phase launch rejection,
exact C0 dependency and available CPU cap. Source neuroscience T=2999 was read
from the supplied data. No GPU was initialized. Self-audit passed for this
bounded executability/cost question; no independent reviewer was used.


## Observed timeout and bounded retry

Attempt140 completed Lorenz cSMC/APF and neuroscience cSMC (nine cases).
The neuroscience BPF completed one finite case in about42 Octave seconds,
then hit the45-second driver cap. No observed nonfinite likelihood caused
the stop. Retry only this driver, with the same seed and all three selected
cells, at most130 seconds and at most the remaining aggregate pilot allowance
minus2 seconds for finalization. Reuse the nine completed cases by summary
hash and source-lock identity; do not recompute or select among them. Both
attempts count toward the unchanged210-second pilot cap. If it still times out,
retain partial evidence and cost the continuation; do not loop indefinitely.

The Lorenz APF Octave timers summed to7.16 seconds while Python monotonic
reported5.17 seconds. The cause is not established. Preserve both clocks and
flag the mismatch; use conservative reservations, not exact timing claims.
This does not change likelihoods or authorize a numerical method modification.
A larger source run should record CPU time as an additional timing diagnostic.
