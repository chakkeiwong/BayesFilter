# D4R4a: adaptive MAMS full-model cost and source preflight

Date: 12 October 2026. This is the next bounded execution phase of the
[generic repair](bayesfilter-generic-trajectory-adaptation-repair-2026-10-11.md),
using the [D4R3 result](bayesfilter-mams-adaptation-feasibility-result-2026-10-12.md).
It is **cost-only**. It cannot verify a posterior, rank methods, or qualify a
reference. NUTS remains excluded.

## Question and evidence contract

Can the already implemented shared discarded MAMS proposer run on the four
full-data nonlinear targets (PP-SGQF, PP-UKF, SIR and structural UKF) with
finite numerical status, preserved source/seed provenance and measurable
pilot/short-trajectory cost under a 7,200-second aggregate GPU allowance?
The comparator is the actual public finite target and the existing saved
32-transition native pricing records. Historical nine-grid costs are a
descriptive planning baseline, not an adapted sampler comparison.

The pass criterion is execution of the same proposer code without model-name
branches, correct prepared-coordinate target and score/status, all planned
target/candidate denominators, durable pilot and short-path records, and a
bounded elapsed/work report. A changed target/data/math source, invalid
score/status, source drift during execution, broken stream identity or missing
memory growth is a continuation veto for the affected run. A candidate
rejection, unreliable IACT, clipped duration domain, or budget exhaustion is
an explicit candidate/resource outcome and repair trigger. Pilot acceptance,
IACT, local movement, elapsed time and short-path health are explanatory only.
No posterior R-hat/ESS/MCSE is used or computed for tuning membership here.

## Source and default audit

The four archived full-model posterior references fail their exact source-hash
loaders because five inference/tuning modules changed after generation; saved
observation inputs and target-source files still match. Three local-geometry
archives also contain a changed geometry driver. Four current full-horizon
value/score parity tests passed against independent evaluators, but probe
parity alone cannot reissue a numerical reference. Therefore this phase does
not load or use those reference means. A later versioned source-compatibility
audit or fresh reference must precede posterior comparison.

The saved local center/factor is an **unadmitted warm-start hypothesis**. The
new consumer checks target ID, observation hash and bytes, dimensions,
positive-definite factor/covariance and finite starts under the current target
code. It records every source mismatch against the geometry archive; it does
not stamp the old geometry as current-source evidence. All methods in a later
paired study must receive the same declared geometry and starts.

Explicit price hypotheses: initial epsilon 0.2 (historical full-model grid
warm start), requested mean duration sqrt(d) (MAMS paper starting hypothesis),
epsilon domain [0.001, 4] (D4R3 bounded search hypothesis), count cap 128,
six four-transition epsilon probes, 32 settling draws, two 64-draw frozen
windows with 16 lags, source coefficient 0.3, acceptance target 0.9 and
duration neighborhood [1,2]. They are **not** generic or model-calibrated
defaults. Early diagnostics are feasibility, boundary hits, finite status,
actual work and window reliability. The local metric scale is one in prepared
coordinates; it can miss global curvature and is not a posterior criterion.

Per target, price each frozen proposal with 32 fresh transitions per chain,
resetting the same four prepared starts. This is inherited from D4a to make
costs interpretable, and is too short for posterior conclusions. Retain every
generated, omitted, failed and unstarted proposal in the inventory. New
qualification and posterior sampling require a separate D4R4b design.

The source closure includes the new price consumer and all dynamic target
factories. Commit a coherent snapshot before GPU execution; the shared
checkout's untracked historical full-model scripts are not implicit source
authority. Keep the four old reference/geometry archives immutable.

## Budget, steps and stop conditions

Reserve at most 1,800 aggregate worker-seconds for focused implementation,
CPU-only source/geometry checks and tests, and 7,200 aggregate GPU-supervisor
seconds for one four-target price attempt and a localized retry if needed.
This is a suballocation of the settled 154,011.091554 seconds, leaving
145,011.091554 seconds unallocated while reserved. Use a fresh versioned root
under `N5/generic-adaptation/full-model-pricing-01/`. Reconcile actual
supervisor elapsed and release unused reservation at terminal. The parent
enforces the total cap and preserves every target row; it does not impose a
fixed per-target timeout when machine load varies. Check cooperative budget
before each pilot window and each priced member. A nonpreemptible compiled
call is bounded by the parent process. No automatic budget expansion.

1. Add an explicit warm-start-only geometry preparation mode, leaving its
   strict default unchanged. Check data and covariance against the current
   target, record old source mismatches, and issue a fresh binding identity.
2. Add one cost-only consumer that calls `propose_mams_trajectory_candidates`
   and the existing native execution binding. It issues no tuning artifact.
   Give all four targets the same algorithmic settings and fresh named seeds.
3. Run focused CPU hidden-GPU mechanics tests and a tiny target smoke. Review
   the staged source tree; commit it. Probe a trusted GPU with memory growth,
   then run the four-target price in an immutable source snapshot.
4. Audit source/operator hashes, actual versus conservative work, status
   health, member denominators, target differences and budget. Refresh D4R4b
   from observed costs; no full posterior launch without a reference bridge.

## Skeptical pre-execution review

The old grid is not an adequately adapted baseline; it enters only as a cost
planning comparator. Short native paths and pilot IACT do not become
qualification or performance promotion criteria. The four targets share a
finite full-data target factory but have different likelihood costs, so the
same transition count is not a fair compute comparison. Prior local geometry
is a hypothesis, not a qualified map; source mismatches remain explicit.
Saved references are unusable under current strict loaders, so a posterior
run would not answer the scientific question now. GPU/XLA, memory growth,
source snapshot and total cap must match the execution plan. These conditions
make cost-only pricing meaningful and bounded; the plan is suitable to
implement and run without implying D4R4b confirmation.

The full-model source preflight also found that the conservative working-tree
import closure includes unrelated untracked validation modules. Build and
audit the launch closure from the committed immutable snapshot, where the
actual four-target runtime modules and the new consumer must all be present.
Do not stage unrelated shared-worktree modules merely to satisfy a static
working-tree closure. Fresh seed block 5200 is a convenience hypothesis,
disjoint from the prior D4R3/D4a blocks and recorded per call.

Planned worker command, after a trusted GPU probe selects the host device and
the scoped source commit is archived (replace `<gpu>` and `<snapshot>` with
those recorded values):

```bash
env CUDA_VISIBLE_DEVICES=<gpu> TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python <snapshot>/scripts/price_mams_adaptation_full_models.py \
  --geometry-root /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5 \
  --output /home/ubuntu/python/BayesFilter/docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/full-model-pricing-01 \
  --targets NL-PP-SGQF NL-PP-UKF NL-SIR NL-STR --seed 5200 --max-wall-seconds 7200
```
