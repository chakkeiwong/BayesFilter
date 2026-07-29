# PP-UKF true-HMC continuation repair and attempt-11 plan

Date: 2026-07-30

Status: `PLAN_REVIEWED_READY_FOR_EXECUTION`

Parent plan: `docs/plans/bayesfilter-pp-ukf-true-hmc-continuation-plan-2026-07-23.md`

Failed attempt log:
`docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-10-launch.log`

## Research intent ledger

| Field | Binding decision |
|---|---|
| Main question | Do the exact archived `L=9,12,17` chains censored at 3,000 retained draws pass the existing sequential HMC gates when continued to at most 10,000? |
| Mechanism under test | Fixed-identity PP-UKF NeuTra HMC with the frozen candidate-specific epsilon and leapfrog count |
| Expected failure mode | Folded rank-normalized R-hat remains above `1.01` for one or more parameters despite adequate ESS |
| Promotion criterion | Cumulative all-parameter R-hat `<=1.01`, bulk ESS `>=1000`, tail ESS `>=400`, and all declared chunk health checks pass |
| Promotion veto | Nonfinite state, target, or log acceptance; invalid target telemetry; no chain movement; nonfinite convergence diagnostic; or native divergence when exposed |
| Continuation veto | Prefix identity/hash/content mismatch, output collision, aggregate campaign budget exhaustion, or a harness exception that invalidates the artifact |
| Repair trigger | A localized controller, serialization, checkpoint, or harness failure under the unchanged scientific contract |
| Explanatory only | Acceptance, finite extreme log-acceptance count, per-chunk runtime, and descriptive differences among viable candidates |
| Nonclaims | No candidate ranking, sampler superiority, posterior recovery, exact-likelihood, default-readiness, or production claim |

## Attempt-10 classification and budget

Attempt 10 is an infrastructure failure, not a candidate result. The first
new `L=9` HMC chunk returned from the GPU, but the local wrapper passed an ad
hoc object without `BatchedHMCConfig.payload()` to the shared summarizer. The
process then raised `AttributeError` before archiving samples or writing
progress. `L=12` and `L=17` did not start. Attempt 10 contains no candidate
artifact eligible for reuse.

The attempt-10 log existed from `2026-07-23 12:45:26.178208 +08:00` to
`2026-07-23 12:51:09.542062 +08:00`. Charge the conservative full interval
`343.363854 s` even though no scientific artifact survived:

| Accounting item | Seconds |
|---|---:|
| Aggregate through attempt 09 | `42,403.504540` |
| Attempt-10 conservative charge | `343.363854` |
| Attempt-11 carry-in | `42,746.868394` |
| Campaign cap | `86,400.000000` |
| Remaining before attempt 11 | `43,653.131606` |

## Evidence contract

| Field | Binding value |
|---|---|
| Target signature | `d3ed745b4f755582bfce46b24992e9d626e10c1409c46b0518ca8cfc673fc2f5` |
| Frozen transport SHA-256 | `b7a558db1e9a48fcd79333e65771d933342a1933e93869a8d5193ce166019221` |
| Prefix progress SHA-256 | `acd34ab3d4bd1ecf0907c193cb87a4aeed1fa95c6a2d637ece8b6bb8fdd4eec8` |
| Candidates | Exactly manifest indices `(1,2,5)`, corresponding to `L=(9,12,17)` |
| Starting state | Final latent state of each verified 3,000-draw cumulative prefix |
| Next seed | Original retained seed passed to `sequential_chunk_seed` at chunk index `6` |
| Maximum | 10,000 retained transitions per chain, in 500-transition chunks |
| Output | Fresh `docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11/` plus sibling launch log |
| Hardware | TensorFlow/TFP float64, GPU, XLA enabled, TF32 state recorded, memory growth verified before device initialization |

## End-to-end failure analysis

The code trace identified these material defects or untested boundaries:

1. **Shared-config interface mismatch.** The local continuation wrapper passed
   a temporary object lacking `payload()` into `_summarize_batched_hmc_output`.
2. **Private-controller duplication.** The benchmark imported private build and
   summary helpers, allowing its behavior to drift from the shared controller.
3. **Missing one-chunk contract test.** Tests validated prefix metadata and row
   merging but never ran one continuation chunk through the real summarizer.
4. **Incomplete provenance in replacement rows.** The wrapper emitted empty
   warmup checks and only new archives instead of preserving the verified
   attempt-09 warmup and retained-prefix evidence.
5. **Candidate-boundary-only progress.** Completed chunks were not reflected in
   `progress.json`, so a later process loss could waste all work for a candidate.
6. **Incorrect timing evidence.** Each local chunk summary recorded
   `elapsed_seconds=0.0`.
7. **Insufficient prefix-content validation.** Individual chunk hashes and one
   cumulative hash were checked, but their tensor concatenation equivalence was
   not proved.
8. **Weak post-transform checks.** Continuation did not explicitly validate the
   model-coordinate output shape and finiteness before convergence diagnostics.
9. **Convergence-veto propagation.** Nonfinite convergence diagnostics could
   fail the gate without being copied into the row hard-veto ledger.
10. **Non-atomic tensor archives.** A process loss during `write_bytes` could
    leave a truncated file under its final name.
11. **Ambiguous and invalid CLI inputs.** Simultaneous selection modes and
    negative/nonfinite carry-in values were not rejected before execution.
12. **Non-durable exceptions.** A harness error was preserved only in the shell
    log, with no structured failure artifact under the attempt root.
13. **Budget carry-in drift.** Attempt 10's failed runtime was not yet charged.

## Default and assumption audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| Shared retained-continuation API | Existing `bayesfilter.inference.neutra_hmc` controller | A second local sampler drifts from policy | Unit test using the public API and real `BatchedHMCConfig` | Required repair |
| Archived prefix reuse | Attempt-09 hash-bound artifacts | Wrong or internally inconsistent chain prefix | Hash, shape, finiteness, seed schedule, and chunk-concatenation parity | Binding invariant |
| Preserve prior checks | Attempt-09 rows already passed warmup and health | Replacement row falsely appears to lack warmup evidence | Row-level provenance test | Binding invariant |
| Per-chunk checkpoint | Campaign repair/retry policy | Long candidate work is lost after a process failure | Callback test and emitted merged ten-row progress | Required repair |
| 500-draw chunks | Frozen attempt-09 controller configuration | Seed schedule or workload changes | Exact config comparison and chunk-index tests | Reviewed inherited setting |
| Attempt-11 carry-in | Conservative attempt-10 log interval | Campaign silently exceeds cap | CLI validation and result/manifest accounting | Binding hard stop |

## Implementation sequence

1. Add a public shared retained-continuation function to
   `bayesfilter.inference.neutra_hmc` that uses `BatchedHMCConfig`, the shared
   health summarizer, deterministic chunk seeds, cumulative diagnostics,
   archive callbacks, per-chunk checkpoint callbacks, and a pre-chunk stop
   callback.
2. Export and test that API with deterministic CPU fake programs, including
   pass, health veto, convergence hard veto, and callback sequencing.
3. Replace PP-UKF private-helper use with the shared public API.
4. Strengthen prefix validation to prove cumulative tensors equal the ordered
   chunk tensors and preserve warmup/retained checks plus prefix archive
   metadata in replacement rows.
5. Make tensor archive publication atomic and validate model-coordinate shape
   and finiteness before diagnostics.
6. Write a merged ten-row `progress.json` after every completed continuation
   chunk. Mark the in-flight replacement row explicitly incomplete until a
   pass, hard veto, cap, or budget stop makes it terminal.
7. Validate selection modes and carry-in budget before creating the output
   root or importing TensorFlow. Emit structured `failure.json` for exceptions
   after root creation, then re-raise for a nonzero process exit.
8. Run focused CPU tests, compile/static checks, route-policy enforcement, and
   the no-sampling real-prefix preflight.
9. Commit only PP-UKF/shared-HMC lane files and launch attempt 11 with carry-in
   `42,746.868394 s`.
10. Verify tmux, PID, GPU, memory growth/XLA log, and either the first durable
    checkpoint or a structured failure before handoff.

## Skeptical plan audit

- **Wrong baseline:** attempt 09 remains the exact prefix source; attempt 10
  produced no reusable candidate evidence. No chain is restarted or retuned.
- **Proxy promotion:** acceptance and runtime remain explanatory only. The
  original all-parameter R-hat/ESS and health gates remain primary.
- **Missing stop conditions:** output collision, prefix mismatch, chunk health,
  convergence nonfiniteness, aggregate budget, and harness exceptions all have
  explicit terminal handling.
- **Unfair comparison:** candidate controls, prefix states, seed schedules,
  chain count, chunk size, target, transport, and diagnostics remain frozen.
- **Hidden assumptions:** the failed chunk is deliberately discarded because
  it had no archive or diagnostic artifact; deterministic seed reuse recreates
  the same transition from the unchanged prefix.
- **Stale context:** attempt-09 progress is hash-bound; attempt-10 is charged
  but not treated as scientific evidence.
- **Environment mismatch:** serious execution remains trusted GPU/XLA with
  fail-closed memory-growth setup; CPU tests are explicitly mechanics only.
- **Artifact adequacy:** per-chunk archives and merged checkpoints answer both
  scientific continuation and crash-recovery questions; `failure.json`
  answers a future harness-failure question.
- **Plan-could-pass-while-misleading:** preserving only new checks would make a
  continuation row look self-contained when it is not. The repair must retain
  explicit prefix provenance and prior diagnostics.

Audit verdict: `PASS_FOR_IMPLEMENTATION_AND_ATTEMPT_11` only if steps 1--8 pass.
Any test that bypasses the public shared summarizer, any replacement row that
drops prior evidence, or any launch using the old carry-in fails this audit.

## Implementation and prelaunch audit result

Status: `PASS_FOR_ATTEMPT_11_GPU_PREFLIGHT`

The implementation trace and focused verification closed all thirteen defects
listed above:

- retained continuation is now a public function in
  `bayesfilter.inference.neutra_hmc` and constructs a real
  `BatchedHMCConfig` for every chunk;
- the PP-UKF route no longer imports either private HMC builder/summarizer;
- shared-controller mechanics tests execute the exact summary/config boundary
  that failed attempt 10;
- replacement rows preserve attempt-09 warmup checks, retained checks, and
  archive provenance, while appending new evidence;
- a merged ten-row progress payload is written after every completed chunk and
  marks selected rows nonterminal until pass, hard veto, or the 10,000 cap;
- actual chunk elapsed time is recorded;
- real attempt-09 cumulative latent/raw tensors were proved bitwise equal to
  the ordered concatenation of their six archived chunks for all three
  candidates;
- model-coordinate shape/finiteness and convergence hard-veto propagation are
  enforced;
- tensor archives use temporary-file replacement;
- ambiguous selection and invalid carry-in inputs fail before root creation;
- post-root exceptions create `failure.json` and exit nonzero;
- a 900-second pre-chunk reserve, rounded above the largest preserved
  attempt-09 chunk runtime (`752.734436 s`), prevents likely cap overrun; and
- attempt 11 uses the corrected `42,746.868394 s` carry-in.

Focused CPU-only mechanics and artifact tests passed: `26 passed`. Python
compile checks and `git diff --check` passed. These checks do not support an
HMC scientific claim.

The repository-wide NeuTra route-ledger audit remains blocked by 20 pre-existing
unledgered routes across unrelated HMC lanes, including historical and current
benchmark scripts. Its generated ledger is not tracked, and adding only PP-UKF
would not make the repository check pass. This is recorded as repository debt,
not an attempt-11 scientific or execution veto. A focused committed-source test
passes and enforces that this PP-UKF route calls
`run_sequential_neutra_hmc`/`run_retained_neutra_hmc_continuation` and contains
no private sampler-helper import or local sampler bypass.

Post-implementation skeptical audit: `PASS_FOR_ATTEMPT_11_GPU_PREFLIGHT`.
The repair changes harness correctness, provenance, durability, and budget
enforcement only. It does not change the target, transport, candidate controls,
prefix draws, seed schedule, chain count, chunk size, convergence thresholds,
promotion criteria, or nonclaims.

Trusted GPU preflight passed on 2026-07-30: one NVIDIA GeForce RTX 4080 SUPER
was visible; TensorFlow created `/device:GPU:0`; memory growth was verified on
the physical GPU before logical-device initialization; full-device
preallocation was disabled; and TF32 was enabled. The attempt-11 output root
and launch-log path were both absent. Prelaunch status is
`PASS_FOR_ATTEMPT_11_LAUNCH` after the final focused test/compile gate.

## Launch command

After all prelaunch gates pass:

```bash
tmux new-session -d -s pp_ukf_hmc_24h_20260730_attempt11 \
  "cd /home/chakwong/BayesFilter && \
  TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
    docs/benchmarks/run_pp_ukf_true_hmc_validation_20260722.py \
    --output-root docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11 \
    --replace-candidate-index 1 --replace-candidate-index 2 --replace-candidate-index 5 \
    --resume-progress docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-09/progress.json \
    --prior-elapsed-seconds 42746.868394 > \
    docs/plans/artifacts/bayesfilter-pp-ukf-true-hmc-validation-20260722/attempt-11-launch.log 2>&1"
```

## Terminal interpretation

A candidate that reaches 10,000 without passing is rejected under this frozen
screen, but that does not reject PP-UKF as a research direction. A budget stop
before the cap leaves the candidate incomplete rather than rejected. Viable
candidates remain unranked unless a separate predeclared uncertainty analysis
supports ranking. The terminal note must include decision and inference-status
tables and state the strongest alternative explanation for any rejection.
