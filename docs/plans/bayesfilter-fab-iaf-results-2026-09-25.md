# FAB TensorFlow port, chapter and execution status

The author FAB components are ported to TensorFlow, and the canonical IAF and
the existing q20/T30 UKF approximate posterior are preserved. The modern
importance-sampling chapter is written, audited and compiled. **No FAB-trained
q20 map has been produced yet.** All three saved runs still have zero optimizer
updates. The replay path is now repaired and passes a complete GPU/XLA smoke,
but the owner's Sep25 18:00 Shanghai campaign deadline passed before the repaired
fits could resume. A deadline extension was requested; it does not increase
the existing compute allocation.

The implementation is on branch `codex/neutra-fab-20260925` in
`/tmp/BayesFilter-neutra-fab-20260925`, based on canonical commit `6ccfebc02`.
The first implementation/documentation commit is `4d3f75108`; a subsequent
local repair commit preserves the final replay fix. Shared `main` has older
transport files and extensive unrelated dirty changes. Those were not
overwritten or reset. Shared documentation contains the new chapter and these
notes; use the isolated canonical checkout for execution.

## Deliverables

- `bayesfilter/inference/neutra_fab.py`: alpha-2 AIS, identity-mass HMC or
  Gaussian Metropolis mutations, detached loss, corrected prioritized replay,
  rollback of invalid optimizer updates and complete checkpoint/resume.
- `docs/reference/neutra-fab.md`: pinned source correspondence, licensing,
  numerical and algorithmic departures, precision and downstream boundaries.
- `docs/benchmarks/run_q20_fab_2026_09_25.py`: bounded GPU worker, memory-growth
  verification, initial/final 1,000-point probes and beta-1 artifact binding.
- `docs/benchmarks/diagnose_q20_fab_2026_09_25.py`: prepared independent
  before/after posterior-weight and region diagnostic; not yet run on a
  trained q20 map because none exists.
- `docs/chapters/ch26d_modern_importance_sampling.tex`: foundations and
  dimensional failure, weighted kernels and mixtures, defensive/adaptive IS,
  AIS/SMC, AFT/CRAFT, FAB derivation/replay/tails, IAF inversion and frozen
  NeuTra, Stein methods, annealing flows and controlled diffusions.
- `docs/plans/bayesfilter-fab-iaf-audit-2026-09-25.md`: mathematical,
  source-code, implementation and rendered-document audit dispositions.

The complete monograph builds from both checkouts. The shared build has 596
pages; the canonical branch build has 595 because of unrelated manuscript
differences. The new chapter has 12 pages in both. It was visually inspected,
including repaired running headers and final pagination. All references and
citations resolve, and the new chapter has no layout warnings in the final
build. Existing warnings elsewhere in the monograph remain.

Preserved PDFs are in
`docs/plans/artifacts/neutra-fab-2026-09-25/modern-importance-sampling-chapter.pdf`
and `BayesFilter-with-modern-importance-sampling.pdf`. The separate chapter
extract has 12 readable pages; the full monograph retains its bibliography and
cross-chapter context. Human prose review remains pending.

## Audit and engineering findings

MathDevMCP's seven bounded algebra checks returned `equivalent`. Its
whole-document audit crashed with `KeyError: 'evidence_refs'`; its supported
source-bound audits of the FAB gradient, replay and AIS expectation completed
as **unverified**, with no mismatch. The detailed audit note supplies the
manual derivations and explicit assumptions. These outcomes are not formal
verification of integrability or differentiation under the integral for q20.

Two implementation defects were found and repaired before any q20 optimizer
updates:

1. The author initializes replay with `floor(minimum/batch)+1` passes. A
   configured 40-batch minimum therefore means 41 fill passes. The first port
   started one batch early. The owned workers were interrupted while still
   filling; their valid prefixes were preserved and the boundary was repaired.
2. TensorFlow's `StatelessShuffle` had no XLA_GPU kernel. Initial compiled
   tests covered AIS and updates separately but missed replay selection.
   The repair sorts independent FP64 random keys, preserving the permutation
   operation for distinct keys, and adds a full compiled replay regression.
   A complete GPU/XLA smoke subsequently passed, including two optimizer
   updates and checkpoint round-trip on a two-dimensional normal reference.

The beta-1 target-adapter signature was also corrected before training/export
so the existing frozen-map consumer can load the eventual map. Calibration
checkpoints 01–03 used the underlying target signature; they have zero updates
and are not map handoffs.

Focused tests cover bridge and replay identities, detached gradients, inverse
parameter finite differences, matching and shifted Gaussian AIS laws,
without-replacement sampling, correction capping versus uncapped priority
refresh, invalid-target rejection, exact JSON checkpoint continuation, XLA
update parity, and both mutations. All 16 tests pass in 72.01 seconds. The final test log is preserved as
`artifacts/neutra-fab-2026-09-25/focused-tests-final.txt`. The GPU smoke is
`replay-gpu-smoke-r1/result.json`. CPU tests intentionally hide GPUs and make
no q20 training-quality claim.

## Executed calibration

All rows below use 32 particles, preserve the same posterior, and contain
zero optimizer updates. Their metrics are descriptive, not method rankings.

| Attempt | Mutation / interior temperatures | Steady pass cost | Observed weight ESS | Interpretation |
|---|---|---:|---:|---|
| calibration-01 | HMC, 10, five leapfrog steps, epsilon .01 | About 123 s | About 1–1.92 of 32 | Near-unit acceptance did not resolve concentrated weights |
| calibration-02 | HMC, 10, five leapfrog steps, epsilon .1 | About 123 s | About 1–1.02 of 32 | Larger movement remained insufficient in these batches |
| calibration-03 | HMC, 32, five leapfrog steps, epsilon .07 | 409.58 s, one pass | 2.07 of 32 | Numerically valid but expensive; one batch cannot establish coverage improvement |
| metropolis-calibration-01 | Metropolis, 10, one mutation, epsilon .3 | 26.52 s | 1–1.01 of 32 | Replay initialization became affordable; weight concentration persisted |

No invalid proposals were reported during these calibrations or the completed
q20 initialization prefixes. This excludes an observed numerical failure in
those evaluations; it does not prove global target validity or finite `p²/q`.
The initial radial checks are explanatory, not a tail-integrability proof.

## Saved state and compute

| Seed | Last complete checkpoint | Initialization passes | Replay rows | Optimizer updates | Initial 1,000-point residual median |
|---|---|---:|---:|---:|---:|
| 0 | `replay-seed-0-r2/checkpoint.json` | 40 | 1,280 | 0 | 456.187 |
| 1 | `replay-seed-1-r2/checkpoint.json` | 41 | 1,312 | 0 | 424.053 |
| 2 | `replay-seed-2-r2/checkpoint.json` | 41 | 1,312 | 0 | 412.879 |

All initial probes completed with 1,000 finite, valid rows. These maps use
fresh prior-centered/scaled initialization, so the large residuals describe
the untrained baseline. They must not be compared as if they were final FAB
training results or the earlier reverse-KL maps.

The checkpoint after the last complete pass is the continuation authority.
Do not use a failure checkpoint for these retries: it can contain step-size
adaptation from a partially completed outer pass. Exact checkpoint hashes,
commands, source hashes, environment, devices, memory policy and seeds are
preserved in each manifest and `prepared-continuation.json`.

Charged GPU worker time is **5,997.991 seconds (1.666 hours)**. It includes all
four calibrations, all interrupted/failed q20 workers and the GPU replay smoke.
**8,402.009 seconds (2.334 hours)** remain in the four-hour FAB allocation.
The larger inherited campaign ledger has 17,039.168 seconds remaining after
these costs, but does not authorize exceeding the FAB cap or the wall deadline.
Waiting for tool approval and CPU document work are not GPU worker charges.
CPU mechanics checks remain within their separate 600-second allowance.

Conditional continuation uses three workers bounded to 2,300 seconds internally
and 2,350 externally, with 80/79/79 additional-pass ceilings. Three independent
posterior-weight diagnostics are each capped at 350 seconds. The worst-case
8,100 additional seconds fit in the 8,402.009 remaining. The initial probe can
be reused only after exact map/configuration/target/seed matching; final probes
remain required. The prepared continuation requires the pending deadline
extension before launch.

## Decision and inference status

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain repaired FAB implementation | Focused mechanics and complete GPU replay smoke pass | Known initialization and compilation failures repaired | q20 replay optimizer path and full trained-map diagnostics still unexecuted | Resume saved prefixes after deadline extension | Correctly trained NeuTra or posterior readiness |
| Pause campaign at wall deadline | Budget remains but calendar authorization expired | Deadline is the continuation veto | Owner's desired extension | Preserve prepared commands and checkpoints | Failure of FAB as a scientific idea |
| Retain compiled chapter | Sources, derivations, bibliography and rendering checked | MathDevMCP higher-level certification unavailable | General integral assumptions and human prose review | Use audit note in ordinary manuscript review | A machine proof of the complete chapter |

| Inference status | Finding |
|---|---|
| Hard veto screen | GPU replay compilation failed before q20 updates; repair passed a full reference GPU smoke. Calendar deadline currently stops campaign continuation |
| Statistically supported ranking | None |
| Descriptive-only differences | Calibration ESS, acceptance, movement, runtime and untrained-map residuals |
| Default readiness | Not established; no q20 FAB optimizer update completed |
| Next evidence needed | Actual replay updates, mandatory final 1,000-point probes, independent posterior-weight/region diagnostics; later downstream HMC checks for posterior claims |

The failure so far is an implementation/integration failure followed by a wall
deadline, not evidence against the target, the canonical IAF or FAB. The strongest
remaining scientific alternative explanation is poor finite-sample exploration
or a nonintegrable auxiliary tail even after the code runs. A finite loss or
balanced unweighted sign count cannot rule those out. Continued bounded training
and the independent diagnostics remain justified; promotion does not.
