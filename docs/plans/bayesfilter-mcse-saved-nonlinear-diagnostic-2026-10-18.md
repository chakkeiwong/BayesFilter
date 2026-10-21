# Saved nonlinear-HMC precision diagnostic

18 October 2026. Status: complete; independent terminal audit passed.
This is a CPU-reference diagnostic within N5 of the modern-HMC master.
It generates no new HMC transitions and changes no numerical default or prior
member disposition. The concurrent SIR and stationary-interval runs continue.

## Question and evidence contract

How sensitive are the reported MCSEs of the terminal NL-PP-SGQF cohorts to
the six already-declared variance estimators and batch schedules? The exact
baseline is each member's original saved retained-window lugsail report.
Replay every available original retained look on the same saved quantities
under the four existing policies and the two twice-square-root policies from
`bayesfilter-mcse-dependence-interval-study-2026-10-18.md`. Use the same reviewed
immutable precision overlay. Do not simulate counterfactual stopping after
the original endpoint, pool candidates as independent replications, or use a
more favorable estimate to change delivery, reference or tuning decisions.

There is no estimator-promotion or method-ranking criterion. Engineering
success requires a complete inventory of the four terminal cohorts and their
member rows, preserved original dispositions, verified source/input hashes,
draw/chain/quantity counts and retained-only identity, reproduced original
lugsail estimates within a declared floating-point comparison, and explicit
missing/unavailable data. A source, shape, quantity, policy or integrity mismatch
vetoes interpretation. A negative or unavailable variance is a diagnostic
finding, not a harness failure. Estimator differences, reference errors and
MCSE/SD ratios are explanatory only. Passing replay cannot establish MCSE
coverage, convergence, burn-in sufficiency, a superior method or release
readiness: these are adaptively selected, single-seed, already-stopped chains.

The artifact preserves a row for every member, including failed, incomplete
or unavailable outcomes. Fixed HMC, randomized HMC, SNAPER-proposed randomized
HMC and fully refreshed MAMS are separate cohorts. The finite-filter reference
and its original verdict may be copied as context; it is not an exact posterior
oracle and no new confidence-coverage verdict will be constructed from it.

## Intent, assumptions and early falsification

| Item | Role or provenance | Failure and early check |
| --- | --- | --- |
| Main question | Sensitivity of saved nonlinear estimates to variance estimation | Do not substitute synthetic stationary coverage for this question |
| Original saved lugsail | Exact baseline, with original policy and looked-at sample counts | Replay estimate, MCSE, validity and batch metadata; reject contradictory records |
| Six estimator arms | Frozen diagnostic hypotheses from the interval study, not selected from its results | Check names and exact `min(isqrt(4*n), n//20)` schedule |
| Original retained looks only | Inherited observed endpoints | Require matching saved check counts; never manufacture post-stop draws |
| Archived transformed quantities | Produced by the original shared posterior controller | Match names, dtype, shape and retained count to result and checkpoint identity; exclude warm-up |
| Every original member | Complete denominator, not an independent replication sample | Include original failures and explicit missing reasons; never average across candidates as if independent |
| Source overlay | Previously reviewed numerical authority | Validate `.validation-source.json`, precision SHA and all imported numerical modules |
| Replay tolerance | Engineering hypothesis: relative `1e-10`, absolute `1e-12` for CPU/XLA roundoff | Tiny known and corrupted fixtures; report largest observed discrepancy; never use as a posterior tolerance |
| CPU-only, two TF/OMP threads, XLA on | Reference exception, keeps GPU free | Environment set before import; fail if GPUs are exposed |
| 1,800 CPU-worker seconds total | Convenience cap, not a performance or scientific default | Meter focused preflight, smoke, numerical replay and terminal inspection; stop incomplete at the cap |

The interval study holds 7,200 seconds, leaving `127651.56848495119` seconds
unallocated from the released SGQF balance. Reserve 1,800 seconds separately
for this diagnostic, leaving `125851.56848495119` seconds while both holds are
active. This is no new grant and does not change SIR caps. Do not include the
separate 3,600-second hold or historical 1,579.592954-second balance. Write an
allocation before metered commands, charge inclusive work once, and return
unused funds to the same balance. No automatic second numerical attempt.

## Execution

Use a fresh root
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N5/generic-adaptation/saved-nonlinear-precision-20261018-01/`.
The explicit input design lists the four existing terminal cohort result files
and records their hashes before execution. Do not scan for favorable members
or read a changing SIR cohort as if terminal. The terminal SIR cohort can be
added later through a new design and versioned output.

Implement a diagnostic reader using the public precision calculation, not a
new numerical estimator. Validate retained tensor counts, names, original
policy and terminal state before assessment. Check committed retained chunks
or an already-audited archive binding so warm-up or changed raw tensors cannot
silently replace the assessed window. Preserve per-chain batch metadata and
input/source hashes. Keep source and raw data read-only. Write a small manifest,
one member record at a time, a progress index and a terminal result; archive no
raw duplicates.

The pre-execution investigation found all 26 members have retained tensors,
with 159 original retained looks in total (fixed 50, randomized 46, SNAPER 38,
MAMS 25). All have 18 named quantities. These measured counts are the frozen
inventory, not a new sample-size choice. Check the original terminal manifests'
source/input hashes. Read committed bundles without opening a checkpoint writer;
verify each bundle checksum, identity/input digests, tensor receipt, original
policy, active count and recorded seed. Partition chronological chunks using
the original warm-up count, require that the boundary falls between chunks,
and reconstruct retained quantities using the original frozen
`model_quantities` function. Compare this reconstruction with
`retained_raw.tensor` before replay. Its function's file hash must match the
original run's source receipts. This avoids relying on uneven historical
archive hashing across methods, and detects warm-up substitution. Never
instantiate `DurableTensorCheckpoint` for a read-only audit.

An independent terminal reader will recompute means, SDs, batch components,
validity and MCSEs for all five batch-based arms using diagnostic NumPy, plus
inventory, input hashes and original-disposition checks. Autocorrelation uses
the public TensorFlow authority; its numerical reimplementation is outside
this reader's scope and must be reported as such. Original lugsail replay is
checked against every saved report, including per-chain batch components.

Focused tests cover warm-up/retained confusion, wrong shape/count/names,
duplicate candidates, stale hashes, omitted failed members, absent retained
draws, constant/antithetic quantities, exact experimental batch sizes, replay
mismatch and capped partial denominators. Follow with one original-member
smoke, then one bounded complete inventory replay if the preflight passes.
Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, CPU hidden before framework
import, `TF_FORCE_GPU_ALLOW_GROWTH=true`, all TF/OMP thread counts two, XLA on.
Freeze the tested driver and helper before the numerical attempt. Record exact
commands, source, environment, original seeds, wall time and artifacts in the
manifest. No new random sampling seed is needed for deterministic replay.

At closeout independently check the complete design/member inventory, source
and input hashes, replay discrepancy and unchanged original verdicts. Write
a result note with decision and inference-status tables. A replay mismatch
triggers diagnosis of saved-data/source/precision identity, not selective
retuning. Stop at integrity failure or the inclusive allocation; retain every
partial outcome. Refresh the master and next phase from the findings.

## Skeptical pre-execution audit

The comparator is the actual original estimator on its actual assessed saved
window. The principal risk is that saved `retained_raw.tensor` already contains
transformed named quantities, while checkpoint tensors contain source states;
reapplying the transform would be wrong. Require explicit names and reproduce
the original report before interpreting alternatives. Another risk is survivor
bias: all original members remain in the inventory, and no difference is
called calibration or statistical superiority. The six policies are frozen
before the concurrent interval study finishes. No reference error, availability
rate or descriptive ratio promotes a default. This plan passes the skeptical
audit subject to the stated source, retained-data and smoke checks; the
read-only committed-chunk reconstruction resolves the archive-binding risk.
No running numerical source, candidate membership or posterior policy is edited.

The implementation audit found and repaired two completion-reporting defects
before preflight: a capped smoke could claim success without replaying a member,
and the auditor could accept a complete label with missing records. The focused
suite now exercises both capped CLI modes and rejects false completion. It also
rejects null validity flags rather than coercing them to false. These are
diagnostic integrity repairs, not changes to a variance estimator.

`allocation-01.json` holds the declared 1,800 seconds. The bounded command meter
records each actual command, CPU environment, log, exit and elapsed time without
overwriting prior receipts. Initial convenience watchdogs are 240 seconds for
focused tests, 60 for inventory preparation, 180 for the original-member smoke,
120 for each audit, and 960 for the full replay (900-second cooperative cap).
These bounds are engineering stop limits, not scientific defaults or predicted
durations. Check measured preflight costs and remaining allocation before the
full launch; reserve terminal-audit time and reduce its cap if necessary.

The next preflight exposed a wrong archive-schema assumption: original run
manifests store a structured target identity, while their bound design stores
the model name. The repaired reader binds both to the original design checksum,
geometry checksum, six-dimensional 20-observation model and data identity.
Realistic fixtures now exercise that schema and reject mismatched model,
dimension, horizon, geometry and missing design bindings. The skeptical audit
permits this localized reader repair: it changes neither the saved inputs nor
the original numerical computation, and all failed preflight time remains
charged. Freeze and test the repair before retrying inventory preparation.
Continuation manifests use a second inspected schema: they bind the prior
manifest through `continuation_inputs_sha256` and retain the same frozen design.
The reader checks that parent checksum and exact design agreement before
validating the original structured target. It rejects unbound, changed or
multi-hop parents rather than inferring identity from directory names.

The first original-member smoke failed its reconstruction check before any
look was replayed. The traced original call chain is
`assess_full_model_members -> run_hmc_posterior -> HMCCandidateRetainedRunner.run_sequential
-> binding.position_samples -> LatentAffineHMCTransform.latent_to_position`.
Committed chunks contain active whitened states `z`; named quantities use
`q = center + z @ factor.T`. The saved-data reader incorrectly treated `z`
as `q`. Repair both diagnostic reconstructions using the checksum-bound saved
geometry, retain the original replay tolerance, and test a shifted, nonsymmetric
factor so omission and transposition fail. A fresh design must bind the explicit
geometry descriptor; preserve design-01 and smoke-01 as failed preflight evidence.
The skeptical audit passes this localized repair: it restores the original
computation, changes no sampler or decision, and all attempts remain charged
against the same 1,800-second allocation. No scientific replay has launched.

The repaired frozen `source-09` passed all 60 focused tests. Fresh `design-02`
binds all 1,157 inputs and each member's explicit geometry descriptor. The
original-member `smoke-02` replayed ten looks in 11.742579 inclusive seconds;
maximum reconstruction and original-report absolute differences were
`1.4210854715202004e-14` and `3.3084646133829665e-13`. Its independent audit
passed 50 batch-arm checks and source/input/disposition checks in 2.772380
inclusive seconds. The declared tolerances are unchanged. A separate tiny-cap
preflight tests the complete denominator without replaying a member; its
terminal audit must pass before the full launch.

The continuation audit retains the original 900-second cooperative / 960-second
watchdog full-replay bounds, with at least 120 seconds reserved for independent
terminal audit. Measured preflight costs leave over 1,600 seconds before launch.
These are conservative engineering caps, not predicted runtimes. The full
attempt remains one deterministic replay of the frozen 26-member, 159-look
inventory. No result from the live interval study enters its six frozen arms.

The single complete attempt has now replayed all 26 members and 159 original
looks. The independent terminal audit passed all 795 batch-arm checks and all
retained reconstruction/input/disposition checks. The
[terminal result](bayesfilter-mcse-saved-nonlinear-result-2026-10-18.md) records
the complete denominator, diagnostic findings, limits and settlement. Total
inclusive charge is `131.3668131614104` seconds; `1668.6331868385896` seconds
are released from this reservation. No new sampler, estimator/default change,
counterfactual stopping decision or additional scientific attempt is needed.
