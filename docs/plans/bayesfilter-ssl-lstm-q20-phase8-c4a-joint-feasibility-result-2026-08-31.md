# Phase 8 C4A K=4 joint mixture-RKL feasibility result

Date: 2026-08-31  
Status: `PASS_C4A_JOINT_FEASIBILITY_NO_PROMOTION`

Subplan:
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4a-joint-feasibility-subplan-2026-08-31.md`

## Result

The bounded GPU/XLA pilot completed one q=20 `K=4`, `B=32`, beta-0.5 row in
454.691 seconds. TensorFlow float64 used the strict
`tensorflow_eigh_strict` backend with one visible RTX 4080 SUPER GPU and
memory growth configured before logical-device creation. The largest allocator
peak was 3,401,816,064 bytes, below the 4-GiB cap. All finite/status,
checkpoint-replay, learned-map reliability, alpha-normalization, route-scan,
and exact-work checks passed.

Manifest:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c4a-joint-feasibility/attempt-01/run_manifest.json`

The pilot used independent and joint copies restored from the same four
beta-0.5 start checkpoints. Each independent chart received eight updates;
the joint trainer received eight updates and trained the mixture logits. Every
joint update reported one target call over 128 flattened rows and 512
cross-density work units. The 16-update joint resource forecast was 117.723
seconds, below the 3,600-second envelope.

## Evidence

| Arm | Held-out mixture objective | Target calls per update | Cross-density work per update | Alpha / entropy | Exact duplicate states |
|---|---:|---:|---:|---|---:|
| Independent (four charts) | 169.3460 | 4 aggregate | 0 for training; 2,048 for held-out bank | `(0.25,0.25,0.25,0.25)` / 1.386294 | 0 |
| Joint mixture-RKL | 155.1721 | 1 over `K B` rows | 512 | `(0.25025,0.24969,0.24998,0.25008)` / 1.386294 | 0 |

The joint pilot's held-out objective is 14.174 lower than the independent
copy's objective on this one fresh bank. This is a descriptive, eight-update
within-root contrast. It is not a statistical comparison and does not show
that joint training improves the posterior approximation. Pullback score RMS
remained large for both arms, so the run supplies no whitening evidence.

## Decision

| Decision | Primary criterion | Hard-veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| C4A implementation | Correct joint objective, work accounting, finite maps, replay, reliability, and resource bounds | Pass | One root and short pilot | Run a fresh-root C4B replication before any freeze | No production joint arm |
| Joint resource envelope | 16-update forecast <= 3,600 s and peak < 4 GiB | Pass (117.723 s; 3.40 GiB peak) | Peak is close enough to require monitoring at larger banks | Keep K=4 as a bounded candidate only | No high-dimensional cost claim |
| Joint objective contrast | Held-out joint objective versus matched independent copies | Descriptive only | One bank, eight updates, no uncertainty interval | Replicate with a fresh root/architecture | No superiority or ranking |
| Alpha behavior | Finite positive normalized logits without exact collapse | Pass; alpha remains nearly uniform | Alpha is not a regional-mass estimator | Report entropy in the replication | No posterior mode masses |
| Whitening | Pullback score residuals | Gate remains closed; residuals large | Capacity/objective may be inadequate | Preserve whitening/HMC veto | No IID-Gaussian claim |

## Inference status

| Evidence class | Result |
|---|---|
| Hard veto screen | Pass for this bounded feasibility row; no target, bridge, graph, checkpoint, reliability, alpha, duplicate-state, GPU, or allocator veto fired. |
| Statistically supported ranking | None. One root and eight updates provide no uncertainty-supported ranking. |
| Descriptive-only differences | Joint versus independent held-out loss, update time, alpha entropy, and pullback residuals. The joint objective is lower on the recorded bank, but this is descriptive. |
| Default readiness | Not established. No target-specific joint tuning, multi-root replication, retained chains, ESS/R-hat, downstream agreement, or posterior validation exists. |
| Next evidence needed | Fresh-root/architecture C4B replication with the same exact work and memory accounting, followed by a C5 freeze decision. |

## Post-run red-team

The strongest alternative explanation for the lower joint held-out objective is
that the two objectives are evaluated after different stochastic trajectories,
and eight updates are too short to compare optimization quality. The alpha
logits barely moved, so the pilot did not test meaningful weight adaptation.
The 3.40-GiB peak also leaves less headroom than the short wall time suggests;
larger validation banks or more components may breach the cap. The flattened
target call is batch-native, but the independent comparator's four calls and
the joint call's one call are not equal-cost wall-time paths.

An implementation conclusion would be overturned by a fresh-root replay/hash
mismatch, a wrong `[K,K,B]` tensor or target-call count, a nonfinite score or
map, exact chart collapse, or an allocator/memory-growth violation. None
occurred here. The weakest evidence is the held-out loss difference and the
short alpha trajectory.

## Provenance and nonclaims

The manifest includes the C3 helper dependency, route modules, runner,
subplan, C3B manifest, and C3B provenance receipt in its source hashes. The
run used no posterior draws, target-derived replay bank, particle replacement,
or state-dependent chart selection.

This result does not establish whitening, IID Gaussian pullback, mode
discovery, exhaustive coverage, posterior regional masses, convergence, HMC
readiness, statistical superiority, architecture superiority, or
high-dimensional scaling.
