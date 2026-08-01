# Phase 1 Specification: NeuTra Batch-Native Knowledge Transfer

Date: 2026-07-14

## Scope And Verdict

The prior DSGE work contains two useful but different target-evaluation
topologies:

1. a compiled leading-batch TensorFlow value/score route with custom-gradient
   score injection; and
2. a persistent CPU process pool that shards each optimizer step's current
   target batch.

The first is the selected mechanism for BayesFilter's exact LGSSM GPU/XLA
route. The second is an alternative repair/comparator topology. Neither means
that reverse-KL NeuTra uses a pre-generated training dataset: both generate
stateless base noise for the current optimizer step, transform it, and evaluate
the resulting parameter batch.

## Source Identity

| Source | SHA-256 |
| --- | --- |
| `~/python/src/dsge_hmc/experiment_adapters/ssm_equivalence.py` | `1cf029d599d86dce9dfb1613804373ed8c2857c3cf986ec2d9398cf9b9aa2a78` |
| `~/python/scripts/train_nk_svd_ukf_neutra_phase2_canary.py` | `659b1e6f144f49b5330ecf0c47c43b2a760673a0a0c560198e8fc27c549f49da` |
| `~/python/scripts/run_neutra_paper_style_at_baseline.py` | `0c15613ff2d4ea48f86995963e1ca5f5b1ba63409c47a788fce2ca30dbb64316` |
| SGU serious launch summary | `d2a98758120c58926c3379ecd6229d718751dc7af9daa9be103b7a8a6304d509` |
| Rotemberg serious launch summary | `4be482d632dfb1b7b70fce5330560b4afde3d5a936f88ba86af429cfce95b108` |
| `bayesfilter/linear/experimental_batched_kalman_tf.py` | `2297525950cb56e2a3967df042f0bba9bcb5f87f220f837d1b4055aa38de2ec9` |
| `bayesfilter/testing/multidim_triangular_lgssm_tf.py` | `92951662ef8f211b4e99500855fa594fce28f02c85a21b5964ca796df3e767c9` |
| `bayesfilter/linear/kalman_svd_derivatives_tf.py` | `ba0e77672221409e01c1fb5fbc4139f8b661e61f3cf9ddc01e628a0d05ecfcad` |

Hashes identify the inspected dirty-worktree snapshots; they are evidence
anchors, not claims that the external repository is clean or immutable.

## Mechanism Ledger

| Mechanism | Exact anchor | Classification | Local transfer |
| --- | --- | --- | --- |
| Direct rank-2 batch adapter contract | `ssm_equivalence.py:655` | `transfer_reusable` | exact adapter method accepts `[B,18]`, validates rank/width, and returns `[B]`, `[B,18]`, status |
| Model-owned batched analytical prior | `ssm_equivalence.py:582` | `transfer_reusable` only when direct batch helper exists | implement the LGSSM Gaussian prior with axis reductions |
| Scalar prior mapped with `tf.map_fn` | `ssm_equivalence.py:595` | `rejected_policy_incompatible` | no fallback; missing batch prior is a construction error |
| Compiled fixed-shape target boundary | `train_nk_svd_ukf_neutra_phase2_canary.py:303` | `transfer_reusable` | compile the actual batch callable with XLA and tensor-only outputs |
| Reviewed score attached with `tf.custom_gradient` | `train_nk_svd_ukf_neutra_phase2_canary.py:568` | `transfer_reusable` | already implemented generically in `neutra_batching.py`; certify same-call value/score |
| Python optimizer-step loop and NumPy diagnostic arrays | `train_nk_svd_ukf_neutra_phase2_canary.py:550` | `rejected_policy_incompatible` for active training | retain only as historical design evidence; current trainer uses a TensorFlow loop |
| Leading batch-axis Kalman state and score | `experimental_batched_kalman_tf.py:294` | `transfer_reusable` | use `[B,...]` model state and `[B,P,...]` sensitivities with one time loop |
| Cholesky factor/solve law | `experimental_batched_kalman_tf.py:461` | `target_specific_hypothesis` and comparator only | do not substitute for exact SVD/eigh graph-status math |
| Scalar SVD/eigh solve/logdet and score | `kalman_svd_derivatives_tf.py:271` | `transfer_reusable` as mathematical authority | lift operations over leading batch axes without changing formulas |
| Scalar graph-status accumulation | `kalman_svd_derivatives_tf.py:400` and `kalman_svd_derivatives_tf.py:489` | `transfer_reusable` | accumulate per-row invalid input, floor count, minimum eigenvalue, and maximum condition |
| One scalar time `tf.while_loop` | `kalman_svd_derivatives_tf.py:535` | `transfer_reusable` | one batched time loop; no sample loop |
| Persistent CPU worker initialization and shard pool | `run_neutra_paper_style_at_baseline.py:1252` and `run_neutra_paper_style_at_baseline.py:1374` | `alternative_topology` | possible Phase 6 repair comparator, not selected default |
| Scalar row loop inside old CPU worker | `run_neutra_paper_style_at_baseline.py:1315` | `rejected_policy_incompatible` | only the batched-worker variant is relevant even as an alternative |
| Parent host bridge using `.numpy()` | `run_neutra_paper_style_at_baseline.py:5572` | `alternative_topology`, not graph-native | cannot satisfy selected GPU/XLA batch-target admission |
| SGU `B=480`, 96 workers, five rows/worker | SGU launch summary lines 284-350 | `historical_evidence_only` | configuration proves a designed topology, not measured speed or correctness |
| Rotemberg `B=480`, 96 workers, five rows/worker | Rotemberg launch summary lines 261-327 | `historical_evidence_only` | same limitation; no LGSSM default transfer |

## Selected LGSSM Tensor Contract

Let `B` be the proposal batch, `P=18`, state dimension `N=4`, observation
dimension `M=4`, and horizon `T=120`.

| Quantity | Shape |
| --- | --- |
| raw parameters | `[B,P]` |
| initial/transition/observation offsets | `[B,N]`, `[B,N]`, `[B,M]` |
| initial, transition, observation covariances | `[B,N,N]`, `[B,N,N]`, `[B,M,M]` |
| transition and observation matrices | `[B,N,N]`, `[B,M,N]` |
| vector derivatives | `[B,P,N]` or `[B,P,M]` |
| matrix derivatives | `[B,P,N,N]`, `[B,P,M,N]`, or `[B,P,M,M]` |
| likelihood/posterior value | `[B]` |
| likelihood/posterior score | `[B,P]` |
| every status field | `[B]` |

The `T` axis is not materialized into model tensors because this fixture is
time-invariant. Observations remain shared `[T,M]`.

## Stationary Covariance Transfer

The scalar implementation solves

`(I - A kron A) vec(P) = vec(Q)`.

The batch implementation uses `[B,16,16]` systems and `[B,16,1]` right-hand
sides. Its sensitivities satisfy

`L(A) vec(dP_p) = vec(dA_p P A' + A P dA_p' + dQ_p)`.

All `P=18` right-hand sides share `L(A)` for a row, so they are passed to one
batched `tf.linalg.solve` as `[B,16,P]`. This is a target-local optimization,
not a mathematical change. Scalar materialization parity, Lyapunov residuals,
and derivative parity are hard gates.

## Training And Sample Semantics

| Object | Generation/evaluation | Policy role |
| --- | --- | --- |
| reverse-KL base noise `z` | stateless TensorFlow RNG inside compiled optimizer program | selected training route |
| transformed target batch `theta=T_phi(z)` | GPU flow forward pass | selected training route |
| exact target value/score/status | one GPU/XLA batch call | selected target route |
| CPU-worker target shards | host transfer plus persistent CPU TensorFlow workers | alternative Phase 6 topology |
| offline replay/training dataset | not used by current reverse-KL objective | new contract if introduced |
| HMC posterior samples | generated after training and tuning | downstream scientific validation, not training data |

## Defaults Not Transferred

The following remain hypotheses or historical context, not LGSSM defaults:

- batch `480`, worker count `96`, and five rows per worker;
- NK canary batch `128`, optimizer, learning rate, flow width, and step count;
- Cholesky/QR/sigma-point target mathematics;
- CPU host bridges and NumPy orchestration;
- DSGE timing, launch, convergence, or transport-quality claims.

## Phase 2 Handoff

Phase 2 will implement only materialization, analytical prior, stationary
covariance, and first derivatives. It will not implement the batch SVD/eigh
time recursion or bind the exact adapter. Its pass gate is row-wise scalar
parity plus XLA/no-loop/no-NumPy evidence.

