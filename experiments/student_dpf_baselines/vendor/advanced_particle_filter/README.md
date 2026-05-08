# Advanced Particle Filter Library

State-space filtering and Bayesian inference research code, organized in three threads:

1. **Classical filters and particle flow filters** — implementations of KF/EKF/UKF, bootstrap PF, and recent particle flow methods (EDH, LEDH, kernel PFF, stochastic PFF). Replication and exploration of the literature.
2. **HMC with differentiable resampling** — Hamiltonian Monte Carlo on parameters of state-space models, where the marginal likelihood is provided by a differentiable particle filter (DPF). Soft and Sinkhorn-OT resamplers, applied to SVSSM calibration.
3. **Neural acceleration of OT resampling** — a learned operator (Set Transformer + attention coupling head) that amortizes the entropy-regularized OT resampling step from Corenflos et al. (2021), trained offline and used as a drop-in replacement inside the DPF.

This is a work-in-progress research codebase. The classical thread is the most polished; the neural-amortization thread is the most recent and active.

## Where to look first

For an overview of the recent neural-amortization work:

1. `docs/amortized_ot_operator.md` — one-page architecture summary.
2. `notebooks/03_neural_amortized_ot/02_train_coupling_operator.ipynb` — operator training and held-out evaluation.
3. `notebooks/03_neural_amortized_ot/03_hmc_amortized_comparison.ipynb` — three-resampler HMC comparison on SVSSM (soft, sinkhorn, amortized).

For HMC+DPF without amortization:

4. `notebooks/02_hmc_differentiable_resampling/01_diagnostics_hmc_dpf_linear.ipynb` — gradient quality diagnostics on a linear-Gaussian-observation variant of the SVSSM (Kalman gives ground truth; isolates DPF bias).
5. `notebooks/02_hmc_differentiable_resampling/02_hmc_dpf_svssm.ipynb` — soft vs OT comparison on the full SVSSM.

For the classical-filtering thread (replication of literature):

6. `notebooks/01_classical_and_differentiable_pf/03_Li2017_acoustic.ipynb` — Li & Coates (2017) acoustic tracking.
7. `notebooks/01_classical_and_differentiable_pf/04_pff_comparison.ipynb` — EDH vs LEDH vs kernel PFF.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── docs/
│   └── amortized_ot_operator.md     One-page architecture note for thread 3.
│
├── notebooks/                       All exploratory and experimental notebooks.
│   ├── README.md                    Per-notebook runtime + topic index.
│   ├── 01_classical_and_differentiable_pf/
│   │   ├── 01_lgssm.ipynb           Kalman on linear-Gaussian SSM.
│   │   ├── 02_nonlinear_ssm.ipynb   EKF/UKF/BPF on range-bearing.
│   │   ├── 03_Li2017_acoustic.ipynb Li & Coates (2017) replication.
│   │   ├── 04_pff_comparison.ipynb  EDH vs LEDH vs kernel PFF.
│   │   ├── 05_kernel_pff.ipynb      Scalar vs matrix kernel; collapse analysis.
│   │   ├── 06_dai2022_seed_sensitivity.ipynb   Dai & Daum (2022) Ex. 1.
│   │   ├── 07_differentiable_pf.ipynb          Soft & OT resampling intro [Colab/GPU].
│   │   ├── 08_sinkhorn_parameter_sweep.ipynb   ε and num_iters sweep [Colab/GPU].
│   │   └── 09_three_model_dpf.ipynb            DPF with normalizing flows [Colab/GPU].
│   │
│   ├── 02_hmc_differentiable_resampling/
│   │   ├── 01_diagnostics_hmc_dpf_linear.ipynb DPF gradient quality on a
│   │   │                                       linear-Gaussian-observation
│   │   │                                       variant of the SVSSM
│   │   │                                       (Kalman gives ground truth).
│   │   ├── 02_hmc_dpf_svssm.ipynb              SVSSM HMC: soft vs Sinkhorn-OT.
│   │   └── 03_mle_laplace_pipeline.ipynb       MAP + Laplace + HMC pipeline.
│   │
│   └── 03_neural_amortized_ot/
│       ├── 01_brenier_gap_evidence.ipynb       Why the original M-MGN
│       │                                       architecture failed (Brenier
│       │                                       vs Kantorovich function class).
│       ├── 02_train_coupling_operator.ipynb    Train the CouplingOperator
│       │                                       on precomputed Sinkhorn targets.
│       └── 03_hmc_amortized_comparison.ipynb   SVSSM HMC: soft vs sinkhorn
│                                               vs amortized (3-resampler).
│
├── scripts/                         Standalone runner scripts for the
│   ├── acoustic_experiment.py       classical thread, used for replication
│   ├── benchmark_numpy_baseline.py  experiments and figure generation.
│   ├── filter_comparison.py
│   ├── dai2022_example1.py
│   ├── dai2022_example2.py
│   └── ledh_vs_spfpf.py
│
├── tests/                           Unit and integration tests.
│   ├── test_basic.py
│   ├── test_filters.py
│   ├── test_kernel_pff.py
│   └── test_tf_migration.py
│
├── archive/                         Stale or deprecated files kept for reference.
│
├── models/                          NumPy state-space model definitions.
│   ├── base.py                      StateSpaceModel dataclass.
│   ├── linear_gaussian.py
│   ├── range_bearing.py
│   ├── acoustic.py                  Multi-target acoustic tracking.
│   └── dai22_example2.py            Dai & Daum (2022) angle-only tracking.
│
├── filters/                         NumPy filtering algorithms.
│   ├── base.py                      FilterResult container.
│   ├── kalman.py                    KF, EKF, UKF.
│   ├── particle.py                  Bootstrap PF (SIR).
│   ├── edh.py                       EDH flow + EDH PF-PF.
│   ├── ledh.py                      LEDH flow + LEDH PF-PF.
│   ├── kernel_pff.py                Kernel PFF (scalar/matrix kernels).
│   ├── stochastic_pff.py            Stochastic PF flow (Dai & Daum 2022).
│   └── stochastic_pfpf.py           SPFF as PF-PF proposal.
│
├── simulation/                      Trajectory generators.
│   └── trajectory.py
│
├── utils/                           Resampling, metrics.
│   ├── resampling.py
│   └── metrics.py
│
├── tf_models/                       TensorFlow model variants.
│   ├── linear_gaussian.py           Generic LG factory.
│   ├── linear_gaussian_obs.py       SVSSM dynamics with linear-Gaussian obs.
│   ├── corenflos_lg.py              Corenflos et al. (2021) bivariate LG model.
│   ├── range_bearing.py             Range-bearing in TF.
│   └── svssm.py                     2-asset stochastic-volatility SSM.
│
├── tf_filters/                      TensorFlow filtering algorithms.
│   ├── base.py
│   ├── kalman.py
│   ├── particle.py                  TF bootstrap PF.
│   └── differentiable_particle.py   DPF with three resampler options:
│                                    soft, sinkhorn, amortized.
│
├── tf_utils/
│   ├── resampling.py                Standard resampling routines.
│   ├── soft_resampler.py            Straight-through soft resampler.
│   ├── sinkhorn.py                  Differentiable Sinkhorn-OT
│   │                                with implicit differentiation.
│   └── amortized_resampler.py       Adapter wrapping the trained
│                                    CouplingOperator into the DPF interface.
│
├── hmc/                             HMC entry points.
│   ├── parameterization.py                  SVSSM unconstrained-space mapping.
│   ├── parameterization_corenflos_lg.py     Corenflos LG parameterization.
│   ├── run_hmc_poc.py                       Main SVSSM HMC driver.
│   ├── run_hmc_corenflos_lg.py              Corenflos LG HMC driver
│   │                                        (Kalman + 3 DPF variants).
│   └── run_hmc_amortized_smoke.py           Amortized-only HMC entry.
│
├── mle/                             MAP + Laplace + HMC pipeline.
│   ├── adam_mle.py                  Multi-restart Adam MAP via DPF gradient.
│   ├── laplace.py                   Finite-difference Hessian at MAP.
│   ├── preconditioned_hmc.py        HMC with mass matrix from |H|.
│   └── ...
│
├── diagnostics/                     Likelihood-landscape and SNR diagnostics.
│   ├── batched_kalman.py            Batched Kalman log-likelihood for ground truth.
│   ├── kalman_ll.py
│   └── landscape.py
│
├── dpf_pretrained/                  Pretrained models, vendored architecture
│   ├── F_theta.weights.h5           code, and the operator-training subpackage.
│   ├── G_phi.weights.h5
│   └── mgn_ot_operator/             Self-contained operator-training subpackage.
│       ├── config.py                Default training config.
│       ├── models/                  Architecture: coupling-predictor (Option B)
│       │                            and the older M-MGN ablation.
│       ├── data/                    Cloud sampler, Sinkhorn target precompute,
│       │                            dataset loader.
│       ├── training/                Train loops and overfit/sanity scripts.
│       ├── evaluation/              Held-out evaluation.
│       └── checkpoints_option_b/    Trained operator weights (≈5 MB).
│           ├── best/ckpt-82.*       Best-val checkpoint, used at inference.
│           └── latest/ckpt-{85..87}.* Three rolling latest checkpoints.
│
├── pfresults/                       Experiment outputs (acoustic experiment).
│                                    Reproducible from scripts/.
│
└── fig/                             Figures referenced by notebooks.
```

## The operator-training subpackage at `dpf_pretrained/mgn_ot_operator/`

The neural operator's training pipeline (config, dataset, training loops,
evaluation) lives at `dpf_pretrained/mgn_ot_operator/`. It is a self-contained
subpackage — both notebooks 01 and 02 in topic 3 use it via a `sys.path`
injection at notebook setup time, so their bare `from data.X`,
`from training.X`, `from models.X` imports resolve to that subpackage rather
than to the top-level `models/`, `filters/`, etc. of `advanced_particle_filter`.

The pretrained weights at `dpf_pretrained/mgn_ot_operator/checkpoints_option_b/`
are loaded by the inference adapter at `tf_utils/amortized_resampler.py`,
which uses an absolute Python import
(`from advanced_particle_filter.dpf_pretrained.mgn_ot_operator.models.operator_b
import CouplingOperator`). The relative imports inside `models/` make both
load paths work without conflict.

If you only need to use the trained operator for filtering or HMC, you do
not need to touch this subpackage; just call `AmortizedOTResampler()` and
the bundled checkpoint is loaded for you.

## Requirements

**NumPy/SciPy core** (all of `models/`, `filters/`, `simulation/`, `utils/`,
notebooks 01-06 in topic 1, scripts in `scripts/`):

```
python >= 3.9
numpy, scipy, matplotlib
```

**TensorFlow side** (all of `tf_*/`, `hmc/`, `mle/`, `diagnostics/`,
notebooks 07-09 in topic 1, all of topics 2 and 3):

```
tensorflow >= 2.12
tensorflow-probability
```

GPU recommended for the TF side. Notebooks in topics 2 and 3 are intended for
Colab with GPU runtime; topic 1 notebooks 07-09 are Colab too.

See `requirements.txt` for the consolidated list.

## Runtime conventions

Two runtime contexts in this codebase:

- **Local CPU**: notebooks 01–06 of topic 1, scripts in `scripts/`, and
  tests in `tests/`. Only NumPy/SciPy needed.
- **Colab GPU**: notebooks 07–09 of topic 1, all of topic 2, and all of
  topic 3. Needs TF + TFP. We ran all GPU notebooks on **Google Colab
  with A100 40GB**. Notebooks 01 and 02 of topic 3 use the operator-
  training subpackage at `dpf_pretrained/mgn_ot_operator/` — their first
  cell injects that path into `sys.path` so the bare `from data.X` /
  `from training.X` imports resolve.

For Colab notebooks, the standard setup pattern is:

1. Runtime → Change runtime type → GPU.
2. Zip the entire `advanced_particle_filter/` folder and upload it via the
   notebook's first code cell (which calls `files.upload()`). It is then
   extracted at `/content/advanced_particle_filter/`. Subsequent runs
   reuse a cached copy from Drive.
3. The first cell installs `tensorflow-probability`.
4. **Drive paths used by topic 3**:
   - Notebook 02 (`02_train_coupling_operator`): expects
     `MyDrive/mgn_ot_operator/data/` for precomputed Sinkhorn `.npz`
     targets and `MyDrive/mgn_ot_operator/checkpoints_option_b/` for
     training checkpoints. If precomputed targets are absent, full
     precompute runs from scratch.
   - Notebook 03 (`03_hmc_amortized_comparison`): optional Drive mount.
     Loads the trained operator from either
     `MyDrive/mgn_ot_operator/checkpoints_option_b/best/` (if present)
     or the bundled copy at
     `dpf_pretrained/mgn_ot_operator/checkpoints_option_b/best/`.

See `notebooks/README.md` for a per-notebook runtime table.

## Quick start (CPU)

```python
from advanced_particle_filter.models import make_lgssm
from advanced_particle_filter.simulation import simulate
from advanced_particle_filter.filters import KalmanFilter

import numpy as np

# Constant-velocity tracking model
A = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], dtype=float)
C = np.array([[1,0,0,0],[0,1,0,0]], dtype=float)
Q = 0.1 * np.eye(4); R = 0.5 * np.eye(2)
m0 = np.zeros(4); P0 = np.eye(4)

model = make_lgssm(A, C, Q, R, m0, P0)
traj = simulate(model, T=100, seed=42)
result = KalmanFilter().filter(model, traj.observations)
print(f"Mean RMSE: {result.mean_rmse(traj.states):.4f}")
```

## Quick start (GPU, amortized OT resampler)

Construct the amortized OT resampler with the bundled checkpoint:

```python
from advanced_particle_filter.tf_utils.amortized_resampler import AmortizedOTResampler
import tensorflow as tf
import numpy as np

# No ckpt_dir argument → uses the vendored checkpoint at
# dpf_pretrained/mgn_ot_operator/checkpoints_option_b/.
resampler = AmortizedOTResampler(d=2, N=1000, eps=0.5)

# Mock weighted cloud
B, N, D = 4, 1000, 2
particles = tf.constant(np.random.randn(B, N, D))
log_w = tf.fill([B, N], -tf.math.log(tf.cast(N, tf.float64)))

resampled, new_log_w = resampler(particles, log_w)
print(resampled.shape, new_log_w.shape)
# (4, 1000, 2) (4, 1000)
```

For end-to-end use inside the differentiable particle filter:

```python
from advanced_particle_filter.tf_filters import TFDifferentiableParticleFilter

dpf = TFDifferentiableParticleFilter(
    n_particles=1000,
    resampler='amortized',
    amortized_d=2,            # state dim must match trained operator
    amortized_eps=0.5,
    # ckpt_dir defaults to the vendored copy
)
```

## Running the test suite

From the parent directory of this package:

```bash
# Integration tests (no pytest needed)
python advanced_particle_filter/tests/test_basic.py

# Unit tests (requires pytest)
pytest advanced_particle_filter/tests/ -v
```

## Design

All NumPy filters operate on a shared `StateSpaceModel` dataclass that bundles
dynamics, observation functions, Jacobians, noise covariances, and precomputed
Cholesky factors. Filters return a uniform `FilterResult` containing means,
covariances, ESS, log-likelihood, and optional particle histories. Filter and
model are decoupled; any filter can be applied to any compatible model.

The TF side mirrors this design but is structured for GPU execution and
autodiff. The DPF (`tf_filters/differentiable_particle.py`) accepts three
resamplers via a string flag: `'soft'`, `'sinkhorn'`, or `'amortized'`. The
amortized variant loads a pretrained CouplingOperator from disk and uses it
in place of online Sinkhorn iterations.
