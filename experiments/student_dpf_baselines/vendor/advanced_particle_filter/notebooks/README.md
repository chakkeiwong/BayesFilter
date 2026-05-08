# Notebooks index

All exploratory and experimental notebooks. Topics 1, 2, 3 progress
from literature replication (1) → research on HMC + DPF (2) → research
on neural amortization of OT resampling (3).

We ran all Colab notebooks on Google Colab with **A100 40GB GPU**.
T4 will work for some smoke tests but is too slow for the full HMC
chains and full operator training.

| Notebook | Runtime | Topic | What it does |
|---|---|---|---|
| **01_classical_and_differentiable_pf/** |  |  |  |
| 01_lgssm.ipynb                          | Local CPU | Classical | KF on linear-Gaussian SSM. |
| 02_nonlinear_ssm.ipynb                  | Local CPU | Classical | EKF/UKF/BPF on range-bearing model. |
| 03_Li2017_acoustic.ipynb                | Local CPU | Classical | Li & Coates (2017) acoustic-tracking replication. |
| 04_pff_comparison.ipynb                 | Local CPU | Classical | EDH vs LEDH vs kernel PFF comparison. |
| 05_kernel_pff.ipynb                     | Local CPU | Classical | Scalar vs matrix kernel; collapse analysis. |
| 06_dai2022_seed_sensitivity.ipynb       | Local CPU | Classical | Dai & Daum (2022) Example 1 seed sensitivity. |
| 07_differentiable_pf.ipynb              | Colab GPU | DPF intro | Differentiable PF with soft and OT resampling. |
| 08_sinkhorn_parameter_sweep.ipynb       | Colab GPU | DPF intro | Sweep over Sinkhorn ε and iteration count. |
| 09_three_model_dpf.ipynb                | Colab GPU | DPF intro | DPF with normalizing-flow proposal across three SSMs. |
| **02_hmc_differentiable_resampling/** |  |  |  |
| 01_diagnostics_hmc_dpf_linear.ipynb     | Colab GPU | Research | DPF gradient quality on a linear-Gaussian-observation variant of the SVSSM (Kalman gives ground truth; isolates DPF bias from observation nonlinearity). |
| 02_hmc_dpf_svssm.ipynb                  | Colab GPU | Research | SVSSM HMC: soft vs Sinkhorn-OT resampler comparison. |
| 03_mle_laplace_pipeline.ipynb           | Colab GPU | Research | Three-phase MAP + Laplace + preconditioned-HMC pipeline. |
| **03_neural_amortized_ot/** |  |  |  |
| 01_brenier_gap_evidence.ipynb           | Colab GPU | Research | Why the original M-MGN architecture failed; motivation for the coupling-predictor design. |
| 02_train_coupling_operator.ipynb        | Colab GPU | Research | Train the amortized OT operator (coupling-predictor) on precomputed Sinkhorn targets. |
| 03_hmc_amortized_comparison.ipynb       | Colab GPU | Research | SVSSM HMC: soft vs Sinkhorn vs amortized (three-resampler comparison). |

## Standard Colab setup pattern

All Colab notebooks follow this convention:

1. **Runtime**: GPU. Set via Runtime → Change runtime type.
2. **Dependencies**: `!pip install -q tensorflow-probability` runs in the
   first code cell. TensorFlow itself is preinstalled in Colab.
3. **Codebase upload**: zip the `advanced_particle_filter/` folder, upload it
   to Colab, extract at `/content/advanced_particle_filter/`. The first
   setup cell does this via `files.upload()` and caches a copy to Drive
   at `MyDrive/advanced_particle_filter_clean.zip` for later sessions.
4. **Drive mount** (topic 3 notebooks 02 and 03):
   - Notebook 02 expects `MyDrive/mgn_ot_operator/data/` for precomputed
     Sinkhorn `.npz` targets and `MyDrive/mgn_ot_operator/checkpoints_option_b/`
     for training output. If targets are absent, full precompute runs.
   - Notebook 03 looks for the trained operator at
     `MyDrive/mgn_ot_operator/checkpoints_option_b/best/` and falls back
     to the bundled copy at
     `dpf_pretrained/mgn_ot_operator/checkpoints_option_b/best/`.

Notebooks 01 and 02 of topic 3 use the operator-training subpackage at
`dpf_pretrained/mgn_ot_operator/`. Their setup cell adds that directory to
`sys.path` so imports like `from data.X`, `from training.X`, and
`from config import CONFIG` resolve to it.

## What you need to run each topic

- **Topic 1**: For local notebooks (01–06), only NumPy/SciPy. For Colab
  notebooks (07–09), only TensorFlow + TFP (installed by the first cell).
- **Topic 2**: All Colab. Need TF + TFP. No external state required —
  notebooks generate their own SVSSM data.
- **Topic 3**: All Colab. Notebook 02 optionally reads precomputed
  Sinkhorn targets from Drive; if none are present it precomputes on
  the fly. Notebook 03 uses the bundled trained operator out of the box.
