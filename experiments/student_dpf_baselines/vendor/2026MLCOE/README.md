# 🚀 MLCOE Summer Associate Internship Assessment (2026)

This repository contains the required code and documentation for the Machine Learning Center of Excellence (MLCOE) 2026 Summer Associate Internship assessment, focused on Time Series and Reinforcement Learning (TSRL).

## Overview

This repository contains the solution code and project report for **Part 1: From Classical Filters to Particle Flows** and **Part 2: Differentiable Sequential Monte Carlo**.

The project explores the evolution of State Space Model (SSM) filtering, starting from foundational Gaussian approximations and Sequential Monte Carlo (SMC) methods, and culminating in advanced, high-dimensional **Particle Flow Filters (PFF)**. The implementation focuses on numerical stability, particle degeneracy mitigation, and the application of invertible and kernel-based flows to enable end-to-end differentiable parameter inference.

---

## 📂 Repository Structure

The project is organized into a modular package structure to support both classical filtering and differentiable deep-learning integration.

```
jpm_pff_project/
├── src/
│   ├── filters/                      # Bayesian Filtering Library
│   │   ├── classical.py              # Foundational Gaussian-based filters (KF, EKF, UKF, ESRF)
│   │   ├── DPF.py                    # Differentiable Particle Filter core
│   │   ├── dpfpf.py                  # Differentiable Particle Flow Particle Filter core
│   │   ├── flow_filters.py           # Particle Flow implementations (EDH, LEDH)
│   │   ├── particle.py               # Sequential Monte Carlo (SMC) methods (BPF, GSMC, UPF)
│   │   ├── stochastic_flow_filter.py # SPF and LEDH with optimized schedules (SPF, LEDH+Optimal Homotopy)
│   │   ├── flows/                    # Particle Flow Vector Field Subroutines
│   │   │   ├── dai_homotopy.py       # BVP solvers for optimal schedules
│   │   │   ├── EDH.py                # Exact Daum-Huang global flow logic
│   │   │   ├── kernel.py             # Kernel-Embedded PFF (Matrix/Scalar)
│   │   │   ├── LEDH.py               # Local Exact Daum-Huang pointwise flow
│   │   │   └── stochastic.py         # Stochastic Particle Flow (SPF) vector fields
│   │   └── resampling/               # Differentiable Resampling Subroutines
│   │       ├── optimal_transport.py  # Sinkhorn/EOT and OT-based transport maps
│   │       ├── soft.py               # Differentiable soft-resampling logic
│   │       └── transformer.py        # Particle Transformer resampling priors
│   ├── inference/                    # Parameter Estimation Engines
│   │   ├── phmc.py                   # Particle Hamiltonian Monte Carlo (PHMC)
│   │   └── pmmh.py                   # Particle Marginal Metropolis-Hastings (PMMH)
│   └── models/                       # State Space Model (SSM) Library
│       ├── base_ssm.py               # Abstract base classes for SSM implementations
│       ├── classic_ssm.py            # SSMs with known transition dynamics and measurement models 
│       └── neural_ssm.py             # GRU/LSTM based transition and likelihood models
├── experiments/                      # Benchmarking & Validation Scripts
│   # (Categorized scripts for Replication, Benchmarking, and DPF Analysis)
├── tests/
│   ├── unit/                         # Mathematical Invariant Verification
│   │   └── test_unit_math.py         # Unit tests for core math & logic
│   └── integration/                  # Pipeline & Gradient Stability
│       └── test_integration_system.py # End-to-end integration tests
└── full_report.pdf                   # Comprehensive Final Report

```

## 📄 File Descriptions

### 📚 Bayesian Filtering Core (`src/filters/`)

* **`classical.py`**: Implementation of foundational filters using `float64` arithmetic globally to prevent numerical drift on high-performance architectures like the Apple M1 Ultra. Includes the **Kalman Filter (KF)** with Joseph-form updates, the **Extended Kalman Filter (EKF)**, the **Unscented Kalman Filter (UKF)** using stable eigen-decomposition sigma points, and the **Ensemble Square Root Filter (ESRF)**.
* **`particle.py`**: Implements SMC methods for non-linear/non-Gaussian state estimation, including the **Bootstrap Particle Filter (BPF)**, the **Unscented Particle Filter (UPF)** for reduced degeneracy, and **Gaussian Sum Monte Carlo (GSMC)** which bridges deterministic and stochastic filtering.
* **`flow_filters.py`**: Advanced filters that transport particles deterministically toward the posterior to avoid the "curse of dimensionality". Includes **Exact Daum-Huang (EDH)**, **Local EDH (LEDH)**, and the **Kernel Particle Flow Filter (KPFF)** utilizing functional gradient descent in RKHS.
* **`stochastic_flow_filters.py`**: Focuses on **Stochastic Particle Flow (SPF)** and localized flows utilizing optimized integration schedules (Dai, 2022) to mitigate numerical stiffness in high-stiffness regimes.
* **`DPF.py`**: A modular orchestrator for **Differentiable Particle Filters**. It decouples filtering logic from resampling subroutines, allowing the use of Neural (Transformer), Soft, or Optimal Transport (Sinkhorn) strategies.
* **`dpfpf.py`**: Implements the **Differentiable Particle Flow Particle Filter (dPFPF)**, which ensures that the marginal likelihood estimate is a differentiable function of model parameters. It serves as the likelihood engine for gradient-based inference methods.

### 🧠 Inference Engines (`src/inference/`)

* **`phmc.py`**: Implements **Particle Hamiltonian Monte Carlo (PHMC)** for efficient parameter estimation. It utilizes the `dPFPF` to obtain likelihood gradients via automatic differentiation, outperforming random-walk MCMC in high-dimensional spaces.
* **`pmmh.py`**: Implements **Particle Marginal Metropolis-Hastings (PMMH)**, targeting the exact posterior by replacing intractable likelihoods with unbiased estimates from a Particle Filter.

### 🧪 Validation Suite (`tests/`)

* **`test_unit_math.py`**: Verifies strict mathematical constraints such as mass conservation in Sinkhorn transport, unbiasedness in soft resampling, and moment reconstruction in UKF sigma points.
* **`test_integration_system.py`**: Ensures end-to-end pipeline stability, including gradient flow from system parameters back to likelihood outputs and verifying MCMC stationary distribution targeting.


## 🚀 Getting Started

### 1. Installation
Ensure you have a Python 3.10+ environment.
```
git clone [https://github.com/ljw9510/2026MLCOE.git](https://github.com/ljw9510/2026MLCOE.git)
cd 2026MLCOE
pip install -r requirements.txt
```

### 2. Running Tests
Tests are configured to run on CPU by default to ensure stability on hardware architectures with specific Metal/GPU precision constraints.
```
# Run Unit Tests
python tests/unit/test_unit_math.py

# Run Integration Tests
python tests/integration/test_integration_system.py
```

### 3. Hardware Optimization Note
The project enforces high-precision `tf.float64` globally to maintain numerical stability during long sequential filtering operations. The test suite explicitly manages device placement to prevent colocation errors during complex graph executions involving Hamiltonian and Sinkhorn subroutines.

### 🧪 Experiment Scripts

#### Literature Replication
* **`run_li17_experiments.py`**: Replicates key results from *Li & Coates (2017)*:
    * **Figure 1:** Visualizes multi-target acoustic tracking trajectories using PFPF-LEDH.
    * **Figure 2:** Benchmarks OMAT (Optimal Mass Transfer) error against baselines (EKF, UKF, GSMC) over 50 time steps.
    * **Figure 4:** Analyzes Effective Sample Size (ESS) evolution to demonstrate how PFPF mitigates weight degeneracy compared to standard SIS.
* **`run_hu21_experiments1.py`**: Replicates the "Marginal Collapse" experiment from *Hu & Van Leeuwen (2021)* on the Lorenz 96 model. It visualizes posterior contours to demonstrate how matrix-valued kernels preserve ensemble diversity in unobserved dimensions.

#### Benchmarking & Validation
* **`run_nonlinear_nonGaussian.py`**: A unified benchmarking script for Nonlinear/Non-Gaussian SSMs. It compares EKF, UKF, and PF on Stochastic Volatility and Range-Bearing models, generating RMSE metrics and trajectory plots.
* **`run_linear_Gaussian_SSM.py`**: Validates the optimality of the Kalman Filter implementation by comparing it against a Conditional Monte Carlo (CMC) baseline. It analyzes the Frobenius norm discrepancy between the analytic KF covariance and the empirical CMC covariance.
* **`run_EKF_UKF_PF_benchmark.py`**: A computational performance benchmark comparing Runtime, Peak Memory, and RMSE of EKF/UKF against Particle Filters with scaling particle counts (N=500 to N=100k) on both CPU and GPU.

#### Advanced Particle Flow Analysis
* **`run_robust_PFF_experiments.py`**: Conducts three specific robustness experiments:
    * **Exp A (Nonlinearity):** Tests filter performance on a highly nonlinear observation model ($y=x^2$).
    * **Exp B (Sparsity):** Evaluates performance on a high-dimensional (D=40) Lorenz 96 model with sparse observations (observing every 4th state).
    * **Exp C (Conditioning):** Analyzes numerical stiffness by measuring flow magnitude and condition numbers as measurement noise $R \to 0$.
* **`run_singularity_test.py`**: Specifically tests the singularity robustness of EDH vs. LEDH vs. Kernel PFF when a target passes close to a sensor origin, creating an infinite Jacobian slope.
* **`run_matrix_kernel_PFF.py`**: A dedicated runner for the Matrix-Valued Kernel PFF, comparing its tracking accuracy (RMSE) against analytic EDH and LEDH wrappers on the Range-Bearing model.
* **`run_EDH_LEDH_KPFF_comparison.py`**: A comparative analysis script that aggregates results from EDH, LEDH, and KPFF runs to generate summary comparison plots.
* **`run_dai22_experiments.py`**: Replicates Figure 2 from *Dai & Daum (2022)* by solving a **Boundary Value Problem (BVP)** to determine the optimal homotopy schedule $\beta(\lambda)$. The script uses a shooting method with bisection to minimize the log-determinant of the information matrix, effectively mitigating the numerical stiffness inherent in the Daum-Huang flow.
* **`run_li17_experiments_dai22.py`**: Extends the *Li & Coates (2017)* acoustic tracking benchmark with the **Dai (2022)** optimized schedule, demonstrating improved OMAT and ESS stability in high-stiffness regimes.

  
#### Differentiable Particle Filter (DPF) Benchmarks
* **`run_DPF_EOT_test.py`**: Analyzes the fundamental **bias-variance trade-off** in Entropic Optimal Transport (EOT) resampling. By varying the entropy parameter ($\epsilon$), it evaluates barycentric shrinkage, numerical stability "cliffs" in gradient flow, and Sinkhorn iteration convergence speed.
* **`run_DPF_market_experiment.py`**: Evaluates Differentiable Particle Filters on **Stochastic Volatility** tracking for market dynamics. It benchmarks Neural (Transformer), Soft, and EOT (Sinkhorn) resampling strategies across GRU and LSTM transition models, measuring ELBO convergence and log-volatility RMSE.

#### Bonus Questions & Parameter Estimation
* **`bonus1a.py`**: Evaluates the Invertible PFPF-LEDH on the highly non-linear benchmark from *Andrieu et al. (2010)*, featuring quadratic observations and a cosine-modulated transition model.
* **`bonus1b.py`**: Executes a comprehensive **PHMC vs. PMMH** grid-search benchmark for parameter estimation. It compares the efficiency (ESS/sec) and accuracy (MSE) of Differentiable Particle Flow gradients against standard Bootstrap Particle Filter likelihood estimates.
      


### 📄 Documentation
* **`Project_Report.pdf`**: The final submission document. It contains:
    * **Literature Review:** A comprehensive in-depth survey of Bayesian filter methods from Kalman Filters to modern Differentiable Particle Filters.
    * **Project tasks:** Detailed analysis of the experiments, including stability diagnostics (Jacobian conditioning), runtime comparisons, and error metrics. Includes both Parts I and II, and bonus questions 1-3. 


## 📚 References

The implementations and experiments in this repository are based on the following literature cited in the project report:

* **[ADH10]** Andrieu, C., Doucet, A., and Holenstein, R. "Particle markov chain monte carlo methods." *Journal of the Royal Statistical Society Series B: Statistical Methodology* (2010).
* **[BBL08]** Bengtsson, T., Bickel, P., and Li, B. "Curse-of-dimensionality revisited: Collapse of the particle filter in very large scale systems." *Probability and statistics: Essays in honor of David A. Freedman* (2008).
* **[CL23]** Chen, X. and Li, Y. "An overview of differentiable particle filters for data-adaptive sequential bayesian inference." *arXiv preprint arXiv:2302.09639* (2023).
* **[CPM25]** Chaudhari, S., Pranav, S., and Moura, J. "Gradnetot: Learning optimal transport maps with gradnets." *arXiv preprint arXiv:2507.13191* (2025).
* **[CTDD21]** Corenflos, A., Thornton, J., Deligiannidis, G., and Doucet, A. "Differentiable particle filtering via entropy-regularized optimal transport." *ICML* (2021).
* **[DC12]** Ding, T. and Coates, M. "Implementation of the daum-huang exact-flow particle filter." *IEEE Statistical Signal Processing Workshop (SSP)* (2012).
* **[DD21]** Dai, L. and Daum, F. "A new parameterized family of stochastic particle flow filters." *arXiv preprint arXiv:2103.09676* (2021).
* **[DD22]** Dai, L. and Daum, F. "Stiffness mitigation in stochastic particle flow filters." *IEEE Transactions on Aerospace and Electronic Systems* (2022).
* **[DDFG01]** Doucet, A., De Freitas, N., and Gordon, N. *Sequential monte carlo methods in practice*. Springer (2001).
* **[DH08]** Daum, F. and Huang, J. "Nonlinear filters with particle flow." *Signal Processing, Sensor Fusion, and Target Recognition XVII* (2008).
* **[DH11]** Daum, F. and Huang, J. "Particle degeneracy: root cause and solution." *Signal Processing, Sensor Fusion, and Target Recognition XX* (2011).
* **[DHN10]** Daum, F., Huang, J., and Noushin, A. "Exact particle flow for nonlinear filters." *Signal processing, sensor fusion, and target recognition XIX* (2010).
* **[DJ11]** Doucet, A. and Johansen, A. "A tutorial on particle filtering and smoothing: Fifteen years later." *The Oxford Handbook of Nonlinear Filtering* (2011).
* **[GSS93]** Gordon, N., Salmond, D., and Smith, A. "Novel approach to nonlinear/non-gaussian bayesian state estimation." *IEE Proceedings F (Radar and Signal Processing)* (1993).
* **[HVL21]** Hu, C. and Van Leeuwen, P. "A particle flow filter for high-dimensional system applications." *Quarterly Journal of the Royal Meteorological Society* (2021).
* **[Jha25]** Jha, P. "From theory to application: A practical introduction to neural operators in scientific computing." *arXiv preprint arXiv:2503.05598* (2025).
* **[JRB18]** Jonschkowski, R., Rastogi, D., and Brock, O. "Differentiable particle filters: End-to-end learning with algorithmic priors." *Robotics: Science and Systems (RSS)* (2018).
* **[JU97]** Julier, S. and Uhlmann, J. "New extension of the kalman filter to nonlinear systems." *Signal processing, sensor fusion, and target recognition VI* (1997).
* **[Kal60]** Kalman, R. "A new approach to linear filtering and prediction problems." *Journal of Basic Engineering* (1960).
* **[KHL18]** Karkus, P., Hsu, D., and Lee, W. "Particle filter networks with application to visual localization." *Conference on Robot Learning (CoRL)* (2018).
* **[LC17]** Li, Y. and Coates, M. "Particle filtering with invertible particle flow." *IEEE Transactions on Signal Processing* (2017).
* **[Rei13]** Reich, S. "A nonparametric ensemble transform method for bayesian inference." *SIAM Journal on Scientific Computing* (2013).
* **[SBBA08]** Snyder, C., Bengtsson, T., Bickel, P., and Anderson, J. "Obstacles to high-dimensional particle filtering." *Monthly Weather Review* (2008).
* **[Sch66]** Schmidt, S. "Application of state-space methods to navigation problems." *Advances in Control Systems* (1966).
* **[Vil21]** Villani, C. *Topics in optimal transportation*. American Mathematical Soc. (2021).
* **[ZAS17]** Zaheer, M., Ahmed, A., and Smola, A. "Latent lstm allocation: Joint clustering and nonlinear dynamic modeling of sequence data." *ICML* (2017).
* **[ZMJ20]** Zhu, M., Murphy, K., and Jonschkowski, R. "Towards differentiable resampling." *arXiv preprint arXiv:2004.11938* (2020).
* **[ZZA+17]** Zheng, X., Zaheer, M., Ahmed, A., Wang, Y., Xing, E., and Smola, A. "State space lstm models with particle mcmc inference." *arXiv preprint arXiv:1711.11179* (2017).



---
*Author: Joowon Lee*
*Context: JP Morgan MLCOE TSRL 2026 Internship Assessment*
