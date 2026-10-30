# KDM native sampling result — 6 October 2026

The pinned author KDE sampler and TensorFlow GPU/XLA implementation passed 28/28 checks on a seven-concentration, three-component replay fixture in both float64 and float32. The source reference ran in the verified author environment with CUDA hidden. The TF path used GPU memory growth, TF32 disabled and one compiled trace per dtype.

The replay preserved source categorical ancestors, per-component normal draws, circular rejection uniforms, source concentration correction 1/(bandwidth+1e-8), internal double-precision circular rejection, angle wrapping and stopped sampling gradients. Source loop count and validity/rejection checks matched. Native 32768 draws in each dtype passed the declared six-standard-error analytic moment and component-frequency diagnostics, including zero-weight categories and angular seam.

This is an implementation and distribution diagnostic. It does not establish training equivalence, a complete checkpoint/stage controller, eleven-trial paper replication, or accepted downstream no-worse stochastic margins. Native draws used a separate stateless TensorFlow RNG; identical integer seeds across Torch and TF were not claimed to produce identical samples.

Attempt 01 failed before numerical work because the generic interpreter lacked the source environment tqdm. Attempt 02 used the pinned reference interpreter and completed. The source CPU reference took 7.032s and the GPU/XLA sampler took 16.410s. ptxas register-spill messages were compilation diagnostics; no invalid result or nonfinite sample occurred.
