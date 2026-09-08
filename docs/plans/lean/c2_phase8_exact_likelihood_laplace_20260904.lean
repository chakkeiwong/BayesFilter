import Mathlib

/-!
Scoped real-algebra certificates for the C2 Phase 8 exact-likelihood Laplace
proposal. These theorems cover the zero-observation score/information limit,
the corresponding scalar stationary point, and exactness of a scalar Newton
solve for a quadratic objective. They do not certify floating-point execution,
matrix factorization, proposal support, importance-weight variance, or the
adaptive total derivative.
-/

theorem c2_zero_observation_score (x xi : ℝ) :
    -(1 : ℝ) / 2 + (0 : ℝ) ^ 2 * Real.exp (-x - 2 * xi) / 2 = -1 / 2 := by
  ring

theorem c2_zero_observation_information (x xi : ℝ) :
    (0 : ℝ) ^ 2 * Real.exp (-x - 2 * xi) / 2 = 0 := by
  ring

theorem c2_zero_observation_stationarity
    (m p : ℝ) (hp : p ≠ 0) :
    ((m - p / 2) - m) / p + 1 / 2 = 0 := by
  field_simp [hp]
  ring

theorem scalar_quadratic_newton_exact
    (h x b : ℝ) (hh : h ≠ 0) :
    x - (h * x - b) / h = b / h := by
  field_simp [hh]
  ring

theorem laplace_gaussian_covariance_inverse (p : ℝ) :
    (p⁻¹)⁻¹ = p := by
  simp
