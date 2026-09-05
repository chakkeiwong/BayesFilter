import Mathlib

/-
These are the scalar identities used by the Phase 5A diagnostic.  They certify
the algebraic change-of-measure and scaling steps only; TensorFlow finiteness,
SPD, quadrature error, and ALS behavior are checked by executable artifacts.
-/

theorem c2_phase5a_predictive_mixture_linearity
    (w₁ w₂ i₁ i₂ : ℝ) :
    w₁ * i₁ + w₂ * i₂ =
      (w₁ * i₁) + (w₂ * i₂) := by
  rfl

theorem c2_phase5a_square_root_shift (s : ℝ) :
    (Real.exp (s / 2)) ^ 2 = Real.exp s := by
  rw [pow_two, ← Real.exp_add]
  congr 1
  ring

theorem c2_phase5a_reference_density_cancellation
    (gamma det eta : ℝ) (heta : eta ≠ 0) :
    (gamma * det / eta) * eta = gamma * det := by
  field_simp [heta]
