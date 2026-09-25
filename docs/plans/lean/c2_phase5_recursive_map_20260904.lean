import Mathlib

/-
These are the scalar algebraic obligations behind the Phase 5 value path.
The TensorFlow implementation additionally checks matrix shapes, finite
values, positive definiteness, and floating-point parity; those properties are
not asserted by these real-arithmetic theorems.
-/

theorem c2_phase5_affine_weighted_mean (a b w x₁ x₂ : ℝ) :
    a * (w * x₁ + (1 - w) * x₂) + b =
      w * (a * x₁ + b) + (1 - w) * (a * x₂ + b) := by
  ring

theorem c2_phase5_two_component_total_covariance
    (a b q w x₁ x₂ : ℝ) :
    w * (q + ((a * x₁ + b) -
      (a * (w * x₁ + (1 - w) * x₂) + b)) ^ 2) +
      (1 - w) * (q + ((a * x₂ + b) -
      (a * (w * x₁ + (1 - w) * x₂) + b)) ^ 2) =
      q + a ^ 2 * (
        w * (x₁ - (w * x₁ + (1 - w) * x₂)) ^ 2 +
        (1 - w) * (x₂ - (w * x₁ + (1 - w) * x₂)) ^ 2) := by
  ring

theorem c2_phase5_affine_inverse (m l u : ℝ) (hl : l ≠ 0) :
    (m + l * u - m) / l = u := by
  field_simp [hl]

