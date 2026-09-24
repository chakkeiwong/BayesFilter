import Mathlib

/-!
Scoped algebraic certificates for the C2 Phase 7 integrated diagnostic.

These theorems certify only the real-algebra cancellation used by a frozen
complete-proposal importance ratio and the normalization of a two-component
defensive mixture.  They do not certify TensorFlow execution, proposal support,
finite variance, filtering consistency, or the hybrid TT fit.
-/

theorem complete_dmis_cancellation
    (b a q gamma phi : ℝ) (ha : a ≠ 0) (hq : q ≠ 0) :
    b * a * q * (gamma / (a * q)) * phi = b * gamma * phi := by
  field_simp [ha, hq]

theorem complete_dmis_cancellation_factored
    (b a q gamma : ℝ) (ha : a ≠ 0) (hq : q ≠ 0) :
    b * (gamma / (a * q)) * (a * q) = b * gamma := by
  field_simp [ha, hq]

theorem two_component_mixture_mass (epsilon : ℝ) :
    (1 - epsilon) + epsilon = 1 := by
  ring

theorem four_equal_component_mass :
    (4 : ℝ) * (1 / 4) = 1 := by
  norm_num

theorem weighted_second_moment_nonnegative
    (w q delta : ℝ) (hw : 0 <= w) (hq : 0 <= q) :
    0 <= w * (q + delta ^ 2) := by
  positivity
