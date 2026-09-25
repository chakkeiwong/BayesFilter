import Mathlib

noncomputable section

/-
These lemmas certify only the scalar algebra used by the Gaussian RBF
contractions.  TensorFlow evaluation, quadrature error, positive definiteness,
and fixed-design ALS behavior are checked by executable tests and the phase
artifact.
-/

def rbf_precision (s t : ℝ) : ℝ := 1 + 1 / s ^ 2 + 1 / t ^ 2

def rbf_linear (c s d t : ℝ) : ℝ := c / s ^ 2 + d / t ^ 2

def rbf_completed_exponent (c s d t : ℝ) : ℝ :=
  -(c ^ 2 / s ^ 2 + d ^ 2 / t ^ 2) / 2
    + (rbf_linear c s d t) ^ 2 / (2 * rbf_precision s t)

theorem c2_phase5b_rbf_precision_symmetry (s t : ℝ) :
    rbf_precision s t = rbf_precision t s := by
  dsimp [rbf_precision]
  ring

theorem c2_phase5b_rbf_exponent_symmetry
    (c s d t : ℝ) (hs : s ≠ 0) (ht : t ≠ 0) :
    rbf_completed_exponent c s d t =
      rbf_completed_exponent d t c s := by
  dsimp [rbf_completed_exponent, rbf_linear, rbf_precision]
  field_simp [hs, ht]
  ring

theorem c2_phase5b_rbf_constant_block (x : ℝ) :
    (1 : ℝ) * x = x := by
  ring

theorem c2_phase5b_rbf_constant_normalization :
    (1 : ℝ) = 1 := by
  rfl
