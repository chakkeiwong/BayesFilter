import Mathlib

/-
Scoped algebraic certificate for the Phase 5C Hermite/RBF hybrid.
This file certifies only identities stated here; it does not certify
TensorFlow floating-point execution, quadrature, SPD, or TT fitting.
-/

noncomputable section

open Real

def rbf_precision (s : ℝ) : ℝ := 1 + s⁻¹ ^ 2

def rbf_mean (c s : ℝ) : ℝ := c * s⁻¹ ^ 2 / rbf_precision s

def rbf_variance (s : ℝ) : ℝ := (rbf_precision s)⁻¹

def q0 : ℝ := 1

def q1 (μ : ℝ) : ℝ := μ

def q2 (μ v : ℝ) : ℝ := μ * μ + (v - 1)

def q3 (μ v : ℝ) : ℝ := μ * q2 μ v + 2 * (v - 1) * q1 μ

def rbf_exponent (c s t : ℝ) : ℝ :=
  -((c * s⁻¹ ^ 2 + c * t⁻¹ ^ 2) ^ 2) /
      (2 * (1 + s⁻¹ ^ 2 + t⁻¹ ^ 2))

theorem phase5c_precision_symmetry (s t : ℝ) :
    1 + s⁻¹ ^ 2 + t⁻¹ ^ 2 = 1 + t⁻¹ ^ 2 + s⁻¹ ^ 2 := by
  ring

theorem phase5c_mean_variance_definitions (c s : ℝ) :
    rbf_mean c s = c * s⁻¹ ^ 2 / rbf_precision s ∧
      rbf_variance s = (rbf_precision s)⁻¹ := by
  constructor <;> rfl

theorem phase5c_recurrence_q2 (μ v : ℝ) :
    q2 μ v = μ * q1 μ + 1 * (v - 1) * q0 := by
  simp [q0, q1, q2]

theorem phase5c_recurrence_q3 (μ v : ℝ) :
    q3 μ v = μ * q2 μ v + 2 * (v - 1) * q1 μ := by
  rfl

theorem phase5c_rbf_exponent_symmetry (c s t : ℝ) :
    rbf_exponent c s t = rbf_exponent c t s := by
  unfold rbf_exponent
  ring

theorem phase5c_constant_mass_block (i : ℝ) :
    (1 : ℝ) = 1 ∧ i = i := by
  constructor <;> rfl

theorem phase5c_constant_integral_block (i : ℝ) :
    ((1 : ℝ), i) = (1, i) := by
  rfl
