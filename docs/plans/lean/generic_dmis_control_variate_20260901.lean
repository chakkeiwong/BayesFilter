/-
  Lean companion for
  bayesfilter-generic-dmis-control-variate-repair-plan-20260901.md.

  Scope: finite banks over a finite index type and a linear ordered field.
  These theorems formalize the algebra used by the continuous propositions:
  complete-mixture cancellation, the known-integral control-variate identity,
  and the explicit directional quotient rule.  They do not formalize
  measurability, integrability, TensorFlow execution, or statistical variance.
-/

import Mathlib

open scoped BigOperators

namespace GenericDMIS

variable {ι K : Type*} [Fintype ι] [Field K]

theorem complete_mixture_cancellation
    (gamma q : ι → K)
    (hq : ∀ i, q i ≠ 0) :
    (∑ i, q i * (gamma i / q i)) = ∑ i, gamma i := by
  classical
  apply Finset.sum_congr rfl
  intro i hi
  field_simp [hq i]

theorem finite_bank_control_variate
    (gamma h q b : ι → K)
    (zH : K)
    (hzH : zH = ∑ i, b i * (h i / q i))
    (hq : ∀ i, q i ≠ 0) :
    zH + ∑ i, b i * ((gamma i - h i) / q i)
      = ∑ i, b i * (gamma i / q i) := by
  classical
  rw [hzH]
  rw [← Finset.sum_add_distrib]
  apply Finset.sum_congr rfl
  intro i hi
  field_simp [hq i]
  ring

theorem frozen_control_variate_tangent
    (gamma h gammaDot hDot q qDot b : ι → K)
    (zHDot : K)
    (hq : ∀ i, q i ≠ 0) :
    zHDot + ∑ i, b i *
        (((gammaDot i - hDot i) * q i
          - (gamma i - h i) * qDot i) / (q i * q i))
      = zHDot + ∑ i, b i *
        ((gammaDot i - hDot i) / q i
          - (gamma i - h i) / q i * (qDot i / q i)) := by
  classical
  apply congrArg (fun z => zHDot + z) ?_
  apply Finset.sum_congr rfl
  intro i hi
  field_simp [hq i]

theorem positive_denominator_is_nonzero
    [LinearOrder K] [IsStrictOrderedRing K]
    (q : K) (hq : 0 < q) : q ≠ 0 := ne_of_gt hq

end GenericDMIS
