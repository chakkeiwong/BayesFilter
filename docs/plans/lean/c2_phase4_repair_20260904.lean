import Mathlib

/-
The first theorem is the exact APF log-space rearrangement used by the
implementation.  The second theorem proves that the scale used for the
normalized backward error is strictly positive.  Neither theorem claims a
floating-point error bound; that bound is a separately tested numerical
policy.
-/
theorem c2_apf_identity (a q g : ℝ) :
    a + q + (g - a - q) - g = 0 := by
  ring

theorem c2_backward_error_scale_positive (a q w g : ℝ) :
    0 < max 1 (|a| + |q| + |w| + |g|) := by
  have h : (0 : ℝ) < 1 := by norm_num
  exact lt_of_lt_of_le h (le_max_left _ _)
