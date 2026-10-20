# SIR score finite-difference localization

This is the predeclared repair triggered by the main tuning plan. At confirmation T=10, the baseline analytical score is (-10739.751291, 4774.539565, -2097.123241). Central differences with steps 1e-4 and 5e-5 disagree in the first two coordinates. The cause is not yet established. Selection remains frozen; these calculations cannot select controls or improve the held-out nomination.

## Question and evidence contract

Does the derivative of the exact same finite value program converge to the shared analytical score as the finite-difference step decreases? The comparator is the baseline public value_and_score result on each original confirmation dataset, N=1008 and design seed 261006411. The diagnostic obtains values from the shared trace endpoint, with initial public endpoint parity checked. It fixes the observations, noise, weights policy, route and controls.

Use central differences at 1e-5, 3e-6, 1e-6, 3e-7, 1e-7 and 3e-8 for all three coordinates at T=10,20,40,50. Report both raw values, all estimates and absolute/scaled discrepancies. Agreement requires two adjacent spacings with abs(FD-score)/(1+abs(score)) <=1e-3 and valid perturbations. Report the smallest discrepancy separately; a single favorable spacing is insufficient. Repeated base values must agree to 1e-10*(1+abs(value)). Numerical validity failures or public/trace disagreement invalidate the diagnostic. Failure to converge keeps score agreement unresolved and triggers branch/tangent localization; it does not prove the analytical formula wrong merely from a coarse difference.

The finite program may have strong curvature or changing sort/cap branches. Smaller steps may also suffer cancellation. The full ladder, two adjacent spacings, and base repeat distinguish these risks better than one chosen step. This check establishes local derivative agreement only, not statistical score accuracy, oracle agreement, posterior correctness, or an SIR improvement. It cannot rescue the rejected tuning alternatives.

## Scope, budget and review

Use the same TensorFlow environment, FP64/XLA, TF32 disabled and trusted GPU with verified memory growth as the main campaign. Run sequentially after its workers finish. A maximum 3600 seconds is charged to the original 28800-second worker budget; calculate the remaining budget before launch. Store an immutable numbered diagnostic directory under the original artifact root and retain command, source hashes, hardware, seeds, data identity and all rows. A local launch failure permits a new numbered retry only within the remaining total budget.

Skeptical audit: the scalar and inputs match the failed comparison; no finite-difference or autodiff value replaces the claim-bearing analytical score. The step ladder is fixed before these additional evaluations. Held-out controls and data remain unchanged. No oracle-accuracy inference is permitted. Proceed as a diagnostic repair within the authorized campaign.
