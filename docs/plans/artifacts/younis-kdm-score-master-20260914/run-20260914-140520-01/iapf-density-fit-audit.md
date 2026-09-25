# Remaining iAPF fitting prerequisite: objective and optimizer semantics

The current fitted-twist consumer correctly labels its local log-quadratic fit and fixed iteration budget as adaptations. It does not implement the density-scale criterion in Guarniero, Johansen and Lee, Section5.1, equation(15), or the particle-count/stopping rule in Algorithm4. The local paper is `.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf`; the checked text anchors are lines650--687 (Algorithms3--4) and908--946 (the criterion, positive floor and diagonal covariance restriction). Earlier source inspection found no original-author implementation, so code parity remains unchecked.

An additional mathematical assumption must be explicit before implementing equation(15). For positive target values y_i and Gaussian density values p_i(m,Sigma), its loss is

    L(m,Sigma,lambda) = sum_i (p_i(m,Sigma) - lambda*y_i)^2.

Write a=sum p_i^2, b=sum p_i*y_i and c=sum y_i^2>0. Then

    L = a - b^2/c + c*(lambda-b/c)^2.

The fitted scale for a fixed Gaussian is lambda=b/c>0. MathDevMCP checked this completion-of-the-square identity with SymPy; its exact result is `iapf-profile-objective-mathdev.json`. It does not certify the subsequent limiting argument or a whole filter.

The unrestricted global argmin need not exist. For Sigma=s^2 I, every Gaussian density value is at most C_s=(2*pi)^(-d/2)*s^(-d). Choosing the positive scale lambda=C_s gives

    0 <= L <= C_s^2 * sum_i (1+y_i)^2 -> 0 as s -> infinity.

Thus the infimum is zero. Unless a finite Gaussian is proportional to the target vector on all support points, that infimum is not attained. A concrete one-dimensional witness is x=(-1,0,1), y=(1,1,1): equal Gaussian density at -1 and1 forces mean0, but its density at0 is strictly larger for every finite positive variance. Therefore no finite covariance and scale achieve zero loss in this witness. The paper's diagonal restriction allows this sequence. Boundedness of each positive twisting function alone does not impose a common upper covariance bound.

This does not refute the paper's reported numerical results or the target-preserving psi-APF identity. A numerical local fit can return a useful finite Gaussian, and Algorithm3 permits other approximation choices. It does mean that a local implementation cannot promise an unrestricted global minimizer or use a vanishing absolute loss as proof of a good-shaped approximation. Broadening the Gaussian lowers the density amplitude as well as changing its shape.

Before the published-objective comparison executes, specify the numerical optimizer, initialization, stopping conditions and any covariance bounds. A bounded or locally terminated optimization can use exactly the density-scale criterion, but the bounds and termination are operational choices requiring their own diagnostics. Report covariance scale, normalized shape residual, the fitted scale, boundary activity and held-out downstream normalizer/score error. Do not silently replace the loss by a normalized correlation or log loss and call it equation(15). The particle-number adaptation and final independent run in Algorithm4 are separate implementation obligations. Preserve the existing log-quadratic comparator while building and testing this additional arm.

Decision: repair the Phase0E specification before implementing a global-argmin claim. Existing valid fixed/fitted psi-APF rows and the nonlinear covariance campaign continue independently. This is a missing mathematical/optimizer specification, not evidence rejecting twisting or permission to relabel the existing local fit as published iAPF.
