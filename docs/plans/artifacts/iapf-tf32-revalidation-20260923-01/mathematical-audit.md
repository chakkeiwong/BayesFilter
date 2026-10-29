# What the repaired consumer computes

The inspected call chain is
`fitted_twist_adapter.py:30` / `iapf_adapter.py:113,154` →
`fitted_twist_tf.make_fitted_twist_kernel` →
`fitted_twist_tf.twisted_transition` at the transition inside the time loop.
The revalidation calls that shared factory and compares its complete outputs
with the independent R reconstruction. It does not test the entire adaptive
fitting/stopping program in every new precision scope. The separate prior
adaptive FP64 comparison remains the evidence for that boundary.

Let the prior transition be N(m,Q) and the nonconstant part of the frozen
guide be N(c,V). Their product, normalized as a density in x, has covariance
and mean

    C = (Q^-1 + V^-1)^-1,
    a = C(Q^-1 m + V^-1 c) = m + Q(Q+V)^-1(c-m).

The equality uses the Gaussian product identity and does not assume diagonal
or commuting covariances. Put K=Q(Q+V)^-1. The implemented repair evaluates
`a = m - Km + Kc`. It is the same real-arithmetic mean. With the guide frozen,
the analytical differential is

    da = dm - K dm - (dK)m + (dK)c,
    dK = (dQ - K dQ)(Q+V)^-1.

These are the terms in `twisted_transition`; the noise term additionally
includes the Cholesky differential of C. The earlier compiler reproducer
computed an erroneous later-time shift K(c0-ct) in the fused unsplit route.
That error is mathematically different from the required update, even if a
downstream likelihood happens to look plausible. The independent R transition
uses precision addition and so checks an independently expressed calculation.

Ordinary rounding has a different signature. For a multinomial ancestor let
Fj be the cumulative particle weights and let U be the shared uniform draw.
The selected ancestor is the first j for which U<Fj. If an arithmetic change
gives cumulative weights Ftilde, the ancestor can change only when U lies
between some Fj and Ftildej. For an ideal uniform, the union bound is

    P(ancestor differs) <= min(1, sum_j |Fj - Ftildej|).

For the actual M=2^23 lattice, each interval probability is its lattice-point
count divided by M, with endpoint conventions set by right-sided search. A
small CDF difference can therefore yield an O(1) cloud difference on a rare
shared seed: the two runs copy different particles. The Gaussian/floor mixture
selection has the same threshold property. Once a choice differs, subsequent
clouds and scores do not provide a pure arithmetic-parity comparison. This
does not prove that the likelihood distributions agree; it explains why
trajectory disagreement alone does not prove likelihood bias.

Let Z be the exact Kalman likelihood for the local model, and define each
numerical estimate's normalized likelihood r=exp(log(Zhat)-log(Z)). Comparing
paired averages of r_TF32-r_FP64 or r_TF32-r_FP32 directly targets the difference
in mean numerical likelihood estimates at a fixed dataset and guide. It does
not compare E[log(Zhat)] with log(Z): those differ even for an unbiased
positive Zhat, by Jensen's inequality. Nor does it assert exact unbiasedness
for finite precision, discretized random draws or a finite replication set.
The phase's ±0.02 mean-difference margin is an explicit engineering criterion;
the simultaneous intervals and tail-contribution veto determine whether these
replications can support it. A tail-unresolved interval stays unresolved.

Finally, the returned six-component derivative holds fitted guides and
discrete selections fixed locally. Its checked target is the derivative of
that finite value computation inside the same selection region. It is not
automatically the physical likelihood score d log p(y)/d theta. Successful
precision checks cannot close the separate marginal-score question.

Source/math self-review: the quantities compared, x0-to-y1 timing, frozen-guide
dependence, derivative target and quantized uniform construction are explicit.
No paper-fitting identity or canonical LEDH claim is inferred. The primary
fixture exporter compares Kalman recursion with a separately assembled full
observation covariance to detect a shared timing mistake before GPU analysis.
