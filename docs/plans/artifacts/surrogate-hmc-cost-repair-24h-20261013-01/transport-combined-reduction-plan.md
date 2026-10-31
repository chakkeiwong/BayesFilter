# Shared transport reduction: bounded phase-2 candidate

Question: can one stable exponential supply both logsumexp and softmax in each
Sinkhorn tile faster than the existing two TensorFlow calls, without changing
the analytical all-parameter derivative or the selected finite program?
The previous fixed-kernel cache passed parity but was slower in both KSC and
LGSSM primitive probes (warmed ratios 1.093 and 1.197, respectively); reject
that cache and leave the working filter unchanged.

For finite logits z_j=C_ij+a_j, m=max_j z_j, e_j=exp(z_j-m), S=sum_j e_j:
L_i=m+log S and W_ij=e_j/S. These exactly equal logsumexp(z) and softmax(z)
in real arithmetic. The existing analytical derivative remains
sum_j W_ij(dC_ij+da_j), so no derivative term is removed. With finite logits,
S>=1; no all-underflow denominator is possible. The diagnostic falls back to
the original operations for nonfinite logits. This is a proposed computational
rewrite, not a changed scientific target.

Comparator is the existing XLA logsumexp plus softmax on identical synthetic
B2/K2016 inputs with model-matched packed contraction widths. Test KSC and LGSSM
as opposite observed transport-cost cases. CPU FP64 and all-direction finite
checks come first. Each GPU probe is capped at 300 worker-wall seconds under
the existing 86,400-second/120-attempt campaign and a unique output root.
Run 100 alternating warmed pairs, sync outputs, and record cold/warm timings,
source and input hashes, finite outputs, fallback and GPU allocator peak.
The primitive must pass <=1% scaled value/contraction error and be at least
10% faster in *both* model shapes to nominate a full-filter implementation.
This 10% threshold avoids spending on noise-sized primitive differences; it is
not a scientific accuracy threshold. If nominated, full-horizon/off-centre
value, every score component, health, memory and actual consumer timing decide
whether to retain it; fresh exact-scope tuning is needed after any source edit.

Skeptical audit: XLA may already share the exponential; if so the primitive
will be unchanged or slower. This primitive excludes packing and enclosing
loop/fusion effects, so even a large gain cannot promote the filter. The
finite-input branch must preserve nonfinite behavior through a fallback; CPU
extreme/nonfinite checks test it early. Two shapes are a cost screen, not a
statistical ranking, posterior claim or canonical LEDH admission. A failed
candidate triggers the next computational assessment rather than ending the
research direction. No target, particle count, controls or arithmetic policy
changes are authorized by this probe.
