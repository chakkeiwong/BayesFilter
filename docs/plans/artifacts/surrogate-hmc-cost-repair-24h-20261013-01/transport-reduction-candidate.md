# Combined transport reduction candidate

Question: does sharing one stable exponentiation/reduction between log mass and
conditional weights reduce full finite-filter runtime? Baseline is the completed
shared-mixture candidate, or the original batch source if that candidate fails.
For finite logits a, let m=max(a), e=exp(a-m), s=sum(e). Then logsumexp(a)=m+log(s)
and softmax(a)=e/s. The analytical derivative remains the same weighted contraction.
No Sinkhorn iteration, potential, kernel, epsilon or convergence rule changes.
The compiler may already reuse these operations, so speedup is unproved.

Audit: test shifted logits, FP32/FP64 and nonfinite validity; then full-horizon,
off-centre values/all scores and same-process paired runtime/memory. Owner 1%
arithmetic screen applies. Staging or elementary timing cannot promote the change.
Missing finite rows veto promotion. A slowdown or candidate failure triggers
reversion/next planned repair, not termination of the campaign.

CPU precheck: finite cases passed. The initial nonfinite test required identical
per-element NaN placement, but TensorFlow softmax can leave a finite element in
a row containing NaN. The actual validity contract rejects the entire row.
Corrected the test to compare row validity; no kernel change or accepted-value
tolerance change was made.
