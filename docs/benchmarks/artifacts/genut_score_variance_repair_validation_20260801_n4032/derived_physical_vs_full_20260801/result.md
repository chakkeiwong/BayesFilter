# Austria Physical Versus Full-Filter Growth

This derived diagnostic compares the deterministic nominal RK4 transition with the full particle-filter tangent.
The physical curve is 18-D; the full-filter curves are N x 18 and are pooled over the three attempt-08 seeds and eight probes.

| Step | Physical transition | Full diagonal | Full pairwise |
|---:|---:|---:|---:|
| 1 | +0.486070 | +0.510657 | +0.511968 |
| 2 | +0.633596 | +0.559829 | +0.577504 |
| 3 | +0.530335 | +1.088280 | +1.003553 |
| 4 | +0.375486 | +0.466064 | +0.493369 |
| 5 | +0.196010 | +0.853910 | +0.722397 |
| 6 | +0.005408 | +1.566862 | +0.382061 |
| 7 | -0.165905 | +0.418569 | +0.174623 |
| 8 | -0.271746 | +0.282340 | +0.355391 |
| 9 | -0.282231 | -0.012790 | +0.194983 |
| 10 | -0.237049 | -0.066265 | +0.062746 |
| 11 | -0.206295 | +0.184721 | +0.222565 |
| 12 | -0.205721 | -0.005861 | +0.529352 |
| 13 | -0.218472 | -0.036888 | +0.100977 |
| 14 | -0.232200 | -0.094817 | +0.045139 |
| 15 | -0.242409 | -0.007795 | +0.097320 |
| 16 | -0.247981 | -0.129547 | +0.004432 |
| 17 | -0.248667 | +0.100194 | +0.452507 |
| 18 | -0.244206 | +0.381628 | +0.269693 |
| 19 | -0.234167 | +0.352929 | +0.499548 |
| 20 | -0.218165 | +0.296468 | +0.622039 |

- Physical cumulative log growth: `-1.028309` (factor `0.358`).
- Full diagonal cumulative log growth: `+6.708489` (factor `819`).
- Full pairwise cumulative log growth: `+7.322165` (factor `1.51e+03`).
- Interpretation: the physical transition is strongly expanding early and becomes contracting later; the full particle map remains near-neutral or expansive at several later steps, so filtering operations add amplification beyond the physical transition.
- Nonclaim: this is not a common-dimensional operator-norm comparison and does not identify a causal percentage for OT/reset stages.
