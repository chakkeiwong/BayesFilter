# Austria Physical Versus Full-Filter Growth

This derived diagnostic compares the deterministic nominal RK4 transition with the full particle-filter tangent.
The physical curve is 18-D; the full-filter curves are N x 18 and are pooled over the three attempt-08 seeds and eight probes.

| Step | Physical transition | Full diagonal | Full pairwise |
|---:|---:|---:|---:|
| 1 | +0.486070 | +0.522265 | +0.523477 |
| 2 | +0.633596 | +0.623177 | +0.622309 |
| 3 | +0.530335 | +1.296727 | +0.988925 |
| 4 | +0.375486 | +0.332284 | +0.395644 |
| 5 | +0.196010 | +0.696516 | +0.926506 |
| 6 | +0.005408 | +0.446881 | +0.269068 |
| 7 | -0.165905 | +0.445896 | +0.052829 |
| 8 | -0.271746 | +0.087154 | -0.039790 |
| 9 | -0.282231 | +0.092215 | -0.015288 |
| 10 | -0.237049 | -0.000873 | +0.151834 |
| 11 | -0.206295 | +0.214792 | +0.021631 |
| 12 | -0.205721 | +0.129719 | -0.027103 |
| 13 | -0.218472 | +0.015489 | +0.091573 |
| 14 | -0.232200 | -0.158311 | -0.064372 |
| 15 | -0.242409 | -0.028913 | +0.094093 |
| 16 | -0.247981 | +0.116280 | +0.103206 |
| 17 | -0.248667 | +0.200214 | +0.137811 |
| 18 | -0.244206 | +0.045190 | -0.013321 |
| 19 | -0.234167 | +0.114056 | +0.298287 |
| 20 | -0.218165 | +0.256688 | +0.270357 |

- Physical cumulative log growth: `-1.028309` (factor `0.358`).
- Full diagonal cumulative log growth: `+5.447448` (factor `232`).
- Full pairwise cumulative log growth: `+4.787676` (factor `120`).
- Interpretation: the physical transition is strongly expanding early and becomes contracting later; the full particle map remains near-neutral or expansive at several later steps, so filtering operations add amplification beyond the physical transition.
- Nonclaim: this is not a common-dimensional operator-norm comparison and does not identify a causal percentage for OT/reset stages.
