# Structural tanh Model B confirmation

Fresh data; FP32, 4032 particles, conditional FP64 reset. The finite LEDH likelihood is the HMC endpoint.

| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Sampling | True | VERIFIED | Finite likelihood approximation | Read reference and posterior diagnostics | Exact posterior accuracy |

| Inference status | Finding |
|---|---|
| Hard veto screen | Per-worker numerical validity; failures remain explicit in state.json |
| Statistically supported ranking | None |
| Descriptive differences | analysis.json contains likelihood, score, MCSE and refinement at every probe |
| Default readiness | No default change |
| Next evidence | Resolve material reference discrepancies before claiming exact-model accuracy |

Post-run alternative explanation: mixing can be good for a biased finite likelihood. Weakest evidence is long-series independent score bias. A refined reference changing beyond its measured uncertainty would overturn the accuracy interpretation.

| Parameter | Truth | Posterior mean | 95% interval |
|---|---:|---:|---|
| rho | 0.7 | 0.712934 | [0.404718, 0.944087] |
| sigma | 0.25 | 0.250943 | [0.131254, 0.3945] |
| beta | 0.8 | 0.668383 | [0.201201, 1.59612] |
