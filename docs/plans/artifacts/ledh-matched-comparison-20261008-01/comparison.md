# Matched T=50 comparison

N=1008, FP64 GPU/XLA, TF32 disabled; original data seed 26100611. Four paired designs. Old and new receive identical initial/process tensors. The candidate also uses its required independent categorical variates.

| Model | Method | Mean log likelihood | Mean absolute likelihood error | Mean score L2 error |
|---|---|---:|---:|---:|
| lgssm | old | -132.975242030 | 0.037781598 | 0.099477128 |
| lgssm | old_covariance_only | -132.975233260 | 0.037655567 | 0.100616519 |
| lgssm | new | -132.968394486 | 0.028170090 | 0.083998506 |
| ksc | old | -123.347057715 | 0.464825069 | 0.203998471 |
| ksc | old_covariance_only | -123.445629525 | 0.511588642 | 0.218406021 |
| ksc | new | -122.819548683 | 0.211916762 | 0.063659364 |
| predator_prey | old | -240.275854361 | 0.111998179 | 1.241568242 |
| predator_prey | old_covariance_only | -240.273322222 | 0.113256135 | 1.160679385 |
| predator_prey | new | -240.229966929 | 0.043259414 | 1.234030630 |
| sir_d18 | old | -1902.553778428 | 233.443220888 | 2925.095282127 |
| sir_d18 | old_covariance_only | -1893.723439238 | 224.612881699 | 4845.454420802 |
| sir_d18 | new | -1669.180420776 | 0.181067681 | 31.631943709 |

All individual values, score coordinates, reference uncertainties, input hashes, validity checks and paired differences are in `comparison.json`.
