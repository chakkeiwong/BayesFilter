# Archived posterior interval audit

All intervals use retained draws only, with four chains pooled. R-hat is recomputed after returning to physical parameter coordinates.

| Model | Parameter | Truth | 95% credible interval | Contains truth | R-hat |
|---|---|---:|---|---|---:|
| LGSSM | phi (first state) | 0.1347053 | [-0.06360755, 0.5339627] | Yes | 1.005685 |
| LGSSM | process variance multiplier | 0.18 | [0.1138141, 0.2909688] | Yes | 1.004952 |
| LGSSM | observation variance multiplier | 0.12 | [0.1196892, 0.3094956] | Yes | 1.007082 |
| LGSSM | initial mean amplitude | 0.04 | [-1.689935, 1.717321] | Yes | 1.002958 |
| KSC SV | gamma | 0.9331928 | [0.5714284, 0.9789423] | Yes | 1.005456 |
| KSC SV | beta | 1 | [0.2958376, 2.252152] | Yes | 1.005998 |
| Predator-prey | r | 0.6 | [0.4425996, 0.6369291] | Yes | 1.000578 |
| Predator-prey | K | 114 | [112.7827, 115.3278] | Yes | 1.000559 |
| Predator-prey | a | 25 | [20.31773, 29.70294] | Yes | 1.007034 |
| Predator-prey | s | 0.3 | [0.1203316, 0.6668321] | Yes | 1.001328 |
| Predator-prey | u | 0.5 | [0.01429574, 0.9698361] | Yes | 1.005984 |
| Predator-prey | v | 0.5 | [0.1145067, 0.939572] | Yes | 1.005830 |
| Range bearing | Q variance multiplier | 1 | [0.3940724, 3.251005] | Yes | 1.003620 |
| Range bearing | R variance multiplier | 1 | [0.429801, 1.444021] | Yes | 1.001577 |
| SIR NeuTra | infection-rate multiplier | 1 | [0.9617389, 1.018492] | Yes | 1.009603 |
| SIR NeuTra | recovery-rate multiplier | 1 | [0.9798169, 1.024154] | Yes | 1.003647 |
| SIR NeuTra | observation-noise SD multiplier | 1 | [0.9952579, 1.143531] | Yes | 1.007047 |

| Model | Truths inside intervals | Maximum R-hat | Chain acceptance | Within 65–75% |
|---|---:|---:|---|---|
| LGSSM | 4/4 | 1.007082 | 73.9%–74.5% | Yes |
| KSC SV | 2/2 | 1.005998 | 68.5%–72.6% | Yes |
| Predator-prey | 6/6 | 1.007034 | 71.3%–72.0% | Yes |
| Range bearing | 2/2 | 1.003620 | 68.0%–70.2% | Yes |
| SIR NeuTra | 3/3 | 1.009603 | 75.9%–79.3% | No |

The SIR parameters are dimensionless multipliers of benchmark infection rate, recovery rate and observation-noise standard deviation. Range-bearing multipliers apply to variances. LGSSM phi is the first state's coefficient; the other coefficients are fixed multiples.

A single simulated dataset per model cannot establish repeated-dataset coverage or exact posterior accuracy. These are descriptive checks; the original sampling completion decisions are unchanged.
