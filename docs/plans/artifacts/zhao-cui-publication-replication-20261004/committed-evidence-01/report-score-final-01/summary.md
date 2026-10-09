# Numerical score reference: actual values

These are means of separately fitted finite-sample log-likelihood derivatives. The full CSV retains every radius and fit. The table uses the smallest measured radius at each rank and sample count; this is reporting, not a selection or oracle-admission rule.

Between-fit intervals use a Student t approximation with only three independent fits where available. They exclude shared support error and radius bias. Conditional jackknife errors measure path-sampling variation given the proposal. Neither error measure certifies accuracy. Bootstrap values have their own finite-particle bias; its displayed uncertainty is one MCSE, not a confidence interval.

| Model | Rank | Paths | Parameter | h | Mean score | Between-fit 95% t halfwidth | Conditional mean SE | Bootstrap | Bootstrap MCSE |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| predator_prey | 20 | 10000 | carrying_capacity | 0.00125 | -0.55864759 | 0.035859313 | 0.0058905196 | -0.543943 | 0.0033901849 |
| predator_prey | 20 | 10000 | half_saturation | 0.00125 | 0.020413745 | 0.00020645288 | 0.00011700635 | 0.020529548 | 9.4187819e-05 |
| predator_prey | 20 | 10000 | r | 0.00125 | -33.304023 | 0.39869368 | 0.11470863 | -32.905236 | 0.075497603 |
| predator_prey | 20 | 10000 | s | 0.00125 | 4.5071742 | 0.07683888 | 0.020701044 | 4.487344 | 0.0082724402 |
| predator_prey | 20 | 10000 | u | 0.00125 | -8.7160499 | 0.014133887 | 0.023773972 | -8.7323965 | 0.022002606 |
| predator_prey | 20 | 10000 | v | 0.00125 | 10.84744 | 0.011874644 | 0.029615553 | 10.865452 | 0.027696073 |
| predator_prey | 20 | 100000 | carrying_capacity | 0.0003125 | -0.55078735 | 0.01479031 | 0.0017455561 | -0.543943 | 0.0033901849 |
| predator_prey | 20 | 100000 | half_saturation | 0.0003125 | 0.020627942 | 9.4595672e-05 | 3.2361209e-05 | 0.020529548 | 9.4187819e-05 |
| predator_prey | 20 | 100000 | r | 0.0003125 | -33.051124 | 0.18794098 | 0.033058707 | -32.905236 | 0.075497603 |
| predator_prey | 20 | 100000 | s | 0.0003125 | 4.4686647 | 0.028302594 | 0.0058654152 | 4.487344 | 0.0082724402 |
| predator_prey | 20 | 100000 | u | 0.0003125 | -8.7462792 | 0.026564263 | 0.0066156704 | -8.7323965 | 0.022002606 |
| predator_prey | 20 | 100000 | v | 0.0003125 | 10.883935 | 0.028080287 | 0.0082841917 | 10.865452 | 0.027696073 |
| sir_d18 | 20 | 10000 | log_kappa_scale | 0.01 | 197.38712 | 238.90747 | 36.774927 | 106.31321 | 5.5361736 |
| sir_d18 | 20 | 10000 | log_nu_scale | 0.01 | -113.82512 | 109.71105 | 15.865763 | -65.96031 | 2.4203782 |
| sir_d18 | 20 | 10000 | log_observation_noise_scale | 0.01 | 5.4501184 | 2.9921887 | 0.68306544 | 5.5863495 | 0.035378192 |
| sir_d18 | 40 | 10000 | log_kappa_scale | 0.00015625 | 112.68522 | — | 12.428473 | 106.31321 | 5.5361736 |
| sir_d18 | 40 | 10000 | log_nu_scale | 0.00015625 | -65.165591 | — | 5.4719705 | -65.96031 | 2.4203782 |
| sir_d18 | 40 | 10000 | log_observation_noise_scale | 0.00015625 | 5.5764456 | — | 0.08618324 | 5.5863495 | 0.035378192 |

Parameter differences (TT minus bootstrap): {"predator_prey": [-2.384185793236071e-08, 0.0, 0.0, -1.1920928966180355e-08, 0.0, 0.0], "sir_d18": [0.0, 0.0, 0.0]}. The saved PP comparator rounds r and s to float32; the separately recorded common-path likelihood check measures that shift. A likelihood-shift check alone is not a score-shift bound.

No filter ranking or production-readiness claim is made. Interpret radius, rank, sample-count and independent-reference differences together.
