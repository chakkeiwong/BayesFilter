# Full-filter fitting comparison: numerical tables

Calibration means/SDs describe complete replicas only. Failures and planned counts are shown;
successful-subset summaries never establish a failed arm's accuracy.

| Cell | Method | Complete/planned | Failures | Mean ratio | SD | Mean N | Total mean seconds | Resampling |
|---|---|---|---|---|---|---|---|---|
| attempt01-calibration-d5-qr | qr | 4/4 | 0 | 1.01878 | 0.0783314 | 1000 | 1.90725 | 1 |
| attempt02-calibration-d20-qr | qr | 4/4 | 0 | 1.03296 | 0.114446 | 1000 | 4.20125 | 9.75 |
| attempt03-calibration-d5-box0.5 | box0.5 | 4/4 | 0 | 0.971678 | 0.0658183 | 1000 | 9.68275 | 2 |
| attempt04-calibration-d20-box0.5 | box0.5 | 4/4 | 0 | 0.645968 | 0.250265 | 2000 | 36.655 | 83.5 |
| attempt05-calibration-d5-box1 | box1 | 4/4 | 0 | 0.933848 | 0.250634 | 2250 | 19.12 | 2.25 |
| attempt06-calibration-d20-box1 | box1 | 4/4 | 0 | 1.03276 | 0.51411 | 1750 | 26.7615 | 86.25 |
| attempt07-calibration-d5-box2 | box2 | 4/4 | 0 | 0.978273 | 0.0205977 | 1500 | 10.4738 | 2.25 |
| attempt08-calibration-d20-box2 | box2 | 4/4 | 0 | 0.855452 | 0.431524 | 2000 | 39.0393 | 93 |
| attempt09-validation-89200020-box1 | box1 | 16/16 | 0 | 0.959316 | 0.500993 | 1687.5 | 22.9954 | 86.25 |
| attempt10-validation-89200020-baselines | qr | 16/16 | 0 | 0.970611 | 0.120285 | 1000 | 4.32231 | 9.3125 |
| attempt10-validation-89200020-baselines | bpf | 16/16 | 0 | 6.23861e-23 | 2.43518e-22 | 10000 | 1.06275 | 99 |
| attempt10-validation-89200020-baselines | fully_adapted | 16/16 | 0 | 0.999757 | 0.320233 | 5000 | 0.516562 | 85.375 |
| attempt10-validation-89200020-baselines | sis | 16/16 | 0 | 0 | 0 | 10000 | 0.983875 | 0 |

All complete-cell and paired intervals are retained in summary.json.
Worker use: 1073.525299/2400 seconds; 10/16 launches.
