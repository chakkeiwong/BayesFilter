# Controlled DPF Baseline Fixed-Grid Result

Decision: `mp6_fixed_grid_ok`.

## Summary

- planned records: `15`
- ok records: `15`
- blocked records: `0`
- failed records: `0`
- runtime warnings: `0`

## Records

| Fixture | Seed | N | Flow steps | Status | Position RMSE | Observation proxy RMSE | Runtime seconds |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| range_bearing_gaussian_low_noise | 31 | 128 | 20 | `ok` | 0.049016 | 0.0178128 | 3.7792 |
| range_bearing_gaussian_low_noise | 43 | 128 | 20 | `ok` | 0.047458 | 0.0184954 | 3.81109 |
| range_bearing_gaussian_low_noise | 59 | 128 | 20 | `ok` | 0.0499242 | 0.0180015 | 3.86542 |
| range_bearing_gaussian_low_noise | 71 | 128 | 20 | `ok` | 0.0470935 | 0.0184322 | 3.87061 |
| range_bearing_gaussian_low_noise | 83 | 128 | 20 | `ok` | 0.0445755 | 0.0178205 | 3.78411 |
| range_bearing_gaussian_moderate | 31 | 128 | 10 | `ok` | 0.0666544 | 0.0734124 | 1.88745 |
| range_bearing_gaussian_moderate | 43 | 128 | 10 | `ok` | 0.0623641 | 0.0708784 | 1.89526 |
| range_bearing_gaussian_moderate | 59 | 128 | 10 | `ok` | 0.0605931 | 0.0717997 | 1.89236 |
| range_bearing_gaussian_moderate | 71 | 128 | 10 | `ok` | 0.0677156 | 0.0773265 | 1.8968 |
| range_bearing_gaussian_moderate | 83 | 128 | 10 | `ok` | 0.0578863 | 0.0723264 | 1.88648 |
| range_bearing_gaussian_moderate | 31 | 128 | 20 | `ok` | 0.0675154 | 0.0742734 | 3.7581 |
| range_bearing_gaussian_moderate | 43 | 128 | 20 | `ok` | 0.0623821 | 0.070475 | 3.75877 |
| range_bearing_gaussian_moderate | 59 | 128 | 20 | `ok` | 0.0638511 | 0.0716356 | 3.77455 |
| range_bearing_gaussian_moderate | 71 | 128 | 20 | `ok` | 0.0671637 | 0.0769289 | 3.76405 |
| range_bearing_gaussian_moderate | 83 | 128 | 20 | `ok` | 0.0579492 | 0.0725413 | 3.76736 |

## Notes

- MP6 fixed first-target grid only.
- Metrics are proxy diagnostics, not correctness certificates.
