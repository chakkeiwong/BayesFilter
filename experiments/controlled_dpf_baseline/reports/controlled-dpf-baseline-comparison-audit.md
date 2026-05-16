# Controlled DPF Baseline Comparison Audit

Decision: `mp7_ready_for_final_archive`.

This audit compares clean-room fixed-grid proxy metrics against frozen student aggregate evidence.  It does not execute student code and does not treat student agreement as correctness evidence.

## Inputs

- clean-room fixed-grid summary: `experiments/controlled_dpf_baseline/reports/outputs/clean_room_controlled_baseline_fixed_grid_summary.json`
- frozen student aggregate summary: `experiments/student_dpf_baselines/reports/outputs/full_horizon_edh_pfpf_confirmation_summary_2026-05-12.json`

## Comparison Table

| Cell | Clean position RMSE | Clean observation proxy RMSE | 2026MLCOE class | advanced_particle_filter class | Final class |
| --- | ---: | ---: | --- | --- | --- |
| range_bearing_gaussian_low_noise/N128/steps20 | 0.047458 | 0.0180015 | `same_qualitative_regime` | `same_qualitative_regime` | `same_qualitative_regime` |
| range_bearing_gaussian_moderate/N128/steps10 | 0.0623641 | 0.0723264 | `same_qualitative_regime` | `same_qualitative_regime` | `same_qualitative_regime` |
| range_bearing_gaussian_moderate/N128/steps20 | 0.0638511 | 0.0725413 | `same_qualitative_regime` | `same_qualitative_regime` | `same_qualitative_regime` |

## Interpretation

- All three fixed-grid cells are in the same qualitative proxy regime as both frozen student aggregate implementations under the fixed 2.0x rule.
- Moderate-noise steps10 and steps20 remain diagnostic variants; MP7 does not claim a universal moderate-noise winner.
- State RMSE, position RMSE, observation proxy RMSE, ESS, resampling, and runtime remain proxy or diagnostic evidence only.
- Student agreement is not a correctness certificate and does not authorize production or monograph claims.
