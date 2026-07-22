# One-Seed GenUT/Zhao-Cui Reduced SIR Diagnostic

Status: `diagnostic_only_timing_mismatch`.

| T | GenUT value | Zhao-Cui value | value diff | GenUT score | Zhao-Cui score |
|---:|---:|---:|---:|---|---|
| 2 | -1.5089705 | -1.6710325 | 0.16206202 | [-0.0001209358379128389, 0.0020177580881863832, -0.6295980215072632] | [0.00020916482606682152, -0.003877762466372039, 0.7530906834738442] |
| 5 | -3.6728172 | -3.828794 | 0.1559768 | [-0.0004758623254019767, 0.008695369586348534, -1.2481797933578491] | [-0.0009156352954667788, 0.00866108019881179, 2.432623833037599] |
| 10 | -9.0571308 | -9.7496232 | 0.69249237 | [-0.001283282064832747, 0.01837574504315853, 0.37709593772888184] | [-0.006564923479933317, 0.03845483437375052, 7.42310964788647] |

The Zhao-Cui diagnostic observes the initial state first; GenUT observes after transition. Values and scores are therefore not directly comparable.

JSON: `/home/chakwong/BayesFilter/docs/benchmarks/artifacts/genut_sir_zhaocui_one_seed_20260722/attempt01/result.json`
