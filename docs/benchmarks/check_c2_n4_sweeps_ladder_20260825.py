import math, os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/home/chakwong/BayesFilter")
sys.path.insert(0, "/home/chakwong/BayesFilter/tests/highdim")
import test_c2_gaussian_engine_oracle as T
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import run_value_filter_branch_axis_gaussian

for sweeps in (8, 16, 32):
    adapter, ys, steps, model = T._lgssm_fixture(4, 4, 46)
    ih, ph = T._exact_hint_factories(model)
    config = EngineConfig(basis_degree=6, rank=3, row_count=8192, sweeps=sweeps,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=95106, row_design="sobol")
    v, d = run_value_filter_branch_axis_gaussian(adapter, ys, config,
        predictive_moment_hint=ph, initial_moment_hint=ih)
    gap = abs(float(v.numpy()) - sum(math.log1p(x["tau_t"]) for x in d) - sum(steps))
    per = max(abs(x["log_increment"] - math.log1p(x["tau_t"]) - k) for x, k in zip(d, steps))
    print(f"N4SWEEPS s={sweeps:2d}: gap={gap:.3e} per_step_max={per:.3e}", flush=True)
print("N4SWEEPS DONE", flush=True)
