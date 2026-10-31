"""CPU diagnostic degree probe with a frozen reference fixture and TF hints."""
import math, os, sys
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.highdim import test_c2_gaussian_engine_oracle as T
from bayesfilter.highdim.gaussian_moment_hints_tf import prepare_lgssm_moment_hints
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
from bayesfilter.highdim.squared_tt_engine_gaussian_tf import run_value_filter_branch_axis_gaussian

for deg, rank in ((4, 2), (6, 3)):
    adapter, ys, steps, model = T._lgssm_fixture(4, 4, 46)
    hints = prepare_lgssm_moment_hints(ys, *(model[key] for key in ("A", "Q", "H", "R", "m0", "P0")))
    initial_hint, predictive_hint = hints.callbacks()
    config = EngineConfig(basis_degree=deg, rank=rank, row_count=8192, sweeps=8,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0, seed=95100+deg, row_design="sobol")
    v, d = run_value_filter_branch_axis_gaussian(adapter, ys, config,
        predictive_moment_hint=predictive_hint, initial_moment_hint=initial_hint)
    gap = abs(float(v.numpy()) - sum(math.log1p(x["tau_t"]) for x in d) - sum(steps))
    ess = min(x["row_ess"] for x in d)
    print(f"N4LOWDEG deg={deg} rank={rank}: gap={gap:.3e} ess_min={ess:.0f} "
          f"cond={max(x.get('worst_condition',0) for x in d):.1e}", flush=True)
