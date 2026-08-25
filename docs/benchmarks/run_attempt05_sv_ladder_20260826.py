"""attempt05: SV rank ladder under C2 (plan 2026-08-26, owner go).

Modes:
  (orchestrator)            full campaign: refs -> screens -> ladders
  --reference n obs_seed    build one reference (subprocess, CPU)
  --cell n degree rank obs_seed [sweeps]   run one cell (subprocess, GPU)

Per-cell subprocess isolation (A3 LLVM lesson), 60-min timeout,
append-only accumulator, manifests, heartbeats 'A5 ...'.
"""
import json
import math
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "docs", "benchmarks", "artifacts",
                   "c2_completion_20260824", "attempt05")
BAR = 2.5e-3
T_HORIZON = 20
MODEL_SEED = 52
OBS_SEEDS = (42, 142, 242)
DEGREES = (2, 4, 6)
RANKS = (1, 2, 3, 4, 6)
ROWS = 8192
SWEEPS = 32
ALPHA_MAX = 0.8
PF_NS = (400_000, 800_000)
PF_R = 10


def _cell_path(n, degree, rank, obs_seed, sweeps):
    return os.path.join(OUT, f"cell_n{n}_d{degree}_r{rank}_s{obs_seed}_w{sweeps}.json")


def _ref_path(n, obs_seed):
    return os.path.join(OUT, f"reference_n{n}_s{obs_seed}.json")


def _heartbeat(msg):
    print(f"A5 {msg}", flush=True)


# ---------------------------------------------------------------- reference
def run_reference(n, obs_seed):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "docs", "benchmarks"))
    import sv_fixture_c2_20260826 as SV

    model = SV.sv_model(n, MODEL_SEED)
    ys = SV.sv_simulate(model, T_HORIZON, obs_seed)
    record = {"n": n, "obs_seed": obs_seed, "horizon": T_HORIZON}
    if n == 2:
        total = sum(SV.sv_grid_reference_2d(model, ys, width=11.0, points=295))
        record.update({"kind": "exact_grid", "total": total, "valid": True})
    else:
        arms = {}
        for npart in PF_NS:
            pf = SV.sv_particle_reference(model, ys, n_particles=npart,
                                          replicates=PF_R, seed=31 + obs_seed)
            arms[str(npart)] = pf
        lo, hi = arms[str(PF_NS[0])], arms[str(PF_NS[1])]
        joint_se = math.sqrt(lo["se_total"] ** 2 + hi["se_total"] ** 2)
        doubling_ok = abs(lo["mean_total"] - hi["mean_total"]) <= 2.0 * joint_se
        se_ok = hi["se_total"] <= T_HORIZON * BAR / 5.0
        screen_ok = min(lo["min_normalized_ess"], hi["min_normalized_ess"]) >= 0.05
        record.update({
            "kind": "particle", "arms": arms,
            "total": hi["mean_total"], "se_total": hi["se_total"],
            "doubling_ok": doubling_ok, "se_ok": se_ok, "screen_ok": screen_ok,
            "valid": bool(doubling_ok and se_ok and screen_ok),
        })
    with open(_ref_path(n, obs_seed), "w") as fh:
        json.dump(record, fh, indent=1)
    _heartbeat(f"reference n={n} seed={obs_seed} valid={record['valid']} "
               f"total={record['total']:+.4f}")


# ---------------------------------------------------------------- cell
def run_cell(n, degree, rank, obs_seed, sweeps):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "docs", "benchmarks"))
    import numpy as np
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    import sv_fixture_c2_20260826 as SV
    from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig
    from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
        student_t_nu_criterion,
    )
    from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import (
        run_value_filter_branch_axis_gaussian_xla,
    )

    nu = student_t_nu_criterion(ALPHA_MAX, 12.0)
    model = SV.sv_model(n, MODEL_SEED)
    ys = SV.sv_simulate(model, T_HORIZON, obs_seed)
    adapter = SV.sv_adapter(model)
    ih_raw, ph_raw = SV.sv_gh_hint_factory(model, gh_points=9)
    alphas = []

    def _alpha_from_cov(cov):
        eig = float(np.min(np.linalg.eigvalsh(np.asarray(cov)[:n, :n])))
        alphas.append(1.0 - eig / SV.SIGMA**2)

    def initial_hint(y0):
        m, c = ih_raw(y0)
        _alpha_from_cov(c.numpy())
        return m, c

    def predictive_hint(t, y_t):
        m, c = ph_raw(t, y_t)
        _alpha_from_cov(c.numpy())
        return m, c

    config = EngineConfig(
        basis_degree=degree, rank=rank, row_count=ROWS, sweeps=sweeps,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0,
        seed=98000 + 100 * n + 10 * degree + rank, row_design="sobol",
    )
    t0 = time.perf_counter()
    value, diags = run_value_filter_branch_axis_gaussian_xla(
        adapter, tf.constant(ys, tf.float64), config,
        predictive_moment_hint=predictive_hint, initial_moment_hint=initial_hint,
        defensive_nu=nu,
    )
    wall = time.perf_counter() - t0
    corrected = float(value.numpy()) - sum(math.log1p(d["tau_t"]) for d in diags)
    design_width = rank * (degree + 1) * rank if rank > 1 else (degree + 1)
    record = {
        "n": n, "degree": degree, "rank": rank, "obs_seed": obs_seed,
        "sweeps": sweeps, "rows": ROWS, "nu": nu, "horizon": T_HORIZON,
        "corrected_total": corrected, "wall_s": wall,
        "row_ess_min": min(d["row_ess"] for d in diags),
        "ess_floor": 5 * design_width,
        "tau_max_seen": max(d["tau_t"] for d in diags),
        "rms_max": max(d.get("weighted_fit_rms", 0.0) for d in diags),
        "cond_max": max(d.get("worst_condition", 0.0) for d in diags),
        "alpha_max_seen": max(alphas),
        "device": [tf.config.experimental.get_device_details(g).get("device_name")
                   for g in gpus],
        "memory_growth_verified": all(
            tf.config.experimental.get_memory_growth(g) for g in gpus
        ) if gpus else "no_gpu_visible",
    }
    with open(_cell_path(n, degree, rank, obs_seed, sweeps), "w") as fh:
        json.dump(record, fh, indent=1)


# ---------------------------------------------------------------- orchestrator
def _spawn(args, timeout=3600):
    proc = subprocess.run([sys.executable, os.path.abspath(__file__)] + args,
                          capture_output=True, text=True, timeout=timeout)
    sys.stdout.write(proc.stdout)
    sys.stdout.flush()
    return proc.returncode


def _evaluate_cell(n, degree, rank, obs_seed, sweeps):
    ref = json.load(open(_ref_path(n, obs_seed)))
    cell = json.load(open(_cell_path(n, degree, rank, obs_seed, sweeps)))
    per_step = abs(cell["corrected_total"] - ref["total"]) / T_HORIZON
    vetoes = []
    if not ref["valid"]:
        vetoes.append("reference_invalid")
    if not math.isfinite(cell["corrected_total"]):
        vetoes.append("non_finite")
    if cell["row_ess_min"] < cell["ess_floor"]:
        vetoes.append("row_ess_floor")
    if cell["tau_max_seen"] >= 1e-4:
        vetoes.append("tau_at_cap")
    if cell["alpha_max_seen"] > ALPHA_MAX:
        vetoes.append("alpha_exceeds_declared")
    passed = (per_step <= BAR) and not vetoes
    row = {**cell, "reference_total": ref["total"], "per_step_gap": per_step,
           "vetoes": vetoes, "passed": passed}
    with open(os.path.join(OUT, "rows.jsonl"), "a") as fh:
        fh.write(json.dumps(row) + "\n")
    _heartbeat(f"cell n={n} d={degree} r={rank} seed={obs_seed} "
               f"gap={per_step:.2e} pass={passed} vetoes={vetoes} "
               f"wall={cell['wall_s']:.0f}s alpha={cell['alpha_max_seen']:.2f}")
    return passed, per_step


def _run_cell_guarded(n, degree, rank, obs_seed, sweeps=SWEEPS):
    if os.path.exists(_cell_path(n, degree, rank, obs_seed, sweeps)):
        _heartbeat(f"cell n={n} d={degree} r={rank} seed={obs_seed} RESUMED")
        return _evaluate_cell(n, degree, rank, obs_seed, sweeps)
    try:
        rc = _spawn(["--cell", str(n), str(degree), str(rank),
                     str(obs_seed), str(sweeps)])
    except subprocess.TimeoutExpired:
        _heartbeat(f"cell n={n} d={degree} r={rank} seed={obs_seed} TIMEOUT")
        with open(os.path.join(OUT, "rows.jsonl"), "a") as fh:
            fh.write(json.dumps({"n": n, "degree": degree, "rank": rank,
                                 "obs_seed": obs_seed, "vetoes": ["timeout"],
                                 "passed": False}) + "\n")
        return False, float("inf")
    if rc != 0:
        _heartbeat(f"cell n={n} d={degree} r={rank} seed={obs_seed} CRASH rc={rc}")
        with open(os.path.join(OUT, "rows.jsonl"), "a") as fh:
            fh.write(json.dumps({"n": n, "degree": degree, "rank": rank,
                                 "obs_seed": obs_seed, "vetoes": ["crash"],
                                 "passed": False}) + "\n")
        return False, float("inf")
    passed, gap = _evaluate_cell(n, degree, rank, obs_seed, sweeps)
    # Declared repair (plan sec 2): one retry at 2x sweeps when the fit
    # residual capped tau; recorded as its own accumulator row.
    if not passed and sweeps == SWEEPS:
        row = json.load(open(_cell_path(n, degree, rank, obs_seed, sweeps)))
        ref_valid = json.load(open(_ref_path(n, obs_seed)))["valid"]
        if (row["tau_max_seen"] >= 1e-4 and ref_valid
                and math.isfinite(row["corrected_total"])):
            _heartbeat(f"cell n={n} d={degree} r={rank} seed={obs_seed} "
                       f"RETRY at sweeps={2 * SWEEPS} (declared repair)")
            return _run_cell_guarded(n, degree, rank, obs_seed, 2 * SWEEPS)
    return passed, gap


def orchestrate():
    os.makedirs(OUT, exist_ok=True)
    manifest = {
        "plan": "bayesfilter-attempt05-sv-rank-ladder-plan-2026-08-26.md",
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                 capture_output=True, text=True).stdout.strip(),
        "bar_per_step": BAR, "horizon": T_HORIZON, "model_seed": MODEL_SEED,
        "obs_seeds": OBS_SEEDS, "rows": ROWS, "sweeps": SWEEPS,
        "alpha_max": ALPHA_MAX, "pf_ns": PF_NS, "pf_r": PF_R,
    }
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=1)

    # n=4 references in a parallel CPU worker while n=2 proceeds
    n4_ref_procs = [
        subprocess.Popen([sys.executable, os.path.abspath(__file__),
                          "--reference", "4", str(s)],
                         stdout=subprocess.PIPE, text=True)
        for s in OBS_SEEDS if not os.path.exists(_ref_path(4, s))
    ]
    for s in OBS_SEEDS:
        if not os.path.exists(_ref_path(2, s)):
            _spawn(["--reference", "2", str(s)], timeout=1800)

    verdict = {}
    for n in (2, 4):
        if n == 4:
            for proc in n4_ref_procs:
                out, _ = proc.communicate(timeout=7200)
                sys.stdout.write(out)
                sys.stdout.flush()
                if proc.returncode != 0:
                    raise RuntimeError("n=4 reference build failed (fail closed)")
        # degree screen (explanatory, one seed, rank 6)
        screen = {}
        for degree in DEGREES:
            deg_passed, gap = _run_cell_guarded(n, degree, 6, OBS_SEEDS[0])
            screen[degree] = {"passed": deg_passed, "gap": gap}
        passing = [d for d in DEGREES if screen[d]["passed"]]
        working_degree = (min(passing) if passing
                          else min(screen, key=lambda d: screen[d]["gap"]))
        if not passing:
            _heartbeat(f"screen n={n}: NO veto-clean passing degree — "
                       f"gap-nominated degree {working_degree}; ladder runs "
                       f"under that recorded caveat")
        _heartbeat(f"screen n={n}: "
                   f"{ {d: (v['passed'], round(v['gap'], 8)) for d, v in screen.items()} } "
                   f"working_degree={working_degree}")
        # rank ladder
        r_star = None
        for rank in RANKS:
            results = [_run_cell_guarded(n, working_degree, rank, s)
                       for s in OBS_SEEDS]
            if all(p for p, _ in results):
                r_star = rank
                break
        verdict[f"n{n}"] = {"working_degree": working_degree,
                            "degree_screen": screen, "r_star": r_star}
        _heartbeat(f"VERDICT n={n}: r_star={r_star} degree={working_degree}")
    with open(os.path.join(OUT, "verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=1)
    _heartbeat(f"DONE {json.dumps(verdict)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--reference":
        run_reference(int(sys.argv[2]), int(sys.argv[3]))
    elif len(sys.argv) > 1 and sys.argv[1] == "--cell":
        run_cell(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
                 int(sys.argv[5]),
                 int(sys.argv[6]) if len(sys.argv) > 6 else SWEEPS)
    else:
        orchestrate()
