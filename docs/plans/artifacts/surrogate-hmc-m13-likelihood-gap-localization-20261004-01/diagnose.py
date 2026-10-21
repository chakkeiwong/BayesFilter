"""Diagnostic only: localize the frozen M13 likelihood gap without changing it."""
import json, pathlib, subprocess, sys, time
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PREVIOUS = HERE.parent / "surrogate-hmc-m13-score-before-after-20261004-01"
sys.path.insert(0, str(PREVIOUS))
import measure as m
PLAN = "docs/plans/surrogate-hmc-m13-likelihood-gap-localization-2026-10-04.md"

def worker():
    started = time.monotonic()
    tf, environment = m.setup_tf(PREVIOUS / "source_after", False)
    from bayesfilter.highdim.ledh_range_bearing_target_tf import make_range_bearing_model
    from bayesfilter.highdim.ledh_canonical_batch_fused_tf import _single_cloud_model
    from bayesfilter.highdim.ledh_canonical_score_tf import canonical_value_and_analytical_score
    from bayesfilter.testing.surrogate_hmc_m13_reference import quadrature
    import tensorflow_probability as tfp
    environment["tensorflow_probability"] = tfp.__version__
    common = m.read(PREVIOUS / "inputs.json")
    prior = m.read(PREVIOUS / "attempt-02/after-float64.json")
    checked_sources = {}
    for relative, expected in prior["imported_source_hashes"].items():
        path = PREVIOUS / "source_after" / relative
        assert m.sha(path) == expected, relative
        checked_sources[relative] = expected
    inputs = m.input_tensors(tf, common, tf.float64)
    controls = common["controls"] | {"reset_design": inputs.pop("reset_design")}
    controls["flow_substeps"] = controls.pop("substeps")
    coefficients = common["model_coefficients_rounded_fp32"]
    model = make_range_bearing_model(
        *[tf.constant(coefficients[k], tf.float64) for k in ("A", "Q", "R")],
        dtype=tf.float64)
    q_diagonal = tf.linalg.diag_part(tf.constant(coefficients["Q"], tf.float64))

    @tf.function(input_signature=[tf.TensorSpec([2], tf.float64),
                                  tf.TensorSpec([2], tf.float64)],
                 jit_compile=True, autograph=False)
    def inspect(theta, direction):
        value, score, trace, status = canonical_value_and_analytical_score(
            _single_cloud_model(model, theta, direction), theta, **inputs,
            **controls, with_score=True, return_trace=True, return_status=True)
        qdiag = tf.exp(2 * theta[0]) * q_diagonal
        rows = []
        for s in trace:
            w = s["posterior_weights"]
            cloud = s["children"]
            mean = tf.reduce_sum(w[:, None] * cloud, axis=0)
            centered = cloud - mean
            covariance = tf.einsum("n,ni,nj->ij", w, centered, centered)
            reset = s["states_after_reset"]
            reset_mean = tf.reduce_mean(reset, axis=0)
            rc = reset - reset_mean
            reset_cov = tf.einsum("ni,nj->ij", rc, rc) / tf.cast(tf.shape(reset)[0], tf.float64)
            displacement = tf.reduce_sum(
                tf.square(cloud - s["pre_flow"]) / qdiag[None], axis=1)
            rows.append(dict(
                log_increment=tf.reduce_logsumexp(s["posterior_logits"]),
                score_increment=tf.reduce_sum(w * s["d_posterior_logits"]),
                ess=1 / tf.reduce_sum(tf.square(w)),
                maximum_weight=tf.reduce_max(w),
                particles_for_99_percent_mass=1 + tf.reduce_sum(tf.cast(
                    tf.cumsum(tf.sort(w, direction="DESCENDING")) < .99, tf.int32)),
                prior_log_normalizer=s["prior_observation_log_normalizer"],
                prior_ess=1 / tf.reduce_sum(tf.square(s["prior_observation_weights"])),
                ukf_variance_over_process_variance=tf.reduce_mean(
                    tf.linalg.diag_part(s["predicted_covariances"]), axis=0) / qdiag,
                flow_displacement_process_sd_rms=tf.sqrt(tf.reduce_mean(displacement)),
                weighted_flow_displacement_process_sd_rms=tf.sqrt(tf.reduce_sum(w * displacement)),
                weighted_mean=mean, weighted_covariance=covariance,
                reset_mean=reset_mean, reset_covariance=reset_cov,
                reset_mean_error_in_filter_sd=tf.reduce_max(
                    tf.abs(reset_mean - mean) / tf.sqrt(tf.linalg.diag_part(covariance))),
                reset_relative_covariance_error=tf.linalg.norm(reset_cov - covariance)
                    / tf.linalg.norm(covariance),
                program_valid=s["program_valid"]))
        return dict(value=value, score=score[0], status=status, steps=rows)

    # Same model law, with the exact rounded A,Q,R of the measured LEDH route.
    fixture = m.fixture_from_data(tf, common["data"])
    for name in ("A", "Q", "R"):
        setattr(fixture, name, tf.constant(coefficients[name], tf.float64))
    quadratures = {n: quadrature(fixture, n) for n in (512, 1024)}
    report = dict(schema="bayesfilter.m13.likelihood_gap_trace.v1",
        question="Why is the frozen LEDH likelihood score far from the acceptance score?",
        plan=PLAN, environment=environment, fp64_reference_exception=True,
        input_manifest_sha256=m.sha(PREVIOUS / "inputs.json"),
        checked_source_hashes=checked_sources, rows=[], status="RUNNING",
        new_hmc_transitions=0, statistical_ranking_supported=False)
    m.write(HERE / "report.json", report)
    for i, probe in enumerate(common["probes"]):
        theta = tf.constant(probe["theta"], tf.float64)
        traces = [m.serial(inspect(theta, direction)) for direction in tf.unstack(tf.eye(2, dtype=tf.float64))]
        base = prior["base"][i // 4]
        value_expected = base["value"][i % 4]
        score_expected = base["score"][i % 4]
        value_error = max(abs(row["value"] - value_expected) for row in traces)
        score_error = max(abs(row["score"] - score_expected[j]) for j, row in enumerate(traces))
        parity = (value_error <= 1e-7 * (1 + abs(value_expected))
                  and all(abs(row["score"] - score_expected[j]) <= 1e-7 * (1 + abs(score_expected[j]))
                          for j, row in enumerate(traces)))
        assert parity, (probe["label"], value_error, score_error)
        assert all(row["status"]["value_valid"] and row["status"]["force_valid"] for row in traces)
        t1 = []
        for n, evaluate in quadratures.items():
            central = m.serial(evaluate(theta))
            assert central["valid"], (n, probe["label"])
            derivatives = {}
            for h in (3e-4, 1e-4):
                scores = []
                for j in range(2):
                    d = tf.one_hot(j, 2, dtype=tf.float64) * h
                    plus, minus = evaluate(theta + d), evaluate(theta - d)
                    assert bool(plus["valid"]) and bool(minus["valid"])
                    scores.append(float((plus["value"] - minus["value"]) / (2 * h)))
                derivatives[str(h)] = scores
            t1.append(dict(intervals=n, center=central, score=derivatives))
        row = dict(probe=probe, value=value_expected, score=score_expected,
                   trace_value_error=value_error, trace_score_error=score_error,
                   parity_passed=parity, directions=traces, first_observation_quadrature=t1)
        report["rows"].append(row)
        report["wall_seconds"] = time.monotonic() - started
        m.write(HERE / "report.json", report)
        print(probe["label"], "parity", parity,
              "min_ESS", min(s["ess"] for s in traces[0]["steps"]),
              "T1_LED H", traces[0]["steps"][0]["log_increment"],
              "T1_quadrature", t1[-1]["center"]["value"], flush=True)
    for relative, expected in checked_sources.items():
        assert m.sha(PREVIOUS / "source_after" / relative) == expected
    report.update(status="COMPLETED", wall_seconds=time.monotonic() - started,
                  gpu_memory=tf.config.experimental.get_memory_info("GPU:0"))
    m.write(HERE / "report.json", report)

def supervisor():
    assert not (HERE / "run-manifest.json").exists(), "never overwrite evidence"
    command = [m.PYTHON, str(pathlib.Path(__file__).resolve()), "--worker"]
    manifest = dict(plan=PLAN, command=command, cwd=str(ROOT),
        git_commit=m.git("rev-parse", "HEAD").decode().strip(),
        source_snapshot=str(PREVIOUS / "source_after"),
        conda_environment="tftwogpu", executable=m.PYTHON,
        hardware="GPU/XLA FP64 diagnostic; FP32 remains runtime default",
        data="data1704/T20; exact serialized inputs from previous measurement",
        seeds="same frozen designs and observations as inputs.json",
        timeout_seconds=600, previous_budget_seconds=1921.8366,
        total_budget_seconds=3600, started_utc=m.stamp(),
        script_sha256=m.sha(pathlib.Path(__file__)),
        plan_sha256=m.sha(ROOT / PLAN), log=str(HERE / "worker.log"),
        report=str(HERE / "report.json"), result=str(HERE / "result.md"),
        status="RUNNING")
    m.write(HERE / "run-manifest.json", manifest)
    start = time.monotonic()
    with (HERE / "worker.log").open("x") as log:
        try:
            proc = subprocess.run(command, cwd=ROOT, stdout=log,
                                  stderr=subprocess.STDOUT, timeout=600)
            exit_code = proc.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
    manifest.update(exit_code=exit_code, wall_seconds=time.monotonic() - start,
                    completed_utc=m.stamp(),
                    status="COMPLETED" if exit_code == 0 else "FAILED")
    manifest["remaining_budget_seconds"] = 3600 - 1921.8366 - manifest["wall_seconds"]
    if (HERE / "report.json").exists():
        manifest["report_sha256"] = m.sha(HERE / "report.json")
    manifest["log_sha256"] = m.sha(HERE / "worker.log")
    m.write(HERE / "run-manifest.json", manifest)
    print(json.dumps({k:manifest[k] for k in ("exit_code","wall_seconds","remaining_budget_seconds")}))
    sys.exit(exit_code)

if __name__ == "__main__":
    worker() if "--worker" in sys.argv else supervisor()
