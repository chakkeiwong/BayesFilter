"""Independent standard-library audit of the completed precision diagnostic."""
from pathlib import Path
import hashlib
import json
import math
import statistics

root = Path(__file__).resolve().parent
load = lambda p: json.loads(p.read_text())
read_rows = lambda p: [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
execution = load(root / "execution.json")
report = load(root / "report.json")
assert execution["status"] == "completed" and len(execution["attempts"]) == 9
assert all(a["exit_code"] == 0 for a in execution["attempts"])
assert execution["wall_seconds"] < execution["budget_seconds"]
all_rows, summaries = {}, []
for n in (1008, 2016, 4032):
    controls, data_hashes = None, None
    for mode in ("fp64", "fp32", "fp32_tf32"):
        directory = root / f"N{n}-{mode}"
        m = load(directory / "manifest.json")
        assert m["source_sha256"] == execution["source_sha256"]
        assert m["graph_trace_count"] == 1
        assert m["memory_policy"]["all_physical_devices_memory_growth"]
        assert m["memory_policy"]["configured_before_logical_device_initialization"]
        assert m["scope"]["dtype"] == ("float64" if mode == "fp64" else "float32")
        assert m["scope"]["tf32"] == (mode == "fp32_tf32")
        assert m["scope"]["jit_compile"] and not m["scope"]["cpu_only"]
        assert m["scope"]["chunk"] == min(n, 2016)
        assert m["cuda_visible_devices"] == "GPU-68251639-fe82-8f81-3ccc-2953c32e805b"
        if controls is None:
            controls, data_hashes = m["selected_controls"], m["data_sha256"]
        assert m["selected_controls"] == controls and m["data_sha256"] == data_hashes
        r = read_rows(directory / "measurements.jsonl")
        pre = read_rows(directory / "preflight.jsonl")
        by_key = {(x["data_seed"], x["point"], x["seed"]): x for x in r}
        assert len(by_key) == len(r)
        if m["status"] == "completed":
            expected = {(ds, pt, 31001 + 100*i + k) for i, ds in enumerate((1704, 18020, 18021))
                        for pt in (0, 1) for k in range(8)}
            assert set(by_key) == expected
        elif m["status"] == "failed_preflight":
            assert not r and any(not x["valid"] for x in pre)
        else:
            raise AssertionError("Worker did not reach a terminal status")
        for x in r:
            if x["valid"]:
                assert x["steps_completed"] == [20, 20]
                assert all(math.isfinite(v) for v in (x["value"], *x["score"]))
                assert x["primal_direction_difference"] <= (1e-9 if mode == "fp64" else 1e-4)
        all_rows[n, mode] = by_key
        failures = []
        for x in pre+r:
            if not x["valid"]:
                trace = dict(zip(x["trace_fields"], x["trace"][x["steps_completed"][0]-1]))
                failures.append(dict(data_seed=x["data_seed"], point=x["point"], steps=x["steps_completed"],
                                     covariance_relative_error=trace["reset_covariance_relative_error"],
                                     covariance_guard=100*2**-23*4 if mode != "fp64" else 100*2**-52*4))
        summaries.append(dict(particles=n, mode=mode, valid=sum(x["valid"] for x in r), count=len(r),
                              failures=failures, cold_warm_repeat_value_difference=abs(pre[0]["value"]-pre[2]["value"]),
                              cold_warm_repeat_score_difference=math.dist(pre[0]["score"],pre[2]["score"])))

checks = []
for c in report["cells"]:
    if not c["complete"]:
        continue
    n, mode, ds, point = (c[k] for k in ("particles", "mode", "data_seed", "point"))
    fp64, lower = all_rows[n, "fp64"], all_rows[n, mode]
    keys = [k for k in fp64 if k[:2] == (ds, point)]
    assert len(keys) == 8 and all(fp64[k]["input_sha256"] == lower[k]["input_sha256"] for k in keys)
    dscores = [math.dist(fp64[k]["score"], lower[k]["score"]) for k in keys]
    dvalues = [abs(fp64[k]["value"]-lower[k]["value"]) for k in keys]
    sr = math.sqrt(sum(v*v for v in dscores)/8)
    vr = math.sqrt(sum(v*v for v in dvalues)/8)
    assert math.isclose(c["score_difference_rms"], sr, rel_tol=1e-12)
    assert math.isclose(c["value_difference_rms"], vr, rel_tol=1e-12)
    assert math.isclose(c["score_change_ratio"], sr/c["fp64_n1008_n2016_score_change_rms"], rel_tol=1e-12)
    assert math.isclose(c["value_change_ratio"], vr/c["fp64_n1008_n2016_value_change_rms"], rel_tol=1e-12)
    checks.append(dict(particles=n, mode=mode, data_seed=ds, point=point, exact_input_hash_match=True))

for row in report["timings"]:
    if row["complete"]:
        values = all_rows[row["particles"], row["mode"]].values()
        assert len(values) == 48 and all(x["valid"] for x in values)
        assert row["median_seconds"] == statistics.median(x["wall_seconds"] for x in values)
    else:
        assert row["median_seconds"] is None

result = dict(passed=True, worker_summaries=summaries, paired_cell_checks=checks,
              report_sha256=hashlib.sha256((root/"report.json").read_bytes()).hexdigest(),
              checks="source/data/control/input identity; GPU growth/XLA/TF32; complete seeds; valid steps; independent arithmetic; no early-exit timing")
(root / "terminal-review.json").write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps(dict(passed=True, complete_paired_cells=len(checks), summaries=summaries)))
