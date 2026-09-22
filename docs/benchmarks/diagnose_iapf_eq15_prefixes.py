"""Post-run independent-reference diagnostics; never imported by a runtime lane."""
import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "docs/plans/artifacts/iapf-r-24hour-campaign-20260921-01"
NEW = ROOT / "docs/plans/artifacts/iapf-r-equation15-resolution-20260921-01"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    destination = NEW / "analysis-v1"
    destination.mkdir(exist_ok=False)
    source = OLD / "phase03-first-study-1000/analysis-v1/summary.json"
    old = json.loads(source.read_text())
    cell = next(c for c in old["cells"] if c["dimension"] == 80)
    methods = set(cell["methods"])
    per_rep, per_time, events = defaultdict(dict), defaultdict(dict), defaultdict(list)
    situations, parents = {}, []
    for parent in cell["validity"]["parents"]:
        folder = OLD / parent["attempt"] / "results"
        path = folder / "prefixes.csv"
        checksum = digest(path)
        assert checksum == parent["files"]["prefixes.csv"]
        rr = rows(path)
        assert len(rr) == 500
        identities = set()
        for r in rr:
            rep, time = int(r["replication"]), int(r["time"])
            method, situation = r["method"], r["situation"]
            assert method in methods and 3001 <= rep <= 4000 and 1 <= time <= 100
            assert (rep, method, time) not in identities
            identities.add((rep, method, time))
            assert situations.setdefault(time, situation) == situation
            loss = math.expm1(float(r["log_prefix_error"])) ** 2
            assert math.isfinite(loss)
            key = (method, situation)
            per_rep[key].setdefault(rep, []).append(loss)
            per_time[key].setdefault(time, []).append(loss)
            events[key].append((loss, rep, time, float(r["log_prefix_error"])))
        parents.append(dict(attempt=parent["attempt"], prefix_sha256=checksum))
    assert len(parents) == 1000
    result = dict(question="Concentration of the preserved d80 conditional prefix error",
        classification="post-run explanatory diagnostic; all observations retained",
        prior_summary=str(source.relative_to(ROOT)), prior_summary_sha256=digest(source),
        diagnostic_source_sha256=digest(Path(__file__)), parents=parents, cells=[])
    for (method, situation), replicas in sorted(per_rep.items()):
        assert set(replicas) == set(range(3001, 4001))
        ntime = sum(s == situation for s in situations.values())
        assert all(len(v) == ntime for v in replicas.values())
        losses = {r: statistics.mean(v) for r, v in replicas.items()}
        ordered = sorted(losses.items(), key=lambda pair: pair[1], reverse=True)
        total = math.fsum(losses.values())
        reference = next((r["mean_mse"] for r in cell["conditional_comparisons"]
            if r["method"] == method and r["situation"] == situation), None)
        if reference is None:
            reference = next(r["comparator_mean_mse"] for r in cell["conditional_comparisons"]
                if r["comparator"] == method and r["situation"] == situation)
        mean = statistics.mean(losses.values())
        assert math.isclose(mean, reference, rel_tol=1e-12, abs_tol=1e-14)
        all_events = sorted(events[method, situation], reverse=True)
        time_totals = sorted(((t, math.fsum(v)) for t, v in per_time[method, situation].items()),
            key=lambda pair: pair[1], reverse=True)
        event_total = math.fsum(v for _, v in time_totals)
        result["cells"].append(dict(method=method, situation=situation, replicas=1000,
            time_points=ntime, mean_mse=mean, reproduced_prior_mean=True,
            median_replica_mse=statistics.median(losses.values()),
            replicas_mse_above_one=sum(v > 1 for v in losses.values()),
            replica_contribution_fraction={str(n): math.fsum(v for _, v in ordered[:n])/total
                for n in (1, 5, 10, 50)},
            largest_replicas=[dict(replication=r, mse=v) for r, v in ordered[:10]],
            largest_times=[dict(time=t, loss_fraction=v/event_total) for t, v in time_totals[:10]],
            largest_events=[dict(loss=v, replication=r, time=t, log_prefix_error=e,
                loss_fraction=v/event_total) for v, r, t, e in all_events[:10]]))

    oracle_path = NEW / "runs/exact-oracle-d80/results"
    # Same saved data, checked numerically despite potentially different CSV formatting.
    prior_path = OLD / cell["validity"]["parents"][0]["attempt"] / "results"
    for name in ("observations.csv", "kalman.csv"):
        before, after = rows(prior_path/name), rows(oracle_path/name)
        assert len(before) == len(after)
        for x, y in zip(before, after):
            assert x.keys() == y.keys()
            assert all(float(x[k]) == float(y[k]) for k in x)
    theory, observed = rows(oracle_path/"oracle-theory.csv"), rows(oracle_path/"oracle-prefixes.csv")
    assert len(theory) == 100 and len(observed) == 1600
    assert {(int(r["replication"]), int(r["time"])) for r in observed} == {
        (rep, t) for rep in range(1, 17) for t in range(1, 101)}
    oracle = dict(data_equal_to_preserved_d80=True, particles=1000, replicas=16,
        terminal_log_error_max=max(abs(float(r["terminal_error"])) for r in observed),
        prefix_identity_error_max=max(abs(float(r["identity_error"])) for r in observed),
        second_moment_margin_min=min(float(r["second_moment_margin"]) for r in theory),
        files={f.name: digest(f) for f in oracle_path.glob("*.csv")}, situations={})
    for situation in ("ordinary", "large_innovation"):
        expected = [float(r["expected_prefix_MSE"]) for r in theory if r["situation"] == situation]
        empirical = [math.expm1(float(r["log_prefix_error"]))**2 for r in observed
            if situations[int(r["time"])] == situation]
        oracle["situations"][situation] = dict(expected_mse=statistics.mean(expected),
            observed_mse=statistics.mean(empirical), maximum_time_expected_mse=max(expected),
            interpretation="Exact variance, small empirical sample; rare tails make this no ranking")
    result["oracle"] = oracle
    (destination/"prefix-concentration.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(prior_means_reproduced=True, parent_hashes_checked=len(parents),
        oracle=oracle, output=str(destination/"prefix-concentration.json")), indent=2))


if __name__ == "__main__":
    main()
