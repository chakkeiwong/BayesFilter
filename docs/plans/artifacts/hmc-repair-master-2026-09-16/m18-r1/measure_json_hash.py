"""Diagnostic same-payload hash parity and alternating host timing."""
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
sys.path.insert(0, str(REPO))
from bayesfilter.inference.hmc_candidate_set_tuning import _sha256


def native_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def main():
    paths = [Path(p) for p in json.loads((ROOT / "host-hash-inputs.json").read_text())]
    rows = [json.loads(p.read_text()) for p in paths]
    payloads = [r["execution"] if p.name == "execution_spec.json" else r for p, r in zip(paths, rows)]
    expected = [_sha256(x) for x in payloads]
    observed = [native_hash(x) for x in payloads]
    if expected != observed:
        raise ValueError("JSON-native normalization differs on saved records")
    times = {"generic": [], "native": []}
    for repeat in range(5):
        for name, fn in ((("generic", _sha256), ("native", native_hash)) if repeat % 2 == 0 else
                         (("native", native_hash), ("generic", _sha256))):
            started = time.perf_counter()
            hashes = [fn(x) for x in payloads]
            times[name].append(time.perf_counter() - started)
            assert hashes == expected
    result = {"record_count": len(paths), "input_bytes": sum(p.stat().st_size for p in paths),
        "exact_hash_parity": True, "input_hashes": dict(zip(map(str, paths), expected)),
        "seconds": times, "median_seconds": {k: statistics.median(v) for k, v in times.items()},
        "interpretation": "descriptive same-payload host costs; no end-to-end or sampler speed claim",
        "source_hash": hashlib.sha256((REPO / "bayesfilter/inference/hmc_candidate_set_tuning.py").read_bytes()).hexdigest()}
    with (ROOT / "host-hash-comparison.json").open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
    print(json.dumps({k: v for k, v in result.items() if k != "input_hashes"}, indent=2))


if __name__ == "__main__":
    main()
