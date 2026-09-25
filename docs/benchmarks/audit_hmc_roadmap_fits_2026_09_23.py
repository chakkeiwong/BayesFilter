"""Audit saved M32/M34 outcomes; no sampling and no statistical promotion."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--root", type=Path, required=True)
parser.add_argument("--phase", type=int, choices=(32, 34), required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
from bayesfilter.testing.inference_validation.storage import read_json, read_tensor
from bayesfilter.testing.inference_validation.designs import ValidationDesign
from docs.benchmarks.audit_hmc_m27_2026_09_23 import audit_runtime
from docs.benchmarks.audit_hmc_m30_2026_09_23 import normalize, numerical_rows
from docs.benchmarks.audit_hmc_m29_2026_09_23 import normalized_tuning
from tests.inference_validation.test_reuse_multimodel import compare_pair
import tensorflow as tf


def check_attempt(name):
    attempt = args.root / f"m{args.phase}-r1" / name
    outer = read_json(attempt / "execution.json")
    assert outer["exit_code"] == 0
    manifest = read_json(attempt / "fit/manifest.json")
    assert manifest["design_identity"] == ValidationDesign.from_payload(manifest["design"]).identity
    audit_runtime(manifest["runtime"], "gpu")
    assert read_json(attempt / "fit/exit.json")["result_present"]
    root = attempt / "fit/replication-0000"
    return root, manifest


if args.phase == 32:
    rows = []
    for target in ("rotated_gaussian", "dirichlet", "residual"):
        a, am = check_attempt(target+"-static-gpu-r1")
        b, bm = check_attempt(target+"-dynamic-gpu-r1")
        assert am["source"] == bm["source"]
        assert am["design_identity"] == bm["design_identity"]
        rows.append({"target": target, **compare_pair(a, b)})
    result = {"phase": 32, "rows": rows, "interpretation": "GPU/XLA numerical parity only"}
else:
    rows, pipelines = [], []
    for arm in ("baseline", "noop", "quarter", "half"):
        root, manifest = check_attempt(arm+"-dynamic-gpu-r1")
        pipeline = read_json(root / "pipeline.json")
        pipelines.append((root, pipeline))
        record = read_json(root.parent / "result.json")
        rows.append({"arm": arm, "verified_count": len(pipeline["verified_candidate_ids"]),
                     "endpoint": record["endpoint"], "design_identity": manifest["design_identity"]})
    aroot, a = pipelines[0]
    broot, b = pipelines[1]
    assert a["verified_candidate_ids"] and a["verified_candidate_ids"] == b["verified_candidate_ids"]
    assert a["selection"] == b["selection"]
    numerical, tuning = [], []
    from bayesfilter.testing.inference_validation.designs import digest
    for root, pipeline in pipelines[:2]:
        evidence = [read_json(p) for p in (root / "tuning/numerical_evidence").glob("*.json")]
        rows_by_work, hashes = numerical_rows({digest(e): e for e in evidence})
        numerical.append(rows_by_work)
        tuning.append(normalized_tuning(read_json(pipeline["tuning_path"]), hashes))
    assert numerical[0] == numerical[1] and tuning[0] == tuning[1]
    for am, bm in zip(a["members"], b["members"]):
        assert am["candidate_id"] == bm["candidate_id"] and am["status"] == bm["status"]
        if am["status"] == "assessed":
            for key in ("draws_path", "warmup_path"):
                tf.debugging.assert_equal(read_tensor(am[key]), read_tensor(bm[key]))
            assert normalize(am["posterior"], expected_l=am["L"], fit_root=aroot) == normalize(
                bm["posterior"], expected_l=bm["L"], fit_root=broot)
    result = {"phase": 34, "rows": rows, "noop_exact_parity": True,
              "interpretation": "paired activation only; no independent-replication or achieved-power claim"}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as out:
    json.dump(result, out, indent=2, allow_nan=False)
print(json.dumps(result))
