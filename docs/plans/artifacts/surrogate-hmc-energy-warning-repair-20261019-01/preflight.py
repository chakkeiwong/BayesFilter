"""Archive validation; no new HMC transitions or inference claims."""
import os
import sys
gpu = "--gpu" in sys.argv
os.environ["CUDA_VISIBLE_DEVICES"] = ("GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a" if gpu else "-1")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))
from dataclasses import replace
import json
import tensorflow as tf
if gpu:
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
else:
    memory = {"cpu_only": True}
tf.config.experimental.enable_tensor_float_32_execution(False)
from bayesfilter.testing import surrogate_hmc_five_model as runtime
from bayesfilter.inference import hmc_tensorflow_resume as resume
from bayesfilter.inference import hmc_tensorflow_verified as verified
from scripts import run_surrogate_hmc_five_model_master as master
from scripts import run_surrogate_hmc_48h_queue_repair as repair

selection = (repair.queue.RANGE_PREPARATION /
             "range_bearing/filter-tuning/range_bearing/filter-selection.json")
case = runtime.configure_prior(runtime.load_case("range_bearing"), master.LITERATURE_PRIOR)
case = master.restore_selected_scope(runtime, case, selection)
parts = runtime.bindings(case)
parent = repair.energy_resume_manifest(repair.queue.OUTPUT)
payload = json.loads(parent.read_text())
cfg = replace(verified.VerifiedTensorFlowHMCConfig.from_payload(payload["config"]),
              resume_manifest=str(parent))
path, saved, nomination, migration = resume.prepare_resume(
    adapter=parts["adapter"], config=cfg, runner_binding=parts["binding"])
replay = resume.replay_then_continue(config=cfg, nomination=nomination,
    binding=parts["binding"], source_root=path.parent, records=saved["sequence"]["chunks"])
assert replay["status"] == "BUDGET_EXHAUSTED"
assert replay["counts"] == saved["sequence"]["counts"]
assert replay["used_seeds"] == saved["sequence"]["used_seeds"]
assert all(all(event["checks"].values()) for event in replay["events"])
assert all("energy_valid" not in event["checks"] for event in replay["events"])
sir_parent = repair.queue.resume_manifest("sir_d18")
sir_migration = resume.validate_source_migration(json.loads(sir_parent.read_text())["source_hashes"])
result = dict(status="PASS", cpu_only=not gpu, memory_policy=memory, new_transitions=0,
    source_manifest=str(path), original_status=saved["sequence"]["status"],
    new_replay_status=replay["status"], counts=replay["counts"],
    chunk_count=len(saved["sequence"]["chunks"]),
    energy_warnings=[event.get("diagnostics", {}) for event in replay["events"]],
    migration=migration, sir_migration=sir_migration["kind"],
    frozen_config_unchanged=json.loads(json.dumps(replace(cfg, resume_manifest=None).payload()))==saved["config"],
    checked="actual adapter/binding/scope identity; nomination and tensor checksums; all archived states/seeds/counts; aggregate equality; original stop replay; warning-only replay")
out = Path(__file__).with_name("preflight-result.json")
out.write_text(json.dumps(result, indent=2)+"\n")
print(json.dumps({key:value for key,value in result.items() if key not in ("migration", "energy_warnings")}))
