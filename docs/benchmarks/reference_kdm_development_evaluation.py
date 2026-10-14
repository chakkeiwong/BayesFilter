"""CPU-only independent author evaluation of completed development checkpoints.

The TensorFlow checkpoint is converted only for this diagnostic comparator.
This is not a BayesFilter runtime, qualification, or training implementation.
"""
import argparse
import ast
import copy
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
AUTHOR = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
PIN = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
PLAN = ROOT / "docs/plans/controlled-smc-kdm-development-evaluation-20261008.md"
RESULT = "docs/plans/controlled-smc-kdm-development-evaluation-result-20261008.md"
SOURCE_RUN = BASE / "kdm-complete-training-20261007-03"
TF_RUN = BASE / "kdm-complete-training-20261007-04"
SOURCE_PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu2/bin/python"
DATA_SEED = 202610081
FILTER_SEEDS = (202610082, 202610083)
ROWS, BATCH, HORIZON, PARTICLES = 256, 64, 150, 25
CAP, ROUTINE = 1800., 120.

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_completion_valid(manifest, diagnostic, audit):
    """Accept audited historical recovery without inventing an exit code."""
    if manifest["status"] != "COMPLETED":
        return False
    if manifest["returncode"] == 0:
        return True
    if not (manifest["returncode"] is None and
            manifest.get("process_returncode_observed") is False and
            manifest.get("completion_basis") == "recovered_complete_artifacts_exit_unobserved" and
            audit.get("completion_recovery", {}).get("validated") is True):
        return False
    proof = manifest["completion_recovery"]
    archived = Path(proof["archived_manifest"])
    assert sha(archived) == proof["archived_manifest_sha256"]
    old = json.loads(archived.read_text())
    assert old["status"] == "RUNNING" and "returncode" not in old
    assert old["hashes"] == manifest["hashes"] and old["child_pid"] == manifest["child_pid"]
    assert proof["supervisor_tool_exit_code"] == 143 and proof["processes_absent_at_recovery"] is True
    assert sha(diagnostic/"result.json") == proof["result_sha256"]
    assert sha(diagnostic/"progress.json") == proof["terminal_progress_sha256"]
    assert json.loads((diagnostic/"progress.json").read_text()) == {"terminal": True, "status": "PASS"}
    result = json.loads((diagnostic/"result.json").read_text())
    assert result["status"] == "PASS" and result["source_training_protocol_complete"]
    return True

def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)

def verify(path):
    path = Path(path)
    relative = path.relative_to(AUTHOR)
    original = subprocess.check_output(["git", "-C", str(AUTHOR), "show", PIN + ":" + str(relative)])
    if path.read_bytes() != original:
        raise ValueError("Changed author source: " + str(relative))
    return original

def worker(out, source_run=None, source_audit=None, tf_run=None, tf_audit=None):
    if (source_run is None) != (source_audit is None):
        raise ValueError('Source override requires its independent terminal audit')
    if (tf_run is None) != (tf_audit is None):
        raise ValueError('TF override requires its independent terminal audit')
    active_source = SOURCE_RUN if source_run is None else source_run.resolve()
    active_tf = TF_RUN if tf_run is None else tf_run.resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["OMP_NUM_THREADS"] = "1"
    if not sys.flags.isolated:
        raise RuntimeError("The independent author worker must use isolated Python")
    started = time.monotonic()
    out.mkdir()
    helper_path = ROOT / "docs/benchmarks/reference_kdm_neural_exact_replay.py"
    spec = importlib.util.spec_from_file_location("source_evaluation_helpers", helper_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    torch = ref.torch
    import numpy as np  # Explicit independent author data-generation exception.
    import yaml
    from loss_functions.loss_functions import create_loss_function
    from datasets.bearings_only_dataset import BearingsOnlyDataset
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    subprocess.run(["git", "-C", str(AUTHOR), "diff", "--quiet", PIN, "--", "src", "packages"], check=True)
    source_files = [
        AUTHOR / "src/evaluations/full_sequence_evaluations/full_sequence_evaluation_base.py",
        AUTHOR / "src/loss_functions/loss_functions.py",
        AUTHOR / "src/datasets/bearings_only_dataset.py",
        AUTHOR / "src/models/particle_transformer.py",
        AUTHOR / "src/models/kde_particle_filter/kde_particle_filter.py",
        AUTHOR / "src/problems/bearings_only/bearings_only_problem.py",
    ]
    for p in source_files:
        verify(p)
    eval_path = source_files[0]
    tree = ast.parse(eval_path.read_text())
    metric_nodes = [node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name in ("EvaluationMetric", "NonSweepEvaluationMetric")]
    namespace = dict(torch=torch, create_loss_function=create_loss_function, os=os,
                     tqdm=lambda values, **kwargs: values)
    exec(compile(ast.Module(body=metric_nodes, type_ignores=[]), str(eval_path), "exec"), namespace)
    eval_class = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                      and node.name == "FullSequenceEvaluationBase")
    forward_node = next(node for node in eval_class.body if isinstance(node, ast.FunctionDef)
                        and node.name == "run_the_model")
    exec(compile(ast.Module(body=[forward_node], type_ignores=[]), str(eval_path), "exec"), namespace)
    run_the_model = namespace["run_the_model"]
    Metric = namespace["NonSweepEvaluationMetric"]
    transformer_class = ref.source_member(source_files[-1], "BearingsOnlyObservationTransformer")
    experiment = ref.EXPERIMENT
    eval_config_path = experiment / "configs/common_parameters_evaluation.yaml"
    data_config_path = experiment / "configs/dataset_params_evaluation.yaml"
    eval_config = yaml.safe_load(verify(eval_config_path))
    data_config = yaml.safe_load(verify(data_config_path))["dataset_params"]
    assert data_config["subsequence_length"] == HORIZON
    assert "sparse_ground_truth_keep_modulo" not in data_config
    assert eval_config["evaluation_params"]["number_of_particles"] == PARTICLES
    metric_config = eval_config["evaluation_params"]["quantitative_evaluation_params"]["metrics"]
    assert set(metric_config) == {"nll", "rmse"}
    data_config["dataset_sizes"] = {"evaluation": ROWS}
    data_config["dataset_saves"] = {"evaluation": str(out / "evaluation-dataset.pt")}
    np.random.seed(DATA_SEED)
    random.seed(DATA_SEED)
    torch.manual_seed(DATA_SEED)
    dataset = BearingsOnlyDataset(data_config, "evaluation")
    assert len(dataset) == ROWS and dataset.get_subsequence_length() == HORIZON
    data_rows = [dataset[i] for i in range(ROWS)]
    assert all("ground_truth_mask" not in row for row in data_rows)
    data = {name: torch.stack([row[name] for row in data_rows]) for name in ("states", "observations")}
    assert tuple(data["states"].shape) == (ROWS, HORIZON, 3)
    assert tuple(data["observations"].shape) == (ROWS, HORIZON, 1)
    for value in data.values():
        assert bool(torch.isfinite(value).all())
    data_value_control = None
    if source_audit is not None:
        prior_root = BASE / "kdm-development-evaluation-20261008-03/diagnostic"
        prior_report = json.loads((prior_root/"result.json").read_text())
        prior_file = prior_root/"evaluation-dataset.pt"
        assert sha(prior_file) == prior_report["dataset_sha256"]
        prior_data = torch.load(prior_file,map_location="cpu",weights_only=False)
        prior_tensors = {"states":prior_data["car_states"],"observations":prior_data["sensor_obs"]}
        for name in data:
            assert data[name].dtype == prior_tensors[name].dtype
            assert data[name].shape == prior_tensors[name].shape
            assert torch.equal(data[name],prior_tensors[name])
        def tensor_digest(tensors):
            payload = {name:{"dtype":str(value.dtype),"shape":list(value.shape),
                             "values":value.tolist()} for name,value in tensors.items()}
            return hashlib.sha256(json.dumps(payload,sort_keys=True,allow_nan=False).encode()).hexdigest()
        data_value_control = dict(exact_tensor_equality=True,
            reference_path=str(prior_file),reference_file_sha256=sha(prior_file),
            tensor_sha256=tensor_digest(data),reference_tensor_sha256=tensor_digest(prior_tensors),
            rationale="Author dataset pickle includes the versioned save path; compare actual dtype, shape and values")
    true_bearing = torch.atan2(data["states"][..., 1], data["states"][..., 0] + 5.)
    residual = data["observations"][..., 0] - true_bearing
    residual = torch.atan2(torch.sin(residual), torch.cos(residual)).abs()
    masks = {
        "first_17": torch.arange(HORIZON)[None, :].expand(ROWS, -1) < 17,
        "after_17": torch.arange(HORIZON)[None, :].expand(ROWS, -1) >= 17,
        "bearing_residual_le_0p5": residual <= .5,
        "bearing_residual_gt_0p5": residual > .5,
        "angle_seam_0p1": data["states"][..., 2].abs() >= math.pi - .1,
    }
    write(out / "data-design.json", dict(seed=DATA_SEED, rows=ROWS, horizon=HORIZON,
          data_config=data_config, source_dataset_sha256=sha(out/"evaluation-dataset.pt"),
          situation_counts={name:int(mask.sum()) for name,mask in masks.items()},
          role="DEVELOPMENT_ONLY_NEVER_UNTOUCHED_QUALIFICATION"))
    reviewed = json.loads((BASE/"kdm-complete-training-reviewed-20261007.json").read_text())
    assert reviewed["decision"]["scope"] == "COMPLETE_SOURCE_AND_TF_DEVELOPMENT_TRAINING"
    for run in reviewed["training_runs"]:
        for key in ("manifest", "result", "audit"):
            assert sha(ROOT/run[key]) == run[key+"_sha256"]
    if source_audit is not None:
        audit = json.loads(source_audit.read_text())
        active_manifest = active_source / "manifest.json"
        audited_manifest = json.loads(active_manifest.read_text())
        assert Path(audit["run"]).resolve() == active_source
        assert audit["verdict"] == "PASS" and not audit["failures"]
        assert sha(active_manifest) == audit["run_manifest_sha256"]
        assert source_completion_valid(audited_manifest, active_source/"diagnostic", audit)
        assert audited_manifest["result_sha256"] == sha(active_source/"diagnostic/result.json")
    if tf_audit is not None:
        audit = json.loads(tf_audit.read_text())
        active_manifest = json.loads((active_tf/"manifest.json").read_text())
        assert Path(audit["run"]).resolve() == active_tf
        assert audit["verdict"] == "PASS" and not audit["failures"]
        assert sha(active_tf/"manifest.json") == audit["run_manifest_sha256"]
        assert active_manifest["status"] == "COMPLETED" and active_manifest["returncode"] == 0
        assert active_manifest["mode"] == "tf" and active_manifest["epoch_mode"] == "full"
        assert active_manifest["result_sha256"] == sha(active_tf/"diagnostic/result.json")
    source_result = json.loads((active_source/"diagnostic/result.json").read_text())
    configurations = json.loads((active_source/"diagnostic/configurations.json").read_text())
    stage_configs = {entry["name"]:entry for entry in configurations["executed_stages"]}
    stage_results = {entry["stage"]:entry for entry in source_result["stages"]}
    tf_result = json.loads((active_tf/"diagnostic/result.json").read_text())
    if tf_audit is not None:
        assert tf_result["status"] == "PASS" and tf_result["source_training_protocol_complete"]
        assert tf_result["checks_failed"] == 0
        # Bind each pair to its own constructors, including the first pilot.
        tf_reference = Path(active_manifest["reference"])
        assert tf_result["reference_sha256"] == sha(tf_reference)
        assert active_manifest["reference_sha256"] == sha(tf_reference)
        paired_reference = json.loads(tf_reference.read_text())
        assert paired_reference["source_pin"] == source_result["source_pin"] == PIN
        assert paired_reference["source_training_configs"] == source_result["source_training_configs"]
        for section in ("datasets", "fixtures"):
            assert set(paired_reference[section]) == set(source_result[section])
            for name, item in paired_reference[section].items():
                counterpart = source_result[section][name]
                assert item["sha256"] == counterpart["sha256"]
                assert sha(item["path"]) == item["sha256"]
                assert sha(counterpart["path"]) == counterpart["sha256"]
        proof = BASE/"stage-refresh-20261011/stage-refresh-01/verification/diagnostic/result.json"
        assert tf_result["verification"]["path"] == str(proof)
        assert tf_result["verification"]["sha256"] == sha(proof)
    tf_stages = {entry["stage"]:entry for entry in tf_result["completed_stages"]}
    checks = []
    def check(condition, label):
        if not bool(condition):
            raise ValueError(label)
        checks.append(label)
    def snapshot(model):
        return {name:value.detach().clone() for name,value in model.state_dict().items()}
    def owner(model, selected_dataset):
        return SimpleNamespace(model=model, device=torch.device("cpu"), evaluation_dataset=selected_dataset,
            number_of_particles=PARTICLES,
            problem=SimpleNamespace(observation_transformer=transformer_class()))
    def forward(model, part, selected_dataset=dataset):
        return run_the_model(owner(model, selected_dataset), part)
    def valid_outputs(outputs):
        check(len(outputs)==HORIZON, "evaluation horizon")
        for i,output in enumerate(outputs):
            check(all(torch.is_tensor(output.get(name)) for name in ("particles","particle_weights","bandwidths")), "required filter outputs")
            for name,value in output.items():
                if torch.is_tensor(value) and value.is_floating_point():
                    check(torch.isfinite(value).all(), "finite/"+name+"/"+str(i))
            for name in ("particle_weights", "resampling_particle_weights"):
                if output.get(name) is not None:
                    value = output[name]
                    check((value>=0).all() and torch.allclose(value.sum(-1),torch.ones(value.shape[0]),atol=1e-5,rtol=0),
                          "probabilities/"+name+"/"+str(i))
            for name in ("bandwidths", "resampling_bandwidths"):
                if output.get(name) is not None:
                    check((output[name]>0).all(), "bandwidths/"+name+"/"+str(i))
    variants = []
    provenance = []
    all_records = []
    for variant in ("mdpf", "am_dpf"):
        cfg = stage_configs[variant]
        torch.manual_seed(202610084)
        directory = "experiment0002_importance" if variant=="mdpf" else "experiment0003_importance_init"
        architecture_paths = [experiment/"configs/models.yaml", experiment/directory/"config.yaml"]
        architecture = {}
        for architecture_path in architecture_paths:
            architecture.update(yaml.safe_load(verify(architecture_path))["models"])
            source_files.append(architecture_path)
        # JSON snapshots stringify integer KDE dimension keys. Build from pinned
        # YAML and check its serialized identity against the training snapshot.
        check(json.loads(json.dumps(architecture))==cfg["architecture"],variant+"/source architecture identity")
        model = ref.KDEParticleFilter(cfg["configuration"]["model_params"], architecture)
        checkpoints = stage_results[variant]["checkpoints"]
        checkpoint = next(Path(path) for path in checkpoints if Path(path).name=="full_dpf_model_best.pt")
        check(sha(checkpoint)==checkpoints[str(checkpoint)], variant+"/source checkpoint hash")
        source_state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(source_state, strict=True)
        model.eval()
        tf_path = active_tf / ("diagnostic/"+variant+"-best.json")
        check(sha(tf_path)==tf_stages[variant]["best_checkpoint_sha256"],variant+"/TF checkpoint hash")
        tf_state = json.loads(tf_path.read_text())["model"]
        check(tf_state["variant"]==variant and tf_state["dtype"]=="float32",variant+"/TF checkpoint identity")
        tf_model = copy.deepcopy(model)
        parameters = dict(tf_model.named_parameters())
        check(set(parameters)==set(tf_state["parameters"]),variant+"/parameter names")
        with torch.no_grad():
            for name,p in parameters.items():
                value = torch.tensor(tf_state["parameters"][name],dtype=torch.float32)
                check(value.shape==p.shape and torch.isfinite(value).all(),variant+"/parameter shape and finite/"+name)
                p.copy_(value)
                check(p.tolist()==tf_state["parameters"][name],variant+"/exact parameter conversion/"+name)
        constants = dict(residual_scale=model.proposal_model.residual_scale_factor,
                         min_bandwidths=model.weighted_bandwidth_predictor.min_bandwidths)
        if model.resampling_weighted_bandwidth_predictor is not None:
            constants["resampling_min_bandwidths"] = model.resampling_weighted_bandwidth_predictor.min_bandwidths
        check(set(constants)==set(tf_state["constants"]),variant+"/constant names")
        for name,value in constants.items():
            check(torch.equal(value,torch.tensor(tf_state["constants"][name],dtype=value.dtype)),variant+"/constant/"+name)
        source_buffers = dict(model.named_buffers())
        for name,value in tf_model.named_buffers():
            check(torch.equal(value,source_buffers[name]),variant+"/source buffer preserved/"+name)
        # The identical conversion of source tensors must reproduce the original
        # source endpoint, not just a list of parameter shapes.
        roundtrip = copy.deepcopy(model)
        with torch.no_grad():
            for name,p in roundtrip.named_parameters():
                p.copy_(torch.tensor(dict(model.named_parameters())[name].tolist(),dtype=p.dtype))
        tiny = {name:value[:4,:17] for name,value in data.items()}
        tiny_dataset = SimpleNamespace(get_subsequence_length=lambda:17)
        torch.manual_seed(202610085)
        expected = forward(model,tiny,tiny_dataset)
        torch.manual_seed(202610085)
        actual = forward(roundtrip,tiny,tiny_dataset)
        for step,(left,right) in enumerate(zip(expected,actual)):
            for name,value in left.items():
                if torch.is_tensor(value):
                    check(torch.equal(value,right[name]),variant+"/conversion endpoint/"+str(step)+"/"+name)
        for metric_name,metric_params in metric_config.items():
            metrics=[Metric(metric_name,m,metric_params,str(out)) for m in (model,roundtrip)]
            for metric,outputs in zip(metrics,(expected,actual)):
                metric.compute_and_add_metric(outputs,tiny["states"],torch.arange(4),tiny_dataset)
            check(metrics[0].get_value()==metrics[1].get_value(),variant+"/conversion metric/"+metric_name)
        provenance.append(dict(variant=variant,source_checkpoint=str(checkpoint),source_sha256=sha(checkpoint),
                               tf_checkpoint=str(tf_path),tf_sha256=sha(tf_path)))
        for label,current in (("source_trained",model),("tf_trained",tf_model)):
            before = snapshot(current)
            for seed in FILTER_SEEDS:
                torch.manual_seed(seed)
                metrics = {name:Metric(name,current,params,str(out)) for name,params in metric_config.items()}
                tick=time.monotonic()
                for begin in range(0,ROWS,BATCH):
                    indices=torch.arange(begin,min(begin+BATCH,ROWS))
                    part={name:value[indices] for name,value in data.items()}
                    outputs=forward(current,part)
                    valid_outputs(outputs)
                    for metric in metrics.values():
                        metric.compute_and_add_metric(outputs,part["states"],indices,dataset)
                    write(out/"progress.json",dict(variant=variant,checkpoint=label,seed=seed,
                          trajectories_done=begin+len(indices),elapsed_seconds=time.monotonic()-started))
                check(all(torch.equal(value,before[name]) for name,value in current.state_dict().items()),
                      variant+"/"+label+"/evaluation model immutable")
                arrays={name:torch.cat(metric.sequence_level_metric_values) for name,metric in metrics.items()}
                values={name:metric.get_value() for name,metric in metrics.items()}
                check(all(math.isfinite(x) for x in values.values()),"finite aggregate metrics")
                check(all(tuple(x.shape)==(ROWS,HORIZON) for x in arrays.values()),"metric coverage")
                conditional={}
                for name,mask in masks.items():
                    count=mask.sum(1); keep=count>0
                    if not keep.any():
                        conditional[name]=dict(status="UNAVAILABLE",states=0,trajectories=0)
                        continue
                    nll=(arrays["nll"]*mask).sum(1)[keep]/count[keep]
                    mse=(arrays["rmse"]*mask).sum(1)[keep]/count[keep]
                    conditional[name]=dict(nll=float(nll.mean()),rmse=float(torch.sqrt(mse).mean()),
                        states=int(mask.sum()),trajectories=int(keep.sum()))
                row=dict(variant=variant,checkpoint=label,seed=seed,metrics=values,
                    conditional=conditional,wall_seconds=time.monotonic()-tick)
                all_records.append(row)
                name=variant+"-"+label+"-"+str(seed)+".json"
                write(out/name,dict(**row,per_step_nll=arrays["nll"].tolist(),
                                    per_step_squared_error=arrays["rmse"].tolist(),
                                    note="RMSE source metric retains squared errors until final per-trajectory aggregation"))
                variants.append(dict(path=str(out/name),sha256=sha(out/name)))
        del model,tf_model,roundtrip,expected,actual
    comparisons=[]
    for variant in ("mdpf","am_dpf"):
        for seed in FILTER_SEEDS:
            rows={x["checkpoint"]:x for x in all_records if x["variant"]==variant and x["seed"]==seed}
            a,b=rows["source_trained"],rows["tf_trained"]
            comparisons.append(dict(variant=variant,seed=seed,
                source=a["metrics"],tf_trained=b["metrics"],
                nll_difference=b["metrics"]["nll"]-a["metrics"]["nll"],
                rmse_ratio=b["metrics"]["rmse"]/a["metrics"]["rmse"],
                conditional={name:dict(nll_difference=b["conditional"][name]["nll"]-a["conditional"][name]["nll"],
                    rmse_ratio=b["conditional"][name]["rmse"]/a["conditional"][name]["rmse"])
                    for name in masks if a["conditional"][name].get("states",0)>0}))
    result=dict(schema="kdm_development_checkpoint_evaluation_v1",
        decision=dict(status="PASS",scope="ONE_TRAINING_PAIR_COMMON_SOURCE_EVALUATOR",
                      replication="INCOMPLETE",scientific_verdict="DEVELOPMENT_EVALUATION_COMPLETED"),
        comparisons=comparisons,check_count=len(checks),failures=[],checks=checks,
        checkpoint_provenance=provenance,metric_artifacts=variants,source_pin=PIN,
        source_files={str(p.relative_to(AUTHOR)):sha(p) for p in source_files+[eval_config_path,data_config_path]},
        dataset_sha256=sha(out/"evaluation-dataset.pt"),data_value_control=data_value_control,
        data_seed=DATA_SEED,filter_seeds=FILTER_SEEDS,
        rows=ROWS,batch_size=BATCH,horizon=HORIZON,particles=PARTICLES,
        gpu_intentionally_hidden=True,threads=1,torch_version=torch.__version__,numpy_version=np.__version__,
        wall_seconds=time.monotonic()-started,training_units=1,
        source_training_run=str(active_source),
        source_terminal_audit=str(source_audit) if source_audit is not None else None,
        source_terminal_audit_sha256=sha(source_audit) if source_audit is not None else None,
        tf_training_run=str(active_tf),
        tf_terminal_audit=str(tf_audit) if tf_audit is not None else None,
        tf_terminal_audit_sha256=sha(tf_audit) if tf_audit is not None else None,
        nonclaims="No trained-unit ranking, no-worse qualification, TF evaluator parity, replication, HMC or default readiness.")
    write(out/"result.json",result)
    print(json.dumps(dict(decision=result["decision"],checks=len(checks),comparisons=[
        {k:x[k] for k in ("variant","seed","nll_difference","rmse_ratio")} for x in comparisons])))
    return 0

def supervise(out):
    if out.parent!=BASE or not out.name.startswith("kdm-development-evaluation-20261008-"):
        raise ValueError("Use the phase's fresh versioned output directory")
    prior=list(BASE.glob("kdm-development-evaluation-20261008-*/manifest.json"))
    if len(prior)>=3:
        raise RuntimeError("Three-attempt budget exhausted")
    used=ROUTINE
    for path in prior:
        previous=json.loads(path.read_text())
        if previous["status"]=="RUNNING":
            raise RuntimeError("A development evaluation is already active")
        used+=previous["wall_seconds"]
    remaining=CAP-used
    if remaining<=0:
        raise RuntimeError("Phase CPU budget exhausted")
    out.mkdir(exist_ok=False)
    command=[SOURCE_PYTHON,"-I","-B",str(Path(__file__).resolve()),"--worker","--output",str(out/"diagnostic")]
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib",
             OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1")
    manifest=dict(schema="kdm_development_evaluation_manifest_v1",status="RUNNING",resource="cpu",
        command=command,launcher_command=[sys.executable,*sys.argv],cwd=str(ROOT),
        plan=str(PLAN.relative_to(ROOT)),result=RESULT,git_commit=subprocess.check_output(
            ["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),supervisor_pid=os.getpid(),
        environment={k:env[k] for k in ("CUDA_VISIBLE_DEVICES","LD_LIBRARY_PATH","OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS")},
        cpu_gpu_status="CPU only independent author reference; GPU intentionally hidden",
        data_seed=DATA_SEED,filter_seeds=FILTER_SEEDS,timeout_seconds=remaining,
        hashes={str(p.relative_to(ROOT)):sha(p) for p in (PLAN,Path(__file__).resolve(),
            ROOT/"docs/benchmarks/reference_kdm_neural_exact_replay.py",BASE/"kdm-complete-training-reviewed-20261007.json")})
    write(out/"manifest.json",manifest)
    start=time.monotonic()
    process=None
    try:
        with (out/"worker.log").open("w") as log:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            manifest["child_pid"]=process.pid
            write(out/"manifest.json",manifest)
            try:
                rc=process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGTERM)
                try: process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGKILL)
                    process.wait()
                rc=124
        manifest.update(returncode=rc,status="COMPLETED" if rc==0 else "FAILED")
        if rc==0:
            result=out/"diagnostic/result.json"
            report=json.loads(result.read_text())
            manifest.update(result_sha256=sha(result),decision=report["decision"])
    except BaseException:
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL)
                process.wait()
        manifest.update(status="FAILED",error=traceback.format_exc())
        raise
    finally:
        manifest["wall_seconds"]=time.monotonic()-start
        write(out/"manifest.json",manifest)
        print(json.dumps({k:manifest[k] for k in ("status","wall_seconds","returncode") if k in manifest}))
    return manifest.get("returncode",1)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--worker",action="store_true")
    parser.add_argument("--source-run",type=Path)
    parser.add_argument("--source-audit",type=Path)
    parser.add_argument("--tf-run",type=Path)
    parser.add_argument("--tf-audit",type=Path)
    args=parser.parse_args()
    if not args.worker and any(x is not None for x in (args.source_run,args.source_audit,args.tf_run,args.tf_audit)):
        parser.error("Run overrides require the bounded calibration supervisor")
    return worker(args.output.resolve(),args.source_run,args.source_audit,args.tf_run,args.tf_audit) if args.worker else supervise(args.output.resolve())

if __name__=="__main__":
    sys.exit(main())
