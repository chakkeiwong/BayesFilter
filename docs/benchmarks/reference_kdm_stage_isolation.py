"""CPU-only author reference for KDM stage isolation; diagnostic, never runtime."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "1"
import argparse, copy, hashlib, json, math, statistics, sys, time
from pathlib import Path
from types import MethodType, SimpleNamespace
ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
CAL = BASE / "paired-calibration-20261011/calibration-01"
SEED = 2026101310
PREFIXES = ("weight_model.", "observation_encoder.", "particle_encoder_for_weights_model.")
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(x): return hashlib.sha256(json.dumps(x, sort_keys=True, allow_nan=False).encode()).hexdigest()
def write(p, x):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)
def main(args):
    out = args.output; out.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(ROOT / "docs/benchmarks"))
    import reference_kdm_source_stages as source
    ref, torch = source.ref, source.torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    checks, inputs, rows = [], {}, []
    def check(name, passed):
        if not passed: raise AssertionError(name)
        checks.append(name)
    def record(p):
        inputs[str(p.relative_to(ROOT))] = sha(p)
        return read(p)
    def progress(**kw): write(out / "progress.json", kw)
    wp = ref.AUTHOR / "src/trainers/specific_trainers/weight_model_trainer.py"
    fullpath = ref.AUTHOR / "src/trainers/specific_trainers/full_sequence_trainer.py"
    tpath = ref.AUTHOR / "src/trainers/trainer.py"
    wforward = source.member(wp, "WeightModelTrainer", "do_forward_pass")
    masked = source.member(wp, "WeightModelTrainer", "get_true_masked_data")
    cloudcls = source.extract_clouds(wp)
    fullforward = source.member(fullpath, "FullSequenceTrainer", "do_forward_pass")
    convert = source.member(fullpath, "FullSequenceTrainer", "convert_to_model_dtype")
    validate = source.member(tpath, "Trainer", "do_validation_epoch")
    validate.__globals__["tqdm"] = lambda values, **kwargs: values
    transform = source.member(ref.AUTHOR / "src/problems/bearings_only/bearings_only_problem.py",
                              "BearingsOnlyObservationTransformer")
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and Path(filename).suffix == ".py" and Path(filename).is_relative_to(ref.AUTHOR):
            ref.verify(Path(filename))
    summaries = []
    for unit in ((args.unit,) if args.unit is not None else ((1, 2) if args.mode == "audit" else (1,))):
        unitname = "unit%02d" % unit
        sr = BASE / "kdm-source-variability-20261008-01" / (unitname + "-full/diagnostic")
        tr = CAL / unitname / "training/diagnostic"
        cfgs = record(sr / "configurations.json")["executed_stages"]
        cfg = next(x for x in cfgs if x["name"] == "mdpf")
        weightcfg = next(x for x in cfgs if x["name"] == "weights")
        fixture = record(sr / "mdpf-constructor.json")
        template = record(tr / "mdpf-starting-state.json")
        # JSON stringifies dimension keys; execute the pinned native YAML.
        native_common = ref.yaml.safe_load(ref.verify(ref.EXPERIMENT / "configs/common_parameters_training.yaml"))
        native_arch = ref.yaml.safe_load(ref.verify(ref.EXPERIMENT / "configs/models.yaml"))["models"]
        native_arch.update(ref.yaml.safe_load(ref.verify(
            ref.EXPERIMENT / "experiment0002_importance/config.yaml"))["models"])
        check("native model configuration matches saved", json.loads(json.dumps(
            native_common["model_params"])) == cfg["configuration"]["model_params"])
        check("native architecture matches saved", json.loads(json.dumps(native_arch)) == cfg["architecture"])
        check("native dimension key types", all(type(k) is int for k in native_arch["dpf_full_training"]["kde"]["dims"]))
        model = ref.KDEParticleFilter(native_common["model_params"], native_arch)
        params = dict(model.named_parameters())
        check("constructor coverage", set(params) == set(fixture["parameters"]) == set(template["parameters"]))
        with torch.no_grad():
            for name, value in params.items(): value.copy_(torch.tensor(fixture["parameters"][name]))
        settings = cfg["configuration"]["pre_trained_models"]
        for item in settings.values():
            path = Path(item); inputs[str(path.relative_to(ROOT))] = sha(path)
        model.load_pretrained(settings, torch.device("cpu"))
        check("source model constants", source.constants(model) == fixture["constants"])
        check("TF model constants", template["constants"] == {
            key: value for key, value in fixture["constants"].items()
            if key != "resampling_min_bandwidths" or value is not None})
        author_start = ref.host(dict(model.named_parameters()))
        source_state = copy.deepcopy(template); source_state["parameters"] = author_start
        tf_pretrained = record(tr / "weights-best.json")["model"]
        for name, value in author_start.items():
            transferred = name.startswith(PREFIXES)
            check("TF transfer " + name, template["parameters"][name] ==
                  (tf_pretrained["parameters"][name] if transferred else fixture["parameters"][name]))
            if not transferred:
                check("source nontransferred " + name, value == fixture["parameters"][name])
        check("rates match source fixture", template["rates"] == fixture["learning_rates"])
        check("trainable names match", template["trainable_names"] == fixture["trainable_names"])
        check("fresh stage optimizer", template["step"] == 0 and all(
            all(v == 0 for v in torch.tensor(x).flatten().tolist())
            for key in ("first_moments", "second_moments") for x in template[key].values()))
        differences = [name for name in author_start if author_start[name] != template["parameters"][name]]
        check("differences confined to measurement modules", all(n.startswith(PREFIXES) for n in differences))
        write(out / (unitname + "-author-start.json"), source_state)
        data = record(sr / "validation.json")
        check("validation coverage", data["rows"] == 1000 and data["row_ids"] == list(range(1000)))
        batches = [{k: torch.tensor(v[a:min(a+256, 1000)], dtype=torch.bool if k == "ground_truth_mask" else torch.float32)
                    for k, v in data["data"].items()} for a in range(0, 1000, 256)]
        model.eval()
        if args.mode == "audit":
            native_config = ref.yaml.safe_load(ref.verify(ref.EXPERIMENT / "individual_training/config.yaml"))
            native_spec = next(v for e in native_config["experiments"] for n, v in e.items()
                               if n == "full_dpf_weight_training_fixed_bands")
            training = native_spec["training_params"]
            saved_training = copy.deepcopy(weightcfg["configuration"]["training_params"])
            check("saved loader is CPU reference", saved_training.pop("num_cpu_cores_for_dataloader", None) == 1)
            check("native numerical pretraining configuration", json.loads(json.dumps(training)) == saved_training)
            generator = cloudcls(training["particle_cloud_params"])
            trainer = SimpleNamespace(main_model=model, device=torch.device("cpu"),
                problem=SimpleNamespace(observation_transformer=transform()), particle_cloud_generator=generator,
                loss_function=ref.KDE_NLL(training["loss_params"], model))
            trainer.get_true_masked_data = MethodType(masked, trainer)
            original_cloud = generator.create_cloud
            capture = []
            def captured(states):
                cloud = original_cloud(states); capture.append(digest(ref.host(cloud)))
                return cloud
            generator.create_cloud = captured
            candidates = {"author_pretrained": author_start, "tf_pretrained": template["parameters"]}
            for draw in range(8):
                equal_clouds, equal_rng = [], []
                for candidate, values in candidates.items():
                    with torch.no_grad():
                        for name, value in params.items(): value.copy_(torch.tensor(values[name]))
                    before = digest(ref.host(dict(model.named_parameters())))
                    seed = SEED + unit * 10000 + draw
                    torch.manual_seed(seed); capture.clear()
                    terms, effective = [], 0
                    progress(mode=args.mode, unit=unit, draw=draw, candidate=candidate)
                    with torch.no_grad():
                        for index, batch in enumerate(batches):
                            loss, count = wforward(trainer, batch, None, 0)
                            value = float(loss); check("finite pretraining NLL", math.isfinite(value))
                            check("selected labeled count", count == len(batch["states"]) * 5)
                            terms.append(value*count); effective += count
                            rows.append(dict(mode=args.mode, unit=unit, draw=draw, candidate=candidate, seed=seed,
                                batch=index, count=count, loss=value, cloud_sha256=capture[-1]))
                    check("all labeled rows", effective == 5000)
                    check("evaluation immutable", before == digest(ref.host(dict(model.named_parameters()))))
                    equal_clouds.append(list(capture)); equal_rng.append(digest(torch.get_rng_state().tolist()))
                    summaries.append(dict(unit=unit, draw=draw, candidate=candidate, loss=math.fsum(terms)/effective))
                check("common realized pretraining clouds", equal_clouds[0] == equal_clouds[1])
                check("common ending pretraining RNG", equal_rng[0] == equal_rng[1])
        else:
            check("trained path provided", args.trained is not None)
            trained = record(args.trained / "mdpf-best.json")["model"]
            old = record(tr / "mdpf-best.json")["model"]
            ap = sr / "mdpf/models/full_dpf_model_best.pt"
            inputs[str(ap.relative_to(ROOT))] = sha(ap)
            authors = torch.load(ap, map_location="cpu", weights_only=True)
            class Plotter:
                def add_value(self, value): check("finite plot value", math.isfinite(value))
            trainer = SimpleNamespace(main_model=model, device=torch.device("cpu"),
                problem=SimpleNamespace(observation_transformer=transform()), number_of_particles=25,
                skip_first_step_for_loss=False, max_sequence_length=None, truncated_bptt_modulo=4, loss_shaper=None,
                loss_function=ref.KDE_NLL(dict(kde_params=native_arch["dpf_full_training"]["kde"]), model),
                models={"full": (model, False)}, validation_dataset=None, validation_loader=batches,
                all_data_plotters={"validation_loss_plotter": Plotter()})
            trainer.convert_to_model_dtype = MethodType(convert, trainer)
            trainer.do_forward_pass = MethodType(fullforward, trainer)
            raw_forward = trainer.do_forward_pass
            captured_batches = []
            def observed(batch, dataset, epoch):
                loss, count = raw_forward(batch, dataset, epoch)
                captured_batches.append(dict(count=count, loss=float(loss)))
                return loss, count
            trainer.do_forward_pass = observed
            candidates = {"author_best": ref.host(authors), "old_tf_best": old["parameters"],
                          "author_start_tf_best": trained["parameters"]}
            for ci, (candidate, values) in enumerate(candidates.items()):
                if candidate == "author_best": model.load_state_dict(authors, strict=True)
                else:
                    with torch.no_grad():
                        for name, value in params.items(): value.copy_(torch.tensor(values[name]))
                before = digest(ref.host(dict(model.named_parameters())))
                for draw in range(8):
                    seed = SEED + 900000 + ci * 1000 + draw
                    torch.manual_seed(seed); captured_batches.clear()
                    progress(mode=args.mode, unit=unit, candidate=candidate, draw=draw)
                    loss = float(validate(trainer, 0))
                    check("finite native loss", math.isfinite(loss))
                    check("native batch coverage", [x["count"] for x in captured_batches] == [256, 256, 256, 232])
                    check("native aggregation", loss == sum(x["loss"]*x["count"] for x in captured_batches)/1000)
                    check("native evaluation immutable", before == digest(ref.host(dict(model.named_parameters()))))
                    rows.append(dict(mode=args.mode, unit=unit, candidate=candidate, draw=draw, seed=seed,
                                     loss=loss, batches=copy.deepcopy(captured_batches)))
                    summaries.append(dict(unit=unit, candidate=candidate, draw=draw, loss=loss))
        write(out / "inputs.json", inputs)
        write(out / "rows.json", rows)
        write(out / "summaries.json", summaries)
    stats = []
    for unit in sorted({x["unit"] for x in summaries}):
        for candidate in sorted({x["candidate"] for x in summaries if x["unit"] == unit}):
            values = [x["loss"] for x in summaries if x["unit"] == unit and x["candidate"] == candidate]
            stats.append(dict(unit=unit, candidate=candidate, n=len(values), mean=statistics.mean(values),
                              stdev=statistics.stdev(values), minimum=min(values), maximum=max(values)))
    write(out / "result.json", dict(status="PASS", mode=args.mode, checks=len(checks), check_names=checks,
        statistics=stats, source_commit=ref.PIN, author_hashes=ref.HASHES, cpu_only=True, gpu_intentionally_hidden=True,
        source_files={str(p.relative_to(ROOT)):sha(p) for p in (wp, fullpath, tpath)},
        seed_base=SEED, input_hashes=inputs, inference="checkpoint-conditional diagnostic; no population ranking",
        replication="INCOMPLETE"))
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode",choices=("audit","evaluate"),required=True); p.add_argument("--trained",type=Path)
    p.add_argument("--unit",type=int,choices=(1,2))
    main(p.parse_args())
