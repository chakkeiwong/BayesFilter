"""Complete-data epoch cost diagnostic; not full source training or qualification."""
import copy
import hashlib
import json
import math
from pathlib import Path
import time

def run_epochs(tf, reference, baseline, args, out, result, boolean, run, write,
               *, require_same_reference=True, seed_base=20261007, epoch_limit=1,
               selected_stages=("weights", "mdpf", "am_dpf"), initial_states=None):
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.kdm_source_stages_tf import SourceWeightPretraining, SourceFullTraining
    from bayesfilter.replication.kdm_stage_transfer_tf import transfer_stage_parameters
    from bayesfilter.replication.kdm_training_controller import SourceEpochController
    root = Path(__file__).resolve().parents[2]
    if args.verification is None:
        raise ValueError("Explicit completed verification required")
    proof = json.loads(args.verification.read_text())
    if (proof["status"] != "PASS" or proof["mode"] != "verify" or proof["checks_failed"]
            or (require_same_reference and proof["reference_sha256"] != result["reference_sha256"])
            or not proof.get("stage_transfers_checked")):
        raise ValueError("Source-stage verification incomplete or stale")
    proof_manifest = json.loads((args.verification.parent.parent/"manifest.json").read_text())
    if hashlib.sha256(args.verification.read_bytes()).hexdigest() != proof_manifest["result_sha256"]:
        raise ValueError("Verification result checksum mismatch")
    for name, digest in proof_manifest["hashes"].items():
        if name.startswith("bayesfilter/replication/"):
            if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
                raise ValueError("Runtime changed since verification: "+name)
    result["verification"] = dict(path=str(args.verification),
        sha256=hashlib.sha256(args.verification.read_bytes()).hexdigest(),
        scope="same reference" if require_same_reference else "unchanged runtime components; fresh inputs are cost-only")
    data = {}
    for split, count in (("training",5000),("validation",1000)):
        exported = json.loads(Path(reference["datasets"][split]["path"]).read_text())
        if exported["rows"] != count or exported["row_ids"] != list(range(count)):
            raise ValueError("Incomplete source dataset")
        data[split] = [tf.constant(exported["data"][key], tf.bool if key=="ground_truth_mask" else tf.float32)
                      for key in ("states","observations","ground_truth_mask")]
    reports = []
    result["epochs"] = reports
    result["completed_stages"] = []
    previous = None
    if selected_stages != ("weights", "mdpf", "am_dpf"):
        if selected_stages != ("mdpf",) or set(initial_states or {}) != {"mdpf"}:
            raise ValueError("Only the declared MDPF starting-state diagnostic may isolate a stage")
    with tf.device("/GPU:0"):
        for stage_index, name in enumerate(("weights","mdpf","am_dpf")):
            if name not in selected_stages:
                continue
            stage_start = time.monotonic()
            if name == "weights":
                fixture = next(c for c in reference["pretraining_cases"] if c["dtype"]=="float32")
                cfg = reference["pretraining_config"]
            else:
                fixture = next(c for c in baseline["cases"] if c["dtype"]=="float32" and c["variant"]==name)
                cfg = reference.get("full_training_configs", {}).get(name, reference["full_training_config"])
            model = BearingsNeuralReplay(fixture)
            transfer = None if previous is None else transfer_stage_parameters(model, previous,
                "weights_to_mdpf" if name=="mdpf" else "mdpf_to_am_dpf")
            if initial_states is not None and name in initial_states:
                if previous is not None:
                    raise ValueError("Starting-state diagnostic cannot override a stage transfer")
                model.load_state_dict(initial_states[name])
                if float(model.step.numpy()) != 0.:
                    raise ValueError("Stage isolation must begin before any full-training update")
                transfer = dict(stage="diagnostic_explicit_start", optimizer_reset=True,
                    original_stage_index=stage_index)
            if (cfg["training_batch_size"],cfg["validation_batch_size"]) != (64,256):
                raise ValueError("Unreviewed source batch dimensions")
            source_cap = cfg["epochs"]
            if source_cap != (400 if name=="weights" else 800):
                raise ValueError("Wrong author epoch cap")
            cap = source_cap if epoch_limit is None else min(source_cap, epoch_limit)
            scheduler = {k:v for k,v in cfg["lr_scheduler_params"].items()
                         if k not in ("scheduler_type","verbose")}
            controller = SourceEpochController(fixture["learning_rates"],scheduler,
                patience=cfg["early_stopping_patience"],start_offset=cfg["early_stopping_start_offset"])
            views = {b:(SourceWeightPretraining(model,b) if name=="weights"
                        else SourceFullTraining(model.for_batch(b))) for b in (64,8,256,232)}
            boolean(name+"/batch_views_share_parameters",
                all(v.model.parameters[k] is p for v in views.values() for k,p in model.parameters.items()))
            model_before = model.state_dict()
            frozen = {k:v for k,v in model_before["parameters"].items() if k not in fixture["trainable_names"]}
            checkpoint_start = out/(name+"-starting-state.json")
            write(checkpoint_start, model_before)
            previous_orders = None
            for epoch in range(cap):
                epoch_start = time.monotonic()
                report = dict(stage=name,epoch=epoch,source_epoch_cap=source_cap,executed_epoch_cap=cap,
                    initial_checkpoint_role=result.get("initialization", "development cost fixture; weights seeded from historical fitted model"),
                    stage_transfer=transfer,splits={})
                reports.append(report)
                with tf.device("/CPU:0"):
                    order_train = tf.random.experimental.stateless_shuffle(
                        tf.range(5000), tf.constant([seed_base+epoch*100,stage_index],tf.int64))
                orders = order_train.numpy().tolist()
                if previous_orders is not None:
                    boolean(name+f"/epoch{epoch}/distinct_shuffle",orders != previous_orders)
                previous_orders = orders
                for split,count,width in (("training",5000,64),("validation",1000,256)):
                    with tf.device("/CPU:0"):
                        order = order_train if split=="training" else tf.range(count)
                    rows = order.numpy().tolist()
                    prefix = name+f"/epoch{epoch}/"+split
                    boolean(prefix+"/row_order_on_cpu", "CPU" in order.device)
                    boolean(prefix+"/each_row_once", sorted(rows)==list(range(count)))
                    terms, weights, elapsed = [], [], []
                    before_validation = model.state_dict() if split=="validation" else None
                    tick_split = time.monotonic()
                    for batch_index,start in enumerate(range(0,count,width)):
                        indexes=order[start:start+width]
                        batch=int(indexes.shape[0])
                        values=[tf.gather(value,indexes) for value in data[split]]
                        view=views[batch]
                        function = (view.native_update if split=="training" else view.native_evaluate) if name=="weights" else (
                                    view.update if split=="training" else view.evaluate)
                        key=tf.constant([seed_base+1+stage_index,epoch*20000+(0 if split=="training" else 10000)+batch_index],tf.int64)
                        before_call=time.monotonic()
                        got=run(prefix+f"/batch{batch_index}",function,*values,key)
                        seconds=time.monotonic()-before_call
                        valid=bool(got["valid"].numpy())
                        committed=bool(got["committed"].numpy()) if split=="training" else None
                        loss=float(got["loss"].numpy())
                        if not valid or not math.isfinite(loss) or (split=="training" and not committed):
                            raise RuntimeError("Invalid training batch: "+prefix+"/"+str(batch_index))
                        effective=batch*5 if name=="weights" else batch
                        weights.append(effective); terms.append(loss*effective); elapsed.append(seconds)
                        row=dict(stage=name,epoch=epoch,split=split,batch_index=batch_index,sequence_batch=batch,
                            effective_batch=effective,row_ids=rows[start:start+width],seed=key.numpy().tolist(),
                            loss=loss,valid=valid,committed=committed,wall_seconds=seconds)
                        with (out/"batches.jsonl").open("a") as stream:
                            stream.write(json.dumps(row,allow_nan=False)+"\n")
                    mean=math.fsum(terms)/sum(weights)
                    boolean(prefix+"/complete_batches",len(weights)==(79 if split=="training" else 4))
                    boolean(prefix+"/partial_batch",batch==(8 if split=="training" else 232))
                    if split=="validation":
                        boolean(prefix+"/immutable",model.state_dict()==before_validation)
                    report["splits"][split]=dict(rows=count,batches=len(weights),
                        row_order_sha256=hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
                        weighted_loss=mean,total_effective_weight=sum(weights),row_order_device=order.device,
                        wall_seconds=time.monotonic()-tick_split,first_call_seconds=elapsed[0],
                        partial_call_seconds=elapsed[-1],regular_steady_seconds=elapsed[1:-1],
                        optimizer_step=float(model.step.numpy()))
                event=controller.observe(report["splits"]["validation"]["weighted_loss"])
                model.set_learning_rates(event["learning_rates"])
                state=model.state_dict()
                boolean(name+f"/epoch{epoch}/frozen_parameters",all(state["parameters"][k]==v for k,v in frozen.items()))
                boolean(name+f"/epoch{epoch}/optimizer_updates",int(model.step.numpy())==79*(epoch+1))
                boolean(name+f"/epoch{epoch}/controller_epoch",controller.epoch==epoch)
                checkpoint=dict(model=state,controller=controller.state_dict(),stage=name,next_epoch=epoch+1,
                    role="COMPLETE_DEVELOPMENT_TRAINING" if epoch_limit is None else "BOUNDED_EPOCH_DIAGNOSTIC",
                    reference_sha256=result["reference_sha256"],data=reference["datasets"],
                    seed_policy=dict(shuffle_base=seed_base,shuffle_epoch_stride=100,
                                     native_base=seed_base+1,native_epoch_stride=20000))
                write(out/(name+"-latest.json"),checkpoint)
                if event["save_best"]:
                    write(out/(name+"-best.json"),checkpoint)
                traces={}
                for b,view in views.items():
                    function=(view.native_update if b in (64,8) else view.native_evaluate) if name=="weights" else (
                              view.update if b in (64,8) else view.evaluate)
                    traces[str(b)]=function.experimental_get_tracing_count()
                boolean(name+f"/epoch{epoch}/one_trace_per_batch_shape",all(v==1 for v in traces.values()))
                report.update(event=event,trace_counts=traces,wall_seconds=time.monotonic()-epoch_start,
                    allocator=tf.config.experimental.get_memory_info("GPU:0"),complete_epoch=True,
                    training_complete=epoch_limit is None and (event["stop"] or epoch+1==source_cap))
                with (out/"epochs.jsonl").open("a") as stream:
                    stream.write(json.dumps(report,allow_nan=False)+"\n")
                write(out/"epochs.partial.json",dict(latest=report,completed_stages=result["completed_stages"]))
                if event["stop"]: break
            completed=epoch+1
            if epoch_limit is not None and completed != cap:
                raise RuntimeError("Unexpected early stop in bounded epoch diagnostic")
            best=json.loads((out/(name+"-best.json")).read_text())
            boolean(name+"/best_epoch_matches_controller",best["next_epoch"]-1==controller.best_checkpoint_epoch)
            previous=best["model"]
            summary=dict(stage=name,epochs=completed,source_cap=source_cap,
                termination="SOURCE_EARLY_STOP" if event["stop"] else ("SOURCE_CAP" if epoch_limit is None else "DIAGNOSTIC_CAP"),
                best_epoch=controller.best_checkpoint_epoch,wall_seconds=time.monotonic()-stage_start,
                best_checkpoint=str(out/(name+"-best.json")),
                best_checkpoint_sha256=hashlib.sha256((out/(name+"-best.json")).read_bytes()).hexdigest(),
                full_protocol=epoch_limit is None,optimizer_updates=int(model.step.numpy()))
            result["completed_stages"].append(summary)
            write(out/"stages.partial.json",result["completed_stages"])
    result["source_training_protocol_complete"]=(epoch_limit is None and
        selected_stages == ("weights", "mdpf", "am_dpf"))
    result["selected_stages_complete"] = epoch_limit is None
    return reports
