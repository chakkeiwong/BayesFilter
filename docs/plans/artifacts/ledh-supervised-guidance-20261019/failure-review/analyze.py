"""Post-run diagnostic arithmetic only; no fitting, selection, or admission."""

import collections
import hashlib
import json
import math
from pathlib import Path
import statistics


def mean(values):
    return statistics.fmean(values)


def read_json(path):
    return json.loads(path.read_text())


def analyze(root):
    manifest = read_json(root / "manifest.json")
    labels = read_json(root / "labels.json")
    rows = [json.loads(line) for line in (root / "rows.jsonl").read_text().splitlines()]
    scales = labels["scales"]
    keys = [(row["partition"], row["observation_seed"], row["particle_seed"], row["method"])
            for row in rows]
    assert len(keys) == len(set(keys)) == 96
    groups = collections.defaultdict(list)
    discrepancy = 0.0
    for row in rows:
        if not row["valid"]:
            continue
        target = labels["datasets"][str(row["observation_seed"])]["score"]
        recomputed = [((prediction-label)/scale)**2
                      for prediction, label, scale in zip(row["score"], target, scales)]
        discrepancy = max(discrepancy, *(abs(left-right)/(1+abs(right))
                                        for left,right in zip(recomputed,row["loss"])))
        groups[(row["partition"],row["method"],row["observation_seed"])].append(row)
    assert discrepancy < 1e-12
    partitions = {}
    for partition, seeds in manifest["partitions"].items():
        methods = {}
        for method in sorted({row["method"] for row in rows}):
            dataset_rows = [groups.get((partition,method,seed),[]) for seed in seeds]
            if any(len(observed) != 2 for observed in dataset_rows):
                methods[method] = dict(valid=False)
                continue
            losses, mean_errors, dispersions = [], [], []
            component_losses = []
            per_dataset = {}
            for seed, observed in zip(seeds,dataset_rows):
                target = labels["datasets"][str(seed)]["score"]
                predictions = [row["score"] for row in observed]
                average = [mean(values) for values in zip(*predictions)]
                component = [mean(values) for values in zip(*(row["loss"] for row in observed))]
                error = [((prediction-label)/scale)**2
                         for prediction,label,scale in zip(average,target,scales)]
                dispersion = [mean([((row[coordinate]-average[coordinate])/scales[coordinate])**2
                                     for row in predictions]) for coordinate in range(3)]
                assert all(abs(total-left-right) < 1e-9*(1+total)
                           for total,left,right in zip(component,error,dispersion))
                losses.append(sum(component)); mean_errors.append(sum(error))
                dispersions.append(sum(dispersion)); component_losses.append(component)
                per_dataset[str(seed)] = dict(loss=sum(component), score_mean=average,
                    score_each=predictions, label=target, component_loss=component,
                    mean_label_error=sum(error), particle_dispersion=sum(dispersion))
            methods[method] = dict(valid=True, mean_loss=mean(losses),
                component_loss=[mean(values) for values in zip(*component_losses)],
                mean_label_error=mean(mean_errors), particle_dispersion=mean(dispersions),
                per_dataset=per_dataset)
        baseline = methods["unguided"]
        for method, result in methods.items():
            if not result["valid"]:
                continue
            deltas = [result["per_dataset"][str(seed)]["loss"]-
                      baseline["per_dataset"][str(seed)]["loss"] for seed in seeds]
            result["paired_deltas"] = dict(zip(map(str,seeds),deltas))
            result["mean_delta"] = mean(deltas)
            result["leave_one_dataset_out_delta"] = {
                str(seed):mean(deltas[:index]+deltas[index+1:])
                for index,seed in enumerate(seeds)}
            result["noise_coordinate_fraction"] = result["component_loss"][2]/result["mean_loss"]
            result["particle_dispersion_fraction"] = result["particle_dispersion"]/result["mean_loss"]
        partitions[partition] = methods
    fits = {}
    for path in sorted(root.glob("fit-*.json")):
        fit = read_json(path)
        initial = [value[0] for value in fit["initial_ess"]]
        controlled = [value[0] for value in fit["controlled_ess"]]
        fits[path.stem.removeprefix("fit-")] = dict(
            initial_mean_ess=mean(initial[1:]), controlled_mean_ess=mean(controlled[1:]),
            initial_min_ess=min(initial[1:]), controlled_min_ess=min(controlled[1:]),
            projected=fit["projected"], raw_minimum=fit["raw_minimum"],
            projection_error_multiplier=[left/right for left,right in
                                         zip(fit["projected_error"],fit["quadratic_error"])],
            kernel_constant_loss_ratio=[left/right for left,right in
                                       zip(fit["selected_error"],fit["constant_error"])],
            bandwidth=fit["selected_bandwidth"], active=fit["active"])
    source_audit = {}
    for name,digest in manifest["source_hashes"].items():
        path = Path(name)
        source_audit[name] = hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return dict(horizon=manifest["horizon"], scales=scales, valid_rows=sum(row["valid"] for row in rows),
        arithmetic_relative_discrepancy=discrepancy, partitions=partitions, fits=fits,
        source_hash_matches=source_audit, nominal_independent_datasets=len(labels["datasets"]),
        particle_streams_reused_across_partitions=True,
        unconditional_noise_label_variance=18*manifest["horizon"],
        interpretation="descriptive exact arithmetic, not population bias/variance or statistical ranking")


def gaussian_example(guide_linear):
    flow_scale = math.sqrt(.5)
    flow_shift = 1.0
    guide_precision = 1.0
    unaligned_base_variance = 1/(1+guide_precision)
    unaligned_base_mean = unaligned_base_variance*guide_linear
    pulled_precision = flow_scale**2*guide_precision
    pulled_linear = flow_scale*(guide_linear-guide_precision*flow_shift)
    pulled_base_variance = 1/(1+pulled_precision)
    pulled_base_mean = pulled_base_variance*pulled_linear
    aligned_mean = flow_shift+flow_scale*pulled_base_mean
    aligned_variance = flow_scale**2*pulled_base_variance
    target_mean = (2+guide_linear)/3
    target_variance = 1/3
    assert math.isclose(aligned_mean,target_mean)
    assert math.isclose(aligned_variance,target_variance)
    unaligned_mean = flow_shift+flow_scale*unaligned_base_mean
    unaligned_variance = flow_scale**2*unaligned_base_variance

    def divergence(mean, variance):
        return .5*(math.log(variance/target_variance)
                   +(target_variance+(target_mean-mean)**2)/variance-1)

    baseline_divergence = divergence(1., .5)
    unaligned_divergence = divergence(unaligned_mean, unaligned_variance)
    if guide_linear == 1.0:
        assert unaligned_divergence > baseline_divergence > 0.
    for point in (-2.,-.5,0.,1.,3.):
        transformed = flow_scale*point+flow_shift
        direct = -.5*guide_precision*transformed**2+guide_linear*transformed
        pulled = -.5*pulled_precision*point**2+pulled_linear*point
        constant = -.5*guide_precision*flow_shift**2+guide_linear*flow_shift
        assert math.isclose(direct,pulled+constant,abs_tol=1e-14)
    return dict(prior_mean=0.,prior_variance=1.,current_observation=2.,observation_variance=1.,
        plain_posterior=dict(mean=1.,variance=.5), future_observation=guide_linear,
        exact_future_conditioned=dict(mean=target_mean,variance=target_variance),
        unaligned_preflow_guide=dict(mean=unaligned_mean,variance=unaligned_variance),
        pulled_back_guide=dict(mean=aligned_mean,variance=aligned_variance),
        kl_target_to_proposal=dict(unguided=baseline_divergence,
                                  unaligned=unaligned_divergence, aligned=0.),
        status="exact scalar counterexample to guide-coordinate equivalence, not a measured SIR effect")


if __name__ == "__main__":
    output = Path(__file__).resolve().parent
    root = output.parent
    result = dict(pilots={name:analyze(root/name) for name in ("pilot-02","pilot-03")},
                  gaussian_flow_examples=[gaussian_example(center) for center in (1., 3.)])
    (output/"analysis.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    for pilot, result in result["pilots"].items():
        print(pilot,"valid",result["valid_rows"],"scales",result["scales"])
        for partition in ("validation","test"):
            for method, summary in result["partitions"][partition].items():
                if summary["valid"]:
                    print(partition,method,"loss",round(summary["mean_loss"],3),
                          "noise share",round(summary["noise_coordinate_fraction"],4),
                          "particle spread share",round(summary["particle_dispersion_fraction"],4),
                          "leave-one-out delta range",
                          [round(operation(summary["leave_one_dataset_out_delta"].values()),3)
                           for operation in (min,max)])
