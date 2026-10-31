"""Fresh restricted-message diagnostics for the SIR neural proposal study."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

if __name__ == "__main__":
    configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
from bayesfilter.score_study.moment_training_campaign import append, save, serializable
from bayesfilter.score_study.moment_training_tf import (
    make_guide_adapters, make_moment_endpoint, random_direction_descent,
)
from bayesfilter.score_study.neural_kdm_study import NeuralKDMStudy
from bayesfilter.score_study.neural_moment_evidence import paired_interval, validate_holdout
from bayesfilter.score_study.run_source_snapshot import verify_source_snapshot
from bayesfilter.score_study.sir_backward_guidance_diagnostic_tf import SIRGuideFitter


def remaining_work_projection(evaluation_times, fit_times):
    evaluations = {}
    for family in ("csmc", "rbf"):
        timings = evaluation_times[family]
        warm = timings[1:]
        if not warm:
            raise ValueError("Both comparator endpoints need measured warm evaluations")
        evaluations[family] = max(0, 2176 - len(timings)) * math.fsum(warm) / len(warm)
    warm_fits = fit_times[1:]
    if not warm_fits:
        raise ValueError("Message fitting needs measured warm executions")
    fitting = max(0, 144 - len(fit_times)) * math.fsum(warm_fits) / len(warm_fits)
    return dict(evaluation_seconds=evaluations, fitting_seconds=fitting,
                projected_seconds=1.2 * (math.fsum(evaluations.values()) + fitting),
                cost_margin=1.2, maximum_evaluations_per_family=2176,
                maximum_distinct_message_fits=144)


def analyze_restricted_comparisons(neural_rows, rows, datasets):
    reference, regimes = validate_holdout(neural_rows, datasets)
    arms = ("csmc_unadapted", "csmc_adapted", "rbf_unadapted", "rbf_adapted")
    if any(row.get("arm") not in arms for row in rows):
        raise ValueError("Unexpected restricted comparator")
    indexed = {}
    for arm in arms:
        replacement = [dict(row, arm="trained") for row in rows if row["arm"] == arm]
        records, _ = validate_holdout(
            [row for row in neural_rows if row["arm"] != "trained"] + replacement, datasets)
        indexed[arm] = records
    subsets = dict(all=datasets)
    subsets.update({name: [dataset for dataset in datasets if regimes[dataset][0] == name]
                    for name in ("low", "high")})
    subsets.update({name: [dataset for dataset in datasets if regimes[dataset][1] == name]
                    for name in ("nearer_boundary", "interior")})
    summaries, comparisons = [], []
    for regime, subset in subsets.items():
        if not subset:
            continue
        neural_means = [math.fsum(reference[dataset, "trained", replicate]["loss"]
                        for replicate in range(4)) / 4. for dataset in subset]
        for arm in arms:
            means = [math.fsum(indexed[arm][dataset, "trained", replicate]["loss"]
                     for replicate in range(4)) / 4. for dataset in subset]
            blocks = [math.fsum(indexed[arm][dataset, "trained", replicate]["blocks"][block]
                      for dataset in subset for replicate in range(4)) / (4 * len(subset))
                      for block in range(2)]
            differences = [neural - baseline for neural, baseline in zip(neural_means, means)]
            mean = math.fsum(means) / len(means)
            difference = math.fsum(differences) / len(differences)
            summaries.append(dict(regime=regime, arm=arm, datasets=len(subset), loss=mean, blocks=blocks))
            comparisons.append(dict(regime=regime, comparator=arm, datasets=len(subset),
                mean_difference=difference, paired_cluster_interval95=paired_interval(differences),
                relative_reduction=-difference / mean if mean else None,
                role="explanatory; not the primary promotion criterion", multiplicity_adjusted=False))
    missing = [name for name, subset in subsets.items() if not subset]
    return dict(status="incomplete_required_regimes" if missing else "complete",
        rows=len(rows), all_valid=True, summaries=summaries,
        comparisons=comparisons, missing_regimes=missing,
        bootstrap=dict(draws=5000, seed=263045001, unit="dataset with all four particle replications"))


class RestrictedStudy(NeuralKDMStudy):
    def __init__(self, arguments):
        super().__init__(arguments)
        factory = lambda parameter, direction: self.spec.model(parameter, direction)[0]
        kernels = {
            "csmc": build_backward_guided_ledh_kernel(factory, None,
                dynamic_quadratic=True, physical_mixture_weight=.5, **self.common),
            "rbf": build_backward_guided_ledh_kernel(factory, None,
                mixture_components=32, mixture_conditioning="mixture_product_moments",
                physical_mixture_weight=.5, **self.common),
        }
        self.message_endpoints = {name: make_moment_endpoint(kernel, self.objective, 18)
                                  for name, kernel in kernels.items()}
        self.adapters = {capacity: make_guide_adapters(5, 18, 32, capacity == "time_specific")
                         for capacity in ("pooled", "time_specific")}
        self.fitter = SIRGuideFitter(5, components=32)
        self.message_cache = {}
        self.batch_cache = {}
        self.validation_cache = {}
        self.evaluation_times = dict(csmc=[], rbf=[])
        self.fit_times = []
        neural_manifest = json.loads((arguments.neural_study / "final-holdout" / "manifest.json").read_text())
        for field in ("controls", "calibration_seed", "reset_seed", "dtype", "tf32", "jit_compile", "horizon", "particles"):
            if neural_manifest[field] != self.manifest[field]:
                raise ValueError(f"Neural and restricted scopes disagree on {field}")
        selection = json.loads((arguments.neural_study / "frozen-selection.json").read_text())
        neural_checkpoint = json.loads(Path(selection["selected"]["path"]).read_text())
        if neural_checkpoint["source_hashes"] != self.source["source_hashes"]:
            raise ValueError("Neural and restricted studies use different package sources")
        for field in ("components", "width", "seed", "learning_rate", "updates",
                      "parameter_names", "parameter_count"):
            self.manifest.pop(field)
        self.manifest.update(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            protocol="docs/plans/artifacts/ledh-sir-neural-kdm-20261020/attempt-01/restricted-comparator-protocol.md",
            baseline_beta=.5, fitting_particles=512, fitting_validation_particles=256,
            fitting_components=32, adapter_seeds=[263047001, 263047002],
            adapter_rates=[.05, .15], adapter_updates=[8, 24],
            batch_size=2, generated_batch_size=4, adapter_parameter_counts=[3, 12],
            method_families=["restricted quadratic cSMC-derived", "restricted positive RBF"],
            training_batch_rows=[0, 1], fitting_seed_namespaces=[284040000, 284042000, 284043000],
            neural_study=str(arguments.neural_study))
        save(self.output / "manifest.json", self.manifest)

    def messages(self, inputs, fit_seed):
        if fit_seed not in self.message_cache:
            self.check_budget()
            begin = time.monotonic()
            quadratic, mixture, diagnostics = self.fitter.fit(inputs[5], fit_seed)
            values = dict(csmc=(quadratic.precision, quadratic.linear),
                          rbf=(mixture.logits, mixture.centers, mixture.covariances, mixture.active))
            if not all(bool(tf.reduce_all(tf.math.is_finite(value)))
                       for coefficients in values.values() for value in coefficients
                       if value.dtype.is_floating):
                raise ValueError("Non-finite exported message coefficients")
            self.message_cache[fit_seed] = values
            self.fit_times.append(time.monotonic() - begin)
            save(self.output / f"message-{fit_seed}.json", diagnostics)
        return self.message_cache[fit_seed]

    def evaluate_message(self, family, capacity, eta, inputs, fit_seed):
        self.check_budget()
        messages = self.messages(inputs, fit_seed)
        begin = time.monotonic()
        adapter = self.adapters[capacity][0 if family == "csmc" else 1]
        coefficients = adapter(eta, *messages[family])
        result = self.message_endpoints[family](*inputs[:-1], *coefficients, inputs[-1])
        float(result["loss"])
        self.evaluation_times[family].append(time.monotonic() - begin)
        return result

    def validate_adapter(self, family, capacity, eta):
        key = (family, capacity, tuple(tf.reshape(eta, [-1]).numpy().tolist()))
        if key in self.validation_cache:
            return self.validation_cache[key]
        losses = []
        for dataset_index in range(32):
            for replicate in range(2):
                inputs = self.row_inputs(tf.constant(263042001 + dataset_index),
                    tf.constant(283020000 + 2 * dataset_index + replicate))
                result = self.evaluate_message(family, capacity, eta, inputs, 284042000 + dataset_index)
                if not bool(result["valid"]):
                    return None
                losses.append(float(result["loss"]))
        loss = math.fsum(losses) / len(losses)
        self.validation_cache[key] = loss if math.isfinite(loss) else None
        return self.validation_cache[key]

    def fit_adapter(self, family, capacity, seed, rate, *, start=0, stop=8, initial=None):
        begin = time.monotonic()
        eta = tf.zeros([4 if capacity == "time_specific" else 1, 3], tf.float64) if initial is None else initial
        run = f"{family}-{capacity}-{seed}-{rate}-{start}-{stop}"
        candidates = []

        def objective(parameters, relative_update):
            update = start + relative_update
            if update not in self.batch_cache:
                self.batch_cache[update] = self.make_batch(tf.constant(263041000 + update),
                                                          tf.constant(283010000 + update))
            batch = self.batch_cache[update]
            losses = []
            for row in range(2):
                inputs = tuple(value[row] for value in batch)
                result = self.evaluate_message(family, capacity, parameters, inputs,
                                               284040000 + 2 * update + row)
                if not bool(result["valid"]):
                    return math.inf
                losses.append(float(result["loss"]))
            return math.fsum(losses) / len(losses)

        def checkpoint(relative_update, parameters):
            loss = self.validate_adapter(family, capacity, parameters)
            candidate = dict(run=run, family=family, capacity=capacity, seed=seed,
                step_size=rate, updates=start + relative_update, eta=parameters,
                validation_loss=loss)
            candidates.append(serializable(candidate))
            append(self.output / "validation.jsonl", candidate)
            print(json.dumps(dict(run=run, updates=start + relative_update,
                                  validation_loss=loss)), flush=True)

        try:
            _, accepted = random_direction_descent(objective, eta, seed=seed,
                updates=stop - start, perturbation=.1, step_size=rate,
                checkpoint=checkpoint, record=lambda row: append(self.output / "training.jsonl",
                    dict(run=run, absolute_update=start + row["update"], **row)))
            status = "complete"
        except ValueError as error:
            if str(error) != "Invalid current training candidate":
                raise
            accepted, status = None, "invalid_training_candidate"
        save(self.output / f"fit-{run}.json", dict(status=status, accepted=accepted, candidates=candidates,
            wall_seconds=time.monotonic() - begin))
        return candidates

    def train_adapters(self):
        for family in ("csmc", "rbf"):
            if self.validate_adapter(family, "pooled", tf.zeros([1, 3], tf.float64)) is None:
                raise ValueError(f"Invalid unadapted {family} comparator")
        projection = remaining_work_projection(self.evaluation_times, self.fit_times)
        remaining = self.arguments.budget_seconds - (time.monotonic() - self.started)
        projection.update(remaining_seconds=remaining,
                          fits_budget=projection["projected_seconds"] <= remaining)
        save(self.output / "budget-projection.json", projection)
        if not projection["fits_budget"]:
            raise TimeoutError("Complete restricted comparison exceeds the measured remaining budget")
        selected = {}
        for family in ("csmc", "rbf"):
            candidates = []
            for capacity in ("pooled", "time_specific"):
                for seed in (263047001, 263047002):
                    for rate in (.05, .15):
                        candidates.extend(self.fit_adapter(family, capacity, seed, rate))
            valid = [candidate for candidate in candidates if candidate["validation_loss"] is not None]
            trained = [candidate for candidate in valid if candidate["updates"] == 8]
            if trained:
                nominee = min(trained, key=lambda candidate: candidate["validation_loss"])
                candidates.extend(self.fit_adapter(family, nominee["capacity"], nominee["seed"] + 1000,
                    nominee["step_size"], start=8, stop=24,
                    initial=tf.constant(nominee["eta"], tf.float64)))
            valid = [candidate for candidate in candidates if candidate["validation_loss"] is not None]
            if not valid:
                raise ValueError(f"No valid {family} restricted baseline")
            selected[family] = min(valid, key=lambda candidate: candidate["validation_loss"])
        save(self.output / "frozen-selection.json", dict(selected=selected,
            holdout_opened=False, selection="validation only"))
        return selected

    def evaluate_holdout(self, selected):
        neural_path = self.arguments.neural_study / "final-holdout" / "holdout.jsonl"
        neural_rows = [json.loads(line) for line in neural_path.read_text().splitlines()]
        datasets = list(range(263043001, 263043065))
        validate_holdout(neural_rows, datasets)
        reference = {(row["dataset"], row["replicate"]): row for row in neural_rows if row["arm"] == "trained"}
        rows = []
        for dataset_index, dataset in enumerate(datasets):
            for replicate in range(4):
                particle_seed = 283030000 + 4 * dataset_index + replicate
                inputs = self.row_inputs(tf.constant(dataset), tf.constant(particle_seed))
                for family in ("csmc", "rbf"):
                    candidate = selected[family]
                    for variant in ("unadapted", "adapted"):
                        eta = tf.constant(candidate["eta"], tf.float64)
                        if variant == "unadapted":
                            eta = tf.zeros_like(eta)
                        result = self.evaluate_message(family, candidate["capacity"], eta, inputs,
                                                       284043000 + dataset_index)
                        row = dict(dataset=dataset, replicate=replicate, particle_seed=particle_seed,
                            arm=f"{family}_{variant}", **result,
                            **{key: reference[dataset, replicate][key]
                               for key in ("observation_regime", "compartment_regime")})
                        append(self.output / "holdout.jsonl", row)
                        rows.append(serializable(row))
            print(f"restricted_holdout_dataset={dataset_index + 1}/64", flush=True)
        return analyze_restricted_comparisons(neural_rows, rows, datasets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--neural-study", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, required=True)
    arguments = parser.parse_args()
    arguments.components, arguments.width, arguments.beta = 2, 16, .5
    arguments.seed, arguments.learning_rate, arguments.updates = 263044001, .001, 24
    arguments.resume, arguments.mode, arguments.profile_update = None, "restricted_baselines", -1
    study = RestrictedStudy(arguments)
    try:
        selected = study.train_adapters()
        result = study.evaluate_holdout(selected)
        result["selected"] = selected
        result.update(wall_seconds=time.monotonic() - study.started,
            source_check=verify_source_snapshot(study.repository, study.source),
            model_score_accuracy_evaluated=False, default_ready=False)
        save(arguments.output / "result.json", result)
        if not result["source_check"]["unchanged"]:
            raise ValueError("Package source changed during restricted comparison")
        if result["missing_regimes"]:
            raise ValueError("Required comparison regimes are absent")
    except Exception as error:
        save(arguments.output / "failure.json", dict(error=repr(error), wall_seconds=time.monotonic() - study.started))
        raise


if __name__ == "__main__":
    main()
