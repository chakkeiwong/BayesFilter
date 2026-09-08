"""Run the predeclared independent C2 alpha/nu DMIS pilot.

This is a bounded calibration diagnostic for the frozen-proposal C2 route.  It
does not alter or reopen the completed fixed-half result.  The pilot uses a
fresh retained-TT fit, an independent half-mixture objective bank, a second
validation bank, and the exact equal-bank component variance formula.  All
runtime numerical work is TensorFlow; JSON/Markdown assembly is host-side
reporting only.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
PLAN = ROOT / "docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-implementation-test-plan-2026-08-29.md"
PLAN_REVIEW = ROOT / "docs/plans/bayesfilter-c2-ukf-guided-defensive-tt-dmis-plan-review-2026-08-29.md"
LATEX = ROOT / "docs/benchmarks/artifacts/c2_completion_20260824/attempt05/ukf_guided_defensive_tt_dmis_analytical_gradient.tex"
N = 4
GAMMA = 0.6
BETA = 0.4
SIGMA = 1.0
DEGREE = 6
RANK = 6
FIT_ROWS = 8192
SWEEPS = 32
RIDGE = 1e-10
TAU = 1e-6
PILOT_PARTICLES = 4096
PILOT_BANK = PILOT_PARTICLES // 2
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_CHUNK = 100
ALPHAS = tuple(0.10 + 0.025 * index for index in range(33))
NU_VALUES = (4.0, 8.0, 16.0, math.inf)
NU_LABELS = ("nu4", "nu8", "nu16", "gaussian_limit")
REFERENCE_SEED = 991001
PILOT_SEED = 991101
VALIDATION_SEED = 991201
BOOTSTRAP_SEED = 991301
DTYPE_NAME = "float64"
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fixture", default=str(FIXTURE))
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonable(value: object) -> object:
    if hasattr(value, "numpy"):
        materialized = value.numpy()
        if hasattr(materialized, "tolist"):
            return _jsonable(materialized.tolist())
        return _jsonable(materialized)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"refusing to serialize non-finite float {value!r}")
        return value
    if hasattr(value, "item"):
        return _jsonable(value.item())
    raise TypeError(f"unsupported JSON value {type(value).__name__}")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot compute a quantile of an empty sequence")
    position = (len(ordered) - 1) * float(probability)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _frozen_adapter(tf, adapter_class, fixture: Mapping[str, object]):
    transition = tf.constant(fixture["transition_matrix"], tf.float64)
    process_chol = tf.linalg.cholesky(tf.constant(fixture["process_covariance"], tf.float64))
    initial_chol = tf.linalg.cholesky(tf.constant(fixture["stationary_covariance"], tf.float64))
    beta = tf.constant(float(fixture["beta"]), tf.float64)

    def _mvn(states, means, chol):
        difference = states - means
        whitened = tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(difference), lower=True))
        return -0.5 * (
            tf.constant(N * math.log(2.0 * math.pi), tf.float64)
            + tf.reduce_sum(tf.square(whitened), axis=1)
        ) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))

    def transition_log_density(current, previous):
        return _mvn(current, tf.linalg.matmul(previous, transition, transpose_b=True), process_chol)

    def observation_log_density(current, observation):
        observed = tf.ensure_shape(tf.convert_to_tensor(observation, tf.float64), [N])
        return tf.reduce_sum(
            -0.5 * tf.constant(math.log(2.0 * math.pi), tf.float64)
            - tf.math.log(beta)
            - 0.5 * current
            - 0.5 * tf.square(observed)[None, :] * tf.exp(-current) / tf.square(beta),
            axis=1,
        )

    def initial_log_density(current):
        return _mvn(current, tf.zeros_like(current), initial_chol)

    return adapter_class(
        state_dim=N,
        transition_log_density=transition_log_density,
        observation_log_density=observation_log_density,
        initial_log_density=initial_log_density,
    )


def _frozen_hint_factory(tf, fixture: Mapping[str, object], horizon: int):
    hints = fixture["moment_hints"][:horizon]
    next_index = {"value": 0}

    def initial_hint(_observation):
        if next_index["value"] != 0:
            raise RuntimeError("frozen hints were consumed out of order")
        next_index["value"] = 1
        return tf.constant(hints[0]["mean"], tf.float64), tf.constant(hints[0]["covariance"], tf.float64)

    def predictive_hint(time_index, _observation):
        if int(time_index) != next_index["value"]:
            raise RuntimeError("frozen hints were consumed out of order")
        next_index["value"] += 1
        return tf.constant(hints[time_index]["mean"], tf.float64), tf.constant(hints[time_index]["covariance"], tf.float64)

    return initial_hint, predictive_hint


def _save_snapshot(tf, snapshot_api, snapshot, output_root: Path) -> Mapping[str, object]:
    metadata, tensors = snapshot_api.gaussian_xla_retained_proposal_snapshot_parts(snapshot)
    step_root = output_root / "snapshots" / f"t{snapshot.time_index:02d}"
    step_root.mkdir(parents=True)
    tensor_rows = {}
    for name, tensor in tensors.items():
        path = step_root / f"{name}.tensor"
        tf.io.write_file(str(path), tf.io.serialize_tensor(tensor))
        tensor_rows[name] = {"path": str(path.relative_to(output_root)), "sha256": _sha256_file(path)}
    metadata = dict(metadata)
    metadata["snapshot_fingerprint"] = snapshot_api.gaussian_xla_retained_proposal_snapshot_fingerprint(snapshot)
    metadata["tensor_files"] = tensor_rows
    metadata_path = step_root / "metadata.json"
    _write_json(metadata_path, metadata)
    return {
        "time_index": int(snapshot.time_index),
        "snapshot_fingerprint": metadata["snapshot_fingerprint"],
        "metadata_path": str(metadata_path.relative_to(output_root)),
        "metadata_sha256": _sha256_file(metadata_path),
        "tensor_count": len(tensors),
    }


def _sample_ancestors(tf, log_auxiliary, count: int, seed: tuple[int, int]):
    probabilities = tf.exp(tf.convert_to_tensor(log_auxiliary, tf.float64))
    cdf = tf.math.cumsum(probabilities)
    cdf = tf.concat([cdf[:-1], tf.ones([1], tf.float64)], axis=0)
    uniforms = tf.random.stateless_uniform([count], [int(seed[0]), int(seed[1])], dtype=tf.float64)
    return tf.searchsorted(cdf, uniforms, side="right", out_type=tf.int32)


def _gaussian_log_density(tf, states, means, chol):
    centered = states - means
    whitened = tf.transpose(tf.linalg.triangular_solve(chol, tf.transpose(centered), lower=True))
    dimension = int(states.shape[1])
    return -0.5 * (
        tf.constant(dimension * math.log(2.0 * math.pi), tf.float64)
        + tf.reduce_sum(tf.square(whitened), axis=1)
    ) - tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol)))


def _draw_defensive(tf, proposal, parents, count: int, seed: tuple[int, int], nu: float):
    if math.isinf(nu):
        normal = tf.random.stateless_normal([count, N], [int(seed[0]), int(seed[1])], dtype=tf.float64)
        means = proposal.conditional_mean(parents)
        chol = tf.linalg.cholesky(proposal.posterior_covariance)
        states = means + tf.einsum("ij,nj->ni", chol, normal)
        log_density = _gaussian_log_density(tf, states, means, chol)
        return states, log_density
    sampled = proposal.sample_with_seed(parents, count, seed, jit_compile=True)
    return sampled["physical_points"], sampled["physical_log_density"]


def _draw_bank(
    tf,
    tt_proposal,
    defensive_proposal,
    parent_states,
    log_weights,
    log_auxiliary,
    count: int,
    seed: tuple[int, int],
    nu: float,
):
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import (
        stateless_proposal_random_inputs,
    )

    ancestors = _sample_ancestors(tf, log_auxiliary, count, (seed[0], seed[1] + 1))
    parents = tf.gather(parent_states, ancestors)
    tt_inputs = stateless_proposal_random_inputs(
        tt_proposal, count, (seed[0], seed[1] + 2)
    )
    tt_sampled = tt_proposal.compiled_sampler(count, jit_compile=True)(*tt_inputs)
    tt_states = tf.ensure_shape(tt_sampled["physical_points"], [count, N])
    tt_log_q = tf.ensure_shape(tt_sampled["physical_log_density"], [count])
    defensive_states, defensive_log_q = _draw_defensive(
        tf, defensive_proposal, parents, count, (seed[0], seed[1] + 20), nu
    )
    return {
        "ancestors": ancestors,
        "parents": parents,
        "tt_states": tt_states,
        "tt_log_q": tt_log_q,
        "defensive_states": tf.ensure_shape(defensive_states, [count, N]),
        "defensive_log_q": tf.ensure_shape(defensive_log_q, [count]),
        "log_weights": tf.gather(log_weights, ancestors),
        "log_auxiliary": tf.gather(log_auxiliary, ancestors),
    }


def _prefix_context(tf, model, branch, observations, theta, time_index: int):
    from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
        prepare_frozen_proposal_apf_program,
        prepare_frozen_proposal_branch,
    )

    prefix = prepare_frozen_proposal_branch(
        observations=observations[:time_index],
        states=branch.states[:time_index],
        initial_log_proposal_density=branch.initial_log_proposal_density,
        ancestors=branch.ancestors[: max(0, time_index - 1)],
        auxiliary_log_probabilities=branch.auxiliary_log_probabilities[: max(0, time_index - 1)],
        transition_log_proposal_density=branch.transition_log_proposal_density[: max(0, time_index - 1)],
        initial_log_base_mass=branch.initial_log_base_mass,
        transition_log_base_mass=branch.transition_log_base_mass[: max(0, time_index - 1)],
    )
    evaluated = prepare_frozen_proposal_apf_program(model, prefix).evaluate(theta)
    return (
        branch.states[time_index - 1],
        evaluated["final_log_weights"],
        branch.auxiliary_log_probabilities[time_index - 1],
    )


def _component_logs(
    tf,
    model,
    theta,
    observation,
    bank,
    time_index: int,
    tt_proposal,
    defensive_proposal,
    nu: float,
):
    tt_parents = bank["parents"]
    d_parents = bank["parents"]
    tt_states = bank["tt_states"]
    d_states = bank["defensive_states"]
    if math.isinf(nu):
        gaussian_chol = tf.linalg.cholesky(defensive_proposal.posterior_covariance)
        tt_log_d = _gaussian_log_density(
            tf,
            tt_states,
            defensive_proposal.conditional_mean(tt_parents),
            gaussian_chol,
        )
    else:
        tt_log_d = defensive_proposal.log_density(tt_states, tt_parents)
    d_log_t = tt_proposal.physical_log_density(d_states)
    tt_fg = model.transition_log_density(theta, tt_parents, tt_states, time_index) + model.observation_log_density(theta, tt_states, observation, time_index)
    d_fg = model.transition_log_density(theta, d_parents, d_states, time_index) + model.observation_log_density(theta, d_states, observation, time_index)
    return {
        "log_pi": tf.concat([bank["log_weights"] + tt_fg, bank["log_weights"] + d_fg], axis=0),
        "log_a": tf.concat([bank["log_auxiliary"], bank["log_auxiliary"]], axis=0),
        "log_q_t": tf.concat([bank["tt_log_q"], d_log_t], axis=0),
        "log_q_d": tf.concat([tt_log_d, bank["defensive_log_q"]], axis=0),
        "tt_log_pi": bank["log_weights"] + tt_fg,
        "d_log_pi": bank["log_weights"] + d_fg,
    }


def _objective_curve(tf, logs: Mapping[str, object], count: int):
    log_pi = logs["log_pi"]
    log_a = logs["log_a"]
    log_q_t = logs["log_q_t"]
    log_q_d = logs["log_q_d"]
    log_m = tf.reduce_logsumexp(tf.stack([log_q_t, log_q_d], axis=1), axis=1) - math.log(2.0)
    values = []
    second_values = []
    delta = tf.exp(log_q_d) - tf.exp(log_q_t)
    delta_squared = tf.square(delta)
    for alpha in ALPHAS:
        log_alpha = math.log(alpha)
        log_one_minus = math.log1p(-alpha)
        log_mix = tf.reduce_logsumexp(tf.stack([log_one_minus + log_q_t, log_alpha + log_q_d], axis=1), axis=1)
        log_term = 2.0 * log_pi - 2.0 * log_a - log_m - log_mix
        values.append(tf.exp(tf.reduce_logsumexp(log_term) - math.log(count)))
        log_second = tf.where(
            delta_squared > 0.0,
            math.log(2.0) + 2.0 * log_pi + tf.math.log(delta_squared) - 2.0 * log_a - log_m - 3.0 * log_mix,
            tf.fill(tf.shape(log_mix), tf.constant(float("-inf"), tf.float64)),
        )
        second_values.append(tf.exp(tf.reduce_logsumexp(log_second) - math.log(count)))
    return tf.stack(values), tf.stack(second_values)


def _validation_curve(tf, logs: Mapping[str, object], bank_count: int):
    values = []
    h_t_values = []
    h_d_values = []
    split = bank_count
    for alpha in ALPHAS:
        log_mix = tf.reduce_logsumexp(tf.stack([math.log1p(-alpha) + logs["log_q_t"], math.log(alpha) + logs["log_q_d"]], axis=1), axis=1)
        h = tf.exp(logs["log_pi"] - logs["log_a"] - log_mix)
        h_t = h[:split]
        h_d = h[split:]
        mean_t = tf.reduce_mean(h_t)
        mean_d = tf.reduce_mean(h_d)
        var_t = tf.reduce_sum(tf.square(h_t - mean_t)) / tf.cast(bank_count - 1, tf.float64)
        var_d = tf.reduce_sum(tf.square(h_d - mean_d)) / tf.cast(bank_count - 1, tf.float64)
        values.append(((1.0 - alpha) ** 2 / bank_count) * var_t + (alpha**2 / bank_count) * var_d)
        h_t_values.append(h_t)
        h_d_values.append(h_d)
    return tf.stack(values), tf.stack(h_t_values, axis=1), tf.stack(h_d_values, axis=1)


def _bootstrap_variance_curves(tf, h_t, h_d, *, seed: tuple[int, int]):
    boot_rows = []
    count = int(h_t.shape[0])
    for start in range(0, BOOTSTRAP_REPLICATES, BOOTSTRAP_CHUNK):
        chunk = min(BOOTSTRAP_CHUNK, BOOTSTRAP_REPLICATES - start)
        indices_t = tf.random.stateless_uniform([chunk, count], [seed[0], seed[1] + start], maxval=count, dtype=tf.int32)
        indices_d = tf.random.stateless_uniform([chunk, count], [seed[0], seed[1] + 100000 + start], maxval=count, dtype=tf.int32)
        sampled_t = tf.gather(h_t, indices_t)
        sampled_d = tf.gather(h_d, indices_d)
        mean_t = tf.reduce_mean(sampled_t, axis=1)
        mean_d = tf.reduce_mean(sampled_d, axis=1)
        var_t = tf.reduce_sum(tf.square(sampled_t - mean_t[:, None, :]), axis=1) / float(count - 1)
        var_d = tf.reduce_sum(tf.square(sampled_d - mean_d[:, None, :]), axis=1) / float(count - 1)
        rows = []
        for index, alpha in enumerate(ALPHAS):
            rows.append(((1.0 - alpha) ** 2 / count) * var_t[:, index] + (alpha**2 / count) * var_d[:, index])
        boot_rows.append(tf.stack(rows, axis=1).numpy())
    return [list(map(float, row)) for row in itertools.chain.from_iterable(boot_rows)]


def _pilot_time(tf, model, observations, theta, time_index, context, tt_proposal, defensive_proposal, nu, seed_base):
    parent_states, log_weights, log_auxiliary = context
    pilot_tt = _draw_bank(tf, tt_proposal, defensive_proposal, parent_states, log_weights, log_auxiliary, PILOT_BANK, (seed_base, 11), nu)
    pilot_logs = _component_logs(
        tf, model, theta, observations[time_index], pilot_tt, time_index,
        tt_proposal, defensive_proposal, nu,
    )
    objective, second_derivative = _objective_curve(tf, pilot_logs, PILOT_PARTICLES)
    validation_tt = _draw_bank(tf, tt_proposal, defensive_proposal, parent_states, log_weights, log_auxiliary, PILOT_BANK, (seed_base + 1, 11), nu)
    validation_logs = _component_logs(
        tf, model, theta, observations[time_index], validation_tt, time_index,
        tt_proposal, defensive_proposal, nu,
    )
    validation, h_t, h_d = _validation_curve(tf, validation_logs, PILOT_BANK)
    bootstrap = _bootstrap_variance_curves(tf, h_t, h_d, seed=(seed_base + 2, 17))
    finite = bool(tf.reduce_all(tf.math.is_finite(tf.concat([objective, second_derivative, validation], axis=0))).numpy())
    return {
        "objective": [float(value) for value in objective.numpy()],
        "analytical_second_derivative": [float(value) for value in second_derivative.numpy()],
        "validation_variance": [float(value) for value in validation.numpy()],
        "bootstrap_variance": bootstrap,
        "finite": finite,
    }


def _select_pair(curves):
    times = sorted(curves[NU_LABELS[0]])
    point = [
        [curves[label][time_index]["validation_variance"] for time_index in times]
        for label in NU_LABELS
    ]
    aggregate = [[statistics.fmean(point[nu_index][time_index][alpha_index] for time_index in range(len(times))) for alpha_index in range(len(ALPHAS))] for nu_index in range(len(NU_LABELS))]
    selected_nu_index, selected_alpha_index = min(
        ((nu_index, alpha_index) for nu_index in range(len(NU_LABELS)) for alpha_index in range(len(ALPHAS))),
        key=lambda pair: aggregate[pair[0]][pair[1]],
    )
    baseline_nu_index = NU_LABELS.index("nu8")
    baseline_alpha_index = min(range(len(ALPHAS)), key=lambda index: abs(ALPHAS[index] - 0.5))
    bootstrap = [
        [curves[label][time_index]["bootstrap_variance"] for time_index in times]
        for label in NU_LABELS
    ]
    aggregate_bootstrap = []
    for replicate in range(BOOTSTRAP_REPLICATES):
        aggregate_bootstrap.append([
            statistics.fmean(bootstrap[nu_index][time_index][replicate][alpha_index] for time_index in range(len(times)))
            for nu_index in range(len(NU_LABELS)) for alpha_index in range(len(ALPHAS))
        ])
    selected_flat = selected_nu_index * len(ALPHAS) + selected_alpha_index
    baseline_flat = baseline_nu_index * len(ALPHAS) + baseline_alpha_index
    selected_boot = [row[selected_flat] for row in aggregate_bootstrap]
    baseline_boot = [row[baseline_flat] for row in aggregate_bootstrap]
    ratios = [selected_value / baseline_value for selected_value, baseline_value in zip(selected_boot, baseline_boot) if baseline_value > 0.0]
    minimizer_stability = sum(
        int((min(range(len(aggregate_bootstrap[0])), key=lambda index: row[index]) // len(ALPHAS)) == selected_nu_index and abs((min(range(len(aggregate_bootstrap[0])), key=lambda index: row[index]) % len(ALPHAS)) - selected_alpha_index) <= 1)
        for row in aggregate_bootstrap
    ) / BOOTSTRAP_REPLICATES
    selected_mean = aggregate[selected_nu_index][selected_alpha_index]
    selected_rse = statistics.stdev(selected_boot) / statistics.fmean(selected_boot) if len(selected_boot) > 1 and statistics.fmean(selected_boot) > 0.0 else float("inf")
    ratio_q95 = _quantile(ratios, 0.95) if ratios else float("inf")
    baseline_positive = aggregate[baseline_nu_index][baseline_alpha_index] > 0.0
    finite = all(math.isfinite(value) and value >= 0.0 for row in aggregate for value in row) and all(math.isfinite(value) for value in ratios)
    selection_pass = bool(
        finite
        and baseline_positive
        and selected_rse <= 0.20
        and minimizer_stability >= 0.80
        and ratio_q95 < 1.0
    )
    return {
        "time_indices": times,
        "alpha_grid": list(ALPHAS),
        "aggregate_validation_variance": aggregate,
        "selected_pair": {"nu": NU_LABELS[selected_nu_index], "alpha": ALPHAS[selected_alpha_index]},
        "fixed_half_baseline": {"nu": "nu8", "alpha": 0.5, "variance": aggregate[baseline_nu_index][baseline_alpha_index]},
        "selected_variance": selected_mean,
        "selected_bootstrap_relative_standard_error": selected_rse,
        "bootstrap_minimizer_stability": minimizer_stability,
        "variance_ratio_to_fixed_half_95th_percentile": ratio_q95,
        "bootstrap_ratio_count": len(ratios),
        "selection_pass": selection_pass,
        "selection_status": "selected_pair_meets_predeclared_pilot_conditions" if selection_pass else "fallback_fixed_half",
        "final_banks_replayed": False,
        "nonclaim": "pilot selection is not retroactively applied to the completed fixed-half serious run",
    }


def _markdown(result: Mapping[str, object]) -> str:
    selection = result["selection"]
    all_finite = all(
        bool(row["finite"])
        for rows in result["curves"].values()
        for row in rows.values()
    )
    all_second_nonnegative = all(
        value >= 0.0 and math.isfinite(value)
        for rows in result["curves"].values()
        for row in rows.values()
        for value in row["analytical_second_derivative"]
    )
    lines = [
        "# C2 UKF-Guided Defensive TT-DMIS Alpha/Nu Pilot", "",
        "This is an independent calibration diagnostic. It does not rewrite the completed fixed-half serious run.", "",
        f"Fresh retained fit: `{result['fit']['snapshot_count']}` transition snapshots; pilot/validation draws per `(nu,time)` are `{PILOT_PARTICLES}` each, split into two `{PILOT_BANK}`-draw banks.", "",
        f"Engineering setup: GPU/XLA `{result['run']['jit_compile']}`, dtype `{result['run']['dtype']}`, memory growth verified `{result['run']['memory_growth_verified']}`.", "",
        f"Pilot validity: all objective/validation curves finite `{all_finite}`; analytical second-derivative estimates nonnegative `{all_second_nonnegative}`.", "",
        "## Selection", "",
        f"Selection status: `{selection['selection_status']}`. Candidate `{selection['selected_pair']['nu']}` at alpha `{selection['selected_pair']['alpha']}`; fixed-half variance `{selection['fixed_half_baseline']['variance']:.10g}`; candidate variance `{selection['selected_variance']:.10g}`.", "",
        f"Bootstrap relative standard error: `{selection['selected_bootstrap_relative_standard_error']:.6g}`; minimizer stability: `{selection['bootstrap_minimizer_stability']:.6g}`; 95th-percentile variance ratio: `{selection['variance_ratio_to_fixed_half_95th_percentile']:.6g}`.", "",
        "The selected pair, even if nominated, was not replayed into final banks because the fixed-half serious run was completed before this follow-up pilot. No default, posterior, pseudo-marginal, HMC, exactness, or superiority claim is made.", "",
        "## Artifacts", "",
        f"Raw JSON: `{result['paths']['result_json']}`", "",
        f"Snapshot/proposal manifest: `{result['paths']['proposal_manifest']}`", "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = _parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root {output_root}")
    output_root.mkdir(parents=True)
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-c2-ukf-guided-dmis")
    sys.path.insert(0, str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    import tensorflow_probability as tfp
    from bayesfilter.highdim.c2_gaussian_hermite_proposal_tf import retained_proposal_from_transition_snapshot
    from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
        C2StochasticVolatilityFrozenAPFModel,
        compile_c2_dmis_proposal_branch,
        transformed_student_proposals,
    )
    from bayesfilter.highdim.squared_tt_engine_v0_tf import DensityKernelAdapter, EngineConfig
    import bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf as snapshot_api

    fixture_path = Path(args.fixture).resolve()
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    if fixture.get("schema_id") != "bayesfilter.c2_sv_frozen_fixture.v1" or int(fixture["state_dimension"]) != N or int(fixture["horizon"]) < 20:
        raise ValueError("fixture identity does not match the C2 pilot scope")
    logical_gpus = tuple(tf.config.list_logical_devices("GPU"))
    if not logical_gpus:
        raise RuntimeError("no logical TensorFlow GPU")
    with tf.device("/GPU:0"):
        placement_probe = tf.reduce_sum(tf.ones([32], tf.float64))
    if "GPU" not in placement_probe.device.upper():
        raise RuntimeError("GPU placement probe did not execute on GPU")
    observations = tf.constant(fixture["observations"][:20], tf.float64)
    theta = tf.constant([GAMMA, math.log(BETA)], tf.float64)
    transition = tf.constant(fixture["transition_matrix"], tf.float64)
    coupling = transition - GAMMA * tf.eye(N, dtype=tf.float64)
    model = C2StochasticVolatilityFrozenAPFModel(coupling_matrix=coupling, sigma=SIGMA)
    adapter = _frozen_adapter(tf, DensityKernelAdapter, fixture)
    initial_hint, predictive_hint = _frozen_hint_factory(tf, fixture, 20)
    config = EngineConfig(
        basis_degree=DEGREE, rank=RANK, row_count=FIT_ROWS, sweeps=SWEEPS,
        ridge=RIDGE, tau=TAU, coordinate_half_width=3.0,
        seed=98000 + 100 * N + 10 * DEGREE + RANK, row_design="sobol",
    )
    run_identity_payload = {
        "fixture_sha256": _sha256_file(fixture_path), "role": "independent_alpha_nu_pilot",
        "horizon": 20, "degree": DEGREE, "rank": RANK, "fit_rows": FIT_ROWS,
        "sweeps": SWEEPS, "ridge": RIDGE, "tau": TAU,
        "pilot_particles": PILOT_PARTICLES, "alpha_grid": ALPHAS, "nu_labels": NU_LABELS,
    }
    run_identity = hashlib.sha256(json.dumps(run_identity_payload, sort_keys=True).encode("utf-8")).hexdigest()
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    direct_value, direct_diagnostics, snapshots = snapshot_api.run_value_filter_branch_axis_gaussian_xla_retained_proposal_diagnostic(
        adapter, observations, config, predictive_moment_hint=predictive_hint,
        initial_moment_hint=initial_hint, capture_steps=tuple(range(1, 20)),
        run_identity=run_identity, defensive_nu=8.0,
    )
    snapshot_rows = [_save_snapshot(tf, snapshot_api, snapshots[t], output_root) for t in range(1, 20)]
    tt_proposals = tuple(retained_proposal_from_transition_snapshot(snapshots[t]) for t in range(1, 20))
    defensive_baseline = transformed_student_proposals(model=model, observations=observations, theta_reference=theta, nu=8.0)
    prefix_compilation = compile_c2_dmis_proposal_branch(
        model=model, observations=observations, theta_reference=theta,
        transition_proposals=tt_proposals, defensive_proposals=defensive_baseline,
        particle_count=PILOT_PARTICLES, seed=REFERENCE_SEED, alpha=0.5, nu=8.0,
        jit_compile_sampler=True,
    )
    curves = {label: {} for label in NU_LABELS}
    for nu_index, (label, nu) in enumerate(zip(NU_LABELS, NU_VALUES)):
        defensive = transformed_student_proposals(model=model, observations=observations, theta_reference=theta, nu=nu if math.isfinite(nu) else 8.0)
        for time_index in range(1, 20):
            context = _prefix_context(tf, model, prefix_compilation.branch, observations, theta, time_index)
            defensive_proposal = defensive[time_index - 1]
            seed_base = PILOT_SEED + 100000 * nu_index + 1000 * time_index
            curves[label][time_index] = _pilot_time(
                tf, model, observations, theta, time_index, context,
                tt_proposals[time_index - 1], defensive_proposal, nu, seed_base,
            )
            _write_json(output_root / "pilot_partial.json", {"curves": curves})
    selection = _select_pair(curves)
    proposal_manifest = {
        "schema_id": "bayesfilter.c2_ukf_guided_defensive_tt_dmis_pilot_manifest.v1",
        "run_identity": run_identity, "fixture_sha256": _sha256_file(fixture_path),
        "snapshot_rows": snapshot_rows, "particle_count": PILOT_PARTICLES,
        "pilot_bank_count": PILOT_BANK, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "alpha_grid": ALPHAS, "nu_candidates": NU_LABELS,
        "proposal_density": "complete_outer_mixture_for_every_sample",
        "joint_factor": "(W_over_a)^2_in_J_pilot",
        "selection": selection,
    }
    _write_json(output_root / "proposal_manifest.json", proposal_manifest)
    result = {
        "schema_id": "bayesfilter.c2_ukf_guided_defensive_tt_dmis_pilot_result.v1",
        "run": {
            "started_at_utc": started_at.isoformat(),
            "wall_seconds": time.perf_counter() - started,
            "dtype": DTYPE_NAME, "jit_compile": True,
            "memory_growth_verified": True, "logical_gpus": [str(device) for device in logical_gpus],
            "placement_probe_device": placement_probe.device,
            "tensorflow_version": tf.__version__, "tensorflow_probability_version": tfp.__version__,
            "memory_policy": memory_policy,
        },
        "fit": {"snapshot_count": len(snapshots), "direct_value": direct_value, "diagnostics": direct_diagnostics},
        "selection": selection, "curves": curves,
        "paths": {"result_json": str((output_root / "result.json").relative_to(ROOT)), "proposal_manifest": str((output_root / "proposal_manifest.json").relative_to(ROOT))},
        "nonclaims": ["exact_pseudo_marginal_likelihood", "posterior_correctness", "HMC_readiness", "default_readiness", "statistical_superiority"],
    }
    _write_json(output_root / "result.json", result)
    (output_root / "result.md").write_text(_markdown(result), encoding="utf-8")
    manifest = {
        "schema_id": "bayesfilter.c2_ukf_guided_defensive_tt_dmis_pilot_run_manifest.v1",
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started,
        "command": " ".join([sys.executable, *sys.argv]),
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip(),
        "fixture_sha256": _sha256_file(fixture_path), "plan_file": str(PLAN.relative_to(ROOT)), "plan_review_file": str(PLAN_REVIEW.relative_to(ROOT)), "latex_file": str(LATEX.relative_to(ROOT)),
        "source_sha256": {str(path.relative_to(ROOT)): _sha256_file(path) for path in (Path(__file__).resolve(), PLAN, PLAN_REVIEW, LATEX, ROOT / "bayesfilter/highdim/c2_sv_frozen_proposal_apf_tf.py", ROOT / "bayesfilter/highdim/c2_transformed_observation_student_proposal_tf.py", ROOT / "bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py")},
        "output_root": str(output_root.relative_to(ROOT)), "trust_basis": TRUST_BASIS,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", "unset"),
        "tensorflow_version": tf.__version__, "tensorflow_probability_version": tfp.__version__,
        "logical_gpus": [str(device) for device in logical_gpus],
        "dtype": DTYPE_NAME, "jit_compile": True, "memory_policy": memory_policy,
        "budget": {"pilot_attempt": 1, "max_attempts": 2, "declared_gpu_minutes": 60},
    }
    _write_json(output_root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
