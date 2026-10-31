"""Compiled SV panel recurrences using the existing quadrature components.

These component-enumerated comparators are not batch-native NeuTra targets.
Clouds and model parameters are tensor inputs; only shapes and branch settings
specialize the bounded function cache.
"""

from functools import lru_cache
from types import SimpleNamespace

import tensorflow as tf

from bayesfilter.nonlinear.fixed_sgqf_compiled_tf import fixed_sgqf_tensor_result
from bayesfilter.nonlinear.svd_cut_tf import tf_svd_cut4_log_likelihood


@lru_cache(maxsize=32)
def augmented_sgqf_program(dates, width, points, branch_config, *, jit_compile=True):
    from bayesfilter.highdim.sv_mixture_cut4 import (
        _actual_transformed_sv_augmented_noise_fixed_sgqf_model,
    )

    d = tf.float64

    @tf.function(
        input_signature=[
            tf.TensorSpec([dates, width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([points, 2], d),
            tf.TensorSpec([points], d),
            tf.TensorSpec([], d),
        ],
        jit_compile=jit_compile,
        autograph=False,
    )
    def run(y, gamma, beta, sigma, nodes, weights, observation_variance_floor):
        cloud = SimpleNamespace(points=nodes, weights=weights)

        def coordinate(axis):
            mean = tf.constant(0.0, d)
            variance = sigma[axis] ** 2 / (1.0 - gamma[axis] ** 2)
            history = tf.zeros([dates, 3], d)

            def step(t, mean, variance, status, failure_date, history):
                model = _actual_transformed_sv_augmented_noise_fixed_sgqf_model(
                    current_mean=mean,
                    current_variance=variance,
                    gamma=gamma[axis],
                    beta=beta[axis],
                    sigma=sigma[axis],
                    time_index=t,
                    observation_variance_floor=observation_variance_floor,
                )
                output = fixed_sgqf_tensor_result(
                    tf.reshape(y[t, axis], [1, 1]),
                    model,
                    cloud,
                    branch_config,
                    jit_compile=False,
                )
                mean = output["filtered_mean"][0]
                variance = output["filtered_covariance"][0, 0]
                status = output["status_code"]
                failure_date = tf.where(status != 0, t, failure_date)
                row = tf.stack([output["log_likelihood"], mean, variance])
                history = tf.tensor_scatter_nd_update(
                    history, tf.reshape(t, [1, 1]), row[None]
                )
                return t + 1, mean, variance, status, failure_date, history

            result = tf.while_loop(
                lambda t, _m, _v, status, _f, _h: (t < dates) & (status == 0),
                step,
                (
                    tf.constant(0),
                    mean,
                    variance,
                    tf.constant(0),
                    tf.constant(-1),
                    history,
                ),
                maximum_iterations=dates,
                parallel_iterations=1,
            )
            return result[5], result[3], result[4]

        history, status, failure_date = tf.map_fn(
            coordinate,
            tf.range(width),
            fn_output_signature=(
                tf.TensorSpec([dates, 3], d),
                tf.TensorSpec([], tf.int32),
                tf.TensorSpec([], tf.int32),
            ),
            parallel_iterations=1,
        )
        return {
            "log_normalizers": tf.reduce_sum(history[:, :, 0], axis=0),
            "mean_path": tf.transpose(history[:, :, 1]),
            "covariance_path": tf.linalg.diag(tf.transpose(history[:, :, 2])),
            "status_code": status,
            "failure_date": failure_date,
        }

    return run


@lru_cache(maxsize=32)
def mixture_program(
    dates,
    width,
    components,
    points,
    branch_config,
    *,
    method="sgqf",
    with_score=False,
    scalar_collapse=False,
    jit_compile=True,
):
    """Preserve component integration and posterior moment collapse in one graph."""
    from bayesfilter.highdim import sv_mixture_cut4 as sv

    if method not in ("sgqf", "cut4", "kalman") or with_score and method != "sgqf":
        raise ValueError("unsupported mixture integration/score combination")
    if scalar_collapse and (width != 1 or method != "cut4"):
        raise ValueError("scalar collapse belongs to the scalar CUT4 comparator")
    d = tf.float64
    parameter_dim = 2 * width
    tuple_count = components**width
    # Fixed result fields, independent of date/component/parameter counts.
    shapes = (
        [],
        [parameter_dim],
        [width],
        [width, width],
        [parameter_dim, width],
        [parameter_dim, width, width],
        [tuple_count],
    )
    component_signature = tuple(tf.TensorSpec(shape, d) for shape in shapes[:6]) + (
        tf.TensorSpec([], tf.int32),
    )

    @tf.function(
        input_signature=[
            tf.TensorSpec([dates, width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([width], d),
            tf.TensorSpec([components], d),
            tf.TensorSpec([components], d),
            tf.TensorSpec([components], d),
            tf.TensorSpec([tuple_count, width], tf.int32),
            tf.TensorSpec([points, width], d),
            tf.TensorSpec([points], d),
            tf.TensorSpec([], d),
        ],
        jit_compile=jit_compile,
        autograph=False,
    )
    def run(
        z,
        gamma,
        beta,
        sigma,
        mixture_weights,
        mixture_means,
        mixture_variances,
        indices,
        nodes,
        weights,
        innovation_floor,
    ):
        mean = tf.zeros([width], d)
        covariance = tf.linalg.diag(sigma**2 / (1.0 - gamma**2))
        d_mean = tf.zeros([parameter_dim, width], d)
        d_covariance = (sv._gamma_seed_covariance_derivatives(gamma, sigma) if with_score
                        else tf.zeros([parameter_dim, width, width], d))
        cloud = SimpleNamespace(points=nodes, weights=weights)
        histories = tuple(tf.zeros([dates, *shape], d) for shape in shapes)
        transition = tf.linalg.diag(gamma)
        process_covariance = tf.linalg.diag(sigma**2)
        observation_offsets = tf.math.log(beta**2)[None] + tf.gather(mixture_means, indices)
        observation_covariances = tf.linalg.diag(tf.gather(mixture_variances, indices))
        component_log_priors = tf.reduce_sum(
            tf.math.log(tf.gather(mixture_weights, indices)), axis=1
        )

        def step(t, mean, covariance, d_mean, d_covariance, _status, histories):
            if method == "kalman":
                mean = tf.where(t == 0, mean, tf.linalg.matvec(transition, mean))
                covariance = tf.where(
                    t == 0,
                    covariance,
                    sv._symmetrize(
                        transition @ covariance @ tf.transpose(transition)
                        + process_covariance
                    ),
                )

            def component(component_index):
                index = indices[component_index]
                offset = tf.gather(mixture_means, index)
                noise = tf.gather(mixture_variances, index)
                status = tf.constant(0)
                score = tf.zeros([parameter_dim], d)
                dm = tf.zeros_like(d_mean)
                dc = tf.zeros_like(d_covariance)
                if method == "kalman":
                    value, cm, cc = sv._kalman_update_identity_observation(
                        mean,
                        covariance,
                        z[t],
                        observation_offsets[component_index],
                        observation_covariances[component_index],
                    )
                elif method == "cut4":
                    if scalar_collapse:
                        model = sv._transformed_sv_component_structural_model(
                            predictive_mean=mean,
                            predictive_covariance=covariance,
                            gamma=gamma[0],
                            beta=beta[0],
                            sigma=sigma[0],
                            mixture_mean=offset[0],
                            mixture_variance=noise[0],
                            time_index=t,
                        )
                    else:
                        model = sv._panel_transformed_sv_component_structural_model(
                            current_mean=mean,
                            current_covariance=covariance,
                            gamma=gamma,
                            beta=beta,
                            sigma=sigma,
                            mixture_means=offset,
                            mixture_variances=noise,
                            time_index=t,
                        )
                    value, means, covariances, _ = tf_svd_cut4_log_likelihood(
                        z[t][None],
                        model,
                        innovation_floor=innovation_floor,
                        return_filtered=True,
                        jit_compile=False,
                    )
                    cm, cc = means[0], covariances[0]
                else:
                    model = (
                        sv._panel_transformed_sv_component_fixed_sgqf_nonlinear_model(
                            current_mean=mean,
                            current_covariance=covariance,
                            gamma=gamma,
                            beta=beta,
                            sigma=sigma,
                            mixture_means=offset,
                            mixture_variances=noise,
                            time_index=t,
                        )
                    )
                    derivatives = None
                    if with_score:
                        derivatives = (
                            sv._panel_transformed_sv_component_fixed_sgqf_derivatives(
                                current_mean=mean,
                                current_covariance=covariance,
                                d_current_mean=d_mean,
                                d_current_covariance=d_covariance,
                                gamma=gamma,
                                beta=beta,
                                sigma=sigma,
                                time_index=t,
                            )
                        )
                    output = fixed_sgqf_tensor_result(
                        z[t][None],
                        model,
                        cloud,
                        branch_config,
                        derivatives,
                        jit_compile=False,
                    )
                    value, cm, cc = (
                        output["log_likelihood"],
                        output["filtered_mean"],
                        output["filtered_covariance"],
                    )
                    status = output["status_code"]
                    if with_score:
                        score, dm, dc = (
                            output["score"],
                            output["d_filtered_mean"],
                            output["d_filtered_covariance"],
                        )
                value += component_log_priors[component_index]
                return value, score, cm, cc, dm, dc, status

            values, scores, means, covariances, d_means, d_covariances, statuses = (
                tf.map_fn(
                    component,
                    tf.range(tuple_count),
                    fn_output_signature=component_signature,
                    parallel_iterations=1,
                )
            )
            log_normalizer = tf.reduce_logsumexp(values)
            normalized_weights = tf.exp(values - log_normalizer)
            if with_score:
                score = tf.reduce_sum(scores * normalized_weights[:, None], axis=0)
                mean, covariance, d_mean, d_covariance = (
                    sv._collapse_gaussian_components_with_derivatives(
                        normalized_weights=normalized_weights,
                        component_means=means,
                        component_covariances=covariances,
                        component_d_means=d_means,
                        component_d_covariances=d_covariances,
                        component_scores=scores,
                    )
                )
            else:
                score = tf.zeros([parameter_dim], d)
                if scalar_collapse:
                    mean = tf.linalg.matvec(tf.transpose(means), normalized_weights)
                    second = tf.reduce_sum(
                        normalized_weights * (covariances[:, 0, 0] + means[:, 0] ** 2)
                    )
                    covariance = tf.reshape(second - mean[0] ** 2, [1, 1])
                else:
                    log_normalizer, mean, covariance, normalized_weights = (
                        sv._collapse_gaussian_components(values, means, covariances)
                    )
            first_failure = tf.argmax(
                tf.cast(statuses != 0, tf.int32), output_type=tf.int32
            )
            status = statuses[first_failure]
            row = (
                log_normalizer,
                score,
                mean,
                covariance,
                d_mean,
                d_covariance,
                normalized_weights,
            )
            histories = tuple(
                tf.tensor_scatter_nd_update(history, tf.reshape(t, [1, 1]), value[None])
                for history, value in zip(histories, row, strict=True)
            )
            return t + 1, mean, covariance, d_mean, d_covariance, status, histories

        result = tf.while_loop(
            lambda t, _m, _c, _dm, _dc, status, _h: (t < dates) & (status == 0),
            step,
            (
                tf.constant(0),
                mean,
                covariance,
                d_mean,
                d_covariance,
                tf.constant(0),
                histories,
            ),
            maximum_iterations=dates,
            parallel_iterations=1,
        )
        history = result[6]
        return {
            "log_normalizers": history[0],
            "score": tf.reduce_sum(history[1], axis=0),
            "mean_path": history[2],
            "covariance_path": history[3],
            "d_mean_path": history[4],
            "d_covariance_path": history[5],
            "component_weights": history[6],
            "status_code": result[5],
        }

    return run


def validate_cloud_mass(cloud, label):
    if abs(cloud.weight_total - 1.0) > max(cloud.zero_weight_tolerance, 1e-12):
        raise ValueError(
            f"{label}: stage=cloud, time_index=0, reason=weight_sum_failure"
        )


def raise_panel_sgqf_failure(output, label):
    """Keep the first legacy component failure at the host assertion boundary."""
    status = tf.reshape(output["status_code"], [-1])
    failed = status != 0
    if not bool(tf.reduce_any(failed).numpy()):
        return
    first = tf.argmax(tf.cast(failed, tf.int32), output_type=tf.int32)
    code = int(status[first].numpy())
    stages = (
        "previous_covariance",
        "predictive_covariance",
        "innovation_covariance",
        "carried_covariance",
        "numerical_output",
    )
    reason = "positive_definiteness_veto" if code <= 4 else "nonfinite_output_veto"
    raise ValueError(
        f"{label}: stage={stages[code - 1]}, time_index=0, reason={reason}"
    )
