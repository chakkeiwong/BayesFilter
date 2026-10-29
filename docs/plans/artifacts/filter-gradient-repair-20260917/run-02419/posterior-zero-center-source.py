def make_posterior_curvature_controller(callback, eligibility, dimension, config, *, jit_compile=True):
    """Compile the whole fixed-center calculation; graph mode is a reference."""
    cfg = config
    rows = max(32, 4 * dimension) if cfg.rows_per_partition is None else cfg.rows_per_partition
    replicates, batch = cfg.replicate_count, cfg.batch_size
    fit_partitions = 2 * replicates
    partitions = fit_partitions + 3
    planned = batch + (fit_partitions + 2) * ((rows + batch - 1) // batch) * batch
    if dimension < 1 or rows < dimension or planned > cfg.max_physical_rows:
        raise ValueError("invalid dimension/design or whole refinement physical-row budget")
    evaluate = make_partition_evaluator(callback, eligibility, dimension, batch, rows, jit_compile=jit_compile)
    nan = float("nan")

    @tf.function(input_signature=[tf.TensorSpec([dimension], D), tf.TensorSpec([dimension, dimension], D),
                                 tf.TensorSpec([], tf.int32)], autograph=False, jit_compile=jit_compile)
    def refine(center, factor, seed):
        state = {
            "status": tf.constant(0), "role": tf.constant(0), "partition_count": tf.constant(0),
            "counts": tf.zeros([8], tf.int64), "partition_counts": tf.zeros([partitions, 8], tf.int64),
            "partition_seeds": tf.zeros([partitions, 2], tf.int32),
            "center_score": tf.zeros([dimension], D), "center_score_z": tf.zeros([dimension], D),
            "designs": tf.zeros([fit_partitions, rows, dimension], D),
            "scores_z": tf.zeros([fit_partitions, rows, dimension], D),
            "fit_count": tf.constant(0), "fit_precisions": tf.zeros([replicates, dimension, dimension], D),
            "fit_metrics": tf.zeros([replicates, 3], D), "fit_spd": tf.zeros([replicates], tf.bool),
            "fit_rank": tf.zeros([replicates], tf.int32), "fit_accepted": tf.zeros([replicates], tf.bool),
            "spread_done": tf.constant(False), "spread": tf.constant(nan, D),
            "audit_done": tf.constant(False), "audit_rmse": tf.constant(nan, D),
            "reconstruction_done": tf.constant(False), "factor_abs": tf.constant(nan, D), "factor_rel": tf.constant(nan, D),
            "norm_done": tf.constant(False), "center_norm": tf.constant(nan, D),
            "proposal_done": tf.constant(False), "proposal_rmse": tf.constant(nan, D),
            "precision": tf.zeros([dimension, dimension], D), "covariance": tf.zeros([dimension, dimension], D),
            "refined_factor": tf.zeros([dimension, dimension], D),
        }

        def record_partition(current, points, count, role, design_seed):
            result = evaluate(points, count)
            slot = tf.reshape(role, [1, 1])
            return ({**current, "status": result["status"], "role": role,
                "partition_count": current["partition_count"] + 1,
                "counts": current["counts"] + result["counts"],
                "partition_counts": tf.tensor_scatter_nd_update(current["partition_counts"], slot, result["counts"][None]),
                "partition_seeds": tf.tensor_scatter_nd_update(current["partition_seeds"], slot, design_seed[None]),
            }, result["scores"])

        state, center_scores = record_partition(state, tf.broadcast_to(center[None], [rows, dimension]),
            tf.constant(1), tf.constant(0), tf.zeros([2], tf.int32))
        center_score = center_scores[0]
        center_z = tf.linalg.matvec(factor, center_score, transpose_a=True)
        state = {**state, "center_score": center_score, "center_score_z": center_z,
                 "status": tf.where((state["status"] == 0) & ~_finite(center_z), 6, state["status"])}

        def partition(current):
            index = current["partition_count"] - 1
            design_seed = partition_seed(seed, index)
            offsets = draw_offsets(rows, dimension, design_seed, cfg.coordinate_half_width, cfg.fit_design)
            points = center[None] + tf.matmul(offsets, factor, transpose_b=True)
            output, scores = record_partition(current, points, tf.constant(rows), index + 1, design_seed)
            transformed = tf.matmul(scores, factor)
            slot = tf.reshape(index, [1, 1])
            return ({**output,
                "status": tf.where((output["status"] == 0) & ~_finite(transformed), 6, output["status"]),
                "designs": tf.tensor_scatter_nd_update(current["designs"], slot, offsets[None]),
                "scores_z": tf.tensor_scatter_nd_update(current["scores_z"], slot, transformed[None]),
            },)

        state, = tf.while_loop(lambda current: (current["status"] == 0) & (current["partition_count"] <= fit_partitions),
            partition, (state,), maximum_iterations=fit_partitions, parallel_iterations=1)
        state = {**state, "role": tf.where(state["status"] == 0, fit_partitions + 3, state["role"])}

        def fit_replicate(current):
            index = current["fit_count"]
            fit = fit_dense_score_precision_tf(current["center_score_z"],
                tf.gather(current["designs"], index), tf.gather(current["scores_z"], index),
                selection_offsets=tf.gather(current["designs"], index + replicates),
                selection_scores=tf.gather(current["scores_z"], index + replicates))
            valid = (_finite(fit["raw_precision"]) & _finite(fit["raw_eigenvalues"]) & fit["raw_spd"]
                & (fit["design_rank"] == dimension)
                & _within(fit["design_condition"], cfg.max_design_condition_number)
                & _within(fit["precision_condition"], cfg.max_precision_condition_number)
                & _within(fit["selection_relative_rmse"], cfg.selection_relative_rmse_cap))
            slot = tf.reshape(index, [1, 1])
            metrics = tf.stack((fit["design_condition"], fit["precision_condition"], fit["selection_relative_rmse"]))
            return ({**current, "status": tf.where(valid, 0, 8), "fit_count": index + 1,
                "fit_precisions": tf.tensor_scatter_nd_update(current["fit_precisions"], slot, fit["raw_precision"][None]),
                "fit_metrics": tf.tensor_scatter_nd_update(current["fit_metrics"], slot, metrics[None]),
                "fit_spd": tf.tensor_scatter_nd_update(current["fit_spd"], slot, fit["raw_spd"][None]),
                "fit_rank": tf.tensor_scatter_nd_update(current["fit_rank"], slot, fit["design_rank"][None]),
                "fit_accepted": tf.tensor_scatter_nd_update(current["fit_accepted"], slot, valid[None]),
            },)

        state, = tf.while_loop(lambda current: (current["status"] == 0) & (current["fit_count"] < replicates),
            fit_replicate, (state,), maximum_iterations=replicates, parallel_iterations=1)

        def consensus(current):
            spread = precision_spread(current["fit_precisions"], jit_compile=jit_compile)
            precision = _symmetric(tf.reduce_sum(current["fit_precisions"] / replicates, axis=0))
            status = tf.where(~_within(spread, cfg.replicate_generalized_eigenvalue_spread_cap), 10,
                              tf.where(~_finite(precision), 11, 0))
            return {**current, "role": tf.constant(fit_partitions + 4), "status": status,
                    "spread_done": tf.constant(True), "spread": spread, "precision": precision}

        state = tf.cond(state["status"] == 0, lambda: consensus(state), lambda: state)

        def audit(current):
            design_seed = partition_seed(seed, tf.constant(fit_partitions))
            offsets = draw_offsets(rows, dimension, design_seed, cfg.coordinate_half_width, cfg.fit_design)
            points = center[None] + tf.matmul(offsets, factor, transpose_b=True)
            output, scores = record_partition(current, points, tf.constant(rows), tf.constant(fit_partitions + 1), design_seed)

            def compare():
                rmse = _relative_response_rmse(current["precision"], current["center_score_z"], offsets, scores @ factor)
                return {**output, "audit_done": tf.constant(True), "audit_rmse": rmse,
                        "status": tf.where(_within(rmse, cfg.audit_relative_rmse_cap), 0, 12)}

            return tf.cond(output["status"] == 0, compare, lambda: output)

        state = tf.cond(state["status"] == 0, lambda: audit(state), lambda: state)

        def factorize(current):
            current = {**current, "role": tf.constant(fit_partitions + 5)}
            cholesky = tf.linalg.cholesky(current["precision"])
            inverse_action = tf.linalg.triangular_solve(cholesky, tf.transpose(factor))
            covariance = _symmetric(tf.matmul(inverse_action, inverse_action, transpose_a=True))
            refined = tf.linalg.cholesky(covariance)
            valid = _finite(cholesky) & _finite(covariance) & _finite(refined) & tf.reduce_all(tf.linalg.diag_part(refined) > 0.0)

            def reconstruction():
                residual = refined @ tf.transpose(refined) - covariance
                absolute = tf.reduce_max(tf.abs(residual))
                scale = tf.reduce_max(tf.abs(covariance))
                relative = tf.linalg.norm(residual / scale) / tf.linalg.norm(covariance / scale)
                passed = (tf.math.is_finite(absolute) & tf.math.is_finite(relative)
                    & ((absolute <= tf.constant(cfg.factor_absolute_tolerance, D)) | (relative <= tf.constant(cfg.factor_relative_tolerance, D))))
                output = {**current, "covariance": covariance, "refined_factor": refined,
                    "reconstruction_done": tf.constant(True), "factor_abs": absolute, "factor_rel": relative,
                    "status": tf.where(passed, 0, 14)}

                def center_norm():
                    normalized = tf.linalg.matvec(refined, current["center_score"], transpose_a=True)
                    scale = tf.reduce_max(tf.abs(normalized))
                    norm = scale * tf.linalg.norm(tf.math.divide_no_nan(normalized, scale))
                    return {**output, "norm_done": tf.constant(True), "center_norm": norm,
                            "status": tf.where(tf.math.is_finite(norm), 0, 15)}

                return tf.cond(passed, center_norm, lambda: output)

            return tf.cond(valid, reconstruction, lambda: {**current, "status": tf.constant(13)})

        state = tf.cond(state["status"] == 0, lambda: factorize(state), lambda: state)

        def proposal(current):
            design_seed = partition_seed(seed, tf.constant(fit_partitions + 1))
            latent = philox_normal_float64([rows, dimension], design_seed)
            delta = tf.matmul(latent, current["refined_factor"], transpose_b=True)
            output, scores = record_partition(current, center[None] + delta, tf.constant(rows),
                tf.constant(fit_partitions + 2), design_seed)

            def compare():
                offsets = tf.transpose(tf.linalg.triangular_solve(factor, tf.transpose(delta)))
                rmse = _relative_response_rmse(current["precision"], current["center_score_z"], offsets, scores @ factor)
                return {**output, "proposal_done": tf.constant(True), "proposal_rmse": rmse,
                        "status": tf.where(_within(rmse, cfg.proposal_relative_rmse_cap), 1, 16)}

            return tf.cond(output["status"] == 0, compare, lambda: output)

        state = tf.cond(state["status"] == 0, lambda: proposal(state), lambda: state)
        del state["designs"], state["scores_z"], state["center_score"], state["center_score_z"]
        return state

    return refine
