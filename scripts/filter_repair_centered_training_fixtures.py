"""Complete centered objectives, updates and target-preparation diagnostics."""

from dataclasses import fields

from filter_repair_centered_fixtures import exact_parent

FIXTURES = ("centered_absolute_score", "centered_absolute_step", "centered_prefit_step",
            "centered_total_score_step", "centered_batch_targets", "centered_prefix_schedule")


def fixture(tf, name, size, jit):
    del jit
    from bayesfilter.highdim import (
        zhao_cui_austria_sir_parameter_density_training_tf as training,
    )

    dtype, rows = tf.float64, 4 * size
    dimensions = {"dimension": 36, "rows": rows, "parameter_count": 3,
        "classification": "existing_extension_or_invention", "canonical_admitted": False}
    if name == "centered_prefix_schedule":
        pool, batch = 32 * size, 8 * size

        def evaluate():
            return training.rotating_prefix_minibatch_indices(
                pool_size=pool, batch_size=batch, update=7, seed=8401)

        dimensions.update(pool_size=pool, batch_size=batch, update=7, seed=8401,
            stream="preserved_original_TensorFlow_Philox_Fisher_Yates",
            boundary="complete_rotating_prefix_minibatch")
        return evaluate, (), dimensions

    parent = exact_parent(tf)
    if name == "centered_batch_targets":
        theta = tf.constant([[0., 0., 0.], [.02, -.01, .03]], dtype)
        noise = .05 * tf.reshape(tf.sin(tf.cast(tf.range(rows * 18), dtype)), [rows, 18])

        def evaluate(theta, initial_noise, transition_noise):
            batch = training.build_t1_parameter_density_batch(parent=parent, theta=theta,
                initial_noise=initial_noise, transition_noise=transition_noise, role="frozen_execution_reference")
            ratio = training.estimate_t1_ratio_score(batch, theta_index=1)
            return (tuple(getattr(batch, field.name) for field in fields(batch) if field.name != "role"),
                    tuple(getattr(ratio, field.name) for field in fields(ratio)))

        dimensions.update(theta_rows=2, state_dimension=18,
            boundary="complete_physical_local_reference_batch_and_ratio_targets")
        return evaluate, (theta, noise, .03 * tf.cos(noise)), dimensions

    trainer = training.CenteredResidualTrainer(parent)
    points = tf.reshape(tf.linspace(tf.constant(-.3, dtype), .4, rows * 36), [rows, 36])
    targets = tf.reshape(tf.linspace(tf.constant(-.2, dtype), .3, rows * 3), [rows, 3])
    weights = tf.linspace(tf.constant(-.2, dtype), .3, rows)
    theta = tf.constant([[0., 0., 0.], [.02, -.01, .03]], dtype)
    dimensions.update(parent_rank=1, residual_rank=1, basis_width=3, gradient_clip_norm=10.)

    if name == "centered_absolute_score":
        def evaluate(theta, points, targets, weights):
            with tf.GradientTape() as tape:
                terms = trainer.absolute_density_loss_arrays(theta, tf.stack([points, points + .03]),
                    tf.stack([weights, weights - .01]), l1_weight=1e-5, l2_weight=1e-4,
                    derivative_points=points, derivative_target_score=targets,
                    derivative_importance_log_weight=weights, derivative_weight=.02)
            return (tuple(getattr(terms, field.name) for field in fields(terms)),
                    tape.gradient(terms.total_loss, trainer.trainable_variables))

        dimensions.update(boundary="complete_centered_absolute_objective_and_gradient",
            l1_weight=1e-5, l2_weight=1e-4, derivative_weight=.02)
        return evaluate, (theta, points, targets, weights), dimensions

    optimizer = tf.keras.optimizers.Adam(learning_rate=.001)
    optimizer.build(trainer.trainable_variables)
    if name == "centered_absolute_step":
        step = training.make_compiled_absolute_train_step(trainer, optimizer,
            l1_weight=1e-5, l2_weight=1e-4, derivative_weight=.02, gradient_clip_norm=10.)
        arguments = (theta, tf.stack([points, points + .03]), tf.stack([weights, weights - .01]),
                     points, targets, weights)
        dimensions.update(l1_weight=1e-5, l2_weight=1e-4, derivative_weight=.02)
    elif name == "centered_prefit_step":
        step = training.make_compiled_origin_score_prefit_step(trainer, optimizer, gradient_clip_norm=10.)
        arguments = (points, targets, weights)
    elif name == "centered_total_score_step":
        step = training.make_compiled_origin_total_score_train_step(trainer, optimizer,
            point_weight=.02, global_weight=.03, prefix_weight=.04, l2_weight=1e-4, gradient_clip_norm=10.)
        arguments = (points, targets, weights, targets[0], tf.ones([3], dtype),
                     points[:, :18], targets, tf.ones([rows, 3], dtype))
        dimensions.update(point_weight=.02, global_weight=.03, prefix_weight=.04, l2_weight=1e-4)
    else:
        raise ValueError(f"Unknown centered fixture: {name}")
    numerical = step.python_function
    variables = (*trainer.trainable_variables, *optimizer.variables)
    initial = tuple(tf.identity(value) for value in variables)

    def evaluate(*values):
        # Each warm call measures the same update, including resource restoration.
        for variable, value in zip(variables, initial, strict=True):
            variable.assign(value)
        terms = numerical(*values)
        return terms, tuple(tf.identity(value) for value in variables)

    dimensions.update(boundary=f"complete_{name}", optimizer="Adam", learning_rate=.001,
        identical_initial_parameters_and_slots_each_call=True)
    return evaluate, arguments, dimensions
