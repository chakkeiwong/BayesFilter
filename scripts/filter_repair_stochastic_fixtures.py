"""Matched complete stochastic TT loss and optimizer-step diagnostics."""

from dataclasses import fields
from math import prod
from types import SimpleNamespace

FIXTURES = ("stochastic_tt_score", "stochastic_tt_step", "stochastic_tt_prefit_step")


def fixture(tf, name, size, jit):
    del jit
    from bayesfilter.highdim import stochastic_density_training as training
    from bayesfilter.highdim.bases import BoundedInterval, LegendreBasis1D, ProductBasis
    from bayesfilter.highdim.diagnostics import (
        DensityMeasure,
        MassMeasure,
        MeasureConvention,
    )

    dtype, dimension, rows = tf.float64, 4 * size, 4 * size
    convention = MeasureConvention(density_measure=DensityMeasure.REFERENCE_MEASURE,
        mass_measure=MassMeasure.REFERENCE_MEASURE, reference_weight_name="omega")
    basis = ProductBasis(tuple(LegendreBasis1D(BoundedInterval(-1., 1.), axis % 2 + 1)
                               for axis in range(dimension)), convention)
    ranks = (1, *((2,) * (dimension-1)), 1)
    shapes = tuple((ranks[axis], width, ranks[axis+1]) for axis, width in enumerate(basis.basis_dim_tuple()))
    sizes = tuple(prod(shape) for shape in shapes)
    position = tf.linspace(tf.constant(.15, dtype), .45, sum(sizes))
    cores = tuple(tf.reshape(value, shape) for value, shape in zip(tf.split(position, sizes), shapes, strict=True))
    config = training.P75TrainableTTConfig(product_basis=basis, ranks=ranks,
        tau=tf.constant(1e-6, dtype), l1_weight=tf.constant(.013, dtype),
        l2_weight=tf.constant(.007, dtype), logz_anchor_weight=tf.constant(.03, dtype),
        logz_reference=tf.constant(-1., dtype))
    trainer = training.TrainableFunctionalTT(config, cores)
    points = tf.reshape(tf.linspace(tf.constant(-.7, dtype), .8, rows * dimension), [rows, dimension])
    targets = tf.exp(-.25 * tf.reduce_sum(tf.square(points), axis=1))
    weights = tf.linspace(tf.constant(.5, dtype), 1.5, rows)
    dimensions = {"dimension": dimension, "rows": rows, "ranks": ranks,
        "widths": basis.basis_dim_tuple(), "l1_weight": .013, "l2_weight": .007,
        "logz_anchor_weight": .03, "logz_reference": -1., "tau": 1e-6,
        "classification": "existing_extension_or_invention", "canonical_admitted": False}

    def numerical_fields(record):
        return tuple(getattr(record, field.name) for field in fields(record)
                     if tf.is_tensor(getattr(record, field.name)))

    if name == "stochastic_tt_score":
        def evaluate(position, points, targets, weights):
            view = object.__new__(type(trainer))
            view.__dict__.update(trainer.__dict__)
            with tf.GradientTape() as tape:
                tape.watch(position)
                view.cores = tuple(tf.reshape(value, shape) for value, shape in
                                   zip(tf.split(position, sizes), shapes, strict=True))
                terms = view.objective(SimpleNamespace(points=points, target_values=targets, weights=weights))
            return numerical_fields(terms), tape.gradient(terms.total_loss, position)

        dimensions["boundary"] = "complete_stochastic_tt_density_loss_and_gradient"
        return evaluate, (position, points, targets, weights), dimensions

    if name in ("stochastic_tt_step", "stochastic_tt_prefit_step"):
        optimizer = training.make_adam_optimizer(config)
        optimizer.build(trainer.variables)
        state = tuple(tf.identity(value) for value in optimizer.variables)
        prefit = name == "stochastic_tt_prefit_step"

        def evaluate(position, points, targets, weights):
            # Restore exactly the same variable/slot schema before every call.
            # This measures one complete update, never a drifting training run.
            values = tuple(tf.reshape(value, shape) for value, shape in
                           zip(tf.split(position, sizes), shapes, strict=True))
            for variable, value in zip((*trainer.variables, *optimizer.variables), (*values, *state), strict=True):
                variable.assign(value)
            batch = SimpleNamespace(points=points, target_values=targets, weights=weights)
            if prefit:
                terms = trainer.square_root_prefit_step(batch, optimizer,
                    reference_cores=cores, reference_l2_weight=.025)
            else:
                terms = trainer.train_step(batch, optimizer)
            return numerical_fields(terms), tuple(tf.identity(value) for value in (*trainer.variables, *optimizer.variables))

        dimensions.update(boundary="complete_stochastic_tt_prefit_update" if prefit else "complete_stochastic_tt_density_update",
                          optimizer="Adam", learning_rate=.001, gradient_clip_norm=10.,
                          identical_initial_parameters_and_slots_each_call=True)
        return evaluate, (position, points, targets, weights), dimensions
    raise ValueError(f"Unknown stochastic training fixture: {name}")
