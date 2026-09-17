"""Matched diagnostic fixtures for complete repaired numerical calculations.

Frozen clouds are constructed independently of either source arm. Host result
assembly is omitted from candidate tensor programs and included in baseline
eager timings when the original endpoint cannot trace; the report names that
execution difference. These fixtures confer no scientific admission.
"""

import math
from dataclasses import replace


FIXTURES = (
    "sv_direct_score", "sv_augmented", "sv_mixture_sgqf", "sv_mixture_score",
    "sv_mixture_kalman", "sv_mixture_cut4", "sv_mixture_ukf", "sv_mixture_ukf_score",
    "filter_dense", "filter_gaussian", "filter_gaussian_tt", "geometry_fit",
)


def frozen_cloud(tf, width):
    from bayesfilter.nonlinear.fixed_sgqf_tf import TFFixedSGQFCloud

    root = math.sqrt(3.0)
    if width == 1:
        points, weights = [[-root], [0.0], [root]], [1 / 6, 2 / 3, 1 / 6]
        indices, coefficients = ((2,),), (1,)
    else:
        points = [[-root, 0.0], [0.0, -root], [0.0, 0.0], [0.0, root], [root, 0.0]]
        weights = [1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6]
        indices, coefficients = ((1, 1), (1, 2), (2, 1)), (-1, 1, 1)
    return TFFixedSGQFCloud(width, 2, tf.constant(points, tf.float64),
        tf.constant(weights, tf.float64), indices, coefficients, 1e-12, 1e-15)


def sv_fixture(tf, name, size, jit):
    from bayesfilter.highdim import sv_mixture_cut4 as panel

    d, dates, width = tf.float64, 2 * size, size
    if name.startswith("sv_mixture_"):
        # The legacy endpoint recompiles each component on every invocation.
        # Keep two horizons within the unchanged 300-second/20-call budget.
        dates, width = size, 1
    inputs = (
        tf.tile(tf.constant([[.6, -1.1], [-.4, .3]], d)[:, :width], [size, 1])[:dates],
        tf.constant([.45, .72], d)[:width], tf.constant([.8, 1.2], d)[:width],
        tf.constant([.65, .9], d)[:width],
    )
    branch = panel.TFFixedSGQFBranchConfig(predictive_epsilon=1e-10, innovation_epsilon=1e-10)
    dimensions = dict(horizon=dates, state=width, parameters=2 * width,
                      cloud="frozen_standard_normal_smolyak_level2", endpoint=name)
    try:
        from bayesfilter.highdim.sv_direct_sgqf_native_tf import direct_sgqf_program
        from bayesfilter.highdim.sv_panel_quadrature_native_tf import augmented_sgqf_program, mixture_program
    except ImportError:
        native = False
    else:
        native = True
    if name in ("sv_direct_score", "sv_augmented"):
        direct = name == "sv_direct_score"
        cloud = frozen_cloud(tf, 1 if direct else 2)
        inputs += (cloud.points, cloud.weights)
        if native:
            call = (direct_sgqf_program(dates, width, cloud.point_count, with_score=True, jit_compile=jit)
                    if direct else augmented_sgqf_program(dates, width, cloud.point_count, branch, jit_compile=jit))

            def evaluate(y, gamma, beta, sigma, points, weights):
                if direct:
                    terms, score, means, covariances = call(tf.math.log(y**2), gamma, beta, sigma,
                                                          tf.reshape(points, [-1]), weights)
                    return tf.reduce_sum(terms), score, means, covariances
                row = call(y, gamma, beta, sigma, points, weights, tf.constant(1e-10, d))
                return (tf.reduce_sum(row["log_normalizers"]), row["log_normalizers"],
                        row["mean_path"], row["covariance_path"])
        else:
            route = (panel.exact_transformed_sv_independent_panel_fixed_sgqf_score if direct
                     else panel.actual_transformed_sv_independent_panel_augmented_noise_fixed_sgqf_filter)

            def evaluate(y, gamma, beta, sigma, points, weights):
                options = {} if direct else dict(jit_compile=jit)
                row = route(y, gamma=gamma, beta=beta, sigma=sigma,
                            cloud=replace(cloud, points=points, weights=weights), **options)
                return row.log_likelihood, row.score if direct else row.log_normalizers, row.mean_path, row.covariance_path
        return evaluate, inputs, dimensions

    cloud = frozen_cloud(tf, width)
    mixture = panel.SVLogChiSquareGaussianMixture(tf.constant([.4, .6], d),
        tf.constant([-1.1, .4], d), tf.constant([.7, 1.4], d), source="repair_frozen_fixture")
    indices = tf.constant(panel._component_tuples(2, width), tf.int32)
    method = name.removeprefix("sv_mixture_")
    score = method in ("score", "ukf_score")
    fields = ("log_likelihood", "score" if score else "log_normalizers", "mean_path", "covariance_path")
    fields += ("d_mean_path", "d_covariance_path") if score else ("component_weights",)
    inputs += (mixture.weights, mixture.means, mixture.variances, indices, cloud.points, cloud.weights)
    dimensions.update(components=2, component_tuples=2**width, score="analytical_recursive" if score else "none")
    if native and method in ("ukf", "ukf_score"):
        def evaluate(y, gamma, beta, sigma, weights, means, variances, indices, points, quadrature_weights):
            from types import SimpleNamespace
            row = panel._compiled_panel_mixture_ukf(tf.math.log(y**2 + 1e-8), gamma, beta, sigma,
                SimpleNamespace(weights=weights, means=means, variances=variances), indices,
                with_score=score, jit_compile=jit)
            if score:
                return tf.reduce_sum(row[0]), tf.reduce_sum(row[1], axis=0), *row[2:6]
            return tf.reduce_sum(row[0]), row[0], row[2], row[3], row[6]
    elif native:
        algorithm = "sgqf" if score else method
        quadrature_count = cloud.point_count if algorithm == "sgqf" else 0
        call = mixture_program(dates, width, 2, quadrature_count, branch,
                               method=algorithm, with_score=score, jit_compile=jit)

        def evaluate(y, gamma, beta, sigma, weights, means, variances, indices, points, quadrature_weights):
            if algorithm != "sgqf":
                points, quadrature_weights = tf.zeros([0, width], d), tf.zeros([0], d)
            row = call(tf.math.log(y**2 + 1e-8), gamma, beta, sigma, weights, means,
                       variances, indices, points, quadrature_weights, tf.constant(1e-12, d))
            return tuple(tf.reduce_sum(row["log_normalizers"]) if field == "log_likelihood" else row[field]
                         for field in fields)
    else:
        suffix = dict(sgqf="fixed_sgqf_filter", score="fixed_sgqf_score",
                      kalman="kalman_filter", cut4="cut4_filter", ukf="ukf_filter", ukf_score="ukf_score")[method]
        route = getattr(panel, "independent_panel_sv_mixture_" + suffix)

        def evaluate(y, gamma, beta, sigma, weights, means, variances, indices, points, quadrature_weights):
            options = (dict(cloud=replace(cloud, points=points, weights=quadrature_weights), branch_config=branch)
                       if method in ("sgqf", "score") else {})
            row = route(y, gamma=gamma, beta=beta, sigma=sigma,
                        mixture=replace(mixture, weights=weights, means=means, variances=variances), **options)
            return tuple(getattr(row, field) for field in fields)
    return evaluate, inputs, dimensions


def filter_fixture(tf, name, size, jit):
    import bayesfilter.highdim as h
    from bayesfilter.highdim import filtering

    d, dates = tf.float64, 2 * size
    convention = h.MeasureConvention(density_measure=h.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=h.MassMeasure.REFERENCE_MEASURE, reference_weight_name="omega")
    dense, fit = name == "filter_dense", name == "filter_gaussian_tt"
    order = 31 if dense else 24
    k = tf.cast(tf.range(1, order), d)
    off = k / tf.sqrt(4 * k * k - 1)
    nodes, vectors = tf.linalg.eigh(tf.linalg.diag(off, k=1) + tf.linalg.diag(off, k=-1))
    weights = 2 * tf.square(vectors[0])
    # Freeze both arms' quadrature preparation, including the legacy wrapper.
    filtering.legendre_gauss_nodes_weights = lambda requested: (nodes, weights) if requested == order else (_ for _ in ()).throw(ValueError(requested))
    basis = h.ProductBasis([h.LegendreBasis1D(h.BoundedInterval(-1., 1.), 6)], convention)
    fit_config = h.FixedTTFitConfig(ranks=(1, 1), ridge=1e-10, max_sweeps=1,
        sweep_order=(0,), row_budget=256, column_budget=32, dense_matrix_byte_budget=100_000,
        normal_matrix_byte_budget=10_000, condition_number_warning=1e10,
        condition_number_veto=1e14, holdout_tolerance=1e6) if fit else None
    coordinate = (h.AffineCoordinateMap(tf.zeros([1], d), tf.constant([[3.]], d))
                  if dense else h.IdentityCoordinateMap(1))
    config = h.FixedBranchFilterConfig(fit_config=fit_config, density_tau=1e-12 if fit else 0.,
        normalizer_floor=1e-14 if fit else 1e-12, denominator_floor=1e-14 if fit else 1e-12,
        retained_storage_byte_budget=10_000_000, coordinate_maps=(coordinate,),
        measure_convention=convention, deterministic_seed="repair-filter-wrapper-v1",
        product_basis=basis if fit else None, fit_quadrature_order=order,
        initial_cores=(h.TTCore(tf.reshape(tf.one_hot(0, 7, dtype=d), [1, 7, 1])),) if fit else None)
    if dense:
        model = h.StochasticVolatilitySSM()
        theta = model.unconstrained_from_physical(.6, .4)
    else:
        model = h.LinearGaussianSSM(initial_mean=tf.zeros([1], d), initial_covariance=tf.ones([1, 1], d),
            transition_matrix=tf.constant([[.7]], d), transition_covariance=tf.constant([[.25]], d),
            observation_matrix=tf.ones([1, 1], d), observation_covariance=tf.constant([[.09]], d))
        theta = tf.zeros([0], d)
    y = tf.tile(tf.constant([[.2], [-.1]], d), [size, 1])
    inputs = (theta, y)
    try:
        from bayesfilter.highdim.filtering_native_tf import scalar_dense_program, gaussian_history_program, gaussian_fit_program
    except ImportError:
        def evaluate(theta, y):
            row = h.FixedBranchSquaredTTFilter(config).log_likelihood(model, theta, y)
            output = (row.log_likelihood, tf.stack([step.log_normalizer for step in row.steps]),
                tf.stack([step.diagnostics["retained_mean"] for step in row.steps]),
                tf.stack([step.diagnostics["retained_covariance"] for step in row.steps]))
            if fit:
                output += (tf.stack([step.fit_result.fitted_tt.cores[0].values for step in row.steps]),
                    tf.stack([step.diagnostics["tt_density_normalizer"] for step in row.steps]))
            return output
    else:
        if dense:
            call = scalar_dense_program(model, y.shape, order, jit_compile=jit)
            points, logdet = coordinate.forward(nodes[:, None])

            def evaluate(theta, y):
                row = call(theta, y, points, weights, logdet)
                moments = row["moments"]
                return row["log_likelihood"], moments[:, 0], moments[:, 1:2], moments[:, 2:3, None]
        else:
            call = gaussian_history_program(dates, 1, 1, jit_compile=jit)
            prepared = gaussian_fit_program(config, dates, 1, jit_compile=jit) if fit else None

            def evaluate(theta, y):
                means, covariances, terms = call(y, model.initial_mean, model.initial_covariance,
                    model.transition_matrix, model.transition_covariance, model.observation_matrix,
                    model.observation_covariance, model.transition_offset, model.observation_offset)
                output = (tf.reduce_sum(terms), terms, means, covariances)
                if fit:
                    row = prepared(means, covariances)
                    cores = tf.reshape(row["fit"]["cores"][:, 0, :7], [dates, 1, 7, 1])
                    output += (cores, row["normalizer"])
                return output
    return evaluate, inputs, dict(horizon=dates, state=1, parameters=2 if dense else 0,
        quadrature_order=order, tt_degree=6 if fit else None, endpoint=name,
        algorithm="existing_dense_quadrature_or_Gaussian_ALS_extension")


def geometry_fixture(tf, size, jit):
    from bayesfilter.inference import quadratic_geometry as geometry

    d, count, width, rank = tf.float64, 50 * size, 3 * size, 2 * size
    # Frozen deterministic, full-rank cloud: no source-arm RNG enters the fit.
    offsets = tf.reshape(tf.cast(tf.range(count * width), d), [count, width])
    z = tf.sin(offsets * .731 + .19) + tf.cos(offsets * offsets * .017)
    q = tf.eye(width, dtype=d)[:, -rank:]
    precision = tf.linalg.diag(tf.linspace(tf.constant(1.2, d), tf.constant(3., d), width))
    center_score = tf.linspace(tf.constant(-.2, d), tf.constant(.3, d), width)
    score = center_score[None] - tf.matmul(z, precision)
    y = .7 + tf.linalg.matvec(z, center_score) - .5 * tf.reduce_sum(z * tf.matmul(z, precision), axis=1)
    fields = ("precision", "linear_term", "mu", "raw_mu", "intercept", "lambda0", "raw_lambda0",
              "loss", "score_rmse", "condition_bound", "score_design_rank", "mu_clipped_count")
    config = geometry.LowRankSPDQuadraticGeometryConfig(rank=rank, eigenvalue_floor=.1,
                                                       max_condition_number=100.)

    def evaluate(z, y, score, q, center_score):
        if hasattr(geometry, "_quadratic_fit_kernel"):
            row = geometry._quadratic_fit_kernel(z, y, score, q, center_score,
                tf.constant(.1, d), tf.constant(100., d), use_xla_svd=jit)
        else:
            row = geometry._fit_constrained_quadratic(z, y, score, q_basis=q, cfg=config,
                                                     dim=width, rank=rank, center_score_z=center_score)
            if row["status"] != "usable":
                raise ValueError(row["status"])
        return tuple(tf.convert_to_tensor(row[field], tf.int64 if field.endswith("_rank") or field == "mu_clipped_count" else d)
                     for field in fields)
    evaluate.execution_backend = ("tensorflow_numerical_fit" if hasattr(geometry, "_quadratic_fit_kernel")
                                  else "legacy_numpy_host_fit_diagnostic_only")
    return evaluate, (z, y, score, q, center_score), dict(samples=count, parameters=width, rank=rank,
        input_design="frozen_trigonometric_cloud_v1", scope="complete_quadratic_fit_not_complete_initializer")


def fixture(tf, name, size, jit):
    if name.startswith("sv_"):
        return sv_fixture(tf, name, size, jit)
    if name.startswith("filter_"):
        return filter_fixture(tf, name, size, jit)
    if name == "geometry_fit":
        return geometry_fixture(tf, size, jit)
    raise ValueError(name)
