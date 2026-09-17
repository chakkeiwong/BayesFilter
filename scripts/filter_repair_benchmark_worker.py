"""Fresh-process engineering diagnostic for the filter repair campaign."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

from measure_filter_xla_memory import build_fixture as audit_fixture
from filter_repair_endpoint_fixtures import FIXTURES as ENDPOINT_FIXTURES, fixture as endpoint_fixture


def host_memory():
    return {row.split()[0].rstrip(":"): int(row.split()[1]) * 1024 for row in Path("/proc/self/status").read_text().splitlines() if row.startswith(("VmRSS:", "VmHWM:"))}


def retained_fixture_config(tf):
    import bayesfilter.highdim as h

    convention = h.MeasureConvention(density_measure=h.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=h.MassMeasure.REFERENCE_MEASURE, reference_weight_name="omega")
    basis = h.ProductBasis([h.LegendreBasis1D(h.BoundedInterval(-1., 1.), 12)], convention)
    return h.FixedBranchFilterConfig(
        fit_config=h.FixedTTFitConfig(ranks=(1, 1), ridge=1e-12, max_sweeps=2,
            sweep_order=(0,), row_budget=256, column_budget=80,
            dense_matrix_byte_budget=100_000, normal_matrix_byte_budget=50_000,
            condition_number_warning=1e10, condition_number_veto=1e14, holdout_tolerance=2e-5),
        density_tau=0., normalizer_floor=1e-12, denominator_floor=1e-12,
        retained_storage_byte_budget=10_000_000,
        coordinate_maps=(h.AffineCoordinateMap(tf.zeros([1], tf.float64), tf.constant([[8.]], tf.float64)),),
        measure_convention=convention, deterministic_seed="repair-retained-fixed-v1",
        product_basis=basis, initial_cores=(h.TTCore(tf.ones([1, 13, 1], tf.float64)),),
        fit_quadrature_order=31)


def squared_fixture_density(tf, correlated=False):
    import bayesfilter.highdim as h

    convention = h.MeasureConvention(density_measure=h.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=h.MassMeasure.REFERENCE_MEASURE, reference_weight_name="omega")
    dimension, degree = (2, 1) if correlated else (1, 0)
    basis = h.ProductBasis([h.LegendreBasis1D(h.BoundedInterval(-1., 1.), degree)
                           for _ in range(dimension)], convention)
    values = (tf.constant([[[1., 0.], [0., 1.]]], tf.float64),
              tf.constant([[[1.], [0.]], [[0.], [.1]]], tf.float64)) if correlated else (tf.ones([1, 1, 1], tf.float64),)
    arguments = dict(sqrt_tt=h.FunctionalTT(tuple(h.TTCore(value) for value in values), basis, convention),
        defensive_density=h.TensorProductReferenceDensity(basis, convention),
        tau=tf.constant(.05 if correlated else .25, tf.float64),
        normalizer_floor=tf.constant(1e-12, tf.float64),
        denominator_floor=tf.constant(1e-12, tf.float64), measure_convention=convention)
    return h.SquaredTTDensity(**arguments, branch_identity=h.SquaredTTDensity.expected_branch_identity(**arguments))


def fixture(tf, name, size, jit):
    if name in ENDPOINT_FIXTURES:
        return endpoint_fixture(tf, name, size, jit)
    if name == "squared_density":
        density = squared_fixture_density(tf, correlated=True)
        prefix = tf.zeros([1, 0], tf.float64)
        grid = tf.linspace(tf.constant(-1., tf.float64), tf.constant(1., tf.float64), 7*size)
        try:
            from bayesfilter.highdim.squared_tt_density_native_tf import density_program
        except ImportError:
            def evaluate(prefix, grid):
                return density.conditional_density(0, prefix, grid), density.normalizer()
        else:
            call, arguments = density_program(density, "conditional", axis=0, prefix=prefix, grid=grid,
                                              jit_compile=jit)
            def evaluate(prefix, grid):
                rows, _, normalizer = call(*arguments[:2], prefix, grid)
                return rows, normalizer
        return evaluate, (prefix, grid), dict(dimension=2, degree=1, rank=2, grid=7*size,
            suffix_grid=33, method="existing_grid_conditional_extension")
    if name == "ttsirt_preparation":
        from bayesfilter.highdim.filtering import IdentityCoordinateMap
        from bayesfilter.highdim.transport import FixedTTSIRTTransport, KRCDFConfig
        from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import compile_fixed_ttsirt_proposal_branch

        steps, count, d = size, 3, tf.float64
        cdf = KRCDFConfig(grid_size=65, bisection_steps=24, monotonicity_tolerance=1e-12,
                         bracket_tolerance=1e-12, denominator_floor=1e-12, max_floor_count=0)
        initial = FixedTTSIRTTransport(squared_fixture_density(tf), cdf)
        transition = FixedTTSIRTTransport(squared_fixture_density(tf, correlated=True), cdf)
        transitions = (transition,) * steps
        coordinate = IdentityCoordinateMap(1)
        inputs = (tf.constant([[.13, .46, .79]], d),
            tf.tile(tf.constant([[.02, .41, .99]], d), [steps, 1]),
            tf.tile(tf.math.log(tf.constant([[.2, .3, .5]], d)), [steps, 1]),
            tf.tile(tf.constant([[[.24, .39, .72]]], d), [steps, 1, 1]))
        try:
            from bayesfilter.highdim.ttsirt_proposal_native_tf import proposal_program
        except ImportError:
            def evaluate(initial_reference, uniforms, auxiliary, reference):
                row = compile_fixed_ttsirt_proposal_branch(observations=tf.zeros([steps+1, 1], d),
                    initial_transport=initial, transition_transports=transitions, coordinate_map=coordinate,
                    initial_reference_points=initial_reference, ancestor_uniforms=uniforms,
                    auxiliary_log_probabilities=auxiliary, transition_reference_points=reference).branch
                return row.states, row.initial_log_proposal_density, row.ancestors, row.transition_log_proposal_density, tf.constant(0)
        else:
            call, packed = proposal_program(initial, transitions, coordinate, count, jit_compile=jit)
            def evaluate(*inputs):
                return call(*packed, *inputs)
        return evaluate, inputs, dict(horizon=steps+1, particles=count, state=1, grid=65,
                                      bisection_steps=24, method="existing_grid_cdf_extension")
    if name.startswith("simulation_"):
        from bayesfilter.highdim.models import StochasticVolatilitySSM, p30_spatial_sir_fixture_model, p30_predator_prey_fixture_model

        family, horizon, seed = name.removeprefix("simulation_"), 3*size, 421
        if family == "sv":
            model = StochasticVolatilitySSM()
            theta = model.unconstrained_from_physical(.8, .6)
        elif family == "sir":
            model = p30_spatial_sir_fixture_model(3)
            theta = tf.zeros([0], tf.float64)
        else:
            model = p30_predator_prey_fixture_model()
            theta = model.true_parameters()
        try:
            from bayesfilter.highdim.model_simulation_tf import model_simulation_program
            from bayesfilter.ops.generator_stream_tf import generator_seed_state
        except ImportError:
            def evaluate(theta):
                return model.simulate(horizon, seed) if family == "sir" else model.simulate(theta, horizon, seed)
        else:
            call = model_simulation_program(model, horizon+1, family, jit_compile=jit)
            seed_state = generator_seed_state(seed)
            def evaluate(theta):
                return call(theta, seed_state)
        return evaluate, (theta,), dict(horizon=horizon, state=model.state_dim(),
            observations=model.observation_dim(), seed=seed, family=family, stream="existing_generator_philox_call_order")
    if name in ("tt_scalar_retained", "tt_panel_retained", "tt_panel_ksc"):
        from bayesfilter.highdim.derivatives import FixedBranchDerivativeConfig
        from bayesfilter.highdim import filtering, sv_mixture_cut4 as panel
        from bayesfilter.highdim.models import StochasticVolatilitySSM

        config = retained_fixture_config(tf)
        derivative = FixedBranchDerivativeConfig(parameter_indices=(0, 1), finite_difference_h=())
        if name == "tt_scalar_retained":
            model = StochasticVolatilitySSM()
            theta = model.unconstrained_from_physical(.6, .4)
            data = tf.tile(tf.constant([[.12], [-.08]], tf.float64), [size, 1])
            try:
                from bayesfilter.highdim.scalar_retained_native_tf import make_scalar_retained_program
            except ImportError:
                def evaluate(theta, data):
                    row = filtering.scalar_nonlinear_fixed_design_tt_score_path(model, theta, data, config,
                        derivative, retained_moment_order=33, retained_propagation_order=41)
                    return row.log_likelihood, row.score
            else:
                call = make_scalar_retained_program(model, config, data.shape,
                    derivative_config=derivative, moment_order=33, propagation_order=41, jit_compile=jit).call
                def evaluate(theta, data):
                    row = call(theta, data)
                    return row["log_likelihood"], row["score"]
            return evaluate, (theta, data), dict(horizon=2*size, state=1, parameters=2, degree=12,
                fit_order=31, moment_order=33, propagation_order=41, score="analytical_recursive")
        width = size
        ksc = name == "tt_panel_ksc"
        data = tf.constant([[.6, -1.1], [-.4, .3]], tf.float64)[:, :width]
        gamma = tf.constant([.45, .72], tf.float64)[:width]
        beta = tf.constant([.8, 1.2], tf.float64)[:width]
        sigma = tf.constant([.65, .9], tf.float64)[:width]
        mixture = panel.ksc_1998_log_chi_square_mixture() if ksc else None
        try:
            from bayesfilter.highdim.sv_panel_retained_native_tf import make_sv_panel_retained_program
        except ImportError:
            route = (panel.independent_panel_sv_mixture_zhaocui_tt_score if ksc
                     else panel.exact_transformed_sv_independent_panel_zhaocui_tt_score)
            extra = dict(mixture=mixture) if ksc else {}
            def evaluate(data, gamma, beta, sigma):
                row = route(data, gamma=gamma, beta=beta, sigma=sigma, config=config,
                            derivative_config=derivative, **extra)
                return row.log_likelihood, row.score
        else:
            call, _ = make_sv_panel_retained_program(config, data.shape, mixture=mixture,
                                                     derivative_config=derivative, jit_compile=jit)
            def evaluate(data, gamma, beta, sigma):
                row = call(tf.math.log(data**2 + (1e-8 if ksc else 0.)), gamma, beta, sigma)
                return row["log_likelihood"], row["score"]
        return evaluate, (data, gamma, beta, sigma), dict(horizon=2, state=width,
            parameters=2*width, degree=12, fit_order=31, moment_order=257, propagation_order=321,
            score="independent_coordinate_analytical_recursive", mixture="ksc" if ksc else "exact")
    if name == "particle_alg1":
        from experiments.dpf_implementation.tf_tfp.filters import ledh_pfpf_alg1_ukf_tf as algorithm
        from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import gaussian_logpdf_tf
        d,horizon,count = tf.float64,2*size,4*size
        y = tf.reshape(tf.linspace(tf.constant(-.1,d),.2,horizon),[horizon,1])
        initial = tf.reshape(tf.linspace(tf.constant(-.2,d),.3,count),[count,1])
        covariance = tf.constant([[.3]],d)
        pseudo_time = tf.constant([.5,.5],d)
        transition = lambda x,seed,date:.72*x+.02*(tf.cast(tf.range(count)[:,None],d)-tf.cast(count-1,d)/2.)
        settings = dict(transition_sample=transition,transition_mean_fn=lambda x,date:.72*x,
            transition_log_density_fn=lambda x,old,date:gaussian_logpdf_tf(x-.72*old,tf.constant([[.04]],d)),
            observation_mean_fn=lambda x,date:x+x**2,
            observation_jacobian_fn=lambda x,date:tf.reshape(1.+2.*x[0],[1,1]),
            observation_log_density_fn=lambda x,y,date:gaussian_logpdf_tf(y-x-x**2,tf.constant([[.05]],d)),
            process_noise_covariance_fn=lambda old,date:tf.constant([[.04]],d),
            observation_covariance_fn=lambda date:tf.constant([[.05]],d),seed=13,
            resampling_route=algorithm.OT_SINKHORN_COVARIANCE_CARRY_ROUTE,ess_threshold_ratio=1.01,
            alpha=1.,beta=2.,kappa=0.,covariance_floor=1e-12,rank_tolerance=1e-12,
            sinkhorn_epsilon=1.,sinkhorn_iterations=80,sinkhorn_tolerance=1e-7,sinkhorn_epsilon_policy="fixed",
            annealed_scaling=.9,annealed_convergence_threshold=1e-3,transport_gradient_mode="filterflow_clipped")
        try:
            from experiments.dpf_implementation.tf_tfp.filters.alg1_execution_tf import make_alg1_filter
        except ImportError:
            def primal(y,initial,covariance,pseudo_time):
                row = algorithm.run_ledh_pfpf_alg1_ukf_tf(observations=y,initial_sample=lambda n,seed:initial,
                    initial_covariance=covariance,pseudo_time_steps=pseudo_time,num_particles=count,**settings)
                return (row.log_likelihood_estimate,row.filtered_means,row.filtered_variances,
                        row.particle_covariances_by_time,row.predicted_covariances_by_time,row.corrected_log_weights_by_time,row.ess_by_time)
        else:
            call = make_alg1_filter(tf.TensorSpec(y.shape,d),tf.TensorSpec(initial.shape,d),
                tf.TensorSpec(pseudo_time.shape,d),jit_compile=jit,**settings)
            def primal(y,initial,covariance,pseudo_time):
                total,history=call(y,initial,covariance,pseudo_time)
                return total,*history[:6]
        def evaluate(y,initial,covariance,pseudo_time):
            with tf.GradientTape() as tape:
                tape.watch(y)
                result=primal(y,initial,covariance,pseudo_time)
            return *result,tape.gradient(result[0],y)
        return evaluate,(y,initial,covariance,pseudo_time),dict(horizon=horizon,particles=count,state=1,
            pseudo_time_steps=2,resampling=settings["resampling_route"],canonical_admitted=False,
            score="observation_autodiff_diagnostic")
    if name == "apf":
        from types import SimpleNamespace
        from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import _evaluate_core
        d, horizon, count, dimension = tf.float64, 3*size, 8*size, 2
        def normal(residual, variance):
            return -.5*(dimension*tf.math.log(tf.constant(2*3.141592653589793*variance,d))
                        +tf.reduce_sum(residual**2,axis=-1)/variance)
        def mark(residual, variance, axis):
            value = tf.reduce_sum(residual,axis=-1)/variance
            return tf.stack((value,tf.zeros_like(value)) if axis == 0 else (tf.zeros_like(value),value),axis=-1)
        model = SimpleNamespace(parameter_dim=lambda:2,
            initial_log_density=lambda theta,x:normal(x-theta[0],1.25),
            observation_log_density=lambda theta,x,y,t:normal(y-x-theta[1],.8),
            transition_log_density=lambda theta,old,x,t:normal(x-.65*old-theta[0],.7),
            initial_log_density_parameter_score=lambda theta,x:mark(x-theta[0],1.25,0),
            observation_log_density_parameter_score=lambda theta,x,y,t:mark(y-x-theta[1],.8,1),
            transition_log_density_parameter_score=lambda theta,old,x,t:mark(x-.65*old-theta[0],.7,0))
        theta = tf.constant([.14,-.08],d)
        observations = tf.reshape(tf.linspace(tf.constant(-.2,d),.3,horizon*dimension),[horizon,dimension])
        states = tf.reshape(tf.sin(tf.cast(tf.range(horizon*count*dimension),d)*.31),[horizon,count,dimension])
        ancestors = tf.math.floormod(tf.reshape(tf.range((horizon-1)*count),[horizon-1,count])*3+1,count)
        log_q = normal(states,1.4)
        auxiliary = tf.nn.log_softmax(tf.cos(tf.cast(ancestors,d)),axis=-1)
        initial_mass = tf.fill([count],-tf.math.log(tf.cast(count,d)))
        transition_mass = tf.broadcast_to(initial_mass,[horizon-1,count])
        def evaluate(theta,observations,states,ancestors,log_q,auxiliary,initial_mass,transition_mass):
            branch = SimpleNamespace(dtype=d,particle_count=count,time_steps=horizon,observations=observations,
                states=states,ancestors=ancestors,initial_log_proposal_density=log_q[0],
                transition_log_proposal_density=log_q[1:],auxiliary_log_probabilities=auxiliary,
                initial_log_base_mass=initial_mass,transition_log_base_mass=transition_mass)
            return _evaluate_core(model,branch,theta)
        return evaluate,(theta,observations,states,ancestors,log_q,auxiliary,initial_mass,transition_mass),dict(
            horizon=horizon,particles=count,state=dimension,parameters=2,score="analytical_recursive_fixed_branch")
    if name == "tt_scalar":
        import bayesfilter.highdim as highdim
        from bayesfilter.highdim.zhao_cui_fixed_adjacent_tt_tf import ScalarAdjacentTTConfig,scalar_adjacent_state_fixed_tt_score
        d,horizon = tf.float64,3*size
        convention = highdim.MeasureConvention(density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
            mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,reference_weight_name="omega")
        coordinate = highdim.AffineCoordinateMap(offset=tf.zeros([1],d),matrix=tf.constant([[6.]],d))
        def settings(dimension,ranks,schedule):
            basis = highdim.ProductBasis([highdim.LegendreBasis1D(highdim.BoundedInterval(-1.,1.),6)
                                          for _ in range(dimension)],convention)
            return highdim.FixedBranchFilterConfig(fit_config=highdim.FixedTTFitConfig(ranks=ranks,
                ridge=1e-10,max_sweeps=2,sweep_order=schedule,row_budget=512,column_budget=128,
                dense_matrix_byte_budget=2_000_000,normal_matrix_byte_budget=200_000,
                condition_number_warning=1e12,condition_number_veto=1e16,holdout_tolerance=1.),
                density_tau=0.,normalizer_floor=1e-14,denominator_floor=1e-14,
                retained_storage_byte_budget=10_000_000,coordinate_maps=(coordinate,),measure_convention=convention,
                deterministic_seed="repair-september-scalar-tt",product_basis=basis,
                initial_cores=highdim.norm_balanced_initial_cores(basis,ranks),fit_quadrature_order=7)
        config = ScalarAdjacentTTConfig(initial=settings(1,(1,1),(0,)),adjacent=settings(2,(1,2,1),(0,1,1,0)),
                                         scalar_coordinate_map=coordinate)
        model = highdim.ExactTransformedSVSSM(sigma=1.)
        theta = model.unconstrained_from_physical(gamma=.6,beta=.4)
        observations = tf.tile(tf.math.log(tf.constant([[.35],[-.22],[.41]],d)**2),[size,1])
        try:
            from bayesfilter.highdim.scalar_adjacent_native_tf import make_scalar_adjacent_state_fixed_tt
        except ImportError:
            def evaluate(theta,y):
                row = scalar_adjacent_state_fixed_tt_score(model,theta,y,config,finite_difference_h=())
                return row.log_likelihood,row.score
        else:
            call = make_scalar_adjacent_state_fixed_tt(model,config,observations.shape,jit_compile=jit)[1]
            def evaluate(theta,y):
                (increments,_,_),score = call(theta,y)
                return tf.reduce_sum(increments),score
        return evaluate,(theta,observations),dict(horizon=horizon,state=1,parameters=2,rank=2,degree=6,
            order=7,sweeps=2,score="same_finite_program_autodiff_diagnostic")
    if name == "particle":
        from experiments.dpf_implementation.tf_tfp.filters.ledh_pfpf_ot_tf import run_ledh_pfpf_ot_tf
        from experiments.dpf_implementation.tf_tfp.flows.ledh_tf import gaussian_logpdf_tf,ledh_flow_batch_tf
        d,horizon,count = tf.float64,2*size,4*size
        scale = tf.constant(.72,d)
        transition = lambda x,seed,date:x*scale+tf.cast(date+1,d)*.03
        transition_log = lambda x,old,date:gaussian_logpdf_tf(x-old*scale,tf.constant([[.2]],d))
        observation_log = lambda x,y,date:gaussian_logpdf_tf(x-y,tf.constant([[.15]],d))
        def flow(x,old,y,date):
            return ledh_flow_batch_tf(pre_flow_particles=x,ancestors=old,observation=y,
                transition_matrix=tf.reshape(scale,[1,1]),transition_covariance=tf.constant([[.2]],d),
                observation_covariance=tf.constant([[.15]],d),observation_fn=lambda x:x,
                observation_jacobian_fn=lambda x:tf.ones([1,1],d),observation_residual_fn=lambda predicted,y:y-predicted)
        y = tf.reshape(tf.linspace(tf.constant(-.1,d),.2,horizon),[horizon,1])
        initial = tf.reshape(tf.linspace(tf.constant(-.2,d),.3,count),[count,1])
        try:
            from experiments.dpf_implementation.tf_tfp.filters.particle_ot_execution_tf import make_particle_ot_filter
        except ImportError:
            def primal(y,initial):
                row = run_ledh_pfpf_ot_tf(observations=y,initial_sample=lambda n,seed:initial,
                    transition_sample=transition,observation_log_density=observation_log,
                    ledh_flow=flow,transition_log_density=transition_log,seed=17,num_particles=count,
                    ess_threshold_ratio=1.1,sinkhorn_epsilon=.8,sinkhorn_iterations=20,sinkhorn_tolerance=1e-6)
                return row.log_likelihood_estimate,row.filtered_means,row.filtered_variances,row.ess_by_time
        else:
            call = make_particle_ot_filter(tf.TensorSpec(y.shape,d),tf.TensorSpec(initial.shape,d),
                transition_sample=transition,observation_log_density=observation_log,ledh_flow=flow,
                transition_log_density=transition_log,seed=17,ess_threshold_ratio=1.1,jit_compile=jit,
                transport_method="fixed_target_sinkhorn",epsilon=.8,sinkhorn_iterations=20,sinkhorn_tolerance=1e-6,
                annealed_scaling=.9,annealed_convergence_threshold=1e-3,retained_teacher_warmstart_fn=None)
            def primal(y,initial):
                value,history=call(y,initial)
                return value,*history[:3]
        def evaluate(y,initial):
            with tf.GradientTape() as tape:
                tape.watch(y)
                result=primal(y,initial)
            return *result,tape.gradient(result[0],y)
        return evaluate,(y,initial),dict(horizon=horizon,particles=count,state=1,
            resampling="active_fixed_sinkhorn",canonical_admitted=False,score="observation_autodiff_diagnostic")
    if name == "retained_moments":
        import inspect
        from types import SimpleNamespace
        from bayesfilter.highdim.squared_tt_engine_v0_tf import _product_basis, _initial_tt_cores
        from bayesfilter.highdim.retained_quadratic_form_tf import prefix_gram_matrix
        from bayesfilter.highdim.retained_moments_tf import retained_reference_moments
        from bayesfilter.highdim.tt import TTCore
        dimension = 2*size
        basis = _product_basis(dimension, 3)
        cores = tuple(core.values for core in _initial_tt_cores(dimension,4,2))
        options = {"jit_compile": jit} if "jit_compile" in inspect.signature(retained_reference_moments).parameters else {}
        def evaluate(*values):
            objects = tuple(TTCore(value) for value in values)
            gram = tf.eye(objects[-1].right_rank,dtype=tf.float64)
            normalizer = tf.einsum("ab,ab->",prefix_gram_matrix(objects,basis),gram)+1e-6
            retained = SimpleNamespace(prefix_cores=objects,prefix_basis=basis,suffix_gram=gram,
                tau=tf.constant(1e-6,tf.float64),z_complete_ref=normalizer)
            return retained_reference_moments(retained, **options)
        return evaluate,cores,dict(dimension=dimension,degree=3,rank=2)
    if name == "contract_e":
        from bayesfilter.highdim.ledh_contract_e_canonical_lgssm_tf import canonical_value_and_score_core
        d,horizon = tf.float64,2*size
        theta = tf.constant([.2,.3,.4,.5,.8],d)
        design = tf.constant([[1.,1.,1.],[1.,-1.,-1.],[-1.,1.,-1.],[-1.,-1.,1.]],d)
        observations = tf.reshape(tf.linspace(tf.constant(-.1,d),.2,horizon*3),[horizon,3])
        initial = tf.stack((design*.25,design*.3))
        noise = tf.repeat(tf.stack((design*.2,design*(-.15)))[:,None],horizon,axis=1)
        mask = tf.math.floormod(tf.range(2)[:,None]+tf.range(horizon)[None,:],2)==0
        residual = tf.broadcast_to(design,[2,horizon,4,3])
        ridge = tf.fill([2,horizon],tf.constant(1e-5,d))
        def evaluate(theta,observations,initial,noise,mask,residual,ridge):
            result = canonical_value_and_score_core(theta,dict(observations=observations,initial_noise=initial,
                transition_noise=noise,fixed_reset_mask=mask,residual_design=residual,prepared_ridge=ridge,
                epsilon=tf.constant(2.,d),scaling=tf.constant(.9,d)),steps=2,balance_steps=100,
                row_chunk_size=4,col_chunk_size=4)
            return result["objective"],result["per_batch_log_likelihood"],result["score"],result["per_batch_score"]
        return evaluate,(theta,observations,initial,noise,mask,residual,ridge),dict(horizon=horizon,
            batch=2,particles=4,state=3,parameters=5,reset="contract_e_chol_v1",canonical_admitted=False)
    if name == "genut":
        from bayesfilter.highdim.cubature_genut_batch_tf import BatchCandidateModelAdapter,batch_finite_value_score
        d,horizon = tf.float64,2*size
        def initial(theta,noise):
            return theta[:,None,:]+noise[None]
        def initial_tangent(theta,noise):
            return tf.ones([theta.shape[0],noise.shape[0],1,1],d)
        def transition(theta,x,noise,date):
            return .8*x+theta[:,None,:]+.1*noise[None]
        def transition_tangent(theta,x,noise,dx,date):
            return .8*dx+1.
        def observation(theta,x,y,date):
            return -.5*tf.reduce_sum((x-y)**2+.1*theta[:,None,:]**2,axis=-1)
        def observation_tangent(theta,x,dx,y,date):
            return -tf.reduce_sum((x-y)[...,None]*dx,axis=2)-.1*theta[:,None,:]
        adapter = BatchCandidateModelAdapter(1,1,initial,initial_tangent,transition,transition_tangent,observation,observation_tangent)
        theta = tf.constant([[.05],[.1]],d)
        noise = tf.constant([[-1.2],[-.4],[.4],[1.2]],d)
        process = tf.broadcast_to(noise,[horizon,4,1])
        observations = tf.zeros([horizon,1],d)
        design = noise/tf.sqrt(tf.reduce_mean(noise**2))
        def evaluate(theta,y,noise,process,design):
            value,score,status = batch_finite_value_score(adapter,theta,y,noise,process,design,
                epsilon=8.,sinkhorn_steps=2,balance_steps=2,ridge=1e-5,
                transition_before_first_observation=True,higher_moment_correction_steps=2,
                higher_moment_strength=.1,higher_moment_floor=1e-5)
            return value,score,status["program_valid"]
        return evaluate,(theta,observations,noise,process,design),dict(horizon=horizon,batch=2,
            particles=4,state=1,parameters=1,higher_moment_steps=2,canonical_admitted=False)
    if name in ("tt", "tt_adapted", "tt_gaussian", "tt_adjoint"):
        from bayesfilter.highdim.squared_tt_engine_v0_tf import DensityKernelAdapter, EngineConfig
        import bayesfilter.highdim.squared_tt_engine_xla_tf as generic
        d, horizon = tf.float64, 3*size
        # Matched frozen September engineering fixture; no prior scientific
        # result supplies data, tuning choices, or promotion evidence.
        def normal(x,variance):
            return -.5*tf.reduce_sum(x*x/variance+tf.math.log(tf.constant(2*3.141592653589793*variance,d)),axis=-1)
        adapter = DensityKernelAdapter(1,lambda x,old: normal(x-.8*old,.05),
            lambda x,y: normal(x-y,.2),lambda x:normal(x,.1))
        observations = tf.zeros([horizon,1],d)
        config = EngineConfig(basis_degree=2,rank=2,row_count=64,sweeps=2,ridge=1e-10,
            tau=1e-6,coordinate_half_width=3.,seed=7781,row_design="sobol")
        dimensions = dict(horizon=horizon,state=1,rank=2,degree=2,rows=64,sweeps=2)
        if name == "tt":
            if hasattr(generic,"make_value_filter_branch_axis_xla"):
                call = generic.make_value_filter_branch_axis_xla(adapter,observations.shape,config,jit_compile=jit)
                def evaluate(y):
                    value,history = call(y)
                    return value,history[:,0]
            else:
                def evaluate(y):
                    value,report = generic.run_value_filter_branch_axis_xla(adapter,y,config)
                    return value,tf.constant([row["log_increment"] for row in report],d)
            return evaluate,(observations,),dimensions
        if name == "tt_adjoint":
            import bayesfilter.highdim.squared_tt_adjoint_engine_tf as adjoint
            kwargs = dict(transition_vjp=lambda x,old,cot:tf.einsum("n,ni->i",cot,(x-.8*old)/.05),
                observation_vjp=lambda x,y,cot:tf.zeros([1],d),initial_vjp=lambda x,cot:tf.zeros([1],d),parameter_dim=1)
            try:
                from bayesfilter.highdim.squared_tt_native_adjoint_engine_tf import make_adjoint_filter
            except ImportError:
                evaluate = lambda y: adjoint.run_adjoint_score_filter(adapter,y,config,**kwargs)
            else:
                call = make_adjoint_filter(adapter,observations.shape,config,jit_compile=jit,**kwargs)
                evaluate = lambda y: call(y)[:2]
            return evaluate,(observations,),dimensions
        means = tf.zeros([horizon-1,2],d)
        covariances = tf.repeat((tf.eye(2,dtype=d)*(.005 if name == "tt_adapted" else .1))[None],horizon-1,axis=0)
        if name == "tt_adapted":
            import bayesfilter.highdim.squared_tt_engine_adapted_xla_tf as adapted
            if hasattr(adapted,"make_value_filter_branch_axis_adapted_xla"):
                call = adapted.make_value_filter_branch_axis_adapted_xla(adapter,observations.shape,config,jit_compile=jit)
                def evaluate(y,means,covariances):
                    value,history = call(y,means,covariances)
                    return value,history[:,0]
            else:
                def evaluate(y,means,covariances):
                    value,report = adapted.run_value_filter_branch_axis_adapted_xla(adapter,y,config,
                        predictive_moment_hint=lambda t,y:(means[t-1],covariances[t-1]))
                    return value,tf.constant([row["log_increment"] for row in report],d)
            return evaluate,(observations,means,covariances),dimensions
        initial_mean,initial_cov = tf.zeros([1],d),tf.eye(1,dtype=d)*.1
        try:
            from bayesfilter.highdim.squared_tt_gaussian_native_tf import make_gaussian_value_filter
        except ImportError:
            from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import run_value_filter_branch_axis_gaussian_xla
            def evaluate(y,initial_mean,initial_cov,means,covariances):
                value,report = run_value_filter_branch_axis_gaussian_xla(adapter,y,config,
                    initial_moment_hint=lambda y:(initial_mean,initial_cov),
                    predictive_moment_hint=lambda t,y:(means[t-1],covariances[t-1]))
                return value,tf.constant([row["log_increment"] for row in report],d)
        else:
            call = make_gaussian_value_filter(adapter,observations.shape,config,jit_compile=jit)
            def evaluate(y,initial_mean,initial_cov,means,covariances):
                value,history = call(y,initial_mean,initial_cov,means,covariances)
                return value,history[0][:,0]
        return evaluate,(observations,initial_mean,initial_cov,means,covariances),dimensions
    if name == "joint_target":
        from bayesfilter.hardbound.joint_target_tf import GateModel, gate_joint_log_prob_batched
        parameters = tf.constant([[.005,-7.6],[.004,-7.4]],tf.float64)
        raw = tf.reshape(tf.linspace(tf.constant(-.2,tf.float64),.2,2*(4*size+1)),[2,4*size+1])
        y = tf.fill([4*size],tf.constant(.005,tf.float64))
        model = GateModel(horizon=4*size)
        def evaluate(parameters,raw,y):
            with tf.GradientTape() as tape:
                tape.watch((parameters,raw))
                value = gate_joint_log_prob_batched(y,parameters,raw,model)
            return (value,*tape.gradient(value,(parameters,raw)))
        return evaluate,(parameters,raw,y),dict(batch=2,horizon=4*size,state=1,parameters=2)
    if name == "cpu_pool":
        from bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf import batch_native_complexity_posterior_target
        target = batch_native_complexity_posterior_target(1,jit_compile=jit)
        rows = tf.tile(tf.constant([[.35,-.08,.65,.05],[.37,-.06,.63,.07]],tf.float64),[size,1])
        return target.neutra_batch_log_prob_and_grad_status,(rows,),dict(batch=2*size,parameters=4,
            consumer="batched_cpu_pool_numerical_shard; process_orchestration_tested_separately")
    if name == "tt_actual":
        from bayesfilter.highdim.zhao_cui_actual_sv_batched_tt_tf import (
            batched_fixed_tt_likelihood_analytic_score_status,
        )
        # Frozen September fixture, independent of either arm's preparation.
        # Every prepared tensor is an explicit measured input.
        order, degree, rank = 25, 10, 2
        k = tf.cast(tf.range(1, order), tf.float64)
        off = k / tf.sqrt(4.0 * k * k - 1.0)
        jacobi = tf.linalg.diag(off, k=1) + tf.linalg.diag(off, k=-1)
        nodes, vectors = tf.linalg.eigh(jacobi)
        weights = tf.square(vectors[0])
        grid = tf.stack((tf.repeat(nodes, order), tf.tile(nodes, [order])), axis=1)
        grid_weights = tf.reshape(weights[:, None] * weights[None, :], [-1])
        polys = tf.TensorArray(tf.float64, degree + 1).write(0, tf.ones_like(nodes)).write(1, nodes)
        def polynomial_body(i, previous, current, array):
            n = tf.cast(i, tf.float64)
            following = ((2.0 * n + 1.0) * nodes * current - n * previous) / (n + 1.0)
            return i + 1, current, following, array.write(i + 1, following)
        _, _, _, polys = tf.while_loop(lambda i, *_: i < degree, polynomial_body,
                                      (tf.constant(1), tf.ones_like(nodes), nodes, polys))
        basis = tf.transpose(polys.stack()) * tf.sqrt(tf.cast(2 * tf.range(degree + 1) + 1, tf.float64))
        initial = tf.reshape(tf.one_hot(0, degree + 1, dtype=tf.float64), [1, degree + 1, 1])
        adjacent0 = tf.reshape(tf.eye(degree + 1, rank, dtype=tf.float64), [1, degree + 1, rank])
        adjacent1 = tf.transpose(adjacent0, [2, 1, 0])
        names = ("transformed_observations", "initial_core", "adjacent_core0", "adjacent_core1",
                 "reference_nodes", "reference_weights", "reference_grid", "reference_grid_weights",
                 "basis_nodes", "basis_grid_axis0", "basis_grid_axis1")
        prepared = (tf.linspace(tf.constant(-1.6, tf.float64), 0.4, 4 * size), initial,
                    adjacent0, adjacent1, nodes, weights, grid, grid_weights, basis,
                    tf.repeat(basis, order, axis=0), tf.tile(basis, [order, 1]))
        theta = tf.constant([[0.30, -0.20], [0.33, -0.22]], tf.float64)
        def evaluate(theta, *prepared):
            return batched_fixed_tt_likelihood_analytic_score_status(theta, **dict(zip(names, prepared)))
        return evaluate, (theta, *prepared), {"batch": 2, "horizon": 4 * size, "degree": degree, "rank": rank, "order": order, "parameters": 2}
    if name in ("rectangular", "factor", "covariance", "sinkhorn_jvp"):
        if size != 1:
            raise ValueError("Legacy audit fixture has one frozen size")
        return audit_fixture(tf, name, jit)
    if name == "sqmc":
        from bayesfilter.highdim.sqmc_tf import hilbert_permutation
        count, dim, bits = 256 * size, 4 * size, 8
        points = tf.reshape(tf.sin(tf.cast(tf.range(count * dim), tf.float64) * 0.31), [count, dim])
        def evaluate(x):
            return hilbert_permutation(x, tf.zeros([dim], x.dtype), tf.ones([dim], x.dtype), bits=bits)
        return evaluate, (points,), {"particles": count, "dimension": dim, "bits": bits}
    if name == "dns":
        import inspect
        from bayesfilter.hardbound.dns_curve_tf import yield_curve
        factors = tf.reshape(tf.linspace(tf.constant(-0.04, tf.float64), 0.05, 96 * size), [32 * size, 3])
        maturities = tf.constant([0.25, 0.5, 1., 2., 5., 10.], tf.float64)
        options = {"jit_compile": jit} if "jit_compile" in inspect.signature(yield_curve).parameters else {}
        def evaluate(x):
            return (yield_curve(x, maturities, 0.5, 0., 0.01, "softplus", **options),)
        return evaluate, (factors,), {"batch": 32 * size, "maturities": 6, "quadrature": 40}
    if name == "sgqf_derivatives":
        from bayesfilter.highdim.models import p30_predator_prey_fixture_model
        from bayesfilter.nonlinear.fixed_sgqf_structural_adapter_tf import (
            tf_predator_prey_to_fixed_sgqf_model,
        )
        model = p30_predator_prey_fixture_model()
        adapter = tf_predator_prey_to_fixed_sgqf_model(model, model.true_parameters(), with_derivatives=True)
        points = tf.reshape(tf.linspace(tf.constant(0.3, tf.float64), 1.6, 32 * size), [16 * size, 2])
        def evaluate(x):
            return adapter.derivatives.transition_state_jacobian_fn(x), adapter.derivatives.d_transition_fn(x)
        return evaluate, (points,), {"points": 16 * size, "state": 2, "parameters": 6}
    raise ValueError(f"Fixture {name} is not qualified yet; missing coverage blocks merge")


def measure(args, result):
    # Set source precedence before any numerical import, including fixture imports.
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    if args.device == "CPU":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import tensorflow as tf

    from bayesfilter.runtime.gpu_memory_policy import (
        configure_tensorflow_gpu_memory_growth,
    )
    result["memory_policy"] = configure_tensorflow_gpu_memory_growth(tf, require_gpu=args.device == "GPU")
    result["tensorflow"] = tf.__version__
    result["tf32"] = tf.config.experimental.tensor_float_32_execution_enabled()
    result["environment"] = {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OPENBLAS_NUM_THREADS", "TF_XLA_FLAGS", "XLA_FLAGS")}
    if tf.config.optimizer.get_jit():
        raise RuntimeError("Global auto-JIT invalidates graph comparison")
    if args.device == "GPU":
        result["hardware"] = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version", "--format=csv"], text=True).strip()
        result["trust_basis"] = "trusted_escalated_gpu_diagnostic"
    else:
        result["hardware"] = {"machine":platform.machine(),"processor":next((row for row in
            Path("/proc/cpuinfo").read_text().splitlines() if row.startswith("model name")),platform.processor())}
    def snapshot():
        out = host_memory()
        if args.device == "GPU":
            out["gpu"] = tf.config.experimental.get_memory_info("GPU:0")
        return out
    def reset_peak():
        if args.device == "GPU":
            tf.config.experimental.reset_memory_stats("GPU:0")
    with tf.device(f"/{args.device}:0"):
        result["phase"] = "preparation"
        start = time.perf_counter()
        evaluate, inputs, dimensions = fixture(tf, args.fixture, args.size, args.jit == "on")
        result["numerical_backend"] = getattr(evaluate, "execution_backend", "tensorflow")
        # Legacy fixture loader prepends its own root. Source packages are already
        # resolved; restore baseline precedence for all subsequent lazy imports.
        sys.path.insert(0, str(source))
        result["preparation_seconds"] = time.perf_counter() - start
        result["dimensions"] = dimensions
        input_values = [value.numpy().tolist() for value in inputs]
        result["input_sha256"] = hashlib.sha256(json.dumps(input_values, allow_nan=False).encode()).hexdigest()
        result["input_shapes"] = [value.shape.as_list() for value in inputs]
        result["input_dtypes"] = [value.dtype.name for value in inputs]
        result["stages"]["prepared"] = snapshot()
        result["phase"] = "trace"
        target = evaluate if args.jit == "eager" else tf.function(evaluate, input_signature=[tf.TensorSpec(value.shape, value.dtype) for value in inputs], autograph=False, jit_compile=args.jit == "on")
        reset_peak()
        start = time.perf_counter()
        concrete = None if args.jit == "eager" else target.get_concrete_function()
        result["trace_seconds"] = time.perf_counter() - start
        result["stages"]["traced"] = snapshot()
        graph = None if concrete is None else concrete.graph.as_graph_def()
        nodes = [] if graph is None else list(graph.node) + [node for fn in graph.library.function for node in fn.node_def]
        result["graph"] = {"nodes": len(nodes), "functions": 0 if graph is None else len(graph.library.function), "callbacks": sorted({node.op for node in nodes} & {"PyFunc", "PyFuncStateless", "EagerPyFunc"}), "must_compile": concrete is not None and bool(concrete.function_def.attr.get("_XlaMustCompile").b), "nested_xla": [] if graph is None else [fn.signature.name for fn in graph.library.function if fn.attr.get("_XlaMustCompile") and fn.attr["_XlaMustCompile"].b], "xla_ops": sorted({node.op for node in nodes if "Xla" in node.op})}
        if args.jit == "eager":
            result["execution_exception"] = "existing_host_execution_reference_only; nested compilation may exist"
        if result["graph"]["callbacks"] or (args.jit == "off" and (result["graph"]["nested_xla"] or result["graph"]["xla_ops"])):
            raise RuntimeError("Invalid callback/compilation boundary")
        def call():
            start = time.perf_counter()
            values = tf.nest.flatten(target(*inputs))
            tf.test.experimental.sync_devices()
            copy_start = time.perf_counter()
            host = [value.numpy() for value in values]
            end = time.perf_counter()
            metrics = {"synchronized_seconds": copy_start - start,
                       "output_copy_seconds": end - copy_start,
                       "end_to_end_seconds": end - start}
            return values, host, metrics
        reset_peak()
        result["phase"] = "first_execution"
        values, host, timing = call()
        result["cold"] = timing
        result["output_shapes"] = [value.shape.as_list() for value in values]
        result["output_dtypes"] = [value.dtype.name for value in values]
        result["output_devices"] = [value.device for value in values]
        result["values"] = [value.tolist() for value in host]
        result["stages"]["cold_outputs_live"] = snapshot()
        del values, host
        gc.collect()
        result["stages"]["cold_outputs_released"] = snapshot()
        result["warm"] = []
        result["phase"] = "warm"
        for iteration in range(20):
            reset_peak()
            values, host, timing = call()
            serialized = [value.tolist() for value in host]
            if serialized != result["values"]:
                raise RuntimeError("Repeated fixed-input output changed")
            del values, host, serialized
            result["warm"].append({"iteration": iteration, **timing, **snapshot()})
        result["trace_count"] = None if args.jit == "eager" else target.experimental_get_tracing_count()
        if args.jit != "eager" and result["trace_count"] != 1:
            raise RuntimeError("Unbounded trace contract")
        if args.jit == "on":
            result["phase"] = "hlo"
            hlo = target.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
            hlo_path = args.output.with_suffix(".hlo.txt")
            with hlo_path.open("x") as handle:
                handle.write(hlo)
            result["hlo_sha256"] = hashlib.sha256(hlo.encode()).hexdigest()
    result["imported_source_sha256"] = imported_sources(source)
    result["phase"] = "complete"
    result["status"] = "passed"


def imported_sources(source):
    imported = {}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename and filename.endswith(".py"):
            path = Path(filename).resolve()
            if path.is_relative_to(source):
                imported[str(path.relative_to(source))] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif getattr(module, "__name__", "").startswith(("bayesfilter.", "experiments.dpf_implementation.")):
                raise RuntimeError(f"Repository import escaped measured source: {filename}")
    return imported


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--jit", choices=("on", "off", "eager"), required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--device", choices=("GPU", "CPU"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    started = time.perf_counter()
    result = {"schema": "filter_repair_measurement.v2", "fixture": args.fixture, "size": args.size, "jit": args.jit, "device": args.device, "source_root": str(args.source_root), "worker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "harness_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(__file__).with_name("measure_filter_xla_memory.py"),Path(__file__).with_name("filter_repair_endpoint_fixtures.py"))}, "stages": {"start": host_memory()}, "status": "failed", "phase": "initialization"}
    try:
        measure(args, result)
    except Exception as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = traceback.format_exc()
        raise
    finally:
        result["imported_source_sha256"] = imported_sources(args.source_root.resolve())
        result["wall_seconds"] = time.perf_counter() - started
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps({key: result[key] for key in ("fixture", "size", "jit", "status", "wall_seconds")}))


if __name__ == "__main__":
    main()
