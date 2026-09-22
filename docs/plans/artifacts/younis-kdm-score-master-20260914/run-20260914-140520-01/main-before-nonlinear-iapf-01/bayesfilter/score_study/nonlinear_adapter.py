"""Master consumer of checked scalar nonlinear references and proposal kernels."""
import math
import time
from .contracts import digest,seed_pair
from .runtime import configure_runtime,memory_usage
from .tuning import TUNABLE_PROPOSALS,candidate_configuration,selected_row


def evaluate_nonlinear(row,context):
    if row.get("role")=="claim" and row["proposal"] in TUNABLE_PROPOSALS:
        row=selected_row(row,context)
    settings=context["study"]["settings"]
    if settings["dimension"]!=1 or settings["observation_dimension"]!=1:
        raise ValueError("nonlinear scalar adapter requires d=o=1")
    c,b=(settings[k] for k in ("transition_curve","observation_curve"))
    if not all(isinstance(x,(int,float)) and math.isfinite(x) for x in (c,b)):
        raise ValueError("explicit finite nonlinear coefficients required")
    reference=settings["reference"]
    points,radius=reference["points"],reference["radius"]
    tolerance,tail_tolerance=reference["relative_tolerance"],reference["tail_tolerance"]
    if (points<65 or points%4!=1 or radius<=0 or not 0<tolerance<=1e-4 or not 0<tail_tolerance<=1e-6):
        raise ValueError("invalid reference refinement specification")
    runtime=configure_runtime(device=settings["device"],tf32=settings["tf32"],jit_compile=settings["jit_compile"])
    import tensorflow as tf
    from .nonlinear_tf import make_data_kernel,make_grid_reference,make_ledh_kernel,make_moment_filter,make_particle_filter
    dtype=tf.as_dtype(settings["dtype"]);N,T=settings["particles"],settings["horizon"]
    theta=tf.constant(settings["theta"],dtype)
    def seed(stream,replicate=None,group=None):
        return seed_pair(master_seed=context["study"]["seed"],model=row["model"],dataset=row["dataset"],
            replicate=row["replicate"] if replicate is None else replicate,stream=stream,
            coupling_group=group or row.get("coupling_group","baseline"))
    data_seed=seed("observations",0,"common_data")
    # Generate and check one physical dataset in FP64 on CPU, including in GPU
    # studies. This avoids treating GPU-roundoff differences as new datasets.
    with tf.device("/CPU:0"):
        theta64=tf.constant(settings["theta"],tf.float64)
        observations64=make_data_kernel(T,c,b,"float64",True)(
            tf.constant(settings["data_theta"],tf.float64),tf.constant(data_seed,tf.int32))
        grid=make_grid_reference(T,points,radius,c,b,"float64",True)
        oracle=grid(theta64,observations64)
        coarse=make_grid_reference(T,(points+1)//2,radius,c,b,"float64",True)(theta64,observations64)
        expanded=make_grid_reference(T,points+((points-1)//2),radius*1.5,c,b,"float64",True)(theta64,observations64)
        def error(other):
            return max(float(tf.reduce_max(tf.abs(a-z)/(1+tf.abs(a))).numpy()) for a,z in zip(oracle[:4],other[:4]))
        mesh_error,domain_error=error(coarse),error(expanded)
        tail=max(float(x[k].numpy()) for x in (oracle,coarse,expanded) for k in (4,5))
        for part in oracle:tf.debugging.assert_all_finite(part,"grid reference")
        if not all(math.isfinite(x) for x in (mesh_error,domain_error,tail)) or max(mesh_error,domain_error)>tolerance or tail>tail_tolerance:
            raise ValueError(f"reference refinement veto: mesh={mesh_error}, domain={domain_error}, tail={tail}")
    diagnostics={"data_seed":data_seed,"data_version":digest(observations64.numpy().tolist()),
        "data_model":"scalar_sine_transition_quadratic_observation","transition_curve":c,"observation_curve":b,
        "initial_seed":seed("initial"),"process_seed":seed("process"),"resampling_seed":seed("resampling"),
        "reset_seed":seed("reset_design"),"state_dimension":1,"parameter_dimension":6,"particle_count":N,"horizon":T,
        "uses_autodiff":False,"oracle_kind":"refined_numerical_grid_not_exact","reference_device":oracle[0].device,
        "reference_dtype":"float64","reference_mesh_relative_error":mesh_error,"reference_domain_relative_error":domain_error,
        "reference_max_tail_mass":tail,"reference_controls":reference}
    proposal=row["proposal"]
    with tf.device("/GPU:0" if settings["device"]=="GPU" else "/CPU:0"):
        obs=tf.cast(observations64,dtype)
        initial=tf.random.stateless_normal([N,1],seed("initial"),dtype=dtype)
        process=tf.random.stateless_normal([T,N,1],seed("process"),dtype=dtype)
        uniforms=tf.random.stateless_uniform([T,N],seed("resampling"),dtype=dtype)
        design=tf.random.stateless_normal([N,1],seed("reset_design"),dtype=dtype)
        started=time.monotonic();calls=1
        if proposal=="grid_reference":
            # A reference row stays on the CPU reference backend and declares it.
            value,score=oracle[:2];kernel=grid
        elif proposal in ("ekf","ukf"):
            kernel=make_moment_filter(T,c,b,proposal,settings["dtype"],settings["jit_compile"])
            outputs=[kernel(theta,direction,obs) for direction in tf.unstack(tf.eye(6,dtype=dtype))]
            value=outputs[0][0];score=tf.stack([x[1] for x in outputs]);calls=6
            diagnostics["minimum_covariance"]=min(float(x[2].numpy()) for x in outputs)
        elif proposal in ("bootstrap","prior_sis","local_linear"):
            kernel=make_particle_filter(N,T,c,b,proposal=="local_linear",proposal!="prior_sis",settings["dtype"],settings["jit_compile"])
            value,score,ess=kernel(theta,obs,initial,process,uniforms)
            diagnostics.update(minimum_particle_ess=float(ess.numpy()),derivative_semantics="locally_fixed_multinomial_labels")
        else:
            controls=row["controls"]
            control={"ledh":1.,"sgqf":row.get("sgqf_level",2),"kdm_covariance":row.get("within_fraction",.5)}[proposal]
            collect_diagnostics=row.get("collect_control_diagnostics",False)
            kernel=make_ledh_kernel(N,T,tuple(sorted(controls.items())),c,b,proposal,control,settings["dtype"],settings["jit_compile"],
                                   return_diagnostics=collect_diagnostics)
            outputs=[kernel(theta,direction,obs,initial,process,design) for direction in tf.unstack(tf.eye(6,dtype=dtype))]
            values=tf.stack([x[0] for x in outputs]);value=values[0];score=tf.stack([x[1] for x in outputs]);calls=6
            tf.debugging.assert_near(values,tf.fill([6],value))
            if collect_diagnostics:
                from .control_diagnostics_tf import materialize_control_diagnostics
                diagnostics["control_diagnostics"]=materialize_control_diagnostics(outputs[0][2])
            diagnostics.update(controls=controls,reset_contract_id="contract_e_chol_v1",
                chunk_policy="dpf_transport_exact_divisor_cap3000_v1",chunk_size=N,
                derivative_semantics="analytical_total_initial_law_flow_genut_contract_e",
                score_consumer="nonlinear model -> shared LEDH executor with "+proposal,
                candidate_configuration=candidate_configuration(row))
        tf.debugging.assert_all_finite(value,"nonlinear value");tf.debugging.assert_all_finite(score,"nonlinear score")
        if value.shape!=() or score.shape!=(6,):raise ValueError("nonlinear output shape veto")
        value_number,score_numbers=float(value.numpy()),score.numpy().tolist()
        runtime.update(dtype=value.dtype.name,value_device=value.device,score_device=score.device,
            kernel_wall_seconds=time.monotonic()-started,kernel_calls=calls,
            traces=kernel.experimental_get_tracing_count(),**memory_usage(settings["device"]))
    diagnostics["score_squared_error"]=float(tf.reduce_sum((tf.cast(score,tf.float64)-oracle[1])**2).numpy())
    diagnostics["value_squared_error"]=float(((tf.cast(value,tf.float64)-oracle[0])**2).numpy())
    estimator=context["registry"].estimators[row["estimator"]]
    return {"value":value_number,"score":score_numbers,"oracle_value":float(oracle[0].numpy()),"oracle_score":oracle[1].numpy().tolist(),
        "value_target":"finite_grid_log_likelihood" if proposal=="grid_reference" else ("gaussian_approximate_log_likelihood" if proposal in ("ekf","ukf") else "finite_particle_log_likelihood"),
        "derivative_target":estimator.target,"comparison_target":row["comparison_target"],"derivative_id":estimator.derivative,
        "proposal_law":context["registry"].proposals[proposal].law,"initial_terms":True,"numerical_validity":"pass",
        "inference_status":"mechanics_only" if context["study"].get("evidence_class")=="mechanics" else "descriptive_only",
        "runtime":runtime,"diagnostics":diagnostics}
