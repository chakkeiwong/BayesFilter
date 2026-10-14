"""GPU/XLA identical-coefficient calibration of optional bounded covariance repair."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
sys.path.insert(0, "/home/chakwong/BayesFilter")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
import multivariate_controlled_smc_tf as kernel
import lorenz_refinement_tf as controller

D = tf.float64


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--eigen-tolerance", type=float)
    parser.add_argument("--eigen-refinement", action="store_true")
    parser.add_argument("--stable-integral",action="store_true")
    parser.add_argument("--activation-margin",type=float)
    args = parser.parse_args()
    start = time.monotonic()
    summary = dict(status="FAIL", replication="INCOMPLETE", device="GPU",
                   gpu_memory_policy=memory, dtype="float64", tf32=False,
                   activation_margin=args.activation_margin,stable_integral=args.stable_integral,jit_compile=True, tensorflow=tf.__version__, eigen_tolerance=args.eigen_tolerance,eigen_refinement=args.eigen_refinement, checks=[], candidates=[],
                   heuristic_dominance_verdict="NOT_ASSESSED_NUMERICAL_REPAIR_ONLY")
    def check(name, passed, **fields):
        summary["checks"].append(dict(name=name, passed=bool(passed), **fields))
    def compare(name, actual, expected, rows):
        expected = tf.constant(expected, D)
        scaled = tf.abs(actual-expected)/(1e-7+1e-9*tf.abs(expected))
        good = bool(tf.reduce_all(tf.math.is_finite(scaled) & (scaled<=1)).numpy())
        rows.append(dict(name=name, passed=good, max_scaled_error=float(tf.reduce_max(scaled).numpy())))
    try:
        path = Path(inspect.getfile(controller.MultivariateGaussianControlledSMC))
        check("consumer_shared_kernel_snapshot", path.read_bytes()==(Path(__file__).parent/path.name).read_bytes())
        payload = json.loads((args.reference/"calibration.json").read_text())
        check("reference_activation_matches",payload.get("activation_margin")==args.activation_margin)
        cases = payload["cases"]
        field = lambda key: tf.constant([row[key] for row in cases], D)
        fit = {key: tf.constant([row["fit"][key] for row in cases], D) for key in cases[0]["fit"]}
        fit["valid"] = tf.constant(True)
        inputs = (field("base_precision"), field("old_A"), fit,
                  field("mean"), field("old_b"), field("old_c"), field("x"))
        signature = tf.nest.map_structure(lambda value: tf.TensorSpec(value.shape, value.dtype), inputs)
        for floor in payload["floors"]:
            def calculation(base, old_a, fitted, mean, old_b, old_c, points):
                changed, safety = kernel.constrain_precision_increment(
                    base, old_a, fitted, "nearest_covariance", relative_floor=floor, eigen_tolerance=args.eigen_tolerance,eigen_refinement=args.eigen_refinement,activation_margin=args.activation_margin)
                integral, valid = kernel.centered_correction_integral(
                    mean, base, -2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(base))),axis=-1), old_a, old_b, old_c,
                    changed["center"], changed["scale"], changed["centered_A"],
                    changed["centered_b"], changed["centered_c"])
                if args.stable_integral:
                    integral,valid=kernel.projected_correction_integral(
                        mean,base,-2*tf.reduce_sum(tf.math.log(tf.linalg.diag_part(tf.linalg.cholesky(base))),axis=-1),
                        old_a,old_b,old_c,fitted["center"],fitted["scale"],
                        fitted["centered_A"],fitted["centered_b"],fitted["centered_c"],
                        changed["projection_added_A"])
                raw = kernel.quadratic(points, changed["A"], changed["b"], changed["c"])
                z = (points-changed["center"][:, None, :])/changed["scale"][:, None, :]
                centered = kernel.quadratic(z, changed["centered_A"], changed["centered_b"], changed["centered_c"])
                return changed, safety, integral, valid, raw, centered
            graph = tf.function(calculation, input_signature=signature, jit_compile=True, autograph=False)
            changed, safety, integral, valid, raw, centered = graph(*inputs)
            expected = [next(v for v in row["expected"] if v["floor"]==floor) for row in cases]
            checks = []
            for key in cases[0]["fit"]:
                compare("identical_coefficients_"+key, changed[key], [row["fit"][key] for row in expected], checks)
            compare("Gaussian_correction_integral", integral, [row["integral"] for row in expected], checks)
            compare("centered_raw_polynomial", raw, centered.numpy().tolist(), checks)
            healthy = tf.constant([not row["invalid"] for row in cases])
            no_fire = bool(tf.reduce_all(tf.boolean_mask(safety["protection_applied"], healthy)==0).numpy())
            unchanged = all(bool(tf.reduce_all(tf.boolean_mask(changed[key], healthy)==
                               tf.boolean_mask(fit[key], healthy)).numpy()) for key in cases[0]["fit"])
            checks.extend([dict(name="healthy_no_fire", passed=no_fire),
                           dict(name="healthy_bitwise_unchanged", passed=unchanged),
                           dict(name="valid_integral_and_coefficients", passed=bool(valid.numpy() and changed["valid"].numpy())),
                           dict(name="one_trace", passed=graph.experimental_get_tracing_count()==1)])
            # Constructor-to-fit wiring of the exact explicit floor, not a separate local policy.
            consumer = controller.LorenzRefinement(forcing=2.5, precision_policy="nearest_covariance", precision_floor=floor,precision_eigen_tolerance=args.eigen_tolerance,precision_eigen_refinement=args.eigen_refinement,precision_stable_integral=args.stable_integral,precision_activation_margin=args.activation_margin)
            checks.append(dict(name="consumer_floor_wiring", passed=consumer.engine.precision_floor==floor and consumer.engine.precision_eigen_tolerance==args.eigen_tolerance and consumer.engine.precision_eigen_refinement==args.eigen_refinement and consumer.engine.precision_stable_integral==args.stable_integral and consumer.engine.precision_activation_margin==args.activation_margin))
            summary["candidates"].append(dict(floor=floor, checks=checks,
                eligible=all(row["passed"] for row in checks) and floor>=1e-6,
                baseline_only=floor<1e-6,maximum_eigen_rotations=float(tf.reduce_max(safety["eigen_rotations"]).numpy()), maximum_eigen_residual=float(tf.reduce_max(safety["eigen_residual"]).numpy()),
                maximum_eigen_orthogonality=float(tf.reduce_max(safety["eigen_orthogonality"]).numpy()), max_A=float(tf.reduce_max(tf.abs(changed["A"])).numpy())))

        if args.activation_margin is not None:
            # Diagonal inputs isolate activation from eigensolver rounding.
            # Exact dyadic boundary at64epsilon; strict above/below otherwise.
            # Healthy but below rho; above/at/below the numerical boundary; invalid.
            boundary=args.activation_margin
            values=([0.005,2*boundary,boundary,0.5*boundary,-0.005] if boundary==64*2.220446049250313e-16 else [0.005,2*boundary,0.5*boundary,0.25*boundary,-0.005])
            minimum=tf.constant(values,D)
            target=tf.linalg.diag(tf.concat([minimum[:,None],tf.ones([5,7],D)],axis=1))
            identity=tf.eye(8,batch_shape=[5],dtype=D)
            fitted_A=.5*(target-identity)
            fixture=dict(A=fitted_A,b=tf.zeros([5,8],D),c=tf.zeros([5],D),
                centered_A=fitted_A,centered_b=tf.zeros([5,8],D),centered_c=tf.zeros([5],D),
                center=tf.zeros([5,8],D),scale=tf.ones([5,8],D),valid=tf.constant(True))
            def boundary_probe(base,fit):
                return kernel.constrain_precision_increment(
                    base,tf.zeros_like(base),fit,"nearest_covariance",relative_floor=0.01,
                    eigen_tolerance=args.eigen_tolerance,eigen_refinement=args.eigen_refinement,
                    activation_margin=args.activation_margin)
            boundary_graph=tf.function(boundary_probe,input_signature=[
                tf.TensorSpec(identity.shape,D),
                tf.nest.map_structure(lambda v:tf.TensorSpec(v.shape,v.dtype),fixture)],
                jit_compile=True,autograph=False)
            changed,observed=boundary_graph(identity,fixture)
            check("feasibility_boundary_activations",
                tf.reduce_all(observed["protection_applied"]==tf.constant([0.,0.,1.,1.,1.],D)).numpy(),
                margins=observed["raw_precision_margin"].numpy().tolist(),
                activations=observed["protection_applied"].numpy().tolist())
            check("healthy_below_repair_floor_unchanged",
                tf.reduce_all(changed["A"][:2]==fixture["A"][:2]).numpy())
            check("boundary_graph_valid",changed["valid"].numpy())
            check("boundary_graph_one_trace",boundary_graph.experimental_get_tracing_count()==1)
        if args.eigen_refinement:
            # Independent constructed stress matrices: repeated and zero eigenvalues,
            # indefinite spectra, and twelve orders of scale in each direction.
            basis=tf.linalg.qr(tf.reshape(tf.sin(tf.cast(tf.range(64),D)),[8,8])+tf.eye(8,dtype=D))[0]
            eigenvalues=tf.constant([[-4.,-1.,0.,0.,1.,1.,2.,2.],
                                     [1.,1.,1.,1.,1.,1.,1.,1.],
                                     [1.,1.,2.,2.,3.,3.,4.,4.]],D)
            matrices=tf.linalg.matmul(basis[None]*eigenvalues[:,None,:],basis,transpose_b=True)
            scaled=tf.reshape(matrices[None]*tf.constant([1e-12,1.,1e12],D)[:,None,None,None],[9,8,8])
            def stress(matrix):
                _,v=tf.linalg.eigh(matrix)
                w,v,n=kernel.refine_symmetric_eigenbasis(matrix,v,args.eigen_tolerance)
                scale=tf.sqrt(tf.reduce_sum(matrix**2,[-2,-1]))
                residual=tf.sqrt(tf.reduce_sum((tf.linalg.matmul(matrix,v)-v*w[:,None,:])**2,[-2,-1]))/scale
                orth=tf.reduce_max(tf.abs(tf.linalg.matmul(v,v,transpose_a=True)-tf.eye(8,dtype=D)),[-2,-1])
                return residual,orth,n
            stress_graph=tf.function(stress,input_signature=[tf.TensorSpec([9,8,8],D)],jit_compile=True,autograph=False)
            residual,orth,n=stress_graph(scaled)
            check("scaled_repeated_indefinite_eigen_residual",
                  tf.reduce_all((residual<=256*8*2.220446049250313e-16)&(orth<=256*8*2.220446049250313e-16)).numpy(),
                  maximum_residual=float(tf.reduce_max(residual).numpy()),
                  maximum_orthogonality=float(tf.reduce_max(orth).numpy()),rotations=int(n.numpy()))
            check("stress_graph_one_trace",stress_graph.experimental_get_tracing_count()==1)
        summary["eligible_floors"] = [row["floor"] for row in summary["candidates"] if row["eligible"]]
        summary["status"] = "PASS" if all(row["passed"] for row in summary["checks"]) else "FAIL"
    except Exception as error:
        summary["error"] = str(error)
        import traceback
        traceback.print_exc()
    summary["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
    summary["wall_seconds"] = time.monotonic()-start
    (args.output/"summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=summary["status"], eligible_floors=summary.get("eligible_floors"),
                          error=summary.get("error"), wall_seconds=summary["wall_seconds"])))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
