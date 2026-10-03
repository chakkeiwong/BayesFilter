"""Numerical mechanisms versus independently written diagnostic oracles."""
from __future__ import annotations
import numpy as np
import tensorflow as tf

from ..targets import ValidationTarget
from ..references import analytic
from ..designs import seed_for
from ..storage import write_json


def run(design, root, deadline=None):
    data = design.options.get("data")
    target=ValidationTarget(design.scenario.target,design.scenario.parameters,data,
        control=design.scenario.control if design.scenario.control in {"wrong_score","omit_jacobian"} else "baseline",
        jit_compile=design.device=="gpu")
    rng=np.random.default_rng(seed_for(design.seed,design.design_id,"probes"))
    probes=rng.normal(size=(design.replications,target.parameter_dim))
    value,score=target.log_prob_and_grad(probes)
    reference=analytic.log_density(target.target_id,probes,target.parameters,data)
    gradients=np.empty_like(probes)
    for j in range(target.parameter_dim):
        h=np.cbrt(np.finfo(float).eps)*np.maximum(1.,np.abs(probes[:,j]))
        plus,minus=probes.copy(),probes.copy(); plus[:,j]+=h; minus[:,j]-=h
        gradients[:,j]=(analytic.log_density(target.target_id,plus,target.parameters,data)-analytic.log_density(target.target_id,minus,target.parameters,data))/(2*h)
    value_error=float(np.max(np.abs(value.numpy()-reference)/(1+np.abs(reference))))
    score_error=float(np.max(np.abs(score.numpy()-gradients)/(1+np.abs(gradients))))
    # FD truncation/rounding diagnostic tolerance, deliberately independent of
    # the implementation's values. Unscaled raw errors are retained as well.
    tolerance=100*np.cbrt(np.finfo(float).eps)**2
    results={"density_relative_error":value_error,"score_relative_error":score_error,
             "score_tolerance":tolerance,"density_tolerance":1e-10,
             "score_checked_against":"independent scipy law centered finite difference",
             "density_passed":value_error<=1e-10,"score_passed":score_error<=tolerance,
             "probes":probes.tolist(),"control":design.scenario.control}
    results["observed_values"]=value.numpy().tolist()
    results["observed_scores"]=score.numpy().tolist()
    if target.target_id=="gaussian":
        from tensorflow_probability.python.mcmc.internal.leapfrog_integrator import SimpleLeapfrogIntegrator
        eps=design.step_size; length=design.leapfrog_steps
        mass=np.array([[2., .3], [.3, .8]])
        inverse=np.linalg.inv(mass)
        kinetic_matrix=tf.constant(mass if design.scenario.control=="wrong_metric" else inverse,tf.float64)
        q=tf.constant(probes,tf.float64); p=tf.constant(rng.normal(size=probes.shape),tf.float64)
        def integrate(q,p,epsilon):
            integrator=SimpleLeapfrogIntegrator(lambda x:target.log_density(x),
                [tf.constant(epsilon, tf.float64)],length)
            momenta,states,_,_=integrator([p],[q],kinetic_energy_fn=lambda p:
                .5*tf.reduce_sum(p*tf.linalg.matvec(kinetic_matrix,p),-1))
            return states[0],momenta[0]
        def roundtrip(q,p):
            q1,p1=integrate(q,p,eps)
            q2,p2=integrate(q1,p1,-eps)
            return q1,p1,q2,p2
        compiled=tf.function(roundtrip,input_signature=[tf.TensorSpec(probes.shape,tf.float64)]*2,
            autograph=False,jit_compile=design.device=="gpu")
        q1,p1,q2,p2=compiled(q,p)
        scale=target.parameters.get("scale",1.)
        # Independent matrix-power harmonic oscillator oracle for full L steps.
        eye=np.eye(2); precision=eye/scale**2
        a=eye-eps**2/2*inverse@precision
        b=eps*inverse
        c=-eps*precision+eps**3/4*precision@inverse@precision
        matrix=np.block([[a,b],[c,a.T]])
        power=np.linalg.matrix_power(matrix,length)
        state=np.concatenate([probes,p.numpy()],-1)@power.T
        err=float(max(np.max(abs(q1.numpy()-state[...,:2])),np.max(abs(p1.numpy()-state[...,2:]))))
        reverse=float(max(tf.reduce_max(abs(q2-q)),tf.reduce_max(abs(p2-p))))
        results.update(leapfrog_error=err,reversal_error=reverse,
            leapfrog_passed=err<1e-9 and reverse<1e-9, integrator="TFP SimpleLeapfrogIntegrator",
            metric_convention="K(p)=p.T @ inverse_mass @ p / 2; fixed dense SPD fixture",
            independent_map_symplectic_error=float(np.max(abs(power.T@np.block([[np.zeros((2,2)),eye],[-eye,np.zeros((2,2))]])@power-np.block([[np.zeros((2,2)),eye],[-eye,np.zeros((2,2))]])))),
            energy_change=(.5*tf.reduce_sum((q1*q1-q*q)/scale**2+p1*tf.linalg.matvec(tf.constant(inverse),p1)-p*tf.linalg.matvec(tf.constant(inverse),p),axis=-1)).numpy().tolist(),
            trace_count=compiled.experimental_get_tracing_count())
    results["finding"]="mechanics_passed" if all(v for k,v in results.items() if k.endswith("passed")) else "mechanics_discrepancy"
    results["accuracy_established"]=False
    write_json(root/"mechanics.json",results)
    return results
