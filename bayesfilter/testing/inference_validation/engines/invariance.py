"""Frozen-kernel random-position ranks and independent two-sample experiments.

Gandy--Scott (2021), Algorithms 1--2 and Proposition 2.3. Here exact reference
draws permit the conditional fixed-data version directly. This experiment tests
the kernel on its declared target, not ordinary initialization or tuning.
"""
from __future__ import annotations
import time
import numpy as np
import tensorflow as tf

from ..designs import seed_for
from ..targets import ValidationTarget
from ..procedures import FrozenTransition
from ..references import analytic
from ..storage import write_json
from .statistics import randomized_rank,rank_uniform_test,two_sample_test,gaussian_energy_test


def run(design,root,deadline=None):
    if "sequential" in design.options:
        from .sequential import run_invariance
        return run_invariance(design, root, deadline, _run_single)
    return _run_single(design, root, deadline)


def _run_single(design,root,deadline=None):
    def check_deadline():
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError("invariance experiment deadline exhausted")
    check_deadline()
    target=ValidationTarget(design.scenario.target,design.scenario.parameters,
        control=design.scenario.control if design.scenario.control in {"wrong_score","omit_jacobian"} else "baseline",
        jit_compile=design.device=="gpu")
    n,m=design.replications,design.rank_draws
    if n<2: raise ValueError("invariance requires two or more independent replications")
    anchor=analytic.draw(target.target_id,n,seed_for(design.seed,design.design_id,"anchor"),target.parameters)
    direct=analytic.draw(target.target_id,n,seed_for(design.seed,design.design_id,"direct"),target.parameters)
    insertion_rng=np.random.default_rng(seed_for(design.seed,design.design_id,"insertion"))
    insert=insertion_rng.integers(m+1,size=n)
    step=FrozenTransition(target,chains=n,step_size=design.step_size,leapfrog_steps=design.leapfrog_steps,
        control=design.scenario.control,jit_compile=design.device=="gpu")
    kernel_power = design.options.get("kernel_power", 1)
    transition = step.powered_step(kernel_power, with_health=True)
    arms=[]
    for arm in ("left","right"):
        q=tf.constant(anchor,tf.float64); history=[q.numpy()]
        for j in range(m):
            check_deadline()
            q,_,healthy=transition(q,tf.constant(seed_for(design.seed,design.design_id,arm,j),tf.int32))
            if not bool(healthy.numpy()):
                raise FloatingPointError(f"nonfinite invariance evidence in {arm} arm, step {j}")
            history.append(q.numpy())
        arms.append(history)
    check_deadline()
    paths=assemble_paths(arms,insert)
    quantities=analytic.test_quantities(target.target_id,paths,target.parameters)
    selected = design.options.get("invariance_quantities", tuple(quantities))
    quantities = {key: quantities[key] for key in selected}
    ranks={}
    rng=np.random.default_rng(seed_for(design.seed,design.design_id,"ties"))
    for key,values in quantities.items():
        ranks[key]=[randomized_rank(row[pos],np.delete(row,pos),rng) for row,pos in zip(values,insert)]
    energy_requested = design.options.get("gaussian_energy_test", False)
    family=max(design.multiplicity,2*len(ranks) + int(energy_requested))
    tests={key:rank_uniform_test(values,m,null_draws=design.null_draws,
        seed=seed_for(design.seed,design.design_id,"null",key),alpha=design.alpha,multiplicity=family)
        for key,values in ranks.items()}
    a=analytic.test_quantities(target.target_id,arms[1][-1],target.parameters)
    b=analytic.test_quantities(target.target_id,direct,target.parameters)
    two={key:two_sample_test(a[key],b[key],seed=seed_for(design.seed,design.design_id,"two",key),
        permutations=design.null_draws,alpha=design.alpha,multiplicity=family) for key in selected}
    energy = ({"gaussian_energy": gaussian_energy_test(arms[1][-1],
        scale=target.parameters.get("scale", 1.), alpha=design.alpha, multiplicity=family)}
        if energy_requested else {})
    if 1/(design.null_draws+1)>design.alpha/family:
        raise ValueError("rank/two-sample multiplicity exceeds null simulation resolution")
    check_deadline()
    moved=float(np.mean(np.any(paths!=anchor[:,None,:],axis=-1)))
    result={"rank_tests":tests,"two_sample_tests":two,"ranks":ranks,"insertion_positions":insert.tolist(),
        "two_sample_observations":{"left":{k:v.tolist() for k,v in a.items()},
                                   "right":{k:v.tolist() for k,v in b.items()}},
        "replications":n,"rank_draws":m,"fraction_moved":moved,"kernel_control":design.scenario.control,
        "trace_count":step.step.experimental_get_tracing_count(),"multiplicity":family,
        "kernel_power":kernel_power,"powered_trace_count":transition.experimental_get_tracing_count(),
        "native_transitions_per_arm":m*kernel_power,
        "all_substep_states_and_log_ratios_finite":True,
        "kernel_power_scope":"fixed-look reversible-kernel experiment; not independent posterior draws",
        "recurrence_fraction":float(np.mean(np.all(paths[:,2:]==paths[:,:-2],axis=-1))) if m>1 else None,
        "mutation_activation":design.scenario.control,
        "finding":"discrepancy_detected" if any(t["finding"]=="discrepancy_detected" for t in [*tests.values(),*two.values(),*energy.values()]) else "no_discrepancy_detected",
        "accuracy_established":False,"mixing_established":False,
        "null":"fixed-target conditional Gandy-Scott rank construction; independent exact reference anchors"}
    if energy_requested:
        result["analytic_tests"] = energy
    if "invariance_quantities" in design.options:
        result["tested_quantities"] = list(selected)
    write_json(root/"invariance.json",result)
    return result


def assemble_paths(arms, insertion):
    """Reversible K is also the reverse kernel (Gandy--Scott Algorithm 2).

    Each arm starts at the same exact anchor, using independent randomness.
    Left-arm draws run away from the anchor and are placed in reverse order.
    No inverse numerical integrator or adaptation is involved.
    """
    left,right=map(np.asarray,arms)  # [distance from anchor, replication, parameter]
    if left.shape!=right.shape or left.ndim!=3 or not np.array_equal(left[0],right[0]):
        raise ValueError("two equally shaped arms with identical anchors required")
    n=left.shape[1]; length=left.shape[0]
    if len(insertion)!=n or any(type(int(p)) is not int or p!=int(p) or not 0<=p<length for p in insertion):
        raise ValueError("one insertion position per replication required")
    return np.stack([np.stack([left[pos-j,r] if j<pos else right[j-pos,r]
                    for j in range(length)]) for r,pos in enumerate(insertion)])
