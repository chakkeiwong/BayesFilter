"""Dedicated Class C non-harm check; zero cap is a diagnostic comparator only."""
import json,os
from pathlib import Path
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(True)
from bayesfilter.highdim.gaussian_cloud_designs_tf import standard_normal_cloud
from bayesfilter.highdim.ledh_unified_correction_tf import batched_higher_moment_shape_jvp
out=Path(os.environ['LEDH_PROGRAM_RESULT_DIR']);dtype=tf.float32;n=4096
cloud=standard_normal_cloud('scrambled_halton_gaussian',num_particles=n,dimension=1,
    seed=20260917,salt=301,dtype=tf.float64)
cloud=(cloud-tf.reduce_mean(cloud))/tf.math.reduce_std(cloud)
x=tf.cast(cloud,dtype)[None]
# Same known-good source/points: a numerics-altering protection must preserve
# these already-correct coordinates within the declared engineering budget.
stress=tf.tensor_scatter_nd_update(x,[[0,0,0]],[tf.constant(1e4,dtype)])
weights=tf.ones([1,n],dtype)/n;dx=tf.zeros([1,1,n,1],dtype);dw=tf.zeros([1,1,n],dtype)
records=[]
def build(cap):
    @tf.function(input_signature=[tf.TensorSpec([1,n,1],dtype)],jit_compile=True)
    def run(points):
        return batched_higher_moment_shape_jvp(x,weights,dx,dw,points,dx,
            correction_steps=2,strength=.2,diagonal_lm_damping=.001,
            diagonal_lm_scale_floor=1e-6,diagonal_trust_radius=.1,
            pairwise_correction_steps=2,pairwise_strength=.02,
            pairwise_particle_rms_cap=2.,coordinatewise_standardized_cap=cap,
            coordinatewise_standardized_cap_power=8)
    return run
reference=build(0.)(x)
assert bool(reference['value_valid'][0])
base=reference['particles']
def moments(points):
    p=tf.cast(points,tf.float64);p=(p-tf.reduce_mean(p))/tf.math.reduce_std(p)
    return dict(skew=float(tf.reduce_mean(p**3)),kurtosis=float(tf.reduce_mean(p**4)))
for cap in (.98,2.,4.,8.,16.,32.):
    run=build(cap);healthy=run(x);bounded=run(stress)
    maximum_error=float(tf.reduce_max(tf.abs(healthy['particles']-base)))
    valid=bool(healthy['value_valid'][0]) and bool(healthy['valid'][0])
    stress_finite=bool(bounded['value_valid'][0]) and bool(tf.reduce_all(tf.math.is_finite(bounded['particles'])))
    displacement=float(bounded['mean_coordinatewise_cap_displacement'][0])
    records.append(dict(cap=cap,healthy_maximum_standardized_displacement=maximum_error,
        healthy_valid=valid,healthy_moments=moments(healthy['particles']),
        stress_finite=stress_finite,stress_cap_displacement=displacement,
        stress_maximum_pre_cap=float(bounded['maximum_coordinatewise_pre_cap_absolute'][0]),
        stress_maximum_post_cap=float(bounded['maximum_coordinatewise_post_cap_absolute'][0]),
        nonharm_pass=valid and maximum_error<=1e-4 and stress_finite and displacement>0))
    print(json.dumps(records[-1]),flush=True)
eligible=[r['cap'] for r in records if r['nonharm_pass']]
report=dict(classification='class_C_nonharm_calibration_not_filter_admission',
    memory_policy=memory,reference='zero cap diagnostic, same full correction',
    healthy_budget=1e-4,baseline_moments=moments(base),records=records,
    selected=min(eligible) if eligible else None)
(out/'cap-calibration.json').write_text(json.dumps(report,indent=2)+'\n')
assert eligible,'No positive coordinate cap satisfies the declared non-harm screen'
