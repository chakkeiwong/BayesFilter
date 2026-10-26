"""Actual-consumer CPU reference diagnostic; no sampler or admission authority.

External fixture validation may use NumPy; value/score arithmetic is TensorFlow.
The shared BayesFilter checkout is the consumer's required runtime authority.
Non-XLA execution is an explicit diagnostic exception for an unqualified target.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

BASE = Path(__file__).resolve().parents[1]
CONSUMER = Path(os.environ.get('BAYESFILTER_DIAGNOSTIC_CONSUMER_ROOT', '/home/ubuntu/python/MacroFinance')).resolve()
PLAN = 'docs/plans/bayesfilter-modern-hmc-residual-qualification-2026-10-24.md'
RESULT = 'docs/plans/bayesfilter-modern-hmc-residual-qualification-result-2026-10-24.md'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, payload):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    report = dict(status='running', checks={}, timings={}, plan=PLAN, result=RESULT,
        gpu_intentionally_hidden=True, jit_compile=False,
        non_xla_reason='actual consumer explicitly declares XLA readiness false; bounded reference diagnostic',
        posterior_qualification=False, default_promoted=False, hmc_transitions=0)
    inputs = json.loads(args.inputs.read_text())
    paths = list((BASE/'bayesfilter').rglob('*.py'))
    paths += [CONSUMER/name for name in ('one_country_affine_macro_phase9_hmc.py',
        'one_country_affine_macro_governance.py', 'one_country_affine_macro_transition_2x2.py',
        'one_country_affine_macro_prior_measurement_contract.py')]
    paths += [Path(__file__).resolve(), args.inputs.resolve()]
    before = {str(p): digest(p) for p in paths}
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(), command=sys.argv,
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=BASE,text=True).strip(),
        source_sha256=before, source_scope='actual shared runtime including dirty-source hashes; consumer diagnostic dependencies',
        environment={k:os.environ.get(k) for k in ('CUDA_VISIBLE_DEVICES','TF_FORCE_GPU_ALLOW_GROWTH',
            'TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS')},
        python=sys.executable, data_version='retained P18 synthetic 48-period input',
        seeds='N/A: fixed saved input; no random sampling', plan=PLAN, result=RESULT)
    write(args.output/'manifest.json', manifest)
    def mark(stage):
        write(args.output/'progress.json', dict(stage=stage,elapsed_seconds=time.monotonic()-start))
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '-1' or os.environ.get('TF_FORCE_GPU_ALLOW_GROWTH') != 'true':
            raise ValueError('explicit CPU hiding and memory-growth environment required before import')
        sys.path.insert(0,str(CONSUMER));sys.path.insert(0,str(BASE))
        mark('import')
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory_policy = configure_tensorflow_gpu_memory_growth(tf,require_gpu=False)
        import one_country_affine_macro_phase9_hmc as target
        from bayesfilter.linear.kalman_tf import tf_kalman_log_likelihood, tf_kalman_filter
        from bayesfilter.linear.kalman_qr_derivatives_tf import tf_qr_sqrt_kalman_score_factors_batched_static
        from bayesfilter.inference.posterior_adapter import require_exact_score_adapter
        report['memory_policy']=memory_policy
        report['tensorflow_version']=tf.__version__
        report['timings']['imports']=time.monotonic()-start
        mark('build_actual_adapter')
        adapter=target._build_generic_target_with_runtime(inputs['observations'],
            dataset_id=inputs['design']['dataset_id'],
            runtime=(tf,tf_kalman_log_likelihood,tf_qr_sqrt_kalman_score_factors_batched_static,memory_policy),
            diagnostic_admission={'role':'source_bound_reference_diagnostic_only','plan':PLAN})
        require_exact_score_adapter(adapter)
        report['adapter_signature']=adapter.adapter_signature()
        report['capability']=adapter.value_score_capability().__dict__
        center=tf.constant(inputs['initial_position'],tf.float64)
        points=tf.stack((center,center+tf.constant(.01,tf.float64)))
        graph=tf.function(adapter.log_prob_and_grad, input_signature=[tf.TensorSpec([2,9],tf.float64)],
            autograph=False,jit_compile=False)
        mark('graph_compile_and_evaluate')
        tick=time.monotonic();gv,gs=graph(points)
        report['timings']['first_graph_evaluation']=time.monotonic()-tick
        tick=time.monotonic();ev,es=adapter.log_prob_and_grad(points)
        report['timings']['eager_reference']=time.monotonic()-tick
        report['checks']['finite_target']=bool(tf.reduce_all(tf.math.is_finite(gv)) & tf.reduce_all(tf.math.is_finite(gs)))
        report['graph_eager_value_error']=float(tf.reduce_max(tf.abs(gv-ev)))
        report['graph_eager_score_error']=float(tf.reduce_max(tf.abs(gs-es)))
        report['checks']['graph_value_parity']=report['graph_eager_value_error']<=1e-8
        report['checks']['graph_score_parity']=report['graph_eager_score_error']<=2e-5
        if not all(report['checks'].values()):raise ValueError('initial target/graph checks failed')
        one=tf.function(adapter.log_prob_and_grad,input_signature=[tf.TensorSpec([1,9],tf.float64)],autograph=False,jit_compile=False)
        singleton=[one(points[i:i+1]) for i in range(2)]
        report['batch_row_value_error']=float(tf.reduce_max(tf.abs(gv-tf.concat([v for v,s in singleton],0))))
        report['batch_row_score_error']=float(tf.reduce_max(tf.abs(gs-tf.concat([s for v,s in singleton],0))))
        report['checks']['batch_row_parity']=report['batch_row_value_error']<=1e-8 and report['batch_row_score_error']<=2e-5
        mark('central_difference_score_check')
        steps=tf.constant(sys.float_info.epsilon**(1/3),tf.float64)*tf.maximum(1.,tf.abs(points))
        rows=[];tick=time.monotonic()
        for j in range(9):
            estimates=[]
            for factor in (1.,.5):
                delta=steps[:,j:j+1]*factor*tf.one_hot(j,9,dtype=tf.float64)[None,:]
                vp,_=graph(points+delta);vm,_=graph(points-delta)
                estimates.append((vp-vm)/(2*factor*steps[:,j]))
            for i in range(2):
                exact=float(gs[i,j]);full=float(estimates[0][i]);half=float(estimates[1][i])
                rows.append(dict(point=i,parameter=target.PARAMETER_NAMES[j],analytic=exact,
                    step=float(steps[i,j]),full_step=full,half_step=half,
                    full_error=abs(full-exact),half_error=abs(half-exact),step_difference=abs(full-half)))
        report['finite_difference']=rows
        report['timings']['finite_difference']=time.monotonic()-tick
        report['checks']['score_fd_both_steps']=all(max(x['full_error'],x['half_error'])<=2e-5 for x in rows)
        obs=tf.constant(inputs['observations'],tf.float64)
        def reference_kalman(**kwargs):
            # Diagnostic-only trace of the identical function body. This keeps
            # the scalar chart's Eig and backward TensorLists outside XLA.
            return tf_kalman_filter.python_function(**kwargs,return_filtered=False)[0]
        @tf.function(input_signature=[tf.TensorSpec([9],tf.float64)],autograph=False,jit_compile=False)
        def reference(q):
            with tf.GradientTape() as tape:
                tape.watch(q)
                value=target._tf_likelihood_single(q,obs,tf,reference_kalman)
                value-=.5*tf.reduce_sum(tf.square(q/tf.constant(.35,tf.float64)))
            return value,tape.gradient(value,q)
        mark('independent_value_and_reverse_score')
        tick=time.monotonic();rv=[];rs=[]
        for i in range(2):
            v,s=reference(points[i]);rv.append(v);rs.append(s)
        report['reference_value_error']=float(tf.reduce_max(tf.abs(gv-tf.stack(rv))))
        report['reference_score_error']=float(tf.reduce_max(tf.abs(gs-tf.stack(rs))))
        report['timings']['independent_reference']=time.monotonic()-tick
        report['independent_reference_jit_compile']=False
        report['independent_reference_backend']='same tf_kalman_filter.python_function body traced without XLA; diagnostic only'
        report['checks']['independent_value_parity']=report['reference_value_error']<=1e-8
        report['checks']['independent_score_parity']=report['reference_score_error']<=2e-5
        definition=graph.get_concrete_function().graph.as_graph_def()
        nodes=list(definition.node)+[n for f in definition.library.function for n in f.node_def]
        report['host_callback_nodes']=[n.name for n in nodes if n.op in ('PyFunc','PyFuncStateless','EagerPyFunc')]
        report['checks']['no_host_callbacks']=not report['host_callback_nodes']
        report['checks']['bounded_trace_count']=graph.experimental_get_tracing_count()==1
        report['values']=[float(v) for v in gv]
        report['scores']=[[float(v) for v in row] for row in gs]
        imports={name:str(Path(m.__file__).resolve()) for name,m in list(sys.modules.items())
            if name.startswith('bayesfilter') and getattr(m,'__file__',None)}
        report['checks']['canonical_imports']=all(Path(p).is_relative_to(BASE/'bayesfilter') for p in imports.values())
        report['runtime_imports']=imports
        report['forbidden_local_imports']=[name for name in sys.modules if name=='filters' or name.startswith('filters.')
            or name.startswith(('inference.hmc','inference.mass_matrix','inference.posterior_adapter'))]
        report['checks']['no_local_filter_hmc']=not report['forbidden_local_imports']
        report['checks']['sources_unchanged']=all(digest(p)==h for p,h in before.items())
        report['status']='passed' if all(report['checks'].values()) else 'failed_checks'
    except Exception as exc:
        report['status']='error';report['error']=repr(exc);report['traceback']=traceback.format_exc()
    report['elapsed_seconds']=time.monotonic()-start
    report['finished_utc']=datetime.now(timezone.utc).isoformat()
    write(args.output/'result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ('status','checks','timings','error','elapsed_seconds')},allow_nan=False))
    return 0 if report['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
