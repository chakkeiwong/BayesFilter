"""Diagnostic consumer of the supported fixed-HMC public tuning procedure.

Every verified candidate is preserved and assessed separately. No posterior
draw is used to tune or pick an efficiency winner. All numerical transitions
use the repository's XLA implementation; scheduling/reporting remain on host.
"""
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback


class TargetAdapter:
    def __init__(self,tf,target):
        self.tf,self.target=tf,target
        self.parameter_dim=target.dimension
        self.batch_rank_policy='rank2_required'
        def score(q):
            with tf.GradientTape() as tape:
                tape.watch(q);value=target.log_prob(q)
            return value,tape.gradient(value,q)
        self._score=tf.function(score,input_signature=[tf.TensorSpec([None,target.dimension],tf.float64)],
                                autograph=False,jit_compile=True)

    def adapter_signature(self):
        path=Path(__file__).with_name('nuts_benchmark_targets.py')
        return hashlib.sha256(json.dumps(dict(target=self.target.name,dimension=self.parameter_dim,
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()),sort_keys=True).encode()).hexdigest()

    def value_score_capability(self):
        from bayesfilter.inference import ValueScoreCapability
        return ValueScoreCapability(value_score_authority='graph_native',xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,target_scope='stan-port-benchmark',runtime_backend='tensorflow',
            evidence_path=__file__,nonclaims=('Standalone analytic diagnostic; no production qualification',))

    def log_prob_and_grad(self,q):
        tf=self.tf;q=tf.convert_to_tensor(q,tf.float64)
        if q.shape.rank==1:
            v,g=self._score(q[None]);return v[0],g[0]
        return self._score(q)

    def target_status_telemetry(self,q):
        tf=self.tf;shape=tf.shape(q)[:-1]
        return dict(status_code=tf.zeros(shape,tf.int32),valid_pre_regularized_score=tf.ones(shape,tf.bool),
                    floor_count_value=tf.zeros(shape,tf.int32))


def run_fixed(tf,target,starts,seed,args,folder):
    from bayesfilter.inference import (HMCKernelTuningConfig,HMCCandidateExecutionConfig,HMCAcceptancePolicy,
        tune_hmc_kernel,export_hmc_candidate_retained_runners,load_hmc_candidate_retained_runners)
    from bayesfilter.inference.hmc_candidate_set_execution import _trace_from_payload
    import benchmark_stan_transplant as common
    began=time.perf_counter();adapter=TargetAdapter(tf,target)
    # Preserve the dirty workspace's actual numerical dependency sources before
    # preparation. Git SHA and a later binding hash alone cannot reconstruct it.
    from bayesfilter.inference.hmc_candidate_set_execution import _source_closure
    closure=_source_closure(adapter,[__file__,str(Path(__file__).with_name('nuts_benchmark_targets.py'))])
    source_root=folder/'source';source_root.mkdir()
    root=Path(__file__).resolve().parents[2]
    for name,digest in closure.items():
        source=Path(name);dest=source_root/source.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        if hashlib.sha256(dest.read_bytes()).hexdigest()!=digest:raise RuntimeError('Source changed during snapshot')
    common.write(folder/'source-hashes.json',closure)
    # Defaults here are the public standard protocol, not a replacement grid.
    cfg=HMCKernelTuningConfig.standard(use_xla=True,target_scope='stan-port-benchmark',seed=(seed,51),
                                       public_timeout_budget_s=180.)
    policy=HMCAcceptancePolicy()
    execution=HMCCandidateExecutionConfig(measurement_num_results=policy.min_decisions_per_chain,
        verification_num_results=policy.min_decisions_per_chain,num_warmup_steps=0,seed=(seed,51),
        acceptance_policy=policy,target_status_trace_policy='none',use_xla=True,
        chain_mode='batched',reuse_leapfrog_graphs=True)
    report=dict(protocol='tune_hmc_kernel, standard operational windowed mass preparation and shared candidate-set search',
        configuration=cfg.payload(),execution=execution.payload(),candidate_results=[],
        representative_selection=None,no_efficiency_winner=True)
    common.write(folder/'protocol.json',report)
    try:
        run=tune_hmc_kernel(adapter=adapter,initial_position=starts[0],config=cfg,execution_config=execution,
            target_lineage=dict(model=target.name,data='analytic regression design seed 20261027,61 when applicable',
                                prior='encoded in target',starts='automatic public preparation starts from common first chain'),
            source_paths=[__file__,str(Path(__file__).with_name('nuts_benchmark_targets.py'))],output_dir=folder/'tuning')
        report['tuning_seconds']=time.perf_counter()-began
        report['completion_status']=run.result.completion_status
        report['verified_candidate_ids']=list(run.result.verified_candidate_ids)
        report['candidate_states']=dict(run.result.candidate_states)
        binding=run.adapter._execution_binding
        t=time.perf_counter()
        members=export_hmc_candidate_retained_runners(candidate_set_result=run.result,
            retained_binding=binding,output_dir=folder/'members')
        runners=load_hmc_candidate_retained_runners(list(members.values()),adapter=adapter)
        report['shared_member_export_reload_seconds']=time.perf_counter()-t
        report['common_preparation_search_export_seconds']=time.perf_counter()-began
        for index,cid in enumerate(run.result.verified_candidate_ids):
            runner=runners[cid]
            out=folder/f'retained-{index}'
            t=time.perf_counter()
            chunk=runner.run(num_results=args.draws,seed=(seed,700+index),output_dir=out)
            elapsed=time.perf_counter()-t
            archive=json.loads(Path(chunk['archive_path']).read_text())
            trace=_trace_from_payload(archive['trace']);draws=chunk['position_samples']
            accepts=tf.exp(tf.minimum(trace['log_accept_ratio'],0.))
            stats=tf.stack((accepts,tf.cast(trace['is_accepted'],tf.float64),
                tf.fill(accepts.shape,tf.cast(runner.candidate.leapfrog_steps,tf.float64)),tf.zeros_like(accepts),
                tf.cast(~tf.math.is_finite(trace['log_accept_ratio']),tf.float64),tf.ones_like(accepts)),-1)
            with tf.device('/CPU:0'):
                diagnostics=common.assess(tf,target,tf.identity(draws),tf.identity(stats))
            row=dict(candidate_id=cid,epsilon=runner.candidate.epsilon,leapfrog_steps=runner.candidate.leapfrog_steps,
                retained_seconds_including_archive_and_checks=elapsed,
                tuned_total_seconds=report['common_preparation_search_export_seconds']+elapsed,
                timing_scope='entire common preparation/search and all-member export/reload plus this member retention; common posterior diagnostics separate',
                diagnostics=diagnostics,archive=chunk['archive_path'])
            report['candidate_results'].append(row)
            common.write(folder/'progress.json',report)
        report['status']='completed_with_candidates' if report['verified_candidate_ids'] else 'no_verified_candidate'
    except Exception as exc:
        report['status']='preparation_or_tuning_failed'
        report['error']=str(exc);report['traceback']=traceback.format_exc()
        # An exception is preserved and is a repair trigger, never a fast winner.
    report['tuned_total_seconds']=time.perf_counter()-began
    report['timing_scope']='entire standard preparation/search plus ALL verified members; per-member totals reported separately'
    return None,None,report
