"""Independent-sequence inference mechanics; CPU diagnostic only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf
from docs.benchmarks import run_observation_tt_independent_filtering as run


def test_inference_pairs_sequences_and_keeps_sparse_regimes_inconclusive(monkeypatch):
    monkeypatch.setattr(run,'tf',tf,raising=False)
    monkeypatch.setattr(run,'D',tf.float64,raising=False)
    entries=[]
    for d in (1,4):
        for i in range(12):
            tables={}
            for method in (*run.HEURISTICS,'tt_sgqf_safeguard'):
                candidate=method=='tt_sgqf_safeguard'
                # Large common sequence effects should cancel in paired differences.
                mse=1000.+30*i+(-.001+i*.00001 if candidate else 0.)
                tables[method]={'regimes':{label:{'count':(0 if label=='near_zero' and i>=3 else 20),'mse':mse}
                    for label in ('all','near_zero','ordinary','large')}}
            entries.append(dict(dimension=d,sequence=i,reference_pass=True,metrics=tables))
    result=run.infer(entries)
    complete=next(c for c in result['contrasts'] if c['regime']=='all')
    assert abs(complete['mean_delta']-(-.000945))<1e-10
    assert complete['upper']<0 and complete['half_width']<.001
    sparse=next(c for c in result['contrasts'] if c['regime']=='near_zero')
    assert sparse['sequence_count']==3 and not sparse['coverage_pass']
    assert not result['candidate_advances']
