"""CPU-only harness checks: independent data and frozen campaign provenance."""
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['TF_NUM_INTRAOP_THREADS']='2'
os.environ['TF_NUM_INTEROP_THREADS']='2'
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import tensorflow as tf
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'docs/benchmarks'))
import run_ledh_matched_comparison as matched
import run_ledh_independent_validation as master


def test_checked_data_are_used_verbatim_and_reject_corruption(tmp_path):
    obs=tf.reshape(tf.range(100,dtype=tf.float64),(50,2))
    spec=SimpleNamespace(target_id='fixture',default_theta=lambda dtype:tf.constant([0.],dtype))
    data=dict(target_id='fixture',data_seed=26100831,dtype='float64',observations=obs.numpy().tolist(),
              observation_sha256=matched.tensor_sha(tf,obs))
    path=tmp_path/'dataset.json';path.write_text(json.dumps(data))
    _,actual,_,reference=matched.data_and_reference(tf,spec,'predator_prey',path,26100831)
    tf.debugging.assert_equal(actual,obs)
    assert reference['value'] is None
    with pytest.raises(ValueError,match='scope mismatch'):
        matched.data_and_reference(tf,spec,'predator_prey',path,26100832)
    data['observations'][0][0]=99.
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='corrupted'):
        matched.data_and_reference(tf,spec,'predator_prey',path,26100831)


def test_frozen_input_identity_rejects_changed_tuning_or_data(tmp_path,monkeypatch):
    source=tmp_path/'original.json';source.write_text('{"beta":[0.1,0.9,0.0]}')
    snap=tmp_path/'snapshot.json';snap.write_bytes(source.read_bytes())
    folder=tmp_path/'inputs';folder.mkdir();data=folder/'dataset.json';data.write_text('{}')
    master.write(tmp_path/'campaign.json',dict(frozen_tuning={'fixture':dict(source=str(source),
        snapshot=str(snap),sha256=master.sha(source))},data=[dict(path=str(folder),sha256=master.sha(data))]))
    monkeypatch.setattr(master,'OUT',tmp_path)
    master.validate_frozen()
    snap.write_text('{"beta":[1.0,0.0,0.0]}')
    with pytest.raises(ValueError,match='tuning changed'):master.validate_frozen()
    snap.write_bytes(source.read_bytes());data.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='dataset changed'):master.validate_frozen()


def test_paired_designs_preserve_common_inputs():
    spec=SimpleNamespace(dimension=2)
    old,new=matched.fixed_inputs(tf,spec,261008301,n=8,h=2)
    assert old[0] is new[0] and old[1] is new[1]
    assert matched.tensor_sha(tf,old[0])==matched.tensor_sha(tf,new[0])
    assert matched.tensor_sha(tf,old[1])==matched.tensor_sha(tf,new[1])
    assert new[2].shape==(2,8,2)
    assert set(master.design_seeds(26100831)).isdisjoint(master.design_seeds(26100832))
    assert set(master.DATA_SEEDS).isdisjoint({26100611,26100820,26100821})
