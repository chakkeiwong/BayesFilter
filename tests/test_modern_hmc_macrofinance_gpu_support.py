"""Source/evidence checks run without TensorFlow, CUDA or external consumers."""
import hashlib
import json
import pytest
from scripts.modern_hmc_macrofinance_gpu_support import verify_gpu_evidence


def fixture(tmp_path):
    base=tmp_path/'base';consumer=tmp_path/'consumer';out=tmp_path/'evidence'
    out.mkdir();sources={}
    for p in (base/'bayesfilter/linear/kalman_qr_derivatives_tf.py',
              base/'bayesfilter/ssm/target_builder.py',consumer/'one_country_affine_macro_phase9_hmc.py'):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text('test source')
        sources[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
    result=dict(status='completed',source_unchanged=True,chain_output_device='/device:GPU:0',checks={k:True for k in
        ('same_value_score','accepted_and_proposed_status','gpu_xla_complete_target','cpu_gpu_target_parity',
         'gpu_xla_shared_chain_body','gpu_xla_deterministic_replay','cpu_gpu_shared_chain_parity')})
    manifest=dict(source_sha256=sources,device='GPU',memory_policy=dict(
        configured_before_logical_device_initialization=True,all_physical_devices_memory_growth=True))
    (out/'result.json').write_text(json.dumps(result));(out/'manifest.json').write_text(json.dumps(manifest))
    return base,consumer,out,result,manifest


def test_saved_gpu_evidence_checked_before_capability(tmp_path):
    b,c,o,_,_=fixture(tmp_path)
    assert len(verify_gpu_evidence(o,base=b,consumer=c)['checked_sources'])==3


@pytest.mark.parametrize('failure',('missing_check','cpu','growth','source','missing_source'))
def test_incomplete_or_changed_evidence_refused(tmp_path,failure):
    b,c,o,r,m=fixture(tmp_path)
    if failure=='missing_check':r['checks'].pop('cpu_gpu_shared_chain_parity')
    if failure=='cpu':m['device']='CPU'
    if failure=='growth':m['memory_policy']['all_physical_devices_memory_growth']=False
    if failure=='source':(c/'one_country_affine_macro_phase9_hmc.py').write_text('changed')
    if failure=='missing_source':m['source_sha256'].pop(str(c/'one_country_affine_macro_phase9_hmc.py'))
    (o/'result.json').write_text(json.dumps(r));(o/'manifest.json').write_text(json.dumps(m))
    with pytest.raises(ValueError):verify_gpu_evidence(o,base=b,consumer=c)
