from pathlib import Path

import pytest

from bayesfilter.testing.inference_validation import execution
from bayesfilter.testing.inference_validation.designs import digest
from bayesfilter.testing.inference_validation.storage import read_json, write_json
from scripts.resume_inference_validation_job import resume_job


def test_resource_retry_preserves_design_prior_cost_and_attempt(design,tmp_path,monkeypatch):
    d=design("sbc","normal_conjugate","reference",replications=2)
    suite={"schema":"bayesfilter.inference_validation_suite.v1","suite_id":"retry",
           "profile":"test","profiles":{"test":["sbc"]},"designs":[d.payload()]}
    root=tmp_path/"suite"
    index=execution.run_suite(suite,root)
    original=index["jobs"][d.design_id]
    original_result=read_json(original["result"])
    index["jobs"][d.design_id]["status"]="timed_out"
    write_json(root/"run_index.json",index)
    def complete(design,command,log,seconds,attempt):
        assert design.identity==d.identity and attempt==2 and seconds==7.
        target=Path(original["result"]).with_name("attempt-002-result.json")
        write_json(target,original_result)
        manifest=read_json(target.with_name("attempt-001-manifest.json"))
        write_json(target.with_name("attempt-002-manifest.json"),manifest)
        return {"attempt":attempt,"elapsed_seconds":2.,"exit_code":0,"status":"complete","command":command}
    monkeypatch.setattr(execution,"_execute_job",complete)
    resume_job(root,d.design_id,seconds=7.,reason="finish interrupted output")
    after=read_json(root/"run_index.json")
    assert len(after["jobs"][d.design_id]["attempts"])==2
    assert after["jobs"][d.design_id]["attempts"][0]==original["attempts"][0]
    assert digest(after["plan"])==digest(index["plan"])
    assert len(after["resource_extensions"])==1
    with pytest.raises(ValueError,match="ended unsuccessful"):
        resume_job(root,d.design_id,seconds=7.,reason="not allowed")
    after["source"]["identity"]="changed"
    write_json(root/"run_index.json",after)
    with pytest.raises(ValueError,match="unchanged frozen source"):
        resume_job(root,d.design_id,seconds=7.,reason="not allowed")
