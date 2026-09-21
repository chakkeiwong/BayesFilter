from dataclasses import replace

import pytest

from bayesfilter.testing.inference_validation.aggregation import aggregate_groups, validate_groups
from bayesfilter.testing.inference_validation.execution import run_suite
from bayesfilter.testing.inference_validation.storage import read_json, write_json


def grouped_suite(design):
    first=design("sbc","normal_conjugate","reference",replications=4,rank_draws=2)
    second=replace(first,design_id="shard-two")
    return {"schema":"bayesfilter.inference_validation_suite.v1","suite_id":"parallel-sbc",
            "profile":"statistical","profiles":{"statistical":["sbc"]},
            "designs":[first.payload(),second.payload()],
            "aggregate_groups":[{"group_id":"combined","design_ids":[first.design_id,second.design_id],
                                 "replications":8}]}


def test_group_definition_rejects_duplicated_and_changed_settings(design):
    suite=grouped_suite(design)
    validate_groups(suite)
    suite["designs"][1]["rank_draws"]=3
    with pytest.raises(ValueError,match="identical"):
        validate_groups(suite)
    suite=grouped_suite(design)
    suite["aggregate_groups"][0]["design_ids"]*=2
    with pytest.raises(ValueError,match="membership"):
        validate_groups(suite)


def test_parallel_sbc_shards_preserve_complete_and_missing_denominators(design,tmp_path):
    suite=grouped_suite(design)
    index=run_suite(suite,tmp_path/"run",max_workers=2)
    assert index["max_workers"]==2
    assert all(j["status"]=="complete" and len(j["attempts"])==1 for j in index["jobs"].values())
    result=read_json(tmp_path/"run"/"combined-aggregate.json")
    assert result["completed"]==result["planned"]==8
    assert len({(r["shard_id"],r["dataset_id"]) for r in result["datasets"]})==8
    index["jobs"][suite["designs"][1]["design_id"]]["status"]="failed"
    write_json(tmp_path/"run"/"run_index.json",index)
    result=aggregate_groups(tmp_path/"run")[0]
    assert result["completed"]==4 and result["planned"]==8
    assert result["finding"]=="calibration_incomplete" and result["conditional_tests_only"]
    assert result["missing_dataset_records"] == 4
    assert result["unstarted_datasets"] is None
    assert result["known_unstarted_datasets"] == 0
    assert result["missing_records_from_incomplete_shards"] == 4
    first=next(iter(index["jobs"].values()))
    from pathlib import Path
    result_path=Path(first["result"])
    manifest_path=result_path.with_name(result_path.name.replace("-result.json","-manifest.json"))
    manifest=read_json(manifest_path)
    manifest["source"]["identity"]="different-source"
    write_json(manifest_path,manifest)
    with pytest.raises(ValueError,match="source or design"):
        aggregate_groups(tmp_path/"run")
    result_path.write_text("{}")
    with pytest.raises(ValueError,match="checksum"):
        aggregate_groups(tmp_path/"run")
