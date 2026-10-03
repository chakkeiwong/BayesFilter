"""A saved report cannot keep claiming current source after the checkout changes."""
import importlib.util
import json
from pathlib import Path


def test_renderer_rechecks_source_instead_of_trusting_saved_current_label(tmp_path):
    script=Path(__file__).resolve().parents[2]/"scripts/render_inference_validation_coverage.py"
    spec=importlib.util.spec_from_file_location("coverage_renderer",script)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report=tmp_path/"report.json"
    report.write_text(json.dumps({"schema":"bayesfilter.inference_validation_report.v1","rows":[{
        "control":"baseline","target":"gaussian","route":"ordinary","device":"cpu_reference",
        "engine":"accuracy","execution_status":"complete","source_status":"matches_current_source",
        "assessment_complete":True,"finding":"pipeline_assessed"}]}))
    (tmp_path/"run_index.json").write_text(json.dumps({"source":{"identity":"older-source"}}))
    md,tex=tmp_path/"coverage.md",tmp_path/"coverage.tex"
    module.render([report],md,tex)
    assert "0 / 1 / 1 | 0" in md.read_text()
    assert "0 / 1 / 1" in tex.read_text()
