"""Independent checks for the diagnostic-only Zhao--Cui source bridge."""
import math
from pathlib import Path
import zipfile

import pytest

from bayesfilter.testing.zhao_cui_reference_utils import (
    alignment_check, derive_full_sol_reference, extract_mlx_code,
    extract_mlx_tree, sha256_file, summarize_log_weights,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party/audit/zhao_cui_tensor_ssm_p10/source"


def test_logmeanexp_is_log_of_mean_weights_not_mean_of_logs():
    result = summarize_log_weights([0.0, math.log(9.0)])
    assert result.valid
    assert result.corrected_logmeanexp == pytest.approx(math.log(5.0))
    assert result.legacy_mean_log_weight == pytest.approx(math.log(3.0))
    assert result.importance_ess == pytest.approx(100.0 / 82.0)


def test_large_logs_remain_finite_and_translation_equivariant():
    result = summarize_log_weights([10000.0, 10000.0])
    assert result.valid
    assert result.corrected_logmeanexp == pytest.approx(10000.0)
    assert result.importance_ess == pytest.approx(2.0)


@pytest.mark.parametrize("invalid", [math.nan, math.inf, -math.inf])
def test_nonfinite_is_retained_as_a_veto_not_silently_removed(invalid):
    result = summarize_log_weights([0.0, invalid])
    assert not result.valid
    assert result.corrected_logmeanexp is None
    assert result.finite_fraction == 0.5


def test_extracts_only_code_in_cell_order(tmp_path):
    source = tmp_path / "setup.mlx"
    xml = """<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body>
        <w:p><w:pPr><w:pStyle w:val="text"/></w:pPr><w:r><w:t>not code</w:t></w:r></w:p>
        <w:p><w:pPr><w:pStyle w:val="code"/></w:pPr><w:r><w:t><![CDATA[function x = setup()
x = 1 < 2;]]></w:t></w:r></w:p>
        <w:p><w:pStyle w:val="code"/><w:r><w:t>end</w:t></w:r></w:p>
      </w:body></w:document>"""
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("matlab/document.xml", xml)
    assert extract_mlx_code(source) == "function x = setup()\nx = 1 < 2;\n\nend\n"


def test_empty_code_is_rejected(tmp_path):
    source = tmp_path / "empty.mlx"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("matlab/document.xml", "<document/>")
    with pytest.raises(ValueError, match="no nonempty code"):
        extract_mlx_code(source)


def test_missing_alignment_fields_cannot_pass():
    report = alignment_check({}, {}, ["initial_law", "observation_sha256"])
    assert report["status"] == "not_aligned"
    assert set(report["mismatches"]) == {"initial_law", "observation_sha256"}


def test_pinned_callbacks_and_derived_class_preserve_source(tmp_path):
    original = SOURCE / "models/full_sol.m"
    before = sha256_file(original)
    records = extract_mlx_tree(SOURCE, tmp_path)
    assert len(records) == 19
    assert all(sha256_file(Path(row["derived"])) == row["derived_sha256"] for row in records)
    generated = tmp_path / "models/full_sol_reference.m"
    record = derive_full_sol_reference(original, generated)
    assert record["source_sha256"] == before == sha256_file(original)
    code = generated.read_text()
    assert "function sol = full_sol_reference" in code
    assert "'raw_log_weight', raw_log_weight" in code
    assert "log(numel(raw_log_weight))" in code



def test_changed_source_density_call_cannot_silently_bypass_stable_logs(tmp_path):
    source = tmp_path / "changed_full_sol.m"
    source.write_text((SOURCE / "models/full_sol.m").read_text().replace(
        "log(priorpdf(sol.model, fulldata(1:d+m, :, 1)))",
        "log(changed_priorpdf(sol.model, fulldata(1:d+m, :, 1)))"))
    with pytest.raises(ValueError, match="source log-density call changed"):
        derive_full_sol_reference(source, tmp_path / "full_sol_reference.m")


def test_fixed_target_adapter_records_changes_without_touching_source(tmp_path):
    from bayesfilter.testing.zhao_cui_reference_utils import derive_fixed_target_callbacks
    original_hash = sha256_file(SOURCE / "models/pp/predator_step.mlx")
    extract_mlx_tree(SOURCE, tmp_path)
    records = derive_fixed_target_callbacks(tmp_path)
    assert len(records) == 4
    assert all(row["classification"] == "extension_or_invention_fixed_target" for row in records)
    assert all(row["before_sha256"] != row["after_sha256"] for row in records)
    assert sha256_file(SOURCE / "models/pp/predator_step.mlx") == original_hash
    # A second application must fail; it cannot silently treat an adapted source
    # as an unmodified author snapshot on another run.
    with pytest.raises(ValueError, match="fourth stage did not match"):
        derive_fixed_target_callbacks(tmp_path)
