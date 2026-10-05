"""Structural HLO reader checks, independent of TensorFlow execution."""

import pytest

from scripts.analyze_filter_repair_locator_optimized_hlo import comparison, computations

SOURCE = '''%f (x: f64[2]) -> f64[2] {
  %x = f64[2]{0} parameter(0)
  %c = f64[2]{0} constant({2, 3})
  ROOT %y = f64[2]{0} subtract(f64[2]{0} %x, f64[2]{0} %c), metadata={op_type="Sub"}
}
'''
CALLER = '''ENTRY %main (a: f64[2]) -> f64[2] {
  %a = f64[2]{0} parameter(0)
  ROOT %answer = f64[2]{0} call(f64[2]{0} %a), to_apply=%f
}
'''


def parse(text):
    return computations(text.splitlines())


def test_renaming_and_source_metadata_do_not_change_content():
    changed = (SOURCE + CALLER).replace('%x', '%foo').replace('%c', '%bar').replace('%y', '%baz')
    changed = changed.replace('op_type="Sub"', 'op_type="Diagnostic" source_file="elsewhere.py"')
    left, right = parse(SOURCE + CALLER), parse(changed)
    assert comparison(left, right)['matched_by_content_and_multiplicity'] == 2


@pytest.mark.parametrize('before,after', [
    ('constant({2, 3})', 'constant({2, 4})'),
    ('subtract(f64[2]{0} %x, f64[2]{0} %c)', 'subtract(f64[2]{0} %c, f64[2]{0} %x)'),
    ('f64', 'f32'),
    ('{0}', '{0:T(2)}'),
    ('parameter(0)', 'parameter(1)'),
    ('subtract(', 'add('),
])
def test_numerical_shape_layout_or_operand_change_changes_caller(before, after):
    left = parse(SOURCE + CALLER)
    right = parse(SOURCE.replace(before, after) + CALLER)
    assert all(a['normalized_sha256'] != b['normalized_sha256'] for a, b in zip(left, right, strict=True))


def test_declaration_order_and_duplicate_computations():
    other = SOURCE.replace('%f ', '%g ').replace('constant({2, 3})', 'constant({4, 5})')
    left, right = parse(SOURCE + other), parse(other + SOURCE)
    assert comparison(left, right)['matched_by_content_and_multiplicity'] == 2
    duplicate = SOURCE.replace('%f ', '%g ')
    result = comparison(parse(SOURCE + duplicate), parse(SOURCE))
    assert result['matched_by_content_and_multiplicity'] == 1
    assert len(result['left_unmatched']) == 1


def test_elided_literals_are_explicitly_incomplete():
    row = parse(SOURCE.replace('constant({2, 3})', 'constant({...})'))[0]
    assert row['opaque_constant_lines'] == [3]
    assert not row['constants_fully_printed']


@pytest.mark.parametrize('text', [SOURCE.replace('%x,', '%missing,'),
    SOURCE.replace('  %x =', '  unsupported %x ='), SOURCE.removesuffix('}\n'),
    SOURCE.replace('ROOT ', ''), CALLER + SOURCE])
def test_unsupported_or_incomplete_ir_fails(text):
    with pytest.raises(AssertionError):
        parse(text)
