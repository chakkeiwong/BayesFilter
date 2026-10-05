"""Independent tiny IR checks for invariant tracing, including tuple comments."""

import pytest

from scripts.inspect_filter_repair_locator_fusions import (
    Body,
    instruction,
    invariant_inputs,
    tuple_index,
)

BODY = Body('%body (arg: (f64[], f64[])) -> (f64[], f64[]) {', [
    '  %arg = (f64[], f64[]) parameter(0)',
    '  %a = f64[] get-tuple-element((f64[], /*index=1*/f64[]) %arg), index=0',
    '  %b = f64[] get-tuple-element((f64[], f64[]) %arg), index=1',
    '  %v = f64[] fusion(f64[] %a, f64[] %a, f64[] %a, f64[] %a, f64[] %a, f64[] %b), kind=kLoop, calls=%selected',
    '  ROOT %result = (f64[], f64[]) tuple(f64[] %a, f64[] %b)',
])
ENTRY = [
    '  %c = f64[] constant(2.4)',
    '  %initial = (f64[], f64[]) tuple(f64[] %c, f64[] %c)',
    '  ROOT %loop = (f64[], f64[]) while((f64[], f64[]) %initial), condition=%cond, body=%body',
]


def test_index_comments_cannot_replace_actual_tuple_index():
    assert tuple_index(instruction(BODY[1])) == 0
    result = invariant_inputs(BODY, ENTRY, 'selected')
    assert result['initial_tuple_size'] == 2
    assert [item['tuple_index'] for item in result['invariants']] == [0, 1]
    assert all(item['initial_literal']['expression'] == 'f64[] constant(2.4)'
        for item in result['invariants'])


@pytest.mark.parametrize('case', ['changed_loop_value', 'opaque_literal', 'dynamic_initial'])
def test_unproven_invariants_fail(case):
    body, entry = Body(BODY.header, BODY), list(ENTRY)
    if case == 'changed_loop_value':
        body[-1] = body[-1].replace('%a, f64[] %b', '%b, f64[] %a')
    elif case == 'opaque_literal':
        entry[0] = entry[0].replace('2.4', '{...}')
    else:
        entry[0] = entry[0].replace('constant(2.4)', 'parameter(0)')
    with pytest.raises(AssertionError):
        invariant_inputs(body, entry, 'selected')
