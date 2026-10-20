"""Diagnostic extraction of the original fit suffix, with unchanged AST body.

Only its function signature is replaced to supply the already evaluated prefix.
The full original initializer remains the numerical authority in the tests.
"""

import ast
import hashlib
from functools import lru_cache

import numpy as np

from tests.test_filter_repair_geometry_control import source
from tests.test_filter_repair_geometry_fit import DIAGNOSTIC_FIELDS, TOP_FIELDS

PARAMETERS = ('value_and_score_fn', 'center_np', 'scale_np', 'q_basis', 'z_train', 'y_train',
    'score_train', 'z_holdout', 'y_holdout', 'center_value', 'center_score_z', 'cfg', 'dim', 'rank',
    'holdout_count', 'finite_sample_count', 'center_score_norm', 'incumbent', 'exact_evaluation_count',
    'exact_candidates', 'diagnostics')


@lru_cache(maxsize=1)
def original_suffix():
    checkpoint, module = source('3582b4ac')
    filename = 'bayesfilter/inference/quadratic_geometry.py'
    tree = ast.parse(checkpoint.sources[filename])
    original = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == 'fit_low_rank_spd_quadratic_geometry')
    index = next(index for index, node in enumerate(original.body)
        if isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'fit')
    body = original.body[index:]
    digest = hashlib.sha256(ast.dump(ast.Module(body=body, type_ignores=[]), include_attributes=False).encode()).hexdigest()
    function = ast.FunctionDef(name='frozen_geometry_fit_suffix', args=ast.arguments(
        posonlyargs=[], args=[ast.arg(arg=name) for name in PARAMETERS], kwonlyargs=[],
        kw_defaults=[], defaults=[]), body=body, decorator_list=[])
    extracted = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(vars(module))
    exec(compile(extracted, '3582b4ac:geometry_fit_suffix', 'exec'), namespace)  # noqa: S102 - exact original diagnostic AST.
    return namespace['frozen_geometry_fit_suffix'], digest, checkpoint


def prefix_context(target, cfg, args):
    """Build the same completed-prefix records outside both timed suffixes."""
    import dataclasses

    checkpoint, module = source('3582b4ac')
    records = checkpoint.load('bayesfilter.inference._exact_incumbent')
    center, scale, basis, z_train, y_train, score_train, z_holdout, y_holdout, center_value, center_score = (
        value.numpy() for value in args[:10])
    dimension, rank = basis.shape
    count = z_train.shape[0] + z_holdout.shape[0]
    original_cfg = module.LowRankSPDQuadraticGeometryConfig(**dataclasses.asdict(cfg))
    diagnostics = module._base_diagnostics(cfg=original_cfg, dim=dimension, rank=rank,
        regression_parameter_count=dimension + rank + 2,
        required_finite_samples=cfg.min_samples_per_parameter * (dimension + rank + 2),
        sample_count=count, center_value=float(center_value), center_score_norm=float(np.linalg.norm(center_score)))
    diagnostics.update(finite_sample_count=count, nonfinite_sample_count=0, pilot={'frozen_basis': True},
        design_evaluation_route='scalar_value_and_score_loop')
    positions = center + np.concatenate((z_holdout, z_train)) * scale
    values = np.r_[y_holdout, y_train]
    scores = np.stack([np.asarray(target(point)[1]) for point in positions])
    candidates = [records.ExactCandidate(center, float(center_value), center_score / scale, 0, 'center'),
        *records.candidates_from_rows(positions, values, scores, start_index=1, source_role='design')]
    incumbent = records.select_exact_incumbent(candidates)
    assert incumbent.evaluation_index == int(args[13])
    arguments = (target, center, scale, basis, z_train, y_train, score_train, z_holdout, y_holdout,
        float(center_value), center_score, original_cfg, dimension, rank, z_holdout.shape[0], count,
        float(np.linalg.norm(center_score)), incumbent, int(args[-1]))
    return arguments, candidates, diagnostics


def invoke_original_suffix(function, context):
    arguments, candidates, diagnostics = context
    return function(*arguments, candidates.copy(), diagnostics.copy())


def result_suffix(result):
    fields = {name: getattr(result, name) for name in TOP_FIELDS}
    fields.update({name: result.diagnostics[name] for name in DIAGNOSTIC_FIELDS if name in result.diagnostics})
    return fields
