"""Diagnostic source-pinned Python chunk oracle, never imported by runtime."""

import ast
import math
import subprocess
from types import SimpleNamespace

import tensorflow as tf


def original_evaluator(callback, dimension, batch_size):
    source = subprocess.check_output([
        "git", "show", "bd36a89b:bayesfilter/inference/batched_quadratic_center.py",
    ], text=True)
    outer = next(node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name == "refine_batched_quadratic_center")
    body = next(node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == "evaluate")
    namespace = {"tf": tf, "math": math, "callback": callback,
                 "cfg": SimpleNamespace(batch_size=batch_size)}
    exec(compile(ast.Module(body=[body], type_ignores=[]), "bd36a89b:evaluate", "exec"), namespace)  # noqa: S102

    def evaluate(points, point_count, record, center, value, score, first_index, selected_index):
        details = {"physical_rows": int(first_index), "padded_rows": 0, "callback_batches": 0,
                   "invalid_rows": 0, "candidate_batches": [], "selected_evaluation_index": int(selected_index)}
        incumbent = [center, value, score]
        namespace.update(details=details, incumbent=incumbent)
        collected = namespace["evaluate"](points[:int(point_count)], "fixture", record=bool(record))
        batches = details["candidate_batches"]
        result = {
            "ok": collected is not None, "physical_rows": details["physical_rows"] - int(first_index),
            "padded_rows": details["padded_rows"], "invalid_rows": details["invalid_rows"],
            "callback_batches": details["callback_batches"], "center": incumbent[0],
            "center_value": incumbent[1], "center_score": incumbent[2],
            "selected_index": details["selected_evaluation_index"],
            "positions": tf.stack([item["positions"] for item in batches]) if batches else tf.zeros([0, batch_size, dimension], tf.float64),
            "values": tf.stack([item["values"] for item in batches]) if batches else tf.zeros([0, batch_size], tf.float64),
            "scores": tf.stack([item["scores"] for item in batches]) if batches else tf.zeros([0, batch_size, dimension], tf.float64),
            "valid": tf.stack([item["valid"] for item in batches]) if batches else tf.zeros([0, batch_size], tf.bool),
        }
        # The extracted function uses module globals instead of the original
        # invocation-local closure. Release those temporary bindings so the
        # oracle retains no extra output tensors after its caller drops them.
        namespace.pop("details")
        namespace.pop("incumbent")
        return result

    return evaluate
