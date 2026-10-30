"""Diagnostic inference validation; never imported by runtime admission code.

Catalog and design inspection are framework-free. Numerical procedures import
TensorFlow only after the executor establishes its declared device policy.
"""
from .catalog import TARGETS, TargetSpec, get_target
from .designs import ScenarioSpec, ValidationDesign

__all__ = ["TARGETS", "TargetSpec", "get_target", "ScenarioSpec", "ValidationDesign"]
