"""Standalone Stan-algorithm TensorFlow/XLA sampler; no native runtime bridge."""
from .api import Config, FitResult, make_functions

__all__ = ['Config', 'FitResult', 'make_functions']
