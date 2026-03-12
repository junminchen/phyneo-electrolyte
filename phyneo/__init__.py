"""
PhyNEO-Electrolyte: A Hybrid Physics-Driven Neural Network Force Field for Liquid Electrolytes

This package provides a hybrid approach combining physics-based models with machine learning
for accurate simulation of liquid electrolytes.
"""
from importlib import import_module

__version__ = "0.1.0"
__author__ = "dreamchen"

__all__ = ["models", "utils"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
