"""
PhyNEO-Electrolyte: A Hybrid Physics-Driven Neural Network Force Field for Liquid Electrolytes

This package provides a hybrid approach combining physics-based models with machine learning
for accurate simulation of liquid electrolytes.
"""

__version__ = "0.1.0"
__author__ = "dreamchen"

from . import models
from . import utils

__all__ = ["models", "utils"]
