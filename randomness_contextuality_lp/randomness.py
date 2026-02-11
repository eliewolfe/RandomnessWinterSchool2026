"""Randomness-oriented utilities acting on ``ContextualityScenario`` objects."""
from __future__ import annotations
import math
import numpy as np
from .scenario import ContextualityScenario
import mosek




def min_entropy_bits(p_guess: float) -> float:
    """Return ``-log2(P_guess)`` from average guessing probability."""
    return float(-math.log2(p_guess))

