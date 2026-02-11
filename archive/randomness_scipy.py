"""Linear-programming routines for randomness quantification."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

from .scenario import ContextualityScenario


def min_entropy_bits(p_guess: float) -> float:
    """Return min-entropy in bits from guessing probability."""
    return float(-math.log2(p_guess))


def eve_optimal_guessing_probability(
    scenario: ContextualityScenario,
    x: int = 0,
    y: int = 0,
) -> float:
    """Compute Eve's optimal guessing probability via LP.

    Variables are a tripartite conditional distribution ``P(a,b,e|x,y)``
    with tensor shape ``(X, Y, A, B, E)``, where ``E = B``.

    Constraints:
    - Data consistency: ``sum_e P(a,b,e|x,y) = P_data(a,b|x,y)``.
    - Preparation opeqs: ``sum_{x,a} c[x,a] P(a,b,e|x,y) = 0`` for each prep-opeq.
    - Measurement opeqs: ``sum_{y,b} d[y,b] P(a,b,e|x,y) = 0`` for each meas-opeq.
    - Positivity: ``P(a,b,e|x,y) >= 0``.

    Objective (maximize):
    - ``sum_b P(B=b, E=b | X=x, Y=y)``, where A is marginalized.
    """
    if x < 0 or x >= scenario.X_cardinality:
        raise ValueError(f"x must be in 0..{scenario.X_cardinality - 1}.")
    if y < 0 or y >= scenario.Y_cardinality:
        raise ValueError(f"y must be in 0..{scenario.Y_cardinality - 1}.")

    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality
    num_a = scenario.A_cardinality
    num_b = scenario.B_cardinality
    num_e = num_b

    num_vars = num_x * num_y * num_a * num_b * num_e
    num_prep = scenario.opeq_preps.shape[0]
    num_meas = scenario.opeq_meas.shape[0]

    num_rows_data = num_x * num_y * num_a * num_b
    num_rows_prep = num_prep * num_y * num_b * num_e
    num_rows_meas = num_meas * num_x * num_a * num_e
    num_rows_total = num_rows_data + num_rows_prep + num_rows_meas

    a_eq = lil_matrix((num_rows_total, num_vars), dtype=float)
    b_eq = np.zeros(num_rows_total, dtype=float)

    def var_index(ix: int, iy: int, ia: int, ib: int, ie: int) -> int:
        return ((((ix * num_y) + iy) * num_a + ia) * num_b + ib) * num_e + ie

    row = 0
    for ix in range(num_x):
        for iy in range(num_y):
            for ia in range(num_a):
                for ib in range(num_b):
                    for ie in range(num_e):
                        a_eq[row, var_index(ix, iy, ia, ib, ie)] = 1.0
                    b_eq[row] = scenario.data[ix, iy, ia, ib]
                    row += 1

    for k in range(num_prep):
        coeffs = scenario.opeq_preps[k]
        for iy in range(num_y):
            for ib in range(num_b):
                for ie in range(num_e):
                    for ix in range(num_x):
                        for ia in range(num_a):
                            coeff = coeffs[ix, ia]
                            if coeff != 0.0:
                                a_eq[row, var_index(ix, iy, ia, ib, ie)] = coeff
                    b_eq[row] = 0.0
                    row += 1

    for k in range(num_meas):
        coeffs = scenario.opeq_meas[k]
        for ix in range(num_x):
            for ia in range(num_a):
                for ie in range(num_e):
                    for iy in range(num_y):
                        for ib in range(num_b):
                            coeff = coeffs[iy, ib]
                            if coeff != 0.0:
                                a_eq[row, var_index(ix, iy, ia, ib, ie)] = coeff
                    b_eq[row] = 0.0
                    row += 1

    c = np.zeros(num_vars, dtype=float)
    for ib in range(num_b):
        for ia in range(num_a):
            c[var_index(x, y, ia, ib, ib)] = -1.0

    result = linprog(
        c=c,
        A_eq=a_eq.tocsr(),
        b_eq=b_eq,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP solve failed: {result.message}")

    return float(-result.fun)
