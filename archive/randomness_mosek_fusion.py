"""Fusion-based LP routines for randomness quantification."""

from __future__ import annotations

import math

import mosek.fusion as mf

from .scenario import ContextualityScenario


def min_entropy_bits(p_guess: float) -> float:
    """Return min-entropy in bits from guessing probability."""
    return float(-math.log2(p_guess))


def eve_optimal_guessing_probability(
    scenario: ContextualityScenario,
    x: int = 0,
    y: int = 0,
) -> float:
    """Compute Eve's optimal guessing probability for one target pair ``(x, y)``."""
    if x < 0 or x >= scenario.X_cardinality:
        raise ValueError(f"x must be in 0..{scenario.X_cardinality - 1}.")
    if y < 0 or y >= scenario.Y_cardinality:
        raise ValueError(f"y must be in 0..{scenario.Y_cardinality - 1}.")

    with mf.Model("single_xy_guessing") as model:
        distribution = _build_tripartite_distribution_with_constraints(
            model=model,
            scenario=scenario,
            num_targets=1,
        )
        objective_expr = _guessing_expression_for_target(
            distribution=distribution,
            scenario=scenario,
            target_index=0,
            x=x,
            y=y,
        )
        model.objective("maximize_guessing_prob", mf.ObjectiveSense.Maximize, objective_expr)
        return _solve_and_get_primal_objective(model)


def eve_optimal_average_guessing_probability(scenario: ContextualityScenario) -> float:
    """Compute average optimal guessing probability over all ``(x, y)`` targets.

    This models one eavesdropper distribution per target pair:
    ``P_t(a,b,e|x,y)``, where ``t`` labels the target ``(x_t, y_t)``.
    """
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality
    num_targets = num_x * num_y

    with mf.Model("average_xy_guessing") as model:
        distribution = _build_tripartite_distribution_with_constraints(
            model=model,
            scenario=scenario,
            num_targets=num_targets,
        )

        pieces = []
        target_index = 0
        for x in range(num_x):
            for y in range(num_y):
                pieces.append(
                    _guessing_expression_for_target(
                        distribution=distribution,
                        scenario=scenario,
                        target_index=target_index,
                        x=x,
                        y=y,
                    )
                )
                target_index += 1

        mean_objective = mf.Expr.mul(1.0 / float(num_targets), mf.Expr.add(pieces))
        model.objective("maximize_average_guessing_prob", mf.ObjectiveSense.Maximize, mean_objective)
        return _solve_and_get_primal_objective(model)


def _build_tripartite_distribution_with_constraints(
    model: mf.Model,
    scenario: ContextualityScenario,
    num_targets: int,
) -> mf.Variable:
    """Create ``P_t(a,b,e|x,y)`` and add all linear constraints.

    Variable shape:
    - ``[target, x, y, a, b, e]`` with ``|e| = |b|``.
    """
    num_x = scenario.X_cardinality
    num_y = scenario.Y_cardinality
    num_a = scenario.A_cardinality
    num_b = scenario.B_cardinality
    num_e = num_b

    # Nonnegative tripartite conditional distributions.
    distribution = model.variable(
        "P",
        [num_targets, num_x, num_y, num_a, num_b, num_e],
        mf.Domain.greaterThan(0.0),
    )

    # Data consistency: sum_e P_t(a,b,e|x,y) = P_data(a,b|x,y), for every target t.
    for t in range(num_targets):
        for x in range(num_x):
            for y in range(num_y):
                for a in range(num_a):
                    for b in range(num_b):
                        slice_expr = distribution.slice(
                            [t, x, y, a, b, 0],
                            [t + 1, x + 1, y + 1, a + 1, b + 1, num_e],
                        )
                        model.constraint(
                            f"data_t{t}_x{x}_y{y}_a{a}_b{b}",
                            mf.Expr.sum(slice_expr),
                            mf.Domain.equalsTo(float(scenario.data[x, y, a, b])),
                        )

    # Preparation operational equivalences:
    # sum_{x,a} c[x,a] P_t(a,b,e|x,y) = 0 for all t,y,b,e.
    for t in range(num_targets):
        for k in range(scenario.opeq_preps.shape[0]):
            coeffs = scenario.opeq_preps[k]
            for y in range(num_y):
                for b in range(num_b):
                    for e in range(num_e):
                        terms = []
                        for x in range(num_x):
                            for a in range(num_a):
                                coeff = float(coeffs[x, a])
                                if coeff != 0.0:
                                    terms.append(
                                        mf.Expr.mul(coeff, distribution.index([t, x, y, a, b, e]))
                                    )
                        lhs = mf.Expr.add(terms) if terms else mf.Expr.constTerm(0.0)
                        model.constraint(
                            f"prep_t{t}_k{k}_y{y}_b{b}_e{e}",
                            lhs,
                            mf.Domain.equalsTo(0.0),
                        )

    # Measurement operational equivalences:
    # sum_{y,b} d[y,b] P_t(a,b,e|x,y) = 0 for all t,x,a,e.
    for t in range(num_targets):
        for k in range(scenario.opeq_meas.shape[0]):
            coeffs = scenario.opeq_meas[k]
            for x in range(num_x):
                for a in range(num_a):
                    for e in range(num_e):
                        terms = []
                        for y in range(num_y):
                            for b in range(num_b):
                                coeff = float(coeffs[y, b])
                                if coeff != 0.0:
                                    terms.append(
                                        mf.Expr.mul(coeff, distribution.index([t, x, y, a, b, e]))
                                    )
                        lhs = mf.Expr.add(terms) if terms else mf.Expr.constTerm(0.0)
                        model.constraint(
                            f"meas_t{t}_k{k}_x{x}_a{a}_e{e}",
                            lhs,
                            mf.Domain.equalsTo(0.0),
                        )

    return distribution


def _guessing_expression_for_target(
    distribution: mf.Variable,
    scenario: ContextualityScenario,
    target_index: int,
    x: int,
    y: int,
) -> mf.Expression:
    """Build ``sum_b sum_a P_t(a,b,e=b|x,y)`` for one target ``t``."""
    num_a = scenario.A_cardinality
    num_b = scenario.B_cardinality

    terms = []
    for b in range(num_b):
        for a in range(num_a):
            terms.append(distribution.index([target_index, x, y, a, b, b]))
    return mf.Expr.add(terms) if terms else mf.Expr.constTerm(0.0)


def _solve_and_get_primal_objective(model: mf.Model) -> float:
    """Solve a Fusion model and return the objective value."""
    model.solve()
    return float(model.primalObjValue())
