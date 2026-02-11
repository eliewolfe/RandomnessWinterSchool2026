"""Construct probability tables from quantum states and measurements."""

from __future__ import annotations

import numpy as np


def QuantumConstructor(rho_x, E_by) -> np.ndarray:
    """Return probabilities p(b|x,y) = Tr(rho_x E_{b|y}).

    Parameters
    ----------
    rho_x : array-like, shape (X, d, d)
        Preparation states (density matrices).
    E_by : array-like, shape (Y, B, d, d)
        POVM effects for each measurement setting y and outcome b.

    Returns
    -------
    np.ndarray
        Probability table with shape (X, Y, B), where entry [x, y, b]
        equals Tr(rho_x E_{b|y}).
    """
    rho = np.asarray(rho_x, dtype=complex)
    effects = np.asarray(E_by, dtype=complex)

    if rho.ndim != 3:
        raise ValueError("rho_x must have shape (X, d, d).")
    if effects.ndim != 4:
        raise ValueError("E_by must have shape (Y, B, d, d).")

    X, d1, d2 = rho.shape
    Y, B, d3, d4 = effects.shape

    if d1 != d2:
        raise ValueError("rho_x matrices must be square.")
    if d3 != d4:
        raise ValueError("E_{b|y} matrices must be square.")
    if d1 != d3:
        raise ValueError("rho_x and E_{b|y} must have matching dimensions.")

    # Tr(A B) = sum_{i,j} A_{i,j} * B_{j,i}
    probabilities = np.einsum("xij,ybji->xyb", rho, effects)

    # Drop tiny imaginary numerical noise if present.
    if np.max(np.abs(np.imag(probabilities))) < 1e-12:
        probabilities = probabilities.real

    return probabilities
