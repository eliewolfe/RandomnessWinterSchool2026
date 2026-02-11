"""End-to-end demos using direct quantum -> scenario constructors."""

from __future__ import annotations

import numpy as np

from randomness_contextuality_lp.randomness import run_quantum_example


def _projector(ket: np.ndarray) -> np.ndarray:
    return np.outer(ket, ket.conj())


def main() -> None:
    np.set_printoptions(precision=6, suppress=True)

    ket0 = np.array([1.0, 0.0], dtype=complex)
    ket1 = np.array([0.0, 1.0], dtype=complex)
    ket_plus = (ket0 + ket1) / np.sqrt(2.0)
    ket_minus = (ket0 - ket1) / np.sqrt(2.0)

    quantum_states = np.array(
        [_projector(ket0), _projector(ket1), _projector(ket_plus), _projector(ket_minus)],
        dtype=complex,
    )  # (X=4, d, d)

    sigma_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    sigma_z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)

    op_x_plus_z = (sigma_x + sigma_z) / np.sqrt(2.0)
    eigvals_pz, eigvecs_pz = np.linalg.eigh(op_x_plus_z)
    ket_pz_minus = eigvecs_pz[:, np.argmin(eigvals_pz)]
    ket_pz_plus = eigvecs_pz[:, np.argmax(eigvals_pz)]

    op_x_minus_z = (sigma_x - sigma_z) / np.sqrt(2.0)
    eigvals_mz, eigvecs_mz = np.linalg.eigh(op_x_minus_z)
    ket_mz_minus = eigvecs_mz[:, np.argmin(eigvals_mz)]
    ket_mz_plus = eigvecs_mz[:, np.argmax(eigvals_mz)]

    # Example 1: effects +/-Z, +/-X, +/- (X+Z).
    effects_example_1 = np.array(
        [
            _projector(ket0),       # +Z
            _projector(ket1),       # -Z
            _projector(ket_plus),   # +X
            _projector(ket_minus),  # -X
            _projector(ket_pz_plus),   # +(X+Z)
            _projector(ket_pz_minus),  # -(X+Z)
        ],
        dtype=complex,
    )
    run_quantum_example(
        title="Example 1: Z, X, and (X+Z) measurements",
        quantum_states=quantum_states,
        quantum_effect_set=effects_example_1,
        target_pair=(0, 0),  # target setting (x=0, y=0)
    )

    # Example 2: effects +/- (X+Z), +/- (X-Z).
    effects_example_2 = np.array(
        [
            _projector(ket_pz_plus),   # +(X+Z)
            _projector(ket_pz_minus),  # -(X+Z)
            _projector(ket_mz_plus),   # +(X-Z)
            _projector(ket_mz_minus),  # -(X-Z)
        ],
        dtype=complex,
    )
    run_quantum_example(
        title="Example 2: (X+Z) and (X-Z) measurements",
        quantum_states=quantum_states,
        quantum_effect_set=effects_example_2,
        target_pair=(0, 1),  # target setting (x=0, y=1)
    )


if __name__ == "__main__":
    main()
