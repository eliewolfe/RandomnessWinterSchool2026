"""Usage example for QuantumConstructor with 2 preparations and 1 measurement."""

import numpy as np

from randomness_contextuality_lp.quantum_constructor import QuantumConstructor


def main() -> None:
    # Two 1-qubit preparations: |0><0| and |1><1|
    ket0 = np.array([[1.0], [0.0]], dtype=complex)
    ket1 = np.array([[0.0], [1.0]], dtype=complex)
    rho_0 = ket0 @ ket0.conj().T
    rho_1 = ket1 @ ket1.conj().T
    rho_x = np.stack([rho_0, rho_1], axis=0)  # shape (X=2, d=2, d=2)

    # One 1-qubit measurement in the computational basis with two outcomes
    proj0 = rho_0
    proj1 = rho_1
    E_by = np.stack([[proj0, proj1]], axis=0)  # shape (Y=1, B=2, d=2, d=2)

    probabilities = QuantumConstructor(rho_x, E_by)

    print("p(b|x,y) shape:", probabilities.shape)
    print(probabilities)


if __name__ == "__main__":
    main()
