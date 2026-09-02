"""Shared protocol constructions for the paper's verification scripts.

Every script under ``paper/scripts/`` imports its scenarios from here, so a
reader checking any numerical claim of the manuscript sees exactly one
definition of each protocol.

Conventions used throughout (matching the manuscript):

- "Operational (OPT) adversary" bounds come from ``QKDNoncontextualLP`` and
  are EXACT (the LP is tight for that adversary class).
- "Quantum adversary" bounds come from ``QKDNoncontextualSDP``. The default
  relaxation level everywhere is ``npa_level_bob = npa_level_eve = 1``, whose
  moment matrix is indexed by the words {identity, Bob operators, Eve
  operators, Bob*Eve products} -- i.e. level 1 of the Moroder-type hierarchy
  ("NPA 1+ABE", with Alice's preparations x indexing the moment-matrix
  blocks). Where a script needs a higher level it says so explicitly.
- ``MORODER_LEVEL_1`` / ``MORODER_LEVEL_2`` bundle the corresponding
  ``ContextualityProtocol`` keyword arguments.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from contextualityqkd.demos.qkd_porac_3_2 import build_porac_scenario
from contextualityqkd.protocol import ContextualityProtocol
from contextualityqkd.quantum import GPTContextualityScenario

G = GPTContextualityScenario

# ContextualityProtocol kwargs for the two hierarchy levels used in the paper.
MORODER_LEVEL_1 = dict(
    sdp_solver="MOSEK",
    sdp_projective_bob=False,
    sdp_projective_eve=False,
    sdp_npa_level_bob=1,
    sdp_npa_level_eve=1,
    sdp_use_u_only=True,
    sdp_threads=2,
    sdp_verbose=0,
)
MORODER_LEVEL_2 = {**MORODER_LEVEL_1, "sdp_npa_level_bob": 2, "sdp_npa_level_eve": 2}

# Preparation angles on the X-Z great circle: |0>, |1>, |+>, |->.
BB84_STATE_ANGLES = (0, sp.pi, sp.pi / 2, -sp.pi / 2)


def _hs(angle):
    return G.projector_hs_vector(G.xz_plane_ket(angle))


def _bb84_states(eta: sp.Rational | int = 1):
    """BB84 states depolarized with visibility eta (eta = 1: pure states)."""
    mixed = (_hs(0) + _hs(sp.pi)) / 2
    return [sp.simplify(eta * _hs(a) + (1 - eta) * mixed) for a in BB84_STATE_ANGLES]


def aligned_bb84_scenario(eta: sp.Rational | int = 1) -> GPTContextualityScenario:
    """BB84 states, Bob measuring in the SAME Z and X bases (plain BB84)."""
    effects = [_hs(a) for a in BB84_STATE_ANGLES]
    return GPTContextualityScenario(
        gpt_states=_bb84_states(eta), gpt_effects=effects,
        measurement_indices=[(0, 1), (2, 3)], verbose=False,
    )


def rotated_bb84_scenario(eta: sp.Rational | int = 1) -> GPTContextualityScenario:
    """BB84 states, Bob measuring in the two intermediate bases (rotated by pi/4)."""
    effects = [_hs(a + sp.pi / 4) for a in BB84_STATE_ANGLES]
    return GPTContextualityScenario(
        gpt_states=_bb84_states(eta), gpt_effects=effects,
        measurement_indices=[(0, 1), (2, 3)], verbose=False,
    )


SINGLE_SETTING_WHERE_KEY = [(0, 1, 2, 3), ()]  # key from y* = 0 only


def xz_ring_scenario(num_states: int, num_meas: int) -> GPTContextualityScenario:
    """Evenly spaced antipodal binary measurements on the X-Z ring."""
    half = num_states // 2
    bases = sorted(set((m * half) // num_meas for m in range(num_meas)))
    assert len(bases) == num_meas, "measurement anchors not unique"
    idx = tuple((i, i + half) for i in bases)
    return G.from_xz_ring(num_states=num_states, measurement_indices=idx, verbose=False)


def hexagon_scenario() -> GPTContextualityScenario:
    return xz_ring_scenario(6, 3)


HEXAGON_ALIGNED_WHERE_KEY = [(0, 3), (1, 4), (2, 5)]


def cabello_18ray_scenario() -> GPTContextualityScenario:
    """Cabello's 18-ray / 9-context Kochen-Specker set in dimension 4."""
    labels = list("123456789ABCDEFGHI")
    rays = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [1, 1, 1, 1], [1, -1, 1, -1],
         [1, -1, -1, 1], [1, -1, -1, -1], [1, -1, 1, 1], [1, 1, 1, -1], [1, 1, 0, 0],
         [0, 0, 1, 1], [0, 0, 1, -1], [0, 1, 0, 1], [0, 1, 0, -1], [1, 0, -1, 0],
         [1, 0, 0, -1], [1, 0, 0, 1], [0, 1, -1, 0]],
        dtype=int,
    )
    contexts = ["12BC", "13DE", "23GH", "45EF", "46GI", "56AB", "78AC", "79HI", "89DF"]
    l2i = {lab: i for i, lab in enumerate(labels)}
    mi = [tuple(l2i[ch] for ch in c) for c in contexts]
    return G.from_integer_rays(rays=rays, measurement_indices=mi, verbose=False)


def porac_scenario(eta: float = 1.0):
    """(3,2)-parity-oblivious random access code (X=8, Y=3, B=2)."""
    return build_porac_scenario(eta=eta)


def protocol(scenario, *, where_key=None, level: int = 1, master="Alice") -> ContextualityProtocol:
    """House-default protocol object (LP: HiGHS; SDP: Moroder level 1 or 2)."""
    sdp_kwargs = MORODER_LEVEL_1 if int(level) == 1 else MORODER_LEVEL_2
    # LP backend: MOSEK interior point. On the large, highly degenerate RAC
    # scenarios simplex methods (HiGHS, MOSEK simplex) stall by orders of
    # magnitude, while interior point solves in under a second; optima agree.
    return ContextualityProtocol(
        scenario=scenario, where_key=where_key, master_key_holder=master,
        lp_solver="mosek", **sdp_kwargs,
    )


def check(label: str, computed: float, expected: float, tol: float = 5e-4) -> bool:
    """Print a one-line comparison and return whether it matches."""
    ok = abs(float(computed) - float(expected)) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {label}: computed {computed:+.6f}, expected {expected:+.6f}")
    return ok


# --------------------------------------------------------------------------- #
# (n,d) random-access-code protocol family                                     #
# --------------------------------------------------------------------------- #

import itertools as _it

from contextualityqkd.scenario import ContextualityScenario


def _mub_bases(n: int, d: int) -> list[np.ndarray]:
    """n mutually unbiased bases in dimension d (columns are basis vectors).

    Bases 0 and 1 (computational and Fourier) exist for every d; further
    bases use the odd-prime construction (or X/Y/Z for d = 2)."""
    comp = np.eye(d, dtype=complex)
    w = np.exp(2j * np.pi / d)
    fourier = np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)
    bases = [comp, fourier]
    if n >= 3:
        if d == 2:
            bases.append(np.array([[1, 1], [1j, -1j]], dtype=complex) / np.sqrt(2))
        else:
            for a in range(1, n - 1):
                bases.append(np.array([[w ** ((a * j * j + j * k) % d) for k in range(d)]
                                       for j in range(d)]) / np.sqrt(d))
    return bases[:n]


def _hermitian_basis(d: int) -> list[np.ndarray]:
    """Orthonormal Hermitian operator basis (Tr[H_i H_j] = delta_ij)."""
    basis = []
    for i in range(d):
        e = np.zeros((d, d), dtype=complex); e[i, i] = 1.0; basis.append(e)
    for i in range(d):
        for j in range(i + 1, d):
            s1 = np.zeros((d, d), dtype=complex); s1[i, j] = s1[j, i] = 1 / np.sqrt(2)
            s2 = np.zeros((d, d), dtype=complex); s2[i, j] = -1j / np.sqrt(2); s2[j, i] = 1j / np.sqrt(2)
            basis.extend((s1, s2))
    return basis


def _op_coords(op: np.ndarray, hb: list[np.ndarray]) -> np.ndarray:
    return np.array([float(np.real(np.trace(h.conj().T @ op))) for h in hb])


def qrac_words(n: int, d: int) -> list[tuple[int, ...]]:
    return list(_it.product(range(d), repeat=n))


def qrac_optimal_states(n: int, d: int) -> tuple[dict, list[np.ndarray]]:
    """Optimal unconstrained (n,d)-QRAC states for the n MUB measurements.

    Each word's state is the top eigenvector of the sum of the addressed
    projectors; for n = 2 this attains the proven-optimal success
    (1 + 1/sqrt(d))/2 [Ambainis et al. 1510.03045, Eq. (45); optimality by
    Farkas & Kaniewski, PRA 99, 032316]."""
    bases = _mub_bases(n, d)
    projs = [[np.outer(b[:, k], b[:, k].conj()) for k in range(d)] for b in bases]
    states = {}
    for x in qrac_words(n, d):
        F = sum(projs[y][x[y]] for y in range(n))
        vals, vecs = np.linalg.eigh(F)
        states[x] = np.outer(vecs[:, -1], vecs[:, -1].conj())
    return states, bases


def dary_parity_directions(n: int, d: int) -> list[tuple[int, ...]]:
    """Hidden directions of the d-ary oblivious multiplexing task: one
    representative per projective class of a in Z_d^n with ALL components
    nonzero (hiding any pure direction would hide a letter itself)."""
    import math as _math
    seen: set = set()
    out = []
    for a in _it.product(range(d), repeat=n):
        if any(ai == 0 for ai in a):
            continue
        cls = frozenset(tuple((c * ai) % d for ai in a)
                        for c in range(1, d) if _math.gcd(c, d) == 1)
        if cls & seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def pom_seesaw_states_and_measurements(n: int, d: int, *, rounds: int = 6,
                                       dirs: list | None = None,
                                       solver: str = "SCS", verbose: bool = False):
    """See-saw for the d-ary oblivious multiplexing protocol.

    Alternates two convex SDPs: optimal oblivious states for fixed
    measurements, then optimal POVMs for fixed states, starting from the n
    MUBs. Returns (success, states dict, list of POVMs)."""
    import cvxpy as cp

    words = qrac_words(n, d)
    hb = _hermitian_basis(d)
    dim = d * d
    id_coords = _op_coords(np.eye(d), hb)
    bases = _mub_bases(n, d)
    povms = [[np.outer(b[:, k], b[:, k].conj()) for k in range(d)] for b in bases]
    if dirs is None:
        dirs = dary_parity_directions(n, d)

    def as_matrix(vec):
        return sum(vec[i] * hb[i] for i in range(dim))

    value = None
    states = None
    for _ in range(rounds):
        # --- states step (obliviousness-constrained) ---
        v = {x: cp.Variable(dim) for x in words}
        cons = []
        for x in words:
            cons += [as_matrix(v[x]) >> 0, v[x] @ id_coords == 1.0]
        for a in dirs:
            cosets: dict = {}
            for x in words:
                cosets.setdefault(sum(ai * xi for ai, xi in zip(a, x)) % d, []).append(v[x])
            keys = sorted(cosets)
            base = sum(cosets[keys[0]])
            for k in keys[1:]:
                cons.append(sum(cosets[k]) == base)
        eff = {x: sum(_op_coords(povms[y][x[y]], hb) for y in range(n)) for x in words}
        prob = cp.Problem(cp.Maximize(sum(v[x] @ eff[x] for x in words) / (n * d ** n)), cons)
        prob.solve(solver=getattr(cp, solver))
        new_value = float(prob.value)
        states = {x: as_matrix(v[x]).value for x in words}
        # --- measurements step (per setting, unconstrained POVM) ---
        for y in range(n):
            M = [cp.Variable((d, d), hermitian=True) for _ in range(d)]
            consm = [m >> 0 for m in M] + [sum(M) == np.eye(d)]
            objm = sum(cp.real(cp.trace(M[x[y]] @ states[x])) for x in words) / (n * d ** n)
            cp.Problem(cp.Maximize(objm), consm).solve(solver=getattr(cp, solver))
            povms[y] = [np.asarray(m.value) for m in M]
        if verbose:
            print(f"  see-saw: states step S = {new_value:.6f}")
        if value is not None and abs(new_value - value) < 1e-7:
            value = new_value
            break
        value = new_value
    # final consistent success value with the last POVMs
    success = sum(float(np.real(np.trace(povms[y][x[y]] @ states[x])))
                  for x in words for y in range(n)) / (n * d ** n)
    return success, states, povms


def rac_scenario_from_states(states: dict, povms: list, n: int, d: int,
                             *, promised_dirs: list | None = None) -> ContextualityScenario:
    """Numeric ContextualityScenario for an (n,d) RAC protocol.

    The behavior table is P(b|x,y) = Tr[rho_x M^y_b]. Preparation
    operational equivalences: the promised d-ary obliviousness relations
    when promised_dirs is given, else the full state-level nullspace of the
    ensemble. Measurement equivalences: effect-level nullspace (for
    projective MUBs this is completeness only)."""
    words = qrac_words(n, d)
    data = np.zeros((len(words), n, d))
    for i, x in enumerate(words):
        for y in range(n):
            for b in range(d):
                data[i, y, b] = float(np.real(np.trace(povms[y][b] @ states[x])))
    hb = _hermitian_basis(d)
    if promised_dirs is None:
        from contextualityqkd.linalg_utils import null_space_basis
        smat = np.array([_op_coords(states[x], hb) for x in words])
        prep = null_space_basis(smat.T, atol=1e-7).reshape(-1, len(words))
    else:
        rows = []
        for a in promised_dirs:
            cosets: dict = {}
            for i, x in enumerate(words):
                cosets.setdefault(sum(ai * xi for ai, xi in zip(a, x)) % d, []).append(i)
            keys = sorted(cosets)
            for k in keys[1:]:
                row = np.zeros(len(words))
                row[cosets[keys[0]]] = 1.0
                row[cosets[k]] = -1.0
                rows.append(row)
        prep = np.array(rows)
    from contextualityqkd.linalg_utils import null_space_basis
    emat = np.array([_op_coords(povms[y][b], hb) for y in range(n) for b in range(d)])
    meas_null = null_space_basis(emat.T, atol=1e-7).reshape(-1, n, d)
    return ContextualityScenario(data=data, opeq_preps=prep, opeq_meas=meas_null,
                                 atol=1e-7, verbose=False)


def repair_povms(povms: list, atol: float = 1e-8) -> list:
    """Return exactly-complete PSD POVMs close to the given ones.

    Symmetrizes each element, clips tiny negative eigenvalues, then conjugates
    by S^{-1/2} (S = sum of elements) so completeness holds to machine
    precision -- solver output from a see-saw is otherwise complete only to
    solver tolerance, which the SDP's exact Naimark identities reject."""
    out = []
    for povm in povms:
        clipped = []
        for M in povm:
            H = (np.asarray(M) + np.asarray(M).conj().T) / 2
            vals, vecs = np.linalg.eigh(H)
            vals = np.clip(vals, 0.0, None)
            clipped.append((vecs * vals) @ vecs.conj().T)
        S = sum(clipped)
        vals, vecs = np.linalg.eigh(S)
        inv_sqrt = (vecs * (1.0 / np.sqrt(np.clip(vals, atol, None)))) @ vecs.conj().T
        out.append([inv_sqrt @ M @ inv_sqrt for M in clipped])
    return out
