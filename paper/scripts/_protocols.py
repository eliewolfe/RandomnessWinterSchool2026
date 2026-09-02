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
    return ContextualityProtocol(
        scenario=scenario, where_key=where_key, master_key_holder=master,
        lp_solver="highs", **sdp_kwargs,
    )


def check(label: str, computed: float, expected: float, tol: float = 5e-4) -> bool:
    """Print a one-line comparison and return whether it matches."""
    ok = abs(float(computed) - float(expected)) <= tol
    print(f"  {'OK ' if ok else 'FAIL'} {label}: computed {computed:+.6f}, expected {expected:+.6f}")
    return ok
