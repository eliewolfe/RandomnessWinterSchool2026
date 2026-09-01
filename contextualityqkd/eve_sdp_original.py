"""Manuscript-faithful SDP backend for contextuality-QKD Eve bounds.

Nonprojective measurements follow Appendix B's Naimark-unitary construction:
- Explicit generators U_k^y and (U_k^y)^dagger for k in [0, K_y-2]
- Last outcome eliminated via completeness identities (Eqs. (11)-(12))
- Data constraints in Eq. (8) form
- Measurement OPEQ constraints in Eq. (10) form
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from itertools import product
from pathlib import Path
import sys
from typing import Callable, Iterable, Literal, Sequence

import mosek
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VENDORED_INFLATION_ROOT = _REPO_ROOT / "external" / "inflation"
if _VENDORED_INFLATION_ROOT.exists() and str(_VENDORED_INFLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDORED_INFLATION_ROOT))

from inflation.sdp.fast_npa import nb_lexmon_to_canonical  # noqa: E402

from .scenario import ContextualityScenario


_IDENTITY: tuple[int, ...] = ()
_ZERO_MONOMIAL: None = None


@dataclass(frozen=True, order=True)
class Operator:
    """One primitive operator in the noncommutative alphabet."""

    party: str
    setting: int
    outcome: int
    lex_index: int
    kind: Literal["projector", "unitary"]
    is_dagger: bool = False
    adjoint_lex_index: int = -1

    @property
    def party_index(self) -> int:
        return 1 if self.party == "B" else 2

    def as_fast_npa_row(self) -> list[int]:
        # Use distinct outcome slots for U and U† so fast_npa ordering keeps both symbols.
        if self.kind == "unitary":
            outcome_slot = (2 * self.outcome) + (1 if self.is_dagger else 0)
            return [self.party_index, 1, self.setting + 1, outcome_slot + 1]
        return [self.party_index, 1, self.setting + 1, self.outcome + 1]


@dataclass
class LinearMomentConstraint:
    """A scalar affine constraint over moment entries."""

    terms: list[tuple[int, int, int, complex]]
    rhs: complex = 0.0
    name: str = ""
    domain: str = "zero_complex"


@dataclass
class MomentMatrixTemplate:
    """Shared template for all preparation-indexed moment matrices."""

    operator_sequence: list[tuple[int, ...]]
    operators: list[Operator]
    notcomm: np.ndarray
    projective_bob: bool = False
    projective_eve: bool = False
    row_mapping: dict[int, tuple[int, ...]] = field(init=False)
    col_mapping: dict[int, tuple[int, ...]] = field(init=False)
    word_to_index: dict[tuple[int, ...], int] = field(init=False)
    measurement_column_constraints: list[LinearMomentConstraint] = field(default_factory=list)
    entry_labels: dict[tuple[int, int], tuple[int, ...] | None] = field(init=False)
    entry_representatives: dict[tuple[int, ...], tuple[int, int]] = field(init=False)
    zero_entries: list[tuple[int, int]] = field(init=False)
    consistency_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = field(init=False)
    _effect_index: dict[tuple[str, int, int], int] = field(init=False)
    _unitary_index: dict[tuple[str, int, int, bool], int] = field(init=False)

    def __post_init__(self) -> None:
        self.row_mapping = {i: word for i, word in enumerate(self.operator_sequence)}
        self.col_mapping = dict(self.row_mapping)
        self.word_to_index = {word: idx for idx, word in self.row_mapping.items()}
        self._effect_index = {}
        self._unitary_index = {}
        for op in self.operators:
            if op.kind == "projector":
                self._effect_index[(op.party, op.setting, op.outcome)] = op.lex_index
            else:
                self._unitary_index[(op.party, op.setting, op.outcome, op.is_dagger)] = op.lex_index
        self.entry_labels = {}
        self.entry_representatives = {}
        self.zero_entries = []
        self.consistency_pairs = []
        self._build_entry_consistency()

    @property
    def dimension(self) -> int:
        return len(self.operator_sequence)

    @property
    def real_dimension(self) -> int:
        return 2 * int(self.dimension)

    def generate_sdp_variable(self, prep_index: int) -> "MomentMatrix":
        return MomentMatrix(template=self, prep_index=int(prep_index))

    def apply_operator_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        constraints = [
            LinearMomentConstraint(
                terms=[(prep_index, 0, 0, 1.0)],
                rhs=1.0,
                name=f"norm_x{prep_index}",
            )
        ]
        for row, col in self.zero_entries:
            constraints.append(
                LinearMomentConstraint(
                    terms=[(prep_index, row, col, 1.0)],
                    rhs=0.0,
                    name=f"zero_x{prep_index}_r{row}_c{col}",
                )
            )

        # Projective idempotency for primitive projectors.
        for op in self.operators:
            if not self._is_projective_operator(op):
                continue
            entry_pp = self.representative_entry_for_word((op.lex_index, op.lex_index))
            entry_p = self.representative_entry_for_word((op.lex_index,))
            if entry_pp is None or entry_p is None:
                continue
            constraints.append(
                LinearMomentConstraint(
                    terms=[
                        (prep_index, int(entry_pp[0]), int(entry_pp[1]), 1.0),
                        (prep_index, int(entry_p[0]), int(entry_p[1]), -1.0),
                    ],
                    rhs=0.0,
                    name=f"idempotent_x{prep_index}_{op.party}{op.setting}_{op.outcome}",
                )
            )

        # Appendix-B style unitarity/conjugacy constraints for nonprojective generators.
        for op in self.operators:
            if self._is_projective_operator(op) or op.kind != "unitary":
                continue
            entry_diag = self.representative_entry_for_word((op.adjoint_lex_index, op.lex_index))
            entry_u = self.representative_entry_for_word((op.lex_index,))
            entry_ud = self.representative_entry_for_word((op.adjoint_lex_index,))
            if entry_diag is None or entry_u is None or entry_ud is None:
                continue
            constraints.append(
                LinearMomentConstraint(
                    terms=[(prep_index, int(entry_diag[0]), int(entry_diag[1]), 1.0)],
                    rhs=1.0,
                    name=f"unitary_diag_x{prep_index}_{op.party}{op.setting}_{op.outcome}_{int(op.is_dagger)}",
                )
            )
            constraints.append(
                LinearMomentConstraint(
                    terms=[
                        (prep_index, int(entry_u[0]), int(entry_u[1]), 1.0),
                        (prep_index, int(entry_ud[0]), int(entry_ud[1]), -1.0),
                    ],
                    rhs=0.0,
                    name=f"unitary_conj_1_x{prep_index}_{op.party}{op.setting}_{op.outcome}_{int(op.is_dagger)}",
                )
            )
            constraints.append(
                LinearMomentConstraint(
                    terms=[
                        (prep_index, int(entry_ud[0]), int(entry_ud[1]), 1.0),
                        (prep_index, int(entry_u[0]), int(entry_u[1]), -1.0),
                    ],
                    rhs=0.0,
                    name=f"unitary_conj_2_x{prep_index}_{op.party}{op.setting}_{op.outcome}_{int(op.is_dagger)}",
                )
            )

        return constraints

    def apply_measurement_opeq_constraints(
        self,
        prep_index: int,
        opeq_meas: np.ndarray,
        *,
        b_cardinality_per_y: np.ndarray,
        party: str = "B",
    ) -> list[LinearMomentConstraint]:
        """Apply measurement OPEQs; nonprojective mode uses Eq. (10)-(12) structure."""

        constraints: list[LinearMomentConstraint] = []
        coeffs_arr = np.asarray(opeq_meas, dtype=float)
        if coeffs_arr.ndim == 2:
            coeffs_arr = coeffs_arr[np.newaxis, :, :]

        is_projective = (party == "B" and self.projective_bob) or (party == "E" and self.projective_eve)
        for k, coeffs in enumerate(coeffs_arr):
            for row_index in range(self.dimension):
                terms: list[tuple[int, int, int, float]] = []
                for y in range(coeffs.shape[0]):
                    b_count = int(b_cardinality_per_y[y])
                    if b_count <= 0:
                        continue
                    if is_projective:
                        beta_last = float(coeffs[y, b_count - 1])
                        if abs(beta_last) > 0.0:
                            terms.append((prep_index, row_index, 0, beta_last))
                        for b in range(max(0, b_count - 1)):
                            coeff = float(coeffs[y, b]) - beta_last
                            if abs(coeff) <= 0.0:
                                continue
                            idx = self.find_operator(party, y, b)
                            entry = self.entry_for_row_right_word(row_index, (int(idx),)) if idx is not None else None
                            if entry is not None:
                                terms.append((prep_index, int(entry[0]), int(entry[1]), coeff))
                        continue

                    # Nonprojective: sum_b beta_b M_b(row)=0 with
                    # M_k = I/2 + (U_k + U_k†)/4 and M_{K-1} eliminated.
                    beta_last = float(coeffs[y, b_count - 1])
                    beta_except = float(np.sum(coeffs[y, : max(0, b_count - 1)]))
                    id_coeff = 0.5 * beta_except + 0.5 * (3.0 - float(b_count)) * beta_last
                    if abs(id_coeff) > 0.0:
                        terms.append((prep_index, row_index, 0, id_coeff))
                    for b in range(max(0, b_count - 1)):
                        eff_coeff = 0.25 * (float(coeffs[y, b]) - beta_last)
                        if abs(eff_coeff) <= 0.0:
                            continue
                        u = self.find_operator(party, y, b, is_dagger=False)
                        ud = self.find_operator(party, y, b, is_dagger=True)
                        if u is not None:
                            entry_u = self.entry_for_row_right_word(row_index, (int(u),))
                            if entry_u is not None:
                                terms.append((prep_index, int(entry_u[0]), int(entry_u[1]), eff_coeff))
                        if ud is not None:
                            entry_ud = self.entry_for_row_right_word(row_index, (int(ud),))
                            if entry_ud is not None:
                                terms.append((prep_index, int(entry_ud[0]), int(entry_ud[1]), eff_coeff))

                if terms:
                    constraints.append(
                        LinearMomentConstraint(
                            terms=terms,
                            rhs=0.0,
                            name=f"meas_opeq_{party}_x{prep_index}_k{k}_row{row_index}",
                        )
                    )
        self.measurement_column_constraints.extend(constraints)
        return constraints

    def word_index(self, word: tuple[int, ...]) -> int | None:
        canonical = self.canonical_word(word)
        if canonical is _ZERO_MONOMIAL:
            return None
        return self.word_to_index.get(canonical)

    def representative_entry_for_word(self, word: tuple[int, ...]) -> tuple[int, int] | None:
        """Return representative (row,col) for the canonical label of ``word``."""
        canonical = self.canonical_word(word)
        if canonical is _ZERO_MONOMIAL:
            return None
        return self._representative_entry_for_label(canonical)

    def entry_for_row_right_word(self, row_index: int, right_word: tuple[int, ...]) -> tuple[int, int] | None:
        """Return representative entry for ``<row_word^† * right_word>``."""
        row_word = self.row_mapping[int(row_index)]
        label = self.canonical_word(self._adjoint_word(row_word) + tuple(int(v) for v in right_word))
        if label is _ZERO_MONOMIAL:
            return None
        assert label is not None
        return self._representative_entry_for_label(label)

    def _representative_entry_for_label(self, label: tuple[int, ...]) -> tuple[int, int] | None:
        rep = self.entry_representatives.get(label)
        if rep is not None:
            return rep
        # Representatives are stored from lower-triangular entries only; if a
        # label appears only in upper-triangular form, resolve via its adjoint.
        adj = self.canonical_word(self._adjoint_word(label))
        if adj is _ZERO_MONOMIAL or adj is None:
            return None
        return self.entry_representatives.get(adj)

    def find_operator(
        self,
        party: str,
        setting: int,
        outcome: int,
        is_dagger: bool | None = None,
    ) -> int | None:
        if is_dagger is None:
            return self._effect_index.get((party, setting, outcome))
        return self._unitary_index.get((party, setting, outcome, bool(is_dagger)))

    def canonical_word(self, word: Iterable[int]) -> tuple[int, ...] | None:
        lexmon = np.asarray(tuple(int(v) for v in word), dtype=np.intc)
        if lexmon.size == 0:
            return _IDENTITY
        canonical = tuple(int(v) for v in np.asarray(nb_lexmon_to_canonical(lexmon, self.notcomm), dtype=int))
        return self._apply_operator_rules(canonical)


    def _build_entry_consistency(self) -> None:
        for row, row_word in enumerate(self.operator_sequence):
            row_adj = self._adjoint_word(row_word)
            for col, col_word in enumerate(self.operator_sequence[: row + 1]):
                label = self.canonical_word(row_adj + col_word)
                self.entry_labels[row, col] = label
                if label is _ZERO_MONOMIAL:
                    self.zero_entries.append((row, col))
                    continue
                if label not in self.entry_representatives:
                    self.entry_representatives[label] = (row, col)
                    continue
                rep = self.entry_representatives[label]
                if rep != (row, col):
                    self.consistency_pairs.append(((row, col), rep))

    def _adjoint_word(self, word: tuple[int, ...]) -> tuple[int, ...]:
        return tuple(int(self.operators[idx].adjoint_lex_index) for idx in reversed(word))

    def _apply_operator_rules(self, word: tuple[int, ...]) -> tuple[int, ...] | None:
        if not word:
            return _IDENTITY

        out: list[int] = []
        for lex_index in word:
            op = self.operators[int(lex_index)]
            if out:
                prev = self.operators[int(out[-1])]
                if self._is_projective_operator(prev) and self._is_projective_operator(op):
                    if prev.party == op.party and prev.setting == op.setting:
                        if prev.outcome != op.outcome:
                            return _ZERO_MONOMIAL
                        out.pop()
                        out.append(int(prev.lex_index))
                        continue
                if prev.kind == "unitary" and op.kind == "unitary":
                    if (
                        prev.party == op.party
                        and prev.setting == op.setting
                        and prev.outcome == op.outcome
                        and prev.is_dagger != op.is_dagger
                    ):
                        out.pop()
                        continue
            out.append(int(lex_index))
        return tuple(out)

    def _is_projective_operator(self, op: Operator) -> bool:
        return op.kind == "projector"


@dataclass
class MomentMatrix:
    """One preparation-indexed moment matrix instance."""

    template: MomentMatrixTemplate
    prep_index: int = 0
    sdp_var: int | None = None

    @property
    def dimension(self) -> int:
        return self.template.dimension

    @property
    def real_dimension(self) -> int:
        return self.template.real_dimension

    def word_index(self, word: tuple[int, ...]) -> int | None:
        return self.template.word_index(word)

    def find_operator(
        self,
        party: str,
        setting: int,
        outcome: int,
        is_dagger: bool | None = None,
    ) -> int | None:
        return self.template.find_operator(party, setting, outcome, is_dagger=is_dagger)


class QKDNoncontextualSDP:
    """Semi-device-independent SDP relaxation for Eve's QKD guessing attack."""

    def __init__(
        self,
        scenario: ContextualityScenario,
        *,
        projective_bob: bool = False,
        projective_eve: bool = False,
        npa_level_bob: int = 1,
        npa_level_eve: int = 1,
        master_key_holder: Literal["Alice", "Bob"] | str = "Alice",
        where_key: Sequence[Sequence[int]] | None = None,
        use_u_only: bool = False,
        threads: int | None = None,
        atol: float | None = None,
        verbose: int | bool = 0,
    ) -> None:
        if not isinstance(scenario, ContextualityScenario):
            raise TypeError("scenario must be a ContextualityScenario instance.")
        self.scenario = scenario
        self.projective_bob = bool(projective_bob)
        self.projective_eve = bool(projective_eve)
        self.npa_level_bob = self._validate_level(npa_level_bob, "npa_level_bob")
        self.npa_level_eve = self._validate_level(npa_level_eve, "npa_level_eve")
        self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        self.where_key = self._normalize_where_key(where_key)
        self.use_u_only = bool(use_u_only)
        self.threads = None if threads is None else int(threads)
        self.atol = scenario.atol if atol is None else float(atol)
        self.verbose = int(verbose)

        self.operators: list[Operator] = []
        self.lexorder: np.ndarray | None = None
        self.notcomm: np.ndarray | None = None
        self.template: MomentMatrixTemplate | None = None
        self.moment_matrices: list[MomentMatrix] = []
        self.constraints: list[LinearMomentConstraint] = []
        self.objective_terms: list[tuple[int, int, int, float]] = []
        self.key_selection_table: np.ndarray | None = None
        self.objective_pair_count: int = 0
        self.eve_success_probability: float | None = None
        self.key_rate_lower_bound: float | None = None
        self.solution_matrices: list[np.ndarray] = []
        self.solution_matrices_real: list[np.ndarray] = []

    def _log(self, level: int, message: str) -> None:
        if self.verbose >= int(level):
            print(message)

    @staticmethod
    def _validate_level(value: int, name: str) -> int:
        level = int(value)
        if level < 0:
            raise ValueError(f"{name} must be nonnegative.")
        return level

    @staticmethod
    def _canonicalize_master_key_holder(master_key_holder: Literal["Alice", "Bob"] | str) -> Literal["Alice", "Bob"]:
        token = str(master_key_holder).strip().lower()
        if token == "alice":
            return "Alice"
        if token == "bob":
            return "Bob"
        raise ValueError("master_key_holder must be 'Alice' or 'Bob'.")

    def _party_is_projective(self, party: str) -> bool:
        return (party == "B" and self.projective_bob) or (party == "E" and self.projective_eve)

    def build_operator_list(self) -> list[Operator]:
        operators: list[Operator] = []
        b_counts = self.scenario.b_cardinality_per_y.astype(int, copy=False)

        for party in ("B", "E"):
            is_projective = self._party_is_projective(party)
            for y in range(self.scenario.Y_cardinality):
                count = int(b_counts[y])
                if is_projective:
                    for b in range(max(0, count - 1)):
                        operators.append(
                            Operator(
                                party=party,
                                setting=y,
                                outcome=b,
                                lex_index=len(operators),
                                kind="projector",
                                is_dagger=False,
                                adjoint_lex_index=len(operators),
                            )
                        )
                    continue
                for b in range(max(0, count - 1)):
                    idx_u = len(operators)
                    idx_ud = idx_u + 1
                    operators.append(
                        Operator(
                            party=party,
                            setting=y,
                            outcome=b,
                            lex_index=idx_u,
                            kind="unitary",
                            is_dagger=False,
                            adjoint_lex_index=idx_ud,
                        )
                    )
                    operators.append(
                        Operator(
                            party=party,
                            setting=y,
                            outcome=b,
                            lex_index=idx_ud,
                            kind="unitary",
                            is_dagger=True,
                            adjoint_lex_index=idx_u,
                        )
                    )

        self.operators = operators
        return list(operators)

    def build_lexorder_and_notcomm(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.operators:
            self.build_operator_list()
        self.lexorder = np.asarray([op.as_fast_npa_row() for op in self.operators], dtype=np.intc)

        n_ops = len(self.operators)
        notcomm = np.zeros((n_ops, n_ops), dtype=bool)
        for i in range(n_ops):
            for j in range(n_ops):
                op_i = self.operators[i]
                op_j = self.operators[j]
                if op_i.party != op_j.party:
                    notcomm[i, j] = False
                elif op_i.setting == op_j.setting:
                    notcomm[i, j] = bool(i != j)
                else:
                    notcomm[i, j] = True

        self.notcomm = notcomm
        return self.lexorder.copy(), self.notcomm.copy()

    def instantiate_moment_matrices(self) -> list[MomentMatrix]:
        if self.notcomm is None:
            self.build_lexorder_and_notcomm()
        assert self.notcomm is not None
        operator_sequence = self._build_operator_sequence()
        self.template = MomentMatrixTemplate(
            operator_sequence=operator_sequence,
            operators=self.operators,
            notcomm=self.notcomm,
            projective_bob=self.projective_bob,
            projective_eve=self.projective_eve,
        )
        self.moment_matrices = [
            self.template.generate_sdp_variable(x)
            for x in range(self.scenario.X_cardinality)
        ]
        if self.verbose >= 1:
            dim = int(self.template.dimension)
            real_dim = int(self.template.real_dimension)
            packed = real_dim * (real_dim + 1) // 2
            self._log(
                1,
                f"[eve_sdp] moment matrices: {len(self.moment_matrices)} blocks, "
                f"complex {dim}x{dim} embedded as real {real_dim}x{real_dim} "
                f"(symmetric packed={packed})",
            )
        if self.verbose >= 1 and self.use_u_only:
            self._log(1, "[eve_sdp] U-only generator mode: dagger operators excluded from NPA word sequence.")
        return list(self.moment_matrices)

    def apply_observed_data_constraints(self) -> list[LinearMomentConstraint]:
        if self.template is None or not self.moment_matrices:
            self.instantiate_moment_matrices()
        assert self.template is not None

        constraints: list[LinearMomentConstraint] = []
        for matrix in self.moment_matrices:
            x = matrix.prep_index
            constraints.extend(self._complex_embedding_constraints(x))
            constraints.extend(self.template.apply_operator_constraints(x))
            constraints.extend(self._moment_consistency_constraints(x))
            constraints.extend(
                self.template.apply_measurement_opeq_constraints(
                    x,
                    self.scenario.opeq_meas_numeric,
                    b_cardinality_per_y=self.scenario.b_cardinality_per_y,
                    party="B",
                )
            )
            constraints.extend(self._measurement_completeness_constraints(x))
            constraints.extend(self._observed_bob_constraints(x))
            constraints.extend(self._bob_eve_marginal_constraints(x))
            constraints.extend(self._probability_positivity_constraints(x))

        constraints.extend(self._preparation_opeq_constraints())
        self.constraints = constraints
        return list(constraints)

    def set_objective(
        self,
        key_selection_function: Callable[[int, int], int] | Sequence[Sequence[int]] | np.ndarray | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        where_key: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if self.template is None:
            self.instantiate_moment_matrices()
        assert self.template is not None
        if master_key_holder is not None:
            self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        if where_key is not None:
            self.where_key = self._normalize_where_key(where_key)
        key_table = self._normalize_key_selection(key_selection_function)
        terms: list[tuple[int, int, int, float]] = []
        pair_count = sum(len(row) for row in self.where_key)
        if pair_count == 0:
            raise ValueError("where_key must contain at least one key-eligible (x,y) pair.")
        weight = 1.0 / float(pair_count)
        for y, row in enumerate(self.where_key):
            b_count = int(self.scenario.b_cardinality_per_y[y])
            for x in row:
                if self.master_key_holder == "Alice":
                    e = int(key_table[x, y])
                    if e < 0 or e >= b_count:
                        raise ValueError(f"key_selection[{x},{y}] is not a valid outcome for y={y}.")
                    prob_terms = self._single_probability_terms(int(x), "E", y, e)
                    for term in prob_terms:
                        terms.append((term[0], term[1], term[2], weight * term[3]))
                    continue

                for b in range(b_count):
                    for term in self._joint_probability_terms(int(x), y, b, b):
                        terms.append((term[0], term[1], term[2], weight * term[3]))
        self.key_selection_table = key_table
        self.objective_pair_count = int(pair_count)
        self.objective_terms = terms

    def solve_sdp(
        self,
        key_selection_function: Callable[[int, int], int] | Sequence[Sequence[int]] | np.ndarray | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        where_key: Sequence[Sequence[int]] | None = None,
    ) -> float:
        if not self.constraints:
            self.apply_observed_data_constraints()
        if key_selection_function is not None or master_key_holder is not None or where_key is not None or not self.objective_terms:
            self.set_objective(
                key_selection_function,
                master_key_holder=master_key_holder,
                where_key=where_key,
            )

        assert self.template is not None
        self._log(
            1,
            f"[eve_sdp] solving (complex-embedded SDP) with {len(self.constraints)} affine constraints and "
            f"{self.scenario.X_cardinality} PSD block(s)",
        )
        with mosek.Env() as env:
            with env.Task(0, 0) as task:
                if self.verbose >= 2:
                    task.set_Stream(mosek.streamtype.log, lambda msg: print(msg, end=""))
                self._build_mosek_task(task)
                termination_code = task.optimize()
                self._assert_mosek_solution_is_not_infeasible_or_unbounded(
                    task,
                    termination_code=termination_code,
                    failure_context="Eve SDP",
                )
                objective = self._get_optimal_primal_objective(task, termination_code)
                self.eve_success_probability = float(objective)
                self.solution_matrices_real = self._extract_solution_matrices_real(task)
                self.solution_matrices = [
                    self._real_block_to_complex(matrix) for matrix in self.solution_matrices_real
                ]
        self.key_rate_lower_bound = self.compute_key_rate(rate_type="reverse_fano")
        return float(self.eve_success_probability)

    def solve(
        self,
        key_selection_function: Callable[[int, int], int] | Sequence[Sequence[int]] | np.ndarray | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        where_key: Sequence[Sequence[int]] | None = None,
    ) -> float:
        return self.solve_sdp(
            key_selection_function=key_selection_function,
            master_key_holder=master_key_holder,
            where_key=where_key,
        )

    def compute_key_rate(
        self,
        *,
        rate_type: Literal["reverse_fano", "min_entropy"] = "reverse_fano",
        other_party_uncertainty: float | None = None,
    ) -> float:
        if self.eve_success_probability is None:
            raise RuntimeError("solve_sdp must be called before compute_key_rate.")

        s_e = float(self.eve_success_probability)
        if s_e <= 0.0:
            raise ValueError("Eve success probability must be positive.")
        eve_uncertainty = self.eve_uncertainty(rate_type=rate_type)
        if other_party_uncertainty is None:
            other_party_uncertainty = self._other_party_uncertainty()
        self.key_rate_lower_bound = float(eve_uncertainty - float(other_party_uncertainty))
        return float(self.key_rate_lower_bound)

    def eve_uncertainty(self, *, rate_type: Literal["reverse_fano", "min_entropy"] = "reverse_fano") -> float:
        if self.eve_success_probability is None:
            raise RuntimeError("solve_sdp must be called before eve_uncertainty.")
        if rate_type == "reverse_fano":
            return self._reverse_fano_bound(float(self.eve_success_probability))
        if rate_type == "min_entropy":
            return float(-math.log2(float(self.eve_success_probability)))
        raise ValueError("rate_type must be 'reverse_fano' or 'min_entropy'.")

    def _build_operator_sequence(self) -> list[tuple[int, ...]]:
        def _party_generator_indices(party: str) -> list[int]:
            ops = [op for op in self.operators if op.party == party]
            if self.use_u_only:
                # Restrict to non-dagger unitaries and all projectors (no U†).
                ops = [op for op in ops if not (op.kind == "unitary" and op.is_dagger)]
            return [op.lex_index for op in ops]

        bob = _party_generator_indices("B")
        eve = _party_generator_indices("E")
        template = MomentMatrixTemplate(
            operator_sequence=[_IDENTITY],
            operators=self.operators,
            notcomm=np.asarray(self.notcomm, dtype=bool),
            projective_bob=self.projective_bob,
            projective_eve=self.projective_eve,
        )
        words: list[tuple[int, ...]] = [_IDENTITY]
        seen = {_IDENTITY}

        bob_words = self._party_words(bob, self.npa_level_bob)
        eve_words = self._party_words(eve, self.npa_level_eve)
        for b_word in bob_words:
            for e_word in eve_words:
                combined = template.canonical_word(b_word + e_word)
                if combined is _ZERO_MONOMIAL or combined in seen:
                    continue
                seen.add(combined)
                words.append(combined)
        return words

    @staticmethod
    def _party_words(operator_indices: list[int], level: int) -> list[tuple[int, ...]]:
        words: list[tuple[int, ...]] = [_IDENTITY]
        for length in range(1, int(level) + 1):
            words.extend(tuple(int(v) for v in word) for word in product(operator_indices, repeat=length))
        return words

    def _moment_consistency_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        assert self.template is not None
        out: list[LinearMomentConstraint] = []
        for (entry, rep) in self.template.consistency_pairs:
            row, col = entry
            rep_row, rep_col = rep
            out.append(
                LinearMomentConstraint(
                    terms=[
                        (prep_index, row, col, 1.0),
                        (prep_index, rep_row, rep_col, -1.0),
                    ],
                    rhs=0.0,
                    name=f"moment_consistency_x{prep_index}_r{row}_c{col}",
                )
            )
        return out

    def _complex_embedding_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        assert self.template is not None
        d = int(self.template.dimension)
        constraints: list[LinearMomentConstraint] = []
        for i in range(d):
            for j in range(d):
                constraints.append(
                    LinearMomentConstraint(
                        terms=[
                            (prep_index, i, j, 1.0),
                            (prep_index, d + i, d + j, -1.0),
                        ],
                        rhs=0.0,
                        name=f"embed_diagblock_x{prep_index}_r{i}_c{j}",
                        domain="zero_real",
                    )
                )
                constraints.append(
                    LinearMomentConstraint(
                        terms=[
                            (prep_index, i, d + j, 1.0),
                            (prep_index, d + i, j, 1.0),
                        ],
                        rhs=0.0,
                        name=f"embed_offblock_x{prep_index}_r{i}_c{j}",
                        domain="zero_real",
                    )
                )
        return constraints

    def _measurement_completeness_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        # The final outcome is reconstructed from completeness in both pathways,
        # so no separate completeness constraints are added here.
        return []

    def _observed_bob_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        assert self.template is not None
        constraints: list[LinearMomentConstraint] = []
        data = self.scenario.data_numeric
        for y in range(self.scenario.Y_cardinality):
            b_count = int(self.scenario.b_cardinality_per_y[y])
            if self.projective_bob:
                for b in range(b_count):
                    terms = self._single_probability_terms(prep_index, "B", y, b)
                    constraints.append(
                        LinearMomentConstraint(
                            terms=terms,
                            rhs=float(data[prep_index, y, b]),
                            name=f"data_proj_x{prep_index}_y{y}_b{b}",
                            domain="zero_complex",
                        )
                    )
                continue

            for b in range(b_count):
                terms = self._single_probability_terms(prep_index, "B", y, b)
                constraints.append(
                    LinearMomentConstraint(
                        terms=terms,
                        rhs=float(data[prep_index, y, b]),
                        name=f"data_naimark_x{prep_index}_y{y}_b{b}",
                        domain="zero_complex",
                    )
                )
        return constraints

    def _bob_eve_marginal_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        constraints: list[LinearMomentConstraint] = []
        for y in range(self.scenario.Y_cardinality):
            b_count = int(self.scenario.b_cardinality_per_y[y])
            for b in range(b_count):
                terms = [
                    (prep_index, row, col, -coeff)
                    for (_, row, col, coeff) in self._single_probability_terms(prep_index, "B", y, b)
                ]
                for e in range(b_count):
                    terms.extend(self._joint_probability_terms(prep_index, y, b, e))
                constraints.append(
                    LinearMomentConstraint(
                        terms=terms,
                        rhs=0.0,
                        name=f"joint_to_bob_x{prep_index}_y{y}_b{b}",
                    )
                )

            for e in range(b_count):
                terms = [
                    (prep_index, row, col, -coeff)
                    for (_, row, col, coeff) in self._single_probability_terms(prep_index, "E", y, e)
                ]
                for b in range(b_count):
                    terms.extend(self._joint_probability_terms(prep_index, y, b, e))
                constraints.append(
                    LinearMomentConstraint(
                        terms=terms,
                        rhs=0.0,
                        name=f"joint_to_eve_x{prep_index}_y{y}_e{e}",
                    )
                )
        return constraints

    def _probability_positivity_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        constraints: list[LinearMomentConstraint] = []

        def add_probability_bounds(terms: list[tuple[int, int, int, float]], name: str) -> None:
            complex_terms = [(prep, row, col, complex(coeff)) for prep, row, col, coeff in terms]
            constraints.append(
                LinearMomentConstraint(
                    terms=complex_terms,
                    rhs=0.0,
                    name=f"{name}_imag_zero",
                    domain="zero_imag",
                )
            )
            constraints.append(
                LinearMomentConstraint(
                    terms=complex_terms,
                    rhs=0.0,
                    name=f"{name}_lower",
                    domain="plus_real",
                )
            )
            constraints.append(
                LinearMomentConstraint(
                    terms=[(prep_index, row, col, -complex(coeff)) for _, row, col, coeff in terms],
                    rhs=-1.0,
                    name=f"{name}_upper",
                    domain="plus_real",
                )
            )

        for party in ("B", "E"):
            for y in range(self.scenario.Y_cardinality):
                for b in range(int(self.scenario.b_cardinality_per_y[y])):
                    add_probability_bounds(
                        self._single_probability_terms(prep_index, party, y, b),
                        f"prob_{party}_x{prep_index}_y{y}_b{b}",
                    )

        for y in range(self.scenario.Y_cardinality):
            count = int(self.scenario.b_cardinality_per_y[y])
            for b in range(count):
                for e in range(count):
                    add_probability_bounds(
                        self._joint_probability_terms(prep_index, y, b, e),
                        f"joint_x{prep_index}_y{y}_b{b}_e{e}",
                    )
        return constraints

    def _preparation_opeq_constraints(self) -> list[LinearMomentConstraint]:
        assert self.template is not None
        constraints: list[LinearMomentConstraint] = []
        opeqs = np.asarray(self.scenario.opeq_preps_numeric, dtype=float)
        if opeqs.ndim == 1:
            opeqs = opeqs[np.newaxis, :]
        for k, coeffs in enumerate(opeqs):
            nz_x = np.flatnonzero(np.abs(coeffs) > 0.0)
            if nz_x.size == 0:
                continue
            for row in range(self.template.dimension):
                for col in range(row + 1):
                    terms = [(int(x), row, col, float(coeffs[x])) for x in nz_x.tolist()]
                    constraints.append(
                        LinearMomentConstraint(
                            terms=terms,
                            rhs=0.0,
                            name=f"prep_opeq_k{k}_r{row}_c{col}",
                        )
                    )
        return constraints

    def _effect_affine_words(self, party: str, y: int, b: int) -> list[tuple[tuple[int, ...], float]]:
        assert self.template is not None
        b_count = int(self.scenario.b_cardinality_per_y[y])
        if self._party_is_projective(party):
            if b_count <= 1:
                return [(_IDENTITY, 1.0)]
            if b < b_count - 1:
                op_idx = self.template.find_operator(party, y, b)
                if op_idx is None:
                    raise RuntimeError(f"Missing projector for {party}, y={y}, b={b}.")
                return [((int(op_idx),), 1.0)]
            terms: list[tuple[tuple[int, ...], float]] = [(_IDENTITY, 1.0)]
            for k in range(b_count - 1):
                op_idx = self.template.find_operator(party, y, k)
                if op_idx is None:
                    raise RuntimeError(f"Missing projector for {party}, y={y}, b={k}.")
                terms.append(((int(op_idx),), -1.0))
            return terms

        if b_count <= 1:
            return [(_IDENTITY, 1.0)]

        terms: list[tuple[tuple[int, ...], float]] = []
        if b < b_count - 1:
            u = self.template.find_operator(party, y, b, is_dagger=False)
            ud = self.template.find_operator(party, y, b, is_dagger=True)
            if u is None or ud is None:
                raise RuntimeError(f"Missing unitary pair for {party}, y={y}, b={b}.")
            terms.append((_IDENTITY, 0.5))
            terms.append(((int(u),), 0.25))
            terms.append(((int(ud),), 0.25))
            return terms

        terms.append((_IDENTITY, 0.5 * (3.0 - float(b_count))))
        for k in range(b_count - 1):
            u = self.template.find_operator(party, y, k, is_dagger=False)
            ud = self.template.find_operator(party, y, k, is_dagger=True)
            if u is None or ud is None:
                raise RuntimeError(f"Missing unitary pair for {party}, y={y}, b={k}.")
            terms.append(((int(u),), -0.25))
            terms.append(((int(ud),), -0.25))
        return terms

    def _single_probability_terms(self, prep_index: int, party: str, y: int, b: int) -> list[tuple[int, int, int, float]]:
        assert self.template is not None
        out: list[tuple[int, int, int, float]] = []
        for word, coeff in self._effect_affine_words(party, y, b):
            entry = self.template.entry_for_row_right_word(0, word)
            if entry is None:
                continue
            out.append((prep_index, int(entry[0]), int(entry[1]), float(coeff)))
        return out

    def _joint_probability_terms(self, prep_index: int, y: int, b: int, e: int) -> list[tuple[int, int, int, float]]:
        assert self.template is not None
        out: list[tuple[int, int, int, float]] = []
        b_terms = self._effect_affine_words("B", y, b)
        e_terms = self._effect_affine_words("E", y, e)
        for b_word, b_coeff in b_terms:
            for e_word, e_coeff in e_terms:
                entry = self.template.entry_for_row_right_word(0, b_word + e_word)
                if entry is None:
                    continue
                out.append((prep_index, int(entry[0]), int(entry[1]), float(b_coeff * e_coeff)))
        return out

    def _normalize_key_selection(
        self,
        key_selection_function: Callable[[int, int], int] | Sequence[Sequence[int]] | np.ndarray | None,
    ) -> np.ndarray:
        shape = (self.scenario.X_cardinality, self.scenario.Y_cardinality)
        if key_selection_function is None:
            return np.asarray(self.scenario.key_selection_by_xy, dtype=int).reshape(shape)
        if callable(key_selection_function):
            table = np.empty(shape, dtype=int)
            for x in range(shape[0]):
                for y in range(shape[1]):
                    table[x, y] = int(key_selection_function(x, y))
            return table
        table = np.asarray(key_selection_function, dtype=int)
        if table.shape != shape:
            raise ValueError(f"key_selection must have shape {shape}.")
        return table

    def _normalize_where_key(
        self,
        where_key: Sequence[Sequence[int]] | None,
    ) -> tuple[tuple[int, ...], ...]:
        num_x = int(self.scenario.X_cardinality)
        num_y = int(self.scenario.Y_cardinality)
        if where_key is None:
            full_row = tuple(range(num_x))
            return tuple(full_row for _ in range(num_y))
        rows_raw = [list(row) for row in where_key]
        if len(rows_raw) != num_y:
            raise ValueError(f"where_key must have {num_y} rows.")
        rows: list[tuple[int, ...]] = []
        for y, row in enumerate(rows_raw):
            row_int = [int(x) for x in row]
            if any(x < 0 or x >= num_x for x in row_int):
                raise ValueError(f"where_key[{y}] contains an out-of-range x index.")
            rows.append(tuple(sorted(set(row_int))))
        return tuple(rows)

    def _other_party_uncertainty(self) -> float:
        if self.key_selection_table is None:
            self.key_selection_table = self._normalize_key_selection(None)
        if self.master_key_holder == "Bob":
            return self._alice_uncertainty_bob_key_weighted()
        return self._bob_uncertainty_key_weighted(self.key_selection_table)

    def _alice_uncertainty_bob_key_weighted(self) -> float:
        if self.objective_pair_count == 0:
            return float("nan")
        total = 0.0
        data = self.scenario.data_numeric
        b_counts = self.scenario.b_cardinality_per_y
        for y, row in enumerate(self.where_key):
            for x in row:
                b_count = int(b_counts[y])
                total += ContextualityScenario._shannon_entropy(data[int(x), y, :b_count], atol=self.atol)
        return float(total / float(self.objective_pair_count))

    def _bob_uncertainty_key_weighted(self, key_table: np.ndarray) -> float:
        if self.objective_pair_count == 0:
            return float("nan")
        total = 0.0
        data = self.scenario.data_numeric
        b_counts = self.scenario.b_cardinality_per_y
        for y, row in enumerate(self.where_key):
            if len(row) == 0:
                continue
            b_count = int(b_counts[y])
            joint = np.zeros((b_count, b_count), dtype=float)
            for x in row:
                key = int(key_table[int(x), y])
                joint[:, key] += data[int(x), y, :b_count] / float(len(row))
            b_marginal = joint.sum(axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cond = np.divide(
                    joint,
                    b_marginal[:, np.newaxis],
                    out=np.zeros_like(joint),
                    where=b_marginal[:, np.newaxis] > 0.0,
                )
            positive = joint > 0.0
            total += float(-np.sum(joint[positive] * np.log2(cond[positive]))) * float(len(row))
        return float(total / float(self.objective_pair_count))


    def _build_mosek_task(self, task: mosek.Task) -> None:
        assert self.template is not None
        dim = int(self.template.real_dimension)

        expanded_rows: list[tuple[str, float, list[tuple[int, int, int, float]]]] = []
        for constraint in self.constraints:
            expanded_rows.extend(self._expand_constraint_rows(constraint))

        task.appendbarvars([dim] * self.scenario.X_cardinality)
        task.appendafes(len(expanded_rows))

        bar_coeffs: dict[tuple[int, int, int, int], float] = {}
        g = np.empty((len(expanded_rows),), dtype=float)
        for row_index, (domain, rhs, terms) in enumerate(expanded_rows):
            g[row_index] = -float(rhs)
            for prep, row, col, coeff in terms:
                k, l, value = self._bar_triplet_entry(row, col, coeff)
                key = (row_index, int(prep), k, l)
                bar_coeffs[key] = bar_coeffs.get(key, 0.0) + value

        if bar_coeffs:
            afeidx: list[int] = []
            barvaridx: list[int] = []
            subk: list[int] = []
            subl: list[int] = []
            valkl: list[float] = []
            for (afe, bar, k, l), value in sorted(bar_coeffs.items()):
                if abs(value) <= self.atol:
                    continue
                afeidx.append(afe)
                barvaridx.append(bar)
                subk.append(k)
                subl.append(l)
                valkl.append(float(value))
            if afeidx:
                task.putafebarfblocktriplet(afeidx, barvaridx, subk, subl, valkl)
        if g.size:
            task.putafegslice(0, int(g.size), g.tolist())
            zero_domain = task.appendrzerodomain(1)
            plus_domain = task.appendrplusdomain(1)
            for row_index, (domain, _, _) in enumerate(expanded_rows):
                if domain == "zero":
                    task.appendaccseq(zero_domain, row_index, None)
                elif domain == "plus":
                    task.appendaccseq(plus_domain, row_index, None)
                else:
                    raise ValueError(f"Unknown affine constraint domain: {domain}")

        obj_coeffs: dict[tuple[int, int, int], float] = {}
        for prep, row, col, coeff in self.objective_terms:
            for emb_prep, emb_row, emb_col, emb_coeff in self._expand_complex_term_real_part(prep, row, col, complex(coeff)):
                k, l, value = self._bar_triplet_entry(emb_row, emb_col, emb_coeff)
                key = (int(emb_prep), k, l)
                obj_coeffs[key] = obj_coeffs.get(key, 0.0) + value
        if obj_coeffs:
            obj_j: list[int] = []
            obj_k: list[int] = []
            obj_l: list[int] = []
            obj_v: list[float] = []
            for (bar, k, l), value in sorted(obj_coeffs.items()):
                if abs(value) <= self.atol:
                    continue
                obj_j.append(bar)
                obj_k.append(k)
                obj_l.append(l)
                obj_v.append(float(value))
            task.putbarcblocktriplet(obj_j, obj_k, obj_l, obj_v)

        if self.threads is not None and self.threads > 0:
            task.putintparam(mosek.iparam.num_threads, int(self.threads))
        task.putobjsense(mosek.objsense.maximize)

    @staticmethod
    def _bar_triplet_entry(row: int, col: int, coeff: float) -> tuple[int, int, float]:
        k = int(max(row, col))
        l = int(min(row, col))
        value = float(coeff)
        if k != l:
            value *= 0.5
        return k, l, value

    def _expand_complex_term_real_part(
        self,
        prep: int,
        row: int,
        col: int,
        coeff: complex,
    ) -> list[tuple[int, int, int, float]]:
        assert self.template is not None
        d = int(self.template.dimension)
        a = float(np.real(coeff))
        b = float(np.imag(coeff))
        out: list[tuple[int, int, int, float]] = []
        if abs(a) > self.atol:
            out.append((int(prep), int(row), int(col), a))
        if abs(b) > self.atol:
            out.append((int(prep), d + int(row), int(col), -b))
        return out

    def _expand_complex_term_imag_part(
        self,
        prep: int,
        row: int,
        col: int,
        coeff: complex,
    ) -> list[tuple[int, int, int, float]]:
        assert self.template is not None
        d = int(self.template.dimension)
        a = float(np.real(coeff))
        b = float(np.imag(coeff))
        out: list[tuple[int, int, int, float]] = []
        if abs(a) > self.atol:
            out.append((int(prep), d + int(row), int(col), a))
        if abs(b) > self.atol:
            out.append((int(prep), int(row), int(col), b))
        return out

    def _expand_constraint_rows(
        self,
        constraint: LinearMomentConstraint,
    ) -> list[tuple[str, float, list[tuple[int, int, int, float]]]]:
        rows: list[tuple[str, float, list[tuple[int, int, int, float]]]] = []
        if constraint.domain == "zero_complex":
            real_terms: list[tuple[int, int, int, float]] = []
            imag_terms: list[tuple[int, int, int, float]] = []
            for prep, row, col, coeff in constraint.terms:
                real_terms.extend(self._expand_complex_term_real_part(prep, row, col, complex(coeff)))
                imag_terms.extend(self._expand_complex_term_imag_part(prep, row, col, complex(coeff)))
            rows.append(("zero", float(np.real(constraint.rhs)), real_terms))
            rows.append(("zero", float(np.imag(constraint.rhs)), imag_terms))
            return rows
        if constraint.domain == "zero_real":
            real_terms: list[tuple[int, int, int, float]] = []
            for prep, row, col, coeff in constraint.terms:
                real_terms.extend(self._expand_complex_term_real_part(prep, row, col, complex(coeff)))
            rows.append(("zero", float(np.real(constraint.rhs)), real_terms))
            return rows
        if constraint.domain == "zero_imag":
            imag_terms: list[tuple[int, int, int, float]] = []
            for prep, row, col, coeff in constraint.terms:
                imag_terms.extend(self._expand_complex_term_imag_part(prep, row, col, complex(coeff)))
            rows.append(("zero", float(np.imag(constraint.rhs)), imag_terms))
            return rows
        if constraint.domain == "plus_real":
            real_terms: list[tuple[int, int, int, float]] = []
            for prep, row, col, coeff in constraint.terms:
                real_terms.extend(self._expand_complex_term_real_part(prep, row, col, complex(coeff)))
            rows.append(("plus", float(np.real(constraint.rhs)), real_terms))
            return rows
        raise ValueError(f"Unknown constraint domain: {constraint.domain}")

    def _extract_solution_matrices_real(self, task: mosek.Task) -> list[np.ndarray]:
        assert self.template is not None
        dim = int(self.template.real_dimension)
        packed_size = dim * (dim + 1) // 2
        matrices: list[np.ndarray] = []
        for x in range(self.scenario.X_cardinality):
            packed = [0.0] * packed_size
            task.getbarxj(mosek.soltype.itr, x, packed)
            matrix = np.zeros((dim, dim), dtype=float)
            index = 0
            for col in range(dim):
                for row in range(col, dim):
                    matrix[row, col] = float(packed[index])
                    matrix[col, row] = float(packed[index])
                    index += 1
            matrices.append(matrix)
        return matrices

    @staticmethod
    def _complex_to_real_block(matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=complex)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
            raise ValueError("matrix must be square.")
        real = np.real(arr)
        imag = np.imag(arr)
        return np.block([[real, -imag], [imag, real]])

    @staticmethod
    def _real_block_to_complex(matrix: np.ndarray) -> np.ndarray:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim != 2 or arr.shape[0] != arr.shape[1] or arr.shape[0] % 2 != 0:
            raise ValueError("embedded real matrix must be square with even dimension.")
        d = arr.shape[0] // 2
        real = np.asarray(arr[:d, :d], dtype=float)
        imag = np.asarray(arr[d:, :d], dtype=float)
        return np.asarray(real + 1j * imag, dtype=complex)

    @staticmethod
    def _get_optimal_primal_objective(task: mosek.Task, termination_code: object | None = None) -> float:
        acceptable = {mosek.solsta.optimal}
        statuses: list[str] = []
        for soltype in (mosek.soltype.itr, mosek.soltype.bas):
            try:
                solsta = task.getsolsta(soltype)
            except mosek.Error:
                statuses.append(f"{soltype}: unavailable")
                continue
            statuses.append(f"{soltype}: {solsta}")
            if solsta in acceptable:
                return float(task.getprimalobj(soltype))
        trm = f" termination={termination_code}." if termination_code is not None else ""
        raise RuntimeError(f"SDP solve failed: MOSEK did not return an optimal solution.{trm} {' | '.join(statuses)}")

    @staticmethod
    def _assert_mosek_solution_is_not_infeasible_or_unbounded(
        task: mosek.Task,
        *,
        termination_code: object | None = None,
        failure_context: str = "SDP",
    ) -> None:
        """Assert that MOSEK did not certify infeasibility/unboundedness."""
        bad_prosta_tokens = ("prim_infeas", "dual_infeas", "prim_and_dual_infeas", "ill_posed")
        status_report: list[str] = []
        for soltype in (mosek.soltype.itr, mosek.soltype.bas):
            try:
                solsta = task.getsolsta(soltype)
            except mosek.Error:
                status_report.append(f"{soltype}: solsta unavailable")
                continue
            try:
                prosta = task.getprosta(soltype)
            except mosek.Error:
                prosta = "unavailable"
            status_report.append(f"{soltype}: solsta={solsta}, prosta={prosta}")
            prosta_str = str(prosta).lower()
            if any(token in prosta_str for token in bad_prosta_tokens):
                trm = f" termination={termination_code}." if termination_code is not None else ""
                raise AssertionError(
                    f"{failure_context} returned an invalid MOSEK problem status ({prosta}).{trm} "
                    f"statuses: {' | '.join(status_report)}"
                )


    @staticmethod
    def _binary_entropy(probability: float, atol: float = 1e-12) -> float:
        p = min(max(float(probability), 0.0), 1.0)
        if p <= atol or p >= 1.0 - atol:
            return 0.0
        return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))

    @staticmethod
    def _reverse_fano_bound(p_guess: float) -> float:
        p = float(p_guess)
        if p <= 0.0:
            raise ValueError("p_guess must be strictly positive.")
        p_eff = min(p, 1.0)
        f = math.floor(1 / p_eff)
        c = f + 1
        return float((c * p_eff - 1) * f * math.log2(f) + (1 - f * p_eff) * c * math.log2(c))


def _shannon_entropy(probabilities: np.ndarray | Sequence[float], atol: float = 1e-9) -> float:
    """Return Shannon entropy in bits for a probability vector."""

    return ContextualityScenario._shannon_entropy(np.asarray(probabilities, dtype=float), atol=atol)


__all__ = [
    "MomentMatrixTemplate",
    "MomentMatrix",
    "QKDNoncontextualSDP",
    "_shannon_entropy",
]
