"""Semi-device-independent Eve SDP relaxations for Bob-outcome scenarios.

The implementation builds one real symmetric moment matrix per preparation.
Moment entries are constrained by observed Bob data, preparation and Bob
measurement operational equivalences, and Bob/Eve marginal consistency.  The
SDP maximizes Eve's probability of guessing a selected key outcome.
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

from inflation.sdp.fast_npa import (  # noqa: E402
    commutation_matrix,
    nb_is_physical,
    nb_lexmon_to_canonical,
    remove_projector_squares,
)

from .scenario import ContextualityScenario


_IDENTITY: tuple[int, ...] = ()
_ZERO_MONOMIAL: None = None


@dataclass(frozen=True, order=True)
class Operator:
    """One Bob or Eve measurement effect."""

    party: str
    setting: int
    outcome: int
    lex_index: int

    @property
    def party_index(self) -> int:
        return 1 if self.party == "B" else 2

    def as_fast_npa_row(self) -> list[int]:
        # [party, source-copy, setting, outcome] matches fast_npa's convention
        # that the last two columns are setting and outcome labels.
        return [self.party_index, 1, self.setting + 1, self.outcome + 1]


@dataclass
class LinearMomentConstraint:
    """A scalar affine equality in moment-matrix entries."""

    terms: list[tuple[int, int, int, float]]
    rhs: float = 0.0
    name: str = ""
    domain: str = "zero"


@dataclass
class MomentMatrixTemplate:
    """Template shared by all preparation-indexed moment matrices."""

    operator_sequence: list[tuple[int, ...]]
    operators: list[Operator]
    notcomm: np.ndarray
    projective_bob: bool = False
    projective_eve: bool = False
    row_mapping: dict[int, tuple[int, ...]] = field(init=False)
    col_mapping: dict[int, tuple[int, ...]] = field(init=False)
    measurement_column_constraints: list[LinearMomentConstraint] = field(default_factory=list)
    entry_labels: dict[tuple[int, int], tuple[int, ...] | None] = field(init=False)
    entry_representatives: dict[tuple[int, ...], tuple[int, int]] = field(init=False)
    zero_entries: list[tuple[int, int]] = field(init=False)
    consistency_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = field(init=False)

    def __post_init__(self) -> None:
        self.row_mapping = {i: word for i, word in enumerate(self.operator_sequence)}
        self.col_mapping = dict(self.row_mapping)
        self.entry_labels = {}
        self.entry_representatives = {}
        self.zero_entries = []
        self.consistency_pairs = []
        self._build_entry_consistency()

    @property
    def dimension(self) -> int:
        return len(self.operator_sequence)

    def generate_sdp_variable(self, prep_index: int) -> "MomentMatrix":
        """Create a preparation-specific matrix placeholder."""

        return MomentMatrix.from_template(self, prep_index=prep_index)

    def apply_operator_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        """Return normalization, zero, and projective-idempotency constraints."""

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

        for op in self.operators:
            if not self._is_projective_operator(op):
                continue
            op_word = (op.lex_index,)
            op_idx = self.word_index(op_word)
            if op_idx is None:
                continue
            constraints.append(
                LinearMomentConstraint(
                    terms=[
                        (prep_index, op_idx, op_idx, 1.0),
                        (prep_index, 0, op_idx, -1.0),
                    ],
                    rhs=0.0,
                    name=f"idempotent_x{prep_index}_{op.party}{op.setting}_{op.outcome}",
                )
            )
        return constraints

    def apply_measurement_opeq_constraints(
        self,
        prep_index: int,
        opeq_meas: np.ndarray,
        *,
        party: str = "B",
    ) -> list[LinearMomentConstraint]:
        """Apply OEM equalities as column-level moment constraints."""

        constraints: list[LinearMomentConstraint] = []
        coeffs_arr = np.asarray(opeq_meas, dtype=float)
        if coeffs_arr.ndim == 2:
            coeffs_arr = coeffs_arr[np.newaxis, :, :]

        for k, coeffs in enumerate(coeffs_arr):
            nz_y, nz_b = np.nonzero(np.abs(coeffs) > 0.0)
            if nz_y.size == 0:
                continue
            for row_index in range(self.dimension):
                terms: list[tuple[int, int, int, float]] = []
                row_word = self.operator_sequence[row_index]
                for y, b in zip(nz_y.tolist(), nz_b.tolist()):
                    op_idx = self.find_operator(party, y, b)
                    if op_idx is None:
                        continue
                    col_index = self.word_index((op_idx,))
                    if col_index is None:
                        continue
                    terms.append((prep_index, row_index, col_index, float(coeffs[y, b])))
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
        try:
            return self.operator_sequence.index(canonical)
        except ValueError:
            return None

    def find_operator(self, party: str, setting: int, outcome: int) -> int | None:
        for op in self.operators:
            if op.party == party and op.setting == setting and op.outcome == outcome:
                return op.lex_index
        return None

    def canonical_word(self, word: Iterable[int]) -> tuple[int, ...] | None:
        lexmon = np.asarray(tuple(int(v) for v in word), dtype=np.intc)
        if lexmon.size == 0:
            return _IDENTITY
        canonical = tuple(int(v) for v in np.asarray(nb_lexmon_to_canonical(lexmon, self.notcomm), dtype=int))
        return self._apply_projector_rules(canonical)

    def _build_entry_consistency(self) -> None:
        for row, row_word in enumerate(self.operator_sequence):
            for col, col_word in enumerate(self.operator_sequence[: row + 1]):
                label = self.canonical_word(tuple(reversed(row_word)) + col_word)
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

    def _apply_projector_rules(self, word: tuple[int, ...]) -> tuple[int, ...] | None:
        if not word:
            return _IDENTITY

        out: list[int] = []
        previous: Operator | None = None
        for lex_index in word:
            op = self.operators[int(lex_index)]
            if previous is not None and self._is_projective_operator(op):
                if (
                    previous.party == op.party
                    and previous.setting == op.setting
                    and previous.outcome != op.outcome
                ):
                    return _ZERO_MONOMIAL
                if previous == op:
                    continue
            out.append(int(lex_index))
            previous = op

        if out:
            rows = np.asarray([self.operators[i].as_fast_npa_row() for i in out], dtype=np.intc)
            if self.projective_bob or self.projective_eve:
                rows = remove_projector_squares(rows)
            # ``nb_is_physical`` is a positivity classifier, not a validity
            # test for all moment entries. We still call it here so fast_npa's
            # representation is exercised without discarding noncommuting
            # same-party words that belong in an NPA relaxation.
            _ = nb_is_physical(rows, sandwich_positivity=True)
            recovered = [
                self.find_operator("B" if int(row[0]) == 1 else "E", int(row[-2]) - 1, int(row[-1]) - 1)
                for row in rows
            ]
            if any(item is None for item in recovered):
                return _ZERO_MONOMIAL
            out = [int(v) for v in recovered]
        return tuple(int(v) for v in out)

    def _is_projective_operator(self, op: Operator) -> bool:
        return (op.party == "B" and self.projective_bob) or (op.party == "E" and self.projective_eve)


@dataclass
class MomentMatrix(MomentMatrixTemplate):
    """Moment matrix instantiated for one preparation."""

    prep_index: int = 0
    sdp_var: int | None = None

    @classmethod
    def from_template(cls, template: MomentMatrixTemplate, *, prep_index: int) -> "MomentMatrix":
        matrix = cls(
            operator_sequence=list(template.operator_sequence),
            operators=list(template.operators),
            notcomm=np.asarray(template.notcomm, dtype=bool),
            projective_bob=template.projective_bob,
            projective_eve=template.projective_eve,
            measurement_column_constraints=[],
            prep_index=int(prep_index),
        )
        matrix.entry_labels = template.entry_labels
        matrix.entry_representatives = template.entry_representatives
        matrix.zero_entries = template.zero_entries
        matrix.consistency_pairs = template.consistency_pairs
        return matrix

    def apply_prep_opeq_constraints(
        self,
        prep_index: int,
        alpha_coeffs: np.ndarray,
    ) -> list[LinearMomentConstraint]:
        """Compatibility hook; preparation OPEQs are assembled globally."""

        _ = prep_index, alpha_coeffs
        return []


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

    def build_operator_list(self) -> list[Operator]:
        """Create Bob and Eve effect operators for all valid scenario outcomes."""

        operators: list[Operator] = []
        b_counts = self.scenario.b_cardinality_per_y.astype(int, copy=False)
        for party in ("B", "E"):
            for y in range(self.scenario.Y_cardinality):
                for b in range(int(b_counts[y])):
                    operators.append(Operator(party=party, setting=y, outcome=b, lex_index=len(operators)))
        self.operators = operators
        return list(operators)

    def build_lexorder_and_notcomm(self) -> tuple[np.ndarray, np.ndarray]:
        """Build fast_npa lexicographic order and noncommutation table."""

        if not self.operators:
            self.build_operator_list()
        self.lexorder = np.asarray([op.as_fast_npa_row() for op in self.operators], dtype=np.intc)
        sources_to_check = np.full((2, 2, 1), 2, dtype=np.uint8)
        self.notcomm = np.asarray(
            commutation_matrix(self.lexorder, sources_to_check, commuting=False),
            dtype=bool,
        )
        return self.lexorder.copy(), self.notcomm.copy()

    def instantiate_moment_matrices(self) -> list[MomentMatrix]:
        """Build the template and one matrix placeholder per preparation."""

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
            packed = dim * (dim + 1) // 2
            self._log(
                1,
                f"[eve_sdp] moment matrices: {len(self.moment_matrices)} blocks, each {dim}x{dim} "
                f"(symmetric packed={packed})",
            )
        return list(self.moment_matrices)

    def apply_observed_data_constraints(self) -> list[LinearMomentConstraint]:
        """Assemble all affine constraints for the SDP."""

        if self.template is None or not self.moment_matrices:
            self.instantiate_moment_matrices()
        assert self.template is not None

        constraints: list[LinearMomentConstraint] = []
        for matrix in self.moment_matrices:
            x = matrix.prep_index
            constraints.extend(self.template.apply_operator_constraints(x))
            constraints.extend(self._moment_consistency_constraints(x))
            constraints.extend(self.template.apply_measurement_opeq_constraints(x, self.scenario.opeq_meas_numeric, party="B"))
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
        """Set the objective to maximize Eve's average correct key guess."""

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
            for x in row:
                if self.master_key_holder == "Alice":
                    e = int(key_table[x, y])
                    if e < 0 or e >= int(self.scenario.b_cardinality_per_y[y]):
                        raise ValueError(f"key_selection[{x},{y}] is not a valid outcome for y={y}.")
                    op_idx = self.template.find_operator("E", y, e)
                    col = self.template.word_index((op_idx,)) if op_idx is not None else None
                    if col is None:
                        raise RuntimeError("Eve key operator is unavailable in the moment template.")
                    terms.append((x, 0, col, weight))
                    continue

                for b in range(int(self.scenario.b_cardinality_per_y[y])):
                    b_op = self.template.find_operator("B", y, b)
                    e_op = self.template.find_operator("E", y, b)
                    col = self.template.word_index((b_op, e_op)) if b_op is not None and e_op is not None else None
                    if col is None:
                        raise RuntimeError("Bob-Eve joint operator is unavailable in the moment template.")
                    terms.append((x, 0, col, weight))
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
        """Build and solve the SDP, returning Eve's success-probability bound."""

        if not self.constraints:
            self.apply_observed_data_constraints()
        if key_selection_function is not None or master_key_holder is not None or where_key is not None or not self.objective_terms:
            self.set_objective(
                key_selection_function,
                master_key_holder=master_key_holder,
                where_key=where_key,
            )

        assert self.template is not None
        with mosek.Env() as env:
            with env.Task(0, 0) as task:
                if self.verbose >= 2:
                    task.set_Stream(mosek.streamtype.log, lambda msg: print(msg, end=""))
                self._build_mosek_task(task)
                self._log(
                    1,
                    f"[eve_sdp] solving with {len(self.constraints)} affine constraints and "
                    f"{self.scenario.X_cardinality} PSD blocks",
                )
                termination_code = task.optimize()
                objective = self._get_optimal_primal_objective(task, termination_code)
                self.eve_success_probability = float(objective)
                self.solution_matrices = self._extract_solution_matrices(task)
        self.key_rate_lower_bound = self.compute_key_rate(rate_type="reverse_fano")
        return float(self.eve_success_probability)

    def solve(
        self,
        key_selection_function: Callable[[int, int], int] | Sequence[Sequence[int]] | np.ndarray | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        where_key: Sequence[Sequence[int]] | None = None,
    ) -> float:
        """Alias for :meth:`solve_sdp` matching the public API example."""

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
        """Compute the key-rate lower bound from the solved Eve bound."""

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
        bob = [op.lex_index for op in self.operators if op.party == "B"]
        eve = [op.lex_index for op in self.operators if op.party == "E"]
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

    def _measurement_completeness_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        assert self.template is not None
        constraints: list[LinearMomentConstraint] = []
        for party in ("B", "E"):
            for y in range(self.scenario.Y_cardinality):
                terms = []
                for b in range(int(self.scenario.b_cardinality_per_y[y])):
                    op_idx = self.template.find_operator(party, y, b)
                    col = self.template.word_index((op_idx,)) if op_idx is not None else None
                    if col is not None:
                        terms.append((prep_index, 0, col, 1.0))
                constraints.append(
                    LinearMomentConstraint(
                        terms=terms,
                        rhs=1.0,
                        name=f"complete_{party}_x{prep_index}_y{y}",
                    )
                )
        return constraints

    def _observed_bob_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        assert self.template is not None
        constraints: list[LinearMomentConstraint] = []
        data = self.scenario.data_numeric
        for y in range(self.scenario.Y_cardinality):
            for b in range(int(self.scenario.b_cardinality_per_y[y])):
                op_idx = self.template.find_operator("B", y, b)
                col = self.template.word_index((op_idx,)) if op_idx is not None else None
                if col is None:
                    continue
                constraints.append(
                    LinearMomentConstraint(
                        terms=[(prep_index, 0, col, 1.0)],
                        rhs=float(data[prep_index, y, b]),
                        name=f"data_x{prep_index}_y{y}_b{b}",
                    )
                )
        return constraints

    def _bob_eve_marginal_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        assert self.template is not None
        constraints: list[LinearMomentConstraint] = []
        for y in range(self.scenario.Y_cardinality):
            b_count = int(self.scenario.b_cardinality_per_y[y])
            for b in range(b_count):
                b_op = self.template.find_operator("B", y, b)
                b_col = self.template.word_index((b_op,)) if b_op is not None else None
                if b_col is None:
                    continue
                terms = [(prep_index, 0, b_col, -1.0)]
                for e in range(b_count):
                    e_op = self.template.find_operator("E", y, e)
                    joint_col = self.template.word_index((b_op, e_op)) if e_op is not None else None
                    if joint_col is not None:
                        terms.append((prep_index, 0, joint_col, 1.0))
                constraints.append(
                    LinearMomentConstraint(terms=terms, rhs=0.0, name=f"joint_to_bob_x{prep_index}_y{y}_b{b}")
                )

            for e in range(b_count):
                e_op = self.template.find_operator("E", y, e)
                e_col = self.template.word_index((e_op,)) if e_op is not None else None
                if e_col is None:
                    continue
                terms = [(prep_index, 0, e_col, -1.0)]
                for b in range(b_count):
                    b_op = self.template.find_operator("B", y, b)
                    joint_col = self.template.word_index((b_op, e_op)) if b_op is not None else None
                    if joint_col is not None:
                        terms.append((prep_index, 0, joint_col, 1.0))
                constraints.append(
                    LinearMomentConstraint(terms=terms, rhs=0.0, name=f"joint_to_eve_x{prep_index}_y{y}_e{e}")
                )
        return constraints

    def _probability_positivity_constraints(self, prep_index: int) -> list[LinearMomentConstraint]:
        assert self.template is not None
        constraints: list[LinearMomentConstraint] = []

        def add_probability_bounds(col: int, name: str) -> None:
            constraints.append(
                LinearMomentConstraint(
                    terms=[(prep_index, 0, col, 1.0)],
                    rhs=0.0,
                    name=f"{name}_lower",
                    domain="plus",
                )
            )
            constraints.append(
                LinearMomentConstraint(
                    terms=[(prep_index, 0, col, -1.0)],
                    rhs=-1.0,
                    name=f"{name}_upper",
                    domain="plus",
                )
            )

        for party in ("B", "E"):
            for y in range(self.scenario.Y_cardinality):
                for b in range(int(self.scenario.b_cardinality_per_y[y])):
                    op_idx = self.template.find_operator(party, y, b)
                    col = self.template.word_index((op_idx,)) if op_idx is not None else None
                    if col is not None:
                        add_probability_bounds(col, f"prob_{party}_x{prep_index}_y{y}_b{b}")

        for y in range(self.scenario.Y_cardinality):
            count = int(self.scenario.b_cardinality_per_y[y])
            for b in range(count):
                b_op = self.template.find_operator("B", y, b)
                for e in range(count):
                    e_op = self.template.find_operator("E", y, e)
                    col = self.template.word_index((b_op, e_op)) if b_op is not None and e_op is not None else None
                    if col is not None:
                        add_probability_bounds(col, f"joint_x{prep_index}_y{y}_b{b}_e{e}")

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
        dim = self.template.dimension
        task.appendbarvars([dim] * self.scenario.X_cardinality)
        task.appendafes(len(self.constraints))

        bar_coeffs: dict[tuple[int, int, int, int], float] = {}
        g = np.empty((len(self.constraints),), dtype=float)
        for row_index, constraint in enumerate(self.constraints):
            g[row_index] = -float(constraint.rhs)
            for prep, row, col, coeff in constraint.terms:
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
            for row_index, constraint in enumerate(self.constraints):
                if constraint.domain == "zero":
                    task.appendaccseq(zero_domain, row_index, None)
                elif constraint.domain == "plus":
                    task.appendaccseq(plus_domain, row_index, None)
                else:
                    raise ValueError(f"Unknown affine constraint domain: {constraint.domain}")

        obj_coeffs: dict[tuple[int, int, int], float] = {}
        for prep, row, col, coeff in self.objective_terms:
            k, l, value = self._bar_triplet_entry(row, col, coeff)
            key = (int(prep), k, l)
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

    def _extract_solution_matrices(self, task: mosek.Task) -> list[np.ndarray]:
        assert self.template is not None
        dim = self.template.dimension
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

    def _bob_key_success_probability(self, key_table: np.ndarray) -> float:
        total = 0.0
        count = 0
        data = self.scenario.data_numeric
        for x in range(self.scenario.X_cardinality):
            for y in range(self.scenario.Y_cardinality):
                total += float(data[x, y, int(key_table[x, y])])
                count += 1
        return float(total / float(count))

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
