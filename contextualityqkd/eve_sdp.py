"""Readable CVXPY SDP backend for contextuality-QKD Eve bounds.

Nonprojective measurements follow Appendix B's Naimark-unitary construction:
- Explicit generators U_k^y and (U_k^y)^dagger for k in [0, K_y-2]
- Last outcome eliminated via completeness identities (Eqs. (11)-(12))
- Data constraints in Eq. (8) form
- Measurement OPEQ constraints in Eq. (10) form

The optimization model is assembled with CVXPY and solved with MOSEK by
default.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from itertools import product
from pathlib import Path
import sys
from typing import Callable, Iterable, Literal, Sequence

import cvxpy as cp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_VENDORED_INFLATION_ROOT = _REPO_ROOT / "external" / "inflation"
if _VENDORED_INFLATION_ROOT.exists() and str(_VENDORED_INFLATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDORED_INFLATION_ROOT))

from inflation.sdp.fast_npa import nb_lexmon_to_canonical  # noqa: E402

from .cvxpy_utils import (
    DualKey,
    add_named_constraint,
    assert_cvxpy_solution_is_optimal,
    is_mosek_solver,
    snapshot_dual_values,
    solve_cvxpy_problem_preserving_duals,
)
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
        complex_moments: bool = True,
        solver: str | None = None,
        threads: int | None = None,
        atol: float | None = None,
        verbose: int | bool = 0,
        data_constraint: Literal["full", "witness"] = "full",
        witness_coeffs: np.ndarray | None = None,
        witness_bound: float | None = None,
        witness_sense: Literal[">=", "<="] = ">=",
    ) -> None:
        if not isinstance(scenario, ContextualityScenario):
            raise TypeError("scenario must be a ContextualityScenario instance.")
        self.scenario = scenario
        # Data-constraint mode. "full" pins Bob's observed marginal entrywise;
        # "witness" keeps only a single linear lower bound on the behavior
        # (outcome normalization is built into the Naimark completeness
        # identities, so no extra normalization constraint is needed).
        self.data_constraint = str(data_constraint).strip().lower()
        if self.data_constraint not in {"full", "witness"}:
            raise ValueError("data_constraint must be 'full' or 'witness'.")
        if self.data_constraint == "witness":
            if witness_coeffs is None or witness_bound is None:
                raise ValueError("witness mode requires witness_coeffs and witness_bound.")
            self.witness_coeffs: np.ndarray | None = np.asarray(witness_coeffs, dtype=float)
            expected = (
                int(scenario.X_cardinality),
                int(scenario.Y_cardinality),
                int(scenario.B_cardinality),
            )
            if self.witness_coeffs.shape != expected:
                raise ValueError(f"witness_coeffs must have shape {expected}.")
            self.witness_bound: float | None = float(witness_bound)
            sense = str(witness_sense).strip()
            if sense not in {">=", "<="}:
                raise ValueError("witness_sense must be '>=' or '<='.")
            self.witness_sense: str = sense
        else:
            if witness_coeffs is not None or witness_bound is not None:
                raise ValueError("witness_coeffs/witness_bound are only valid with data_constraint='witness'.")
            self.witness_coeffs = None
            self.witness_bound = None
            self.witness_sense = ">="
        self.projective_bob = bool(projective_bob)
        self.projective_eve = bool(projective_eve)
        self.npa_level_bob = self._validate_level(npa_level_bob, "npa_level_bob")
        self.npa_level_eve = self._validate_level(npa_level_eve, "npa_level_eve")
        self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        self.where_key = self._normalize_where_key(where_key)
        self.use_u_only = bool(use_u_only)
        self.complex_moments = bool(complex_moments)
        self.solver = cp.MOSEK if solver is None else solver
        self.threads = None if threads is None else int(threads)
        self.atol = scenario.atol if atol is None else float(atol)
        self.verbose = int(verbose)

        self.operators: list[Operator] = []
        self.lexorder: np.ndarray | None = None
        self.notcomm: np.ndarray | None = None
        self.word_sequence: list[tuple[int, ...]] = []
        self.word_to_index: dict[tuple[int, ...], int] = {}
        self.entry_labels: dict[tuple[int, int], tuple[int, ...] | None] = {}
        self.entry_representatives: dict[tuple[int, ...], tuple[int, int]] = {}
        self.zero_entries: list[tuple[int, int]] = []
        self.consistency_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        self.effect_index: dict[tuple[str, int, int], int] = {}
        self.unitary_index: dict[tuple[str, int, int, bool], int] = {}
        self.key_selection_table: np.ndarray | None = None
        self.objective_pair_count: int = 0
        self.eve_success_probability: float | None = None
        self.key_rate_lower_bound: float | None = None
        self.solution_matrices: list[np.ndarray] = []
        self.solution_matrices_real: list[np.ndarray] = []
        self.cvxpy_problem: cp.Problem | None = None
        self.cvxpy_variables: list[cp.Variable] = []
        self.cvxpy_constraints: list[cp.Constraint] = []
        self.cvxpy_objective: cp.Expression | None = None
        self.dual_constraints: dict[DualKey, cp.Constraint] = {}
        self.dual_values: dict[DualKey, object | None] = {}

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

    def set_objective(
        self,
        key_selection_function: Callable[[int, int], int] | Sequence[Sequence[int]] | np.ndarray | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        where_key: Sequence[Sequence[int]] | None = None,
    ) -> None:
        if master_key_holder is not None:
            self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        if where_key is not None:
            self.where_key = self._normalize_where_key(where_key)
        self.key_selection_table = self._normalize_key_selection(key_selection_function)
        pair_count = sum(len(row) for row in self.where_key)
        if pair_count == 0:
            raise ValueError("where_key must contain at least one key-eligible (x,y) pair.")
        self.objective_pair_count = int(pair_count)

    def solve_sdp(
        self,
        key_selection_function: Callable[[int, int], int] | Sequence[Sequence[int]] | np.ndarray | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        where_key: Sequence[Sequence[int]] | None = None,
        solver: str | None = None,
    ) -> float:
        if solver is not None:
            self.solver = solver
        self.set_objective(
            key_selection_function,
            master_key_holder=master_key_holder,
            where_key=where_key,
        )
        self._build_problem()
        assert self.cvxpy_problem is not None
        self._log(
            1,
            f"[eve_sdp] solving CVXPY SDP with {len(self.cvxpy_constraints)} constraints and "
            f"{self.scenario.X_cardinality} PSD block(s)",
        )
        solve_kwargs: dict[str, object] = {"solver": self.solver, "verbose": self.verbose >= 2}
        mosek_params: dict[str, object] = {}
        if is_mosek_solver(self.solver) and self.threads is not None and self.threads > 0:
            mosek_params["MSK_IPAR_NUM_THREADS"] = int(self.threads)
        if mosek_params:
            solve_kwargs["mosek_params"] = mosek_params

        value = solve_cvxpy_problem_preserving_duals(self.cvxpy_problem, solve_kwargs)
        assert_cvxpy_solution_is_optimal(problem=self.cvxpy_problem, value=value, problem_kind="SDP")
        self.eve_success_probability = float(value)
        self.solution_matrices = self._extract_solution_matrices(self.cvxpy_variables)
        self.solution_matrices_real = [self._complex_to_real_block(matrix) for matrix in self.solution_matrices]
        self.dual_values = snapshot_dual_values(self.dual_constraints)
        self.key_rate_lower_bound = self.compute_key_rate(rate_type="reverse_fano")
        return float(self.eve_success_probability)

    def solve(
        self,
        key_selection_function: Callable[[int, int], int] | Sequence[Sequence[int]] | np.ndarray | None = None,
        *,
        master_key_holder: Literal["Alice", "Bob"] | str | None = None,
        where_key: Sequence[Sequence[int]] | None = None,
        solver: str | None = None,
    ) -> float:
        return self.solve_sdp(
            key_selection_function=key_selection_function,
            master_key_holder=master_key_holder,
            where_key=where_key,
            solver=solver,
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
        words: list[tuple[int, ...]] = [_IDENTITY]
        seen = {_IDENTITY}

        bob_words = self._party_words(bob, self.npa_level_bob)
        eve_words = self._party_words(eve, self.npa_level_eve)
        for b_word in bob_words:
            for e_word in eve_words:
                combined = self.canonical_word(b_word + e_word)
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

    def _build_problem(self) -> None:
        self.build_lexorder_and_notcomm()
        self.word_sequence = self._build_operator_sequence()
        self._build_operator_lookup()
        self._build_entry_lookup()
        self.cvxpy_variables = self._make_cvxpy_variables()
        self.cvxpy_constraints = []
        self.dual_constraints = {}
        self.dual_values = {}
        for x, variable in enumerate(self.cvxpy_variables):
            self._add_constraint(("psd", x), variable >> 0)
        for x in range(self.scenario.X_cardinality):
            self._add_normalization_and_operator_constraints(x)
            self._add_moment_consistency_constraints(x)
            self._add_measurement_opeq_constraints(x, party="B")
            if self.data_constraint == "full":
                self._add_observed_bob_constraints(x)
            self._add_bob_eve_marginal_constraints(x)
            self._add_probability_bounds(x)
        if self.data_constraint == "witness":
            self._add_witness_bound_constraint()
        self._add_preparation_opeq_constraints()
        self.cvxpy_objective = self._build_objective_expr()
        self.cvxpy_problem = cp.Problem(cp.Maximize(self.cvxpy_objective), self.cvxpy_constraints)
        if self.verbose >= 1:
            field = "complex Hermitian" if self.complex_moments else "real symmetric"
            self._log(
                1,
                f"[eve_sdp] moment matrices: {self.scenario.X_cardinality} blocks, "
                f"{field} {self.dimension}x{self.dimension} CVXPY PSD variable(s)",
            )
        if self.verbose >= 1 and self.use_u_only:
            self._log(1, "[eve_sdp] U-only generator mode: dagger operators excluded from NPA word sequence.")

    @property
    def dimension(self) -> int:
        return len(self.word_sequence)

    @property
    def real_dimension(self) -> int:
        return 2 * int(self.dimension)

    def _build_operator_lookup(self) -> None:
        self.effect_index = {}
        self.unitary_index = {}
        for op in self.operators:
            if op.kind == "projector":
                self.effect_index[(op.party, op.setting, op.outcome)] = op.lex_index
            else:
                self.unitary_index[(op.party, op.setting, op.outcome, op.is_dagger)] = op.lex_index

    def _build_entry_lookup(self) -> None:
        self.word_to_index = {word: index for index, word in enumerate(self.word_sequence)}
        self.entry_labels = {}
        self.entry_representatives = {}
        self.zero_entries = []
        self.consistency_pairs = []
        for row, row_word in enumerate(self.word_sequence):
            row_adj = self._adjoint_word(row_word)
            for col, col_word in enumerate(self.word_sequence[: row + 1]):
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

    def _add_normalization_and_operator_constraints(self, x: int) -> None:
        self._add_complex_equality(("normalization", x), self._moment(x, 0, 0), 1.0)
        for row, col in self.zero_entries:
            self._add_complex_equality(("zero_entry", x, row, col), self._moment(x, row, col), 0.0)

        for op in self.operators:
            if not self._is_projective_operator(op):
                continue
            entry_pp = self.representative_entry_for_word((op.lex_index, op.lex_index))
            entry_p = self.representative_entry_for_word((op.lex_index,))
            if entry_pp is None or entry_p is None:
                continue
            self._add_complex_equality(
                ("projector_idempotency", x, op.party, op.setting, op.outcome),
                self._moment(x, entry_pp[0], entry_pp[1]) - self._moment(x, entry_p[0], entry_p[1]),
                0.0,
            )

        for op in self.operators:
            if self._is_projective_operator(op) or op.kind != "unitary":
                continue
            entry_diag = self.representative_entry_for_word((op.adjoint_lex_index, op.lex_index))
            entry_u = self.representative_entry_for_word((op.lex_index,))
            entry_ud = self.representative_entry_for_word((op.adjoint_lex_index,))
            if entry_diag is None or entry_u is None or entry_ud is None:
                continue
            self._add_complex_equality(
                ("unitary_diagonal", x, op.party, op.setting, op.outcome, op.is_dagger),
                self._moment(x, entry_diag[0], entry_diag[1]),
                1.0,
            )
            self._add_complex_equality(
                ("unitary_conjugacy", x, op.party, op.setting, op.outcome, op.is_dagger, "forward"),
                self._moment(x, entry_u[0], entry_u[1]) - self._moment(x, entry_ud[0], entry_ud[1]),
                0.0,
            )
            self._add_complex_equality(
                ("unitary_conjugacy", x, op.party, op.setting, op.outcome, op.is_dagger, "reverse"),
                self._moment(x, entry_ud[0], entry_ud[1]) - self._moment(x, entry_u[0], entry_u[1]),
                0.0,
            )

    def _add_moment_consistency_constraints(self, x: int) -> None:
        for (entry, rep) in self.consistency_pairs:
            self._add_complex_equality(
                ("moment_consistency", x, entry[0], entry[1], rep[0], rep[1]),
                self._moment(x, entry[0], entry[1]) - self._moment(x, rep[0], rep[1]),
                0.0,
            )

    def _add_measurement_opeq_constraints(self, x: int, *, party: str = "B") -> None:
        coeffs_arr = np.asarray(self.scenario.opeq_meas_numeric, dtype=float)
        if coeffs_arr.ndim == 2:
            coeffs_arr = coeffs_arr[np.newaxis, :, :]

        is_projective = self._party_is_projective(party)
        for k, coeffs in enumerate(coeffs_arr):
            for row_index in range(self.dimension):
                pieces: list[cp.Expression] = []
                for y in range(coeffs.shape[0]):
                    b_count = int(self.scenario.b_cardinality_per_y[y])
                    if b_count <= 0:
                        continue
                    beta_last = float(coeffs[y, b_count - 1])
                    if is_projective:
                        if abs(beta_last) > self.atol:
                            pieces.append(beta_last * self._moment(x, row_index, 0))
                        for b in range(max(0, b_count - 1)):
                            coeff = float(coeffs[y, b]) - beta_last
                            entry = self._entry_for_row_right_word(row_index, self._operator_word(party, y, b))
                            if abs(coeff) > self.atol and entry is not None:
                                pieces.append(coeff * self._moment(x, entry[0], entry[1]))
                        continue

                    beta_except = float(np.sum(coeffs[y, : max(0, b_count - 1)]))
                    id_coeff = 0.5 * beta_except + 0.5 * (3.0 - float(b_count)) * beta_last
                    if abs(id_coeff) > self.atol:
                        pieces.append(id_coeff * self._moment(x, row_index, 0))
                    for b in range(max(0, b_count - 1)):
                        eff_coeff = 0.25 * (float(coeffs[y, b]) - beta_last)
                        if abs(eff_coeff) <= self.atol:
                            continue
                        for word in (
                            self._operator_word(party, y, b, is_dagger=False),
                            self._operator_word(party, y, b, is_dagger=True),
                        ):
                            entry = self._entry_for_row_right_word(row_index, word)
                            if entry is not None:
                                pieces.append(eff_coeff * self._moment(x, entry[0], entry[1]))
                if pieces:
                    self._add_complex_equality(("measurement_opeq", party, x, k, row_index), sum(pieces), 0.0)

    def _add_observed_bob_constraints(self, x: int) -> None:
        data = self.scenario.data_numeric
        for y in range(self.scenario.Y_cardinality):
            for b in range(int(self.scenario.b_cardinality_per_y[y])):
                self._add_complex_equality(
                    ("observed_bob", x, y, b),
                    self._single_probability_expr(x, "B", y, b),
                    float(data[x, y, b]),
                )

    def _add_witness_bound_constraint(self) -> None:
        """Single inequality ``sum_{x,y,b} c[x,y,b] P(b|x,y) >= witness_bound``."""
        assert self.witness_coeffs is not None and self.witness_bound is not None
        pieces: list[cp.Expression] = []
        for x in range(self.scenario.X_cardinality):
            for y in range(self.scenario.Y_cardinality):
                for b in range(int(self.scenario.b_cardinality_per_y[y])):
                    coeff = float(self.witness_coeffs[x, y, b])
                    if abs(coeff) <= self.atol:
                        continue
                    pieces.append(coeff * self._single_probability_expr(x, "B", y, b))
        if not pieces:
            raise ValueError("witness_coeffs has no support on valid (x, y, b) entries.")
        expr = sum(pieces)
        real_expr = cp.real(expr) if self.complex_moments else expr
        if self.complex_moments:
            self._add_constraint(("witness_bound", "imag_zero"), cp.imag(expr) == 0.0)
        if self.witness_sense == ">=":
            self._add_constraint(("witness_bound", "lower"), real_expr >= float(self.witness_bound))
        else:
            self._add_constraint(("witness_bound", "upper"), real_expr <= float(self.witness_bound))

    def _add_bob_eve_marginal_constraints(self, x: int) -> None:
        for y in range(self.scenario.Y_cardinality):
            b_count = int(self.scenario.b_cardinality_per_y[y])
            for b in range(b_count):
                joint_sum = sum(self._joint_probability_expr(x, y, b, e) for e in range(b_count))
                self._add_complex_equality(
                    ("bob_eve_marginal", "bob", x, y, b),
                    joint_sum - self._single_probability_expr(x, "B", y, b),
                    0.0,
                )
            for e in range(b_count):
                joint_sum = sum(self._joint_probability_expr(x, y, b, e) for b in range(b_count))
                self._add_complex_equality(
                    ("bob_eve_marginal", "eve", x, y, e),
                    joint_sum - self._single_probability_expr(x, "E", y, e),
                    0.0,
                )

    def _add_probability_bounds(self, x: int) -> None:
        for party in ("B", "E"):
            for y in range(self.scenario.Y_cardinality):
                for b in range(int(self.scenario.b_cardinality_per_y[y])):
                    self._add_probability_bound(("probability_bound", party, x, y, b), self._single_probability_expr(x, party, y, b))
        for y in range(self.scenario.Y_cardinality):
            count = int(self.scenario.b_cardinality_per_y[y])
            for b in range(count):
                for e in range(count):
                    self._add_probability_bound(("joint_probability_bound", x, y, b, e), self._joint_probability_expr(x, y, b, e))

    def _add_preparation_opeq_constraints(self) -> None:
        opeqs = np.asarray(self.scenario.opeq_preps_numeric, dtype=float)
        if opeqs.ndim == 1:
            opeqs = opeqs[np.newaxis, :]
        for k, coeffs in enumerate(opeqs):
            nz_x = np.flatnonzero(np.abs(coeffs) > self.atol)
            if nz_x.size == 0:
                continue
            for row in range(self.dimension):
                for col in range(row + 1):
                    expr = sum(float(coeffs[x]) * self._moment(int(x), row, col) for x in nz_x.tolist())
                    self._add_complex_equality(("preparation_opeq", k, row, col), expr, 0.0)

    def _effect_words(self, party: str, y: int, b: int) -> list[tuple[tuple[int, ...], float]]:
        b_count = int(self.scenario.b_cardinality_per_y[y])
        if self._party_is_projective(party):
            if b_count <= 1:
                return [(_IDENTITY, 1.0)]
            if b < b_count - 1:
                return [(self._operator_word(party, y, b), 1.0)]
            terms: list[tuple[tuple[int, ...], float]] = [(_IDENTITY, 1.0)]
            for k in range(b_count - 1):
                terms.append((self._operator_word(party, y, k), -1.0))
            return terms

        if b_count <= 1:
            return [(_IDENTITY, 1.0)]

        terms: list[tuple[tuple[int, ...], float]] = []
        if b < b_count - 1:
            terms.append((_IDENTITY, 0.5))
            terms.append((self._operator_word(party, y, b, is_dagger=False), 0.25))
            terms.append((self._operator_word(party, y, b, is_dagger=True), 0.25))
            return terms

        terms.append((_IDENTITY, 0.5 * (3.0 - float(b_count))))
        for k in range(b_count - 1):
            terms.append((self._operator_word(party, y, k, is_dagger=False), -0.25))
            terms.append((self._operator_word(party, y, k, is_dagger=True), -0.25))
        return terms

    def _single_probability_expr(self, x: int, party: str, y: int, b: int) -> cp.Expression:
        pieces: list[cp.Expression] = []
        for word, coeff in self._effect_words(party, y, b):
            entry = self._entry_for_row_right_word(0, word)
            if entry is None:
                continue
            pieces.append(float(coeff) * self._moment(x, int(entry[0]), int(entry[1])))
        return sum(pieces) if pieces else cp.Constant(0.0)

    def _joint_probability_expr(self, x: int, y: int, b: int, e: int) -> cp.Expression:
        pieces: list[cp.Expression] = []
        b_terms = self._effect_words("B", y, b)
        e_terms = self._effect_words("E", y, e)
        for b_word, b_coeff in b_terms:
            for e_word, e_coeff in e_terms:
                entry = self._entry_for_row_right_word(0, b_word + e_word)
                if entry is None:
                    continue
                pieces.append(float(b_coeff * e_coeff) * self._moment(x, int(entry[0]), int(entry[1])))
        return sum(pieces) if pieces else cp.Constant(0.0)

    def _build_objective_expr(self) -> cp.Expression:
        assert self.key_selection_table is not None
        weight = 1.0 / float(self.objective_pair_count)
        pieces: list[cp.Expression] = []
        for y, row in enumerate(self.where_key):
            b_count = int(self.scenario.b_cardinality_per_y[y])
            for x in row:
                if self.master_key_holder == "Alice":
                    e = int(self.key_selection_table[int(x), y])
                    if e < 0 or e >= b_count:
                        raise ValueError(f"key_selection[{x},{y}] is not a valid outcome for y={y}.")
                    pieces.append(weight * self._single_probability_expr(int(x), "E", y, e))
                    continue
                for b in range(b_count):
                    pieces.append(weight * self._joint_probability_expr(int(x), y, b, b))
        expr = sum(pieces) if pieces else cp.Constant(0.0)
        return cp.real(expr) if self.complex_moments else expr

    def _moment(self, x: int, row: int, col: int) -> cp.Expression:
        return self.cvxpy_variables[int(x)][int(row), int(col)]

    def _add_constraint(self, key: DualKey, constraint: cp.Constraint) -> None:
        add_named_constraint(
            key=key,
            constraint=constraint,
            constraints=self.cvxpy_constraints,
            dual_constraints=self.dual_constraints,
        )

    def _add_complex_equality(self, key: DualKey, expr: cp.Expression, rhs: complex | float) -> None:
        rhs_complex = complex(rhs)
        if not self.complex_moments and abs(np.imag(rhs_complex)) > self.atol:
            raise ValueError("Real SDP mode cannot impose a complex RHS.")
        if self.complex_moments:
            self._add_constraint((*key, "real"), cp.real(expr) == float(np.real(rhs_complex)))
            self._add_constraint((*key, "imag"), cp.imag(expr) == float(np.imag(rhs_complex)))
        else:
            self._add_constraint((*key, "real"), expr == float(np.real(rhs_complex)))

    def _add_probability_bound(self, key: DualKey, expr: cp.Expression) -> None:
        if self.complex_moments:
            self._add_constraint((*key, "imag_zero"), cp.imag(expr) == 0.0)
            self._add_constraint((*key, "lower"), cp.real(expr) >= 0.0)
            self._add_constraint((*key, "upper"), cp.real(expr) <= 1.0)
        else:
            self._add_constraint((*key, "lower"), expr >= 0.0)
            self._add_constraint((*key, "upper"), expr <= 1.0)

    def _operator_word(
        self,
        party: str,
        setting: int,
        outcome: int,
        is_dagger: bool | None = None,
    ) -> tuple[int, ...]:
        op_idx = self.find_operator(party, setting, outcome, is_dagger=is_dagger)
        if op_idx is None:
            kind = "projector" if is_dagger is None else "unitary"
            raise RuntimeError(f"Missing {kind} for {party}, y={setting}, outcome={outcome}.")
        return (int(op_idx),)

    def find_operator(
        self,
        party: str,
        setting: int,
        outcome: int,
        is_dagger: bool | None = None,
    ) -> int | None:
        if is_dagger is None:
            return self.effect_index.get((party, setting, outcome))
        return self.unitary_index.get((party, setting, outcome, bool(is_dagger)))

    def word_index(self, word: tuple[int, ...]) -> int | None:
        canonical = self.canonical_word(word)
        if canonical is _ZERO_MONOMIAL:
            return None
        return self.word_to_index.get(canonical)

    def representative_entry_for_word(self, word: tuple[int, ...]) -> tuple[int, int] | None:
        canonical = self.canonical_word(word)
        if canonical is _ZERO_MONOMIAL:
            return None
        return self._representative_entry_for_label(canonical)

    def _entry_for_row_right_word(self, row_index: int, right_word: tuple[int, ...]) -> tuple[int, int] | None:
        row_word = self.word_sequence[int(row_index)]
        label = self.canonical_word(self._adjoint_word(row_word) + tuple(int(v) for v in right_word))
        if label is _ZERO_MONOMIAL:
            return None
        assert label is not None
        return self._representative_entry_for_label(label)

    def _representative_entry_for_label(self, label: tuple[int, ...]) -> tuple[int, int] | None:
        rep = self.entry_representatives.get(label)
        if rep is not None:
            return rep
        adj = self.canonical_word(self._adjoint_word(label))
        if adj is _ZERO_MONOMIAL or adj is None:
            return None
        return self.entry_representatives.get(adj)

    def canonical_word(self, word: Iterable[int]) -> tuple[int, ...] | None:
        lexmon = np.asarray(tuple(int(v) for v in word), dtype=np.intc)
        if lexmon.size == 0:
            return _IDENTITY
        assert self.notcomm is not None
        canonical = tuple(int(v) for v in np.asarray(nb_lexmon_to_canonical(lexmon, self.notcomm), dtype=int))
        return self._apply_operator_rules(canonical)

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
                        out[-1] = int(prev.lex_index)
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

    @staticmethod
    def _is_projective_operator(op: Operator) -> bool:
        return op.kind == "projector"

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


    def _make_cvxpy_variables(self) -> list[cp.Variable]:
        dim = int(self.dimension)
        variables: list[cp.Variable] = []
        for x in range(self.scenario.X_cardinality):
            name = f"Gamma_{x}"
            if self.complex_moments:
                variables.append(cp.Variable((dim, dim), hermitian=True, name=name))
            else:
                variables.append(cp.Variable((dim, dim), symmetric=True, name=name))
        return variables

    @staticmethod
    def _extract_solution_matrices(variables: Sequence[cp.Variable]) -> list[np.ndarray]:
        matrices: list[np.ndarray] = []
        for variable in variables:
            value = variable.value
            if value is None:
                raise RuntimeError("CVXPY did not populate a moment-matrix solution.")
            matrices.append(np.asarray(value, dtype=complex))
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
    "QKDNoncontextualSDP",
    "_shannon_entropy",
]
