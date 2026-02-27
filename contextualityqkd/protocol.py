"""Protocol-level Bob-outcome QKD analysis built on top of ContextualityScenario."""

from __future__ import annotations

from functools import cached_property
import math
from typing import Sequence

import numpy as np

from .randomness_lp import _solve_eve_guess_bob_by_y_lp_hotstart
from .scenario import ContextualityScenario


class ContextualityProtocol:
    """QKD protocol metrics for Bob-outcome guessing with key-conditioning subsets."""

    scenario: ContextualityScenario
    _where_key_input: object | None

    def __init__(
        self,
        scenario: ContextualityScenario,
        where_key: list[list[int]] | np.ndarray | None = None,
    ) -> None:
        if not isinstance(scenario, ContextualityScenario):
            raise TypeError("scenario must be a ContextualityScenario instance.")
        self.scenario = scenario
        self._where_key_input = where_key

    @staticmethod
    def reverse_fano_bound(p_guess: float) -> float:
        """Return a lower bound on conditional Shannon entropy in bits from guessing probability."""
        p = float(p_guess)
        if p <= 0.0:
            raise ValueError("p_guess must be strictly positive.")
        p_eff = min(p, 1.0)
        f = math.floor(1 / p_eff)
        c = f + 1
        return (c * p_eff - 1) * f * math.log2(f) + (1 - f * p_eff) * c * math.log2(c)

    @staticmethod
    def min_entropy(p_guess: float) -> float:
        """Return min-entropy in bits from guessing probability."""
        return float(-math.log2(float(p_guess)))

    @staticmethod
    def binary_entropy(probability: float, atol: float = 1e-12) -> float:
        """Return binary Shannon entropy ``h2(p)`` with endpoint handling."""
        p = float(probability)
        if p < -float(atol) or p > 1.0 + float(atol):
            raise ValueError("probability must be in [0,1].")
        p = min(max(p, 0.0), 1.0)
        if p <= float(atol) or p >= 1.0 - float(atol):
            return 0.0
        return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))

    @cached_property
    def where_key(self) -> tuple[tuple[int, ...], ...]:
        """Canonicalized key-conditioning index rows, one row per y (dedup+sorted)."""
        num_x = int(self.scenario.X_cardinality)
        num_y = int(self.scenario.Y_cardinality)

        if self._where_key_input is None:
            full_row = tuple(range(num_x))
            return tuple(full_row for _ in range(num_y))

        rows_raw = [list(row) for row in self._where_key_input]
        if len(rows_raw) != num_y:
            raise ValueError(f"where_key must have {num_y} rows (one per y).")

        rows: list[tuple[int, ...]] = []
        for y, row in enumerate(rows_raw):
            row_int = [int(x) for x in row]
            if any(x < 0 or x >= num_x for x in row_int):
                raise ValueError(
                    f"where_key[{y}] contains out-of-range x index; valid range is [0, {num_x - 1}]."
                )
            rows.append(tuple(sorted(set(row_int))))
        return tuple(rows)

    @cached_property
    def key_counts_by_y(self) -> np.ndarray:
        """Number of key-eligible x values per y, shape ``(Y,)``."""
        return np.asarray([len(row) for row in self.where_key], dtype=int)

    @cached_property
    def key_mask_xy(self) -> np.ndarray:
        """Boolean mask marking key-eligible (x,y) pairs, shape ``(X,Y)``."""
        mask = np.zeros((self.scenario.X_cardinality, self.scenario.Y_cardinality), dtype=bool)
        for y, row in enumerate(self.where_key):
            for x in row:
                mask[int(x), int(y)] = True
        return mask

    @cached_property
    def key_pair_count_total(self) -> int:
        """Total number of key-eligible (x,y) pairs under uniform x/y sampling."""
        return int(np.sum(self.key_counts_by_y))

    @cached_property
    def key_generation_probability_per_run(self) -> float:
        """Probability an experimental run is key-generating under uniform x/y sampling."""
        total_pairs = float(self.scenario.X_cardinality * self.scenario.Y_cardinality)
        return float(self.key_pair_count_total / total_pairs)

    @cached_property
    def alice_guess_bob_by_xy(self) -> np.ndarray:
        """Alice correct-guess probability table ``max_b p(b|x,y)`` with shape ``(X,Y)``."""
        table = np.zeros((self.scenario.X_cardinality, self.scenario.Y_cardinality), dtype=float)
        data = self.scenario.data_numeric
        b_counts = self.scenario.b_cardinality_per_y
        for x, y in np.ndindex(self.scenario.X_cardinality, self.scenario.Y_cardinality):
            b_count = int(b_counts[y])
            table[x, y] = float(np.max(data[x, y, :b_count]))
        return table

    @cached_property
    def alice_guess_bob_by_y_key(self) -> np.ndarray:
        """Alice key-conditioned guess vector over y; NaN where ``where_key[y]`` is empty."""
        out = np.full((self.scenario.Y_cardinality,), np.nan, dtype=float)
        for y, row in enumerate(self.where_key):
            if len(row) == 0:
                continue
            out[y] = float(np.mean([self.alice_guess_bob_by_xy[int(x), y] for x in row]))
        return out

    @cached_property
    def alice_guess_bob_by_xy_key_masked(self) -> np.ma.MaskedArray:
        """Alice ``P^guess(B|x,y)`` table masked to key-eligible ``(x,y)`` pairs."""
        return np.ma.array(self.alice_guess_bob_by_xy, mask=~self.key_mask_xy)

    @cached_property
    def alice_guess_bob_key_weighted(self) -> float:
        """Alice key-conditioned guess probability averaged by key-pair counts."""
        if self.key_pair_count_total == 0:
            return float("nan")
        total = 0.0
        for y, row in enumerate(self.where_key):
            for x in row:
                total += float(self.alice_guess_bob_by_xy[int(x), y])
        return float(total / float(self.key_pair_count_total))

    @cached_property
    def alice_uncertainty_bob_by_xy(self) -> np.ndarray:
        """Alice uncertainty table ``H(B|x,y)`` with shape ``(X,Y)``."""
        table = np.zeros((self.scenario.X_cardinality, self.scenario.Y_cardinality), dtype=float)
        data = self.scenario.data_numeric
        b_counts = self.scenario.b_cardinality_per_y
        for x, y in np.ndindex(self.scenario.X_cardinality, self.scenario.Y_cardinality):
            b_count = int(b_counts[y])
            table[x, y] = ContextualityScenario._shannon_entropy(data[x, y, :b_count], atol=self.scenario.atol)
        return table

    @cached_property
    def alice_uncertainty_bob_by_y_key(self) -> np.ndarray:
        """Alice key-conditioned uncertainty vector over y; NaN for empty key rows."""
        out = np.full((self.scenario.Y_cardinality,), np.nan, dtype=float)
        for y, row in enumerate(self.where_key):
            if len(row) == 0:
                continue
            out[y] = float(np.mean([self.alice_uncertainty_bob_by_xy[int(x), y] for x in row]))
        return out

    @cached_property
    def alice_uncertainty_bob_by_xy_key_masked(self) -> np.ma.MaskedArray:
        """Alice ``H(B|x,y)`` table masked to key-eligible ``(x,y)`` pairs."""
        return np.ma.array(self.alice_uncertainty_bob_by_xy, mask=~self.key_mask_xy)

    @cached_property
    def alice_uncertainty_bob_key_weighted(self) -> float:
        """Alice key-conditioned uncertainty averaged by key-pair counts."""
        if self.key_pair_count_total == 0:
            return float("nan")
        total = 0.0
        for y, row in enumerate(self.where_key):
            for x in row:
                total += float(self.alice_uncertainty_bob_by_xy[int(x), y])
        return float(total / float(self.key_pair_count_total))

    @cached_property
    def eve_guess_bob_by_y_lp(self) -> np.ndarray:
        """Eve LP-optimal Bob-guessing probabilities per y (NaN for empty key rows)."""
        return np.asarray(
            _solve_eve_guess_bob_by_y_lp_hotstart(
                scenario=self.scenario,
                where_key=self.where_key,
            ),
            dtype=float,
        )

    @cached_property
    def eve_guess_bob_average_y_lp(self) -> float:
        """Uniform average of Eve LP guess probabilities over non-empty y rows."""
        return float(self._average_over_y(self.eve_guess_bob_by_y_lp, y_distribution=None, skip_nan=True))

    @cached_property
    def eve_uncertainty_bob_min_entropy_by_y_lp(self) -> np.ndarray:
        """Eve min-entropy lower-bound vector from LP guessing probabilities."""
        out = np.full_like(self.eve_guess_bob_by_y_lp, np.nan, dtype=float)
        for y, value in enumerate(self.eve_guess_bob_by_y_lp):
            if not np.isfinite(value) or value <= 0.0:
                continue
            out[y] = float(self.min_entropy(float(value)))
        return out

    @cached_property
    def eve_uncertainty_bob_min_entropy_average_y_lp(self) -> float:
        """Uniform average of Eve min-entropy bounds over non-empty y rows."""
        return float(
            self._average_over_y(
                self.eve_uncertainty_bob_min_entropy_by_y_lp,
                y_distribution=None,
                skip_nan=True,
            )
        )

    @cached_property
    def eve_uncertainty_bob_reverse_fano_by_y_lp(self) -> np.ndarray:
        """Eve reverse-Fano entropy lower-bound vector from LP guessing probabilities."""
        out = np.full_like(self.eve_guess_bob_by_y_lp, np.nan, dtype=float)
        for y, value in enumerate(self.eve_guess_bob_by_y_lp):
            if not np.isfinite(value) or value <= 0.0:
                continue
            out[y] = float(self.reverse_fano_bound(float(value)))
        return out

    @cached_property
    def eve_uncertainty_bob_reverse_fano_average_y_lp(self) -> float:
        """Uniform average of Eve reverse-Fano bounds over non-empty y rows."""
        return float(
            self._average_over_y(
                self.eve_uncertainty_bob_reverse_fano_by_y_lp,
                y_distribution=None,
                skip_nan=True,
            )
        )

    @cached_property
    def key_rate_by_y_reverse_fano_lp(self) -> np.ndarray:
        """Per-y key-rate (reverse-Fano Eve bound minus Alice uncertainty), key-conditioned."""
        return self.eve_uncertainty_bob_reverse_fano_by_y_lp - self.alice_uncertainty_bob_by_y_key

    @cached_property
    def key_rate_per_key_run_reverse_fano_lp(self) -> float:
        """Average key bits per key-generating run using reverse-Fano Eve bound."""
        if self.key_pair_count_total == 0:
            return float("nan")
        weights = self.key_counts_by_y.astype(float)
        rates = np.asarray(self.key_rate_by_y_reverse_fano_lp, dtype=float)
        weighted_total = 0.0
        for y in range(self.scenario.Y_cardinality):
            if weights[y] <= 0.0:
                continue
            if not np.isfinite(rates[y]):
                continue
            weighted_total += float(weights[y] * rates[y])
        return float(weighted_total / float(np.sum(weights)))

    @cached_property
    def key_rate_per_experimental_run_reverse_fano_lp(self) -> float:
        """Average key bits per experimental run using reverse-Fano Eve bound."""
        if self.key_pair_count_total == 0:
            return 0.0
        return float(self.key_generation_probability_per_run * self.key_rate_per_key_run_reverse_fano_lp)

    @cached_property
    def key_rate_by_y_min_entropy_lp(self) -> np.ndarray:
        """Per-y key-rate (min-entropy Eve bound minus Alice uncertainty), key-conditioned."""
        return self.eve_uncertainty_bob_min_entropy_by_y_lp - self.alice_uncertainty_bob_by_y_key

    @cached_property
    def key_rate_per_key_run_min_entropy_lp(self) -> float:
        """Average key bits per key-generating run using min-entropy Eve bound."""
        if self.key_pair_count_total == 0:
            return float("nan")
        weights = self.key_counts_by_y.astype(float)
        rates = np.asarray(self.key_rate_by_y_min_entropy_lp, dtype=float)
        weighted_total = 0.0
        for y in range(self.scenario.Y_cardinality):
            if weights[y] <= 0.0:
                continue
            if not np.isfinite(rates[y]):
                continue
            weighted_total += float(weights[y] * rates[y])
        return float(weighted_total / float(np.sum(weights)))

    @cached_property
    def key_rate_per_experimental_run_min_entropy_lp(self) -> float:
        """Average key bits per experimental run using min-entropy Eve bound."""
        if self.key_pair_count_total == 0:
            return 0.0
        return float(self.key_generation_probability_per_run * self.key_rate_per_key_run_min_entropy_lp)

    def format_alice_guessing_metrics(
        self,
        *,
        precision_table: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Alice guessing metrics with ``(x,y)`` output masked by key-eligibility."""
        lines = [
            "Alice guessing metrics:",
            "P_A^guess(B|x,y) on key-eligible pairs (masked by where_key):",
            self._format_masked_xy_table(self.alice_guess_bob_by_xy_key_masked, precision=precision_table),
            "P_A^guess(B|X,y,key):",
            self._format_numeric_array(self.alice_guess_bob_by_y_key, precision=precision_table),
            "P_A^guess(B|key) = "
            f"{ContextualityScenario.format_numeric(self.alice_guess_bob_key_weighted, precision=precision_scalar)}",
        ]
        return "\n".join(lines)

    def print_alice_guessing_metrics(
        self,
        *,
        precision_table: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Alice guessing metrics with ``(x,y)`` output masked by key-eligibility."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_alice_guessing_metrics(
            precision_table=precision_table,
            precision_scalar=precision_scalar,
        ))

    def format_alice_uncertainty_metrics(
        self,
        *,
        precision_table: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Alice uncertainty metrics with ``(x,y)`` output masked by key-eligibility."""
        lines = [
            "Alice uncertainty metrics:",
            "H_A(B|x,y) on key-eligible pairs (masked by where_key):",
            self._format_masked_xy_table(self.alice_uncertainty_bob_by_xy_key_masked, precision=precision_table),
            "H_A(B|X,y,key):",
            self._format_numeric_array(self.alice_uncertainty_bob_by_y_key, precision=precision_table),
            "H_A(B|key) = "
            f"{ContextualityScenario.format_numeric(self.alice_uncertainty_bob_key_weighted, precision=precision_scalar)}",
        ]
        return "\n".join(lines)

    def print_alice_uncertainty_metrics(
        self,
        *,
        precision_table: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Alice uncertainty metrics with ``(x,y)`` output masked by key-eligibility."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_alice_uncertainty_metrics(
            precision_table=precision_table,
            precision_scalar=precision_scalar,
        ))

    def format_eve_guessing_metrics_lp(
        self,
        *,
        precision_vector: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Eve LP guessing metrics."""
        lines = [
            "Eve LP guessing metrics:",
            "P_E^guess(B|y,key) (LP):",
            self._format_numeric_array(self.eve_guess_bob_by_y_lp, precision=precision_vector),
            "P_E^guess(B|Y,key) (LP) = "
            f"{ContextualityScenario.format_numeric(self.eve_guess_bob_average_y_lp, precision=precision_scalar)}",
        ]
        return "\n".join(lines)

    def print_eve_guessing_metrics_lp(
        self,
        *,
        precision_vector: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Eve LP guessing metrics."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_eve_guessing_metrics_lp(
            precision_vector=precision_vector,
            precision_scalar=precision_scalar,
        ))

    def format_eve_uncertainty_metrics_reverse_fano_lp(
        self,
        *,
        precision_vector: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Eve reverse-Fano uncertainty lower bounds from LP guessing outputs."""
        lines = [
            "Eve uncertainty lower bounds:",
            "H_E(B|y,key) >= RF(P_E^guess) (LP):",
            self._format_numeric_array(self.eve_uncertainty_bob_reverse_fano_by_y_lp, precision=precision_vector),
            "H_E(B|Y,key) >= "
            f"{ContextualityScenario.format_numeric(self.eve_uncertainty_bob_reverse_fano_average_y_lp, precision=precision_scalar)}",
        ]
        return "\n".join(lines)

    def print_eve_uncertainty_metrics_reverse_fano_lp(
        self,
        *,
        precision_vector: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Eve reverse-Fano uncertainty lower bounds from LP guessing outputs."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_eve_uncertainty_metrics_reverse_fano_lp(
            precision_vector=precision_vector,
            precision_scalar=precision_scalar,
        ))

    def format_key_rate_summary_reverse_fano_lp(self, *, header=False, precision: int = 6) -> str:
        """Format reverse-Fano key-rate summary."""
        lines = [
            "bits per key-generating run = "
            f"{ContextualityScenario.format_numeric(self.key_rate_per_key_run_reverse_fano_lp, precision=precision)}",
            "key-generating run probability per experimental run = "
            f"{ContextualityScenario.format_numeric(self.key_generation_probability_per_run, precision=precision)}",
            "bits per experimental run = "
            f"{ContextualityScenario.format_numeric(self.key_rate_per_experimental_run_reverse_fano_lp, precision=precision)}",
        ]
        if header:
            lines = ["Key-rate summary (reverse Fano):"] + lines
        return "\n".join(lines)

    def print_key_rate_summary_reverse_fano_lp(
        self,
        *,
        precision: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print reverse-Fano key-rate summary."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_key_rate_summary_reverse_fano_lp(precision=precision))

    def _validate_y_distribution(self, y_distribution: np.ndarray | Sequence[float] | None) -> np.ndarray:
        """Validate optional y-distribution and return normalized weights of shape ``(Y,)``."""
        num_y = int(self.scenario.Y_cardinality)
        if y_distribution is None:
            return np.full((num_y,), 1.0 / float(num_y), dtype=float)

        weights = np.asarray(y_distribution, dtype=float).reshape(-1)
        if weights.size != num_y:
            raise ValueError(f"y_distribution must have length {num_y}.")
        if np.any(weights < 0.0):
            raise ValueError("y_distribution entries must be nonnegative.")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("y_distribution must have positive total mass.")
        return weights / total

    def _average_over_y(
        self,
        values: np.ndarray | Sequence[float],
        y_distribution: np.ndarray | Sequence[float] | None = None,
        skip_nan: bool = True,
    ) -> float:
        """Average y-indexed values with optional custom y-distribution."""
        arr = np.asarray(values, dtype=float).reshape(-1)
        if arr.size != int(self.scenario.Y_cardinality):
            raise ValueError(f"values must have length {self.scenario.Y_cardinality}.")
        weights = self._validate_y_distribution(y_distribution)

        if skip_nan:
            keep = np.isfinite(arr)
            if not np.any(keep):
                return float("nan")
            keep_weights = weights[keep]
            keep_total = float(np.sum(keep_weights))
            if keep_total <= 0.0:
                return float("nan")
            keep_weights = keep_weights / keep_total
            return float(np.dot(keep_weights, arr[keep]))

        return float(np.dot(weights, arr))

    def _format_numeric_array(
        self,
        values: np.ndarray | Sequence[float],
        *,
        precision: int,
    ) -> str:
        """Format numeric arrays/vectors with scenario-consistent scalar formatting."""
        return np.array2string(
            np.asarray(values, dtype=float),
            formatter={"float_kind": lambda value: ContextualityScenario.format_numeric(value, precision=precision)},
        )

    def _format_masked_xy_table(self, values: np.ma.MaskedArray | np.ndarray, *, precision: int) -> str:
        """Format an ``(X,Y)`` table, rendering masked entries as ``--``."""
        masked = np.ma.asarray(values, dtype=float)
        expected_shape = (self.scenario.X_cardinality, self.scenario.Y_cardinality)
        if masked.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape}, got {masked.shape}.")

        data = np.asarray(masked.data, dtype=float)
        mask = np.asarray(np.ma.getmaskarray(masked), dtype=bool)

        rows: list[str] = []
        for x in range(expected_shape[0]):
            entries: list[str] = []
            for y in range(expected_shape[1]):
                if mask[x, y]:
                    entries.append("--")
                else:
                    entries.append(ContextualityScenario.format_numeric(data[x, y], precision=precision))
            rows.append("[" + ", ".join(entries) + "]")
        return "[\n " + "\n ".join(rows) + "\n]"
