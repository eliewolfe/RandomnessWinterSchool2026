"""Protocol-level QKD analysis with a configurable master-key holder."""

from __future__ import annotations

from copy import deepcopy
from functools import cached_property
import math
from typing import Literal, Sequence

from methodtools import lru_cache
import numpy as np

from .eve_lp import _solve_eve_guess_key_by_y_lp_hotstart
from .eve_sdp import QKDNoncontextualSDP
from .scenario import ContextualityScenario


SolverMethod = Literal["lp", "sdp"]
ReportingMethod = Literal["lp", "sdp", "both"]
RateType = Literal["reverse_fano", "min_entropy"]


class ContextualityProtocol:
    """QKD protocol metrics for key-conditioning subsets and a chosen master-key holder."""

    scenario: ContextualityScenario
    master_key_holder: Literal["Alice", "Bob"]
    atol: float
    _where_key_input: object | None
    _where_key_optimization_result: dict[str, object] | None

    def __init__(
        self,
        scenario: ContextualityScenario,
        where_key: list[list[int]] | np.ndarray | str | None = None,
        master_key_holder: Literal["Alice", "Bob"] | str = "Alice",
        atol: float = 1e-9,
        optimize_verbose: bool | None = None,
        optimize_cluster_tolerance: float = 1e-6,
        optimize_cluster_by: Literal[
            "threshold_uncertainty",
            "threshold_alice_guess_bob_probability",
        ] = "threshold_uncertainty",
        optimize_tie_break: Literal[
            "earliest_optimal_stage",
            "latest_optimal_stage",
        ] = "earliest_optimal_stage",
        sdp_projective_bob: bool = False,
        sdp_projective_eve: bool = False,
        sdp_npa_level_bob: int = 1,
        sdp_npa_level_eve: int = 1,
        sdp_use_u_only: bool = True,
        sdp_threads: int | None = None,
        sdp_verbose: int | bool = 0,
    ) -> None:
        if not isinstance(scenario, ContextualityScenario):
            raise TypeError("scenario must be a ContextualityScenario instance.")
        self.scenario = scenario
        self.atol = float(atol)
        if self.atol < 0.0:
            raise ValueError("atol must be nonnegative.")
        self._where_key_optimization_result = None
        self.master_key_holder = self._canonicalize_master_key_holder(master_key_holder)
        self.sdp_projective_bob = bool(sdp_projective_bob)
        self.sdp_projective_eve = bool(sdp_projective_eve)
        self.sdp_npa_level_bob = int(sdp_npa_level_bob)
        self.sdp_npa_level_eve = int(sdp_npa_level_eve)
        self.sdp_use_u_only = bool(sdp_use_u_only)
        self.sdp_threads = None if sdp_threads is None else int(sdp_threads)
        self.sdp_verbose = int(sdp_verbose)

        if isinstance(where_key, str):
            token = where_key.strip().lower()
            if token not in {"auto", "automatic"}:
                raise ValueError(
                    "Unsupported where_key string token. "
                    "Use one of: 'auto', 'automatic' (case-insensitive)."
                )
            verbosity = bool(self.scenario.verbose) if optimize_verbose is None else bool(optimize_verbose)
            result = self._optimize_where_key_automatic(
                scenario=self.scenario,
                master_key_holder=self.master_key_holder,
                cluster_tolerance=float(optimize_cluster_tolerance),
                cluster_by=optimize_cluster_by,
                tie_tolerance=self.atol,
                tie_break=optimize_tie_break,
                verbose=verbosity,
            )
            self._where_key_optimization_result = result
            self._where_key_input = tuple(
                tuple(int(x) for x in row)
                for row in result["best_stage"]["where_key"]  # type: ignore[index]
            )
            return

        self._where_key_input = where_key

    @staticmethod
    def _canonicalize_master_key_holder(master_key_holder: Literal["Alice", "Bob"] | str) -> Literal["Alice", "Bob"]:
        token = str(master_key_holder).strip().lower()
        if token == "alice":
            return "Alice"
        if token == "bob":
            return "Bob"
        raise ValueError("master_key_holder must be 'Alice' or 'Bob'.")

    @staticmethod
    def _canonicalize_reporting_method(method: ReportingMethod | str) -> ReportingMethod:
        token = str(method).strip().lower()
        if token == "lp":
            return "lp"
        if token == "sdp":
            return "sdp"
        if token == "both":
            return "both"
        raise ValueError("method must be 'lp', 'sdp', or 'both'.")

    @classmethod
    def _expand_reporting_methods(cls, method: ReportingMethod | str) -> tuple[SolverMethod, ...]:
        canonical = cls._canonicalize_reporting_method(method)
        if canonical == "both":
            return ("lp", "sdp")
        return (canonical,)

    @staticmethod
    def _canonicalize_rate_type(rate_type: RateType | str) -> RateType:
        token = str(rate_type).strip().lower()
        if token == "reverse_fano":
            return "reverse_fano"
        if token == "min_entropy":
            return "min_entropy"
        raise ValueError("rate_type must be 'reverse_fano' or 'min_entropy'.")

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

    @classmethod
    def uncertainty_from_guess(
        cls,
        p_guess: float,
        *,
        rate_type: Literal["reverse_fano", "min_entropy"] = "reverse_fano",
    ) -> float:
        """Convert a guessing probability into an Eve uncertainty bound."""
        if rate_type == "reverse_fano":
            return float(cls.reverse_fano_bound(p_guess))
        if rate_type == "min_entropy":
            return float(cls.min_entropy(p_guess))
        raise ValueError("rate_type must be 'reverse_fano' or 'min_entropy'.")

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

    @property
    def where_key_optimization_result(self) -> dict[str, object] | None:
        """Full automatic where_key optimization report, or None when not in auto mode."""
        if self._where_key_optimization_result is None:
            return None
        return deepcopy(self._where_key_optimization_result)

    def format_where_key_optimization_best_stage(self, *, precision: int = 6) -> str:
        """Format the best-stage line from an automatic where_key optimization run."""
        optimization = self.where_key_optimization_result
        if optimization is None:
            raise RuntimeError("Automatic where_key optimization report is unavailable.")
        best_stage = dict(optimization["best_stage"])
        return (
            "Automatic best stage: "
            f"stage={int(best_stage['stage_index'])}, "
            "bits_per_experimental_run="
            + ContextualityScenario.format_numeric(float(best_stage["bits_per_experimental_run"]), precision=precision)
            + ", threshold_uncertainty="
            + ContextualityScenario.format_numeric(float(best_stage["threshold_uncertainty"]), precision=precision)
            + ", threshold_alice_guess_bob_probability="
            + ContextualityScenario.format_numeric(
                float(best_stage["threshold_alice_guess_bob_probability"]),
                precision=precision,
            )
        )

    def print_where_key_optimization_best_stage(
        self,
        *,
        precision: int = 6,
        leading_newline: bool = False,
    ) -> None:
        """Print the best-stage line from automatic where_key optimization."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_where_key_optimization_best_stage(precision=precision))

    @staticmethod
    def _canonicalize_where_key_rows(
        where_key: object,
        *,
        num_x: int,
        num_y: int,
    ) -> tuple[tuple[int, ...], ...]:
        rows_raw = [list(row) for row in where_key]
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

    @classmethod
    def _cluster_metric_values_for_y(
        cls,
        values_by_x: np.ndarray,
        *,
        cluster_by: Literal["threshold_uncertainty", "threshold_alice_guess_bob_probability"],
        tolerance: float,
    ) -> list[dict[str, object]]:
        pairs = [(int(x), float(values_by_x[x])) for x in range(values_by_x.size)]
        if cluster_by == "threshold_uncertainty":
            pairs.sort(key=lambda entry: (entry[1], entry[0]))
        else:
            pairs.sort(key=lambda entry: (-entry[1], entry[0]))

        levels: list[dict[str, object]] = []
        current_x: list[int] = []
        current_values: list[float] = []

        def flush_cluster() -> None:
            if not current_x:
                return
            if cluster_by == "threshold_uncertainty":
                level_value = float(max(current_values))
            else:
                level_value = float(min(current_values))
            levels.append(
                {
                    "x_indices": tuple(sorted(set(int(x) for x in current_x))),
                    "level_value": level_value,
                }
            )

        for x, value in pairs:
            if not current_values:
                current_x = [x]
                current_values = [value]
                continue
            if abs(value - current_values[-1]) <= tolerance:
                current_x.append(x)
                current_values.append(value)
                continue
            flush_cluster()
            current_x = [x]
            current_values = [value]
        flush_cluster()

        if not levels:
            raise ValueError("Cannot build clustering levels from empty metric vector.")
        return levels

    @classmethod
    def _build_where_key_sweep_from_metric(
        cls,
        metric_table_xy: np.ndarray,
        *,
        cluster_by: Literal["threshold_uncertainty", "threshold_alice_guess_bob_probability"],
        tolerance: float,
    ) -> list[tuple[tuple[int, ...], ...]]:
        num_x, num_y = metric_table_xy.shape
        if num_x <= 0 or num_y <= 0:
            raise ValueError("Metric table must have positive X and Y dimensions.")

        levels_by_y: list[list[dict[str, object]]] = []
        for y in range(num_y):
            levels = cls._cluster_metric_values_for_y(
                metric_table_xy[:, y],
                cluster_by=cluster_by,
                tolerance=tolerance,
            )
            levels_by_y.append(levels)

        frontier_by_y = [0 for _ in range(num_y)]
        sweep: list[tuple[tuple[int, ...], ...]] = []

        while True:
            rows: list[tuple[int, ...]] = []
            for y in range(num_y):
                chosen_x: list[int] = []
                for idx in range(frontier_by_y[y] + 1):
                    chosen_x.extend(int(x) for x in levels_by_y[y][idx]["x_indices"])  # type: ignore[index]
                rows.append(tuple(sorted(set(chosen_x))))
            sweep.append(tuple(rows))

            if all(frontier_by_y[y] >= len(levels_by_y[y]) - 1 for y in range(num_y)):
                break

            deltas: list[tuple[float, int]] = []
            for y in range(num_y):
                idx = frontier_by_y[y]
                if idx >= len(levels_by_y[y]) - 1:
                    continue
                current_value = float(levels_by_y[y][idx]["level_value"])  # type: ignore[index]
                next_value = float(levels_by_y[y][idx + 1]["level_value"])  # type: ignore[index]
                if cluster_by == "threshold_uncertainty":
                    delta = max(0.0, next_value - current_value)
                else:
                    delta = max(0.0, current_value - next_value)
                deltas.append((delta, y))

            if not deltas:
                break

            min_delta = min(delta for delta, _ in deltas)
            progressed = False
            for delta, y in deltas:
                if delta <= min_delta + tolerance:
                    frontier_by_y[y] += 1
                    progressed = True
            if not progressed:
                raise RuntimeError("Automatic where_key sweep stalled; no frontier could be advanced.")

        return sweep

    @staticmethod
    def _compute_stage_thresholds(
        where_key: tuple[tuple[int, ...], ...],
        *,
        uncertainty_xy: np.ndarray,
        guess_xy: np.ndarray,
    ) -> tuple[float, float]:
        worst_uncertainty = float("-inf")
        worst_guess = float("inf")
        for y, row in enumerate(where_key):
            if len(row) == 0:
                continue
            x_idx = np.asarray(row, dtype=int)
            worst_uncertainty = max(worst_uncertainty, float(np.max(uncertainty_xy[x_idx, y])))
            worst_guess = min(worst_guess, float(np.min(guess_xy[x_idx, y])))
        if not np.isfinite(worst_uncertainty) or not np.isfinite(worst_guess):
            return float("nan"), float("nan")
        return float(worst_uncertainty), float(worst_guess)

    @classmethod
    def _print_auto_stage_summary(
        cls,
        entry: dict[str, object],
        *,
        total_stages: int,
        precision: int = 6,
    ) -> None:
        stage_index = int(entry["stage_index"])
        print(f"\nSweep stage {stage_index}/{int(total_stages)}")
        if bool(entry["is_uniform_key_count"]):
            uniform_value = int(entry["uniform_key_count"])
            print(f"key counts per y = {uniform_value} (uniform across y)")
        else:
            print(f"key counts per y = {list(entry['key_counts_by_y'])}")
        print(
            "threshold_uncertainty = "
            + ContextualityScenario.format_numeric(float(entry["threshold_uncertainty"]), precision=precision)
        )
        print(
            "threshold_alice_guess_bob_probability = "
            + ContextualityScenario.format_numeric(
                float(entry["threshold_alice_guess_bob_probability"]),
                precision=precision,
            )
        )
        print(
            "key-generating run probability per experimental run = "
            + ContextualityScenario.format_numeric(
                float(entry["key_generation_probability_per_run"]),
                precision=precision,
            )
        )
        print(
            "bits per key-generating run = "
            + ContextualityScenario.format_numeric(
                float(entry["bits_per_key_generating_run"]),
                precision=precision,
            )
        )
        print(
            "bits per experimental run = "
            + ContextualityScenario.format_numeric(float(entry["bits_per_experimental_run"]), precision=precision)
        )

    @classmethod
    def _optimize_where_key_automatic(
        cls,
        *,
        scenario: ContextualityScenario,
        master_key_holder: Literal["Alice", "Bob"],
        cluster_tolerance: float,
        cluster_by: Literal["threshold_uncertainty", "threshold_alice_guess_bob_probability"],
        tie_tolerance: float,
        tie_break: Literal["earliest_optimal_stage", "latest_optimal_stage"],
        verbose: bool,
    ) -> dict[str, object]:
        if cluster_tolerance < 0.0:
            raise ValueError("optimize_cluster_tolerance must be nonnegative.")
        if cluster_by not in {"threshold_uncertainty", "threshold_alice_guess_bob_probability"}:
            raise ValueError(
                "optimize_cluster_by must be 'threshold_uncertainty' or "
                "'threshold_alice_guess_bob_probability'."
            )
        if tie_tolerance < 0.0:
            raise ValueError("Protocol tie tolerance must be nonnegative.")
        if tie_break not in {"earliest_optimal_stage", "latest_optimal_stage"}:
            raise ValueError(
                "optimize_tie_break must be 'earliest_optimal_stage' or "
                "'latest_optimal_stage'."
            )

        baseline = cls(scenario, where_key=None, master_key_holder=master_key_holder, atol=tie_tolerance)
        uncertainty_xy = np.asarray(baseline.alice_uncertainty_bob_by_xy, dtype=float)
        guess_xy = np.asarray(baseline.alice_guess_bob_by_xy, dtype=float)
        metric_table = uncertainty_xy if cluster_by == "threshold_uncertainty" else guess_xy

        sweep_rows = cls._build_where_key_sweep_from_metric(
            metric_table_xy=metric_table,
            cluster_by=cluster_by,
            tolerance=float(cluster_tolerance),
        )
        total_stages = int(len(sweep_rows))

        if verbose:
            print(
                "\nAutomatic where_key optimization started "
                f"(cluster_by={cluster_by}, cluster_tolerance="
                f"{ContextualityScenario.format_numeric(cluster_tolerance, precision=6)})."
            )

        stage_entries: list[dict[str, object]] = []
        for stage_index, where_rows in enumerate(sweep_rows, start=1):
            stage_protocol = cls(
                scenario,
                where_key=where_rows,
                master_key_holder=master_key_holder,
                atol=tie_tolerance,
            )
            key_counts = stage_protocol.key_counts_by_y.astype(int)
            is_uniform = bool(key_counts.size > 0 and np.all(key_counts == key_counts[0]))
            uniform_count = int(key_counts[0]) if is_uniform and key_counts.size > 0 else None
            threshold_uncertainty, threshold_guess = cls._compute_stage_thresholds(
                where_rows,
                uncertainty_xy=uncertainty_xy,
                guess_xy=guess_xy,
            )

            entry: dict[str, object] = {
                "stage_index": int(stage_index),
                "where_key": tuple(tuple(int(x) for x in row) for row in where_rows),
                "key_counts_by_y": [int(v) for v in key_counts.tolist()],
                "is_uniform_key_count": bool(is_uniform),
                "uniform_key_count": uniform_count,
                "threshold_uncertainty": float(threshold_uncertainty),
                "threshold_alice_guess_bob_probability": float(threshold_guess),
                "key_generation_probability_per_run": float(stage_protocol.key_generation_probability_per_run),
                "bits_per_key_generating_run": float(stage_protocol.key_rate_per_key_run_reverse_fano_lp),
                "bits_per_experimental_run": float(stage_protocol.key_rate_per_experimental_run_reverse_fano_lp),
            }
            stage_entries.append(entry)
            if verbose:
                cls._print_auto_stage_summary(entry, total_stages=total_stages, precision=6)

        if not stage_entries:
            raise RuntimeError("Automatic where_key optimization produced no stages.")

        # Tie handling uses protocol-level tolerance and policy.
        best_entry = stage_entries[0]
        best_score = float(best_entry["bits_per_experimental_run"])
        for candidate in stage_entries[1:]:
            candidate_score = float(candidate["bits_per_experimental_run"])
            if candidate_score > best_score + tie_tolerance:
                best_entry = candidate
                best_score = candidate_score
                continue
            if (
                tie_break == "latest_optimal_stage"
                and abs(candidate_score - best_score) <= tie_tolerance
            ):
                best_entry = candidate
                best_score = candidate_score
        best_stage = {
            "stage_index": int(best_entry["stage_index"]),
            "where_key": tuple(tuple(int(x) for x in row) for row in best_entry["where_key"]),  # type: ignore[index]
            "threshold_uncertainty": float(best_entry["threshold_uncertainty"]),
            "threshold_alice_guess_bob_probability": float(best_entry["threshold_alice_guess_bob_probability"]),
            "key_generation_probability_per_run": float(best_entry["key_generation_probability_per_run"]),
            "bits_per_key_generating_run": float(best_entry["bits_per_key_generating_run"]),
            "bits_per_experimental_run": float(best_entry["bits_per_experimental_run"]),
        }

        if verbose:
            print(
                "Best automatic protocol by bits per experimental run: "
                f"stage={best_stage['stage_index']}, "
                "bits_per_experimental_run="
                + ContextualityScenario.format_numeric(
                    float(best_stage["bits_per_experimental_run"]),
                    precision=6,
                )
            )

        return {
            "mode": "automatic",
            "objective": "reverse_fano_bits_per_experimental_run",
            "master_key_holder": master_key_holder,
            "cluster_by": cluster_by,
            "cluster_tolerance": float(cluster_tolerance),
            "tie_tolerance": float(tie_tolerance),
            "tie_break": tie_break,
            "total_stages": total_stages,
            "stages": stage_entries,
            "best_stage": best_stage,
        }

    @cached_property
    def where_key(self) -> tuple[tuple[int, ...], ...]:
        """Canonicalized key-conditioning index rows, one row per y (dedup+sorted)."""
        num_x = int(self.scenario.X_cardinality)
        num_y = int(self.scenario.Y_cardinality)

        if self._where_key_input is None:
            full_row = tuple(range(num_x))
            return tuple(full_row for _ in range(num_y))

        if isinstance(self._where_key_input, str):
            raise ValueError(
                "where_key string tokens are only supported via constructor auto mode "
                "('auto' or 'automatic')."
            )
        return self._canonicalize_where_key_rows(
            self._where_key_input,
            num_x=num_x,
            num_y=num_y,
        )

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
        """Bob best guess probability table ``max_b p(b|x,y)`` with shape ``(X,Y)``."""
        table = np.zeros((self.scenario.X_cardinality, self.scenario.Y_cardinality), dtype=float)
        data = self.scenario.data_numeric
        b_counts = self.scenario.b_cardinality_per_y
        for x, y in np.ndindex(self.scenario.X_cardinality, self.scenario.Y_cardinality):
            b_count = int(b_counts[y])
            table[x, y] = float(np.max(data[x, y, :b_count]))
        return table

    @cached_property
    def alice_guess_bob_by_y_given_go(self) -> np.ndarray:
        """Alice key-conditioned guess vector over y; NaN where ``where_key[y]`` is empty."""
        out = np.full((self.scenario.Y_cardinality,), np.nan, dtype=float)
        for y, row in enumerate(self.where_key):
            if len(row) == 0:
                continue
            out[y] = float(np.mean([self.alice_guess_bob_by_xy[int(x), y] for x in row]))
        return out

    @cached_property
    def bob_outcome_likelihood_by_y_given_go(self) -> np.ndarray:
        """The outcome for Bob probabilities averaged over all x in where_key; NaN where ``where_key[y]`` is empty.
        Returns a 2d array p(b|y), indices 1) Y, 2) B
        """
        out = np.full((self.scenario.Y_cardinality, self.scenario.B_cardinality), np.nan, dtype=float)
        b_counts = self.scenario.b_cardinality_per_y
        data = self.scenario.data_numeric
        for y, row in enumerate(self.where_key):
            if len(row) == 0:
                continue
            for b in range(b_counts[y]):
                out[y, b] = float(np.mean([data[int(x), y, b] for x in row]))
        return out

    @cached_property
    def bob_and_key_joint_distribution_by_y(self) -> np.ndarray:
        """
        We return a joint distribution over key and b probabilities for every y.
        Indices are: 1) Y, 2) B, 3) K
        """
        out = np.full((self.scenario.Y_cardinality, self.scenario.B_cardinality, self.scenario.B_cardinality), np.nan, dtype=float)
        data = self.scenario.data_numeric
        b_counts = self.scenario.b_cardinality_per_y
        for y, row in enumerate(self.where_key):
            for b, k in np.ndindex(b_counts[y], b_counts[y]):
                prob = 0
                for x in row:
                    if k == self.scenario.key_selection_by_xy[int(x), y]:
                        prob += float(data[x, y, b]) / len(row)
                out[y, b, k] = prob
        return out

    @cached_property
    def key_probability_by_b_and_y(self) -> np.ndarray:
        """
        We return a conditional distribution of key probabilities for every y and b.
        Indices are: 1) Y, 2) B, 3) K
        """
        return np.divide(self.bob_and_key_joint_distribution_by_y, self.bob_outcome_likelihood_by_y_given_go[:, :, np.newaxis])
        # np.divide(self.bob_and_key_joint_distribution_by_y, self.bob_outcome_likelihood_by_y_given_go[:, :, np.newaxis], out=np.full_like(self.bob_and_key_joint_distribution_by_y, np.nan), where=self.bob_outcome_likelihood_by_y_given_go[:, :, np.newaxis] > 0)


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
    def bob_uncertainty_key_by_y(self) -> np.ndarray:
        """Bob uncertainty table ``H(K|y,B)`` with shape ``(Y)``."""
        out = np.zeros((self.scenario.Y_cardinality,), dtype=float)
        cond_probs = self.key_probability_by_b_and_y.reshape((self.scenario.Y_cardinality, -1))
        joint_probs = self.bob_and_key_joint_distribution_by_y.reshape((self.scenario.Y_cardinality, -1))
        positive = (cond_probs > 0)
        for y in range(self.scenario.Y_cardinality):
            out[y] = -np.sum(joint_probs[y, positive[y]] * np.log2(cond_probs[y, positive[y]]))
        return out

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
    def eve_guess_key_by_y_lp(self) -> np.ndarray:
        """Eve LP-optimal key-guessing probabilities per y (NaN for empty key rows)."""
        return np.asarray(
            _solve_eve_guess_key_by_y_lp_hotstart(
                scenario=self.scenario,
                where_key=self.where_key,
                master_key="Alice"
            ),
            dtype=float,
        )

    @cached_property
    def eve_guess_key_average_y_lp(self) -> float:
        """Uniform average of Eve LP guess probabilities over non-empty y rows."""
        return float(self._average_over_y(self.eve_guess_key_by_y_lp, y_distribution=None, skip_nan=True))

    @cached_property
    def eve_guess_bob_by_y_lp(self) -> np.ndarray:
        """Eve LP-optimal Bob-guessing probabilities per y (NaN for empty key rows)."""
        return np.asarray(
            _solve_eve_guess_key_by_y_lp_hotstart(
                scenario=self.scenario,
                where_key=self.where_key,
                master_key="Bob"
            ),
            dtype=float,
        )

    @cached_property
    def eve_guess_bob_average_y_lp(self) -> float:
        """Uniform average of Eve LP guess probabilities over non-empty y rows."""
        return float(self._average_over_y(self.eve_guess_bob_by_y_lp, y_distribution=None, skip_nan=True))

    @cached_property
    def eve_uncertainty_bob_min_entropy_by_y_lp(self) -> np.ndarray:
        """Eve min-entropy lower-bound vector for Bob as the master key."""
        out = np.full_like(self.eve_guess_bob_by_y_lp, np.nan, dtype=float)
        for y, value in enumerate(self.eve_guess_bob_by_y_lp):
            if not np.isfinite(value) or value <= 0.0:
                continue
            out[y] = float(self.min_entropy(float(value)))
        return out

    @cached_property
    def eve_uncertainty_bob_min_entropy_average_y_lp(self) -> float:
        """Uniform average of Eve min-entropy bounds for Bob as the master key."""
        return float(self._average_over_y(self.eve_uncertainty_bob_min_entropy_by_y_lp, y_distribution=None, skip_nan=True))

    @cached_property
    def eve_uncertainty_bob_reverse_fano_by_y_lp(self) -> np.ndarray:
        """Eve reverse-Fano entropy lower-bound vector for Bob as the master key."""
        out = np.full_like(self.eve_guess_bob_by_y_lp, np.nan, dtype=float)
        for y, value in enumerate(self.eve_guess_bob_by_y_lp):
            if not np.isfinite(value) or value <= 0.0:
                continue
            out[y] = float(self.reverse_fano_bound(float(value)))
        return out

    @cached_property
    def eve_uncertainty_bob_reverse_fano_average_y_lp(self) -> float:
        """Uniform average of Eve reverse-Fano bounds for Bob as the master key."""
        return float(
            self._average_over_y(
                self.eve_uncertainty_bob_reverse_fano_by_y_lp,
                y_distribution=None,
                skip_nan=True,
            )
        )

    @cached_property
    def eve_guess_master_key_by_y_lp(self) -> np.ndarray:
        """Eve LP-optimal guessing probabilities for the configured master key holder."""
        return np.asarray(
            _solve_eve_guess_key_by_y_lp_hotstart(
                scenario=self.scenario,
                where_key=self.where_key,
                master_key=self.master_key_holder,
            ),
            dtype=float,
        )

    @cached_property
    def eve_guess_master_key_average_y_lp(self) -> float:
        """Uniform average of Eve LP guess probabilities for the configured master key holder."""
        return float(self._average_over_y(self.eve_guess_master_key_by_y_lp, y_distribution=None, skip_nan=True))

    @cached_property
    def eve_uncertainty_master_key_min_entropy_by_y_lp(self) -> np.ndarray:
        """Eve min-entropy lower-bound vector for the configured master key holder."""
        out = np.full_like(self.eve_guess_master_key_by_y_lp, np.nan, dtype=float)
        for y, value in enumerate(self.eve_guess_master_key_by_y_lp):
            if not np.isfinite(value) or value <= 0.0:
                continue
            out[y] = float(self.min_entropy(float(value)))
        return out

    @cached_property
    def eve_uncertainty_master_key_min_entropy_average_y_lp(self) -> float:
        """Uniform average of Eve min-entropy bounds for the configured master key holder."""
        return float(
            self._average_over_y(
                self.eve_uncertainty_master_key_min_entropy_by_y_lp,
                y_distribution=None,
                skip_nan=True,
            )
        )

    @cached_property
    def eve_uncertainty_master_key_reverse_fano_by_y_lp(self) -> np.ndarray:
        """Eve reverse-Fano entropy lower-bound vector for the configured master key holder."""
        out = np.full_like(self.eve_guess_master_key_by_y_lp, np.nan, dtype=float)
        for y, value in enumerate(self.eve_guess_master_key_by_y_lp):
            if not np.isfinite(value) or value <= 0.0:
                continue
            out[y] = float(self.reverse_fano_bound(float(value)))
        return out

    @cached_property
    def eve_uncertainty_master_key_reverse_fano_average_y_lp(self) -> float:
        """Uniform average of Eve reverse-Fano bounds for the configured master key holder."""
        return float(
            self._average_over_y(
                self.eve_uncertainty_master_key_reverse_fano_by_y_lp,
                y_distribution=None,
                skip_nan=True,
            )
        )

    @cached_property
    def other_party_uncertainty_by_y(self) -> np.ndarray:
        """Other-party uncertainty about the key, chosen to match the configured master key holder."""
        return self.bob_uncertainty_key_by_y if self.master_key_holder == "Alice" else self.alice_uncertainty_bob_by_y_key

    @cached_property
    def other_party_uncertainty_key_weighted(self) -> float:
        """Other-party uncertainty averaged over key-eligible ``(x,y)`` pairs."""
        if self.key_pair_count_total == 0:
            return float("nan")
        if self.master_key_holder == "Bob":
            return float(self.alice_uncertainty_bob_key_weighted)
        weights = self.key_counts_by_y.astype(float)
        values = np.asarray(self.bob_uncertainty_key_by_y, dtype=float)
        return float(np.dot(weights, values) / float(np.sum(weights)))

    @cached_property
    def eve_sdp_solver(self) -> QKDNoncontextualSDP:
        """Solved Eve SDP for the configured master-key holder and key subset."""
        solver = QKDNoncontextualSDP(
            self.scenario,
            projective_bob=self.sdp_projective_bob,
            projective_eve=self.sdp_projective_eve,
            npa_level_bob=self.sdp_npa_level_bob,
            npa_level_eve=self.sdp_npa_level_eve,
            use_u_only=self.sdp_use_u_only,
            master_key_holder=self.master_key_holder,
            where_key=self.where_key,
            threads=self.sdp_threads,
            atol=self.atol,
            verbose=self.sdp_verbose,
        )
        try:
            solver.solve_sdp()
        except (AssertionError, RuntimeError):
            if not self.sdp_use_u_only:
                raise
            # U-only is the default fast path, but some scenarios can become
            # infeasible under that relaxation; retry once with full generators.
            if self.sdp_verbose:
                print("[protocol] U-only SDP failed; retrying with full U/U-dagger generator set.")
            solver = QKDNoncontextualSDP(
                self.scenario,
                projective_bob=self.sdp_projective_bob,
                projective_eve=self.sdp_projective_eve,
                npa_level_bob=self.sdp_npa_level_bob,
                npa_level_eve=self.sdp_npa_level_eve,
                use_u_only=False,
                master_key_holder=self.master_key_holder,
                where_key=self.where_key,
                threads=self.sdp_threads,
                atol=self.atol,
                verbose=self.sdp_verbose,
            )
            solver.solve_sdp()
        return solver

    def _lp_guess_vector_for_holder(self, holder: Literal["Alice", "Bob"]) -> np.ndarray:
        return self.eve_guess_key_by_y_lp if holder == "Alice" else self.eve_guess_bob_by_y_lp

    def _lp_guess_average_for_holder(self, holder: Literal["Alice", "Bob"]) -> float:
        return self.eve_guess_key_average_y_lp if holder == "Alice" else self.eve_guess_bob_average_y_lp

    def _lp_uncertainty_vector_for_holder(
        self,
        holder: Literal["Alice", "Bob"],
        *,
        rate_type: RateType,
    ) -> np.ndarray:
        if rate_type == "reverse_fano":
            return (
                self.eve_uncertainty_key_reverse_fano_by_y_lp
                if holder == "Alice"
                else self.eve_uncertainty_bob_reverse_fano_by_y_lp
            )
        return (
            self.eve_uncertainty_key_min_entropy_by_y_lp
            if holder == "Alice"
            else self.eve_uncertainty_bob_min_entropy_by_y_lp
        )

    def _lp_uncertainty_average_for_holder(
        self,
        holder: Literal["Alice", "Bob"],
        *,
        rate_type: RateType,
    ) -> float:
        if rate_type == "reverse_fano":
            return (
                self.eve_uncertainty_key_reverse_fano_average_y_lp
                if holder == "Alice"
                else self.eve_uncertainty_bob_reverse_fano_average_y_lp
            )
        return (
            self.eve_uncertainty_key_min_entropy_average_y_lp
            if holder == "Alice"
            else self.eve_uncertainty_bob_min_entropy_average_y_lp
        )

    @lru_cache(maxsize=None)
    def eve_guessing_metrics(
        self,
        *,
        method: ReportingMethod = "both",
        include_per_y_lp: bool = False,
    ) -> tuple[dict[str, object], ...]:
        """Return cached Eve guessing-metric blocks for LP/SDP reporting."""
        blocks: list[dict[str, object]] = []
        for one_method in self._expand_reporting_methods(method):
            if one_method == "lp":
                blocks.append(
                    {
                        "method": "lp",
                        "master_key_holder": self.master_key_holder,
                        "average": float(self.eve_guess_master_key_average_y_lp),
                        "per_y": (
                            np.asarray(self.eve_guess_master_key_by_y_lp, dtype=float).copy()
                            if include_per_y_lp
                            else None
                        ),
                    }
                )
                continue
            blocks.append(
                {
                    "method": "sdp",
                    "master_key_holder": self.master_key_holder,
                    "average": float(self.eve_guess_master_key_sdp),
                    "per_y": None,
                }
            )
        return tuple(blocks)

    @cached_property
    def eve_guess_master_key_sdp(self) -> float:
        """Eve SDP-optimal average guessing probability over key-eligible pairs."""
        if self.key_pair_count_total == 0:
            return float("nan")
        raw_sdp = float(self.eve_sdp_solver.eve_success_probability)
        # Quantum attacks are contained in the LP attack model. If a chosen SDP
        # relaxation is looser, report the tighter known LP cap at protocol level.
        return float(min(raw_sdp, self.eve_guess_master_key_average_y_lp))

    def eve_uncertainty_master_key_sdp(
        self,
        *,
        rate_type: RateType = "reverse_fano",
    ) -> float:
        """Eve uncertainty lower bound from the SDP guessing probability."""
        sdp_value = float(self.uncertainty_from_guess(self.eve_guess_master_key_sdp, rate_type=rate_type))
        if rate_type == "reverse_fano":
            lp_floor = float(self.eve_uncertainty_master_key_reverse_fano_average_y_lp)
        elif rate_type == "min_entropy":
            lp_floor = float(self.eve_uncertainty_master_key_min_entropy_average_y_lp)
        else:
            raise ValueError("rate_type must be 'reverse_fano' or 'min_entropy'.")
        # LP provides a valid outer-model bound, so protocol-level SDP reporting
        # should never be weaker than the LP uncertainty at the same rate type.
        return float(max(sdp_value, lp_floor))

    def key_rate_per_key_run_sdp(
        self,
        *,
        rate_type: RateType = "reverse_fano",
    ) -> float:
        """Average key bits per key-generating run from the SDP Eve bound."""
        if self.key_pair_count_total == 0:
            return float("nan")
        return float(self.eve_uncertainty_master_key_sdp(rate_type=rate_type) - self.other_party_uncertainty_key_weighted)

    def key_rate_per_experimental_run_sdp(
        self,
        *,
        rate_type: RateType = "reverse_fano",
    ) -> float:
        """Average key bits per experimental run from the SDP Eve bound."""
        if self.key_pair_count_total == 0:
            return 0.0
        return float(self.key_generation_probability_per_run * self.key_rate_per_key_run_sdp(rate_type=rate_type))

    def eve_guessing_probability(
        self,
        *,
        method: SolverMethod = "lp",
    ) -> float:
        """Return Eve's configured-master-key guessing probability."""
        if method == "lp":
            return float(self.eve_guess_master_key_average_y_lp)
        if method == "sdp":
            return float(self.eve_guess_master_key_sdp)
        raise ValueError("method must be 'lp' or 'sdp'.")

    def eve_uncertainty(
        self,
        *,
        method: SolverMethod = "lp",
        rate_type: RateType = "reverse_fano",
    ) -> float:
        """Return Eve's uncertainty bound for LP or SDP security assumptions."""
        if method == "lp":
            if rate_type == "reverse_fano":
                return float(self.eve_uncertainty_master_key_reverse_fano_average_y_lp)
            if rate_type == "min_entropy":
                return float(self.eve_uncertainty_master_key_min_entropy_average_y_lp)
            raise ValueError("rate_type must be 'reverse_fano' or 'min_entropy'.")
        if method == "sdp":
            return float(self.eve_uncertainty_master_key_sdp(rate_type=rate_type))
        raise ValueError("method must be 'lp' or 'sdp'.")

    def key_rate_per_key_run(
        self,
        *,
        method: SolverMethod = "lp",
        rate_type: RateType = "reverse_fano",
    ) -> float:
        """Return key bits per key-generating run for LP or SDP Eve assumptions."""
        if method == "lp":
            if rate_type == "reverse_fano":
                return float(self.key_rate_per_key_run_reverse_fano_lp)
            if rate_type == "min_entropy":
                return float(self.key_rate_per_key_run_min_entropy_lp)
            raise ValueError("rate_type must be 'reverse_fano' or 'min_entropy'.")
        if method == "sdp":
            return float(self.key_rate_per_key_run_sdp(rate_type=rate_type))
        raise ValueError("method must be 'lp' or 'sdp'.")

    def key_rate_per_experimental_run(
        self,
        *,
        method: SolverMethod = "lp",
        rate_type: RateType = "reverse_fano",
    ) -> float:
        """Return key bits per experimental run for LP or SDP Eve assumptions."""
        if method == "lp":
            if rate_type == "reverse_fano":
                return float(self.key_rate_per_experimental_run_reverse_fano_lp)
            if rate_type == "min_entropy":
                return float(self.key_rate_per_experimental_run_min_entropy_lp)
            raise ValueError("rate_type must be 'reverse_fano' or 'min_entropy'.")
        if method == "sdp":
            return float(self.key_rate_per_experimental_run_sdp(rate_type=rate_type))
        raise ValueError("method must be 'lp' or 'sdp'.")

    @cached_property
    def eve_uncertainty_key_min_entropy_by_y_lp(self) -> np.ndarray:
        """Eve min-entropy lower-bound vector from LP guessing probabilities."""
        out = np.full_like(self.eve_guess_key_by_y_lp, np.nan, dtype=float)
        for y, value in enumerate(self.eve_guess_key_by_y_lp):
            if not np.isfinite(value) or value <= 0.0:
                continue
            out[y] = float(self.min_entropy(float(value)))
        return out

    @cached_property
    def eve_uncertainty_key_min_entropy_average_y_lp(self) -> float:
        """Uniform average of Eve min-entropy bounds over non-empty y rows."""
        return float(
            self._average_over_y(
                self.eve_uncertainty_key_min_entropy_by_y_lp,
                y_distribution=None,
                skip_nan=True,
            )
        )

    @cached_property
    def eve_uncertainty_key_reverse_fano_by_y_lp(self) -> np.ndarray:
        """Eve reverse-Fano entropy lower-bound vector from LP guessing probabilities."""
        out = np.full_like(self.eve_guess_key_by_y_lp, np.nan, dtype=float)
        for y, value in enumerate(self.eve_guess_key_by_y_lp):
            if not np.isfinite(value) or value <= 0.0:
                continue
            out[y] = float(self.reverse_fano_bound(float(value)))
        return out

    @cached_property
    def eve_uncertainty_key_reverse_fano_average_y_lp(self) -> float:
        """Uniform average of Eve reverse-Fano bounds over non-empty y rows."""
        return float(
            self._average_over_y(
                self.eve_uncertainty_key_reverse_fano_by_y_lp,
                y_distribution=None,
                skip_nan=True,
            )
        )

    @cached_property
    def key_rate_by_y_reverse_fano_lp(self) -> np.ndarray:
        """Per-y key-rate (Eve reverse-Fano bound minus the other party's uncertainty)."""
        return self.eve_uncertainty_master_key_reverse_fano_by_y_lp - self.other_party_uncertainty_by_y

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
        """Per-y key-rate (min-entropy Eve bound minus the other party's uncertainty)."""
        return self.eve_uncertainty_master_key_min_entropy_by_y_lp - self.other_party_uncertainty_by_y

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
            self._format_numeric_array(self.alice_guess_bob_by_y_given_go, precision=precision_table),
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
        master_key: Literal["Alice", "Bob"] | None = None,
        precision_vector: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Eve LP guessing metrics."""
        holder = self.master_key_holder if master_key is None else self._canonicalize_master_key_holder(master_key)
        guessing_data = self.eve_guess_key_by_y_lp if holder == "Alice" else self.eve_guess_bob_by_y_lp
        average_guessing_data = self.eve_guess_key_average_y_lp if holder == "Alice" else self.eve_guess_bob_average_y_lp

        lines = [
            f"Eve LP guessing metrics, where the master key is held by {holder}:",
            "P_E^guess(master_key|y) (LP):",
            self._format_numeric_array(guessing_data, precision=precision_vector),
            "P_E^guess(master_key|Y) (LP) = "
            f"{ContextualityScenario.format_numeric(average_guessing_data, precision=precision_scalar)}",
        ]
        return "\n".join(lines)

    def print_eve_guessing_metrics_lp(
        self,
        *,
        master_key: Literal["Alice", "Bob"] | None = None,
        precision_vector: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Eve LP guessing metrics."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_eve_guessing_metrics_lp(
            master_key=master_key,
            precision_vector=precision_vector,
            precision_scalar=precision_scalar,
        ))

    def format_eve_uncertainty_metrics_reverse_fano_lp(
        self,
        *,
        master_key: Literal["Alice", "Bob"] | None = None,
        precision_vector: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Eve reverse-Fano uncertainty lower bounds from LP guessing outputs."""
        holder = self.master_key_holder if master_key is None else self._canonicalize_master_key_holder(master_key)
        uncertainty_data = (
            self.eve_uncertainty_key_reverse_fano_by_y_lp if holder == "Alice" else self.eve_uncertainty_bob_reverse_fano_by_y_lp
        )
        average_uncertainty = (
            self.eve_uncertainty_key_reverse_fano_average_y_lp if holder == "Alice" else self.eve_uncertainty_bob_reverse_fano_average_y_lp
        )
        lines = [
            f"Eve uncertainty lower bounds, where the master key is held by {holder}:",
            "H_E(master_key|y) >= RF(P_E^guess(master_key|y)) (LP):",
            self._format_numeric_array(uncertainty_data, precision=precision_vector),
            "H_E(master_key|Y) >= "
            f"{ContextualityScenario.format_numeric(average_uncertainty, precision=precision_scalar)}",
        ]
        return "\n".join(lines)

    def print_eve_uncertainty_metrics_reverse_fano_lp(
        self,
        *,
        master_key: Literal["Alice", "Bob"] | None = None,
        precision_vector: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Eve reverse-Fano uncertainty lower bounds from LP guessing outputs."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_eve_uncertainty_metrics_reverse_fano_lp(
            master_key=master_key,
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
            lines = [f"Key-rate summary (reverse Fano, master key held by {self.master_key_holder}):"] + lines
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

    def format_eve_guessing_metrics(
        self,
        *,
        method: ReportingMethod = "both",
        include_per_y_lp: bool = False,
        precision_vector: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Eve guessing metrics for LP, SDP, or both assumptions."""
        sections: list[str] = []
        for block in self.eve_guessing_metrics(method=method, include_per_y_lp=include_per_y_lp):
            one_method = str(block["method"])
            holder = str(block["master_key_holder"])
            lines = [
                f"Eve {one_method.upper()} guessing metrics, where the master key is held by {holder}:",
            ]
            if one_method == "lp" and include_per_y_lp and block["per_y"] is not None:
                lines.extend(
                    [
                        "P_E^guess(master_key|y) (LP):",
                        self._format_numeric_array(block["per_y"], precision=precision_vector),
                    ]
                )
            summary_label = "P_E^guess(master_key|Y) (LP)" if one_method == "lp" else "P_E^guess(master_key|key-eligible x,y) (SDP)"
            lines.append(
                summary_label
                + " = "
                + ContextualityScenario.format_numeric(float(block["average"]), precision=precision_scalar)
            )
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def print_eve_guessing_metrics(
        self,
        *,
        method: ReportingMethod = "both",
        include_per_y_lp: bool = False,
        precision_vector: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Eve guessing metrics for LP, SDP, or both assumptions."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_eve_guessing_metrics(
            method=method,
            include_per_y_lp=include_per_y_lp,
            precision_vector=precision_vector,
            precision_scalar=precision_scalar,
        ))

    def format_eve_uncertainty_metrics(
        self,
        *,
        method: ReportingMethod = "both",
        rate_type: RateType = "reverse_fano",
        include_per_y_lp: bool = False,
        precision_vector: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Eve uncertainty metrics for LP, SDP, or both assumptions."""
        canonical_rate_type = self._canonicalize_rate_type(rate_type)
        label = "reverse-Fano" if canonical_rate_type == "reverse_fano" else "min-entropy"
        sections: list[str] = []
        for one_method in self._expand_reporting_methods(method):
            lines = [
                f"Eve uncertainty lower bound ({label}, {one_method.upper()}), master key held by {self.master_key_holder}:",
            ]
            if one_method == "lp":
                if include_per_y_lp:
                    lines.extend(
                        [
                            f"H_E(master_key|y) >= {label}(LP):",
                            self._format_numeric_array(
                                self._lp_uncertainty_vector_for_holder(
                                    self.master_key_holder,
                                    rate_type=canonical_rate_type,
                                ),
                                precision=precision_vector,
                            ),
                        ]
                    )
                lines.append(
                    "H_E(master_key|Y) >= "
                    + ContextualityScenario.format_numeric(
                        self._lp_uncertainty_average_for_holder(
                            self.master_key_holder,
                            rate_type=canonical_rate_type,
                        ),
                        precision=precision_scalar,
                    )
                )
            else:
                lines.append(
                    "H_E(master_key) >= "
                    + ContextualityScenario.format_numeric(
                        self.eve_uncertainty(method="sdp", rate_type=canonical_rate_type),
                        precision=precision_scalar,
                    )
                )
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def print_eve_uncertainty_metrics(
        self,
        *,
        method: ReportingMethod = "both",
        rate_type: RateType = "reverse_fano",
        include_per_y_lp: bool = False,
        precision_vector: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Eve uncertainty metrics for LP, SDP, or both assumptions."""
        prefix = "\n" if leading_newline else ""
        print(prefix + self.format_eve_uncertainty_metrics(
            method=method,
            rate_type=rate_type,
            include_per_y_lp=include_per_y_lp,
            precision_vector=precision_vector,
            precision_scalar=precision_scalar,
        ))

    def format_key_rate_summary(
        self,
        *,
        method: ReportingMethod = "both",
        rate_type: RateType = "reverse_fano",
        include_per_y_lp: bool = False,
        header: bool = True,
        precision: int = 6,
    ) -> str:
        """Format key-rate summary for LP, SDP, or both Eve assumptions."""
        canonical_rate_type = self._canonicalize_rate_type(rate_type)
        label = "reverse Fano" if canonical_rate_type == "reverse_fano" else "min entropy"
        sections: list[str] = []
        for one_method in self._expand_reporting_methods(method):
            lines: list[str] = []
            if header:
                lines.append(
                    f"Key-rate summary ({label}, {one_method.upper()}, master key held by {self.master_key_holder}):"
                )
            if one_method == "lp" and include_per_y_lp:
                per_y_rates = (
                    self.key_rate_by_y_reverse_fano_lp
                    if canonical_rate_type == "reverse_fano"
                    else self.key_rate_by_y_min_entropy_lp
                )
                lines.extend(
                    [
                        "bits per key-generating run by y (LP) =",
                        self._format_numeric_array(per_y_rates, precision=precision),
                    ]
                )
            lines.extend(
                [
                    "bits per key-generating run = "
                    + ContextualityScenario.format_numeric(
                        self.key_rate_per_key_run(method=one_method, rate_type=canonical_rate_type),
                        precision=precision,
                    ),
                    "key-generating run probability per experimental run = "
                    + ContextualityScenario.format_numeric(self.key_generation_probability_per_run, precision=precision),
                    "bits per experimental run = "
                    + ContextualityScenario.format_numeric(
                        self.key_rate_per_experimental_run(method=one_method, rate_type=canonical_rate_type),
                        precision=precision,
                    ),
                ]
            )
            sections.append("\n".join(lines))
        return "\n\n".join(sections)

    def print_key_rate_summary(
        self,
        *,
        method: ReportingMethod = "both",
        rate_type: RateType = "reverse_fano",
        include_per_y_lp: bool = False,
        precision: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print key-rate summary for LP, SDP, or both Eve assumptions."""
        prefix = "\n" if leading_newline else ""
        print(
            prefix
            + self.format_key_rate_summary(
                method=method,
                rate_type=rate_type,
                include_per_y_lp=include_per_y_lp,
                precision=precision,
            )
        )

    def format_eve_security_metrics(
        self,
        *,
        method: ReportingMethod = "both",
        rate_type: RateType = "reverse_fano",
        include_per_y_lp: bool = False,
        precision_vector: int = 3,
        precision_scalar: int = 6,
    ) -> str:
        """Format Eve guessing, uncertainty, and key-rate outputs together."""
        sections = [
            self.format_eve_guessing_metrics(
                method=method,
                include_per_y_lp=include_per_y_lp,
                precision_vector=precision_vector,
                precision_scalar=precision_scalar,
            ),
            self.format_eve_uncertainty_metrics(
                method=method,
                rate_type=rate_type,
                include_per_y_lp=include_per_y_lp,
                precision_vector=precision_vector,
                precision_scalar=precision_scalar,
            ),
            self.format_key_rate_summary(
                method=method,
                rate_type=rate_type,
                include_per_y_lp=include_per_y_lp,
                precision=precision_scalar,
            ),
        ]
        return "\n\n".join(sections)

    def print_eve_security_metrics(
        self,
        *,
        method: ReportingMethod = "both",
        rate_type: RateType = "reverse_fano",
        include_per_y_lp: bool = False,
        precision_vector: int = 3,
        precision_scalar: int = 6,
        leading_newline: bool = True,
    ) -> None:
        """Print Eve guessing, uncertainty, and key-rate outputs together."""
        prefix = "\n" if leading_newline else ""
        print(
            prefix
            + self.format_eve_security_metrics(
                method=method,
                rate_type=rate_type,
                include_per_y_lp=include_per_y_lp,
                precision_vector=precision_vector,
                precision_scalar=precision_scalar,
            )
        )

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
