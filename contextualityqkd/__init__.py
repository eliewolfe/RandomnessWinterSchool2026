"""Public package API for contextuality randomness tooling."""

from .linalg_utils import (
    enumerate_cone_extremal_rays,
    null_space_basis,
    select_linearly_independent_rows,
)
from .extremal_finders import (
    cone_h_to_v_cdd,
    cone_v_to_h_cdd,
    cone_h_to_v_mosek,
    cone_v_to_h_mosek,
)
from .scenario import ContextualityScenario
from .protocol import ContextualityProtocol
from .quantum import (
    GPTContextualityScenario,
    QuantumContextualityScenario,
)


__all__ = [
    "ContextualityScenario",
    "ContextualityProtocol",
    "GPTContextualityScenario",
    "QuantumContextualityScenario",
    "null_space_basis",
    "select_linearly_independent_rows",
    "enumerate_cone_extremal_rays",
    "cone_h_to_v_cdd",
    "cone_v_to_h_cdd",
    "cone_h_to_v_mosek",
    "cone_v_to_h_mosek",
]
