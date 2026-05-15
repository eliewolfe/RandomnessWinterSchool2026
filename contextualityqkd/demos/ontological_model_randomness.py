from __future__ import annotations
from pathlib import Path
import sys
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
from contextualityqkd.scenario import ContextualityScenario
from contextualityqkd.protocol import ContextualityProtocol

# q[x, y] = p(b=1 | x, y)
# "Coherent relabeling" symmetry:
# each preparation x is deterministic (b=1) on matching measurement y=x,
# and gives 0.25 for b=1 on the other two measurements.
q = np.array(
    [
        [1.00, 0.25, 0.25],
        [0.25, 1.00, 0.25],
        [0.25, 0.25, 1.00],
    ],
    dtype=float,
)

# data[x, y, b], with b in {0,1}
data = np.zeros((3, 3, 2), dtype=float)
data[:, :, 1] = q
data[:, :, 0] = 1.0 - q

# Measurement OPEQs: coefficients d[y,b] enforcing sum_{y,b} d[y,b] p(b|x,y)=0 for all x
opeq_meas = np.array(
    [
        # (M0_0 + M0_1) = (M1_0 + M1_1)
        [[1, 1], [-1, -1], [0, 0]],

        # (M1_0 + M1_1) = (M2_0 + M2_1)
        [[0, 0], [1, 1], [-1, -1]],

        # M0_0 + M1_0 + M2_0 = M0_1 + M1_1 + M2_1
        [[1, -1], [1, -1], [1, -1]],
    ],
    dtype=float,
)

scenario = ContextualityScenario(
    data=data,
    opeq_preps=None,      # as requested
    opeq_meas=opeq_meas,  # as requested
)

scenario.sanity_check()
print(scenario)
scenario.print_probabilities(as_p_b_given_x_y=True, precision=2)
scenario.print_measurement_operational_equivalences(precision=2)

protocol = ContextualityProtocol(
    scenario=scenario,
    where_key=None,  # where_key=None → all x for every y
    master_key_holder="Bob",
    atol=1e-9,
    sdp_threads=None,
    sdp_verbose=0,
)
protocol.print_eve_security_metrics(
    method="both",
    rate_type="reverse_fano",
    include_per_y_lp=False,
    precision_vector=3,
    precision_scalar=6,
    leading_newline=True,
)
