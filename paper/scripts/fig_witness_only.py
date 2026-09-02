"""Render Fig. 3 (key from a noncontextuality-inequality violation alone).

Reads paper/data/witness_only.json (produced by
verify_witness_only_rates.py). Positive-rate region only; the watermark
layer shows why the rotated-BB84 violation cannot beat the operational
adversary: Eve's LP uncertainty H_E = w never reaches the error-correction
cost h(cos^2 pi/8) within the physically attainable range w <= sqrt(2)-1.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
res = json.loads((BASE / "data" / "witness_only.json").read_text())
BLUE, VERM, GREEN, INKGRAY = "#0072B2", "#D55E00", "#009E73", "#666666"

plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})
fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=300)

bb = res["rotated_bb84"]
hx = res["hexagon"]
cost_bb = bb["honest_cost"]

# --- watermark: rotated-BB84 LP uncertainty vs its EC cost ----------------
f = np.linspace(0, 1, 100)
wm = dict(lw=0.9, alpha=0.4, zorder=2)
ax.plot(f, f * bb["observed"], color=VERM, ls="--", **wm)   # H_E^LP = w
ax.axhline(cost_bb, color=INKGRAY, ls=":", **wm)
ax.text(0.44, cost_bb - 0.035, r"rot. BB84 cost $h(\cos^2\frac{\pi}{8})\approx0.601$",
        fontsize=6.3, color=INKGRAY)
ax.text(0.60, 0.20, r"rot. BB84 $H_{\rm E}^{\rm LP}=w$" "\n" r"(caps at $\sqrt{2}-1<0.601$)",
        fontsize=6.3, color=VERM, alpha=0.9)

# --- foreground: positive key rates ---------------------------------------
w_hx = np.array([r["w"] for r in hx["lp_curve"]]) / hx["observed"]
r_hx = np.array([r["key_rate_per_key_run"] for r in hx["lp_curve"]])
ax.plot(w_hx, r_hx, color=GREEN, lw=1.6, zorder=4,
        label=r"hexagon, OPT adversary (LP, exact): $w/2$")

w_q = np.array([p["w"] for p in bb["sdp_points_level2"]]) / bb["observed"]
r_q = np.array([p["key_rate_per_key_run"] for p in bb["sdp_points_level2"]])
w_star = bb["sdp2_threshold_fraction"]
# join the positive segment from its interpolated zero crossing
w_pos = np.concatenate(([w_star], w_q[r_q >= 0]))
r_pos = np.concatenate(([0.0], r_q[r_q >= 0]))
ax.plot(w_pos, r_pos, color=BLUE, lw=1.6, marker="o", ms=3.5, zorder=4,
        label="rot. BB84, quantum adversary (SDP, Moroder level 2)")
ax.plot([w_star], [0.0], marker="|", ms=7, color=BLUE, zorder=5)
ax.annotate(rf"$w^* \approx{w_star:.2f}\,w_{{\rm obs}}$", xy=(w_star, 0.0),
            xytext=(w_star - 0.02, 0.09), ha="right", fontsize=6.5, color=BLUE)

ax.set_xlabel(r"imposed witness value $w/w_{\rm obs}$")
ax.set_ylabel("bits per key round")
ax.set_xlim(0.0, 1.02)
ax.set_ylim(0.0, 0.72)
ax.legend(frameon=False, fontsize=6.6, loc="upper left")
fig.tight_layout(pad=0.3)
(BASE / "figures").mkdir(exist_ok=True)
fig.savefig(BASE / "figures" / "fig_witness_only.pdf")
fig.savefig("/tmp/claude-0/-home-user-RandomnessWinterSchool2026/68b742d9-84a6-5ee4-a07f-df3376d6cbde/scratchpad/fig3.png", dpi=160)
print("wrote fig_witness_only.pdf")
