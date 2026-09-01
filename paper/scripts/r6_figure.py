"""Figure F3: key rate from noncontextuality-inequality violation alone."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
res = json.loads((BASE / "data" / "r6_witness.json").read_text())
BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"

plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})
fig, ax = plt.subplots(figsize=(3.4, 2.6), dpi=300)
ax.axhline(0.0, color="#bbbbbb", lw=0.6, zorder=1)

# hexagon LP curve, x normalized to fraction of observed violation
hx = res["hexagon"]
w = np.array([r["w"] for r in hx["lp_curve"]]) / hx["observed"]
rate = np.array([r["key_rate_per_key_run"] for r in hx["lp_curve"]])
ax.plot(w, rate, color=GREEN, lw=1.4, ls="-", zorder=3,
        label="hexagon, operational adversary (LP)")

# rotated BB84 LP curve
w = np.array([r["w"] for r in res["lp_curve"]]) / res["observed"]
rate = np.array([r["key_rate_per_key_run"] for r in res["lp_curve"]])
ax.plot(w, rate, color=VERM, lw=1.4, ls="--", zorder=3,
        label="rotated BB84, operational adversary (LP)")

# rotated BB84 quantum SDP level-2 points
w = np.array([p["w"] for p in res["sdp_points_level2"]]) / res["observed"]
rate = np.array([p["key_rate_per_key_run"] for p in res["sdp_points_level2"]])
ax.plot(w, rate, color=BLUE, lw=1.2, ls="-", marker="o", ms=3.5, zorder=4,
        label="rotated BB84, quantum adversary (SDP level 2)")

ax.set_xlabel(r"witness value $w/w_{\rm obs}$")
ax.set_ylabel("key rate per key round (bits)")
ax.set_xlim(0.0, 1.02)
ax.set_ylim(-0.65, 0.62)
ax.legend(frameon=False, fontsize=6.6, loc="upper left")
fig.tight_layout(pad=0.3)
(BASE / "figures").mkdir(exist_ok=True)
fig.savefig(BASE / "figures" / "fig_r6_witness.pdf")
fig.savefig("/tmp/claude-0/-home-user-RandomnessWinterSchool2026/68b742d9-84a6-5ee4-a07f-df3376d6cbde/scratchpad/fig_r6.png", dpi=160)
print("wrote fig_r6_witness.pdf")
