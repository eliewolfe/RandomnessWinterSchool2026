"""Figure F1: key rate vs preparation visibility for the single-setting protocol.

Reads paper/data/r1_results.json; writes paper/figures/fig_r1_visibility.pdf.
Colors: Okabe-Ito CVD-safe pair; linestyle doubles the encoding for print.
"""
from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
FIGS = BASE / "figures"
FIGS.mkdir(exist_ok=True)
res = json.loads((BASE / "data" / "r1_results.json").read_text())

lp = res["visibility_scan_lp"]
sdp2 = res["visibility_scan_sdp_level2"]
eta_star = res["lp_rf_threshold_eta"]

BLUE = "#0072B2"   # quantum adversary (SDP)
VERM = "#D55E00"   # operational adversary (LP)
INK = "#333333"

plt.rcParams.update({
    "font.size": 8.5,
    "font.family": "serif",
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
})

fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)

x_lp = [r["eta"] for r in lp]
y_lp = [r["lp_rate_key_run_RF"] for r in lp]
x_q = [r["eta"] for r in sdp2]
y_q = [r["sdp2_rate_key_run_RF"] for r in sdp2]

ax.axhline(0.0, color="#bbbbbb", lw=0.6, zorder=1)
ax.plot(x_q, y_q, color=BLUE, lw=1.4, ls="-", marker="o", ms=2.5, zorder=3,
        label="quantum adversary (SDP level 2, lower bound)")
ax.plot(x_lp, y_lp, color=VERM, lw=1.4, ls="--", zorder=3,
        label="operational adversary (LP, exact)")
ax.plot([eta_star], [0.0], marker="|", ms=7, color=VERM, zorder=4)
ax.annotate(rf"$\eta^*\approx{eta_star:.3f}$", xy=(eta_star, 0.0),
            xytext=(eta_star - 0.015, 0.12), ha="right", fontsize=7.5, color=VERM)

ax.set_xlabel(r"preparation visibility $\eta$")
ax.set_ylabel("key rate per key round (bits)")
ax.set_xlim(0.5, 1.0)
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(pad=0.3)
fig.savefig(FIGS / "fig_r1_visibility.pdf")
print("wrote", FIGS / "fig_r1_visibility.pdf")
