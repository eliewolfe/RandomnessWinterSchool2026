"""Render Fig. 2 ((3,2)-PORAC vs visibility).

Reads paper/data/porac_visibility.json (produced by
scan_porac_visibility.py). The key-rate axis shows only the positive
region; the watermark layer plots Eve's certified reverse-Fano
uncertainty against Bob's error-correction cost. The LP watermark
(H_E = eta/sqrt(3)) never reaches the cost curve: no key against the
operational adversary at any visibility.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
res = json.loads((BASE / "data" / "porac_visibility.json").read_text())
BLUE, VERM, INKGRAY = "#0072B2", "#D55E00", "#666666"

plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})
fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=300)

scan = res["eta_scan"]
eta = np.array([r["eta"] for r in scan])

# --- watermark: uncertainties vs error-correction cost --------------------
eta_fine = np.linspace(0.5, 1.0, 200)
cost = np.array([-(p := 0.5 + e / (2 * math.sqrt(3))) * math.log2(p) - (1 - p) * math.log2(1 - p)
                 for e in eta_fine])
he_lp = eta_fine / math.sqrt(3)               # 2(1 - G_LP), G_LP = 1 - eta/(2 sqrt 3)
he_q = 2 * (1 - np.array([min(r["sdp1_guess_raw"], r["lp_guess_avg"]) for r in scan]))
wm = dict(lw=0.9, alpha=0.4, zorder=2)
ax.plot(eta_fine, cost, color=INKGRAY, ls=":", **wm)
ax.plot(eta_fine, he_lp, color=VERM, ls="--", **wm)
ax.plot(eta, he_q, color=BLUE, ls="-", **wm)
ax.text(0.515, 0.80, r"$H(K|B)=h\left(\frac{1}{2}+\frac{\eta}{2\sqrt{3}}\right)$", fontsize=6.5, color=INKGRAY)
ax.text(0.60, 0.29, r"$H_{\rm E}^{\rm LP}=\eta/\sqrt{3}$ (never reaches $H(K|B)$)",
        fontsize=6.5, color=VERM, alpha=0.9)
ax.text(0.815, 0.545, r"$H_{\rm E}^{\rm SDP}$", fontsize=6.5, color=BLUE, alpha=0.85)

# --- foreground: quantum key rate, positive region only -------------------
r_q = np.array([r["sdp1_rate_key_run_RF"] for r in scan])
mq = r_q >= 0
ax.plot(eta[mq], r_q[mq], color=BLUE, lw=1.6, marker="o", ms=3, zorder=4,
        label="quantum adversary (SDP, Moroder level 1)")
eta_star = res["sdp1_threshold_eta"]
ax.plot([eta_star], [0.0], marker="|", ms=7, color=BLUE, zorder=5)
ax.annotate(rf"$\eta^*\approx{eta_star:.3f}$", xy=(eta_star, 0.0),
            xytext=(eta_star - 0.012, 0.10), ha="right", fontsize=6.5, color=BLUE)
# The exact OPT rate is negative everywhere; it never enters the plotted region.
ax.plot([], [], color=VERM, lw=1.6, ls="--",
        label=r"OPT adversary (LP, exact): no key for any $\eta$")

ax.set_xlabel(r"preparation visibility $\eta$")
ax.set_ylabel("bits per key round")
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.0, 1.02)
ax.legend(frameon=False, fontsize=6.6, loc="upper left")
fig.tight_layout(pad=0.3)
(BASE / "figures").mkdir(exist_ok=True)
fig.savefig(BASE / "figures" / "fig_porac.pdf")
fig.savefig("/tmp/claude-0/-home-user-RandomnessWinterSchool2026/68b742d9-84a6-5ee4-a07f-df3376d6cbde/scratchpad/fig2.png", dpi=160)
print("wrote fig_porac.pdf")
