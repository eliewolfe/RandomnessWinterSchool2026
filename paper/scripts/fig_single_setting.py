"""Render Fig. 1 (rotated BB84, key from one setting, vs visibility).

Reads paper/data/single_setting_visibility.json (produced by
scan_single_setting_visibility.py). Design: the key-rate axis shows only
the positive-rate region; the mechanism behind the thresholds is shown as
a watermark layer plotting Eve's certified uncertainty (reverse-Fano,
H_E = 2(1 - P_guess)) against Bob's error-correction cost H(K|B) -- the
key rate is their difference, so curve intersections are the critical
visibilities.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
res = json.loads((BASE / "data" / "single_setting_visibility.json").read_text())
BLUE, VERM, INKGRAY = "#0072B2", "#D55E00", "#666666"

plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})
fig, ax = plt.subplots(figsize=(3.4, 2.7), dpi=300)

lp = res["visibility_scan_lp"]
sdp = res["visibility_scan_sdp_level2"]
eta_lp = np.array([r["eta"] for r in lp])
eta_q = np.array([r["eta"] for r in sdp])

# --- watermark layer: uncertainties (bits), same axis as the rates -------
eta_fine = np.linspace(0.5, 1.0, 200)
cost = np.array([-(p := 0.5 + e / (2 * math.sqrt(2))) * math.log2(p) - (1 - p) * math.log2(1 - p)
                 for e in eta_fine])
he_lp = eta_fine / math.sqrt(2)                      # 2(1 - G_LP), G_LP = 1 - sqrt(2) eta / 4
he_q = 2 * (1 - np.array([min(r["sdp2_guess_raw"], 1 - math.sqrt(2) * r["eta"] / 4) for r in sdp]))
wm = dict(lw=0.9, alpha=0.4, zorder=2)
ax.plot(eta_fine, cost, color=INKGRAY, ls=":", **wm)
ax.plot(eta_fine, he_lp, color=VERM, ls="--", **wm)
ax.plot(eta_q, he_q, color=BLUE, ls="-", **wm)
ax.text(0.60, 0.715, r"$H(K|B)=h\left(\frac{1}{2}+\frac{\eta}{2\sqrt{2}}\right)$", fontsize=6.5, color=INKGRAY)
ax.text(0.62, 0.375, r"$H_{\rm E}^{\rm LP}=\eta/\sqrt{2}$", fontsize=6.5, color=VERM, alpha=0.85)
ax.text(0.80, 0.60, r"$H_{\rm E}^{\rm SDP}$", fontsize=6.5, color=BLUE, alpha=0.85)

# --- foreground: key rates, positive region only -------------------------
r_lp = eta_fine / math.sqrt(2) - cost
mask = r_lp >= 0
ax.plot(eta_fine[mask], r_lp[mask], color=VERM, lw=1.6, ls="--", zorder=4,
        label=r"OPT adversary (LP, exact): $\eta/\sqrt{2}-h(\cdot)$")
r_q = np.array([r["sdp2_rate_key_run_RF"] for r in sdp])
mq = r_q >= 0
ax.plot(eta_q[mq], r_q[mq], color=BLUE, lw=1.6, marker="o", ms=3, zorder=4,
        label="quantum adversary (SDP, Moroder level 2)")

idx = np.where(np.diff(np.sign(r_q)) > 0)[0]
i = idx[0]
eta_star_q = eta_q[i] - r_q[i] * (eta_q[i + 1] - eta_q[i]) / (r_q[i + 1] - r_q[i])
for eta_star, color, dy in ((res["lp_rf_threshold_eta"], VERM, 0.10), (float(eta_star_q), BLUE, 0.19)):
    ax.plot([eta_star], [0.0], marker="|", ms=7, color=color, zorder=5)
    ax.annotate(rf"$\eta^*\approx{eta_star:.3f}$", xy=(eta_star, 0.0),
                xytext=(eta_star - 0.012, dy), ha="right", fontsize=6.5, color=color)

ax.set_xlabel(r"preparation visibility $\eta$")
ax.set_ylabel("bits per key round")
ax.set_xlim(0.5, 1.0)
ax.set_ylim(0.0, 1.02)
ax.legend(frameon=False, fontsize=6.6, loc="upper left")
fig.tight_layout(pad=0.3)
(BASE / "figures").mkdir(exist_ok=True)
fig.savefig(BASE / "figures" / "fig_single_setting.pdf")
fig.savefig("/tmp/claude-0/-home-user-RandomnessWinterSchool2026/68b742d9-84a6-5ee4-a07f-df3376d6cbde/scratchpad/fig1.png", dpi=160)
print("wrote fig_single_setting.pdf")
