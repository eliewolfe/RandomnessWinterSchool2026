"""Figure F2: PORAC key rate vs visibility, OPT vs quantum adversary."""
from pathlib import Path
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
res = json.loads((BASE / "data" / "r2_porac.json").read_text())
scan = res["eta_scan"]
BLUE, VERM = "#0072B2", "#D55E00"

plt.rcParams.update({
    "font.size": 8.5, "font.family": "serif", "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
})
fig, ax = plt.subplots(figsize=(3.4, 2.5), dpi=300)
eta = np.array([r["eta"] for r in scan])
lp = np.array([r["lp_rate_key_run_RF"] for r in scan])
q = np.array([r["sdp1_rate_key_run_RF"] for r in scan])

# linear interpolation of the quantum threshold
idx = np.where(np.diff(np.sign(q)) > 0)[0]
eta_star = None
if idx.size:
    i = idx[0]
    eta_star = eta[i] - q[i] * (eta[i+1] - eta[i]) / (q[i+1] - q[i])

ax.axhline(0.0, color="#bbbbbb", lw=0.6, zorder=1)
ax.plot(eta, q, color=BLUE, lw=1.4, ls="-", marker="o", ms=2.5, zorder=3,
        label="quantum adversary (SDP level 1, lower bound)")
ax.plot(eta, lp, color=VERM, lw=1.4, ls="--", zorder=3,
        label="operational adversary (LP, exact)")
if eta_star is not None:
    ax.plot([eta_star], [0.0], marker="|", ms=7, color=BLUE, zorder=4)
    ax.annotate(rf"$\eta^*\approx{eta_star:.3f}$", xy=(eta_star, 0.0),
                xytext=(eta_star - 0.01, -0.20), ha="right", fontsize=7.5, color=BLUE)
    print("quantum threshold eta* ≈", eta_star)

ax.set_xlabel(r"preparation visibility $\eta$")
ax.set_ylabel("key rate per key round (bits)")
ax.set_xlim(0.5, 1.0)
ax.legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(pad=0.3)
(BASE / "figures").mkdir(exist_ok=True)
fig.savefig(BASE / "figures" / "fig_r2_porac.pdf")
fig.savefig("/tmp/claude-0/-home-user-RandomnessWinterSchool2026/68b742d9-84a6-5ee4-a07f-df3376d6cbde/scratchpad/fig_r2.png", dpi=160)
print("wrote fig_r2_porac.pdf")
