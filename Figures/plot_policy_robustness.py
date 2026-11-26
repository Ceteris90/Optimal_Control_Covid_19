"""Plot the optimal controls across posterior samples (run seiq_optimal_control_posterior.py
with weights 200, 1000 and 5000 first; it writes oc_w200.npz, oc_w1000.npz, oc_w5000.npz)."""
import os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
INK, TEAL, CORAL = "#14213D", "#0F7173", "#E76F51"
os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"font.size": 14, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
d = np.load("oc_w1000.npz"); t, u1, u2 = d["t"], d["u1"], d["u2"]      # last column = true parameters
q = np.quantile(u1[:, :-1], [.05, .5, .95], axis=1); q2 = np.quantile(u2[:, :-1], [.05, .5, .95], axis=1)
ax[0].fill_between(t, q[0], q[2], color=TEAL, alpha=.25, lw=0, label="posterior 90% band (300 samples)")
ax[0].plot(t, q[1], color=TEAL, lw=2.4, label="posterior median")
ax[0].plot(t, u1[:, -1], "--", color=CORAL, lw=2, label="true parameters")
ax[0].fill_between(t, q2[0], q2[2], color=INK, alpha=.2, lw=0); ax[0].plot(t, q2[1], color=INK, lw=2, label="treatment u2 (median)")
ax[0].set(xlim=(0, 40), ylim=(0, 0.32), xlabel="day", ylabel="control intensity", title="Across the posterior (w = 1000)")
ax[0].legend(frameon=False, fontsize=11)
for w, col in [(200, CORAL), (1000, TEAL), (5000, INK)]:
    q = np.quantile(np.load(f"oc_w{w}.npz")["u1"][:, :-1], [.05, .5, .95], axis=1)
    ax[1].fill_between(t, q[0], q[2], color=col, alpha=.2, lw=0); ax[1].plot(t, q[1], color=col, lw=2.4, label=f"w = {w}")
ax[1].set(xlim=(0, 40), ylim=(0, 0.55), xlabel="day", title="Vaccination u1 for different cost weights"); ax[1].legend(frameon=False)
fig.tight_layout(); fig.savefig("figures/policy_robustness.png", dpi=160)
