import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

plt.rcParams.update({"font.size": 9, "font.family": "serif",
                     "axes.linewidth": 0.8, "savefig.dpi": 300, "mathtext.default": "regular"})

AUD, LM, SPEC, OK, BAD, SAMP = "#cfe8ff", "#e7d9ff", "#d7f2d0", "#d7f2d0", "#ffd6d6", "#f5ead0"

def box(ax, x, y, w, h, text, fc, fs=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
                 fc=fc, ec="black", lw=0.9))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs)

def arr(ax, x1, y1, x2, y2, color="black", rad=0.0, lw=1.2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
                 lw=lw, color=color, connectionstyle="arc3,rad=%s" % rad, shrinkA=0, shrinkB=0))

# ===========================================================================
# FIGURE 1 — methodology. Panel (b) shows parallel sampling + majority vote.
# ===========================================================================
fig = plt.figure(figsize=(7.2, 5.1))
gs = gridspec.GridSpec(3, 1, height_ratios=[1.0, 2.05, 1.0], hspace=0.38)
axes = [fig.add_subplot(gs[i]) for i in range(3)]
for ax in axes:
    ax.set_xlim(0, 10); ax.axis("off")

# (a)
ax = axes[0]; ax.set_ylim(0, 1.5)
ax.text(0.0, 1.30, "(a) End-to-end (binding fails)", fontsize=9, fontweight="bold")
box(ax, 0.3, 0.45, 1.9, 0.55, "audio", AUD)
box(ax, 3.3, 0.45, 2.5, 0.55, "Audio LM\n(perceive + reason)", LM)
box(ax, 7.0, 0.45, 2.2, 0.55, "wrong\n(0.49)", BAD)
arr(ax, 2.2, 0.725, 3.3, 0.725); arr(ax, 5.8, 0.725, 7.0, 0.725)
ax.text(4.55, 0.20, "attribute stays latent", fontsize=7.0, style="italic", ha="center", color="#a00")

# (b) Elicitation probe (diagnostic): surface the attribute, then reason by consensus
ax = axes[1]; ax.set_ylim(0, 3.15)
ax.text(0.0, 2.95, "(b) Elicitation probe (diagnostic): surface the attribute, then reason by consensus",
        fontsize=9, fontweight="bold")
box(ax, 0.15, 1.05, 1.35, 0.55, "audio", AUD)
box(ax, 2.05, 1.05, 2.05, 0.55, "surface the\nattribute as text", LM, fs=7.4)
box(ax, 4.55, 1.05, 2.05, 0.55, "re-ground:\naudio + attribute", LM, fs=7.4)
arr(ax, 1.5, 1.325, 2.05, 1.325); arr(ax, 4.10, 1.325, 4.55, 1.325)
labels = ["reason sample 1", "reason sample 2", "reason sample K"]
sy = [2.15, 1.325, 0.50]
for lbl, y in zip(labels, sy):
    box(ax, 7.05, y-0.23, 1.9, 0.46, lbl, SAMP, fs=7.0)
    arr(ax, 6.60, 1.325, 7.05, y, color="#666", lw=1.0)
    arr(ax, 8.95, y, 9.35, 1.325, color="#666", lw=1.0)
box(ax, 9.35, 1.05, 0.6, 0.55, "vote", OK, fs=7.4)
ax.text(3.30, 0.72, "with the benchmark's single-hop question: 0.71; self-derived: collapses to SC (0.59)",
        fontsize=6.6, style="italic", ha="center", color="#a00")
ax.text(8.0, 0.02, "K parallel samples, then majority vote", fontsize=7.0, style="italic",
        ha="center", color="#b06a00")

# (c)
ax = axes[2]; ax.set_ylim(0, 1.5)
ax.text(0.0, 1.30, "(c) Decoupled cascade (upper bound)", fontsize=9, fontweight="bold")
box(ax, 0.3, 0.45, 1.5, 0.55, "audio", AUD)
box(ax, 2.3, 0.45, 2.5, 0.55, "specialist\nperceiver", SPEC)
box(ax, 5.3, 0.45, 2.4, 0.55, "text-only LM\n(reason)", LM)
box(ax, 8.2, 0.45, 1.6, 0.55, "0.84\n(3-track)", OK)
arr(ax, 1.8, 0.725, 2.3, 0.725); arr(ax, 4.8, 0.725, 5.3, 0.725); arr(ax, 7.7, 0.725, 8.2, 0.725)
ax.text(4.0, 0.20, "audio leaves the reasoning path", fontsize=7.0, style="italic", ha="center", color="#060")

plt.savefig("fig_method.pdf", bbox_inches="tight"); plt.close()
print("wrote fig_method.pdf")

# ===========================================================================
# FIGURE 2 — HORIZONTAL bars (roomy labels, no congestion)
# ===========================================================================
fig, ax = plt.subplots(figsize=(3.6, 2.9))
labels = ["cascade (upper bound)", "elicit w/ oracle question\n(diagnostic, not deployable)", "SC: 5 samples + vote", "self-elicit (self-derived)", "AF-Next frontier (greedy)", "base: 1 greedy pass", "no-audio (chance)"]
vals   = [0.843, 0.708, 0.600, 0.588, 0.523, 0.485, 0.285]
colors = ["#3aa657", "#8fb98f", "#2f6db5", "#4a90d9", "#f0b26b", "#9ec5e8", "#bbbbbb"]
ys = np.arange(len(vals))
ax.barh(ys, vals, color=colors, edgecolor="black", lw=0.6, height=0.68)
for y, v in zip(ys, vals):
    ax.text(v + 0.012, y, "%.2f" % v, va="center", ha="left", fontsize=7.4)
ax.axvline(0.25, ls=":", lw=0.9, color="gray")
ax.axvline(0.523, ls="--", lw=1.0, color="#c0392b")
ax.set_yticks(ys); ax.set_yticklabels(labels, fontsize=7.0)
ax.set_xlabel("SAKURA multi-hop accuracy", fontsize=8)
ax.set_xlim(0, 1.0); ax.invert_yaxis()
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig("fig_results.pdf", bbox_inches="tight"); plt.close()
print("wrote fig_results.pdf")
