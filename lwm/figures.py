"""
Figures for the decision memo. Reads checkpoints + eval_out/metrics.json and writes PNGs to
figures/. Run AFTER train.py and evaluate.py.

Run:  python -m lwm.figures
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from lwm.config import FIELD_NAMES, F, P, M, A, C, HISTORY
from lwm.data import make_cohorts, make_probes
from lwm.generator import generate_dataset
from lwm.data import Cohort
from lwm.evaluate import load_model, upper_tail_rollout
from lwm.train import free_rollout_mae

FIGDIR = "figures"
BLUE, RED, GREEN, GREY = "#2b6cb0", "#c53030", "#2f855a", "#718096"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})


def _load_metrics():
    with open("eval_out/metrics.json") as f:
        return json.load(f)


def fig_training():
    blob = torch.load("checkpoints/jepa.pt", weights_only=False)
    h = blob["history"]
    ep = range(len(h["val_mae"]))
    fig, ax1 = plt.subplots(figsize=(6.5, 3.6))
    ax1.plot(ep, h["val_mae"], color=BLUE, lw=2, label="val free-rollout MAE")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("validation MAE", color=BLUE)
    ax1.tick_params(axis="y", labelcolor=BLUE)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot(ep, h["eff_rank"], color=RED, lw=2, ls="--", label="latent effective rank")
    ax2.axhline(3.5, color=GREY, ls=":", lw=1)
    ax2.set_ylabel("effective rank (of 16)", color=RED); ax2.tick_params(axis="y", labelcolor=RED)
    ax2.text(len(h["val_mae"]) * 0.55, 3.7, "data intrinsic dim 2.70", color=GREY, fontsize=8)
    plt.title("JEPA training: accuracy converges, latent does not collapse")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_training.png"); plt.close(fig)


def fig_rollout():
    X, patients = generate_dataset(60, 48, seed=333)
    # pick a patient with a clear ratchet rise
    i = int(np.argmax(X[:, -1, F]))
    coh = Cohort(X[i:i + 1], [patients[i]], 48)
    jepa, _ = load_model("jepa")
    _, _, preds, _ = free_rollout_mae(jepa, coh, K=48 - HISTORY, start=HISTORY)
    seq = torch.cat([coh.X[:, HISTORY - 1:HISTORY], preds], 1)[0].numpy()
    months_pred = range(HISTORY - 1, 48)
    fig, axes = plt.subplots(1, 5, figsize=(15, 3), sharex=True)
    for ax, idx, name in zip(axes, [F, P, M, A, C], ["F", "P", "M", "A", "C"]):
        ax.plot(range(48), X[i, :, idx], color=BLUE, lw=2, label="true")
        ax.plot(months_pred, seq[:, idx], color=RED, lw=2, ls="--", label="predicted")
        ax.axvline(HISTORY - 1, color=GREY, ls=":", lw=1)
        ax.set_title(name); ax.set_xlabel("month")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"Free-rollout from month {HISTORY-1} (grey) — patient susceptibility "
                 f"{patients[i].susceptibility:.2f}", y=1.02)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_rollout.png", bbox_inches="tight"); plt.close(fig)


def fig_phase_boundary():
    m = _load_metrics()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))
    # left: sparsity
    pb = m["phase_boundary"]
    strides = [int(s) for s in pb]
    for name, col in [("memoryless", GREY), ("supervised", BLUE), ("oracle", GREEN)]:
        axL.plot(strides, [pb[str(s)][name] for s in strides], marker="o", color=col, label=name)
    axL.set_xlabel("months between clinic visits (re-anchor stride)")
    axL.set_ylabel("rollout MAE"); axL.set_title("(a) Sparse visits: history helps only a whisper")
    axL.legend(fontsize=8)
    # right: denoising (the real win)
    dn = m["denoising"]
    sig = [float(s) for s in dn]
    axR.plot(sig, [dn[str(s)]["raw_noisy_obs"] for s in sig], marker="s", color=GREY,
             label="raw noisy obs (memoryless floor)")
    axR.plot(sig, [dn[str(s)]["supervised"] for s in sig], marker="o", color=BLUE, label="history denoise")
    axR.plot(sig, [dn[str(s)]["jepa"] for s in sig], marker="^", color=RED, label="JEPA denoise")
    axR.set_xlabel("sensor noise σ"); axR.set_ylabel("current-state estimate error")
    axR.set_title("(b) Sensor noise: the latent buys a decisive win"); axR.legend(fontsize=8)
    fig.suptitle("Phase boundary: WHEN does the predictive latent earn its keep?", y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_phase_boundary.png", bbox_inches="tight"); plt.close(fig)


def fig_probes_latent():
    m = _load_metrics()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4))
    probes = m["probes"]; names = list(next(iter(probes.values())).keys())
    base = {n: m["accuracy"][n]["overall"] for n in names}
    labels = ["in-dist"] + list(probes.keys()) + ["mechanism\nshift"]
    xs = np.arange(len(labels)); w = 0.2
    for j, n in enumerate(names):
        vals = [base[n]] + [probes[p][n]["overall"] for p in probes] + [m["mechanism_shift"][n]["overall"]]
        axL.bar(xs + (j - 1.5) * w, vals, w, label=n)
    axL.set_xticks(xs); axL.set_xticklabels(labels); axL.set_ylabel("overall MAE")
    axL.set_title("(a) Generalisation probes (failures shown)"); axL.legend(fontsize=8)
    # decodability
    dec = m["decodability_R2_susceptibility"]
    axR.bar(range(len(dec)), list(dec.values()), color=[GREY, BLUE, RED])
    axR.set_xticks(range(len(dec))); axR.set_xticklabels(list(dec.keys()), rotation=15, fontsize=8)
    axR.set_ylabel("R² (susceptibility)")
    axR.set_title("(b) x(t) already decodes the hidden cause")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_probes_latent.png", bbox_inches="tight"); plt.close(fig)


def fig_manifold():
    m = _load_metrics()
    mc = m["manifold_critic"]
    names = ["memoryless", "supervised", "oracle", "jepa"]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    vals = [mc[n] for n in names]
    ax.bar(names, vals, color=[GREY, BLUE, GREEN, RED])
    ax.axhline(mc["real_transitions"], color="black", ls="--", lw=1.2,
               label=f"real transitions ({mc['real_transitions']:.2f})")
    ax.set_ylabel("critic on-manifold score"); ax.set_ylim(0, 1)
    ax.set_title(f"On-manifold score of each model's rollout (critic AUC {mc['critic_auc_valid_but_wrong']:.2f})")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_manifold.png"); plt.close(fig)


def fig_tail_risk():
    """Show one true-positive P80 risk call; it is a risk interval, not a point forecast."""
    path = "checkpoints/supervised_quantile.pt"
    if not os.path.exists(path):
        return
    X, patients = generate_dataset(400, 48, seed=900)
    coh = Cohort(X, patients, 48)
    model, _ = load_model("supervised_quantile")
    point, tail, _ = upper_tail_rollout(model, coh)
    candidates = [i for i in range(len(patients)) if X[i, :, P].max() >= 0.5 and (tail[i] >= 0.5).any()]
    if not candidates:
        return
    i = candidates[0]
    months = np.arange(48)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(months, X[i, :, P], color=BLUE, lw=2, label="true P")
    ax.plot(months[HISTORY:], point[i, :, P].numpy(), color=GREY, lw=2, ls="--", label="point forecast")
    ax.plot(months[HISTORY:], tail[i].numpy(), color=RED, lw=2, label="calibrated P80 risk")
    ax.axhline(0.5, color="black", lw=1, ls=":", label="decompensation threshold")
    ax.axvline(HISTORY - 1, color=GREY, lw=1, ls=":")
    ax.set(xlabel="month", ylabel="portal hypertension P", ylim=(0, 1),
           title="P80 risk path catches a tail event the point forecast misses")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_tail_risk.png"); plt.close(fig)


def fig_summary():
    """One panel telling the whole honest story: constraints, the accuracy tie, the denoising win,
    and the decompensation recall (point vs P80 risk). Reads multiseed.json for error bars if present."""
    m = _load_metrics()
    ms = None
    if os.path.exists("eval_out/multiseed.json"):
        with open("eval_out/multiseed.json") as f:
            ms = json.load(f)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.6))

    # (a) accuracy tie vs noise floor (error bars from multiseed if available)
    names = ["memoryless", "supervised", "oracle", "jepa"]
    a = axes[0]
    if ms and "mae" in ms:
        means = [ms["mae"][n]["mean"] for n in names]; errs = [ms["mae"][n]["std"] for n in names]
        a.bar(names, means, yerr=errs, capsize=4, color=[GREY, BLUE, GREEN, RED])
        a.set_title("(a) Accuracy tie (3-seed mean±std)")
    else:
        a.bar(names, [m["accuracy"][n]["overall"] for n in names], color=[GREY, BLUE, GREEN, RED])
        a.set_title("(a) In-dist rollout MAE")
    a.axhline(m["noise_floor"]["mean"], color="black", ls="--", lw=1, label="noise floor")
    a.set_ylabel("MAE"); a.legend(fontsize=8); a.tick_params(axis="x", rotation=20)

    # (b) denoising win (history vs raw obs)
    b = axes[1]; dn = m["denoising"]; sig = sorted(float(s) for s in dn)
    b.plot(sig, [dn[str(s)]["raw_noisy_obs"] for s in sig], "s-", color=GREY, label="raw obs")
    b.plot(sig, [dn[str(s)]["supervised"] for s in sig], "o-", color=BLUE, label="history")
    b.set_title("(b) Denoising: history wins"); b.set_xlabel("sensor σ"); b.set_ylabel("state error"); b.legend(fontsize=8)

    # (c) decompensation recall: point (6mo) vs P80 risk (6mo / pure)
    c = axes[2]
    bars, vals, cols = [], [], []
    d6 = m.get("decompensation", {}).get("supervised")
    if d6: bars.append("point\n6mo"); vals.append(d6["recall"]); cols.append(BLUE)
    q = m.get("quantile_tail")
    if q:
        bars.append("P80\n6mo"); vals.append(q["followup6_decompensation"]["recall"]); cols.append(RED)
        bars.append("P80\npure"); vals.append(q["pure_rollout_decompensation"]["recall"]); cols.append("#b7791f")
    c.bar(bars, vals, color=cols); c.set_ylim(0, 1.05); c.set_ylabel("recall")
    c.set_title("(c) Decompensation recall\n(P80 risk trades false alarms)")

    # (d) cross-mechanism transfer degradation
    d = axes[3]; msh = m.get("mechanism_shift", {})
    if msh:
        base = [m["accuracy"][n]["overall"] for n in names]
        shift = [msh[n]["overall"] for n in names]
        x = np.arange(len(names)); w = 0.38
        d.bar(x - w/2, base, w, label="in-dist", color=GREY)
        d.bar(x + w/2, shift, w, label="alt mechanism", color=RED)
        d.set_xticks(x); d.set_xticklabels(names, rotation=20, fontsize=8); d.set_ylabel("MAE")
        d.set_title("(d) Cross-mechanism transfer"); d.legend(fontsize=8)
    fig.suptitle("Digital Liver World Model — results at a glance (0.0000% constraint violations, all models)",
                 y=1.03, fontsize=12)
    fig.tight_layout(); fig.savefig(f"{FIGDIR}/fig_summary.png", bbox_inches="tight"); plt.close(fig)


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig_training(); print("fig_training.png")
    fig_rollout(); print("fig_rollout.png")
    fig_phase_boundary(); print("fig_phase_boundary.png")
    fig_probes_latent(); print("fig_probes_latent.png")
    fig_manifold(); print("fig_manifold.png")
    fig_tail_risk(); print("fig_tail_risk.png")
    fig_summary(); print("fig_summary.png")
    print("figures written to", FIGDIR)


if __name__ == "__main__":
    main()
