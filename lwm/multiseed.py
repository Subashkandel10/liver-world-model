"""
Multi-seed robustness check for the headline numbers. All three prior attempts (and the first
cut of this one) rested load-bearing claims on a single training seed; this script retrains the
models across several seeds and reports mean +/- std for the claims the memo leans on, so a tie
of 0.005 MAE or a one-patient recall difference is read against its actual run-to-run spread.

Run:  python -m lwm.multiseed            (writes eval_out/multiseed.json)
"""

import json
import os
import numpy as np
import torch

from lwm.config import HISTORY, P
from lwm.data import make_cohorts, Cohort, make_mechanism_shift_cohort, make_probes
from lwm.generator import generate_dataset
from lwm.train import train_model, free_rollout_mae
from lwm.model import effective_rank
from lwm.evaluate import (accuracy, conformal_tail_offset, quantile_decompensation_detection)

SEEDS = [0, 1, 2]


def _denoise_error(model, coh, sigma=0.10):
    window = coh.X[:, :HISTORY]
    ctx_win = coh.ctx[:, :HISTORY]
    true_now = coh.X[:, HISTORY - 1]
    torch.manual_seed(0)
    noisy = window + sigma * torch.randn_like(window)
    raw = (noisy[:, -1] - true_now).abs().mean().item()
    with torch.no_grad():
        est = model.denoise(noisy, ctx_win)
    return raw, (est - true_now).abs().mean().item()


def _decomp_recall(models, stride=6, n=400, seed=900, n_months=48):
    from lwm.config import decompensation_month, DECOMP_P_THRESHOLD
    from lwm.evaluate import _rollout_reanchor_seq
    X, patients = generate_dataset(n, n_months, seed=seed)
    coh = Cohort(X, patients, n_months)
    true_dm = [decompensation_month(X[i]) for i in range(n)]
    true_pos = [i for i, dm in enumerate(true_dm) if dm is not None and dm > HISTORY]
    out = {}
    for name, m in models.items():
        seq = _rollout_reanchor_seq(m, coh, stride).numpy()
        det = sum(1 for i in true_pos for hit in [np.where(seq[i, :, P] >= DECOMP_P_THRESHOLD)[0]] if len(hit))
        out[name] = det / max(len(true_pos), 1)
    return out, len(true_pos)


def run():
    per_seed = []
    for s in SEEDS:
        print(f"\n===== seed {s} =====")
        coh_tr, coh_va, coh_te = make_cohorts(seed=s)   # select on val, report on test
        rec = {"seed": s, "mae": {}, "denoise": {}, "eff_rank": None, "recall": {}}
        clean = {}
        for name in ["memoryless", "supervised", "graph", "oracle", "jepa"]:
            m, _, _ = train_model(name, coh_tr, coh_va, epochs=40, seed=s, verbose=False)
            clean[name] = m
            mae, _, _, _ = free_rollout_mae(m, coh_te)
            rec["mae"][name] = mae
        # effective rank of the jepa latent
        z = clean["jepa"].encode_online(coh_te.X[:, :HISTORY], coh_te.ctx[:, :HISTORY]).detach()
        rec["eff_rank"] = effective_rank(z)
        # noisy suite for denoise
        for name in ["supervised", "jepa"]:
            m, _, _ = train_model(name, coh_tr, coh_va, epochs=40, seed=s, obs_noise=0.10, verbose=False)
            raw, err = _denoise_error(m, coh_te, sigma=0.10)
            rec["denoise"]["raw"] = raw
            rec["denoise"][name] = err
        # decompensation recall
        rec["recall"], rec["n_true"] = _decomp_recall(clean)

        # quantile model: point MAE + calibrated P80 tail recall (pure + 6mo follow-up)
        qm, _, _ = train_model("supervised_quantile", coh_tr, coh_va, epochs=40, seed=s, verbose=False)
        _, cal, _ = make_cohorts(seed=s + 2026)
        off = conformal_tail_offset(qm, cal)
        rec["quantile_mae"] = free_rollout_mae(qm, coh_te)[0]
        rec["quantile_recall_pure"] = quantile_decompensation_detection(qm, offset=off, stride=999)["recall"]
        rec["quantile_recall_6mo"] = quantile_decompensation_detection(qm, offset=off, stride=6)["recall"]

        # cross-mechanism transfer + susceptibility probe (per model, held-out)
        mech = make_mechanism_shift_cohort()
        susc = make_probes()["susceptibility"]
        rec["mechanism"] = {k: accuracy(m, mech)[0] for k, m in clean.items()}
        rec["probe_susc"] = {k: accuracy(m, susc)[0] for k, m in clean.items()}

        per_seed.append(rec)
        print(f"  MAE {({k: round(v,4) for k,v in rec['mae'].items()})}  eff_rank {rec['eff_rank']:.2f}")
        print(f"  denoise@0.1 raw {rec['denoise']['raw']:.3f} sup {rec['denoise']['supervised']:.3f} jepa {rec['denoise']['jepa']:.3f}")
        print(f"  decomp recall {({k: round(v,2) for k,v in rec['recall'].items()})}")
        print(f"  quantile MAE {rec['quantile_mae']:.4f}  P80 recall pure {rec['quantile_recall_pure']:.2f} / 6mo {rec['quantile_recall_6mo']:.2f}")

    # aggregate
    def agg(path):
        vals = [_get(r, path) for r in per_seed]
        return {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    def _get(r, path):
        cur = r
        for p in path:
            cur = cur[p]
        return cur

    summary = {"seeds": SEEDS, "n_true_decomp": per_seed[0]["n_true"]}
    for name in ["memoryless", "supervised", "graph", "oracle", "jepa"]:
        summary.setdefault("mae", {})[name] = agg(["mae", name])
        summary.setdefault("recall", {})[name] = agg(["recall", name])
    summary["eff_rank"] = agg(["eff_rank"])
    summary["denoise_raw"] = agg(["denoise", "raw"])
    summary["denoise_supervised"] = agg(["denoise", "supervised"])
    summary["denoise_jepa"] = agg(["denoise", "jepa"])
    summary["quantile_mae"] = agg(["quantile_mae"])
    summary["quantile_recall_pure"] = agg(["quantile_recall_pure"])
    summary["quantile_recall_6mo"] = agg(["quantile_recall_6mo"])
    for name in ["memoryless", "supervised", "graph", "oracle", "jepa"]:
        summary.setdefault("mechanism", {})[name] = agg(["mechanism", name])
        summary.setdefault("probe_susc", {})[name] = agg(["probe_susc", name])
    summary["per_seed"] = per_seed

    os.makedirs("eval_out", exist_ok=True)
    with open("eval_out/multiseed.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n===== MULTI-SEED SUMMARY (mean +/- std over seeds", SEEDS, ") =====")
    for name in ["memoryless", "supervised", "graph", "oracle", "jepa"]:
        a = summary["mae"][name]
        print(f"  MAE {name:12s} {a['mean']:.4f} +/- {a['std']:.4f}")
    print(f"  eff_rank       {summary['eff_rank']['mean']:.2f} +/- {summary['eff_rank']['std']:.2f}")
    print(f"  denoise raw    {summary['denoise_raw']['mean']:.3f} +/- {summary['denoise_raw']['std']:.3f}")
    print(f"  denoise sup    {summary['denoise_supervised']['mean']:.3f} +/- {summary['denoise_supervised']['std']:.3f}")
    print(f"  denoise jepa   {summary['denoise_jepa']['mean']:.3f} +/- {summary['denoise_jepa']['std']:.3f}")
    for name in ["memoryless", "supervised", "graph", "oracle", "jepa"]:
        a = summary["recall"][name]
        print(f"  recall {name:12s} {a['mean']:.2f} +/- {a['std']:.2f}")
    qm, qp, q6 = summary["quantile_mae"], summary["quantile_recall_pure"], summary["quantile_recall_6mo"]
    print(f"  quantile MAE   {qm['mean']:.4f} +/- {qm['std']:.4f}")
    print(f"  P80 recall pure {qp['mean']:.2f} +/- {qp['std']:.2f}   6mo {q6['mean']:.2f} +/- {q6['std']:.2f}")
    for name in ["memoryless", "supervised", "graph", "oracle", "jepa"]:
        a = summary["mechanism"][name]; b = summary["probe_susc"][name]
        print(f"  {name:12s} cross-mech {a['mean']:.4f}+/-{a['std']:.4f}   susc-probe {b['mean']:.4f}+/-{b['std']:.4f}")


if __name__ == "__main__":
    run()
