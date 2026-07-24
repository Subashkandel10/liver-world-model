"""
JEPA-loss ablation: isolate what the JEPA *objective* (the latent-prediction / invariance term)
buys, by training the identical model with that term's weight set to 0 (recon + VICReg only) and
comparing. This answers the memo's open question -- "JEPA buys auditability / on-manifold behaviour"
-- with a measurement instead of an assertion.

Run:  python -m lwm.ablation      (writes eval_out/ablation.json)
"""

import json
import os
import numpy as np
import torch

from lwm.config import HISTORY
from lwm.data import make_cohorts
from lwm.train import train_model, free_rollout_mae
from lwm.model import effective_rank
from lwm.evaluate import manifold_critic


def run(epochs=40, seed=0):
    coh_tr, coh_va, coh_te = make_cohorts(seed=seed)   # select on val, report on test
    variants = {
        "jepa_full": None,                       # default weights (latent-prediction ON)
        "jepa_no_latent": {"latent": 0.0},       # ablation: latent-prediction term OFF
    }
    trained, rows = {}, {}
    for label, ov in variants.items():
        m, _, _ = train_model("jepa", coh_tr, coh_va, epochs=epochs, seed=seed,
                              weight_overrides=ov, verbose=False)
        trained[label] = m
        mae, _, _, _ = free_rollout_mae(m, coh_te)
        z = m.encode_online(coh_te.X[:, :HISTORY], coh_te.ctx[:, :HISTORY]).detach()
        rows[label] = {"rollout_mae": mae, "effective_rank": effective_rank(z)}

    # on-manifold score for the two variants under one shared critic
    mc = manifold_critic(trained, coh_te)
    for label in variants:
        rows[label]["manifold_score"] = mc[label]
    result = {
        "seed": seed,
        "critic_auc": mc["critic_auc_valid_but_wrong"],
        "real_transitions": mc["real_transitions"],
        "variants": rows,
        "delta_from_latent_term": {
            "rollout_mae": rows["jepa_full"]["rollout_mae"] - rows["jepa_no_latent"]["rollout_mae"],
            "effective_rank": rows["jepa_full"]["effective_rank"] - rows["jepa_no_latent"]["effective_rank"],
            "manifold_score": rows["jepa_full"]["manifold_score"] - rows["jepa_no_latent"]["manifold_score"],
        },
    }
    os.makedirs("eval_out", exist_ok=True)
    with open("eval_out/ablation.json", "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== JEPA-loss ablation (latent-prediction term ON vs OFF) ===")
    print(f"{'variant':16s} {'rollout MAE':>12} {'eff rank':>10} {'manifold':>10}")
    for label, r in rows.items():
        print(f"{label:16s} {r['rollout_mae']:>12.4f} {r['effective_rank']:>10.2f} {r['manifold_score']:>10.3f}")
    d = result["delta_from_latent_term"]
    print(f"\nWhat the latent-prediction term buys (full - ablation):")
    print(f"  accuracy:      {d['rollout_mae']:+.4f} MAE   ({'better' if d['rollout_mae']<0 else 'worse/none'})")
    print(f"  effective rank:{d['effective_rank']:+.2f}")
    print(f"  manifold score:{d['manifold_score']:+.3f}")
    return result


if __name__ == "__main__":
    run()
