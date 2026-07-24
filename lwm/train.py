"""
Training: scheduled-sampling multistep free-rollout, with a JEPA latent-prediction objective
and VICReg anti-collapse for the JEPA model, reconstruction for the peers.

Why multistep + scheduled sampling: a one-step teacher-forced model never sees its own
mistakes, so it drifts badly on autoregressive rollout at eval. We roll K steps during
training and, with a ramping probability, feed the model's own prediction back into its
history window -- so it learns to recover from its own errors.
"""

import time
import numpy as np
import torch

from lwm.config import HISTORY, ROLLOUT_TRAIN, FIELD_WEIGHTS, P
from lwm.model import MODEL_REGISTRY, vicreg_loss, effective_rank
from lwm.data import make_cohorts

_FW = torch.tensor(FIELD_WEIGHTS)
_FW = _FW / _FW.mean()          # normalised so the overall loss scale is unchanged


def _weighted_mse(pred, target):
    return torch.mean(_FW * (pred - target) ** 2)


def _pinball(pred, target, quantile):
    """Asymmetric quantile loss: under-predicting an upper-tail target costs more."""
    err = target - pred
    return torch.maximum(quantile * err, (quantile - 1.0) * err).mean()


def _sample_anchors(N, T, H, K, batch, rng):
    idx = rng.integers(0, N, size=batch)
    t0 = rng.integers(H, T - K, size=batch)          # need H history + K future steps
    return torch.tensor(idx), torch.tensor(t0)


def rollout_loss(model, coh, idx, t0, K, p_ss, weights, obs_noise=0.0):
    """One batched K-step rollout. Returns (loss, logs, latents_for_metric).

    obs_noise>0 injects sensor noise into the OBSERVED history window (targets stay clean) and
    trains a denoising objective for models that can filter -- the setup where a history latent
    can beat a memoryless model that is stuck with the raw noisy observation.
    """
    X, ctx, ercp, susc = coh.X, coh.ctx, coh.ercp, coh.susc
    B = idx.shape[0]
    H = HISTORY

    # initial history window ending at t0 (indices t0-H+1 .. t0)
    win_idx = (t0[:, None] - (H - 1) + torch.arange(H)[None, :])     # [B, H]
    window = X[idx[:, None], win_idx]                                # [B, H, 8]
    ctx_win = ctx[idx[:, None], win_idx]                             # [B, H, CTX]

    denoise_l = 0.0
    if obs_noise > 0:
        clean_anchor = window[:, -1].clone()
        window = window + obs_noise * torch.randn_like(window)       # noisy measurements
        if hasattr(model, "denoise"):
            denoise_l = torch.mean((model.denoise(window, ctx_win) - clean_anchor) ** 2)

    recon, latent, var_t, cov_t, tail_t = 0.0, 0.0, 0.0, 0.0, 0.0
    last_z = None
    extra = susc[idx].unsqueeze(1) if model.__class__.__name__ == "OracleModel" else None

    for k in range(K):
        tgt_t = t0 + k + 1
        prev_x = window[:, -1]
        ercp_now = ercp[idx, tgt_t]
        pred, z = model.step(window, ctx_win, prev_x, ercp_now, extra=extra)
        true_next = X[idx, tgt_t]
        recon = recon + _weighted_mse(pred, true_next)

        if getattr(model, "uses_tail_loss", False):
            _, tail_p, _ = model.step_with_tail(window, ctx_win, prev_x, prev_x[:, P], ercp_now)
            tail_t = tail_t + _pinball(tail_p, true_next[:, P], model.tail_quantile)

        if model.uses_jepa_loss:
            # target: EMA encoder on the TRUE window ending at tgt_t
            tw_idx = (tgt_t[:, None] - (H - 1) + torch.arange(H)[None, :])
            z_tgt = model.encode_target(X[idx[:, None], tw_idx], ctx[idx[:, None], tw_idx])
            latent = latent + torch.mean((z - z_tgt) ** 2)
            v, c = vicreg_loss(z)
            var_t, cov_t = var_t + v, cov_t + c
            last_z = z

        # scheduled sampling: with prob p_ss slide the model's own prediction into the window
        use_pred = (torch.rand(B) < p_ss).float().unsqueeze(1)
        next_state = use_pred * pred + (1 - use_pred) * true_next
        window = torch.cat([window[:, 1:], next_state.unsqueeze(1)], dim=1)
        ctx_win = torch.cat([ctx_win[:, 1:], ctx[idx, tgt_t].unsqueeze(1)], dim=1)

    loss = weights["recon"] * recon / K
    logs = {"recon": float((recon / K).detach())}
    if obs_noise > 0 and not isinstance(denoise_l, float):
        loss = loss + weights.get("denoise", 1.0) * denoise_l
        logs["denoise"] = float(denoise_l.detach())
    if model.uses_jepa_loss:
        loss = loss + (weights["latent"] * latent + weights["var"] * var_t
                       + weights["cov"] * cov_t) / K
        logs.update(latent=float((latent / K).detach()), var=float((var_t / K).detach()),
                    cov=float((cov_t / K).detach()))
    if getattr(model, "uses_tail_loss", False):
        loss = loss + weights["tail"] * tail_t / K
        logs["tail"] = float((tail_t / K).detach())
    return loss, logs, last_z


@torch.no_grad()
def free_rollout_mae(model, coh, K=None, start=HISTORY):
    """Held-out free-rollout MAE: encode the first `start` months, then predict forward K steps
    feeding predictions back. This is the honest autoregressive number, not one-step teacher-forced.
    """
    X, ctx, ercp, susc = coh.X, coh.ctx, coh.ercp, coh.susc
    N, T = X.shape[0], X.shape[1]
    if K is None:
        K = T - start
    H = HISTORY
    window = X[:, start - H:start].clone()
    ctx_win = ctx[:, start - H:start].clone()
    extra = susc.unsqueeze(1) if model.__class__.__name__ == "OracleModel" else None
    preds, trues = [], []
    for k in range(K):
        tgt_t = start + k
        if tgt_t >= T:
            break
        prev_x = window[:, -1]
        pred, _ = model.step(window, ctx_win, prev_x, ercp[:, tgt_t], extra=extra)
        preds.append(pred)
        trues.append(X[:, tgt_t])
        window = torch.cat([window[:, 1:], pred.unsqueeze(1)], dim=1)
        ctx_win = torch.cat([ctx_win[:, 1:], ctx[:, tgt_t].unsqueeze(1)], dim=1)
    preds = torch.stack(preds, 1)
    trues = torch.stack(trues, 1)
    per_field = (preds - trues).abs().mean(dim=(0, 1))
    return float((preds - trues).abs().mean()), per_field.numpy(), preds, trues


def train_model(name, coh_tr, coh_va, epochs=40, batch=256, steps=60, lr=2e-3,
                K=ROLLOUT_TRAIN, seed=0, verbose=True, obs_noise=0.0, weight_overrides=None):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = MODEL_REGISTRY[name]()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    weights = {"recon": 1.0, "latent": 0.5, "var": 1.0, "cov": 0.05, "denoise": 1.0, "tail": 2.0}
    if weight_overrides:                          # e.g. {"latent": 0.0} for the JEPA-loss ablation
        weights.update(weight_overrides)
    N, T = coh_tr.X.shape[0], coh_tr.X.shape[1]
    H = HISTORY
    history = {"val_mae": [], "eff_rank": []}
    best = {"val": 1e9, "state": None, "epoch": -1}

    for ep in range(epochs):
        p_ss = min(0.5, ep / (epochs * 0.5)) * 0.5      # ramp teacher-forcing -> self-rollout
        model.train()
        for _ in range(steps):
            idx, t0 = _sample_anchors(N, T, H, K, batch, rng)
            loss, logs, _ = rollout_loss(model, coh_tr, idx, t0, K, p_ss, weights, obs_noise=obs_noise)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 5.0)
            opt.step()
            if model.uses_jepa_loss:
                model.update_target()
        model.eval()
        val_mae, _, _, _ = free_rollout_mae(model, coh_va, K=T - H, start=H)
        er = np.nan
        if model.uses_jepa_loss:
            with torch.no_grad():
                idx, t0 = _sample_anchors(N, T, H, K, 400, rng)
                _, _, z = rollout_loss(model, coh_tr, idx, t0, K, 0.3, weights)
                er = effective_rank(z) if z is not None else np.nan
        history["val_mae"].append(val_mae)
        history["eff_rank"].append(er)
        if val_mae < best["val"]:
            best.update(val=val_mae, state={k: v.clone() for k, v in model.state_dict().items()},
                        epoch=ep)
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            msg = f"[{name}] ep{ep:2d}  val_rollout_MAE={val_mae:.4f}"
            if model.uses_jepa_loss:
                msg += f"  eff_rank={er:.2f}  (recon={logs['recon']:.4f} lat={logs.get('latent',0):.4f})"
            print(msg)

    model.load_state_dict(best["state"])
    if verbose:
        print(f"[{name}] best val_rollout_MAE={best['val']:.4f} @ epoch {best['epoch']}")
    return model, history, best


def train_all(epochs=40, seed=0, save_dir="checkpoints", obs_noise=0.10):
    import os
    os.makedirs(save_dir, exist_ok=True)
    t0 = time.time()
    coh_tr, coh_va, _ = make_cohorts(seed=seed)   # test split untouched during training
    results = {}
    # clean suite: the fair peer comparison on fully-observed data
    for name in ["memoryless", "supervised", "graph", "supervised_quantile", "oracle", "jepa"]:
        model, hist, best = train_model(name, coh_tr, coh_va, epochs=epochs, seed=seed)
        torch.save({"state": model.state_dict(), "history": hist, "best_val": best["val"],
                    "best_epoch": best["epoch"], "name": name},
                   os.path.join(save_dir, f"{name}.pt"))
        results[name] = best["val"]
    # noisy suite: sensor-noise regime where a history latent can denoise the state estimate
    for name in ["memoryless", "supervised", "jepa"]:
        model, hist, best = train_model(name, coh_tr, coh_va, epochs=epochs, seed=seed,
                                        obs_noise=obs_noise, verbose=False)
        torch.save({"state": model.state_dict(), "history": hist, "best_val": best["val"],
                    "best_epoch": best["epoch"], "name": name, "obs_noise": obs_noise},
                   os.path.join(save_dir, f"{name}_noisy.pt"))
        results[name + "_noisy"] = best["val"]
    print(f"\nAll models trained in {time.time()-t0:.1f}s")
    print("best held-out free-rollout MAE:")
    for k, v in results.items():
        print(f"  {k:14s} {v:.4f}")
    return results


if __name__ == "__main__":
    import sys
    ep = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    train_all(epochs=ep)
