"""
Evaluation harness -- honest numbers, and the experiments that could falsify the thesis.

Sections:
  1. constraint_audit        -- violation rate per field (must be 0 by construction)
  2. noise_floor             -- irreducible aleatoric error (spread of independent re-runs)
  3. accuracy                -- held-out free-rollout MAE per field, read against the floor
  4. collapse                -- effective rank of the JEPA latent vs the data's intrinsic dim
  5. probes                  -- three generalisation probes (susceptibility / timing / long)
  6. decodability            -- what does the latent BUY? linear R^2(susceptibility) from
                                x(t) vs history-latent vs JEPA-latent, and the ORACLE GAP
  7. phase_boundary          -- the headline: how the oracle gap / history advantage OPENS
                                as clinical observation becomes sparse (re-anchor every k months)
  8. manifold_critic         -- a learned "is this transition on-manifold?" critic; shows
                                0 violations != on-manifold
  9. counterfactual          -- "UDCA 6 months earlier", model vs a shared-noise generator re-run

Run:  python -m lwm.evaluate   ->  eval_out/metrics.json + printed report
"""

import json
import os
import numpy as np
import torch

from lwm.config import FIELD_NAMES, N_FIELDS, HISTORY, MONOTONE_UP, FIELD_MAX, F, S, C, M, P, A, DECOMP_P_THRESHOLD, decompensation_month
from lwm.generator import (
    generate_dataset, simulate, draw_noise, resimulate, Patient, MONOTONE_UP as GEN_MONO,
)
from lwm.data import Cohort, make_cohorts, make_probes, make_mechanism_shift_cohort
from lwm.model import MODEL_REGISTRY, effective_rank
from lwm.train import free_rollout_mae


# --------------------------------------------------------------------------- helpers
def _bootstrap_recall_ci(hit_flags, alpha=0.10, n_boot=3000, seed=0):
    """90% bootstrap CI on a recall estimated from a small event pool.

    hit_flags: 1/0 per true-positive patient (detected or not). Returns [lo, hi]. Small-N event
    metrics (here ~11 decompensators) cannot support a point ranking; the CI makes that explicit.
    """
    hit_flags = np.asarray(hit_flags, dtype=float)
    if len(hit_flags) == 0:
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    boots = [hit_flags[rng.integers(0, len(hit_flags), len(hit_flags))].mean() for _ in range(n_boot)]
    return [float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))]


# --------------------------------------------------------------------------- load
def load_model(name, ckpt_dir="checkpoints"):
    blob = torch.load(os.path.join(ckpt_dir, f"{name}.pt"), weights_only=False)
    model = MODEL_REGISTRY[name]()
    model.load_state_dict(blob["state"])
    model.eval()
    return model, blob


# --------------------------------------------------------------- 1. constraint audit
@torch.no_grad()
def constraint_audit(model, coh, start=HISTORY):
    """Fraction of predicted transitions that violate a hard constraint. By construction of
    the head this should be exactly 0; we check the realised rollout anyway (the whole point
    is not to trust the loss to have learned it)."""
    _, _, preds, _ = free_rollout_mae(model, coh, K=coh.n_months - start, start=start)
    # prepend the true anchor state so we can check the first predicted step's monotonicity
    anchor = coh.X[:, start - 1:start]
    seq = torch.cat([anchor, preds], dim=1).numpy()      # [N, K+1, 8]
    ercp = coh.ercp.numpy()
    viol = {FIELD_NAMES[i]: 0 for i in MONOTONE_UP}
    viol["S_offERCP"] = 0
    viol["bounds"] = 0
    total = 0
    for i in range(seq.shape[0]):
        for t in range(1, seq.shape[1]):
            total += 1
            for idx in MONOTONE_UP:
                if seq[i, t, idx] < seq[i, t - 1, idx] - 1e-5:
                    viol[FIELD_NAMES[idx]] += 1
            month = start + t - 1
            if month < ercp.shape[1] and ercp[i, month] < 0.5:
                if seq[i, t, S] < seq[i, t - 1, S] - 1e-5:
                    viol["S_offERCP"] += 1
            if (seq[i, t, :7] < -1e-5).any() or (seq[i, t, :6] > 1 + 1e-5).any() or seq[i, t, M] > 2 + 1e-5:
                viol["bounds"] += 1
    return {k: v / max(total, 1) for k, v in viol.items()}, total


# ------------------------------------------------------------------ 2. noise floor
def noise_floor(n=300, n_months=48, seed=500):
    """Irreducible per-field error: MAE between two independent realisations of the SAME
    patient context (different flare/process noise). No model can beat this on the fast,
    stochastic channels -- so field errors are read against this, not against zero."""
    _, patients = generate_dataset(n, n_months, seed=seed)
    diffs = []
    for p in patients:
        a_base = 0.15 + 0.10 * p.disease_class
        x1 = simulate(p, n_months, draw_noise(np.random.default_rng(p.seed + 11), n_months, a_base, a_base))
        x2 = simulate(p, n_months, draw_noise(np.random.default_rng(p.seed + 29), n_months, a_base, a_base))
        diffs.append(np.abs(x1[HISTORY:] - x2[HISTORY:]).mean(0))
    floor = np.mean(diffs, 0)
    return {FIELD_NAMES[i]: float(floor[i]) for i in range(N_FIELDS)}, float(floor.mean())


# --------------------------------------------------------------------- 3. accuracy
@torch.no_grad()
def accuracy(model, coh):
    overall, per_field, _, _ = free_rollout_mae(model, coh, K=coh.n_months - HISTORY, start=HISTORY)
    return overall, {FIELD_NAMES[i]: float(per_field[i]) for i in range(N_FIELDS)}


@torch.no_grad()
def upper_tail_rollout(model, coh, start=HISTORY, offset=0.0, stride=999):
    """Autoregressive P80 trajectory, alongside the ordinary point rollout.

    The history window is updated with the point state; only P carries a separate upper-tail
    recurrence.  This avoids treating the P80 path as a physically observed state while retaining
    its monotone, bounded risk guarantee.  With stride<horizon the model re-anchors the window
    (and restarts the risk path from the observed P) at each clinic visit -- the realistic
    follow-up regime; stride>=horizon is the harder pure-rollout regime.
    """
    X, ctx, ercp = coh.X, coh.ctx, coh.ercp
    window, ctx_win = X[:, start - HISTORY:start].clone(), ctx[:, start - HISTORY:start].clone()
    tail_prev = window[:, -1, P].clone()
    points, tails, trues = [], [], []
    for k, tgt_t in enumerate(range(start, X.shape[1])):
        point, raw_tail_p, _ = model.step_with_tail(window, ctx_win, window[:, -1], tail_prev, ercp[:, tgt_t])
        # Split-conformal adjustment of the upper risk path.  Adding a non-negative scalar and
        # re-clipping preserves the P constraint while correcting systematic under-coverage.  The
        # *raw* tail remains the recurrence anchor so a fixed calibration correction is not added
        # once per month and spuriously compounded.
        tail_p = torch.clamp(raw_tail_p + offset, 0.0, 1.0)
        points.append(point); tails.append(tail_p); trues.append(X[:, tgt_t, P])
        visit = (k + 1) % stride == 0
        tail_prev = X[:, tgt_t, P].clone() if visit else raw_tail_p
        nxt = X[:, tgt_t] if visit else point
        window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
        ctx_win = torch.cat([ctx_win[:, 1:], ctx[:, tgt_t].unsqueeze(1)], dim=1)
    return torch.stack(points, 1), torch.stack(tails, 1), torch.stack(trues, 1)


@torch.no_grad()
def quantile_tail_metrics(model, coh, offset=0.0):
    """Calibration and proper-score metrics for the P80 risk path, not point accuracy."""
    _, tail, true = upper_tail_rollout(model, coh, offset=offset)
    q = model.tail_quantile
    err = true - tail
    pinball = torch.maximum(q * err, (q - 1.0) * err).mean().item()
    return {
        "quantile": q,
        "coverage": float((true <= tail).float().mean()),
        "pinball_loss": float(pinball),
        "tail_minus_truth_mae": float((tail - true).abs().mean()),
    }


@torch.no_grad()
def quantile_decompensation_detection(model, n=400, seed=900, n_months=48, offset=0.0, stride=999):
    """Event detection with the P80 risk path. stride>=horizon is pure rollout; stride=6 is the
    realistic 6-month follow-up. Reports the risk/false-alarm trade-off and a bootstrap recall CI."""
    X, patients = generate_dataset(n, n_months, seed=seed)
    coh = Cohort(X, patients, n_months)
    _, tail, _ = upper_tail_rollout(model, coh, offset=offset, stride=stride)
    true_dm = [decompensation_month(X[i]) for i in range(n)]
    true_pos = [i for i, dm in enumerate(true_dm) if dm is not None and dm > HISTORY]
    predicted = [np.where(tail[i].numpy() >= DECOMP_P_THRESHOLD)[0] for i in range(n)]
    detected = [i for i in true_pos if len(predicted[i])]
    false_alarms = [i for i, dm in enumerate(true_dm) if dm is None and len(predicted[i])]
    timing = [abs((HISTORY + int(predicted[i][0])) - true_dm[i]) for i in detected]
    return {
        "stride": stride,
        "n_true": len(true_pos),
        "recall": len(detected) / max(len(true_pos), 1),
        "recall_ci90": _bootstrap_recall_ci([1 if len(predicted[i]) else 0 for i in true_pos]),
        "detected": len(detected),
        "false_alarms": len(false_alarms),
        "timing_mae_months": float(np.mean(timing)) if timing else None,
    }


@torch.no_grad()
def conformal_tail_offset(model, calibration_coh):
    """One-sided split-conformal correction for the P80 path.

    This cohort is distinct from both model fitting and the reported evaluation cohort.  The
    returned residual quantile is applied uniformly at rollout time, preserving monotonicity and
    bounds while targeting marginal 80% coverage.
    """
    _, raw_tail, true = upper_tail_rollout(model, calibration_coh)
    residual = (true - raw_tail).reshape(-1)
    return float(torch.quantile(residual, model.tail_quantile).clamp(min=0.0))


# ---------------------------------------------------------- 6. decodability + oracle gap
@torch.no_grad()
def _latents_and_susc(model, coh, uses_history):
    """Encode each patient's history window at month HISTORY into a latent (or use x(t) for the
    memoryless case) and pair it with the true hidden susceptibility."""
    window = coh.X[:, :HISTORY]
    ctx_win = coh.ctx[:, :HISTORY]
    if uses_history and hasattr(model, "encode_online"):
        z = model.encode_online(window, ctx_win).numpy()
    elif uses_history and hasattr(model, "encoder"):
        z = model.encoder(window, ctx_win).numpy()
    else:
        z = coh.X[:, HISTORY - 1].numpy()               # x(t): the memoryless "latent"
    return z, coh.susc.numpy()


def _linear_r2(Z, y, Ztest, ytest):
    """Ridge-regression R^2 of y from Z (fit on train split, scored on test split)."""
    Z = np.concatenate([Z, np.ones((Z.shape[0], 1))], 1)
    Ztest = np.concatenate([Ztest, np.ones((Ztest.shape[0], 1))], 1)
    lam = 1e-2
    w = np.linalg.solve(Z.T @ Z + lam * np.eye(Z.shape[1]), Z.T @ y)
    pred = Ztest @ w
    ss_res = ((ytest - pred) ** 2).sum()
    ss_tot = ((ytest - ytest.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-12))


def decodability(models, coh_tr, coh_te):
    """R^2 of the hidden susceptibility recovered linearly from each representation.

    This is the direct measurement of "what does the latent hold that x(t) cannot": if the
    JEPA latent decodes susceptibility much better than raw x(t), the latent has abstracted the
    hidden cause; if not, x(t) is already sufficient and the latent buys nothing here.
    """
    out = {}
    reps = {
        "x(t) [memoryless]": (models["memoryless"], False),
        "history [supervised]": (models["supervised"], True),
        "JEPA latent": (models["jepa"], True),
    }
    for label, (m, uses_hist) in reps.items():
        Ztr, ytr = _latents_and_susc(m, coh_tr, uses_hist)
        Zte, yte = _latents_and_susc(m, coh_te, uses_hist)
        out[label] = _linear_r2(Ztr, ytr, Zte, yte)
    return out


# --------------------------------------------------------------- 7. phase boundary
@torch.no_grad()
def reanchor_rollout_mae(model, coh, stride, start=HISTORY):
    """Predict every month, but RE-ANCHOR to the true state every `stride` months (a clinic
    visit) -- between visits the model rolls its own prediction forward. Large stride = must
    extrapolate far from the last observation, where the hidden progression rate matters."""
    X, ctx, ercp, susc = coh.X, coh.ctx, coh.ercp, coh.susc
    N, T = X.shape[0], X.shape[1]
    H = HISTORY
    window = X[:, start - H:start].clone()
    ctx_win = ctx[:, start - H:start].clone()
    extra = susc.unsqueeze(1) if model.__class__.__name__ == "OracleModel" else None
    errs = []
    for k in range(T - start):
        tgt_t = start + k
        prev_x = window[:, -1]
        pred, _ = model.step(window, ctx_win, prev_x, ercp[:, tgt_t], extra=extra)
        errs.append((pred - X[:, tgt_t]).abs().mean().item())
        # re-anchor to truth at a visit month, else feed the prediction forward
        if (k + 1) % stride == 0:
            nxt = X[:, tgt_t]
        else:
            nxt = pred
        window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
        ctx_win = torch.cat([ctx_win[:, 1:], ctx[:, tgt_t].unsqueeze(1)], dim=1)
    return float(np.mean(errs))


def phase_boundary(models, coh, strides=(1, 3, 6, 12)):
    """The headline experiment: oracle gap and history advantage as a function of visit sparsity."""
    out = {}
    for stride in strides:
        row = {name: reanchor_rollout_mae(m, coh, stride) for name, m in models.items()}
        row["oracle_gap"] = row["memoryless"] - row["oracle"]
        row["history_gain"] = row["memoryless"] - row["supervised"]
        out[str(stride)] = row
    return out


# --------------------------------------------------------------- 8. manifold critic
def _repair_valid(prev, cand):
    """Make a candidate successor satisfy every HARD constraint (so a negative is valid-but-wrong,
    forcing the critic to learn dynamics rather than re-check bounds)."""
    cand = cand.copy()
    for idx in MONOTONE_UP:
        cand[:, idx] = np.maximum(cand[:, idx], prev[:, idx])
    cand[:, S] = np.maximum(cand[:, S], 0.0)          # S may fall (ERCP), but not below 0
    fmax = FIELD_MAX
    return np.clip(cand, 0.0, fmax)


def _build_negatives(X, seed=0):
    """Three grades of valid-but-wrong successor, from easy to hard:

      cross_patient  -- successor stolen from a different patient/month (easy: wrong scale)
      same_patient   -- successor is this patient's true next-state at a DIFFERENT month (medium:
                        right scale, wrong dynamics/timing)
      perturbed_real -- the true successor nudged off-manifold but kept valid (hard: almost right)
    """
    N, T, d = X.shape
    prev = X[:, :-1].reshape(-1, d)
    real_next = X[:, 1:].reshape(-1, d)
    rng = np.random.default_rng(seed)
    M = prev.shape[0]

    cross = _repair_valid(prev, real_next[rng.permutation(M)])

    # same-patient wrong-month: reshape to [N, T-1, d], roll the time axis within each patient
    rn = real_next.reshape(N, T - 1, d)
    shift = rng.integers(1, T - 1)
    same = _repair_valid(prev, np.roll(rn, shift, axis=1).reshape(M, d))

    # perturbed-real: push ratchets up a touch and jitter the free fields -- plausible but wrong
    pert = real_next.copy()
    for idx in MONOTONE_UP:
        pert[:, idx] = pert[:, idx] + np.abs(rng.normal(0, 0.03, M))
    for idx in (A, C):
        pert[:, idx] = pert[:, idx] + rng.normal(0, 0.06, M)
    perturbed = _repair_valid(prev, pert)
    return prev, real_next, {"cross_patient": cross, "same_patient": same, "perturbed_real": perturbed}


def _auc(pos_s, neg_s):
    return float((pos_s[:, None] > neg_s[None, :]).mean())


def manifold_critic(models, coh, epochs=400):
    """Train a critic to separate real transitions from valid-but-wrong ones, then score each
    model's rollout. Uses graded-difficulty negatives and reports per-grade AUC + calibration, so a
    weak overall AUC is diagnosed (which grade defeats it) rather than reported as one opaque number."""
    X = coh.X.numpy()
    prev, real_next, negs = _build_negatives(X)
    neg_all = np.concatenate(list(negs.values()), 0)
    prev_rep = np.tile(prev, (len(negs), 1))
    pos = np.concatenate([prev, real_next], 1)
    neg = np.concatenate([prev_rep, neg_all], 1)
    Z = np.concatenate([pos, neg], 0)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    Zt = torch.tensor(Z, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    critic = torch.nn.Sequential(torch.nn.Linear(2 * N_FIELDS, 128), torch.nn.SiLU(),
                                 torch.nn.Linear(128, 128), torch.nn.SiLU(),
                                 torch.nn.Linear(128, 64), torch.nn.SiLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(critic.parameters(), lr=2e-3)
    lossf = torch.nn.BCEWithLogitsLoss()
    n = len(y)
    rng = np.random.default_rng(0)
    for _ in range(epochs):
        bi = rng.integers(0, n, size=2048)
        opt.zero_grad()
        loss = lossf(critic(Zt[bi]).squeeze(1), yt[bi])
        loss.backward(); opt.step()
    with torch.no_grad():
        real_s = torch.sigmoid(critic(torch.tensor(pos, dtype=torch.float32)).squeeze(1)).numpy()
        per_grade_auc, neg_means = {}, {}
        for grade, fake in negs.items():
            neg_s = torch.sigmoid(critic(torch.tensor(np.concatenate([prev, fake], 1),
                                                      dtype=torch.float32)).squeeze(1)).numpy()
            per_grade_auc[grade] = _auc(real_s, neg_s)
            neg_means[grade] = float(neg_s.mean())
        all_neg_s = torch.sigmoid(critic(torch.tensor(neg, dtype=torch.float32)).squeeze(1)).numpy()
        result = {
            "critic_auc_valid_but_wrong": _auc(real_s, all_neg_s),
            "per_grade_auc": per_grade_auc,
            "real_transitions": float(real_s.mean()),
            "neg_mean_by_grade": neg_means,
        }
        for name, m in models.items():
            _, _, preds, _ = free_rollout_mae(m, coh, K=coh.n_months - HISTORY, start=HISTORY)
            anchor = coh.X[:, HISTORY - 1:HISTORY]
            seq = torch.cat([anchor, preds], 1)
            p = seq[:, :-1].reshape(-1, N_FIELDS)
            nx = seq[:, 1:].reshape(-1, N_FIELDS)
            result[name] = float(torch.sigmoid(critic(torch.cat([p, nx], 1)).squeeze(1)).mean().item())
    return result


# ------------------------------------------------- 5b. decompensation detection (clinical)
@torch.no_grad()
def _rollout_reanchor_seq(model, coh, stride, start=HISTORY):
    """Rollout that re-anchors to the true state every `stride` months (a clinic visit). Returns
    the predicted sequence [N, K+1, 8] including the anchor. stride>=horizon => pure free rollout."""
    X, ctx, ercp, susc = coh.X, coh.ctx, coh.ercp, coh.susc
    N, T = X.shape[0], X.shape[1]
    H = HISTORY
    window = X[:, start - H:start].clone()
    ctx_win = ctx[:, start - H:start].clone()
    extra = susc.unsqueeze(1) if model.__class__.__name__ == "OracleModel" else None
    seq = [X[:, start - 1]]
    for k in range(T - start):
        tgt_t = start + k
        pred, _ = model.step(window, ctx_win, window[:, -1], ercp[:, tgt_t], extra=extra)
        seq.append(pred)
        nxt = X[:, tgt_t] if (k + 1) % stride == 0 else pred
        window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
        ctx_win = torch.cat([ctx_win[:, 1:], ctx[:, tgt_t].unsqueeze(1)], dim=1)
    return torch.stack(seq, 1)


@torch.no_grad()
def decompensation_detection(models, n=400, seed=900, n_months=48, stride=6):
    """Clinical event metric under realistic follow-up: the model re-anchors to the true state
    every `stride` months and must flag portal-hypertension decompensation (P crossing threshold)
    BEFORE it happens. Reports recall, false-alarm count, and timing error. `stride` large ==
    the (much harder) pure-free-rollout regime."""
    from lwm.config import decompensation_month, DECOMP_P_THRESHOLD
    X, patients = generate_dataset(n, n_months, seed=seed)
    coh = Cohort(X, patients, n_months)
    true_dm = [decompensation_month(X[i]) for i in range(n)]
    true_pos = [i for i, dm in enumerate(true_dm) if dm is not None and dm > HISTORY]
    out = {}
    for name, m in models.items():
        seq = _rollout_reanchor_seq(m, coh, stride).numpy()
        predP = seq[:, :, P]
        pred_dm = {}
        for i in range(n):
            hit = np.where(predP[i] >= DECOMP_P_THRESHOLD)[0]
            pred_dm[i] = (HISTORY - 1 + int(hit[0])) if len(hit) else None
        detected = [i for i in true_pos if pred_dm[i] is not None]
        false_alarms = [i for i in range(n) if pred_dm[i] is not None and (true_dm[i] is None)]
        timing_err = [abs(pred_dm[i] - true_dm[i]) for i in detected]
        hit_flags = [1 if pred_dm[i] is not None else 0 for i in true_pos]
        out[name] = {
            "stride": stride,
            "n_true": len(true_pos),
            "recall": len(detected) / max(len(true_pos), 1),
            "recall_ci90": _bootstrap_recall_ci(hit_flags),
            "detected": len(detected),
            "false_alarms": len(false_alarms),
            "timing_mae_months": float(np.mean(timing_err)) if timing_err else None,
            "mean_pred_final_P_on_true": float(np.mean([predP[i, -1] for i in true_pos])),
            "mean_true_final_P_on_true": float(np.mean([X[i, -1, P] for i in true_pos])),
        }
    return out


# ---------------------------------------------------- 7b. denoising (the real latent win)
@torch.no_grad()
def denoising_experiment(ckpt_dir, coh, sigmas=(0.05, 0.10, 0.15)):
    """Estimate the TRUE current state from a NOISY history window. A memoryless model can only
    return the raw noisy observation; a history latent can filter. This is the regime where the
    predictive latent buys ACCURACY, not just auditability.

    Reports, per noise level: the raw-observation error (the memoryless floor) and the
    denoise error for the history/JEPA models trained under noise.
    """
    window = coh.X[:, :HISTORY]
    ctx_win = coh.ctx[:, :HISTORY]
    true_now = coh.X[:, HISTORY - 1]
    models = {}
    for n in ["supervised", "jepa"]:
        path = os.path.join(ckpt_dir, f"{n}_noisy.pt")
        if os.path.exists(path):
            m = MODEL_REGISTRY[n]()
            m.load_state_dict(torch.load(path, weights_only=False)["state"])
            m.eval()
            models[n] = m
    out = {}
    torch.manual_seed(0)
    for sigma in sigmas:
        noisy = window + sigma * torch.randn_like(window)
        raw_err = (noisy[:, -1] - true_now).abs().mean().item()   # memoryless: stuck with noise
        row = {"raw_noisy_obs": raw_err}
        for n, m in models.items():
            est = m.denoise(noisy, ctx_win)
            row[n] = (est - true_now).abs().mean().item()
        out[str(sigma)] = row
    return out


# ------------------------------------------- 7c. noisy + irregular full forecast (JEPA home turf)
@torch.no_grad()
def noisy_irregular_forecast(ckpt_dir, n=200, n_months=48, sigma=0.10, strides=(1, 3, 6), seed=920):
    """The regime the memo claims should favour a history latent: observations arrive only every
    `stride` months AND are corrupted by sensor noise. The model re-anchors to the NOISY observation
    at each visit and rolls forward between; error is measured against the CLEAN truth. A memoryless
    model is stuck with the noisy anchor; a history model can filter it. Uses the noise-augmented
    suite (trained for this regime) so the comparison is fair."""
    names = ["memoryless", "supervised", "jepa"]
    models = {}
    for nm in names:
        path = os.path.join(ckpt_dir, f"{nm}_noisy.pt")
        if os.path.exists(path):
            m = MODEL_REGISTRY[nm]()
            m.load_state_dict(torch.load(path, weights_only=False)["state"])
            m.eval(); models[nm] = m
    X, patients = generate_dataset(n, n_months, seed=seed)
    coh = Cohort(X, patients, n_months)
    Xc, ctx, ercp = coh.X, coh.ctx, coh.ercp
    H, start = HISTORY, HISTORY
    T = Xc.shape[1]
    torch.manual_seed(0)
    out = {}
    for stride in strides:
        row = {}
        for name, m in models.items():
            window = Xc[:, start - H:start] + sigma * torch.randn(Xc.shape[0], H, N_FIELDS)
            ctx_win = ctx[:, start - H:start].clone()
            errs = []
            for k in range(T - start):
                tgt_t = start + k
                pred, _ = m.step(window, ctx_win, window[:, -1], ercp[:, tgt_t])
                errs.append((pred - Xc[:, tgt_t]).abs().mean().item())     # vs CLEAN truth
                visit = (k + 1) % stride == 0
                nxt = (Xc[:, tgt_t] + sigma * torch.randn(Xc.shape[0], N_FIELDS)) if visit else pred
                window = torch.cat([window[:, 1:], nxt.unsqueeze(1)], dim=1)
                ctx_win = torch.cat([ctx_win[:, 1:], ctx[:, tgt_t].unsqueeze(1)], dim=1)
            row[name] = float(np.mean(errs))
        row["history_gain"] = row.get("memoryless", 0) - row.get("supervised", 0)
        row["jepa_vs_supervised"] = row.get("supervised", 0) - row.get("jepa", 0)
        out[str(stride)] = row
    return {"sigma": sigma, "by_stride": out}


# --------------------------------------------- 8b. graph-attention causal alignment
@torch.no_grad()
def graph_attention_readout(graph_model, coh):
    """What the causal-graph-attention encoder attends to. Because attention is hard-masked to
    the disease's causal parents, every weight already sits on a biological edge; this reports,
    per child field, how it splits its attention across those parents -- an auditable, causal
    account of the encoder's information flow (the auditability the memo claims for the route)."""
    if not hasattr(graph_model, "encoder") or not hasattr(graph_model.encoder, "last_attn"):
        return {}
    graph_model.encoder(coh.X[:, :HISTORY], coh.ctx[:, :HISTORY])   # populates last_attn
    attn = graph_model.encoder.last_attn.numpy()                     # [child, parent], row-normalised
    from lwm.config import CAUSAL_PARENTS
    out = {}
    for child, parents in CAUSAL_PARENTS.items():
        row = {FIELD_NAMES[p]: round(float(attn[child, p]), 3) for p in parents}
        out[FIELD_NAMES[child]] = dict(sorted(row.items(), key=lambda kv: -kv[1]))
    return out


# --------------------------------------------------------------- 9. counterfactual
@torch.no_grad()
def counterfactual(model, seed=700, n=120, n_months=48, shift=6):
    """Move UDCA start 6 months EARLIER for responders; compare the model's counterfactual
    rollout to a shared-noise generator re-run (ground-truth counterfactual)."""
    _, patients = generate_dataset(n, n_months, seed=seed)
    resp = [p for p in patients if p.responder and 8 <= p.udca_start < 40]
    if not resp:
        return {"n": 0}
    model_dm, true_dm = [], []
    for p in resp:
        new_start = max(2, p.udca_start - shift)
        # ground truth counterfactual via shared-noise generator re-run
        base_true = resimulate(p, n_months)
        cf_true = resimulate(p, n_months, udca_start=new_start)
        true_dm.append(cf_true[-1, M] - base_true[-1, M])
        # model counterfactual: same patient, edited UDCA timeline in the context
        base_coh = Cohort(base_true[None], [p], n_months)
        p_cf = Patient(**{**p.__dict__}); p_cf.udca_start = new_start
        cf_coh = Cohort(cf_true[None], [p_cf], n_months)
        _, _, bp, _ = free_rollout_mae(model, base_coh, K=n_months - HISTORY, start=HISTORY)
        _, _, cp, _ = free_rollout_mae(model, cf_coh, K=n_months - HISTORY, start=HISTORY)
        model_dm.append(float(cp[0, -1, M] - bp[0, -1, M]))
    true_dm, model_dm = np.array(true_dm), np.array(model_dm)
    return {
        "n": len(resp),
        "true_mean_deltaM": float(true_dm.mean()),
        "model_mean_deltaM": float(model_dm.mean()),
        "sign_agreement": float(np.mean(np.sign(true_dm) == np.sign(model_dm))),
        "corr": float(np.corrcoef(true_dm, model_dm)[0, 1]) if len(true_dm) > 2 else None,
    }


# --------------------------------------------------------------------- main
def run_all(ckpt_dir="checkpoints", out_dir="eval_out"):
    os.makedirs(out_dir, exist_ok=True)
    names = ["memoryless", "supervised", "graph", "oracle", "jepa"]
    models = {n: load_model(n, ckpt_dir)[0] for n in names}
    quantile_model = None
    quantile_path = os.path.join(ckpt_dir, "supervised_quantile.pt")
    if os.path.exists(quantile_path):
        quantile_model, _ = load_model("supervised_quantile", ckpt_dir)
    # coh_te is the reporting split: disjoint from both the training patients and the val
    # patients used to pick each model's checkpoint, so these numbers carry no selection bias.
    coh_tr, _coh_va, coh_te = make_cohorts(seed=0)
    probes = make_probes(long_months=96)
    mechanism_shift = make_mechanism_shift_cohort()
    metrics = {}

    # 1-3: accuracy, floor, violations
    floor_field, floor_mean = noise_floor()
    metrics["noise_floor"] = {"per_field": floor_field, "mean": floor_mean}
    metrics["accuracy"] = {}
    metrics["violations"] = {}
    for n in names:
        ov, pf = accuracy(models[n], coh_te)
        metrics["accuracy"][n] = {"overall": ov, "per_field": pf}
        v, total = constraint_audit(models[n], coh_te)
        metrics["violations"][n] = {"rate": v, "n_transitions": total}

    # 4: collapse
    z, _ = _latents_and_susc(models["jepa"], coh_te, True)
    metrics["collapse"] = {"jepa_effective_rank": effective_rank(torch.tensor(z)),
                           "latent_dim": models["jepa"].latent_dim}

    # 5: probes
    metrics["probes"] = {}
    base_ov = metrics["accuracy"]["jepa"]["overall"]
    for pname, pc in probes.items():
        row = {}
        for n in names:
            ov, pf = accuracy(models[n], pc)
            row[n] = {"overall": ov}
        metrics["probes"][pname] = row

    # A valid, separately parameterised disease mechanism.  This is deliberately not treated
    # as a benchmark to optimise: it makes the generator-inverter limitation measurable.
    metrics["mechanism_shift"] = {}
    for n in names:
        ov, pf = accuracy(models[n], mechanism_shift)
        metrics["mechanism_shift"][n] = {"overall": ov, "per_field": pf}

    if quantile_model is not None:
        # Fixed, independent split used only to calibrate the risk band, never to fit weights
        # and disjoint from the reported test split.
        _, calibration_coh, _ = make_cohorts(seed=2026)
        tail_offset = conformal_tail_offset(quantile_model, calibration_coh)
        metrics["quantile_tail"] = {
            "conformal_offset": tail_offset,
            "raw_calibration": quantile_tail_metrics(quantile_model, calibration_coh),
            "in_distribution": quantile_tail_metrics(quantile_model, coh_te, offset=tail_offset),
            "pure_rollout_decompensation": quantile_decompensation_detection(quantile_model, offset=tail_offset, stride=999),
            "followup6_decompensation": quantile_decompensation_detection(quantile_model, offset=tail_offset, stride=6),
        }

    # 6: decodability + oracle gap  (probe fit on train, scored on the held-out test split)
    metrics["decodability_R2_susceptibility"] = decodability(models, coh_tr, coh_te)
    om, _ = accuracy(models["oracle"], coh_te)
    mm, _ = accuracy(models["memoryless"], coh_te)
    metrics["oracle_gap_in_dist"] = mm - om

    # 7: phase boundary (headline)
    metrics["phase_boundary"] = phase_boundary(models, coh_te)
    metrics["phase_boundary_probe_susc"] = phase_boundary(models, probes["susceptibility"])

    # 5b: decompensation detection (clinical event metric) — realistic follow-up + pure rollout
    metrics["decompensation"] = decompensation_detection(models, stride=6)
    metrics["decompensation_pure_rollout"] = decompensation_detection(models, stride=999)

    # 7b: denoising (sensor-noise regime)
    metrics["denoising"] = denoising_experiment(ckpt_dir, coh_te)

    # 7c: noisy + irregular full forecast (does the history latent win when observation is degraded?)
    metrics["noisy_irregular"] = noisy_irregular_forecast(ckpt_dir)

    # 8: manifold critic
    metrics["manifold_critic"] = manifold_critic(models, coh_te)

    # 8b: graph-attention causal alignment (auditability of the recommended encoder)
    if "graph" in models:
        metrics["graph_attention"] = graph_attention_readout(models["graph"], coh_te)

    # 9: counterfactual
    metrics["counterfactual"] = counterfactual(models["jepa"])

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _print_report(metrics)
    return metrics


def _print_report(m):
    print("\n" + "=" * 70)
    print("DIGITAL LIVER WORLD MODEL — EVALUATION REPORT")
    print("=" * 70)
    print("\n[1] Held-out free-rollout MAE (months 12–48), read against the noise floor")
    print(f"    aleatoric noise floor (mean over fields): {m['noise_floor']['mean']:.4f}")
    for n, d in m["accuracy"].items():
        print(f"    {n:12s} overall MAE = {d['overall']:.4f}")
    print("\n[2] Constraint-violation rate (by construction -> expect 0)")
    for n, d in m["violations"].items():
        mx = max(d["rate"].values())
        print(f"    {n:12s} max field violation = {mx:.6f}  over {d['n_transitions']} transitions")
    print(f"\n[3] JEPA latent effective rank = {m['collapse']['jepa_effective_rank']:.2f} / "
          f"{m['collapse']['latent_dim']}  (measured data intrinsic dim 2.70)")
    print("\n[4] Generalisation probes (overall MAE)")
    for pname, row in m["probes"].items():
        s = "  ".join(f"{n}={row[n]['overall']:.3f}" for n in row)
        print(f"    {pname:14s} {s}")
    print("\n[4b] Cross-mechanism transfer (valid alternate generator; models were never trained on it)")
    for n, d in m.get("mechanism_shift", {}).items():
        print(f"    {n:12s} overall MAE = {d['overall']:.4f}")
    print("\n[5] What does the latent BUY? R^2(susceptibility) by representation")
    for k, v in m["decodability_R2_susceptibility"].items():
        print(f"    {k:22s} R^2 = {v:.3f}")
    print(f"    oracle gap (in-distribution) = {m['oracle_gap_in_dist']:+.4f}  "
          f"(<=0 => x(t) already sufficient)")
    if "quantile_tail" in m:
        q = m["quantile_tail"]
        d = q["in_distribution"]; e = q["pure_rollout_decompensation"]
        f6 = q.get("followup6_decompensation")
        print("\n[5a] P80 tail-risk forecast (proper score and calibration; not point accuracy)")
        print(f"    split-conformal offset={q['conformal_offset']:.4f}; coverage={d['coverage']:.3f} "
              f"(target {d['quantile']:.2f})  pinball={d['pinball_loss']:.4f}")
        ci = e.get("recall_ci90", [float('nan'), float('nan')])
        print(f"    pure-rollout   decomp recall={e['recall']:.2f} ({e['detected']}/{e['n_true']}) "
              f"CI90 [{ci[0]:.2f},{ci[1]:.2f}]  false_alarms={e['false_alarms']}")
        if f6:
            ci6 = f6.get("recall_ci90", [float('nan'), float('nan')])
            print(f"    6mo follow-up  decomp recall={f6['recall']:.2f} ({f6['detected']}/{f6['n_true']}) "
                  f"CI90 [{ci6[0]:.2f},{ci6[1]:.2f}]  false_alarms={f6['false_alarms']}")
    print("\n[6] PHASE BOUNDARY — MAE as clinic visits get sparse (re-anchor every k months)")
    print(f"    {'stride':>6}  {'memoryless':>10} {'supervised':>10} {'oracle':>8} "
          f"{'oracle_gap':>10} {'hist_gain':>9}")
    for stride, row in m["phase_boundary"].items():
        print(f"    {stride:>6}  {row['memoryless']:>10.4f} {row['supervised']:>10.4f} "
              f"{row['oracle']:>8.4f} {row['oracle_gap']:>+10.4f} {row['history_gain']:>+9.4f}")
    print("\n[5b] Decompensation detection — realistic 6-month follow-up (re-anchor) vs pure rollout")
    for tag, key in [("6mo follow-up", "decompensation"), ("pure rollout", "decompensation_pure_rollout")]:
        print(f"  {tag}:")
        for n, d in m.get(key, {}).items():
            tm = f"{d['timing_mae_months']:.1f}mo" if d["timing_mae_months"] is not None else "n/a"
            ci = d.get("recall_ci90", [float('nan'), float('nan')])
            print(f"    {n:12s} recall={d['recall']:.2f} ({d['detected']}/{d['n_true']}) "
                  f"CI90 [{ci[0]:.2f},{ci[1]:.2f}]  false_alarms={d['false_alarms']}  timing_MAE={tm}")
    print("\n[6b] DENOISING — estimate true current state from a NOISY window")
    print(f"    {'sigma':>6}  {'raw noisy obs':>14} {'supervised':>11} {'jepa':>8}")
    for sigma, row in m.get("denoising", {}).items():
        print(f"    {sigma:>6}  {row['raw_noisy_obs']:>14.4f} {row.get('supervised', float('nan')):>11.4f}"
              f" {row.get('jepa', float('nan')):>8.4f}")
    if "noisy_irregular" in m:
        ni = m["noisy_irregular"]
        print(f"\n[6c] Noisy (σ={ni['sigma']}) + irregular full forecast — rollout MAE vs clean truth "
              f"(noise-trained models)")
        print(f"    {'stride':>6}  {'memoryless':>10} {'supervised':>10} {'jepa':>8} {'hist_gain':>10} {'jepa-sup':>9}")
        for stride, row in ni["by_stride"].items():
            print(f"    {stride:>6}  {row.get('memoryless',float('nan')):>10.4f} "
                  f"{row.get('supervised',float('nan')):>10.4f} {row.get('jepa',float('nan')):>8.4f} "
                  f"{row.get('history_gain',0):>+10.4f} {row.get('jepa_vs_supervised',0):>+9.4f}")
    print("\n[7] Manifold critic (graded valid-but-wrong negatives; per-model rollout score)")
    mc = m["manifold_critic"]
    print(f"    overall AUC = {mc['critic_auc_valid_but_wrong']:.3f}  real transitions = {mc['real_transitions']:.3f}")
    if "per_grade_auc" in mc:
        pg = "  ".join(f"{g}={a:.3f}" for g, a in mc["per_grade_auc"].items())
        print(f"    per-grade AUC: {pg}")
    for n in ["memoryless", "supervised", "oracle", "jepa"]:
        print(f"    {n:12s} rollout on-manifold score = {mc[n]:.3f}")
    print("\n[8] Counterfactual: UDCA 6 months earlier (model vs shared-noise generator re-run)")
    cf = m["counterfactual"]
    if cf["n"]:
        print(f"    n={cf['n']}  true ΔM={cf['true_mean_deltaM']:+.4f}  model ΔM={cf['model_mean_deltaM']:+.4f}"
              f"  sign agreement={cf['sign_agreement']:.2f}")
    print("=" * 70)


if __name__ == "__main__":
    run_all()
