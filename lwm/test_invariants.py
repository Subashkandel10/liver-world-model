"""
Invariant tests: prove the hard guarantees hold as PROPERTIES of the parameterisation, not as
something the trained weights happened to learn. Every test drives the constraint head with
random weights and adversarial random inputs -- if a guarantee can be broken, an untrained model
with wild activations is the most likely thing to break it.

Run:  python -m lwm.test_invariants     (exits non-zero on any failure -- CI-gateable)
"""

import sys
import numpy as np
import torch

from lwm.config import F, D, S, P, A, C, M, FLARE, FIELD_MAX, N_FIELDS, cirrhosis_stage
from lwm.model import constraint_head, N_RAW, JEPAModel, QuantileHistory, effective_rank
from lwm.generator import generate_dataset, MONOTONE_UP

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print(f"  [{PASS if cond else FAIL}] {name}")
    assert cond, name          # real assertion so pytest (and the CI script) fail on violation


def test_monotone_by_construction():
    """Ratchet fields never decrease, for any random previous state and any random raw signals."""
    rng = torch.Generator().manual_seed(0)
    prev = torch.rand(5000, N_FIELDS, generator=rng)
    prev[:, M] *= 2
    raws = torch.randn(5000, N_RAW, generator=rng) * 5      # adversarially wild
    ercp = (torch.rand(5000, generator=rng) < 0.5).float()
    nxt = constraint_head(prev, raws, ercp)
    for idx in MONOTONE_UP:
        check(f"{['F','D','S','P','A','C','M','flare'][idx]} never decreases",
              (nxt[:, idx] >= prev[:, idx] - 1e-6).all().item())


def test_S_drops_only_at_ercp():
    rng = torch.Generator().manual_seed(1)
    prev = torch.rand(5000, N_FIELDS, generator=rng)
    raws = torch.randn(5000, N_RAW, generator=rng) * 5
    no_ercp = torch.zeros(5000)
    nxt = constraint_head(prev, raws, no_ercp)
    check("S never decreases when there is no ERCP", (nxt[:, S] >= prev[:, S] - 1e-6).all().item())


def test_bounds():
    rng = torch.Generator().manual_seed(2)
    prev = torch.rand(5000, N_FIELDS, generator=rng)
    prev[:, M] *= 2
    raws = torch.randn(5000, N_RAW, generator=rng) * 5
    ercp = (torch.rand(5000, generator=rng) < 0.5).float()
    nxt = constraint_head(prev, raws, ercp)
    fmax = torch.tensor(FIELD_MAX)
    check("all fields stay within [0, field_max]",
          ((nxt >= -1e-6).all() and (nxt <= fmax + 1e-6).all()).item())


def test_M_gated_by_FC():
    """M cannot rise when there is no sustained F*C (C = 0), for any raw signal."""
    rng = torch.Generator().manual_seed(3)
    prev = torch.rand(5000, N_FIELDS, generator=rng)
    prev[:, M] *= 2
    prev[:, C] = 0.0                                        # no cholestasis -> no hazard
    raws = torch.randn(5000, N_RAW, generator=rng) * 5
    nxt = constraint_head(prev, raws, torch.zeros(5000))
    check("M is frozen when C=0 (hazard gated by F*C)", (nxt[:, M] - prev[:, M]).abs().max().item() < 1e-6)


def test_cirrhosis_derived_monotone():
    F_vals = np.linspace(0, 1, 100)
    stage = cirrhosis_stage(F_vals)
    check("derived cirrhosis stage is monotone non-decreasing in F", bool((np.diff(stage) >= 0).all()))
    check("cirrhosis is never stored as a state channel", N_FIELDS == 8)


def test_head_expressive():
    """The head must be able to represent ANY valid transition, so the guarantee costs no reachable
    accuracy. We invert real transitions, including ERCP drops and saturated boundaries, solve
    for the raw signals, and round-trip them through the head."""
    X, patients = generate_dataset(50, 24, seed=7)
    prev = torch.tensor(X[:, :-1].reshape(-1, N_FIELDS))
    nxt = torch.tensor(X[:, 1:].reshape(-1, N_FIELDS))
    ercp = torch.tensor(np.array([
        [float(t in set(p.ercp_months)) for t in range(1, X.shape[1])]
        for p in patients
    ], dtype=np.float32).reshape(-1))
    # Closed-form inverse of the head.  At a true ERCP decrease, set the upward component
    # effectively to zero and route the full drop through the ERCP-only relief component.
    raws = torch.zeros(prev.shape[0], N_RAW)
    inv_softplus = lambda y: torch.log(torch.expm1(torch.clamp(y, min=1e-12)))
    raws[:, 0] = inv_softplus(nxt[:, F] - prev[:, F])
    raws[:, 1] = inv_softplus(nxt[:, D] - prev[:, D])
    raws[:, 2] = inv_softplus(nxt[:, P] - prev[:, P])
    fc = (prev[:, F] * prev[:, C]).clamp(min=1e-6)
    raws[:, 3] = inv_softplus((nxt[:, M] - prev[:, M]).clamp(min=1e-6) / fc)
    s_delta = nxt[:, S] - prev[:, S]
    s_drop = (ercp > 0.5) & (s_delta < 0)
    raws[:, 4] = inv_softplus(torch.where(s_drop, torch.full_like(s_delta, 1e-12), s_delta.clamp(min=1e-12)))
    logit = lambda y: torch.log(y.clamp(1e-6, 1 - 1e-6) / (1 - y.clamp(1e-6, 1 - 1e-6)))
    raws[:, 5] = torch.where(s_drop, logit((-s_delta).clamp(1e-6, 1 - 1e-6)),
                             torch.full_like(s_delta, -30.0))
    raws[:, 6] = logit(nxt[:, A]); raws[:, 7] = logit(nxt[:, C]); raws[:, 8] = logit(nxt[:, FLARE])
    recon = constraint_head(prev, raws, ercp)
    err = (recon - nxt).abs().max().item()
    check("head reproduces valid transitions, including ERCP drops (< 1e-3)", err < 1e-3)


def test_ema_target_no_grad():
    """The JEPA target encoder must receive no gradient while the online encoder does."""
    m = JEPAModel()
    w = torch.rand(8, 12, N_FIELDS)
    from lwm.config import CTX_DIM
    c = torch.rand(8, 12, CTX_DIM)
    z_online = m.encode_online(w, c)
    z_target = m.encode_target(w, c)
    (z_online.sum()).backward()
    online_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in m.encoder.parameters())
    target_has_grad = any(p.grad is not None for p in m.target_encoder.parameters())
    check("online encoder receives gradient", online_has_grad)
    check("EMA target encoder receives NO gradient", not target_has_grad)


def test_tail_risk_constraints():
    """The P80 risk path itself must stay ordered, monotone, and bounded under random weights."""
    rng = torch.Generator().manual_seed(4)
    m = QuantileHistory()
    window = torch.rand(256, 12, N_FIELDS, generator=rng)
    window[:, :, M] *= 2
    from lwm.config import CTX_DIM
    ctx = torch.rand(256, 12, CTX_DIM, generator=rng)
    prev = window[:, -1]
    tail_prev = torch.rand(256, generator=rng)
    point, tail_p, _ = m.step_with_tail(window, ctx, prev, tail_prev, torch.zeros(256))
    check("P80 risk path is bounded, monotone, and above the point P",
          ((tail_p >= tail_prev - 1e-6) & (tail_p >= point[:, P] - 1e-6) & (tail_p <= 1 + 1e-6)).all().item())


def test_no_split_leakage():
    """Regression test for the model-selection leak: the train / val / test cohorts must be
    disjoint patients, so the checkpoint is never chosen on the set it is later reported on."""
    from lwm.data import make_cohorts
    tr, va, te = make_cohorts(n_train=60, n_val=40, n_test=40, n_months=18, seed=0)
    s_tr = {p.seed for p in tr.patients}
    s_va = {p.seed for p in va.patients}
    s_te = {p.seed for p in te.patients}
    check("train/val patients are disjoint", len(s_tr & s_va) == 0)
    check("train/test patients are disjoint", len(s_tr & s_te) == 0)
    check("val/test patients are disjoint (selection set != reporting set)", len(s_va & s_te) == 0)


def test_graph_attention_is_causal():
    """The causal-graph-attention encoder must place ZERO attention weight off the disease's
    causal edges -- information flow is causal by construction, not learned."""
    from lwm.model import GraphAttnHistory
    from lwm.config import CTX_DIM, causal_adjacency
    m = GraphAttnHistory()
    m.encoder(torch.rand(16, 12, N_FIELDS), torch.rand(16, 12, CTX_DIM))
    attn = m.encoder.last_attn.numpy()                    # [child, parent]
    off_edge = attn[~causal_adjacency()]
    check("graph-attention puts no weight on non-causal edges",
          float(off_edge.max()) < 1e-6 if off_edge.size else True)


def main():
    print("Invariant tests (random weights + adversarial inputs):")
    test_monotone_by_construction()
    test_S_drops_only_at_ercp()
    test_bounds()
    test_M_gated_by_FC()
    test_cirrhosis_derived_monotone()
    test_head_expressive()
    test_ema_target_no_grad()
    test_tail_risk_constraints()
    test_no_split_leakage()
    test_graph_attention_is_causal()
    n_fail = sum(1 for _, ok in results if not ok)
    print(f"\n{len(results) - n_fail}/{len(results)} passed.")
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
