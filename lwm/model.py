"""
Models for the Digital Liver world model.

Design stance (argued in the memo): every model shares ONE by-construction constraint head
and ONE decoder, and they differ only in how they build the representation that feeds the
decoder. That is deliberate: it makes the comparison a clean test of "what does the learned
predictive latent buy?", because the constraint guarantee and the decoder capacity are held
fixed across all of them.

  MemorylessModel   -- "x(t) is the latent": MLP(x_t, ctx) -> raws.            (peer)
  SupervisedHistory -- history GRU -> z -> raws, trained on reconstruction.    (peer)
  OracleModel       -- memoryless + the TRUE hidden susceptibility as input.   (upper bound)
  JEPAModel         -- history GRU -> z, predictor -> z_pred, EMA target       (the subject)
                       encoder + VICReg anti-collapse; z_pred decoded by construction.

The head is where the physics lives:
  * F, D, P            next = prev + softplus(raw)                     (monotone up)
  * M                  next = prev + softplus(raw) * (prev_F * prev_C) (hazard of sustained F*C)
  * S                  next = prev + softplus(up) - is_ercp*sigmoid(relief)  (down only at ERCP)
  * A, C, flare        next = sigmoid(raw)                            (free, bounded)
A decrease in a ratchet field, or M rising with no F*C, is UNREPRESENTABLE -- not merely
penalised. The M coupling gate is the key move on the assignment's "coupling" tension: it
enforces the interaction structurally instead of hoping a learned scalar rate finds it.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from lwm.config import (
    F, D, S, P, A, C, M, FLARE, N_FIELDS, FIELD_MAX, CTX_DIM, causal_adjacency,
)

# raw-signal layout produced by every decoder: one per ratchet + 2 for S + one per free field
N_RAW = 9
R_F, R_D, R_P, R_M, R_S_UP, R_S_RELIEF, R_A, R_C, R_FLARE = range(N_RAW)


def constraint_head(prev_x, raws, ercp_now):
    """Map (previous state, raw decoder signals, ERCP flag) -> a guaranteed-valid next state.

    prev_x   [B, 8]  the state we are stepping from
    raws     [B, 9]  unconstrained decoder outputs
    ercp_now [B]     1.0 if an ERCP happens at the target month, else 0.0
    """
    B = prev_x.shape[0]
    nxt = torch.empty(B, N_FIELDS, device=prev_x.device, dtype=prev_x.dtype)
    sp = Fn.softplus

    # Ratchets: previous value + a strictly non-negative increment -> monotone non-decreasing.
    nxt[:, F] = prev_x[:, F] + sp(raws[:, R_F])
    nxt[:, D] = prev_x[:, D] + sp(raws[:, R_D])
    nxt[:, P] = prev_x[:, P] + sp(raws[:, R_P])

    # M: hazard accumulator. Increment is gated by prev_F * prev_C, so M can rise ONLY as a
    # hazard of sustained fibrosis-and-cholestasis. Coupling is structural, not learned-rate.
    m_inc = sp(raws[:, R_M]) * (prev_x[:, F] * prev_x[:, C])
    nxt[:, M] = prev_x[:, M] + m_inc

    # S: creeps up; an ERCP month unlocks a bounded step DOWN, otherwise S is monotone up too.
    s_up = sp(raws[:, R_S_UP])
    s_relief = ercp_now * torch.sigmoid(raws[:, R_S_RELIEF])
    nxt[:, S] = prev_x[:, S] + s_up - s_relief

    # Free fast fields: bounded but may move either way.
    nxt[:, A] = torch.sigmoid(raws[:, R_A])
    nxt[:, C] = torch.sigmoid(raws[:, R_C])
    nxt[:, FLARE] = torch.sigmoid(raws[:, R_FLARE])

    fmax = torch.tensor(FIELD_MAX, device=prev_x.device, dtype=prev_x.dtype)
    return torch.clamp(nxt, torch.zeros_like(fmax), fmax)


class MLP(nn.Module):
    def __init__(self, sizes, act=nn.SiLU):
        super().__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MemorylessModel(nn.Module):
    """'x(t) is the latent.' A Markov one-step predictor: MLP(x_t, ctx) -> raws -> head.

    Also given three hand-coded coupling features [F*C, flare*A, flare*C] so it does not
    have to rediscover the generator's multiplicative terms from scratch (an MLP would). The
    causal graph-attention encoder that routes these interactions structurally is built and
    measured as the `graph` peer (see CausalGraphEncoder and the memo): it ties this and the GRU
    on accuracy while adding an auditable causal-attention readout.
    """
    uses_jepa_loss = False
    latent_dim = 0

    def __init__(self, hidden=64, extra_in=0):
        super().__init__()
        in_dim = N_FIELDS + CTX_DIM + 3 + extra_in
        self.net = MLP([in_dim, hidden, hidden, N_RAW])

    def _features(self, x_t, ctx, extra=None):
        coup = torch.stack([x_t[:, F] * x_t[:, C], x_t[:, FLARE] * x_t[:, A],
                            x_t[:, FLARE] * x_t[:, C]], dim=1)
        parts = [x_t, ctx, coup]
        if extra is not None:
            parts.append(extra)
        return torch.cat(parts, dim=1)

    def step(self, window, ctx_seq, prev_x, ercp_now, extra=None):
        ctx_t = ctx_seq[:, -1]
        raws = self.net(self._features(prev_x, ctx_t, extra))
        return constraint_head(prev_x, raws, ercp_now), None


class OracleModel(MemorylessModel):
    """Upper bound: memoryless, but also handed the TRUE hidden susceptibility.

    The gap between this and MemorylessModel measures how much the unobserved susceptibility
    is worth *at all* -- the ceiling any latent that recovers it could hope to reach.
    """
    def __init__(self, hidden=64):
        super().__init__(hidden=hidden, extra_in=1)

    def step(self, window, ctx_seq, prev_x, ercp_now, extra=None):
        # `extra` carries the true susceptibility [B, 1]
        ctx_t = ctx_seq[:, -1]
        raws = self.net(self._features(prev_x, ctx_t, extra))
        return constraint_head(prev_x, raws, ercp_now), None


def ratchet_slope(window):
    """Per-field progression rate read from the observed window: (last - first)/(H-1).

    This is the serial-measurement slope a clinician reads off a chart -- and, for the ratchet
    fields, a direct observable proxy for the hidden per-patient susceptibility (which scales
    that slope). A memoryless model sees a single state and structurally cannot compute it; this
    feature is exactly what lets a history model extrapolate the rate for a fast progressor.
    """
    H = window.shape[1]
    return (window[:, -1] - window[:, 0]) / max(H - 1, 1)     # [B, 8]


class HistoryEncoder(nn.Module):
    """GRU over a window of (state, context) -> a latent z that can encode the hidden,
    slowly-varying per-patient susceptibility from the *shape* of the recent trajectory."""
    def __init__(self, latent_dim=16, hidden=48):
        super().__init__()
        self.gru = nn.GRU(N_FIELDS + CTX_DIM, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent_dim)

    def forward(self, window, ctx_seq):
        seq = torch.cat([window, ctx_seq], dim=-1)     # [B, H, 8+ctx]
        _, h = self.gru(seq)
        return self.to_latent(h[-1])                    # [B, latent]


class CausalGraphEncoder(nn.Module):
    """The recommended 'graph-attention encoder', built to measure rather than assert.

    Each of the 8 clinical fields is a node. Per timestep we embed every field's scalar value
    (plus a learned per-field type embedding and the shared context), run ONE multi-head
    self-attention layer over the 8 nodes **masked by the disease causal graph** so a field can
    attend only to its causal parents, mean-pool the nodes, and feed the per-timestep summary
    through a GRU over the window. Same forward signature as ``HistoryEncoder`` -> [B, latent],
    so it drops into the shared decoder/head unchanged and the comparison stays clean.

    Because the mask is the generator's own parent structure, all information flow is along
    biological edges by construction -- the attention weights are an auditable readout of which
    causal parent the encoder leaned on (exposed via ``last_attn`` for explainability).
    """
    def __init__(self, latent_dim=16, hidden=48, node_dim=24, heads=4):
        super().__init__()
        self.node_dim = node_dim
        self.value_proj = nn.Linear(1, node_dim)
        self.type_emb = nn.Parameter(torch.randn(N_FIELDS, node_dim) * 0.1)
        self.ctx_proj = nn.Linear(CTX_DIM, node_dim)
        self.attn = nn.MultiheadAttention(node_dim, heads, batch_first=True)
        self.ffn = MLP([node_dim, hidden, node_dim])
        self.norm = nn.LayerNorm(node_dim)
        self.gru = nn.GRU(node_dim, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent_dim)
        # additive attention mask: 0 on a causal edge, -inf elsewhere (query=child, key=parent)
        adj = torch.tensor(causal_adjacency())
        self.register_buffer("attn_mask", torch.where(
            adj, torch.zeros(N_FIELDS, N_FIELDS), torch.full((N_FIELDS, N_FIELDS), float("-inf"))))
        self.last_attn = None

    def forward(self, window, ctx_seq):
        B, H, _ = window.shape
        # node features per (batch, time): value + type + context, shape [B*H, 8, node_dim]
        vals = window.reshape(B * H, N_FIELDS, 1)
        nodes = self.value_proj(vals) + self.type_emb.unsqueeze(0)
        nodes = nodes + self.ctx_proj(ctx_seq.reshape(B * H, CTX_DIM)).unsqueeze(1)
        attended, attn_w = self.attn(nodes, nodes, nodes, attn_mask=self.attn_mask,
                                     need_weights=True, average_attn_weights=True)
        h = self.norm(nodes + attended)
        h = self.norm(h + self.ffn(h))
        self.last_attn = attn_w.detach().reshape(B, H, N_FIELDS, N_FIELDS).mean(dim=(0, 1))
        pooled = h.mean(dim=1).reshape(B, H, self.node_dim)     # mean over the 8 nodes
        _, hn = self.gru(pooled)
        return self.to_latent(hn[-1])


class SupervisedHistory(nn.Module):
    """History GRU -> z -> decode -> head. Trained on reconstruction only (no JEPA objective).

    Isolates whether the JEPA *latent-prediction* objective buys anything over simply giving a
    supervised decoder the same history.
    """
    uses_jepa_loss = False

    def __init__(self, latent_dim=16, hidden=48):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = HistoryEncoder(latent_dim, hidden)
        self.decoder = MLP([latent_dim + N_FIELDS + CTX_DIM + N_FIELDS, hidden, N_RAW])
        # denoise head: estimate the TRUE current state from the (possibly noisy) window.
        # A memoryless model structurally cannot do this -- it only sees the noisy anchor.
        self.denoise_head = MLP([latent_dim + N_FIELDS + CTX_DIM, hidden, N_FIELDS])

    def denoise(self, window, ctx_seq):
        z = self.encoder(window, ctx_seq)
        return self.denoise_head(torch.cat([z, window[:, -1], ctx_seq[:, -1]], dim=1))

    def step(self, window, ctx_seq, prev_x, ercp_now, extra=None):
        z = self.encoder(window, ctx_seq)
        slope = ratchet_slope(window)
        raws = self.decoder(torch.cat([z, prev_x, ctx_seq[:, -1], slope], dim=1))
        return constraint_head(prev_x, raws, ercp_now), z


class GraphAttnHistory(SupervisedHistory):
    """SupervisedHistory with the causal-graph-attention encoder in place of the GRU.

    Identical training objective, decoder, and by-construction head as ``supervised`` -- the ONLY
    change is how the history window is encoded. So the head-to-head against ``supervised`` isolates
    exactly what routing information along the disease's causal edges buys, and lets the memo report
    the graph-attention route as a measured result rather than a rejected assertion.
    """
    def __init__(self, latent_dim=16, hidden=48):
        super().__init__(latent_dim=latent_dim, hidden=hidden)
        self.encoder = CausalGraphEncoder(latent_dim, hidden)


class QuantileHistory(SupervisedHistory):
    """History model with a separate, constraint-preserving upper-tail forecast for P.

    The ordinary output remains a point forecast, scored with the same reconstruction loss as
    ``SupervisedHistory``.  A second raw signal learns the 80th conditional quantile of portal
    hypertension through pinball loss.  It is intentionally reported as *risk*, not as a more
    accurate point prediction: a calibrated P80 path should be above the realised P about 80% of
    the time and may trade false alarms for missed decompensations.
    """
    uses_tail_loss = True
    tail_quantile = 0.80

    def __init__(self, latent_dim=16, hidden=48):
        super().__init__(latent_dim=latent_dim, hidden=hidden)
        # The final component is a P-only upper-tail increment.  Everything else still uses the
        # exact same shared by-construction head as the point forecast.
        self.decoder = MLP([latent_dim + N_FIELDS + CTX_DIM + N_FIELDS, hidden, N_RAW + 1])

    def _raws(self, window, ctx_seq, prev_x):
        z = self.encoder(window, ctx_seq)
        slope = ratchet_slope(window)
        return self.decoder(torch.cat([z, prev_x, ctx_seq[:, -1], slope], dim=1)), z

    def step(self, window, ctx_seq, prev_x, ercp_now, extra=None):
        raws, z = self._raws(window, ctx_seq, prev_x)
        return constraint_head(prev_x, raws[:, :N_RAW], ercp_now), z

    def step_with_tail(self, window, ctx_seq, prev_x, tail_prev_p, ercp_now):
        """Return the point state and a monotone, bounded P80 state value.

        The tail forecast is parameterised as ``tail_prev + softplus(raw)`` and then ordered
        above the point P.  Thus a risk path can neither reverse portal hypertension nor leave
        [0, 1], even under autoregressive rollout.
        """
        raws, z = self._raws(window, ctx_seq, prev_x)
        point = constraint_head(prev_x, raws[:, :N_RAW], ercp_now)
        tail_p = torch.clamp(tail_prev_p + Fn.softplus(raws[:, N_RAW]), 0.0, 1.0)
        return point, torch.maximum(point[:, P], tail_p), z


class JEPAModel(nn.Module):
    """History-window JEPA: encode -> predict next latent (matched to an EMA target encoder
    on the true next window) -> decode by construction. VICReg guards the latent against
    dimensional collapse.
    """
    uses_jepa_loss = True

    def __init__(self, latent_dim=16, hidden=48, ema_decay=0.99):
        super().__init__()
        self.latent_dim = latent_dim
        self.ema_decay = ema_decay
        self.encoder = HistoryEncoder(latent_dim, hidden)
        self.predictor = MLP([latent_dim, hidden, latent_dim])
        self.decoder = MLP([latent_dim + N_FIELDS + CTX_DIM + N_FIELDS, hidden, N_RAW])
        self.denoise_head = MLP([latent_dim + N_FIELDS + CTX_DIM, hidden, N_FIELDS])
        # EMA target encoder: a stop-grad copy updated by exponential moving average.
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def denoise(self, window, ctx_seq):
        z = self.encoder(window, ctx_seq)
        return self.denoise_head(torch.cat([z, window[:, -1], ctx_seq[:, -1]], dim=1))

    @torch.no_grad()
    def update_target(self):
        for tp, op in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            tp.mul_(self.ema_decay).add_(op, alpha=1 - self.ema_decay)

    def step(self, window, ctx_seq, prev_x, ercp_now, extra=None):
        z = self.encoder(window, ctx_seq)
        z_pred = self.predictor(z)
        slope = ratchet_slope(window)
        raws = self.decoder(torch.cat([z_pred, prev_x, ctx_seq[:, -1], slope], dim=1))
        return constraint_head(prev_x, raws, ercp_now), z_pred

    def encode_online(self, window, ctx_seq):
        return self.encoder(window, ctx_seq)

    @torch.no_grad()
    def encode_target(self, window, ctx_seq):
        return self.target_encoder(window, ctx_seq)

    def predict_latent(self, z):
        return self.predictor(z)


# --------------------------- anti-collapse (VICReg) + collapse metric --------------------

def vicreg_loss(z, gamma=1.0, eps=1e-4):
    """VICReg variance-hinge + covariance-decorrelation on a batch of latents.

    variance: hinge each dimension's std up to gamma, so no dimension is allowed to go silent.
    covariance: push off-diagonal covariance to zero, so dimensions cannot become copies.
    Returns (var_term, cov_term).
    """
    z = z - z.mean(0, keepdim=True)
    std = torch.sqrt(z.var(0) + eps)
    var_term = torch.mean(Fn.relu(gamma - std))
    n, d = z.shape
    cov = (z.T @ z) / (n - 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_term = (off_diag ** 2).sum() / d
    return var_term, cov_term


@torch.no_grad()
def effective_rank(z):
    """Participation ratio of the latent covariance eigenspectrum: exp(entropy of normalised
    eigenvalues). A soft-collapse detector -- catches redundancy a per-dim variance check misses.
    Compare against the data's INTRINSIC dimension, not the nominal latent width.
    """
    z = z - z.mean(0, keepdim=True)
    cov = (z.T @ z) / (z.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp(min=0)
    p = eig / (eig.sum() + 1e-12)
    p = p[p > 0]
    return float(torch.exp(-(p * torch.log(p)).sum()))


MODEL_REGISTRY = {
    "memoryless": MemorylessModel,
    "supervised": SupervisedHistory,
    "graph": GraphAttnHistory,
    "supervised_quantile": QuantileHistory,
    "oracle": OracleModel,
    "jepa": JEPAModel,
}
