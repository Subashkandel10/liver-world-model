"""
Data pipeline: turn generator output into model-ready tensors, define the train/val split,
and build the three generalisation-probe cohorts.

Everything a model may condition on lives in the context vector built here; the hidden
susceptibility is carried separately and only ever used by the OracleModel and by the
evaluation's decodability probe -- never leaked into a normal model's inputs.
"""

import numpy as np
import torch

from lwm.config import CTX_DIM, N_DISEASE_CLASSES, N_MONTHS_TRAIN
from lwm.generator import (
    generate_dataset, generate_mechanism_shift_dataset, SUSC_PROBE_LO, SUSC_PROBE_HI,
)


def build_context(patients, n_months):
    """Return (ctx [N, T, CTX_DIM], ercp [N, T], susc [N]).

    Context = disease-class one-hot(3) + age + sex + responder + udca_active(t) + ercp_now(t).
    responder is a GIVEN context constant per the spec, so the only hidden driver is
    susceptibility.
    """
    N = len(patients)
    ctx = np.zeros((N, n_months, CTX_DIM), dtype=np.float32)
    ercp = np.zeros((N, n_months), dtype=np.float32)
    susc = np.zeros(N, dtype=np.float32)
    for i, p in enumerate(patients):
        ercp_set = set(p.ercp_months)
        for t in range(n_months):
            ctx[i, t, p.disease_class] = 1.0
            ctx[i, t, N_DISEASE_CLASSES + 0] = p.age
            ctx[i, t, N_DISEASE_CLASSES + 1] = p.sex
            ctx[i, t, N_DISEASE_CLASSES + 2] = p.responder
            ctx[i, t, N_DISEASE_CLASSES + 3] = 1.0 if t >= p.udca_start else 0.0
            ctx[i, t, N_DISEASE_CLASSES + 4] = 1.0 if t in ercp_set else 0.0
        susc[i] = p.susceptibility
    return ctx, ercp, susc


class Cohort:
    """A bundle of aligned tensors for one set of patients."""
    def __init__(self, X, patients, n_months):
        ctx, ercp, susc = build_context(patients, n_months)
        self.X = torch.tensor(X)                 # [N, T, 8]
        self.ctx = torch.tensor(ctx)             # [N, T, CTX]
        self.ercp = torch.tensor(ercp)           # [N, T]
        self.susc = torch.tensor(susc)           # [N]
        self.patients = patients
        self.n_months = n_months
        self.N = X.shape[0]


def make_cohorts(n_train=800, n_val=200, n_test=200, n_months=N_MONTHS_TRAIN, seed=0):
    """Standard in-distribution train / val / test cohorts (censored susceptibility band).

    Three disjoint splits, drawn from non-overlapping per-patient seed ranges
    (``generate_dataset`` keys patient i off ``seed*100003 + i``):

      train -- fit model weights
      val   -- select the epoch / checkpoint (early stopping). NEVER reported.
      test  -- report every headline number. Touched only at evaluation time.

    Keeping selection (val) and reporting (test) on different patients is what makes the
    in-distribution accuracy honest: the checkpoint is chosen to look best on val, so
    reporting on val would be selection-biased -- an especially sharp risk here, where the
    model-vs-model gaps the memo leans on are ~0.005 MAE, well inside that bias.
    """
    Xtr, ptr = generate_dataset(n_train, n_months, seed=seed)
    Xva, pva = generate_dataset(n_val, n_months, seed=seed + 1)
    Xte, pte = generate_dataset(n_test, n_months, seed=seed + 2)
    return (Cohort(Xtr, ptr, n_months), Cohort(Xva, pva, n_months),
            Cohort(Xte, pte, n_months))


def make_probes(n=200, n_months=N_MONTHS_TRAIN, long_months=None, seed=100):
    """Three falsification probes, each varying ONE axis away from training:

      susceptibility -- held-out fast progressors (susc in the probe band, never trained on)
      timing         -- UDCA starts late (months 26-40), genuinely beyond the training regime
                        (training draws udca_start in 2-23, so 26-40 is never seen)
      long           -- a longer trajectory for the long-horizon rollout probe
    """
    Xs, ps = generate_dataset(n, n_months, seed=seed, susc_range=(SUSC_PROBE_LO, SUSC_PROBE_HI))
    Xt, pt = generate_dataset(n, n_months, seed=seed + 1, udca_start_range=(26, 40))
    probes = {
        "susceptibility": Cohort(Xs, ps, n_months),
        "timing": Cohort(Xt, pt, n_months),
    }
    if long_months is not None:
        Xl, pl = generate_dataset(n, long_months, seed=seed + 2)
        probes["long"] = Cohort(Xl, pl, long_months)
    return probes


def make_mechanism_shift_cohort(n=200, n_months=N_MONTHS_TRAIN, seed=404):
    """Evaluation-only cohort from a valid but independently parameterised mechanism."""
    X, patients = generate_mechanism_shift_dataset(n, n_months, seed=seed)
    return Cohort(X, patients, n_months)
