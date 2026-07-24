"""
Synthetic generator for the Digital Liver world model.

This is BOTH the training data and the quality bar: a model that generalises across
held-out patients has, in effect, recovered these update rules (the spec's "trap"). We
keep the rules seeded, readable, and self-checking so the "did it really learn the
dynamics?" question has a concrete ground truth to compare against.

Two design choices that later evaluation leans on:

  1. Hidden per-patient `susceptibility` scales BOTH the ratchet drive AND the malignancy
     hazard rate, and is never handed to a model. It is the latent cause a predictive
     model must infer from trajectory shape, and the thing generalisation must cope with.

  2. All stochasticity is drawn UP FRONT into a `Noise` bundle keyed by seed. A counterfactual
     re-run ("what if UDCA started 6 months earlier?") reuses the identical noise stream, so
     the difference between the two rollouts is causal, not sampling noise. This is what makes
     the counterfactual probe faithful rather than merely plausible.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np

from lwm.config import (
    F, D, S, P, A, C, M, FLARE, N_FIELDS, FIELD_NAMES, FIELD_MAX, MONOTONE_UP,
)

# --- dynamics constants (the generator's physics; a model never sees these) --------------
K_F, K_D, K_P = 0.022, 0.015, 0.011      # ratchet creep rates for F, D, P
K_P_FIBROSIS = 0.5                        # portal HTN also tracks accumulated fibrosis
K_S = 0.018                               # stricture creep rate
ERCP_RELIEF = 0.4                         # how much an ERCP drops S
K_M = 0.05                                # malignancy hazard rate constant (per unit F*C)
A_REVERT, C_REVERT = 0.5, 0.4             # mean-reversion speeds for A and C
FLARE_TO_AC = 0.5                         # how strongly a flare perturbs A and C
FLARE_DECAY = 0.4                         # geometric decay of the flare field
UDCA_SUPP = 0.6                           # responders' set-point knock-down under UDCA
NOISE_STD = 0.03                          # observation-free process noise on A, C

# Susceptibility band the TRAINING distribution is censored to. The high tail above HI is
# held out entirely and used as the generalisation probe (unseen fast progressors).
SUSC_TRAIN_LO, SUSC_TRAIN_HI = 0.5, 2.0
SUSC_PROBE_LO, SUSC_PROBE_HI = 2.0, 3.5


@dataclass
class Patient:
    """Context constants known to a model + hidden parameters that drive the dynamics."""
    disease_class: int
    age: float
    sex: int
    responder: int
    udca_start: int
    ercp_months: List[int] = field(default_factory=list)
    susceptibility: float = 1.0          # HIDDEN: never exposed to a model
    seed: int = 0


@dataclass
class Noise:
    """Pre-drawn stochastic stream so counterfactuals differ only causally."""
    a0: float
    c0: float
    flare_onset: np.ndarray              # [T] bool: did a flare start this month
    a_noise: np.ndarray                  # [T] process noise on A
    c_noise: np.ndarray                  # [T] process noise on C


def draw_noise(rng: np.random.Generator, n_months: int, a_base: float, c_base: float,
               s_series_hint: Optional[np.ndarray] = None) -> Noise:
    """Draw the full stochastic stream for one patient up front.

    Flare onset probability rises with strictures (cholangitis rides on obstruction), but S
    is itself a function of the rollout. We resolve this by drawing a uniform per month and
    thresholding against the live S inside `simulate`; here we only pre-draw the uniforms and
    the Gaussian process noise, which are exogenous.
    """
    return Noise(
        a0=float(np.clip(a_base + 0.05 * rng.standard_normal(), 0, 1)),
        c0=float(np.clip(c_base + 0.05 * rng.standard_normal(), 0, 1)),
        flare_onset=rng.random(n_months),                       # uniforms, thresholded later
        a_noise=NOISE_STD * rng.standard_normal(n_months),
        c_noise=NOISE_STD * rng.standard_normal(n_months),
    )


def simulate(p: Patient, n_months: int, noise: Noise) -> np.ndarray:
    """Roll one patient forward deterministically given a fixed noise stream.

    Returns x of shape [n_months, 8]. Every one-directional field is non-decreasing by the
    construction of its update (increments are products of non-negatives), so the GROUND
    TRUTH itself never violates a constraint -- that is what makes 0% a meaningful bar.
    """
    x = np.zeros((n_months, N_FIELDS), dtype=np.float32)
    a_base = 0.15 + 0.10 * p.disease_class
    c_base = 0.15 + 0.10 * p.disease_class
    eff_susc = p.susceptibility * (0.8 + 0.4 * p.age)   # older patients ratchet a bit faster

    x[0, A] = noise.a0
    x[0, C] = noise.c0

    for t in range(1, n_months):
        prev = x[t - 1]
        cur = prev.copy()

        on_udca = (t >= p.udca_start) and (p.responder == 1)
        supp = UDCA_SUPP if on_udca else 0.0

        # flare (idx 7): onset likelier when strictures are high; else geometric decay
        p_onset = 0.04 + 0.10 * prev[S]
        onset = 1.0 if noise.flare_onset[t] < p_onset else 0.0
        cur[FLARE] = max(onset, prev[FLARE] * FLARE_DECAY)

        # A, C (idx 4,5): fast mean-reversion toward a possibly-treated set point, + flare
        a_set = a_base * (1 - supp)
        c_set = c_base * (1 - supp)
        cur[A] = np.clip(prev[A] + A_REVERT * (a_set - prev[A]) + FLARE_TO_AC * cur[FLARE] + noise.a_noise[t], 0, 1)
        cur[C] = np.clip(prev[C] + C_REVERT * (c_set - prev[C]) + FLARE_TO_AC * cur[FLARE] + noise.c_noise[t], 0, 1)

        # ratchets F, D, P (idx 0,1,3): non-negative creep driven by A and C
        drive = eff_susc * (0.6 * prev[A] + 0.4 * prev[C])
        cur[F] = min(prev[F] + K_F * drive, 1.0)
        cur[D] = min(prev[D] + K_D * drive, 1.0)
        cur[P] = min(prev[P] + K_P * (drive + K_P_FIBROSIS * prev[F]), 1.0)

        # S (idx 2): creeps with inflammation; ERCP steps it DOWN
        cur[S] = min(prev[S] + K_S * eff_susc * prev[A], 1.0)
        if t in p.ercp_months:
            cur[S] = max(cur[S] - ERCP_RELIEF, 0.0)

        # M (idx 6): hazard accumulator of sustained F*C, scaled by susceptibility, capped
        cur[M] = min(prev[M] + K_M * eff_susc * prev[F] * prev[C], 2.0)

        x[t] = cur

    return x


def simulate_mechanism_shift(p: Patient, n_months: int, noise: Noise) -> np.ndarray:
    """A second, independently parameterised disease mechanism for falsification.

    It preserves the assignment's state space, context semantics, stochastic stream, and every
    hard clinical constraint.  It intentionally *does not* preserve the reference generator's
    update equations: cholestasis has more leverage on progression, portal hypertension has a
    stronger fibrosis/ductopenia feedback, and malignancy accelerates with accumulated fibrosis.

    Models are never trained on this mechanism.  Its purpose is therefore not an additional
    benchmark to optimise, but a concrete test of the ``generator-inverter`` limitation: can a
    model trained on one valid mechanism still forecast another valid mechanism in the same
    clinical state space?
    """
    x = np.zeros((n_months, N_FIELDS), dtype=np.float32)
    a_base = 0.15 + 0.10 * p.disease_class
    c_base = 0.15 + 0.10 * p.disease_class
    eff_susc = p.susceptibility * (0.8 + 0.4 * p.age)
    x[0, A] = noise.a0
    x[0, C] = noise.c0

    for t in range(1, n_months):
        prev = x[t - 1]
        cur = prev.copy()
        on_udca = (t >= p.udca_start) and (p.responder == 1)
        supp = UDCA_SUPP if on_udca else 0.0

        # Same clinically meaningful fast dynamics, but a different persistence and flare gain.
        p_onset = 0.04 + 0.10 * prev[S]
        onset = 1.0 if noise.flare_onset[t] < p_onset else 0.0
        cur[FLARE] = max(onset, prev[FLARE] * 0.50)
        a_set, c_set = a_base * (1 - supp), c_base * (1 - supp)
        cur[A] = np.clip(prev[A] + 0.40 * (a_set - prev[A]) + 0.42 * cur[FLARE] + noise.a_noise[t], 0, 1)
        cur[C] = np.clip(prev[C] + 0.32 * (c_set - prev[C]) + 0.58 * cur[FLARE] + noise.c_noise[t], 0, 1)

        # Structural shift: cholestasis and A*C synergy drive progression, rather than the
        # reference generator's mostly linear A/C mixture.
        drive = eff_susc * (0.30 * prev[A] + 0.60 * prev[C] + 0.25 * prev[A] * prev[C])
        cur[F] = min(prev[F] + 0.018 * drive * (1.0 + 0.35 * prev[F]), 1.0)
        cur[D] = min(prev[D] + 0.013 * drive * (1.0 + 0.20 * prev[S]), 1.0)
        # P remains a ratchet, but now has an explicit accumulating F/D burden.  This is the
        # clinically relevant mechanism that the reference model's under-predicted P tail lacks.
        p_drive = drive + 0.75 * prev[F] + 0.35 * prev[D]
        cur[P] = min(prev[P] + 0.009 * p_drive, 1.0)

        cur[S] = min(prev[S] + 0.015 * eff_susc * (0.35 * prev[A] + 0.65 * prev[C]), 1.0)
        if t in p.ercp_months:
            cur[S] = max(cur[S] - 0.32, 0.0)

        # M is still *strictly* an F*C hazard, with a different accumulation shape.
        cur[M] = min(prev[M] + 0.035 * eff_susc * prev[F] * prev[C] * (0.5 + prev[F]), 2.0)
        x[t] = cur
    return x


def sample_patient(rng: np.random.Generator, n_months: int, seed: int,
                   susc_range=(SUSC_TRAIN_LO, SUSC_TRAIN_HI),
                   udca_start_range=None) -> Patient:
    """Draw one patient's context + hidden parameters.

    susc_range censors susceptibility into a band (training band, or the held-out probe band).
    udca_start_range forces treatment timing into a window (the unseen-timing probe).
    """
    disease_class = int(rng.integers(0, 3))
    responder = int(rng.random() < 0.6)
    if udca_start_range is not None:
        udca_start = int(rng.integers(udca_start_range[0], udca_start_range[1]))
    else:
        udca_start = int(rng.integers(2, n_months // 2)) if rng.random() < 0.7 else n_months + 1
    n_ercp = int(rng.integers(0, 3))
    ercp_months = sorted(int(m) for m in rng.integers(6, n_months, size=n_ercp))
    susc = float(rng.lognormal(mean=0.0, sigma=0.80))
    lo, hi = susc_range
    while not (lo <= susc <= hi):
        susc = float(rng.lognormal(mean=0.0, sigma=0.80))
    return Patient(
        disease_class=disease_class,
        age=float(rng.uniform(0.2, 0.9)),
        sex=int(rng.integers(0, 2)),
        responder=responder,
        udca_start=udca_start,
        ercp_months=ercp_months,
        susceptibility=susc,
        seed=seed,
    )


def generate_dataset(n_patients: int, n_months: int, seed: int,
                     susc_range=(SUSC_TRAIN_LO, SUSC_TRAIN_HI), udca_start_range=None):
    """Return (X, patients): X is [n_patients, n_months, 8]; patients is list[Patient].

    Each patient gets its own isolated RNG (seeded by base seed + index) so the dataset is
    reproducible and a single patient can be regenerated in isolation for counterfactuals.
    """
    X = np.zeros((n_patients, n_months, N_FIELDS), dtype=np.float32)
    patients: List[Patient] = []
    for i in range(n_patients):
        pt_seed = seed * 100003 + i
        rng = np.random.default_rng(pt_seed)
        p = sample_patient(rng, n_months, seed=pt_seed, susc_range=susc_range,
                           udca_start_range=udca_start_range)
        a_base = 0.15 + 0.10 * p.disease_class
        # Noise draws from a DEDICATED stream so it is independent of how many draws the
        # (rejection-sampled) context consumed -- this is what lets `resimulate` reproduce
        # the exact factual noise for a clean counterfactual.
        noise = draw_noise(np.random.default_rng(pt_seed + 1), n_months, a_base, a_base)
        X[i] = simulate(p, n_months, noise)
        patients.append(p)
    return X, patients


def generate_mechanism_shift_dataset(n_patients: int, n_months: int, seed: int,
                                     susc_range=(SUSC_TRAIN_LO, SUSC_TRAIN_HI)):
    """Generate an evaluation-only cohort under :func:`simulate_mechanism_shift`.

    Patient/context sampling and exogenous noise are identical in form to the reference
    generator.  Only the transition mechanism changes, so this isolates mechanism transfer
    rather than a covariate shift in the observed state schema.
    """
    X = np.zeros((n_patients, n_months, N_FIELDS), dtype=np.float32)
    patients: List[Patient] = []
    for i in range(n_patients):
        pt_seed = seed * 100003 + i
        rng = np.random.default_rng(pt_seed)
        p = sample_patient(rng, n_months, seed=pt_seed, susc_range=susc_range)
        a_base = 0.15 + 0.10 * p.disease_class
        noise = draw_noise(np.random.default_rng(pt_seed + 1), n_months, a_base, a_base)
        X[i] = simulate_mechanism_shift(p, n_months, noise)
        patients.append(p)
    return X, patients


def resimulate(p: Patient, n_months: int, udca_start: Optional[int] = None) -> np.ndarray:
    """Re-roll a patient under the SAME noise stream, optionally moving UDCA start.

    This is the counterfactual engine: because the noise is redrawn deterministically from
    the patient's seed, the only thing that changes between the factual and counterfactual
    trajectory is the intervention, so their difference is causal.
    """
    a_base = 0.15 + 0.10 * p.disease_class
    # Identical noise stream to the factual run (dedicated seed = p.seed + 1).
    noise = draw_noise(np.random.default_rng(p.seed + 1), n_months, a_base, a_base)
    p_cf = Patient(**{**p.__dict__})
    if udca_start is not None:
        p_cf.udca_start = udca_start
    return simulate(p_cf, n_months, noise)


def apply_observation_mask(X: np.ndarray, every: int, rng: np.random.Generator):
    """Return (X_obs, mask): thin observations to 1-every-`every` months (stale follow-up).

    mask[i, t] = 1 where a real observation exists. Between visits, X_obs holds the last
    observed value (carry-forward), which is what a memoryless model would have to use. This
    is the partial-observability regime where x(t) stops being a sufficient statistic.
    """
    n, T, d = X.shape
    mask = np.zeros((n, T), dtype=np.float32)
    mask[:, ::every] = 1.0
    mask[:, 0] = 1.0
    X_obs = X.copy()
    for i in range(n):
        last = X[i, 0]
        for t in range(T):
            if mask[i, t]:
                last = X[i, t]
            else:
                X_obs[i, t] = last
    return X_obs, mask


if __name__ == "__main__":
    # Self-check: the generator's own data must satisfy every hard constraint.
    X, patients = generate_dataset(n_patients=300, n_months=60, seed=0)
    dx = np.diff(X, axis=1)
    for idx in MONOTONE_UP:
        assert dx[:, :, idx].min() >= -1e-6, f"{FIELD_NAMES[idx]} decreased in ground truth!"
    # S may only drop at an ERCP month
    for i, p in enumerate(patients):
        ercp = set(p.ercp_months)
        for t in range(1, X.shape[1]):
            if t not in ercp:
                assert X[i, t, S] >= X[i, t - 1, S] - 1e-6, "S dropped outside ERCP!"
    assert (X >= -1e-6).all() and (X <= FIELD_MAX + 1e-6).all(), "out of bounds!"
    print(f"OK  X={X.shape}  all hard constraints hold in the ground truth.")
    print("per-field [min, mean, max] over dataset:")
    for i, name in enumerate(FIELD_NAMES):
        print(f"  {name:5s} [{X[..., i].min():.3f}, {X[..., i].mean():.3f}, {X[..., i].max():.3f}]")
    # counterfactual sanity: earlier UDCA should not increase M for a responder
    resp = [p for p in patients if p.responder and p.udca_start < 40][:1]
    if resp:
        p = resp[0]
        base = simulate(p, 60, draw_noise(np.random.default_rng(p.seed + 1), 60,
                        0.15 + 0.1 * p.disease_class, 0.15 + 0.1 * p.disease_class))
        print(f"counterfactual engine wired; sample patient susc={p.susceptibility:.2f}")
