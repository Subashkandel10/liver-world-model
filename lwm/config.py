"""
Single source of structural truth for the Digital Liver world model.

Everything downstream (generator, model, constraint head, evaluation, tests)
imports the state schema and the monotonicity contract from here, so there is
exactly one place the "what the fields are and how they may move" definition
lives. If the spec changes, it changes here.

State x(t) in R^8, monthly timesteps. All fields in [0, 1] except M in [0, 2].

    idx  field  meaning                        temporal behaviour
    0    F      fibrosis                        ratchet, non-decreasing
    1    D      ductopenia (duct loss)          ratchet, irreversible
    2    S      biliary strictures              non-decreasing, steps DOWN at ERCP
    3    P      portal hypertension             ratchet, non-decreasing
    4    A      inflammatory activity           fast, mean-reverting
    5    C      cholestasis                     fast, with flares
    6    M      malignancy hazard accumulator   monotone non-decreasing, in [0, 2]
    7    flare  acute cholangitis flare         transient, decays

Context constants (supplied to a model, NOT predicted): disease_class, age, sex,
responder in {0,1}, udca_start (month), ercp_months (list).
"""

import numpy as np

# --- field indices: the single source imported everywhere -------------------------------
F, D, S, P, A, C, M, FLARE = range(8)
FIELD_NAMES = ["F", "D", "S", "P", "A", "C", "M", "flare"]
N_FIELDS = 8

# Fields that must never decrease month-to-month. S is monotone-up too, but it is the one
# exception that may step DOWN at an ERCP event, so it is handled separately by the head.
MONOTONE_UP = (F, D, P, M)
RATCHET_FIELDS = (F, D, P, M)          # decoded as prev + positive increment
FREE_FIELDS = (A, C, FLARE)            # decoded as bounded sigmoid, may move either way
FIELD_MAX = np.array([1, 1, 1, 1, 1, 1, 2, 1], dtype=np.float32)

# Per-field reconstruction weights. The fast free fields (A, C, flare) are dominated by
# irreducible flare noise at the aleatoric floor, so squared error there is mostly un-learnable
# and, left equal-weighted, its gradient drowns out the small but LEARNABLE ratchet increments.
# We upweight the slow, low-noise, clinically decisive ratchets (P for decompensation, F for
# cirrhosis, M for malignancy) so capacity goes to the reducible signal.
#                       F    D    S    P    A    C    M   flare
FIELD_WEIGHTS = np.array([4.0, 3.0, 2.0, 5.0, 1.0, 1.0, 4.0, 1.0], dtype=np.float32)

# --- the disease causal graph (parents of each field in the generator's update) ---------
# Read straight off `generator.simulate`: a child field attends only to the fields that
# actually drive its next-step update, plus itself. This is the mask a graph-attention encoder
# routes information along, so information flow is causal *by construction* and inspectable.
#   F,D  <- A, C            (ratchet drive = susc*(0.6 A + 0.4 C))
#   P    <- A, C, F         (drive + fibrosis feedback)
#   S    <- A               (creep proportional to inflammation; ERCP relief is a context flag)
#   A,C  <- flare           (flare perturbs both; mean-reversion + treatment are self/context)
#   M    <- F, C            (hazard of sustained F*C)
#   flare<- S               (onset probability rises with strictures)
CAUSAL_PARENTS = {
    F: (F, A, C),
    D: (D, A, C),
    S: (S, A),
    P: (P, A, C, F),
    A: (A, FLARE),
    C: (C, FLARE),
    M: (M, F, C),
    FLARE: (FLARE, S),
}


def causal_adjacency():
    """Boolean [8, 8] mask; entry [child, parent] = True where the parent drives the child
    (self-edges included). Used to mask attention so a field can only look at its causal parents."""
    adj = np.zeros((N_FIELDS, N_FIELDS), dtype=bool)
    for child, parents in CAUSAL_PARENTS.items():
        for p in parents:
            adj[child, p] = True
    return adj


# --- context layout (what a model is allowed to condition on) ---------------------------
# disease_class one-hot (3) + age + sex + responder + udca_active flag + ercp_now flag = 8
N_DISEASE_CLASSES = 3
CTX_DIM = N_DISEASE_CLASSES + 5

# --- horizons ---------------------------------------------------------------------------
N_MONTHS_TRAIN = 48        # length of a training trajectory
HISTORY = 12               # months of context the history-encoder sees before predicting
ROLLOUT_TRAIN = 6          # multistep free-rollout horizon used during training
LONG_ROLLOUT = 96          # long-horizon generalisation probe length

# --- derived clinical readouts (never stored in x; always recomputed from F) ------------
# Cirrhosis is a fixed monotone function of fibrosis, so it can never disagree with F.
CIRRHOSIS_THRESHOLD = 0.60   # F at/above which the patient is "cirrhotic"
DECOMP_P_THRESHOLD = 0.50    # portal-hypertension level we call clinical decompensation


def cirrhosis_stage(F_value):
    """Derived, monotone in F. 0 = none, 1 = bridging, 2 = cirrhotic."""
    F_value = np.asarray(F_value)
    return (F_value >= 0.35).astype(int) + (F_value >= CIRRHOSIS_THRESHOLD).astype(int)


def is_decompensated(x):
    """Decompensation = portal hypertension crossing its clinical threshold.

    Pure function of the stored state, so a reviewer can recompute it by hand.
    """
    x = np.asarray(x)
    return x[..., P] >= DECOMP_P_THRESHOLD


def decompensation_month(traj):
    """First month P crosses the decompensation threshold, or None if it never does."""
    hits = np.where(traj[:, P] >= DECOMP_P_THRESHOLD)[0]
    return int(hits[0]) if len(hits) else None
