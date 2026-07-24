# Digital Liver World Model

**A JEPA-style predictive world model for liver-disease trajectories, with hard clinical constraints enforced by construction.**

**Author:** Subash Kandel
**Repository:** https://github.com/Subashkandel10/liver-world-model

---

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Formulation](#problem-formulation)
3. [System Overview](#system-overview)
4. [Pipeline Architecture](#pipeline-architecture)
5. [Core Components](#core-components)
6. [Constraint Enforcement](#constraint-enforcement)
7. [Models Compared](#models-compared)
8. [Representation Collapse Control](#representation-collapse-control)
9. [Evaluation Design](#evaluation-design)
10. [Experimental Results](#experimental-results)
11. [Explainability](#explainability)
12. [Installation and Setup](#installation-and-setup)
13. [Usage](#usage)
14. [Project Structure](#project-structure)
15. [Reproducibility](#reproducibility)
16. [Limitations and Residual Risk](#limitations-and-residual-risk)
17. [License](#license)
18. [Citation](#citation)
19. [Contact](#contact)

---

## Introduction

This project builds and evaluates a world model over a bounded 8-dimensional clinical state vector
`x(t)` describing a patient's liver disease as it moves month to month. Given a patient's trajectory
so far, the model predicts how the disease progresses — while guaranteeing that its predictions stay
physically possible.

The problem has three requirements that pull against each other:

- **Accuracy** — low error on held-out patients.
- **Hard constraints** — fibrosis, ductopenia, portal hypertension and malignancy hazard are
  *one-directional*. A trajectory in which fibrosis drifts back down is not a small numerical error;
  it describes something that cannot happen in a real liver.
- **Explainability** — a reviewer asking "why did the model predict decompensation?" must get a real
  answer, not an appeal to network weights.

The working direction explored here is a **JEPA-style predictive latent** — a model that predicts the
*representation* of a future state rather than its raw values — paired with a causal graph-attention
encoder. That route is implemented, measured against four peers, and reported on honestly, including
where it does not pay off.

### Headline finding

On this clean, fully-observed generator, **`x(t)` is a near-sufficient statistic**, so a learned
predictive latent buys **auditability and robustness, not one-step accuracy**. Rather than assert
this, the repository *measures* it and maps the boundary where it flips (it flips under sensor
noise). The constraint mechanism, by contrast, is an unambiguous win: **zero violations across
7,200 held-out transitions, at no cost in reachable accuracy.**

---

## Problem Formulation

Given a history `x(t₀…t_k)` and per-patient context `c`, predict `x̂(t_{k+1}…t_n)`. Here `k = 12`
months of history and `n = 48` months of horizon, on monthly timesteps.

### The state vector

| idx | field | meaning | range | temporal behaviour |
|----|-------|---------|-------|--------------------|
| 0 | `F` | fibrosis | [0,1] | ratchet, non-decreasing |
| 1 | `D` | ductopenia (duct loss) | [0,1] | ratchet, irreversible |
| 2 | `S` | biliary strictures | [0,1] | non-decreasing, steps **down** only at an ERCP event |
| 3 | `P` | portal hypertension | [0,1] | ratchet, non-decreasing |
| 4 | `A` | inflammatory activity | [0,1] | fast, mean-reverting |
| 5 | `C` | cholestasis | [0,1] | fast, with flares |
| 6 | `M` | malignancy hazard | [0,2] | accumulates from sustained `F·C` |
| 7 | `flare` | acute cholangitis | [0,1] | transient, decays |

**Context** (supplied, never predicted): disease class, age, sex, responder status, UDCA start month,
ERCP event months.

**Hidden** (never supplied): per-patient **susceptibility** — the latent progression rate. This is
what the sharpest generalisation probe holds out.

**Derived:** cirrhosis stage is a fixed monotone function of `F`, recomputed on demand and never
stored, so it can never disagree with `F`.

### Where the constraints get hard

Per-field monotonicity has known parameterisations and is the easy part. The difficulty is the
**coupling**, because the dynamics do not factor the way a per-field guarantee does:

- `M` accumulates as a hazard of *sustained* `F·C`.
- A flare perturbs `A` and `C` **together**, then decays.
- Treatment suppresses `C` and `A`, but only for responders.

Enforcing each channel's constraint independently while keeping these interactions faithful is the
real tension. How this repository resolves it — and what it deliberately leaves unresolved — is
documented in [Constraint Enforcement](#constraint-enforcement).

---

## System Overview

The system trains five models that share **one constraint head and one decoder**, so any measured
difference is attributable to the *representation* rather than to the constraint machinery.

### Key features

- **Constraints by construction** — validity is a property of the parameterisation, not a penalty
  term. Violation is unrepresentable, and an inverse round-trip test proves the head can still
  express *any* valid transition.
- **Five-model controlled comparison** — memoryless, supervised history, causal graph-attention,
  JEPA, and an oracle handed the true hidden susceptibility.
- **Falsification-oriented evaluation** — four generalisation probes, a second valid disease
  mechanism never trained on, a learned on-manifold critic, and a direct ablation of the JEPA
  objective.
- **Collapse control with a loss-independent metric** — VICReg plus an EMA stop-gradient target,
  audited by the effective rank of the latent covariance.
- **Three-layer explainability** — an increment ledger, input-gradient saliency, and latent
  read-back of the inferred progression rate.
- **Calibrated tail risk** — a pinball + conformal P80 risk path for portal hypertension.
- **Fully seeded** — every headline number is reported over three seeds; single-seed results are
  labelled as such.

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SEEDED TRAJECTORY GENERATOR                             │
│   ratchets driven by A, C and hidden susceptibility · A mean-reverts and    │
│   spikes on flares · treatment suppresses C/A for responders · ERCP         │
│   relieves S · M accumulates as a hazard of sustained F·C                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   DISJOINT COHORTS  (train / val / test)                    │
│   fit on train · select checkpoint on val · report on test                  │
│   guarded by a regression test that fails if the sets intersect             │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ENCODER  (swapped per model)                          │
│                                                                             │
│   ┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌─────────────┐ │
│   │  memoryless   │  │  supervised   │  │ graph-attn    │  │    JEPA     │ │
│   │ x(t) is the   │  │  history GRU  │  │ masked to the │  │  GRU + EMA  │ │
│   │    latent     │  │               │  │ causal graph  │  │  + VICReg   │ │
│   └───────────────┘  └───────────────┘  └───────────────┘  └─────────────┘ │
│                                                                             │
│   JEPA only:  predicted latent ──vs── EMA target encoder  (stop-gradient)   │
│               + VICReg variance hinge & covariance penalty                  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                   DECODER  →  raw 9-vector  (shared by all models)          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│              CONSTRAINT HEAD  (fixed, unlearned, shared by all)             │
│                                                                             │
│    F, D, P   →  prev + softplus(raw)                      monotone up       │
│    M         →  prev + softplus(raw) · (prev_F · prev_C)   hazard-gated     │
│    S         →  prev + softplus(up) − 1[ERCP] · σ(relief)  down only at ERCP│
│    A, C, fl  →  σ(raw)                                     bounded, free    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    x̂(t+1)  —  VALID BY CONSTRUCTION                         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          EVALUATION HARNESS                                 │
│  accuracy vs aleatoric floor · constraint-violation rate · 4 generalisation │
│  probes · alternate-mechanism transfer · effective rank · manifold critic   │
│  · P80 tail detection with bootstrap CIs · counterfactual vs generator      │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. `config.py` — the structural contract

Single source of truth for the state schema: field indices, which fields ratchet, per-field bounds,
the causal parent graph, and the derived clinical readouts (cirrhosis stage, decompensation
threshold). Everything downstream imports from here, so the specification lives in exactly one place.

### 2. `generator.py` — the data source

Seeded dynamical model producing unlimited labelled trajectories. Supports a **second, valid
alternate mechanism** (same state, bounds and treatment/ERCP semantics; different coupling equations)
used as a falsification probe, and **shared-noise re-runs** for counterfactual validation.

### 3. `model.py` — constraint head and encoders

The by-construction decode head, the five encoders, VICReg regularisation, the EMA target encoder,
and the quantile head that produces a calibrated P80 risk path for portal hypertension.

### 4. `data.py` — cohorts and probes

Context-vector assembly and **disjoint train/val/test cohorts**, plus the held-out-susceptibility,
treatment-timing, long-rollout and cross-mechanism probe cohorts.

### 5. `train.py` — training

Scheduled-sampling multistep rollout with a combined JEPA + reconstruction + VICReg objective and a
field-weighted loss that directs capacity toward the low-noise, clinically decisive ratchets rather
than the noise-saturated fast channels.

### 6. `evaluate.py` — the honest-numbers harness

Accuracy against the measured aleatoric floor, constraint-violation rates, all generalisation probes,
cross-mechanism transfer, the graded manifold critic, the noisy + irregular forecast, and P80 tail
detection with bootstrap confidence intervals.

### 7. `ablation.py`, `multiseed.py` — attribution and variance

A direct ablation isolating what the latent-prediction term buys, and a three-seed spread over every
headline number so no ranking rests on one lucky run.

### 8. `explain.py` — why the model said this

Increment ledger, input-gradient saliency, and linear read-back of the inferred susceptibility for a
single held-out patient.

### 9. `test_invariants.py` — the guarantee, tested

Seventeen random-weight and adversarial-input tests asserting that the constraints hold for
*arbitrary* network outputs, not merely for trained ones. Pytest-discoverable and CI-gateable.

---

## Constraint Enforcement

The decoder emits nine raw signals which a fixed, unlearned head maps to a guaranteed-valid next
state:

```
F, D, P      ←  prev + softplus(raw)
M            ←  prev + softplus(raw) · (prev_F · prev_C)
S            ←  prev + softplus(up) − 1[ERCP] · sigmoid(relief)
A, C, flare  ←  sigmoid(raw)
```

**Why a sign-constrained increment rather than a penalty or a projection.** A loss penalty leaves
violation reachable and merely expensive. A post-hoc projection repairs the output without the model
ever learning from the repair. A sign-constrained *increment* makes the constraint an invariant of
the forward pass: there is no parameter setting and no input for which it can be violated.

### The coupling decisions

The assignment names three couplings. They are not equally amenable to a by-construction guarantee,
and this project treats them differently and says why:

| coupling | form | decision | reasoning | cost accepted |
|---|---|---|---|---|
| `M ← sustained F·C` | monotone accumulation | **enforced** | reduces cleanly to a non-negative increment gated by `F·C` | the model *inherits* the law, so `M` is not counted as evidence of learned dynamics |
| `flare → A, C` | soft, bidirectional, transient | **left learned** | not a monotone increment; forcing it would distort it | fidelity checked empirically, not guaranteed |
| `treatment → A, C` | soft, responder-gated, reversible | **left learned** | encoding it exactly amounts to re-deriving the generator | fidelity checked empirically, not guaranteed |

### What the guarantee costs

**Nothing reachable.** An inverse round-trip test solves for the raw vector producing any real
transition and reconstructs it to `< 1e-3`. The head removes only the freedom to be *invalid*.

**But validity is not fidelity.** Zero violations does not mean on-manifold. A learned critic trained
to separate real transitions from *graded valid-but-wrong* ones — cross-patient, wrong-month, and
perturbed-real, all of which satisfy every hard constraint — reaches **AUC ≈ 0.93** and scores every
model's rollout below real transitions. These are measured as separate properties.

---

## Models Compared

| model | what it is | role |
|---|---|---|
| `memoryless` | `x(t)` *is* the latent — a Markov map | the honest strong baseline |
| `supervised` | history GRU → decode; drops **only** the JEPA objective | the peer that matters |
| `graph` | attention hard-masked to the disease's causal parents | the recommended route, measured |
| `jepa` | history → predicted latent vs. EMA target, + VICReg | the working direction |
| `oracle` | memoryless + the **true** hidden susceptibility | information upper bound |
| `supervised_quantile` | supervised history + pinball/conformal P80 path for `P` | tail risk |

**A plain Neural-ODE was reasoned-rejected.** The dynamics are not smooth: flares are discrete
monthly jumps and ERCP is a discrete step-down. A continuous vector field must either smooth those
events away or bolt on a discrete handler, at which point the continuous-time claim is doing no work.

---

## Representation Collapse Control

Collapse is a live risk the moment a learned latent is introduced. Three mechanisms guard against it:

1. **Reconstruction loss** — full collapse is unrepresentable, since a constant latent cannot decode
   a moving state.
2. **VICReg** — a variance hinge keeps dimensions alive; a covariance penalty keeps them
   decorrelated.
3. **EMA stop-gradient target** — blocks the encode-everything-to-a-point shortcut that a
   shared-weight target invites.

**The metric is effective rank** — the participation ratio of the latent covariance eigenspectrum —
chosen because it is *independent of the loss*, so a collapse driven by an unregularised term still
shows up.

| quantity | value |
|---|---|
| latent effective rank (of 16) | **4.77** reference seed; 4.15 ± 0.44 over three seeds |
| measured data intrinsic effective rank | **2.70** |
| reading | uses more dimensions than the disease has, but not sixteen — neither collapsed nor padded |

Removing the covariance term drives the rank toward 1, and the metric catches it. That is the check
worth running in CI, not the training curve.

---

## Evaluation Design

The evaluation was designed to be able to falsify the model. Each probe varies **one** axis and holds
the rest fixed, and each is reported with what it *cannot* establish.

| test | varies | can establish | **cannot** establish |
|---|---|---|---|
| held-out susceptibility | train `s ≤ 2.0`, test `s ≥ 2.0` (disjoint) | extrapolation to fast progressors | anything about unseen mechanisms |
| treatment timing | train UDCA months 2–23, test 26–40 (disjoint bands) | whether the treatment lever transfers over time | fidelity of the lever's magnitude |
| long rollout | train to 48 months, roll to 96 | compounding-error behaviour | real-liver long-horizon validity |
| alternate mechanism | same state/bounds/semantics, *different* coupling equations | **risk** of generator inversion | a real-liver claim — it is one designed shift |
| manifold critic | real vs. graded valid-but-wrong transitions | that validity ≠ plausibility | a calibrated ranking (one critic seed) |
| counterfactual | UDCA −6 months, shared-noise re-run | sign and rough magnitude of an intervention | per-patient reliability (corr. 0.51) |

### The identifiability trap, stated plainly

Training and testing inside **one** generator means a model that generalises across held-out patients
has, in effect, recovered that generator's update rule — and here the generator *is* the disease. So
"world model vs. generator-inverter" is **not identifiable from this data**, by construction. In-
distribution accuracy here measures *recovery of one mechanism*; it is not evidence of disease
understanding, and is not reported as such.

---

## Experimental Results

All numbers are on held-out **test** cohorts, reproduced by the harness. Multi-seed figures are mean
± standard deviation over seeds {0, 1, 2}.

### Constraint violations

| model | violations | transitions |
|---|---|---|
| memoryless | **0** | 7,200 |
| supervised | **0** | 7,200 |
| graph | **0** | 7,200 |
| jepa | **0** | 7,200 |
| oracle | **0** | 7,200 |

Checked per field: `F`, `D`, `P`, `M` monotonicity, `S` off-ERCP, and all bounds. Rate is `0.0000%`
in every cell. Zero is the *expected* result — the head makes violation unrepresentable — so this
table functions as a regression test on the implementation rather than as evidence about the model.

### Predictive accuracy

| model | seed 0 | 3-seed mean ± sd | reading |
|---|---|---|---|
| memoryless | 0.085 | 0.075 ± 0.007 | strong baseline |
| supervised | 0.078 | 0.072 ± 0.005 | ties for best |
| **graph** | 0.077 | **0.070 ± 0.005** | ties; wins on auditability |
| jepa | 0.087 | 0.082 ± 0.005 | **nominally weakest** |
| oracle | 0.094 | 0.081 ± 0.009 | **not** an upper bound |
| supervised_quantile | 0.076 | 0.070 ± 0.005 | best point predictor |
| **aleatoric noise floor** | — | **0.081** | irreducible flare/activity noise |

Every band overlaps the floor and every other band, so the tie is real rather than an artefact of one
seed. Handing a model the *true* susceptibility does not help — the oracle gap is **−0.009**.

### The aggregate tie hides a real structural-field gap

Three of the eight fields are noise-saturated and carry roughly 60% of the aggregate MAE, which is
what flattens the headline number. Decomposing by field group (reference seed 0):

| model | overall | ratchets `F,D,S,P` | fast `A,C,flare` | vs. memoryless on ratchets |
|---|---|---|---|---|
| memoryless | 0.085 | 0.069 | 0.117 | — |
| supervised | 0.078 | 0.057 | 0.118 | **−18.4%** |
| **graph** | 0.077 | **0.055** | 0.115 | **−20.7%** |
| jepa | 0.087 | 0.073 | 0.117 | +4.7% |
| oracle | 0.094 | 0.085 | 0.120 | +22.0% |
| noise floor | 0.081 | 0.032 | 0.164 | — |

History *does* buy something real — about 20% lower error on the four ratchet fields, and **−30% on
portal hypertension** specifically (0.057 → 0.040), the field that defines decompensation. But it is
**history**, not the JEPA objective: JEPA is 4.7% *worse* than memoryless on exactly these fields.

### Why the latent buys so little

| representation | linear decodability of hidden susceptibility (R²) |
|---|---|
| raw `x(t)` | **0.57** |
| JEPA latent | 0.15 |
| history latent | −0.03 |

The ratchets are a running integral of susceptibility, so the current state already carries the
hidden cause. On a clean, fully-observed, low-dimensional state there is almost no nuisance detail
for a predictive latent to discard — which is *why* the oracle adds nothing and the latents tie.

### Generalisation probes — failures shown

| probe | JEPA | best model | Δ vs. in-dist | verdict |
|---|---|---|---|---|
| in-distribution | 0.087 | 0.077 (graph) | — | baseline |
| held-out susceptibility | 0.158 | 0.144 (supervised) | **×1.8** | ✗ the real gap |
| treatment timing (months 26–40) | 0.081 | 0.072 (graph) | −7% | ✓ lever transfers |
| long rollout (48 → 96 months) | 0.126 | 0.109 (graph) | **+45%** | ✗ compounding drift |
| alternate mechanism | 0.096 | 0.084 (graph) | **+17%** | ✗ does not transfer |

**Treatment timing does not degrade at all**, and the bands are genuinely disjoint — the model
learned "UDCA suppresses `C`", not "month 12 is special". On the **alternate valid generator**, JEPA
degrades 0.082 → 0.096 (+17%) and supervised 0.072 → 0.085 (+18%) over three seeds. The hard
guarantees still hold; the accuracy does not transfer.

### The phase boundary — when does a latent earn its keep?

**(a) Sparse visits** — rollout MAE by months between clinic visits:

| stride (months) | 1 | 3 | 6 | 12 |
|---|---|---|---|---|
| memoryless | 0.0291 | 0.0401 | 0.0489 | 0.0593 |
| supervised | 0.0282 | 0.0388 | 0.0469 | 0.0560 |
| graph | **0.0279** | **0.0378** | **0.0455** | **0.0546** |
| jepa | 0.0298 | 0.0411 | 0.0501 | 0.0611 |
| **history gain** | +0.0009 | +0.0013 | +0.0020 | +0.0032 |

The history advantage grows monotonically as visits thin, but stays a whisper — again because `x(t)`
already carries susceptibility.

**(b) Sensor noise** — current-state estimate error:

| σ | 0.05 | 0.10 | 0.15 |
|---|---|---|---|
| raw noisy observation | 0.0393 | 0.0814 | 0.1139 |
| supervised history denoise | **0.0154** | **0.0188** | **0.0225** |
| JEPA denoise | 0.0228 | 0.0297 | 0.0379 |

A **decisive win** — a single state structurally cannot filter. But it is a win for *history*, not
for the JEPA objective: the plain supervised denoiser beats the JEPA one at every noise level.

### A negative result: JEPA's best case still ties

The regime the latent-prediction objective should most favour is degraded observation — noisy *and*
irregular. Full multi-step forecast at σ = 0.10:

| stride | memoryless | supervised | JEPA | JEPA − supervised |
|---|---|---|---|---|
| 1 month | 0.0711 | 0.0708 | 0.0708 | −0.00004 |
| 3 months | 0.0776 | 0.0779 | 0.0785 | −0.0005 |
| 6 months | 0.0808 | 0.0815 | 0.0818 | −0.0003 |

The denoising win on isolated *state estimation* does **not** propagate to multi-step *forecast*
accuracy. This closes, rather than opens, the "JEPA wins under degraded observation" argument.

### Ablation — what the latent-prediction term actually buys

| variant | rollout MAE | effective rank | manifold score |
|---|---|---|---|
| JEPA, full objective | 0.0871 | **4.77** | 0.509 |
| JEPA, latent term → 0 | 0.0957 | 3.44 | **0.672** |
| **Δ from the latent term** | **−0.009** | **+1.33** | −0.163 |

The objective buys **anti-collapse** (+1.3 effective rank) — the risk it was adopted to manage. It
does **not** buy on-manifold behaviour; that score gets *worse*, so the claim was withdrawn.

### Decompensation detection (`P` crosses 0.50, 11 events)

| regime | recall | false alarms |
|---|---|---|
| point forecast, 6-month follow-up | 0.55 – 0.73 | 0 |
| **calibrated P80 risk, 6-month follow-up** | **1.00** (11/11) | ~3 |
| calibrated P80 risk, pure rollout (3-seed) | 0.21 ± 0.17 | 0 |
| **point forecast, pure 36-month rollout** | **0.00** (0/11) | 0 |

Under pure rollout the model reaches a final `P ≈ 0.25` against a true 0.63: it under-predicts the
accelerating portal-hypertension tail and misses every event. **This is the honest ceiling.** A
calibrated quantile path converts that missed tail into a manageable false-alarm rate — but only
given periodic re-observation.

### Counterfactual (UDCA started 6 months earlier)

Validated against a generator re-run on the *same noise draws*, so the only difference is the
intervention.

| quantity | value |
|---|---|
| true mean ΔM | −0.0139 |
| model mean ΔM | −0.0120 |
| sign agreement | **0.92** |
| per-patient correlation | 0.51 |

Right direction and roughly right magnitude, with substantial per-patient scatter — enough to say
"earlier treatment reduces malignancy hazard on average", not enough to counsel an individual.

### Causal graph-attention, measured

Each field is a node; one attention layer is hard-masked to the disease's causal parents, so
information can only flow along biological edges and the weights are the computation itself rather
than a post-hoc attribution.

| target | attention mass over permitted parents |
|---|---|
| `F` | `F` .58, `C` .32, `A` .10 |
| `D` | `D` .58, `C` .33, `A` .10 |
| `S` | `S` .67, `A` .33 |
| `P` | `F` .40, `C` .29, `P` .25, `A` .06 |
| `A` | flare .74, `A` .26 |
| `C` | flare .66, `C` .34 |
| `M` | `F` .39, `M` .31, `C` .30 |
| `flare` | flare .52, `S` .48 |

`M` splits its mass almost evenly across `F` and `C` — it recovered the `F·C` hazard without being
told the product form. The recommended encoder therefore earns its place on **auditability at equal
accuracy**; building and measuring it overturned an earlier draft that had asserted it "does not earn
its cost".

---

## Explainability

For a held-out patient — disease class 2, non-responder, hidden susceptibility 1.91, true
decompensation at month 33, model flags month 36 under 6-month follow-up — the prediction is audited
at three levels:

1. **Increment ledger.** Because `P = prev + a non-negative increment`, its entire rise is a
   printable sum of monthly steps. The parameterisation makes the trajectory auditable for free.
2. **Input-gradient saliency.** The prediction leans on the structural fields — `P`-history,
   strictures, `F`, `D` — not the noisy fast channels, consistent with the per-field error
   decomposition.
3. **Latent read-back.** A linear read of susceptibility from the latent returns **1.87** against a
   true 1.91, so the model inferred "fast progressor" from twelve months of history, and that drives
   the steep `P` projection.

**The caveat belongs in the same breath as the claim.** The correct statement is *"given follow-up
data, the model projected `P` across the threshold three months late"* — not *"it foresaw
decompensation from month 12"*. Without re-observation it misses the event entirely.

Run it with `python -m lwm.explain`.

---

## Installation and Setup

### Prerequisites

- Python ≥ 3.11
- No GPU required — the full pipeline runs on CPU

### Installation

```bash
# Clone repository
git clone https://github.com/Subashkandel10/liver-world-model.git
cd liver-world-model

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Dependencies are `torch ≥ 2.2`, `numpy ≥ 1.26`, and `matplotlib ≥ 3.8`.

---

## Usage

### Reproduce everything

```bash
python run_all.py                # full pipeline
python run_all.py --quick        # skip the slow 3-seed pass
```

This chains: generator self-check → invariant tests → training → evaluation → ablation → multiseed →
figures, and writes `checkpoints/manifest.json` recording seed, epochs, best validation score and git
hash per checkpoint.

### Run individual stages

```bash
python -m lwm.generator          # self-check: constraints hold in the ground truth
python -m lwm.test_invariants    # 17 random-weight guarantee tests (non-zero exit on failure)
python -m lwm.train 40           # train clean + noise-augmented suites  (~7 min CPU)
python -m lwm.evaluate           # -> eval_out/metrics.json + printed report
python -m lwm.ablation           # JEPA-loss ablation -> eval_out/ablation.json
python -m lwm.multiseed          # 3-seed spread -> eval_out/multiseed.json  (~18 min)
python -m lwm.explain            # the decompensation walk-through
python -m lwm.figures            # -> figures/
```

### Run the test suite

```bash
pip install pytest
pytest
```

The suite asserts the constraints hold for **arbitrary** network outputs, not merely trained ones,
and includes a regression test that fails if the train/val/test cohorts ever intersect.

---

## Project Structure

```
liver-world-model/
│
├── lwm/                          # source package
│   ├── config.py                 # state schema, monotonicity contract, causal graph
│   ├── generator.py              # seeded trajectory generator + alternate mechanism
│   ├── model.py                  # constraint head, five encoders, VICReg, P80 risk path
│   ├── data.py                   # disjoint cohorts, probe and cross-mechanism sets
│   ├── train.py                  # scheduled-sampling multistep rollout, field-weighted loss
│   ├── evaluate.py               # the honest-numbers harness
│   ├── ablation.py               # what the latent-prediction objective buys
│   ├── multiseed.py              # 3-seed spread on every headline number
│   ├── explain.py                # increment ledger, saliency, latent read-back
│   ├── figures.py                # all figures, incl. fig_summary
│   └── test_invariants.py        # 17 random-weight guarantee tests
│
├── docs/
│   └── memo.pdf                  # the decision memo — primary deliverable (3 pages)
│
├── checkpoints/                  # GENERATED — trained models + manifest.json
├── eval_out/                     # GENERATED — metrics.json, ablation.json, multiseed.json
├── figures/                      # GENERATED — result figures, incl. fig_summary.png
│
├── run_all.py                    # one-command pipeline
├── conftest.py                   # puts the repo root on sys.path for a fresh checkout
├── pyproject.toml                # package metadata + pytest configuration
└── requirements.txt              # dependencies
```

---

## Reproducibility

- **All randomness is seeded** — generator, cohort assignment, initialisation, and augmentation.
- **Headline numbers are reported over seeds {0, 1, 2}**; anything single-seed is labelled where it
  appears.
- **The three `GENERATED` directories are committed** (~1.1 MB). Together with
  `checkpoints/manifest.json` they make every number in the memo auditable **without retraining**,
  which is the point of shipping an evaluation harness at all.
- **Split discipline is tested, not assumed.** An earlier version of this work selected the
  checkpoint on the validation set and then reported in-distribution numbers on that same set. With
  inter-model gaps around 0.005 MAE that bias was decisive — it had inflated headline accuracy by
  0.010–0.017 MAE. The cohorts are now disjoint, a regression test guards them, and every number
  published here is post-fix.

---

## Limitations and Residual Risk

Stated plainly, because each is a place the evaluation was *designed* to expose:

1. **The accelerating-`P` tail is improved, not solved.** The calibrated risk head catches every
   decompensator under 6-month follow-up, but pure-rollout recall is 0.21 ± 0.17 across seeds.
2. **Extrapolation to the fastest progressors fails** (×1.8 error). The model cannot emit increments
   outside its training range — the named, expected break.
3. **One generator.** The world-model-versus-inverter question is unfalsifiable here by construction.
   The alternate mechanism is a controlled shift, not an independent source.
4. **Evidence is thin in places.** The manifold ranking rests on one critic seed; decompensation on
   11 events with wide confidence intervals; the counterfactual correlation is only 0.51.
5. **The headline may be wrong.** "History buys robustness" rests on a denoising comparison in which
   the history models were noise-augmented and the memoryless baseline was given no denoiser. That is
   a fair *structural* comparison — a single state genuinely cannot filter — but it assumes the gap is
   structural rather than an under-trained baseline.

### What I would do next, in priority order

1. **A family of independently parameterised generators with interventional validation** — the only
   item that makes the central question *testable* rather than merely acknowledged.
2. **Attack the susceptibility break at its cause** — widen the training rate range, or change the
   output parameterisation so out-of-range progression is representable at all.
3. **Calibrate decompensation for deployment** — a cost-weighted threshold on the P80 risk path.

---

## License

Released under the MIT License. See [`LICENSE`](LICENSE).

---

## Citation

```bibtex
@software{kandel_liver_world_model_2026,
  author = {Kandel, Subash},
  title  = {Digital Liver World Model: A JEPA-Style Predictive World Model
            with By-Construction Clinical Constraints},
  year   = {2026},
  url    = {https://github.com/Subashkandel10/liver-world-model}
}
```

---

## Contact

**Author:** Subash Kandel
**Repository:** https://github.com/Subashkandel10/liver-world-model

For questions or discussion, please open an issue on GitHub.
