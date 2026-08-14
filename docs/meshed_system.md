# Meshed System — Feature Spec

**Sobriety Prediction Model × AI Companion**
Nyx Dynamics LLC | Status: design sketch (serving layer scaffolded, models not yet trained on real data)

## 1. Purpose

Fuse the batch-trained relapse-prediction engine (`SobrietyLSTM` + Bayesian
latent-class + Phase-5 pathway attribution) with a conversational AI companion so
that a supportive daily-check-in agent doubles as the model's sensing layer, and
the model's risk read modulates the companion's behavior — behind a hard
governance wall that keeps clinical inference out of user-facing text.

This document is the feature spec. Implementation seam lives in `src/serving/`.

## 2. Architecture at a glance

```
companion conversation ──► WindowObservation ──► contract.py (encode + stress recursion)
                                                     │
                                                     ▼
                                               state.py (per-user window stream)
                                                     │
                                                     ▼
                                          predictor.py ──► SobrietyLSTM
                                                     │      hazard→survival→risk + attention
                                                     ▼
                                               policy.py (risk band → escalation)
                                                     │   numbers stay server-side
                                                     ▼
                                                  app.py (FastAPI)
```

## 3. Feature set

### F1 — Sensing (conversation as instrument)
| # | Feature | Implemented by |
|---|---|---|
| F1.1 | Conversational elicitation → 13 time-varying features per 30-day window | `contract.WindowObservation` |
| F1.2 | Enrollment profile → 27 static features (dx flags, comorbidities, demographics) | `contract.StaticProfile` |
| F1.3 | Server-derived cumulative stress (`0.8·s_{t-1}+sev−0.3·pos`, clip [0,10]) | `contract.update_cumulative_stress` |
| F1.4 | Missed check-ins flagged (`sparse_window`), not dropped — silence as signal | `contract.WindowObservation.sparse_window` |
| F1.5 | **Conversation→features distiller** — companion chat → structured contract via Claude structured outputs; **transcript discarded**, only the 13 numeric features cross | `companion/distiller.py` (`claude-opus-4-8`) |

### F6 — Data portability (transportability for the user)
| # | Feature | Implemented by |
|---|---|---|
| F6.1 | JSON export — the user's full record as a self-contained, versioned document | `portability.export_session` / `GET /sessions/{id}/export` |
| F6.2 | JSON import — reconstruct a session from an export; consent + contract re-validated | `portability.import_session` / `POST /sessions/import` |
| F6.3 | Interoperable JSON Schema so any tool can read/validate an export | `portability.EXPORT_JSON_SCHEMA` / `GET /portability/schema` |

### F2 — Prediction (three heads, one read)
| # | Feature | Implemented by |
|---|---|---|
| F2.1 | Per-window hazard curve → survival → risk at 90/180/360 days | `predictor.Predictor.score` |
| F2.2 | Temporal attention weights → which recent windows drive risk ("why now") | `model_def.TemporalAttention` |
| F2.3 | Remission vs Cycling latent class (persona routing) | `predictor._maybe_latent_class` *(stubbed — see §6)* |
| F2.4 | Pathway attribution → which lever matters (treatment/monitoring/support/self-advocacy) | Phase-5 `pathway_results.json` *(not yet wired to serving)* |

### F3 — Response (risk-aware companion behavior)
| # | Feature | Implemented by |
|---|---|---|
| F3.1 | Escalation ladder: steady → monitor → outreach → urgent | `policy.decide` |
| F3.2 | Non-numeric companion directives (never a probability) | `policy.DIRECTIVES` |
| F3.3 | Attention-informed rationale (names the driving window, not the number) | `policy.decide` |
| F3.4 | Clinician alert + human routing at `urgent` | `policy.Decision.clinician_alert` |

### F4 — Governance (the hard wall)
| # | Feature | Implemented by |
|---|---|---|
| F4.1 | Risk numbers stay server-side; only `escalation` + `companion_directive` cross back | `app.observe` (`_internal` block) |
| F4.2 | Escalation is a deterministic, auditable rule — not the LLM | `policy.py` |
| F4.3 | Supportive vs clinical output separation (companion states no risk, makes no dx claim) | policy boundary by design |

### F5 — Systems / lifecycle
| # | Feature | Implemented by |
|---|---|---|
| F5.1 | Stateful online serving (per-user append-only window stream + running stress) | `state.Session` |
| F5.2 | Fail-fast contract enforcement (served schema checked vs trained `meta.pkl`) | `predictor.Predictor.__init__` |
| F5.3 | Graceful degradation (boots + reports readiness before models trained; scoring 503s) | `app.get_predictor` |
| F5.4 | Pluggable state store (in-memory → Redis/Postgres) | `state.SessionStore` |

## 4. Interfaces

- `POST /sessions` — enroll a user with a `StaticProfile`.
- `POST /sessions/{user_id}/observe` — append one 30-day check-in → escalation + directive (+ `_internal` risk for logging).
- `GET /health` — liveness + artifact readiness.

The **feature contract** (`contract.FEATURE_COLS`, 40 features in fixed order +
11 continuous scaled columns) is the stable companion↔model interface. It is
asserted equal to the trained tensor's `meta['feature_cols']` at load.

## 5. Non-goals (this is the model seam, not a product)

Authn, rate limiting, and the full clinical-safety review are out of scope for
the serving sketch; see §7 privacy features for what the meshed product must add.

## 6. Known gaps

1. Architecture is duplicated between `train_lstm.py` and `serving/model_def.py`
   — training should import the canonical class.
2. Bayesian latent class (F2.3) is stubbed until `train_bayesian.py` persists its
   continuous-standardization constants for scoring new patients.
3. Pathway attribution (F2.4) exists as batch output but is not yet exposed via serving.
4. Fits are synthetic-first; clinical validity unestablished (PAFs ≈ 0, MH HR HDIs straddle 1).

## 7. Privacy & data-protection features

> **Non-negotiable stance: this MUST NOT be a surveillance tool.** Encryption,
> PHI enforcement, third-party blocking, and transparency are paramount and are
> enforced in code fail-closed — not left to convention. Implementation:
> `serving/privacy.py`, `serving/state.py`, `serving/app.py`.

| # | Privacy feature | Rationale / status |
|---|---|---|
| P1 | **Data minimization at the contract boundary** — only the 40 defined features are ingested; free text / transcripts / unknown metadata are **rejected**, not stored | `privacy.reject_unknown_fields` + dataclass contract; **ENFORCED** |
| P2 | **Server-side derivation of sensitive state** — `cumulative_stress` computed server-side, never round-tripped through the client | `contract.update_cumulative_stress`; **ENFORCED** |
| P3 | **Risk-number containment** — relapse probabilities never reach the companion/user; the check-in endpoint returns only a non-numeric directive | `privacy.to_companion` allowlist + `assert_egress_clean`; **ENFORCED** |
| P4 | **PHI encryption at rest (fail-closed)** — session state serialized and Fernet-encrypted; **no key ⇒ the store refuses to start**. Backing map/Redis/Postgres only ever sees ciphertext | `privacy.Encryptor` + `state.SessionStore`; **ENFORCED** |
| P5 | **Retention & right-to-erasure** — `DELETE /sessions/{id}` hard-deletes; `purge_expired` drops sessions past the retention window | `state.delete` / `state.purge_expired` / `app.erase`; **ENFORCED** |
| P6 | **Consent gate (no covert collection)** — enrollment requires explicit consent; check-ins refused without it | `app.enroll` / `app.observe`; **ENFORCED** |
| P7 | **Subject access** — a user can retrieve exactly what is held about them | `app.my_data` (`GET /sessions/{id}/data`); **ENFORCED** |
| P8 | **Third-party / model exfiltration block** — the companion LLM receives ONLY the non-numeric directive (allowlist). Risk numbers live behind a separate **authenticated** clinical route (`X-Clinical-Token`), never the companion path | `privacy.COMPANION_SAFE_KEYS` + `app.clinical`; **ENFORCED** |
| P9 | **Always-on transparency** — open, unauthenticated disclosure of what is collected, why, retention, that nothing is sold/shared, and the user's rights | `privacy.disclosure` (`GET /transparency`); **ENFORCED** |
| P10 | **Measurement/intervention channel separation** — the channel that measures mood is kept distinct from the one that influences it, limiting endogenous feedback and covert profiling | policy boundary by design; **partially realized** |
| P11 | **Authentication / per-user isolation** — per-user bearer token minted at enrollment (only its HMAC stored); every user-scoped route requires it, so user A's token cannot reach user B. Fail-closed without `SUD_AUTH_PEPPER` | `auth.py` + `app.require_user`; **ENFORCED** |
| P12 | **Immutable, tamper-evident audit log** — append-only SHA-256 hash chain over every enroll / check-in / clinical access / subject access / erasure / auth-denial. PHI-free (pseudonymous `subject_ref`, escalation band only); any edit breaks `GET /audit/verify` | `audit.py` + `app` route hooks; **ENFORCED** (ship to WORM in prod) |
| P13 | **De-identification for training** — inference state never fed back into training without consent + de-id | pipeline de-identified; **policy TODO** for the inference→training loop |
| P14 | **Regulatory posture** — treat as potential SaMD/Clinical Decision Support (HIPAA + FDA CDS guidance); the governance wall is a mitigation, not a sign-off | **open** — legal review |

**Enforced fail-closed in code today:** P1–P12.
**Required before any real PHI:** P13 (training-loop policy), P14 (legal review).

### Required environment (all fail-closed)
| Var | Purpose |
|---|---|
| `SUD_PHI_KEY` | Fernet key — PHI encryption at rest (P4). No key ⇒ store refuses to start. |
| `SUD_AUTH_PEPPER` | HMAC pepper — token hashing + pseudonymization (P11/P12). No pepper ⇒ auth refuses. |
| `SUD_CLINICAL_TOKEN` | Gates the risk-bearing `/clinical` route (P8). |
| `SUD_AUDIT_LOG` | Audit sink path (default `audit/audit.jsonl`). |
| `SUD_RETENTION_DAYS` | Retention window for `purge_expired` (P5). |
