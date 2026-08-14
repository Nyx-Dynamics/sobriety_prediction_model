# Serving layer — inference-server wrapper

Turns the batch-trained `SobrietyLSTM` checkpoint into a **stateful, per-user
online endpoint** an AI companion can call: one 30-day check-in in → relapse
risk curve + attention + a *governed* escalation decision out.

```
companion conversation
        │  (distills a check-in)
        ▼
  WindowObservation ──► contract.py  (encode + cumulative-stress recursion)
        │
        ▼
     state.py  (append-only window stream per user, running stress state)
        │
        ▼
   predictor.py  ──► SobrietyLSTM  ──► hazard curve → survival → risk@{90,180,360}d
        │                                              + attention weights
        ▼
    policy.py  (risk band → escalation; numbers stay server-side)
        │
        ▼
      app.py  (FastAPI: /sessions, /sessions/{id}/observe, /health)
```

## Files
| file | role |
|---|---|
| `model_def.py` | canonical `SobrietyLSTM` arch (import target so training/serving never drift) |
| `contract.py` | the 40-feature schema, categorical encoders, cumulative-stress recursion — the companion↔model interface |
| `state.py` | per-user session store (in-memory reference; swap for Redis/Postgres) |
| `predictor.py` | loads `lstm_best.pt` + `scaler.pkl` + `meta.pkl`, scores a session |
| `policy.py` | deterministic risk→action map; the governance boundary |
| `app.py` | FastAPI endpoints |

## Run
```bash
pip install -r src/serving/requirements.txt
python src/run_pipeline.py            # produce scaler.pkl + meta.pkl + tensors
python src/pipeline/train_lstm.py     # produce models/lstm/lstm_best.pt
cd src && uvicorn serving.app:app --reload
```
`/health` reports `artifacts_ready:false` until the pipeline + training have run;
scoring endpoints return `503` (not a crash) until then.

## Two follow-ups before production
1. **De-dupe the arch.** Patch `pipeline/train_lstm.py` to
   `from serving.model_def import SobrietyLSTM` instead of redefining it, so the
   trained state_dict and the served class can never diverge.
2. **Persist the Bayesian scaler.** `train_bayesian.py` fits `alpha` but only
   saves the trace + *test-set* class probs — not the continuous
   standardization constants needed to score a **new** patient. Dump those
   (e.g. `models/bayesian/bayesian_scaler.json`) to enable the optional
   Remission-vs-Cycling latent class in `predictor._maybe_latent_class`.

## Not included (this is the model seam, not a product)
Authn, rate limiting, audit logging, PHI encryption/retention, and the hard wall
keeping risk numbers out of user-facing companion text (only `escalation` +
`companion_directive` cross back). See the meshability notes.
```
