# Sharing Guide — FLP on Databricks

How to present the French-Learning-Perceptions ML pipeline to the thesis jury
(the "tiers") **without giving them any access** to the dataset, the API, the
catalog, or any credential.

---

## 1. Share freely (zero access required)

| Artifact | Where | What the jury sees |
|---|---|---|
| Public dashboard | https://student-plurilingual-representation-french-learning-htyukpvkoo.streamlit.app/ | Interactive bilingual UI (FR/EN) with H1–H4 predictions, served live from Databricks Model Serving (token stays server-side) |
| Live inference demo | `python databricks/invoke_endpoint.py all` (run on **your** machine) | Real-time predictions from the Databricks serving endpoint |

The dashboard and the demo script are safe: they never expose data or tokens.

---

## 2. Screenshot checklist (export from your authenticated session)

Take these screenshots from **your own browser** — the jury gets images only,
never the links themselves.

- [ ] **Catalog Explorer** — `flp_catalog` with schemas `raw`, `processed`, `models`, `monitoring`
- [ ] **Data** — `raw.survey_responses` (500 rows, 34 cols) and `processed.h1_features`…`h4_features` (495 rows, consent-filtered)
- [ ] **Workflow job** — "FLP — Full ML Pipeline (Databricks Serverless)", job id `562108964197707`, last run with **8/8 green tasks**
      (setup → preprocess → train_h1 → train_h2 → train_h3 → train_h4 → evaluate → serving_bundle)
- [ ] **MLflow experiments** — `/Shared/FLP_H1_Multilingual_Repertoire`, `FLP_H2_French_Representations`,
      `FLP_H3_Plurilingual_Exposure`, `FLP_H4_Local_Language_Integration` with metrics below
- [ ] **Model registry** — 9 registered models in `flp_catalog.models` (8 hypothesis models + `flp_all` serving bundle)
- [ ] **Serving endpoint** — `https://dbc-9e268203-7090.cloud.databricks.com/ml/endpoints/flp-all-models` (state: **READY**)
- [ ] **Pedagogical report** — `flp_catalog.monitoring.pedagogical_report`

### Key results to highlight

| Hypothesis | Metric | Value | Threshold | Status |
|---|---|---|---|---|
| H1 — Multilingual repertoire → daily mobilization | F1-macro / ROC-AUC | 0.79 / 0.84 | ≥ 0.70 / ≥ 0.75 | ✅ VALIDATED |
| H2 — French representations → motivation & difficulties | F1-weighted / F1-micro | 0.95 / 0.80 | ≥ 0.65 / ≥ 0.72 | ✅ VALIDATED |
| H3 — Plurilingual exposure → attitudes | MAE | 0.47 | ≤ 0.50 | ⚠️ partial |
| H4 — Local language integration → engagement | Spearman ρ / subset acc | 1.00 / 1.00 | ≥ 0.55 / ≥ 0.45 | ⚠️ partial (F1-motivation 0.55) |

> The ⚠️ marks are honest, expected outcomes — use them to discuss model
> limitations and next steps (Phase 9 hyperparameter tuning) in the defense.

---

## 3. 5-minute live demo script (defense day)

Run everything on **your** laptop while sharing your screen.

1. Open the endpoint page in your browser:
   `https://dbc-9e268203-7090.cloud.databricks.com/ml/endpoints/flp-all-models`
   → show "READY" state.
2. Live inference — all 8 models in one call:
   ```powershell
   python databricks/invoke_endpoint.py all
   ```
3. Targeted hypotheses (faster, clearly scoped):
   ```powershell
   python databricks/invoke_endpoint.py h1
   python databricks/invoke_endpoint.py h3r
   ```
4. Show the catalog in your browser (Catalog Explorer → `flp_catalog`).
5. Show the workflow job's last run (Job runs → all green).

Total: ~5 minutes. The jury sees everything live with zero credentials.

---

## 4. Never share with the jury (do-not-share list)

- ❌ The workspace URL with login: `https://dbc-9e268203-7090.cloud.databricks.com/`
- ❌ The Databricks personal access token (`dapi...`) — it is a full-access credential
- ❌ The raw dataset `data_FLP.csv` (or any export of `raw.survey_responses`)
- ❌ The GitHub repo, if its public files contain data-derived artifacts (model pickles)
- ❌ Any notebook execution link that embeds credentials

If a jury member asks for data or access, offer a **redacted/anonymized summary**
(aggregate statistics only) instead.

---

## 5. Data-privacy notes (talking points)

- 500 raw respondents → **495 retained** after the consent filter ("J'accepte")
- Anonymized: timestamps and identifiers removed before any processing
- Demo path (dashboard / serving endpoint) transmits only 60 engineered features
  per prediction — never raw responses
