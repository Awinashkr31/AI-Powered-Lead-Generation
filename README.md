# AI-Powered Lead Generation — Lead Scoring

Not all leads are worth the same phone call. This project trains a machine-learning model that scores every lead from **0 to 100** by its probability of converting, and groups leads into **Hot / Warm / Cold** bands — so a sales team spends its time on the leads most likely to buy.

## Problem

Sales teams waste most of their effort on leads that were never going to convert. Lead scoring fixes the order of work: predict each lead's conversion probability, rank the list, call the hottest leads first. This project implements the full loop — training, evaluation, scoring, banding, and a ranked sample output.

## Dataset

**UCI Bank Marketing dataset** (Moro, Cortez & Rita, 2014) — 45,211 client records from phone-marketing campaigns of a Portuguese bank, with features such as age, job, marital status, education, account balance, housing/personal loans, contact channel, and the outcome of the previous campaign.

- Source: UCI Machine Learning Repository — https://archive.ics.uci.edu/dataset/222/bank+marketing (file `bank-full.csv`)
- **What "conversion" means here, honestly:** the dataset's target `y` records whether the contacted client subscribed to a term deposit (yes/no). In this project each contacted client is treated as a *lead* and a subscription as a *conversion*, so the model learns to score leads by conversion likelihood. This is public benchmark data used to demonstrate the lead-scoring method — it is not any company's proprietary lead data.
- Overall conversion rate: **11.70%** (5,289 of 45,211).

### A deliberate honesty choice: `duration` is excluded

The dataset contains `duration` — the length of the last call in seconds. It is strongly predictive, but it is only known *after* the call has ended, so no real lead-scoring system could use it to decide whom to call. The dataset authors flag it for exactly this reason. It is dropped from the features here; the model scores leads only on information available *before* contact.

## Approach

All in one reproducible script, `lead_scoring.py`:

1. **Cleaning** — parse the `;`-separated CSV, map the target `y` (yes/no) to 1/0, drop the leakage column `duration`.
2. **EDA** — conversion rate, age range, and average balance of converters (€1,804) vs. non-converters (€1,304).
3. **Preprocessing** — sklearn `ColumnTransformer` in a `Pipeline`: median imputation + scaling for numerics, most-frequent imputation + one-hot encoding for categoricals.
4. **Split** — stratified 80/20 train/test split (36,168 train / 9,043 test, `random_state=42`).
5. **Models** — Logistic Regression, Random Forest (300 trees), XGBoost (300 trees), with `class_weight="balanced"` / `scale_pos_weight` for the ~11.7% positive class.
6. **Evaluation & selection** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix on the held-out test set; best model picked by F1.
7. **Lead scoring** — the best model's probabilities become a 0–100 score (`score_leads()` function): **Hot ≥ 70**, **Warm 40–69**, **Cold < 40**. Test-set leads are scored and the top 10 are saved to `results/sample_scored_leads.csv`.

## Results

Measured on the held-out test set (9,043 leads) from the actual run — full details in [`results/metrics.json`](results/metrics.json), raw console log in [`results/run_output.txt`](results/run_output.txt):

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.7548 | 0.2662 | 0.6238 | 0.3732 | 0.7722 |
| Random Forest | **0.8792** | **0.4810** | 0.4064 | 0.4406 | 0.7927 |
| **XGBoost** ✅ | 0.8237 | 0.3559 | **0.6257** | **0.4537** | **0.7971** |

**Best model: XGBoost** (highest F1, 0.4537; ROC-AUC 0.7971). Test-set confusion matrix: TN 6,787 · FP 1,198 · FN 396 · TP 662.

### Do the score bands actually separate good leads? Yes:

| Band | Leads (test set) | Actual conversion rate |
|---|---|---|
| 🔥 Hot (score ≥ 70) | 935 | **51.66%** |
| Warm (40–69) | 2,080 | 13.94% |
| Cold (< 40) | 6,028 | 4.73% |

Calling only the Hot band means roughly **1 in 2 calls converts**, versus ~1 in 21 in the Cold band and ~1 in 9 leads overall (11.70% base rate). That separation is the entire point of lead scoring.

### Sample output — top scored leads

From [`results/sample_scored_leads.csv`](results/sample_scored_leads.csv) (top of the ranked test set; full table in the file):

| Age | Job | Education | Balance (€) | Previous campaign | Score | Band | Actually converted |
|---|---|---|---|---|---|---|---|
| 20 | student | secondary | 215 | success | 99 | Hot | ✅ yes |
| 54 | technician | tertiary | 876 | success | 99 | Hot | ✅ yes |
| 60 | retired | secondary | 523 | success | 99 | Hot | ✅ yes |
| 36 | technician | tertiary | 341 | success | 99 | Hot | ✅ yes |
| 72 | retired | tertiary | 3,561 | success | 99 | Hot | ✅ yes |
| 39 | management | tertiary | 224 | success | 98 | Hot | ✅ yes |

A clear pattern the model picks up: a **successful previous campaign outcome** is the strongest conversion signal.

The trained best-model pipeline is saved to `models/best_lead_scoring_model.pkl` (joblib) by the script.

## How to run

```bash
pip install -r requirements.txt
python lead_scoring.py
```

If `data/bank-full.csv` is not present, the script downloads the dataset zip from the UCI source above and extracts it automatically. Outputs: printed metrics and band analysis, `results/metrics.json`, `results/sample_scored_leads.csv`, and the saved model in `models/`.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · joblib

## Project structure

```
ai-lead-generation/
├── lead_scoring.py             # full pipeline: load → train → evaluate → score leads
├── requirements.txt
├── data/                       # dataset (downloaded automatically if missing)
├── results/
│   ├── metrics.json            # measured metrics from the real run
│   ├── sample_scored_leads.csv # top-10 ranked leads with scores & bands
│   └── run_output.txt          # console log of the run
└── models/
    └── best_lead_scoring_model.pkl  # trained pipeline, regenerated by the script
```

## Reference

Moro, S., Cortez, P., & Rita, P. (2014). *A data-driven approach to predict the success of bank telemarketing.* Decision Support Systems, 62, 22–31.
