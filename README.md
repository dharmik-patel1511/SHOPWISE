# SHOPWISE — E-Commerce Conversion Intelligence

## IBM Internship Project

**Project type:** Data Analytics + optional Machine Learning  
**Objective:** Convert online shopping-session data into business KPIs, behavioral insights, and actionable recommendations.

### Business question

> Which browsing and session characteristics are associated with successful online purchases, and how can an e-commerce team use those signals to improve conversion?

The project follows the internship's Data → Information → Insight → Decision → Action approach.

## Dataset

This project uses the **UCI Online Shoppers Purchasing Intention Dataset (ID 468)**.

- Official source: https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
- Dataset size: 12,330 sessions
- Target: `Revenue` (purchase / no purchase)
- Contains browsing activity, engagement metrics, page-value, visitor, traffic, month, and weekend attributes.

**Important submission check:** verify that this dataset is not the exact dataset used in your internship masterclass. The internship session explicitly prohibited reuse of the masterclass learning dataset.

## Deliverables

1. `DharmikPatel_SHOPWISE.py` — single executable project code file.
2. `requirements.txt` — Python dependencies.
3. `DharmikPatel_ProjectReport.docx` — project report.
4. `README.md` — project documentation and dataset source.

## How to run

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install:

```bash
pip install -r requirements.txt
```

Run:

```bash
python DharmikPatel_SHOPWISE.py
```

The script downloads the public dataset through `ucimlrepo`, performs cleaning and feature engineering, calculates business KPIs, creates visualizations, and trains two optional classification models.

Generated files:

```text
outputs/
├── kpis.csv
├── visitor_type_performance.csv
├── monthly_performance.csv
├── traffic_performance.csv
├── model_metrics.csv
├── purchase_outcome.png
├── visitor_conversion.png
└── monthly_conversion.png
```

## Main KPIs

- Total sessions
- Purchasers
- Conversion rate
- Average Page Value
- Conversion by visitor type
- Conversion by month
- Conversion by traffic type

## Feature engineering

- `TotalPages`
- `TotalDuration`
- `AvgSecondsPerPage`
- `EngagementScore`

## Optional predictive layer

Prediction is not required by the internship guidance. This project includes it as an additional analytical layer using:

- Logistic Regression
- Random Forest

Evaluation:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

## Action framework

The analysis is designed to translate findings into actions such as:

- Improve high-exit/high-bounce sessions.
- Prioritize high-intent traffic segments.
- Tailor interventions for new vs. returning visitors.
- Use page-value and engagement signals to identify high-intent sessions.
- Review monthly and traffic patterns before allocating campaigns.

Final actions should be based on the generated outputs after execution.

## Reproducibility

Run the single Python file from the project root. The train/test split and Random Forest use fixed random seed `42`.

## Dataset citation

Sakar, C. O. & Kastro, Y. (2018). *Online Shoppers Purchasing Intention Dataset*. UCI Machine Learning Repository. DOI: 10.24432/C5F88Q.

## Project structure

```text
SHOPWISE/
├── DharmikPatel_SHOPWISE.py
├── requirements.txt
├── README.md
├── DharmikPatel_ProjectReport.docx
└── outputs/                 # generated after execution
```
