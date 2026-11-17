# Enterprise AI Model Risk Portfolio

**Two realistic, reproducible industry case studies inspired by healthcare model validation and enterprise software renewal analytics.** These are **independent personal projects** using **100% synthetic data**. No code, datasets, findings, or employer-specific implementation are sourced from CVS Health or VMware. The project names describe industry use cases, not employer endorsement.

## Projects

| Case study | Business problem | Technical demonstration |
|---|---|---|
| Healthcare Readmission Risk — Independent Validation | Evaluate whether a model identifying patients at elevated 30-day readmission risk is reliable and equitable | Synthetic member data, holdout validation, gradient boosting vs logistic challenger, ROC AUC / PR AUC / Brier, subgroup audit, permutation importance, PSI stress scenario, risk report |
| Enterprise SaaS License Renewal — Churn Risk | Identify enterprise accounts at risk of not renewing software licenses | Synthetic account usage and support history, gradient boosting vs logistic baseline, renewal risk quintiles, model interpretation, simulated usage drift, customer-success recommendations |

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/healthcare_validation.py
python src/vmware_churn.py
pytest -q
```

Generated JSON reports appear in `reports/`. Included sample reports were produced by running the scripts locally.

## How the work mirrors a real enterprise workflow

1. **Define intended use and risk:** distinguish healthcare clinical-risk triage from software customer-success prioritization.
2. **Construct data and baseline:** use explicitly synthetic features and outcomes, with a simple logistic regression challenger.
3. **Validate independently:** compare discrimination, precision/recall, calibration-oriented Brier score and subgroup performance.
4. **Stress and monitor:** simulate feature distribution shifts and calculate Population Stability Index (PSI).
5. **Document limitations:** report data provenance, holdout limitations, model risks and proposed human review.

## Interpretation and safeguards

- These scripts are **educational demonstrations**, not clinically validated models or production software.
- The healthcare model is **not for clinical decision-making** and contains no real patient or PHI data.
- Subgroup metrics alone do not establish fairness. In a real setting, add statistical confidence intervals, applicable policy review, representative data, and a documented decision threshold.
- PSI reflects feature distribution change, **not proof of model performance degradation**.
- A random train/test split is only a starting point. Real-world deployment requires temporal validation, data quality controls, leakage review, governance approvals, and ongoing monitoring.
- Avoid claiming that synthetic results reproduce metrics from prior employers.

## Interview explanation

> I created two independent portfolio case studies reflecting problems I discuss in my resume: healthcare model risk validation and enterprise software renewal prediction. I generated synthetic data, built champion/challenger comparisons, audited model performance, and documented limitations. The code is my personal work, not a copy of any employer system.

## Possible next iterations

- Temporal validation and confidence intervals
- Fairlearn subgroup metric dashboards and SHAP visualizations
- Streamlit dashboard and automated PDF model validation report
- MLflow tracking and CI pipeline
- SQL warehouse mock data and data lineage diagrams

## Suggested GitHub repository name

`enterprise-ai-model-risk-case-studies`

## License

MIT (see `LICENSE`).
