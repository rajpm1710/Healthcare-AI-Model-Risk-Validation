"""Synthetic enterprise SaaS renewal/churn prediction workflow."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from common import metrics, psi, write_report

def generate(n=4000, seed=31):
    rng = np.random.default_rng(seed)
    seats = rng.integers(25, 4000, n)
    utilization = rng.beta(3.5, 2.0, n)
    support_tickets = rng.poisson(2.0 + 3 * (1 - utilization), n)
    incidents = rng.poisson(0.4 + support_tickets / 10, n)
    adoption = rng.beta(2.5, 2.3, n)
    renewal_days = rng.integers(15, 365, n)
    price_increase = rng.uniform(0, 0.20, n)
    segment = rng.choice(['SMB', 'Mid-market', 'Enterprise'], n, p=[0.3, 0.45, 0.25])
    logit = -2.6 + 2.0 * (1 - utilization) + 1.4 * (1 - adoption) + 0.13 * support_tickets + 0.25 * incidents + 3.0 * price_increase - 0.00015 * seats
    churn = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    return pd.DataFrame({'licensed_seats': seats, 'product_utilization': utilization, 'support_tickets_90d': support_tickets, 'critical_incidents_90d': incidents, 'feature_adoption': adoption, 'days_to_renewal': renewal_days, 'price_increase_pct': price_increase, 'customer_segment': segment, 'did_not_renew': churn})

def run():
    df = generate()
    train, test = train_test_split(df, test_size=0.3, stratify=df.did_not_renew, random_state=42)
    cols = ['licensed_seats', 'product_utilization', 'support_tickets_90d', 'critical_incidents_90d', 'feature_adoption', 'days_to_renewal', 'price_increase_pct']
    Xtr, Xte = train[cols], test[cols]
    ytr, yte = train.did_not_renew, test.did_not_renew
    model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=12, l2_regularization=4, random_state=42)
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(Xtr, ytr)
    baseline.fit(Xtr, ytr)
    pred = model.predict_proba(Xte)[:, 1]
    base = baseline.predict_proba(Xte)[:, 1]
    imp = permutation_importance(model, Xte, yte, scoring='roc_auc', n_repeats=3, random_state=42)
    ranking = sorted(zip(cols, imp.importances_mean), key=lambda x: -x[1])
    results = test[['customer_segment', 'did_not_renew']].copy()
    results['risk'] = pred
    results['priority'] = pd.qcut(results['risk'].rank(method='first'), 5, labels=['Q1 low', 'Q2', 'Q3', 'Q4', 'Q5 high'])
    by_tier = results.groupby('priority', observed=True).agg(accounts=('risk', 'size'), churn_rate=('did_not_renew', 'mean'), average_risk=('risk', 'mean')).reset_index()
    report = {'case_study': 'Synthetic enterprise software license-renewal churn prediction', 'data_provenance': 'Entirely synthetic; no VMware customer data', 'records': len(df), 'test_churn_rate': round(float(yte.mean()), 4), 'gradient_boosting': metrics(yte, pred), 'logistic_regression_baseline': metrics(yte, base), 'risk_quintiles': [{ 'tier': str(r.priority), 'accounts': int(r.accounts), 'observed_churn_rate': round(float(r.churn_rate), 4), 'mean_predicted_risk': round(float(r.average_risk), 4)} for r in by_tier.itertuples()], 'feature_importance_permutation_auc': [{'feature': k, 'importance': round(float(v), 4)} for k, v in ranking], 'simulated_usage_drop_psi': psi(test.product_utilization.to_numpy(), (test.product_utilization * 0.8).to_numpy()), 'business_action': 'Prioritize human customer-success review for top-risk accounts; avoid automated account actions.', 'limitations': ['Synthetic outcomes and features; no actual business ROI established', 'Random holdout rather than forward-looking renewal cohort', 'No cost-sensitive decision threshold or causal intervention analysis', 'Risk scores require calibration and monitoring before use']}
    write_report('enterprise_churn', report)
    return report

if __name__ == '__main__': run()
