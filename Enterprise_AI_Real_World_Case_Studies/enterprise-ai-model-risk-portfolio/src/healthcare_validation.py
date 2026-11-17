"""Synthetic healthcare readmission-risk independent validation demonstration."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from common import metrics, psi, write_report

def generate(n=3500, seed=17):
    rng = np.random.default_rng(seed)
    age = rng.integers(18, 90, n)
    chronic = rng.poisson(1.6, n).clip(0, 7)
    prior = rng.poisson(0.7 + 0.2 * chronic, n).clip(0, 8)
    adherence = rng.beta(5, 2, n)
    followup = rng.binomial(1, 0.7, n)
    region = rng.choice(['North', 'South', 'West'], n)
    # Simulated group is used for auditing, not as a predictive input.
    group = rng.choice(['Group A', 'Group B'], n)
    logit = -3.0 + 0.028 * (age - 50) + 0.30 * chronic + 0.43 * prior - 1.8 * adherence - 0.35 * followup
    risk = 1 / (1 + np.exp(-logit))
    y = rng.binomial(1, risk)
    return pd.DataFrame({'age': age, 'chronic_conditions': chronic, 'prior_admissions': prior, 'medication_adherence': adherence, 'followup_completed': followup, 'region': region, 'audit_group': group, 'readmitted_30d': y})

def run():
    df = generate()
    # Independent holdout with no data leakage; in a real deployment use temporal and site-based validation.
    train, test = train_test_split(df, test_size=0.3, stratify=df.readmitted_30d, random_state=42)
    cols = ['age', 'chronic_conditions', 'prior_admissions', 'medication_adherence', 'followup_completed']
    Xtr, Xte = train[cols], test[cols]
    ytr, yte = train.readmitted_30d, test.readmitted_30d
    champion = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=12, l2_regularization=3, random_state=42)
    challenger = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    champion.fit(Xtr, ytr)
    challenger.fit(Xtr, ytr)
    cp, lp = champion.predict_proba(Xte)[:, 1], challenger.predict_proba(Xte)[:, 1]
    audit = {}
    for grp in sorted(test.audit_group.unique()):
        mask = (test.audit_group == grp).to_numpy()
        audit[grp] = {'sample_size': int(mask.sum()), **metrics(yte.to_numpy()[mask], cp[mask], threshold=0.15)}
    importance = permutation_importance(champion, Xte, yte, n_repeats=3, random_state=42, scoring='roc_auc')
    top = sorted(zip(cols, importance.importances_mean), key=lambda x: -x[1])
    shifted = test.copy()
    shifted['prior_admissions'] += 1  # deliberate stress scenario, not observed production drift
    report = {'case_study': 'Synthetic 30-day hospital readmission model validation', 'data_provenance': 'Entirely synthetic; no CVS Health data', 'records': len(df), 'prevalence_test': round(float(yte.mean()), 4), 'champion': metrics(yte, cp, threshold=0.15), 'logistic_challenger': metrics(yte, lp, threshold=0.15), 'fairness_audit_by_simulated_group': audit, 'audit_note': 'Group metrics are descriptive; no fairness conclusion without uncertainty, context and policy review.', 'feature_importance_permutation_auc': [{ 'feature': k, 'importance': round(float(v), 4)} for k, v in top], 'stress_scenario_prior_admissions_psi': psi(test.prior_admissions.to_numpy(), shifted.prior_admissions.to_numpy()), 'limitations': ['Synthetic data and simulated labels are not clinical evidence', 'Random holdout does not establish prospective temporal validity', 'Calibration and subgroup performance require confidence intervals before deployment', 'Not a medical device or clinical decision support system']}
    write_report('healthcare_validation', report)
    return report

if __name__ == '__main__': run()
