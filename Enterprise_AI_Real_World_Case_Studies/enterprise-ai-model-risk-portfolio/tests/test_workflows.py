import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from healthcare_validation import generate as generate_health, run as run_health
from vmware_churn import generate as generate_churn, run as run_churn
from common import psi

def test_health_reproducible():
    assert generate_health(100, 2).equals(generate_health(100, 2))

def test_churn_reproducible():
    assert generate_churn(100, 2).equals(generate_churn(100, 2))

def test_psi_identical():
    assert psi(list(range(100)), list(range(100))) == 0

def test_health_pipeline():
    r = run_health()
    assert 0 <= r['champion']['roc_auc'] <= 1
    assert len(r['fairness_audit_by_simulated_group']) == 2

def test_churn_pipeline():
    r = run_churn()
    assert len(r['risk_quintiles']) == 5
    assert 0 <= r['gradient_boosting']['roc_auc'] <= 1
