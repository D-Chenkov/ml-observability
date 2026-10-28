"""
Project #5: Production ML Observability & Drift-Triggered Retraining Platform
Step 1: Baseline Evidently AI DataDriftReport on Telco Churn Data

Demonstrates:
- Reference vs. Current dataset comparison
- Numerical drift (Wasserstein / KS-test)
- Categorical drift (Chi-squared / PSI)
- HTML report generation and JSON metrics extraction for Prometheus/Grafana export
"""

import json
import os
import numpy as np
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset


def generate_telco_churn_data(n_samples: int, seed: int = 42, drift_factor: float = 0.0) -> pd.DataFrame:
    """Generate realistic synthetic customer churn features with optional injected covariate shift."""
    rng = np.random.default_rng(seed)

    # Numerical features
    # If drift_factor > 0: simulate price increases and shorter tenures (covariate shift)
    tenure_mean = max(5.0, 32.0 - (drift_factor * 12.0))
    tenure = np.clip(rng.gamma(shape=2.5, scale=tenure_mean / 2.5, size=n_samples), 1, 72).round()

    monthly_base = 65.0 + (drift_factor * 35.0)  # shift mean monthly charges upward
    monthly_charges = np.clip(rng.normal(loc=monthly_base, scale=25.0, size=n_samples), 18.0, 140.0).round(2)
    total_charges = np.round(tenure * monthly_charges * rng.uniform(0.92, 1.05, size=n_samples), 2)

    # Categorical features
    if drift_factor == 0.0:
        contract_probs = [0.55, 0.25, 0.20]  # Month-to-month, One year, Two year
        payment_probs = [0.35, 0.25, 0.20, 0.20]  # Elec check, Mailed check, Bank transfer, Credit card
    else:
        # Drifted traffic: surge in month-to-month contracts and electronic check payments
        contract_probs = [0.85, 0.10, 0.05]
        payment_probs = [0.65, 0.15, 0.10, 0.10]

    contract_types = ["Month-to-month", "One year", "Two year"]
    payment_methods = ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
    internet_services = ["DSL", "Fiber optic", "No"]

    contracts = rng.choice(contract_types, size=n_samples, p=contract_probs)
    payments = rng.choice(payment_methods, size=n_samples, p=payment_probs)
    internet = rng.choice(internet_services, size=n_samples, p=[0.34, 0.44, 0.22])

    df = pd.DataFrame({
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Contract": contracts,
        "PaymentMethod": payments,
        "InternetService": internet,
    })
    return df


def run_drift_analysis():
    reports_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "reports")
    os.makedirs(reports_dir, exist_ok=True)

    print("=" * 70)
    print("📊 PROJECT #5: EVIDENTLY AI DATA DRIFT PIPELINE")
    print("=" * 70)

    # 1. Generate Reference (Baseline) and Current (Production) Datasets
    print("\n[1/4] Generating datasets...")
    reference_df = generate_telco_churn_data(n_samples=2500, seed=42, drift_factor=0.0)
    healthy_current_df = generate_telco_churn_data(n_samples=1000, seed=101, drift_factor=0.0)
    drifted_current_df = generate_telco_churn_data(n_samples=1000, seed=202, drift_factor=0.85)

    print(f"  - Reference baseline : {len(reference_df)} rows")
    print(f"  - Healthy current    : {len(healthy_current_df)} rows (no distribution shift)")
    print(f"  - Drifted current    : {len(drifted_current_df)} rows (injected pricing + contract shift)")

    # 2. Run Drift Report on Healthy Batch
    print("\n[2/4] Running DataDriftReport on healthy production batch...")
    healthy_report = Report(metrics=[DataDriftPreset()])
    healthy_report.run(reference_data=reference_df, current_data=healthy_current_df)
    healthy_html_path = os.path.join(reports_dir, "drift_report_healthy.html")
    healthy_report.save_html(healthy_html_path)
    healthy_dict = healthy_report.as_dict()

    dataset_drift_healthy = healthy_dict["metrics"][0]["result"]["dataset_drift"]
    drift_share_healthy = healthy_dict["metrics"][0]["result"]["drift_share"]
    print(f"  -> Healthy Batch Result: dataset_drift = {dataset_drift_healthy} (drift_share = {drift_share_healthy:.1%})")
    print(f"  -> Saved HTML: {healthy_html_path}")

    # 3. Run Drift Report on Drifted Batch
    print("\n[3/4] Running DataDriftReport on drifted production batch...")
    drifted_report = Report(metrics=[DataDriftPreset()])
    drifted_report.run(reference_data=reference_df, current_data=drifted_current_df)
    drifted_html_path = os.path.join(reports_dir, "drift_report_drifted.html")
    drifted_report.save_html(drifted_html_path)
    drifted_dict = drifted_report.as_dict()

    drift_res = drifted_dict["metrics"][0]["result"]
    dataset_drift_flag = drift_res["dataset_drift"]
    number_of_drifted_columns = drift_res["number_of_drifted_columns"]
    total_columns = drift_res["number_of_columns"]
    drift_share = drift_res["drift_share"]

    print(f"  -> Drifted Batch Result: dataset_drift = {dataset_drift_flag}")
    print(f"  -> Drifted columns     : {number_of_drifted_columns} of {total_columns} ({drift_share:.1%})")
    print(f"  -> Saved HTML: {drifted_html_path}")

    # 4. Print Per-Feature Statistical Test Breakdown
    print("\n[4/4] Per-Feature Drift Telemetry (Evidently -> Prometheus Schema):")
    print("-" * 75)
    print(f"{'Feature':<18} | {'Type':<12} | {'Test Used':<16} | {'Score/P-Val':<12} | {'Drift?':<6}")
    print("-" * 75)

    metrics_export = {
        "dataset_drift": bool(dataset_drift_flag),
        "number_of_drifted_columns": int(number_of_drifted_columns),
        "total_columns": int(total_columns),
        "drift_share": float(drift_share),
        "features": {}
    }

    table_res = drifted_dict["metrics"][1]["result"]
    drift_by_columns = table_res["drift_by_columns"]
    for col_name, col_data in drift_by_columns.items():
        col_type = col_data["column_type"]
        stattest_name = col_data["stattest_name"]
        drift_score = float(col_data["drift_score"])
        is_drifted = bool(col_data["drift_detected"])

        metrics_export["features"][col_name] = {
            "type": col_type,
            "test": stattest_name,
            "score": float(drift_score),
            "drift_detected": bool(is_drifted)
        }

        print(f"{col_name:<18} | {col_type:<12} | {stattest_name:<16} | {drift_score:<12.4f} | {str(is_drifted):<6}")

    print("-" * 75)

    # Save summary JSON for Prometheus scrape endpoint integration
    json_path = os.path.join(reports_dir, "drift_summary.json")
    with open(json_path, "w") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"\n✅ Drift metrics JSON exported to: {json_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_drift_analysis()
