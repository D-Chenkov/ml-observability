# Project #5: Production ML Observability & Drift-Triggered Retraining Platform

> **Flagship Portfolio Project (Scheduled: Weeks 9B & 10 — Nov 2 to Nov 15, 2026)**
> Bridges 3.5+ years of C3 AI production ML pipeline validation & behavioral drift detection directly into an open-source MLOps & GCP PMLE reference architecture.

## Architecture & Scope
```
[Reference + Live Batches] 
       │
       ▼
[FastAPI Model Endpoint (/predict, /metrics)] ──► [Prometheus] ──► [Grafana Dashboards (Latency p99 + Drift Score)]
       │
       ▼
[Evidently AI Drift Evaluator (Scheduled GitHub Action)]
       │
       ├─► Drift <= Threshold: Log healthy status
       └─► Drift > Threshold : Trigger Retraining Pipeline -> Champion vs. Challenger Gate -> Redeploy
```

## Milestone Checklist
- [ ] **Step 1 (Wk 9B Mon–Thu):** Evidently `DataDriftReport` + FastAPI instrumented with `prometheus-client` (`Counter`, `Histogram`) + Grafana dashboard in Docker Compose.
- [ ] **Step 2 (Wk 9B Fri–Sun):** Baseline Churn/Fraud classifier + serialized artifact + K8s manifests + scheduled Evidently batch drift evaluator.
- [ ] **Step 3 (Wk 10 Mon–Thu):** Custom PromQL drift score panel in Grafana + simulated covariate/concept drift + GitHub Actions auto-retraining trigger with champion/challenger evaluation gate.
- [ ] **Step 4 (Wk 10 Fri–Sat):** Full `README.md` with Mermaid architecture diagram, operational runbook, and 2–3 min Loom walkthrough.
