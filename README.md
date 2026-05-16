# 📊 Data Science & AI Portfolio

End-to-end machine learning projects, data engineering pipelines, and agentic AI systems — from EDA through modeling, evaluation, and business operationalization.

---

## Projects

### 1. [🤖 Agentic AI — Multi-Source Intelligence System](./agentic-ai-funding-agent/)
Multi-source AI agent combining qualitative knowledge (wikis, PDFs) with quantitative data analysis (SQL) to serve as a unified intelligence layer for enterprise operations.

| Component | Detail |
|-----------|--------|
| Architecture | Multi-agent orchestrator with intent routing |
| Data Sources | SQL database, BI dashboards, Wiki, PDF documents |
| Key Feature | Natural language queries across 8 programs with row-level security |
| Extension | Proposed predictive revenue model for partner performance scoring |

---

### 2. [🔄 ETL Pipeline — Multi-Source Data Warehouse](./etl-data-warehouse-pipeline/)
Production ETL pipeline consolidating 3 disparate source systems into a unified analytical layer with currency normalization, deduplication, billing reconciliation, and data lake export.

| Component | Detail |
|-----------|--------|
| Scale | 200+ columns, 500K+ records, 15+ source tables, daily refresh |
| Architecture | 5-layer pipeline (Extract → Consolidate → SOT → Stage → Report) |
| Key Techniques | Multi-source COALESCE, window function dedup, currency conversion, workflow history pivot, Parquet export |
| Consumers | BI Dashboards, AI Agent, Finance, Leadership |

---

### 3. [🔄 ETL Pipeline — Incentive Program Data Warehouse](./etl-partner-funding-pipeline/)
Production ETL pipeline consolidating operational incentive data from 3 source systems into a unified analytical layer with currency normalization, deduplication, and data lake export.

| Component | Detail |
|-----------|--------|
| Scale | 200+ columns, 500K+ records, 15+ source tables |
| Layers | Extract → Consolidate → Stage → Report |
| Key Techniques | Multi-source COALESCE, window function dedup, currency conversion, Parquet export |
| Consumers | Dashboards, AI Agent, Finance, Leadership |

---

### 4. [🔒 PayPal Fraud Detection](./paypal-fraud-detection/)
Detecting fraudulent transactions with a two-threshold operationalization system (LOCK / ALERT / ALLOW).

| Metric | Result |
|--------|--------|
| Model | XGBoost Classifier |
| AUC-ROC | 0.969 |
| Key Technique | Probability-based multi-action decision system |

---

### 5. [✈️ Airfare Price Prediction](./airfare-price-prediction/)
Predicting flight ticket prices using XGBoost Regressor with log-transformed targets.

| Metric | Result |
|--------|--------|
| Model | XGBoost Regressor |
| MAPE | 52.60% |
| Key Technique | Log transformation of skewed target |

---

## Skills Demonstrated

| Skill | Projects |
|-------|----------|
| Agentic AI / Multi-Agent Systems | Intelligence Agent |
| RAG Architecture | Intelligence Agent |
| ETL Pipeline Design | Data Warehouse, Incentive Pipeline |
| Data Warehousing | Data Warehouse, Incentive Pipeline |
| Advanced SQL (CTEs, Window Functions) | Data Warehouse, Incentive Pipeline |
| Multi-Source Data Integration | Data Warehouse, Incentive Pipeline, Intelligence Agent |
| Data Quality Engineering | Data Warehouse |
| Currency Normalization | Data Warehouse, Incentive Pipeline |
| ML Proposal & Business Case Writing | Intelligence Agent |
| Feature Engineering | Airfare, Fraud Detection |
| Handling Class Imbalance | Fraud Detection |
| Hyperparameter Tuning (RandomSearch) | Airfare, Fraud Detection |
| Recursive Feature Elimination (RFE) | Airfare, Fraud Detection |
| Log Transformation | Airfare |
| Threshold Optimization | Fraud Detection |
| Business Operationalization | Fraud Detection, Intelligence Agent |
| Multicollinearity Detection | Fraud Detection |
| Security & Access Control (RLS) | Intelligence Agent |

---

## Tech Stack
Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn · Multi-Agent Orchestration · SQL · Cloud Data Warehouse · S3 · Parquet · BI Dashboards · RAG
