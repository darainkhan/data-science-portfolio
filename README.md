# 📊 Data Science & AI Portfolio

End-to-end machine learning projects, data engineering pipelines, and agentic AI systems — from EDA through modeling, evaluation, and business operationalization.

> **Note**: These projects reflect real work built in my professional role. Due to confidentiality, proprietary data and business logic have been anonymized — public datasets (e.g., airfare pricing, PayPal transactions) are used to demonstrate the same techniques and methodologies applied in production.

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

### 3. [🎯 Customer Segmentation with K-Means & PCA](./customer-segmentation-kmeans/)

Unsupervised learning pipeline that segments customers into actionable clusters based on loyalty program participation, purchase behavior, and retention metrics.

| Metric | Result |
|--------|--------|
| Method | PCA (74 → 13 features) + K-Means (k=4) |
| Variance Captured | 90%+ with 13 principal components |
| Segments Found | 4 distinct clusters with clear business interpretation |
| Key Techniques | StandardScaler, PCA component analysis, elbow method, cluster profiling |

---

### 4. [🔒 PayPal Fraud Detection](./paypal-fraud-detection/)

Detecting fraudulent transactions with a two-threshold operationalization system (LOCK / ALERT / ALLOW).

| Metric | Result |
|--------|--------|
| Model | XGBoost Classifier |
| AUC-ROC | 0.969 |
| Key Technique | Probability-based multi-action decision system |

---

### 5. [🍕 DoorDash Fraudulent Customer Detection](./doordash-fraud-detection/)

Detecting fraudulent consumer accounts using behavioral and transactional features with XGBoost, featuring missingness-as-a-signal engineering and precision-recall threshold analysis.

| Metric | Result |
|--------|--------|
| Model | XGBoost Classifier |
| F1 (Fraud) | 0.81 |
| Recall (Fraud) | 84% |
| Key Techniques | Missing indicator features, RFECV feature selection, fraud loss quantification |

---

### 6. [✈️ Airfare Price Prediction](./airfare-price-prediction/)

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
| Normalization | Data Warehouse, Incentive Pipeline |
| ML Proposal & Business Case Writing | Intelligence Agent |
| Unsupervised Learning (PCA, K-Means) | Customer Segmentation |
| Cluster Analysis & Profiling | Customer Segmentation |
| Dimensionality Reduction | Customer Segmentation |
| Feature Engineering | Airfare, Fraud Detection (PayPal), Fraud Detection (DoorDash), Customer Segmentation |
| Handling Class Imbalance | Fraud Detection (PayPal), Fraud Detection (DoorDash) |
| Hyperparameter Tuning (Grid/RandomSearch) | Airfare, Fraud Detection (PayPal), Fraud Detection (DoorDash) |
| Recursive Feature Elimination (RFE/RFECV) | Airfare, Fraud Detection (PayPal), Fraud Detection (DoorDash) |
| Log Transformation | Airfare |
| Threshold Optimization | Fraud Detection (PayPal), Fraud Detection (DoorDash) |
| Missing Value Engineering | Fraud Detection (DoorDash) |
| Business Operationalization | Fraud Detection (PayPal), Fraud Detection (DoorDash), Intelligence Agent, Customer Segmentation |
| Multicollinearity Detection | Fraud Detection |
| Security & Access Control (RLS) | Intelligence Agent |

---

## Tech Stack

Python · pandas · NumPy · scikit-learn · XGBoost · matplotlib · seaborn · Multi-Agent Orchestration · SQL · Cloud Data Warehouse · S3 · Parquet · BI Dashboards · RAG
