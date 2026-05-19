# Partner Program Analytics: Unsupervised Segmentation

> ⚠️ **DISCLAIMER:** All data in this repository is entirely synthetic/fake and generated for learning and demonstration purposes only. No real partner data, business metrics, or proprietary information is included. This project showcases methodology and analytical thinking — not real results.

An end-to-end machine learning pipeline that segments technology/consulting partners into actionable clusters based on program participation, revenue performance, and retention metrics.

## Overview

This project applies unsupervised learning techniques to a partner program health dataset to identify distinct partner segments. The goal is to surface data-driven insights that inform partner development strategies, program investment decisions, and retention initiatives.

**Key Techniques:**
- Feature engineering (numerical, categorical, binary)
- StandardScaler normalization
- PCA for dimensionality reduction (75 features → 13 components, 90%+ variance explained)
- K-Means clustering
- Cluster profiling and business interpretation

## Results

The analysis identifies **4 distinct partner segments**:

| Cluster | Label | Size | Key Characteristics |
|---------|-------|------|---------------------|
| 0 | Basic Partners | ~95% | Moderate program adoption, lower retention, geographically diverse |
| 1 | High Performers | ~2% | High revenue (2.5x avg), balanced partner types, concentrated geography |
| 2 | Elite Partners | ~0.5% | Exceptional performance (8-10x avg), technology-focused, single-region |
| 3 | Program Specialists | ~2% | Highest retention (86-100%), consulting-focused, globally diverse |

## Project Structure

```
partner-program-analytics/
├── README.md                              # This file
├── METHODOLOGY.md                         # Detailed step-by-step analysis walkthrough
├── generate_synthetic_data.py             # Generates realistic synthetic dataset
└── requirements.txt                       # Python dependencies
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Generate Synthetic Data

The repository includes a synthetic data generator that produces a dataset matching the schema and statistical properties of the original data:

```bash
python generate_synthetic_data.py
```

This produces `synthetic_partner_program_health.csv` (~25,000 rows, 2,500 partners, 55 features).

### Analysis Walkthrough

See [METHODOLOGY.md](METHODOLOGY.md) for a complete step-by-step walkthrough of the analysis pipeline, including code snippets, PCA results, cluster profiles, and business recommendations.

## Methodology

### 1. Data Preparation
- Load partner-level program health data (multiple monthly snapshots per partner)
- Drop identifiers and date columns not needed for modeling
- Classify features into numerical (34), categorical (3), and binary (16) groups

### 2. Feature Engineering
- One-hot encode categorical variables (partner type, tier, region)
- Standardize numerical features using `StandardScaler`
- Preserve binary flags as-is
- Aggregate multi-row partner data to single partner-level representation

### 3. Dimensionality Reduction (PCA)
- Apply PCA to the 75-feature matrix
- Retain 13 principal components capturing ~90% of variance
- Analyze component loadings to understand what each PC represents:
  - **PC1** (40%): Overall partner scale (launched ARR, qualified opportunities)
  - **PC2** (17%): Program engagement depth (program-specific customer counts)
  - **PC3-5**: Program-specific performance and maturity
  - **PC6-13**: Funding, channel revenue, retention patterns

### 4. Clustering (K-Means)
- Apply K-Means with k=4 on PCA-reduced features
- Profile each cluster across program participation, partner type/tier, geography, and retention

### 5. Business Interpretation
- Map clusters to actionable partner development strategies
- Identify retention best practices from high-performing segments
- Recommend targeted program adoption strategies per segment

## Key Insights

- **Retention correlates with program depth** — partners enrolled in multiple programs show 2-3x higher retention rates
- **Geographic concentration in elite tiers** — top-performing partners are heavily concentrated in one region, suggesting opportunity for global expansion
- **Partner type divergence** — different partner types excel at different things (retention vs. revenue scale)
- **Clear progression ladder** — data supports a partner development pathway from basic → high performer → elite

## Tech Stack

- Python 3.10+
- pandas, numpy — data manipulation
- scikit-learn — StandardScaler, PCA, KMeans
- matplotlib, seaborn — visualization

## Data Note

This repository uses **synthetic data** generated to match the statistical distributions and schema of the original dataset. No proprietary or confidential information is included. All field names, program names, and region identifiers have been anonymized. The synthetic data generator (`generate_synthetic_data.py`) is provided for full reproducibility.

## License

MIT
