# Methodology: Partner Program Segmentation

A step-by-step walkthrough of the unsupervised learning pipeline used to segment partners into actionable clusters.

---

## Step 1 — Data Loading & Initial Cleanup

**Input:** Partner program health dataset (~25,000 rows, ~2,500 unique partners, 55+ features)

Each row represents a monthly snapshot of a partner's program participation, revenue metrics, and retention status. Multiple rows exist per partner (one per month observed).

**Actions:**
- Load CSV into pandas DataFrame
- Drop columns not useful for modeling (identifiers, date fields, name fields)
- Verify zero null values across all remaining columns

**Why:** We want a clean, feature-rich dataset without leaking identifiers into the model.

---

## Step 2 — Feature Classification

Split the remaining columns into three groups based on their data type and how they should be preprocessed:

### Numerical Features (34 columns)
Continuous metrics that need standardization:
- **Maturity:** Years as a partner
- **Pre-treatment metrics:** Number of launched/qualified opportunities and associated ARR in the year before treatment
- **6-month performance:** Launched opportunities and ARR across various programs in the trailing 6 months
- **Funding:** Approved cash, approved credits, claimed cash in trailing 6 months
- **Lifetime totals:** Total opportunities, total customers, total launched ARR across all programs
- **Marketplace:** Total offers, total contract value
- **Channel:** Channel revenue, channel customers

### Categorical Features (3 columns)
Need one-hot encoding:
- `partner_type` — Two partner type categories
- `partner_tier` — Four tier levels
- `partner_region` — 21 geographic regions

### Binary Features (16 columns)
Already 0/1, kept as-is:
- **Program flags (11):** Whether the partner participates in each of 11 different programs
- **Retention flags (5):** Partner and customer retention indicators across different channels/programs

---

## Step 3 — Preprocessing

### 3a. One-Hot Encoding
```python
dummy_vars = pd.get_dummies(df[categorical_columns], drop_first=True).astype(int)
```
- Uses `drop_first=True` to avoid multicollinearity (the dropped category becomes the reference)
- Produces ~25 dummy columns

### 3b. Standardization
```python
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numerical), ...)
```
- Centers each numerical feature to mean=0, std=1
- Critical because K-Means is distance-based — features with larger scales would dominate otherwise

### 3c. Combine
```python
df_final = pd.concat([df_scaled, dummy_vars, df_binary], axis=1)
```
- Final feature matrix: **~75 columns × ~25,000 rows**

---

## Step 4 — Aggregate to Partner Level

Since each partner has ~10 monthly snapshots, we aggregate to one row per partner:

```python
df_aggregated = df_final.groupby('partner_id').mean().reset_index()
```

- **Result:** ~2,500 rows × 75 features
- Using `mean` captures the partner's average behavior over time
- For binary flags, the mean represents the proportion of months where the flag was active

---

## Step 5 — PCA (Dimensionality Reduction)

### Why PCA?
- 75 features is high-dimensional — K-Means suffers from the "curse of dimensionality"
- Many features are correlated (e.g., program opportunity counts and program customer counts)
- PCA finds orthogonal directions of maximum variance

### Implementation
```python
pca = PCA(n_components=13)
X_pca = pca.fit_transform(X)
```

### Results — Variance Explained

| Component | Individual | Cumulative |
|-----------|-----------|------------|
| PC1 | 39.8% | 39.8% |
| PC2 | 17.1% | 56.9% |
| PC3 | 8.3% | 65.2% |
| PC4 | 6.1% | 71.3% |
| PC5 | 3.6% | 74.9% |
| PC6 | 3.3% | 78.2% |
| PC7 | 2.8% | 81.0% |
| PC8 | 2.1% | 83.1% |
| PC9 | 2.0% | 85.1% |
| PC10 | 1.7% | 86.8% |
| PC11 | 1.3% | 88.1% |
| PC12 | 1.2% | 89.3% |
| PC13 | 1.0% | 90.3% |

**13 components capture ~90% of the total variance** in 75 features.

### Component Interpretation (Top Loadings)

- **PC1 (40%)** — Overall partner scale: launched ARR, qualified opportunities, program-specific ARR
- **PC2 (17%)** — Program engagement depth: program-specific customer and opportunity counts
- **PC3 (8%)** — Program-specific performance: specific program ARR, approved funding
- **PC4 (6%)** — Funding & marketplace: approved cash, program opps, total offers
- **PC5 (4%)** — Maturity & channel: partner age, channel revenue, channel customers
- **PC6-13** — Capture retention patterns, geographic effects, certification/specialization flags

---

## Step 6 — K-Means Clustering

### Choosing k
- Used the **elbow method** (plotting inertia vs. k)
- Also validated with domain knowledge — 4 clusters produce interpretable, actionable segments

### Implementation
```python
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
```

### Cluster Distribution

| Cluster | Size | Percentage |
|---------|------|------------|
| 0 | ~2,353 | 95.5% |
| 1 | ~55 | 2.2% |
| 2 | ~12 | 0.5% |
| 3 | ~43 | 1.8% |

---

## Step 7 — Cluster Profiling & Interpretation

### Cluster 0: Basic Partners (95.5%)
- **Tier:** Dominated by Tier 1 (78%)
- **Type:** 64% Type A, 36% Type B
- **Programs:** High adoption in 2-3 programs (58-73%), moderate in others
- **Retention:** ~28% across channels
- **Geography:** Diverse distribution across regions
- **Insight:** The broad base. Moderate engagement, room for growth.

### Cluster 1: High Performers (2.2%)
- **Tier:** Tier 1 (58%) + Tier 3 (42%) — no lower tiers
- **Type:** Balanced — 55% Type B, 45% Type A
- **Programs:** High adoption across the board; near-universal in top programs
- **Retention:** 38-75% depending on channel
- **Geography:** Concentrated in top region (82%)
- **Insight:** Revenue leaders (2.5x above average). Blueprint for advancement.

### Cluster 2: Elite Partners (0.5%)
- **Tier:** Tier 1 (75%) + Tier 3 (25%)
- **Type:** Type B dominated (75%)
- **Programs:** Near-universal participation (83-100% across most programs)
- **Retention:** Highest in one program (92%), but lower in channel retention (25%)
- **Geography:** Almost entirely in top region (92%)
- **Insight:** Exceptional performance (8-10x above average). Revenue machines with retention gaps.

### Cluster 3: Program Specialists (1.8%)
- **Tier:** Tier 3 dominated (72%)
- **Type:** Almost exclusively Type A (98%)
- **Programs:** 100% in two key programs, 94% in specialization, 81% in another
- **Retention:** Highest across all metrics — 86-100%
- **Geography:** Globally diverse distribution
- **Insight:** Retention champions. Deep program engagement drives loyalty.

---

## Step 8 — Business Recommendations

### For Cluster 0 (Basic Partners)
1. Improve customer retention rates (currently ~28%)
2. Encourage adoption of underutilized programs
3. Develop pathways to move partners toward higher tiers
4. Create targeted strategies for underrepresented regions

### For Cluster 1 (High Performers)
1. Use as blueprint for Cluster 0 advancement
2. Focus on improving retention in weaker channels
3. Expand geographic presence beyond primary region
4. Create pathway to Elite (Cluster 2) status

### For Cluster 2 (Elite Partners)
1. Address retention rate disparity (high in one program, low in channel)
2. Develop elite partners in other regions
3. Document and share best practices
4. Balance program participation with retention

### For Cluster 3 (Program Specialists)
1. Use as retention best-practice model
2. Expand successful model to other regions
3. Document partner success model for this type
4. Leverage global diversity for expansion strategies

---

## Tools & Libraries

| Tool | Purpose |
|------|---------|
| pandas | Data manipulation and aggregation |
| numpy | Numerical operations |
| scikit-learn | StandardScaler, PCA, KMeans |
| matplotlib | Scree plots, cluster visualizations |
| seaborn | Heatmaps, styled plots |

---

## Reproducibility

To reproduce this analysis with synthetic data:

```bash
pip install -r requirements.txt
python generate_synthetic_data.py
# Then run the analysis steps above in a Jupyter notebook or script
```

The synthetic data generator produces a dataset with matching schema and similar statistical distributions, without exposing any proprietary information.
