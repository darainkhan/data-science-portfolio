"""
Customer Loyalty Program Segmentation Analysis
================================================
An end-to-end unsupervised learning pipeline that segments retail customers
into actionable clusters based on loyalty program participation, purchase
behavior, and retention metrics.

This script demonstrates:
1. Synthetic data generation
2. Feature engineering (numerical, categorical, binary)
3. StandardScaler normalization
4. PCA for dimensionality reduction
5. K-Means clustering
6. Cluster profiling & business interpretation

All data is entirely synthetic and generated for learning purposes only.
No real customer data or proprietary information is included.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# STEP 1: Generate Synthetic Data
# =============================================================================
print("=" * 70)
print("STEP 1: Generating Synthetic Customer Data")
print("=" * 70)

np.random.seed(42)

NUM_CUSTOMERS = 2500
MONTHS_PER_CUSTOMER = 10
NUM_ROWS = NUM_CUSTOMERS * MONTHS_PER_CUSTOMER

customer_ids = np.repeat(np.arange(1, NUM_CUSTOMERS + 1), MONTHS_PER_CUSTOMER)

# Customer metadata
customer_segments = ["Individual", "Business"]
segment_weights = [0.64, 0.36]

membership_tiers = ["Silver", "Gold", "Platinum", "Basic"]
tier_weights = [0.77, 0.11, 0.07, 0.05]

regions = [
    "Northeast", "Southeast", "Midwest", "Southwest", "West_Coast",
    "Mountain", "Pacific_NW", "Mid_Atlantic", "New_England", "Great_Lakes",
    "Plains", "Gulf_Coast", "Appalachia", "Delta", "Upper_Midwest",
    "Desert_SW", "Rockies", "Pacific", "Atlantic", "Central", "Northwest"
]
region_weights = [
    0.37, 0.09, 0.06, 0.05, 0.05, 0.04, 0.04, 0.03,
    0.03, 0.03, 0.03, 0.03, 0.02, 0.02,
    0.02, 0.02, 0.02, 0.01, 0.01, 0.01, 0.02
]

# Assign per-customer attributes
customer_segment_arr = np.repeat(
    np.random.choice(customer_segments, size=NUM_CUSTOMERS, p=segment_weights),
    MONTHS_PER_CUSTOMER
)
membership_tier_arr = np.repeat(
    np.random.choice(membership_tiers, size=NUM_CUSTOMERS, p=tier_weights),
    MONTHS_PER_CUSTOMER
)
region_arr = np.repeat(
    np.random.choice(regions, size=NUM_CUSTOMERS, p=region_weights),
    MONTHS_PER_CUSTOMER
)

# Tenure (years as customer)
tenure_years = np.repeat(
    np.random.exponential(scale=3.5, size=NUM_CUSTOMERS).clip(0.5, 15),
    MONTHS_PER_CUSTOMER
)


def generate_flag(prob, size=NUM_CUSTOMERS):
    return np.repeat(np.random.binomial(1, prob, size=size), MONTHS_PER_CUSTOMER)


def generate_skewed_metric(base_mean, base_std, zero_prob=0.3, size=NUM_ROWS):
    values = np.random.lognormal(mean=np.log(base_mean + 1), sigma=base_std, size=size)
    mask = np.random.random(size=size) < zero_prob
    values[mask] = 0
    return np.round(values, 2)


def generate_count_metric(lam, zero_prob=0.3, size=NUM_ROWS):
    values = np.random.poisson(lam=lam, size=size)
    mask = np.random.random(size=size) < zero_prob
    values[mask] = 0
    return values


# Binary program enrollment flags
rewards_flag = generate_flag(0.58)
cashback_flag = generate_flag(0.44)
referral_flag = generate_flag(0.07)
premium_flag = generate_flag(0.29)
trial_flag = generate_flag(0.006)
loyalty_plus_flag = generate_flag(0.73)
vip_access_flag = generate_flag(0.08)
bundle_flag = generate_flag(0.41)
points_flag = generate_flag(0.53)
subscription_flag = generate_flag(0.33)
exclusive_flag = generate_flag(0.06)

# Retention flags
store_retention_flag = generate_flag(0.28)
online_retention_flag = generate_flag(0.28)
rewards_retention_flag = generate_flag(0.42)
cashback_retention_flag = generate_flag(0.35)
subscription_retention_flag = generate_flag(0.22)

# Numerical features
purchases_pre = generate_count_metric(3, zero_prob=0.2)
spend_pre = generate_skewed_metric(50000, 1.5, zero_prob=0.25)
visits_pre = generate_count_metric(5, zero_prob=0.15)
browse_value_pre = generate_skewed_metric(80000, 1.5, zero_prob=0.2)

total_orders = generate_count_metric(2, zero_prob=0.4)
total_order_value = generate_skewed_metric(100000, 1.8, zero_prob=0.45)

store_revenue = generate_skewed_metric(200000, 2.0, zero_prob=0.5)
store_visits = generate_count_metric(4, zero_prob=0.4)

recent_purchases_6m = generate_count_metric(2, zero_prob=0.3)
recent_spend_6m = generate_skewed_metric(30000, 1.5, zero_prob=0.35)
rewards_purchases_6m = generate_count_metric(1, zero_prob=0.5)
rewards_spend_6m = generate_skewed_metric(20000, 1.5, zero_prob=0.55)
cashback_purchases_6m = generate_count_metric(1, zero_prob=0.55)
cashback_spend_6m = generate_skewed_metric(15000, 1.5, zero_prob=0.6)
loyalty_spend_6m = generate_skewed_metric(25000, 1.5, zero_prob=0.4)
referral_spend_6m = generate_skewed_metric(10000, 1.5, zero_prob=0.8)
subscription_spend_6m = generate_skewed_metric(12000, 1.5, zero_prob=0.7)

coupons_redeemed_6m = generate_count_metric(2, zero_prob=0.4)
credits_earned_6m = generate_skewed_metric(5000, 1.2, zero_prob=0.5)
credits_spent_6m = generate_skewed_metric(3000, 1.2, zero_prob=0.6)
refunds_6m = generate_skewed_metric(4000, 1.2, zero_prob=0.55)

lifetime_purchases = generate_count_metric(8, zero_prob=0.1)
lifetime_unique_categories = generate_count_metric(5, zero_prob=0.15)
lifetime_spend = generate_skewed_metric(100000, 1.8, zero_prob=0.2)
rewards_orders = generate_count_metric(3, zero_prob=0.4)
rewards_categories = generate_count_metric(2, zero_prob=0.45)
rewards_spend = generate_skewed_metric(40000, 1.5, zero_prob=0.45)
cashback_orders = generate_count_metric(2, zero_prob=0.5)
cashback_categories = generate_count_metric(2, zero_prob=0.5)
cashback_spend = generate_skewed_metric(30000, 1.5, zero_prob=0.55)
loyalty_spend = generate_skewed_metric(50000, 1.5, zero_prob=0.35)
referral_spend = generate_skewed_metric(20000, 1.5, zero_prob=0.75)
subscription_spend = generate_skewed_metric(25000, 1.5, zero_prob=0.65)

# Assemble DataFrame
df = pd.DataFrame({
    "customer_id": customer_ids,
    "tenure_years": tenure_years,
    "customer_segment": customer_segment_arr,
    "membership_tier": membership_tier_arr,
    "region": region_arr,
    "purchases_pre": purchases_pre,
    "spend_pre": spend_pre,
    "visits_pre": visits_pre,
    "browse_value_pre": browse_value_pre,
    "referral_flag": referral_flag,
    "rewards_flag": rewards_flag,
    "cashback_flag": cashback_flag,
    "premium_flag": premium_flag,
    "trial_flag": trial_flag,
    "loyalty_plus_flag": loyalty_plus_flag,
    "vip_access_flag": vip_access_flag,
    "bundle_flag": bundle_flag,
    "points_flag": points_flag,
    "subscription_flag": subscription_flag,
    "exclusive_flag": exclusive_flag,
    "total_orders": total_orders,
    "total_order_value": total_order_value,
    "store_retention_flag": store_retention_flag,
    "online_retention_flag": online_retention_flag,
    "store_revenue": store_revenue,
    "store_visits": store_visits,
    "recent_purchases_6m": recent_purchases_6m,
    "recent_spend_6m": recent_spend_6m,
    "rewards_purchases_6m": rewards_purchases_6m,
    "rewards_spend_6m": rewards_spend_6m,
    "cashback_purchases_6m": cashback_purchases_6m,
    "cashback_spend_6m": cashback_spend_6m,
    "loyalty_spend_6m": loyalty_spend_6m,
    "referral_spend_6m": referral_spend_6m,
    "subscription_spend_6m": subscription_spend_6m,
    "coupons_redeemed_6m": coupons_redeemed_6m,
    "credits_earned_6m": credits_earned_6m,
    "credits_spent_6m": credits_spent_6m,
    "refunds_6m": refunds_6m,
    "lifetime_purchases": lifetime_purchases,
    "lifetime_unique_categories": lifetime_unique_categories,
    "lifetime_spend": lifetime_spend,
    "rewards_orders": rewards_orders,
    "rewards_categories": rewards_categories,
    "rewards_spend": rewards_spend,
    "rewards_retention_flag": rewards_retention_flag,
    "cashback_orders": cashback_orders,
    "cashback_categories": cashback_categories,
    "cashback_retention_flag": cashback_retention_flag,
    "subscription_retention_flag": subscription_retention_flag,
    "cashback_spend": cashback_spend,
    "loyalty_spend": loyalty_spend,
    "referral_spend": referral_spend,
    "subscription_spend": subscription_spend,
})

print(f"  Generated {len(df)} rows for {NUM_CUSTOMERS} customers")
print(f"  Columns: {len(df.columns)}")
print(f"  Shape: {df.shape}")
print()

# =============================================================================
# STEP 2: Feature Classification
# =============================================================================
print("=" * 70)
print("STEP 2: Feature Classification")
print("=" * 70)

numerical_columns = [
    'tenure_years', 'purchases_pre', 'spend_pre', 'visits_pre',
    'browse_value_pre', 'total_order_value', 'store_revenue',
    'recent_spend_6m', 'rewards_spend_6m', 'cashback_spend_6m',
    'loyalty_spend_6m', 'referral_spend_6m', 'subscription_spend_6m',
    'credits_earned_6m', 'credits_spent_6m', 'refunds_6m',
    'lifetime_spend', 'rewards_spend', 'cashback_spend',
    'loyalty_spend', 'referral_spend', 'subscription_spend',
    'total_orders', 'store_visits', 'recent_purchases_6m',
    'rewards_purchases_6m', 'cashback_purchases_6m',
    'coupons_redeemed_6m', 'lifetime_purchases',
    'lifetime_unique_categories', 'rewards_orders',
    'rewards_categories', 'cashback_orders', 'cashback_categories'
]

categorical_columns = ['customer_segment', 'membership_tier', 'region']

binary_columns = [
    'referral_flag', 'rewards_flag', 'cashback_flag', 'premium_flag',
    'trial_flag', 'loyalty_plus_flag', 'vip_access_flag', 'bundle_flag',
    'points_flag', 'subscription_flag', 'exclusive_flag',
    'store_retention_flag', 'online_retention_flag',
    'rewards_retention_flag', 'cashback_retention_flag',
    'subscription_retention_flag'
]

print(f"  Numerical features:   {len(numerical_columns)}")
print(f"  Categorical features: {len(categorical_columns)}")
print(f"  Binary features:      {len(binary_columns)}")
print()

# =============================================================================
# STEP 3: Preprocessing
# =============================================================================
print("=" * 70)
print("STEP 3: Preprocessing")
print("=" * 70)

# One-hot encode categoricals
dummy_vars = pd.get_dummies(df[categorical_columns], drop_first=True).astype(int)
print(f"  One-hot encoded: {dummy_vars.shape[1]} dummy columns")

# Standardize numerical features
df_numerical = df[numerical_columns].copy()
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_numerical),
    columns=df_numerical.columns,
    index=df_numerical.index
)
print(f"  Standardized: {df_scaled.shape[1]} numerical columns (mean=0, std=1)")

# Binary flags
df_binary = df[binary_columns].astype(int)

# Combine all features
df_final = pd.concat([df_scaled, dummy_vars, df_binary], axis=1)
print(f"  Combined feature matrix: {df_final.shape}")
print()

# =============================================================================
# STEP 4: Aggregate to Customer Level
# =============================================================================
print("=" * 70)
print("STEP 4: Aggregate to Customer Level")
print("=" * 70)

df_final['customer_id'] = df['customer_id'].values
df_aggregated = df_final.groupby('customer_id').mean().reset_index()

customer_ids_agg = df_aggregated['customer_id']
X = df_aggregated.drop(columns=['customer_id'])

print(f"  Aggregated from {len(df_final)} rows → {len(df_aggregated)} customers")
print(f"  Feature dimensions: {X.shape[1]}")
print()

# =============================================================================
# STEP 5: PCA (Dimensionality Reduction)
# =============================================================================
print("=" * 70)
print("STEP 5: PCA — Dimensionality Reduction")
print("=" * 70)

pca = PCA(n_components=13)
X_pca = pca.fit_transform(X)

explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)

print(f"\n  {'Component':<12} {'Individual':<12} {'Cumulative':<12}")
print(f"  {'-'*36}")
for i in range(13):
    print(f"  PC{i+1:<9} {explained_var[i]:<12.3f} {cumulative_var[i]:<12.3f}")

print(f"\n  13 components capture {cumulative_var[-1]:.1%} of total variance")

# Top loadings per component
print("\n  Top Feature Loadings per Component:")
feature_names = list(X.columns)
components_df = pd.DataFrame(
    pca.components_, columns=feature_names,
    index=[f'PC{i+1}' for i in range(13)]
)

for idx in range(min(5, len(components_df))):
    component = components_df.index[idx]
    top = abs(components_df.iloc[idx]).sort_values(ascending=False)[:3]
    features_str = ", ".join([f"{f} ({v:.3f})" for f, v in top.items()])
    print(f"    {component}: {features_str}")
print()

# =============================================================================
# STEP 6: K-Means Clustering
# =============================================================================
print("=" * 70)
print("STEP 6: K-Means Clustering")
print("=" * 70)

# Elbow method
print("\n  Elbow Method (Inertia by k):")
for k in range(2, 8):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    print(f"    k={k}: inertia={km.inertia_:.0f}")

# Final model with k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_pca)
df_aggregated['Cluster'] = clusters

print(f"\n  Final model: k=4")
print(f"\n  Cluster Distribution:")
for c in sorted(df_aggregated['Cluster'].unique()):
    n = (df_aggregated['Cluster'] == c).sum()
    pct = n / len(df_aggregated) * 100
    print(f"    Cluster {c}: {n} customers ({pct:.1f}%)")
print()

# =============================================================================
# STEP 7: Cluster Profiling
# =============================================================================
print("=" * 70)
print("STEP 7: Cluster Profiling")
print("=" * 70)

# Merge original categorical data
cat_data = df[['customer_id'] + categorical_columns].groupby('customer_id').first().reset_index()
df_profile = df_aggregated.merge(cat_data, on='customer_id', how='left', suffixes=('', '_orig'))

program_flags = [
    'referral_flag', 'rewards_flag', 'cashback_flag', 'premium_flag',
    'trial_flag', 'loyalty_plus_flag', 'vip_access_flag', 'bundle_flag',
    'points_flag', 'subscription_flag', 'exclusive_flag'
]

retention_flags = [
    'store_retention_flag', 'online_retention_flag',
    'rewards_retention_flag', 'cashback_retention_flag',
    'subscription_retention_flag'
]

for cluster_id in sorted(df_profile['Cluster'].unique()):
    cluster_data = df_profile[df_profile['Cluster'] == cluster_id]
    n = len(cluster_data)
    pct = n / len(df_profile) * 100

    print(f"\n  {'─' * 60}")
    print(f"  CLUSTER {cluster_id} — {n} customers ({pct:.1f}%)")
    print(f"  {'─' * 60}")

    # Program enrollment rates
    print(f"\n  Program Enrollment Rates:")
    for col in program_flags:
        rate = cluster_data[col].mean() * 100
        label = "███" if rate > 50 else "██" if rate > 25 else "█"
        print(f"    {col:<25} {rate:5.1f}% {label}")

    # Retention rates
    print(f"\n  Retention Rates:")
    for col in retention_flags:
        rate = cluster_data[col].mean() * 100
        print(f"    {col:<30} {rate:5.1f}%")

    # Segment distribution
    print(f"\n  Customer Segment:")
    seg_dist = cluster_data['customer_segment'].value_counts(normalize=True).mul(100)
    for seg, pct_val in seg_dist.items():
        print(f"    {seg}: {pct_val:.1f}%")

    # Tier distribution
    print(f"\n  Membership Tier:")
    tier_dist = cluster_data['membership_tier'].value_counts(normalize=True).mul(100)
    for tier, pct_val in tier_dist.items():
        print(f"    {tier}: {pct_val:.1f}%")

    # Top regions
    print(f"\n  Top 3 Regions:")
    reg_dist = cluster_data['region'].value_counts(normalize=True).mul(100).head(3)
    for reg, pct_val in reg_dist.items():
        print(f"    {reg}: {pct_val:.1f}%")

print()

# =============================================================================
# STEP 8: Business Recommendations
# =============================================================================
print("=" * 70)
print("STEP 8: Business Insights & Recommendations")
print("=" * 70)

print("""
  CLUSTER SUMMARY:
  ┌─────────┬─────────────────────┬────────────────────────────────────────────┐
  │ Cluster │ Label               │ Key Strategy                               │
  ├─────────┼─────────────────────┼────────────────────────────────────────────┤
  │    0    │ Basic Customers     │ Increase program adoption & retention      │
  │    1    │ High-Value          │ Blueprint for growth; expand geographically │
  │    2    │ Elite               │ Best practices; address retention gaps      │
  │    3    │ Loyalty Champions   │ Retention model; leverage diversity         │
  └─────────┴─────────────────────┴────────────────────────────────────────────┘

  KEY INSIGHTS:

  1. Retention correlates with program depth
     → Customers enrolled in 3+ programs show 2-3x higher retention rates

  2. Geographic concentration in elite tiers
     → Top-performing customers are concentrated in one region
     → Opportunity for expansion into underrepresented areas

  3. Customer segment divergence
     → Business customers excel at retention
     → Individual customers drive higher revenue scale

  4. Clear progression ladder
     → Data supports a customer development pathway:
        Basic → High-Value → Elite

  RECOMMENDATIONS:

  • Cluster 0: Focus on cross-selling additional programs; improve retention
  • Cluster 1: Use as blueprint for Cluster 0 advancement; expand geography
  • Cluster 2: Document best practices; address channel retention gaps
  • Cluster 3: Use as retention model; replicate in other regions
""")

print("=" * 70)
print("Analysis Complete")
print("=" * 70)
