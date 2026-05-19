# Customer Segmentation with K-Means & PCA

> ⚠️ **All data in this project is entirely synthetic and generated for learning/demonstration purposes only. No real customer data is used.**

An end-to-end unsupervised learning pipeline that segments retail customers into actionable clusters based on loyalty program participation, purchase behavior, and retention metrics.

## What It Does

- Generates synthetic customer data (2,500 customers, 55 features)
- Engineers and preprocesses numerical, categorical, and binary features
- Reduces dimensionality with PCA (74 features → 13 components)
- Clusters customers using K-Means (k=4)
- Profiles each cluster and produces business recommendations

## Run It

```bash
pip install numpy pandas scikit-learn
python customer_segmentation_analysis.py
```

Single file, no dependencies beyond numpy/pandas/sklearn. Generates its own data and prints the full analysis to stdout.

## Techniques

| Step | Method |
|------|--------|
| Encoding | One-hot (drop_first) |
| Scaling | StandardScaler |
| Dimensionality reduction | PCA |
| Clustering | K-Means |
| Validation | Elbow method |
