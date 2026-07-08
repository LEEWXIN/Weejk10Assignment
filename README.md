# From Clusters to Product: Customer Segmentation on UCI Online Retail

BDS23114 Data Analytics — Week 10 Assignment

## 1. Dataset

- **Source:** Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning
  Repository. https://doi.org/10.24432/C5BW33
  Official page: https://archive.ics.uci.edu/dataset/352/online+retail
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0) —
  sharing and adaptation permitted with attribution.
- **Raw size:** 541,909 transaction line items, 8 columns, UK-based online
  gift retailer, Dec 2010 – Dec 2011.
- Place the raw file at `data/Online_Retail.xlsx` before running anything
  (not included in this repo due to file size — download from the link above).

**Problem framing:** the raw file is one row per *transaction line*, not one
row per *customer*. The clustering question is: what natural customer
segments exist in the transaction history, and what should be done
differently for each one? This required aggregating transactions into a
customer-level feature table (RFM + behavioural features) before any
clustering — 4,338 customers, 8 features, above the assignment's
200-observation / 3-feature minimum.

## 2. Method

| Step | Choice | Why |
|---|---|---|
| Feature engineering | Recency, Frequency, Monetary, AvgUnitPrice, DistinctProducts, TotalItems, AvgBasketValue, ReturnRate | Captures spend, loyalty, product breadth and return behaviour, not just plain RFM |
| Scaling | `log1p` on the 6 heavily right-skewed features (Frequency, Monetary, AvgUnitPrice, DistinctProducts, TotalItems, AvgBasketValue), then `RobustScaler` | Raw skewness ranged from ~6 to ~50; after log-transform it dropped to 0.04–3.3. RobustScaler (median/IQR) limits the influence of the remaining large-spender outliers |
| Clustering | **HDBSCAN** (density-based) | EDA showed uneven customer density (a dense mass of casual buyers, a sparse spread of high-frequency buyers) and heavy-tailed spend — this rules out K-Means, which assumes roughly equal-sized spherical clusters |
| Dimensionality reduction (visualization only) | **UMAP** | PCA is excluded by the assignment brief as the primary technique; UMAP is used only to plot the 2D cluster map, never to drive cluster assignment |
| Hyperparameter tuning | Grid search over `min_cluster_size` and `min_samples` (two passes — a coarse grid, then a finer grid around the best region) | Selected `min_cluster_size=25, min_samples=5`, the best-silhouette combination subject to a noise ratio ≤ 35%. A finer search confirmed this dataset naturally supports only **2** well-separated clusters — pushing for 3+ clusters always pushed noise above 55%, so the model was not forced into an artificial extra split |
| Evaluation | Silhouette Score, Davies–Bouldin Index, Calinski–Harabasz Index, computed only on non-noise points | Reports at least two internal validation metrics as required |

## 3. Results

Final model: `HDBSCAN(min_cluster_size=25, min_samples=5, metric="euclidean")`

| Metric | Value |
|---|---|
| Silhouette Score | 0.311 |
| Davies–Bouldin Index | 0.832 |
| Calinski–Harabasz Index | 80.0 |
| Number of clusters | 2 |
| Noise points | 1,486 (34.3%) |

**Cluster profiles** (medians):

| Cluster | Name | Recency | Frequency | Monetary | Return rate | Size |
|---|---|---|---|---|---|---|
| 0 | At-risk, high-return sleeper customers | 247 days | 1 | £326 | 50% | 30 |
| 1 | Core active customers | 40 days | 2 | £660 | 0% | 2,822 |
| -1 (noise) | Heterogeneous / edge customers | 81 days | 2 | £721 | 9.7% | 1,486 |

**Business read:** Cluster 1 is the stable core of the customer base —
recent, low-return, repeat buyers, the group worth protecting with loyalty
incentives. Cluster 0 is tiny (30 customers) but high-signal: a 50% return
rate on a single order 8 months ago is a clear churn/product-fit warning
worth investigating manually. The noise group (34%) is not bad data — their
metrics sit in the moderate range across the board, meaning they genuinely
don't match either archetype tightly; a real deployment would need a
secondary or manual-review process for this group rather than automated
targeting.

A UMAP projection of the customer base (`outputs/figures` or the notebook
output) shows two macro-level "islands" that HDBSCAN's cluster assignment
does not cleanly align with — noted as a limitation and a direction for
future work (e.g. sub-clustering within each UMAP island) rather than
smoothed over.

## 4. Project structure

```
data/Online_Retail.xlsx        # raw dataset (download separately, see Section 1)
<your_notebook>.ipynb          # EDA, preprocessing, HDBSCAN tuning, evaluation
app.py                         # Streamlit web app (loads the saved pipeline, live inference)
outputs/
  cleaned_transactions.csv
  customer_features.csv
  customer_features_clustered.csv
  model/clustering_pipeline.joblib   # scaler + fitted HDBSCAN + UMAP reducer
requirements.txt
```

## 5. How to run

```bash
pip install pandas numpy openpyxl scikit-learn hdbscan umap-learn matplotlib seaborn joblib streamlit

# 1. Open the notebook and run all cells top to bottom.
#    This cleans the data, builds the RFM feature table, tunes and fits
#    HDBSCAN, evaluates it, and saves outputs/model/clustering_pipeline.joblib

# 2. Launch the web app (loads the saved pipeline, does not retrain)
python -m streamlit run app.py
```

The app opens at `http://localhost:8501`. The left panel shows the existing
customer base in UMAP space, colored by cluster, with an expandable table of
cluster profiles. The right panel lets you enter a new/hypothetical
customer's RFM + behavioural values; on submit it log-transforms and scales
the input with the *saved* preprocessing objects, calls
`hdbscan.approximate_predict()` for a live cluster assignment with a
membership-strength confidence score, and plots the new point on the
existing UMAP map.

## 6. Limitations

- HDBSCAN's noise category covers 34.3% of customers — documented above as
  a modeling and business finding, not hidden. These customers would need a
  secondary or manual-review process in a real deployment.
- The dataset naturally supports only 2 clusters at a noise level ≤ 35%; a
  finer hyperparameter grid confirmed this rather than assuming it.
- Sample covers a single UK retailer over ~1 year; segments may not
  generalize to other markets or seasons without re-fitting.

## 7. Academic integrity note

Dataset selection, algorithm choice (HDBSCAN over the excluded
K-Means/GMM/PCA family), the noise-ratio tuning trade-off, and cluster
interpretation were reasoned through interactively and verified against the
actual notebook output at each step; code was AI-assisted (Claude) and
should be understood, not just submitted, before final submission.
