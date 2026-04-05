# Experiment 11 — Clustering Rental Vehicles by Usage Behavior

This repository contains **Experiment 11** for a Data Mining Lab, which focuses on segmenting a fleet of rental vehicles based on customer usage behavior to derive actionable business insights.

## Objective

A synthetic dataset of 400 rental vehicles is analyzed to understand how they are being used by customers. The goal is to group vehicles with similar usage patterns—such as rental frequency, trip duration, distance traveled, and maintenance needs—to make informed decisions regarding:
- Maintenance scheduling
- Dynamic pricing
- Fleet expansion and management

## Dataset Features

The dataset is synthetically generated and consists of 400 vehicles with the following behavioral and derived features used for clustering:
- `total_rentals`
- `avg_trip_duration_hrs`
- `avg_distance_km`
- `utilization_rate`
- `revenue_per_km` *(Clamped between $0.10 and $5.00 to prevent mathematical outlier ghost clusters)*
- `log_maintenance` *(Log-transformed count of maintenance incidents)*
- `log_damage` *(Log-transformed damage count)*
- `avg_days_between_rentals`

Additionally, profiling-only categorical features (`vehicle_type`, `customer_type_mode`) are included for post-clustering analysis but are omitted from the K-Means algorithm itself.

## Workflow & Notebook Structure

The analysis is performed in standard Python Data Science environments leveraging `pandas`, `NumPy`, `matplotlib`, `seaborn`, `Scikit-Learn`, and `SciPy`. The primary steps in the Jupyter Notebook include:

1. **Section 1: Problem Statement** — Outline the goals of clustering the vehicle fleet.
2. **Section 2: Imports & Data Generation** — Generation of the 400 realistic vehicle fleet records, deriving specific ratios like revenue per distance, and cleanly separating clustering features from profiling metadata.
3. **Section 3: Exploratory Data Analysis (EDA)** — Visualizing distributions (histograms), checking for skewed variables, finding correlations, and dealing with extreme values.
4. **Section 4: Data Preprocessing** — Standardizing data so feature variances correspond correctly within Euclidean space algorithms and performing PCA.
5. **Section 5: K-Means Clustering implementation** — Testing algorithms, tuning for the optimal number of clusters (`K`), and avoiding ghost clusters from extreme outliers.
6. **Section 6: Cluster Validation** — Evaluating results through Silhouette Scores, Davies-Bouldin Scores, Calinski-Harabasz Scores, and checking alignment via Hierarchical Clustering (Dendrograms).
7. **Section 7: Recommendations and Business Profiling** — Turning pure mathematical segments into actionable recommendations using group averages and the metadata columns.

## Recent Fixes
- `revenue_per_km` calculation was refined and explicitly clamped by incorporating realistic minimums and margins, fixing a known edge condition where extreme derived outliers forced K-Means to produce empty or tiny ghost clusters.

## Running the Notebook
Ensure you have the required Python libraries installed:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```
Start Jupyter and run all cells sequentially:
```bash
jupyter notebook Experiment_11_Vehicle_Clustering.ipynb
```
