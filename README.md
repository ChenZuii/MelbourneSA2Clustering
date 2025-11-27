# Melbourne-SA2-Clustering
This repository provides a complete, end-to-end analytical pipeline for examining the socio-economic landscape of Melbourne’s Statistical Area Level 2 (SA2) regions using the Australian Bureau of Statistics (ABS) Socio-Economic Indexes for Areas (SEIFA 2021) dataset.

The project applies K-means clustering, implemented both manually (no scikit-learn dependency) and paired with structured exploratory data analysis (EDA), data preparation routines, feature transformation, elbow-method validation, and a suite of visualisation tools.

This script is designed for researchers, urban planners, data scientists, and students interested in urban analytics, neighbourhood typology classification, and data-driven planning approaches.

Key Features
1. Automated Data Exploration
- The script performs comprehensive exploratory analysis:
- Dataframe structure inspection
- Summary statistics (numeric + categorical)
- Missing value reports
- Duplicate row checks
- Categorical distribution analysis
- Structural data quality checks

2. Data Cleaning & SA2 Filtering
- The pipeline transforms the national SEIFA dataset into a curated Victorian-only subset:
- Remove duplicate rows
- Convert SA2 codes to zero-padded 9-digit strings
- Filter for Victorian SA2 regions (SA2 code starts with "2")
- Remove invalid or zero-population areas
- Compute population density
- Save a cleaned file to disk for downstream reuse

3. Distribution Plots (Australia & Victoria)
- The script automatically generates:
- Population distribution histograms
- IRSAD score distribution
- IRSD score distribution
- IRSAD vs IRSD scatter relationships
- Population density for Victoria
- These visualisations help contextualise socio-economic variation across Australia and within Melbourne.

4. Feature Engineering
A structured clustering feature matrix is built using:
- Population density
- IRSAD score
- IRSD score
- IEO score
- IER score

5. Manual Z-Score Standardisation
To make clustering reproducible and transparent, the script manually computes:
- Column means
- Column standard deviations
- Standardised features (z-scores)
- Handling of zero-variance columns

6. Custom Manual K-means Implementation
A full manual implementation of the K-means algorithm is included:
- Random centroid initialisation
- Iterative centroid update
- Convergence detection
- Inertia calculation
- Multiple initialisation runs (n_init)
- Best run selection
- This avoids reliance on scikit-learn and demonstrates full methodological transparency.

7. Elbow Method (manual computation)
- The script calculates inertia for k = 2 to k = 8 and plots a clean elbow curve to guide selection of the optimal number of clusters.

8. Cluster Assignment and Export
- Final cluster labels are:
- Attached to the SA2 metadata
- Exported as melbourne_sa2_clusters.csv
- Used for scatter plotting and feature diagnostics

9. Comprehensive Visualisation Suite
A full range of charts is automatically generated:
- Cluster scatterplots (e.g., density vs IRSAD)
- Cluster size bar charts
- Clustered feature boxplots
- Feature correlation heatmap
- These support interpretative analysis and presentation-ready graphics.

10. Data Requirements
You must download the ABS SEIFA 2021, SA2-level dataset:
Dataset:
Socio-Economic Indexes for Areas (SEIFA) 2021 – SA2
Source: Australian Bureau of Statistics
