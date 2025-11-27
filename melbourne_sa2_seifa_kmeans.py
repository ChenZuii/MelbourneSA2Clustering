from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Section 1.1 - Data Description and Exploration
# ---------------------------------------------------------------------


def describe_dataframe(df: pd.DataFrame) -> None:
    """Print descriptive information about a DataFrame.

    This function prints structural and summary information including:
    - dataset shape
    - first / last rows
    - column names
    - data types
    - info summary
    - statistical summary

    Args:
        df: The DataFrame to describe.

    Returns:
        None
    """
    pd.set_option("display.max_columns", None)

    print("\n--- DATAFRAME SHAPE (ROWS, COLUMNS) ---")
    print(df.shape)

    print("\n--- FIRST FIVE ROWS ---")
    print(df.head())

    print("\n--- LAST FIVE ROWS ---")
    print(df.tail())

    print("\n--- COLUMN NAMES ---")
    print(df.columns)

    print("\n--- DATA TYPES ---")
    print(df.dtypes)

    print("\n--- DATAFRAME INFO ---")
    df.info()

    print("\n--- SUMMARY STATISTICS (INCLUDE ALL TYPES) ---")
    print(df.describe(include="all"))


def find_missing_values(df: pd.DataFrame) -> None:
    """Print counts of missing values and show rows that contain any.

    Args:
        df: The DataFrame to inspect.

    Returns:
        None
    """
    print("\n--- MISSING VALUES PER COLUMN ---")
    print(df.isna().sum())

    print("\n--- ROWS CONTAINING ANY MISSING VALUES ---")
    missing_rows = df[df.isna().any(axis=1)]
    print(missing_rows)


def find_duplicated_rows(df: pd.DataFrame) -> None:
    """Print the number of duplicated rows in the DataFrame.

    Args:
        df: The DataFrame to inspect.

    Returns:
        None
    """
    print("\n--- DUPLICATED ROWS ---")
    duplicate_count = df.duplicated().sum()
    print(f"Number of duplicated rows: {duplicate_count}")


def explore_categorical_data(df: pd.DataFrame) -> None:
    """Explore selected categorical or discrete columns.

    Focuses on:
    - SA2 name
    - SEIFA deciles (IRSAD, IRSD, IEO, IER)

    Args:
        df: The DataFrame that contains the columns.

    Returns:
        None
    """
    categorical_columns = [
        "Statistical Areas Level 2 2021 name",
        "IRSAD Rank within Australia - Decile",
        "IRSD Rank within Australia - Decile",
        "IEO Rank within Australia - Decile",
        "IER Rank within Australia - Decile",
    ]

    for column in categorical_columns:
        if column in df.columns:
            print(f"\nDistinct values in '{column}':")
            print(df[column].unique())
            print(f"\nValue counts in '{column}':")
            print(df[column].value_counts())
        else:
            print(f"\nColumn '{column}' not found in DataFrame.")


def identify_possible_quality_issues(df: pd.DataFrame) -> None:
    """Print indicators of potential structural data quality issues.

    Checks include:
    - Columns with only one unique value.
    - Placeholder-like values:
      * Zero area (Area in square kilometres).
      * Zero population (Usual Resident Population).

    Args:
        df: The DataFrame containing the data.

    Returns:
        None
    """
    print("\n--- POSSIBLE DATA QUALITY ISSUES ---")

    placeholder_rules = {
        "Area in square kilometres": [0.0],
        "Usual Resident Population": [0],
    }

    # Columns with only one unique value.
    for column in df.columns:
        if df[column].nunique(dropna=False) == 1:
            print(f"Column '{column}' has only one unique value.")

    # Column-specific placeholder patterns.
    for column, placeholders in placeholder_rules.items():
        if column in df.columns:
            col_series = df[column]
            for value in placeholders:
                if (col_series == value).any():
                    print(
                        f"Column '{column}' contains the placeholder-like "
                        f"value '{value}'."
                    )


def plot_population_distribution(df: pd.DataFrame, title_suffix: str) -> None:
    """Plot a histogram of usual resident population.

    Args:
        df: The DataFrame containing the data.
        title_suffix: Suffix describing the spatial extent
            (e.g. 'Australia, all SA2s' or 'Victoria, SA2s').

    Returns:
        None
    """
    if "Usual Resident Population" not in df.columns:
        print("Column 'Usual Resident Population' not found. Skipping plot.")
        return

    print("\n--- PLOTTING POPULATION DISTRIBUTION ---")
    plt.figure()
    df["Usual Resident Population"].plot(
        kind="hist",
        bins=30,
        title=f"Distribution of Usual Resident Population ({title_suffix})",
    )
    plt.xlabel("Usual Resident Population")
    plt.ylabel("Number of SA2s")
    plt.tight_layout()
    plt.show()


def plot_irsad_score_distribution(df: pd.DataFrame, title_suffix: str) -> None:
    """Plot a histogram of IRSAD scores.

    Args:
        df: The DataFrame containing the data.
        title_suffix: Suffix describing the spatial extent.

    Returns:
        None
    """
    if "IRSAD Score" not in df.columns:
        print("Column 'IRSAD Score' not found. Skipping plot.")
        return

    print("\n--- PLOTTING IRSAD SCORE DISTRIBUTION ---")
    plt.figure()
    df["IRSAD Score"].plot(
        kind="hist",
        bins=30,
        title=f"Distribution of IRSAD Scores ({title_suffix})",
    )
    plt.xlabel("IRSAD Score")
    plt.ylabel("Number of SA2s")
    plt.tight_layout()
    plt.show()


def plot_irsd_score_distribution(df: pd.DataFrame, title_suffix: str) -> None:
    """Plot a histogram of IRSD scores.

    Args:
        df: The DataFrame containing the data.
        title_suffix: Suffix describing the spatial extent.

    Returns:
        None
    """
    if "IRSD Score" not in df.columns:
        print("Column 'IRSD Score' not found. Skipping plot.")
        return

    print("\n--- PLOTTING IRSD SCORE DISTRIBUTION ---")
    plt.figure()
    df["IRSD Score"].plot(
        kind="hist",
        bins=30,
        title=f"Distribution of IRSD Scores ({title_suffix})",
    )
    plt.xlabel("IRSD Score")
    plt.ylabel("Number of SA2s")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Section 1.2 - Data Preparation (Melbourne / Victoria SA2s)
# ---------------------------------------------------------------------


def prepare_sa2_data_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare SA2-level SEIFA data for clustering analysis.

    This function:
    - Removes duplicated rows.
    - Resets the index.
    - Cleans and converts SA2 code to a zero-padded string.
    - Filters to Victorian SA2s using the first digit of the code (state=2).
    - Removes SA2s with zero or missing population or area.
    - Computes population density (persons per square kilometre).
    - Selects key socio-economic indicators and returns a tidy DataFrame.

    Args:
        df: Raw SA2 SEIFA dataset (national).

    Returns:
        Prepared dataset of Victorian SA2s with features
        needed for clustering.
    """
    df_prepared = df.copy()

    # Remove duplicated rows and reset index.
    df_prepared = df_prepared.drop_duplicates()
    df_prepared = df_prepared.reset_index(drop=True)

    code_col = "Statistical Areas Level 2 2021 code"
    if code_col not in df_prepared.columns:
        raise KeyError(
            "Expected column "
            "'Statistical Areas Level 2 2021 code' not found."
        )

    # Coerce SA2 code to numeric, dropping non-numeric placeholders.
    raw_codes = df_prepared[code_col]
    numeric_codes = pd.to_numeric(raw_codes, errors="coerce")

    # Remove rows with invalid / non-numeric codes.
    df_prepared = df_prepared[~numeric_codes.isna()].copy()
    numeric_codes = numeric_codes[~numeric_codes.isna()]

    # Convert to nullable integer, then to 9-digit zero-padded string.
    df_prepared[code_col] = (
        numeric_codes.astype("Int64")
        .astype(str)
        .str.zfill(9)
    )

    # Filter to Victorian SA2s: state code "2".
    df_prepared = df_prepared[df_prepared[code_col].str.startswith("2")].copy()

    pop_col = "Usual Resident Population"
    area_col = "Area in square kilometres"

    for required_col in [pop_col, area_col]:
        if required_col not in df_prepared.columns:
            raise KeyError(
                f"Expected column '{required_col}' not found "
                "in SA2 SEIFA dataset."
            )

    # Drop rows with missing or zero population/area.
    df_prepared = df_prepared.dropna(subset=[pop_col, area_col])
    df_prepared = df_prepared[
        (df_prepared[pop_col] > 0) & (df_prepared[area_col] > 0)
    ].copy()

    # Compute population density.
    df_prepared["Population density (persons per sq km)"] = (
        df_prepared[pop_col] / df_prepared[area_col]
    )

    columns_to_keep = [
        code_col,
        "Statistical Areas Level 2 2021 name",
        pop_col,
        area_col,
        "Population density (persons per sq km)",
        "IRSAD Score",
        "IRSD Score",
        "IEO Score",
        "IER Score",
    ]

    missing_columns = [col for col in columns_to_keep if col not in df_prepared.columns]
    if missing_columns:
        raise KeyError(
            "The following expected columns are missing from the dataset: "
            f"{missing_columns}"
        )

    df_prepared = df_prepared[columns_to_keep].copy()

    print("\n=== PREPARED SA2 DATASET FOR CLUSTERING (VICTORIA) ===")
    print(df_prepared.head())

    output_dir = Path(r"C:\Users\bobby\Desktop\SA2")
    output_dir.mkdir(parents=True, exist_ok=True)
    prepared_file_csv = output_dir.joinpath(
        "melbourne_sa2_prepared_for_clustering.csv"
    )
    df_prepared.to_csv(prepared_file_csv, index=False)
    print(f"Prepared dataset saved to: {prepared_file_csv}")

    return df_prepared


def plot_density_distribution(df: pd.DataFrame) -> None:
    """Plot a histogram of population density for prepared SA2s.

    Args:
        df: Prepared SA2 dataset with density column.

    Returns:
        None
    """
    density_col = "Population density (persons per sq km)"
    if density_col not in df.columns:
        print(f"Column '{density_col}' not found. Skipping plot.")
        return

    print("\n--- PLOTTING POPULATION DENSITY DISTRIBUTION (VIC SA2s) ---")
    plt.figure()
    df[density_col].plot(
        kind="hist",
        bins=30,
        title="Population Density Distribution (Victorian SA2s)",
    )
    plt.xlabel("Population density (persons per sq km)")
    plt.ylabel("Number of SA2s")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# Section 2.1 - Feature Matrix and Standardisation (manual)
# ---------------------------------------------------------------------


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Construct the feature matrix used for K-means clustering.

    Uses the following indicators:
    - Population density (persons per sq km).
    - IRSAD Score (relative advantage/disadvantage).
    - IRSD Score (relative disadvantage).
    - IEO Score (education and occupation).
    - IER Score (economic resources).

    Args:
        df: Prepared SA2 dataset.

    Returns:
        A tuple of:
        - feature_df: Feature matrix (numeric columns only).
        - df_aligned: Subset of df aligned to feature_df index for
          metadata (codes, names, etc.).
    """
    feature_columns = [
        "Population density (persons per sq km)",
        "IRSAD Score",
        "IRSD Score",
        "IEO Score",
        "IER Score",
    ]

    for col in feature_columns:
        if col not in df.columns:
            raise KeyError(
                f"Expected feature column '{col}' not found "
                "in the prepared SA2 dataset."
            )

    feature_df = df[feature_columns].copy()
    feature_df = feature_df.dropna()
    df_aligned = df.loc[feature_df.index].copy()

    print("\n=== FEATURE MATRIX FOR CLUSTERING ===")
    print(feature_df.head())

    return feature_df, df_aligned


def standardize_features(
    feature_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Standardize feature matrix using manual z-score transformation.

    For each column:
    - Subtract the column mean.
    - Divide by the column standard deviation (population std, ddof=0).

    Columns with zero standard deviation are left unscaled (std set to 1).

    Args:
        feature_df: Feature matrix for clustering.

    Returns:
        A tuple of:
        - X_scaled: Standardised features.
        - means: Feature means.
        - stds: Feature standard deviations (zeros replaced by 1.0).
    """
    means = feature_df.mean(axis=0)
    stds = feature_df.std(axis=0, ddof=0)
    stds_replaced = stds.replace(0, 1.0)

    X_scaled_df = (feature_df - means) / stds_replaced

    print("\n=== STANDARDISED FEATURE MATRIX (HEAD) ===")
    print(X_scaled_df.head())

    return X_scaled_df, means, stds_replaced


# ---------------------------------------------------------------------
# Section 2.2 - Manual K-means Clustering and Elbow Method
# ---------------------------------------------------------------------


def _initialize_centroids(
    X_array: np.ndarray, n_clusters: int, rng: np.random.Generator
) -> np.ndarray:
    """Initialise centroids by sampling points from the dataset."""
    n_samples = X_array.shape[0]
    if n_clusters > n_samples:
        raise ValueError("Number of clusters cannot exceed number of samples.")

    indices = rng.choice(n_samples, size=n_clusters, replace=False)
    centroids = X_array[indices, :].copy()
    return centroids


def _assign_clusters(X_array: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each sample to the nearest centroid."""
    diffs = X_array[:, np.newaxis, :] - centroids[np.newaxis, :, :]
    distances_sq = np.sum(diffs**2, axis=2)
    labels = np.argmin(distances_sq, axis=1)
    return labels


def _update_centroids(
    X_array: np.ndarray,
    labels: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Update centroids based on current assignments."""
    n_features = X_array.shape[1]
    centroids = np.zeros((n_clusters, n_features), dtype=float)

    for k in range(n_clusters):
        members = X_array[labels == k]
        if members.size == 0:
            random_index = rng.integers(0, X_array.shape[0])
            centroids[k] = X_array[random_index]
        else:
            centroids[k] = members.mean(axis=0)

    return centroids


def _compute_inertia(
    X_array: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
) -> float:
    """Compute K-means inertia (sum of squared distances to centroids)."""
    diffs = X_array - centroids[labels]
    inertia = float(np.sum(diffs**2))
    return inertia


def manual_kmeans_fit(
    X_scaled: pd.DataFrame,
    n_clusters: int,
    n_init: int = 10,
    max_iter: int = 300,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit a K-means model using a manual implementation.

    Performs multiple random initialisations and retains the solution
    with the lowest inertia.
    """
    X_array = X_scaled.to_numpy(dtype=float)
    n_samples = X_array.shape[0]

    if n_clusters <= 0:
        raise ValueError("n_clusters must be a positive integer.")
    if n_clusters > n_samples:
        raise ValueError("Number of clusters cannot exceed number of samples.")

    best_inertia = np.inf
    best_centroids = None
    best_labels = None

    print(
        f"\n=== MANUAL K-MEANS FIT "
        f"(k = {n_clusters}, n_init = {n_init}) ==="
    )

    for init_index in range(n_init):
        rng = np.random.default_rng(seed=random_state + init_index)
        centroids = _initialize_centroids(X_array, n_clusters, rng)

        for iteration in range(max_iter):
            labels = _assign_clusters(X_array, centroids)
            new_centroids = _update_centroids(
                X_array,
                labels,
                n_clusters,
                rng,
            )

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        inertia = _compute_inertia(X_array, centroids, labels)
        print(
            f"  Run {init_index + 1}/{n_init}: "
            f"inertia = {inertia:.2f}, iterations = {iteration + 1}"
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids
            best_labels = labels

    print(f"Best inertia across runs: {best_inertia:.2f}")

    return best_centroids, best_labels, best_inertia


def compute_elbow_curve(
    X_scaled: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 10,
) -> pd.DataFrame:
    """Compute inertia values for a range of cluster numbers."""
    results: list[dict[str, float]] = []

    print("\n=== COMPUTING ELBOW CURVE (MANUAL K-MEANS INERTIA) ===")
    for k in range(k_min, k_max + 1):
        print(f"\n--- Fitting manual K-means with k = {k} ---")
        _, _, inertia = manual_kmeans_fit(
            X_scaled,
            n_clusters=k,
            n_init=5,
            max_iter=200,
            random_state=42,
        )
        results.append({"k": k, "inertia": inertia})

    elbow_df = pd.DataFrame(results)
    print("\n--- ELBOW CURVE DATA ---")
    print(elbow_df)

    return elbow_df


def plot_elbow_curve(elbow_df: pd.DataFrame) -> None:
    """Plot the elbow curve for K-means inertia vs number of clusters."""
    if not {"k", "inertia"}.issubset(elbow_df.columns):
        print("Elbow DataFrame must contain 'k' and 'inertia' columns.")
        return

    print("\n--- PLOTTING ELBOW CURVE ---")
    plt.figure()
    elbow_df.plot(
        x="k",
        y="inertia",
        kind="line",
        marker="o",
        title="Elbow Curve for Manual K-means Clustering",
    )
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.tight_layout()
    plt.show()


def run_kmeans_clustering(
    X_scaled: pd.DataFrame,
    n_clusters: int,
) -> tuple[np.ndarray, pd.Series, float]:
    """Run manual K-means clustering on the standardised features."""
    centroids, labels_array, inertia = manual_kmeans_fit(
        X_scaled,
        n_clusters=n_clusters,
        n_init=10,
        max_iter=300,
        random_state=42,
    )

    labels_series = pd.Series(
        labels_array,
        index=X_scaled.index,
        name="Cluster",
    )

    print("\n--- FINAL CLUSTER LABEL COUNTS ---")
    print(labels_series.value_counts().sort_index())

    return centroids, labels_series, inertia


def attach_cluster_labels(
    df_aligned: pd.DataFrame,
    labels: pd.Series,
    label_column: str = "Cluster",
) -> pd.DataFrame:
    """Attach cluster labels to the aligned SA2 metadata DataFrame."""
    df_clustered = df_aligned.copy()
    df_clustered[label_column] = labels

    print("\n=== SA2 DATA WITH CLUSTER LABELS (HEAD) ===")
    print(df_clustered.head())

    output_dir = Path(r"C:\Users\bobby\Desktop\SA2")
    output_dir.mkdir(parents=True, exist_ok=True)
    clustered_file_csv = output_dir.joinpath("melbourne_sa2_clusters.csv")
    df_clustered.to_csv(clustered_file_csv, index=False)
    print(f"Clustered dataset saved to: {clustered_file_csv}")

    return df_clustered


# ---------------------------------------------------------------------
# Section 2.3 - Cluster and Feature Visualisation (extended)
# ---------------------------------------------------------------------


def plot_cluster_scatter(
    df_clustered: pd.DataFrame,
    x_col: str,
    y_col: str,
    cluster_col: str = "Cluster",
) -> None:
    """Plot a scatter chart of SA2s coloured by cluster label."""
    for col in [x_col, y_col, cluster_col]:
        if col not in df_clustered.columns:
            print(
                f"Column '{col}' not found in clustered DataFrame. "
                "Skipping scatter plot."
            )
            return

    print("\n--- PLOTTING CLUSTERS SCATTER PLOT ---")
    plt.figure()
    unique_clusters = sorted(df_clustered[cluster_col].unique())

    for cluster_value in unique_clusters:
        subset = df_clustered[df_clustered[cluster_col] == cluster_value]
        plt.scatter(
            subset[x_col],
            subset[y_col],
            label=f"Cluster {cluster_value}",
            alpha=0.7,
        )

    plt.title(
        "SA2 Clusters by "
        f"{x_col} and {y_col}"
    )
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_feature_correlation_heatmap(feature_df: pd.DataFrame) -> None:
    """Plot a correlation heatmap for the clustering feature matrix."""
    print("\n--- PLOTTING FEATURE CORRELATION HEATMAP ---")
    corr = feature_df.corr()

    plt.figure()
    im = plt.imshow(corr, interpolation="nearest")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Correlation Heatmap of Clustering Features")
    plt.tight_layout()
    plt.show()


def plot_irsad_vs_irsd(df: pd.DataFrame, title_suffix: str) -> None:
    """Plot IRSAD score vs IRSD score for visualising their relationship."""
    if ("IRSAD Score" not in df.columns) or ("IRSD Score" not in df.columns):
        print(
            "Columns 'IRSAD Score' and 'IRSD Score' not found. "
            "Skipping scatter plot."
        )
        return

    print("\n--- PLOTTING IRSAD VS IRSD SCATTER ---")
    plt.figure()
    plt.scatter(df["IRSD Score"], df["IRSAD Score"], alpha=0.6)
    plt.xlabel("IRSD Score (Disadvantage)")
    plt.ylabel("IRSAD Score (Advantage / Disadvantage)")
    plt.title(f"IRSAD vs IRSD Scores ({title_suffix})")
    plt.tight_layout()
    plt.show()


def plot_cluster_sizes(labels: pd.Series) -> None:
    """Plot a bar chart of cluster sizes."""
    print("\n--- PLOTTING CLUSTER SIZE BAR CHART ---")
    counts = labels.value_counts().sort_index()

    plt.figure()
    counts.plot(
        kind="bar",
        title="Cluster Sizes (Number of Victorian SA2s per Cluster)",
    )
    plt.xlabel("Cluster label")
    plt.ylabel("Number of SA2s")
    plt.tight_layout()
    plt.show()


def plot_cluster_boxplots(
    df_clustered: pd.DataFrame,
    feature_cols: list[str],
    cluster_col: str = "Cluster",
) -> None:
    """Plot boxplots of each feature by cluster label."""
    print("\n--- PLOTTING FEATURE BOXPLOTS BY CLUSTER ---")

    for feature in feature_cols:
        if feature not in df_clustered.columns:
            print(f"Feature column '{feature}' not found. Skipping.")
            continue

        plt.figure()
        df_clustered.boxplot(
            column=feature,
            by=cluster_col,
        )
        plt.title(f"{feature} by Cluster")
        plt.suptitle("")  # Remove automatic suptitle from pandas.
        plt.xlabel("Cluster")
        plt.ylabel(feature)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------


if __name__ == "__main__":

    # Path to the SA2 SEIFA dataset (national) in the desktop SA2 folder.
    sa2_data_file = (
        r"C:\Users\bobby\Desktop\SA2\ABS_Socio_Economic_Indexes_for_Areas_"
        r"SEIFA_by_2021_SA2_-5598362606160867168.csv"
    )

    sa2_df = pd.read_csv(sa2_data_file)

    # Basic exploration and data quality checks.
    describe_dataframe(sa2_df)
    find_missing_values(sa2_df)
    find_duplicated_rows(sa2_df)
    explore_categorical_data(sa2_df)
    identify_possible_quality_issues(sa2_df)

    # NATIONAL-LEVEL SEIFA PLOTS (AUSTRALIA, ALL SA2s)
    plot_population_distribution(sa2_df, title_suffix="Australia, all SA2s")
    plot_irsad_score_distribution(sa2_df, title_suffix="Australia, all SA2s")
    plot_irsd_score_distribution(sa2_df, title_suffix="Australia, all SA2s")
    plot_irsad_vs_irsd(sa2_df, title_suffix="Australia, all SA2s")

    # Data preparation (filter to Victorian SA2s and compute density).
    sa2_prepared = prepare_sa2_data_for_clustering(sa2_df)

    # VICTORIA-ONLY PLOTS (PREPARED SA2 DATASET)
    plot_density_distribution(sa2_prepared)
    plot_population_distribution(sa2_prepared, title_suffix="Victoria, SA2s")
    plot_irsad_score_distribution(sa2_prepared, title_suffix="Victoria, SA2s")
    plot_irsd_score_distribution(sa2_prepared, title_suffix="Victoria, SA2s")
    plot_irsad_vs_irsd(sa2_prepared, title_suffix="Victoria, SA2s")

    # Build feature matrix and standardise.
    feature_df, sa2_aligned = build_feature_matrix(sa2_prepared)
    X_scaled, means, stds = standardize_features(feature_df)

    # Feature correlation heatmap (for Victoria features).
    plot_feature_correlation_heatmap(feature_df)

    # Elbow method to inspect plausible values of k.
    elbow_df = compute_elbow_curve(X_scaled, k_min=2, k_max=8)
    plot_elbow_curve(elbow_df)

    # Choose a number of clusters for interpretation.
    n_clusters = 4

    centroids, cluster_labels, final_inertia = run_kmeans_clustering(
        X_scaled,
        n_clusters=n_clusters,
    )

    sa2_clustered = attach_cluster_labels(
        sa2_aligned,
        cluster_labels,
        label_column="Cluster",
    )

    # Visualise clusters in the space of density vs socio-economic score.
    plot_cluster_scatter(
        sa2_clustered,
        x_col="Population density (persons per sq km)",
        y_col="IRSAD Score",
        cluster_col="Cluster",
    )

    # Additional cluster diagnostics: sizes and boxplots by cluster.
    plot_cluster_sizes(cluster_labels)
    plot_cluster_boxplots(
        sa2_clustered,
        feature_cols=[
            "Population density (persons per sq km)",
            "IRSAD Score",
            "IRSD Score",
            "IEO Score",
            "IER Score",
        ],
        cluster_col="Cluster",
    )
