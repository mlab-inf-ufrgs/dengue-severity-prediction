import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path
from scipy.stats import chi2_contingency
from data_processing import process_data, read_parquet_data
from utils.constants import (
    PATIENT_ATTRS, PATIENT_DISEASES, PATIENT_SYMPTOMS,
    NUMERIC_ATTRS
)

CORRELATIONS_TO_CHECK = [
    (col, "severity") for col in PATIENT_ATTRS
] + [
    (col1, col2) for col1 in PATIENT_DISEASES for col2 in PATIENT_SYMPTOMS
] + [
    ("prova_laco", col) for col in PATIENT_SYMPTOMS
]

def cramers_v(contingency_table):
    """ Calculate Cramér's V statistic for categorical-categorical association. """
    chi2, _, _, _ = chi2_contingency(contingency_table)
    n = contingency_table.sum().sum()
    phi2 = chi2 / n
    r, k = contingency_table.shape
    phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    r_corr = r - ((r-1)**2)/(n-1)
    k_corr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2_corr / (min((k_corr-1), (r_corr-1)) + 1e-10))


def compute_cramers_v(df):
    cramers_v_results = {}
    for col1, col2 in CORRELATIONS_TO_CHECK:
        contingency_table = pd.crosstab(df[col1], df[col2], dropna=True)
        v = cramers_v(contingency_table)
        cramers_v_results[(col1, col2)] = v
    return cramers_v_results


def compute_correlation_matrix(df):
    all_cols = list(PATIENT_ATTRS) + ["severity"]
    corr_matrix = pd.DataFrame(index=all_cols, columns=all_cols, dtype=float)

    for i, col1 in enumerate(all_cols):
        for j, col2 in enumerate(all_cols):
            if i <= j:
                contingency_table = pd.crosstab(df[col1], df[col2], dropna=True)
                v = cramers_v(contingency_table)
                corr_matrix.at[col1, col2] = v
                corr_matrix.at[col2, col1] = v
            
    return corr_matrix


def plot_correlation_heatmap(corr_matrix, target_col='severity', output_path=None, annot_threshold=0.15):
    """
    Plots a binned correlation heatmap based on the provided interpretation,
    with the target column highlighted.
    """
    
    # === 1. Define Bins and Colors based on your image ===
    # Bins: [0, 0.05, 0.10, 0.15, 0.25, 1.0]
    # This corresponds to the categories:
    # >0 (No/Very Weak), >0.05 (Weak), >0.10 (Moderate), >0.15 (Strong), >0.25 (Very Strong)
    
    bins = [0, 0.05, 0.10, 0.15, 0.25, 1.0]
    
    # 5 colors for the 5 bins. Using a Yellow-Orange-Red sequential palette.
    colors = [
        "#f7f7f7",  # (0 - 0.05) No/Very Weak (Light Grey)
        "#fff7bc",  # (0.05 - 0.10) Weak (Pale Yellow)
        "#fcc24f",  # (0.10 - 0.15) Moderate (Light Orange)
        "#fc8123",  # (0.15 - 0.25) Strong (Orange)
        "#990000"   # (0.25 - 1.0) Very Strong (Dark Red)
    ]
    
    # Create the discrete colormap and normalization
    my_cmap = ListedColormap(colors)
    my_norm = BoundaryNorm(bins, my_cmap.N)
    # =======================================================
    
    num_features = corr_matrix.shape[0]
    fig_width = max(12, int(num_features * 0.7))
    fig_height = max(10, int(num_features * 0.6))

    plt.figure(figsize=(fig_width, fig_height))
    ax = plt.gca()

    # Conditional Annotation (hides values below threshold)
    annot_labels = corr_matrix.applymap(lambda x: f"{x:.2f}" if abs(x) >= annot_threshold else "")

    sns.heatmap(
        corr_matrix.astype(float),
        annot=annot_labels,
        fmt="",
        # --- Use the new colormap and normalization ---
        cmap=my_cmap,
        norm=my_norm,
        # ---
        square=True,
        linewidths=.5,
        linecolor='white',
        annot_kws={"size": 8},
        ax=ax,
        # --- Update the color bar ---
        cbar_kws={
            "shrink": .8,
            "ticks": bins,  # Put ticks at the bin edges
            "label": "Cramér's V Strength"
        }
    )

    plt.title("Cramér's V Correlation Heatmap (Binned Interpretation)", fontsize=16)
    plt.xlabel("")
    plt.ylabel("")

    plt.xticks(rotation=90, ha='right', fontsize=10)
    plt.yticks(rotation=0, va='center', fontsize=10)

    # --- Highlight the Target Column/Row (same as before) ---
    if target_col in corr_matrix.columns:
        target_idx = corr_matrix.columns.get_loc(target_col)
        num_features = corr_matrix.shape[0]
        
        # Highlight column
        ax.add_patch(patches.Rectangle(
            (target_idx, 0), 1, num_features,
            linewidth=3, edgecolor='lightgreen', facecolor='none', zorder=2
        ))
        # Highlight row
        ax.add_patch(patches.Rectangle(
            (0, target_idx), num_features, 1,
            linewidth=3, edgecolor='lightgreen', facecolor='none', zorder=2
        ))
        # Highlight intersection
        ax.add_patch(patches.Rectangle(
            (target_idx, target_idx), 1, 1,
            linewidth=4, edgecolor='lightgreen', facecolor='none', zorder=3
        ))
    # ---
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
    else:
        plt.show()

def collect_correlated_columns(cramers_v_results, threshold=0.4):
    return [(col1, col2, v) for (col1, col2), v in cramers_v_results.items() if v > threshold]


if __name__ == "__main__":
    input_dir = Path("data/2_silver")
    output_dir = Path("analysis_results")

    output_dir.mkdir(parents=True, exist_ok=True)

    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)

    df = read_parquet_data(input_dir)

    print("=== Original Data Info ===")
    df.info()

    df = process_data(df, as_nominal=True)

    # Normalize numeric attributes
    for col in NUMERIC_ATTRS:
        df[col] = (df[col] - df[col].mean()) / df[col].std()

    print("\n=== Processed Data Info ===")
    df.info()

    print("\n=== Processed Data Description ===")
    df_description = df.describe(include='all')
    with open(output_dir / "data_description.csv", 'w') as f:
        df_description.to_csv(f)

    cramers_v_results = compute_cramers_v(df)

    cramers_v_results = sorted(
        cramers_v_results.items(), key=lambda item: item[1], reverse=True
    )
    cramers_v_results = {k: v for k, v in cramers_v_results}
    
    results_path = output_dir / "cramers_v_results.csv"
    figure_path = output_dir / "cramers_v_heatmap.pdf"

    with open(results_path, 'w') as f:
        f.write("Column_1,Column_2,Cramers_V\n")
        for (col1, col2), v in cramers_v_results.items():
            f.write(f"{col1},{col2},{v:.4f}\n")

    corr_matrix = compute_correlation_matrix(df)
    plot_correlation_heatmap(corr_matrix, output_path=figure_path)
