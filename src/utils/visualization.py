from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

CAPTION_FONT_SIZE = 13
AXIS_FONT_SIZE = 15

def plot_metrics_by_class(y_true, y_pred, target_names, figsize=(10, 6)):
    """
    Plota métricas de classificação (Precision, Recall, F1-Score) por classe.
    
    Parameters:
    -----------
    y_true : array-like
        Labels verdadeiros
    y_pred : array-like
        Labels preditos
    target_names : list
        Lista com nomes das classes
    figsize : tuple, optional
        Tamanho da figura (width, height)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axes
        Objetos da figura e eixos para manipulação adicional
    report_dict : dict
        Dicionário com métricas detalhadas do classification_report
    """
    # Obter report como dicionário
    report_dict = classification_report(y_true, y_pred, target_names=target_names, 
                                       zero_division=0, output_dict=True)
    
    # Criar figura
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(target_names))
    width = 0.25
    
    # Preparar dados
    precision_values = [report_dict[cls]['precision'] for cls in target_names] # type: ignore
    recall_values = [report_dict[cls]['recall'] for cls in target_names] # type: ignore
    f1_values = [report_dict[cls]['f1-score'] for cls in target_names] # type: ignore
    
    # Criar barras
    bars1 = ax.bar(x - width, precision_values, width, label='Precision', alpha=0.8)
    bars2 = ax.bar(x, recall_values, width, label='Recall', alpha=0.8)
    bars3 = ax.bar(x + width, f1_values, width, label='F1-Score', alpha=0.8)
    
    # Adicionar valores no topo das barras
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Configurar gráfico
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Values', fontsize=12)
    ax.set_title('Classification Metrics by Class', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(target_names)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    return fig, ax, report_dict


def print_summary_metrics(report_dict):
    """
    Imprime métricas de classificação resumidas (Accuracy, Macro Avg, Weighted Avg).
    
    Parameters:
    -----------
    report_dict : dict
        Dicionário retornado pelo classification_report com output_dict=True
    """
    print(f"\n{'='*60}")
    print(f"{'CLASSIFICATION METRICS (SUMMARY)':^60}")
    print(f"{'='*60}")
    print(f"\nAccuracy: {report_dict['accuracy']:.3f}\n")
    print(f"{'':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print(f"{'-'*60}")
    print(f"{'Macro Avg':<20} {report_dict['macro avg']['precision']:>12.3f} "
          f"{report_dict['macro avg']['recall']:>12.3f} "
          f"{report_dict['macro avg']['f1-score']:>12.3f}")
    print(f"{'Weighted Avg':<20} {report_dict['weighted avg']['precision']:>12.3f} "
          f"{report_dict['weighted avg']['recall']:>12.3f} "
          f"{report_dict['weighted avg']['f1-score']:>12.3f}")
    print(f"{'='*60}\n")


# Para utilizar: ! pip install jinja2
def print_summary_metrics_latex(report_dict):
    """
    Imprime métricas de classificação resumidas em formato LaTeX.
    
    Parameters:
    -----------
    report_dict : dict
        Dicionário retornado pelo classification_report com output_dict=True
    """
    # Criar DataFrame com as métricas
    metrics_data = {
        'Metric': ['Precision', 'Recall', 'F1-Score'],
        'Macro Avg': [
            report_dict['macro avg']['precision'],
            report_dict['macro avg']['recall'],
            report_dict['macro avg']['f1-score']
        ],
        'Weighted Avg': [
            report_dict['weighted avg']['precision'],
            report_dict['weighted avg']['recall'],
            report_dict['weighted avg']['f1-score']
        ]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    # Gerar LaTeX
    latex_table = df_metrics.to_latex(index=False, float_format="%.3f", 
                                      caption=f"Classification Metrics (Accuracy: {report_dict['accuracy']:.3f})",
                                      label="tab:metrics")
    
    print(latex_table)


def box_plot_from_txt(model: str, validation_dir: str = 'results/validation', urban_hierarchy: bool = False):
    input_path = validation_dir + "/fold_scores/" + f"{model}_fold_scores.txt"

    if urban_hierarchy:
        input_path = input_path.replace('.txt', '_u.txt')

    df_results = pd.read_csv(input_path, sep='\t')

    stats = df_results.groupby(['Model', 'Version'])['F1_Macro'].agg(['mean', 'std']).reset_index()
    stats['Text'] = stats['Model'] + " (" + stats['Version'] + "): " + \
                    stats['mean'].round(4).astype(str) + " ± " + stats['std'].round(4).astype(str)
    stats_text = "\n".join(stats['Text'])

    df_results['Algorithm'] = model

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(bottom=0.25)

    sns.boxplot(
        data=df_results, y='F1_Macro', hue='Version',
        palette='Set2', showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
        ax=ax
    )

    ax.set_ylabel('Macro F1-Score', fontsize=AXIS_FONT_SIZE)
    ax.legend(title='Version', fontsize=CAPTION_FONT_SIZE, title_fontsize=CAPTION_FONT_SIZE)
    ax.tick_params(axis='x', bottom=False, labelbottom=False)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = validation_dir + "/box_plots/" + f"{model}.png"

    if urban_hierarchy:
        output_path = output_path.replace('.png', '_u.png')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")


# função para desenhar dois pares de boxplots lado a lado para comparação, um sem a hierarquia urbana e outro com
def compare_box_plots(model: str, validation_dir: str = 'results/validation'):
    from matplotlib.patches import Patch

    input_path_no_hierarchy = validation_dir + "/fold_scores/" + f"{model}_fold_scores.txt"
    input_path_hierarchy = validation_dir + "/fold_scores/" + f"{model}_fold_scores_u.txt"

    df_no = pd.read_csv(input_path_no_hierarchy, sep='\t')
    df_uh = pd.read_csv(input_path_hierarchy, sep='\t')

    df_no['Urban Hierarchy'] = 'No Urban Hierarchy'
    df_uh['Urban Hierarchy'] = 'With Urban Hierarchy'

    df_combined = pd.concat([df_no, df_uh], ignore_index=True)

    versions = [v for v in ['Multiclass', 'Hierarchical-S', 'Hierarchical-H']
                if v in df_combined['Version'].values]
    uh_groups = ['No Urban Hierarchy', 'With Urban Hierarchy']

    # lighter shade for No UH, darker for With UH
    version_colors = {
        'Multiclass':     {'No Urban Hierarchy': '#a8e6a3', 'With Urban Hierarchy': '#2ecc71'},
        'Hierarchical-S': {'No Urban Hierarchy': '#ffd699', 'With Urban Hierarchy': '#f39c12'},
        'Hierarchical-H': {'No Urban Hierarchy': '#add8e6', 'With Urban Hierarchy': '#3498db'},
    }

    n_versions = len(versions)
    box_width = 0.25
    step = box_width + 0.08
    offsets = np.arange(n_versions) * step
    offsets -= offsets.mean()
    group_centers = np.arange(len(uh_groups), dtype=float) * (n_versions * step + 0.4)

    fig, ax = plt.subplots(figsize=(12, 7))

    for i, uh in enumerate(uh_groups):
        for j, version in enumerate(versions):
            data = df_combined[
                (df_combined['Urban Hierarchy'] == uh) & (df_combined['Version'] == version)
            ]['F1_Macro'].dropna().values

            if len(data) == 0:
                continue

            color = version_colors.get(version, {}).get(uh, '#cccccc')

            ax.boxplot(
                data,
                positions=[group_centers[i] + offsets[j]],
                widths=box_width,
                patch_artist=True,
                boxprops=dict(facecolor=color, color='#333333'),
                medianprops=dict(color='#333333', linewidth=1.5),
                whiskerprops=dict(color='#333333'),
                capprops=dict(color='#333333'),
                flierprops=dict(marker='o', markerfacecolor=color, markeredgecolor='#333333',
                                markersize=4, alpha=0.7),
                showmeans=True,
                meanprops={"marker": "o", "markerfacecolor": "white",
                           "markeredgecolor": "black", "markersize": 8}
            )

    ax.set_xticks(group_centers)
    ax.set_xticklabels(uh_groups, fontsize=AXIS_FONT_SIZE)
    ax.set_ylabel('Macro F1-Score', fontsize=AXIS_FONT_SIZE)
    ax.set_xlabel('')
    ax.tick_params(axis='y', labelsize=CAPTION_FONT_SIZE)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    legend_base_colors = {'Multiclass': '#2ecc71', 'Hierarchical-S': '#f39c12', 'Hierarchical-H': '#3498db'}
    legend_elements = [
        Patch(facecolor=legend_base_colors[v], edgecolor='#333333', label=v)
        for v in versions if v in legend_base_colors
    ]
    ax.legend(handles=legend_elements, title='Version', fontsize=CAPTION_FONT_SIZE, title_fontsize=CAPTION_FONT_SIZE)

    output_path = validation_dir + "/box_plots/" + f"{model}_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Comparison plot saved to {output_path}")