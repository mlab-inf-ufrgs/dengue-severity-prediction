from sklearn.metrics import classification_report
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns

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
        data=df_results, x='Algorithm', y='F1_Macro', hue='Version',
        palette='Set2', showmeans=True,
        meanprops={"marker":"o", "markerfacecolor":"white", "markeredgecolor":"black", "markersize":"8"},
        ax=ax
    )

    ax.set_ylabel('F1-Macro Score')
    ax.set_xlabel('Algorithm')
    ax.legend(title='Version')
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    output_path = validation_dir + "/box_plots/" + f"{model}.png"

    if urban_hierarchy:
        output_path = output_path.replace('.png', '_u.png')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")