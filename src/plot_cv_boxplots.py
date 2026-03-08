from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.utils.class_weight import compute_sample_weight

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from utils.constants import *
from utils.visualization import box_plot_from_txt, compare_box_plots

# --- Helper Functions ---
def get_coarse_fine_data(X, y):
    X_coarse = X.copy()
    y_coarse = y.copy().map(COARSE_LABEL_MAP)

    high_risk_mask = y.isin([1, 2])
    X_fine = X[high_risk_mask].copy()
    y_fine = y[high_risk_mask].copy().map(FINE_LABEL_MAP)

    return X_coarse, y_coarse, X_fine, y_fine


def predict_soft_cascade(model_coarse, model_fine, X, y):
    probs_coarse = model_coarse.predict_proba(X)
    p_high_risk = probs_coarse[:, 1]
    
    probs_fine = model_fine.predict_proba(X) 
    p_severe_given_high = probs_fine[:, 1]
        
    p_severe_global = p_high_risk * p_severe_given_high
    p_alarm_global = p_high_risk * (1 - p_severe_given_high)
    p_low_global = 1.0 - p_high_risk
    
    final_probs = np.vstack([p_low_global, p_alarm_global, p_severe_global]).T
    final_preds = np.argmax(final_probs, axis=1)

    return f1_score(y, final_preds, average='macro')


def predict_hard_cascade(model_coarse, model_fine, X, y, threshold_coarse=0.5, threshold_fine=0.5):
    probs_coarse = model_coarse.predict_proba(X)[:, 1]
    preds_coarse = (probs_coarse >= threshold_coarse).astype(int)
    final_preds = preds_coarse.copy()
    
    high_risk_indices = np.where(preds_coarse == 1)[0]
    
    if len(high_risk_indices) > 0:
        X_high_risk = X.iloc[high_risk_indices]
        probs_fine_local = model_fine.predict_proba(X_high_risk)[:, 1]
        preds_fine_local = (probs_fine_local >= threshold_fine).astype(int)
        preds_fine_global = np.array([FINE_LABEL_MAP_REVERSE[p] for p in preds_fine_local])
        final_preds[high_risk_indices] = preds_fine_global

    return f1_score(y, final_preds, average='macro')


def extract_hier_params(model_key: str):
    """
    Extrai params coarse/fine puramente do BEST_PARAMS[model_key].
    """
    raw = BEST_PARAMS.get(model_key, {}).copy()

    params_coarse = {}
    params_fine = {}
    shared = {}

    for key, value in raw.items():
        if key.endswith('_coarse'):
            params_coarse[key[: -len('_coarse')]] = value
        elif key.endswith('_fine'):
            params_fine[key[: -len('_fine')]] = value
        else:
            shared[key] = value

    use_soft = shared.pop('mode', 'soft') == 'soft'

    params_coarse.update(shared)
    params_fine.update(shared)

    return params_coarse, params_fine, use_soft


def extract_multi_params(model_key: str):
    """
    Retorna os params de BEST_PARAMS[model_key].
    """
    return BEST_PARAMS.get(model_key, {}).copy()


# --- Evaluation Functions ---
def evaluate_multi(model_name, params, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        if model_name == 'RF':
            clf = RandomForestClassifier(**params)
            clf.fit(X_train, y_train)

        elif model_name == 'XGB':
            clf = XGBClassifier(**params)
            w = compute_sample_weight(class_weight='balanced', y=y_train)
            clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], sample_weight=w, verbose=False)
            
        elif model_name == 'CB':
            clf = CatBoostClassifier(**params)
            cat_features = list(CATEGORICAL_COLUMNS.intersection(X_train.columns))
            clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], cat_features=cat_features, verbose=False)

        elif model_name == 'HGB':
            clf = HistGradientBoostingClassifier(**params)
            clf.fit(X_train, y_train)
            
        preds = clf.predict(X_valid) # type: ignore
        scores.append(f1_score(y_valid, preds, average='macro'))

    return scores


def evaluate_hier(model_name, params_coarse, params_fine, use_soft_cascade, threshold_coarse, threshold_fine, X, y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for train_idx, valid_idx in skf.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        X_coarse, y_coarse, X_fine, y_fine = get_coarse_fine_data(X_train, y_train)
        X_coarse_valid, y_coarse_valid, X_fine_valid, y_fine_valid = get_coarse_fine_data(X_valid, y_valid)

        if model_name == 'RF':
            clf_coarse = RandomForestClassifier(**params_coarse)
            clf_fine = RandomForestClassifier(**params_fine)

            clf_coarse.fit(X_coarse, y_coarse)
            clf_fine.fit(X_fine, y_fine)

        elif model_name == 'XGB':
            clf_coarse = XGBClassifier(**params_coarse)
            clf_fine = XGBClassifier(**params_fine)

            w_coarse = compute_sample_weight(class_weight='balanced', y=y_coarse)
            w_fine = compute_sample_weight(class_weight='balanced', y=y_fine)

            clf_coarse.fit(X_coarse, y_coarse, eval_set=[(X_coarse_valid, y_coarse_valid)], sample_weight=w_coarse, verbose=False)
            clf_fine.fit(X_fine, y_fine, eval_set=[(X_fine_valid, y_fine_valid)], sample_weight=w_fine, verbose=False)

        elif model_name == 'CB':
            clf_coarse = CatBoostClassifier(**params_coarse)
            clf_fine = CatBoostClassifier(**params_fine)

            cat_features = list(CATEGORICAL_COLUMNS.intersection(X_train.columns))

            clf_coarse.fit(X_coarse, y_coarse, eval_set=[(X_coarse_valid, y_coarse_valid)], cat_features=cat_features, verbose=False)
            clf_fine.fit(X_fine, y_fine, eval_set=[(X_fine_valid, y_fine_valid)], cat_features=cat_features, verbose=False)
            
        elif model_name == 'HGB':
            clf_coarse = HistGradientBoostingClassifier(**params_coarse)
            clf_fine = HistGradientBoostingClassifier(**params_fine)

            clf_coarse.fit(X_coarse, y_coarse)
            clf_fine.fit(X_fine, y_fine)
            
        if use_soft_cascade:
            f1 = predict_soft_cascade(clf_coarse, clf_fine, X_valid, y_valid) # type: ignore
        else:
            f1 = predict_hard_cascade(clf_coarse, clf_fine, X_valid, y_valid, threshold_coarse, threshold_fine) # type: ignore
        scores.append(f1)
        
    return scores

def validation(models, urban_hierarchy=False):
    print("Loading datasets...")

    if 'RF' in models:
        df_rf = pd.read_csv("data/3_gold/dataset-processed-rf.csv")

        X_rf = df_rf.drop("class", axis=1)
        y_rf = df_rf["class"].map(TARGET_LABEL_MAP)
        X_train_rf, _, y_train_rf, _ = train_test_split(X_rf, y_rf, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y_rf)

    
    if any(m in models for m in ['XGB', 'CB', 'HGB']):
        df_gb = pd.read_csv("data/3_gold/dataset-processed-gb.csv")

        for col in CATEGORICAL_COLUMNS:
            if col in df_gb.columns:
                df_gb[col] = df_gb[col].astype('category')
                #df_gb[col] = df_gb[col].cat.add_categories('None').fillna('None') <--- Investigar
            
        X_gb = df_gb.drop("class", axis=1)
        y_gb = df_gb["class"].map(TARGET_LABEL_MAP)
        X_train_gb, _, y_train_gb, _ = train_test_split(X_gb, y_gb, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y_gb)

    versions = ['Multiclass', 'Hierarchical']
    output_dir = 'results/validation'

    for model in models:
        results = []

        for version in versions:
            print(f"Evaluating {model} - {version}...")

            X_train = X_train_rf if model == 'RF' else X_train_gb # type: ignore
            y_train = y_train_rf if model == 'RF' else y_train_gb # type: ignore

            if version == 'Multiclass':
                params = extract_multi_params(f'{model}_multi')
                scores = evaluate_multi(model, params, X_train, y_train)
            else:
                params_coarse, params_fine, use_soft_cascade = extract_hier_params(f'{model}_hier')
                threshold_coarse = 0.5
                threshold_fine = 0.5
                scores = evaluate_hier(
                    model, params_coarse, params_fine,
                    use_soft_cascade, threshold_coarse, threshold_fine,
                    X_train, y_train
                )
                version += '-S' if use_soft_cascade else '-H'

            for fold, score in enumerate(scores):
                results.append({
                    'Model': model,
                    'Version': version,
                    'Fold': fold + 1,
                    'F1_Macro': score
                })

        df_results = pd.DataFrame(results)

        # --- Salvar dados dos folds ---
        fold_data_path = output_dir + "/fold_scores/" + f"{model}_fold_scores.txt"

        if urban_hierarchy:
            fold_data_path = fold_data_path.replace('.txt', '_u.txt')

        df_results.to_csv(fold_data_path, index=False, sep='\t')
        print(f"Fold scores saved to {fold_data_path}")

def main():
    models = ['RF']
    urban_hierarchy = False

    #validation(models, urban_hierarchy=urban_hierarchy)

    for model in models:
        #box_plot_from_txt(model, urban_hierarchy=urban_hierarchy)
        compare_box_plots(model)

if __name__ == "__main__":
    main()
