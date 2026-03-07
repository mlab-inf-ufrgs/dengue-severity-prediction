# --- Target Class Info ---

# Case classification codes
DISCARDED = "5"
DENGUE = "10"
DENGUE_ALARM = "11"
DENGUE_SEVERE = "12"
CHIKUNGUNYA = "13"

# Case outcome (evolution) codes
DEATH_DENGUE = "2"
DEATH_OTHER = "3"

# --- Scheme Columns Info ---

# Codes
DOES_NOT_APPLY = "6"  # Pregnancy
IGNORED = "9"   # General
YEAR_CODE = 4 # Type of age code indicating years

GEOGRAPHIC_COLUMNS  = {"sigla_uf_residencia", "classificacao_municipio"}
DEMOGRAPHIC_COLUMNS  = {"idade_paciente", "sexo_paciente", "gestante_paciente", "raca_cor_paciente"}
DISEASES_COLUMNS = {
    "possui_doenca_autoimune", "possui_diabetes", "possui_doencas_hematologicas",
    "possui_hepatopatias", "possui_doenca_renal", "possui_hipertensao",
    "possui_doenca_acido_peptica"
}
SYMPTOMS_COLUMNS  = {
    "apresenta_febre", "apresenta_cefaleia", "apresenta_exantema",
    "apresenta_dor_costas", "apresenta_mialgia", "apresenta_vomito", 
    "apresenta_conjutivite", "apresenta_dor_retroorbital", "apresenta_artralgia", 
    "apresenta_artrite", "apresenta_leucopenia", "apresenta_petequias"
}
OTHER_COLUMNS  = {"prova_laco", "dias_sintomas_notificacao"}

ALL_COLUMNS = {
    *GEOGRAPHIC_COLUMNS,
    *DEMOGRAPHIC_COLUMNS,
    *DISEASES_COLUMNS,
    *SYMPTOMS_COLUMNS,
    *OTHER_COLUMNS
}
ALL_COLUMNS_SORTED = sorted(list(ALL_COLUMNS))

BINARY_COLUMNS = {
    "prova_laco", "sexo_paciente", *DISEASES_COLUMNS, *SYMPTOMS_COLUMNS
}
NUMERIC_COLUMNS = {
    "idade_paciente", "dias_sintomas_notificacao"
}
CATEGORICAL_COLUMNS = {
    "gestante_paciente", "raca_cor_paciente", "sigla_uf_residencia", "classificacao_municipio"
}


# --- Training/Testing Constants ---

RANDOM_STATE = 42
TEST_RATIO = 0.15
N_FOLDS = 5
N_CLASSES = 3

TARGET_NAMES = ["low_risk", "alarm", "severe"]
TARGET_NAMES_COARSE = ["low_risk", "high_risk"]
TARGET_NAMES_FINE = ["alarm", "severe"]

TARGET_LABEL_MAP = {name: idx for idx, name in enumerate(TARGET_NAMES)}
LABEL_TARGET_MAP = {idx: name for idx, name in enumerate(TARGET_NAMES)}

COARSE_LABEL_MAP = {0: 0, 1: 1, 2: 1}
FINE_LABEL_MAP = {1: 0, 2: 1}
FINE_LABEL_MAP_REVERSE = {0: 1, 1: 2}

DATASET_RF_PATH = "../../data/3_gold/dataset-processed-rf.csv"
DATASET_GB_PATH = "../../data/3_gold/dataset-processed-gb.csv"

BEST_PARAMS = {
    'CB_hier': {
        'iterations': 2000,
        'loss_function': 'MultiClass',
        'eval_metric': 'AUC',         
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 5,
        'random_state': RANDOM_STATE,
        'cat_features': list(CATEGORICAL_COLUMNS),
        'allow_writing_files': False,

        'learning_rate_coarse': 0.0337314168080377, 
        'depth_coarse': 10, 
        'colsample_bylevel_coarse': 0.8, 
        'min_data_in_leaf_coarse': 7, 
        'bootstrap_type_coarse': 'MVS',
        'learning_rate_fine': 0.07404977644601698, 
        'depth_fine': 10, 
        'colsample_bylevel_fine': 0.9, 
        'min_data_in_leaf_fine': 37, 
        'bootstrap_type_fine': 'MVS',
    },

    'CB_multi': {
        'iterations': 2000,
        'loss_function': 'MultiClass',
        'eval_metric': 'AUC',         
        'auto_class_weights': 'Balanced',
        'early_stopping_rounds': 5,
        'random_state': RANDOM_STATE,
        'cat_features': list(CATEGORICAL_COLUMNS),
        'allow_writing_files': False,
        
        'learning_rate': 0.03158826697434671, 
        'depth': 8, 
        'colsample_bylevel': 0.6, 
        'min_data_in_leaf': 19, 
        'bootstrap_type': 'Bayesian', 
        'bagging_temperature': 0.1985479799912282
    },

    'HGB_hier': {
        'max_iter': 2000,
        'early_stopping': True,
        'n_iter_no_change': 5,
        'tol': 1e-7,
        'categorical_features': list(CATEGORICAL_COLUMNS),
        'class_weight': 'balanced',
        'loss': 'log_loss',
        'random_state': RANDOM_STATE,

        'learning_rate_coarse': 0.018530923118263938, 
        'max_depth_coarse': 9, 
        'min_samples_leaf_coarse': 50, 
        'l2_regularization_coarse': 0.8, 
        'learning_rate_fine': 0.03319143627863521, 
        'max_depth_fine': 10, 
        'min_samples_leaf_fine': 15, 
        'l2_regularization_fine': 1.0, 
        'mode': 'soft'
    },

    'HGB_multi': {
        'max_iter': 2000,
        'early_stopping': True,
        'n_iter_no_change': 5,
        'tol': 1e-7,
        'categorical_features': list(CATEGORICAL_COLUMNS),
        'class_weight': 'balanced',
        'loss': 'log_loss',
        'random_state': RANDOM_STATE,

        'learning_rate': 0.03620016665032439, 
        'max_depth': 10, 
        'min_samples_leaf': 20, 
        'l2_regularization': 0.5
    },

    'RF_hier': {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,

        'n_estimators_coarse': 100,
        'max_depth_coarse': 48,
        'min_samples_split_coarse': 13,
        'min_samples_leaf_coarse': 6,
        'max_features_coarse': 'sqrt',
        'n_estimators_fine': 150, 
        'max_depth_fine': 24, 
        'min_samples_split_fine': 4, 
        'min_samples_leaf_fine': 1, 
        'max_features_fine': 'sqrt', 
        'mode': 'soft'
    },

    'RF_multi': {
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
        'max_features': 'log2',

        'n_estimators': 250, 
        'max_depth': 39, 
        'min_samples_split': 5, 
        'min_samples_leaf': 2
    },

    'XGB_hier': {
        "early_stopping_rounds": 10,
        "enable_categorical": True,
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "tree_method": "hist",
        #"device": "cuda",
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "verbosity": 0,

        'n_estimators_coarse': 1600,
        'max_depth_coarse': 14,
        'learning_rate_coarse': 0.06310200379301271,
        'subsample_coarse': 0.8,
        'colsample_bytree_coarse': 0.5,
        'reg_alpha_coarse': 5,
        'reg_lambda_coarse': 47,
        'min_child_weight_coarse': 9,
        'gamma_coarse': 0.0,
        'n_estimators_fine': 2600,
        'max_depth_fine': 2,
        'learning_rate_fine': 0.0015059118006550702,
        'subsample_fine': 0.8,
        'colsample_bytree_fine': 0.7000000000000001,
        'reg_alpha_fine': 38,
        'reg_lambda_fine': 23,
        'min_child_weight_fine': 7,
        'gamma_fine': 4.5,
        'mode': 'soft'
    },

    'XGB_multi': {
        "n_estimators": 2000,
        "early_stopping_rounds": 5,
        "enable_categorical": True,
        "objective": "multi:softprob",
        "num_class": N_CLASSES,
        "tree_method": "hist",
        #"device": "cuda",
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE,
        "verbosity": 0,

        'max_depth': 16, 
        'learning_rate': 0.107893597297338, 
        'subsample': 0.9632988872870645, 
        'colsample_bytree': 0.9646675197441156, 
        'reg_alpha': 0.5877393757763218,
        'reg_lambda': 22.37101441077761, 
        'min_child_weight': 6, 
        'gamma': 0.11410499025763864
    }
}