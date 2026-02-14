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

GEOGRAPHIC_COLUMNS  = {"sigla_uf_residencia"}
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
    "gestante_paciente", "raca_cor_paciente", "sigla_uf_residencia"
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