# Classificação
DESCARTADO = "5"
DENGUE = "10"
DENGUE_ALARME = "11"
DENGUE_GRAVE = "12"
CHIKUNGUNYA = "13"

# Evolução do caso
OBITO_POR_AGRAVO = "2"
OBITO_POR_OUTRAS_CAUSAS = "3"

PATIENT_ATTRS = {
    "sigla_uf_residencia", "idade_paciente", "sexo_paciente", 
    "raca_cor_paciente", "gestante_paciente",
    "dias_sintomas_notificacao",

    "possui_doenca_autoimune", "possui_diabetes", "possui_doencas_hematologicas",
    "possui_hepatopatias", "possui_doenca_renal", "possui_hipertensao",
    "possui_doenca_acido_peptica",

    "apresenta_febre", "apresenta_cefaleia", "apresenta_exantema",
    "apresenta_dor_costas", "apresenta_mialgia", "apresenta_vomito", 
    "apresenta_conjutivite", "apresenta_dor_retroorbital", "apresenta_artralgia", 
    "apresenta_artrite", "apresenta_leucopenia", "apresenta_petequias", 
    "prova_laco"
}

BINARY_ATTRS = {
    "prova_laco",
    *{key for key in PATIENT_ATTRS if key.startswith(("possui_", "apresenta_"))}
}
NUMERIC_ATTRS = {"idade_paciente", "dias_sintomas_notificacao"}

CATEGORICAL_ATTRS = PATIENT_ATTRS - BINARY_ATTRS - NUMERIC_ATTRS

PATIENT_DISEASES = {
    key for key in PATIENT_ATTRS if key.startswith(("possui_"))
}
PATIENT_SYMPTOMS = {
    key for key in PATIENT_ATTRS if key.startswith("apresenta_")
}