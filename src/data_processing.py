import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any

from utils.constants import *


def process_data(df: pd.DataFrame, one_hot: bool = False) -> pd.DataFrame:
    """Process the input DataFrame according to the specified settings."""
    df = df.copy()

    df = df[df["sexo_paciente"].isin(["M", "F"])]
    df["sexo_paciente"] = df["sexo_paciente"].map({"M": 0, "F": 1})

    df.loc[df["sexo_paciente"] == 0, "gestante_paciente"] = "6"
    df["gestante_paciente"] = df["gestante_paciente"].fillna("9")
    df["raca_cor_paciente"] = df["raca_cor_paciente"].fillna("9")

    for col in BINARY_COLUMNS:
        df[col] = df[col].fillna("2")
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(2)
        df[col] = np.where(df[col] == 1, 1, 0)

    # Remove rows with missing data
    required_cols = ALL_COLUMNS_SORTED + ["evolucao_caso", "classificacao_final"]
    df = df.dropna(axis=0, how="any", subset=required_cols)

    # Remove non-infected cases or Chikungunya cases
    df = df[df["classificacao_final"].isin([DENGUE, DENGUE_ALARM, DENGUE_SEVERE])]

    target_map = {DENGUE: "low_risk", DENGUE_ALARM: "alarm", DENGUE_SEVERE: "severe" }
    df["class"] = df["classificacao_final"].map(target_map)
    df.loc[df["evolucao_caso"] == DEATH_DENGUE, "class"] = "severe"
    df = df.drop(["evolucao_caso", "classificacao_final"], axis=1)

    # df["sigla_uf_residencia"] = df["sigla_uf_residencia"].apply(uf_to_region)
    # df = df.dropna(axis=0, how="any", subset=["sigla_uf_residencia"])

    cols_to_keep = ALL_COLUMNS_SORTED + ["class"]
    df.drop(
        [col for col in df.columns if col not in cols_to_keep], 
        inplace=True, axis=1
    )

    df["idade_paciente"] = df["idade_paciente"].apply(process_age_as_numeric)
    df["dias_sintomas_notificacao"] = df["dias_sintomas_notificacao"].apply(process_diagnosis_delay_as_numeric)

    df = df.dropna(axis=0, how="any", subset=list(NUMERIC_COLUMNS))
    for col in NUMERIC_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors='coerce', downcast='float')

    if one_hot:
        cols_to_encode = [col for col in df.columns if col in CATEGORICAL_COLUMNS]
        prefix_map = {col: f'one_hot_{col}' for col in cols_to_encode}
        df = pd.get_dummies(df, columns=cols_to_encode, prefix=prefix_map, dtype=int)
    
    return df


# --- Conversion Utilities ---

def uf_to_region(uf: str) -> str:
    """
    Converts UF (attribute "sigla_uf_residencia") into region.
    """
    if uf is None: return None 
    uf = uf.upper()
    if uf in {"AC", "AP", "AM", "PA", "RO", "RR", "TO"}: return "N" # North
    if uf in {"AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"}: return "NE" # Northeast
    if uf in {"DF", "GO", "MT", "MS"}: return "CO" # Central-West
    if uf in {"ES", "MG", "RJ", "SP"}: return "SE" # Southeast
    if uf in {"PR", "RS", "SC"}: return "S" # South
    return None 


def group_age(age: str) -> str:
    """
    Converts age (attribute "idade_paciente") into a category 
    based on a clinical grouping.
    """
    if age is None: return None
    
    val_type, age_val = tuple(map(int, age.split("-")))

    if val_type == 3 or val_type == 2:
        return "1" # Infant
    
    if val_type != 4 or age_val > 120:
        return None # Invalid
    
    if age_val <= 4: return "2" # Young children
    if age_val <= 12: return "3" # Children
    if age_val <= 17: return "4" # Adolescents
    if age_val <= 49: return "5" # Young adulds
    if age_val <= 64: return "6" # Older adults
    return "7" # Elderly


def group_diagnosis_delay(diagnosis_delay: str) -> str:
    """
    Converts the symptoms-diagnosis delay (attribute "dias_sintomas_notificacao")
    into a category based on a clinical phase-based grouping.
    """
    try: diagnosis_delay = int(diagnosis_delay)
    except Exception: return None
    
    if diagnosis_delay < 0 or diagnosis_delay > 365: 
        return None # Invalid
    
    if diagnosis_delay <= 3: return "1" # Early diagnosis
    if diagnosis_delay <= 7: return "2" # Critical phase diagnosis
    return "3" # Late diagnosis


def process_age_as_numeric(age: str) -> int:
    """
    Converts age (attribute "idade_paciente") into a numeric value in years.
    Invalid or inconsistent data is set to None.
    """
    if age is None: return None
    
    try:
        val_type, age_val = tuple(map(int, age.split("-")))
    except:
        return None

    if val_type < 4: return 0
    
    if val_type != 4 or age_val > 120 or age_val < 1:
        return None
    
    return age_val


def process_diagnosis_delay_as_numeric(diagnosis_delay: str) -> int:
    """
    Converts the symptoms-diagnosis delay (attribute "dias_sintomas_notificacao")
    into a numeric value in days. Invalid data is set to None.
    """
    try: 
        diagnosis_delay = int(diagnosis_delay)
    except Exception: 
        return None

    if diagnosis_delay < 0: diagnosis_delay = 0
    elif diagnosis_delay > 365: diagnosis_delay = 365

    return diagnosis_delay


# --- Data Loading Utilities ---

def read_parquet_data(input_path: Path) -> pd.DataFrame:
    """Reads the input Parquet file or directory of Parquet files into a DataFrame."""
    input_path = Path(input_path)

    if input_path.is_dir():
        # Read all Parquet files in the directory and join them into a single DataFrame
        df_list = []
        for file_path in input_path.glob("*.parquet"):
            try:
                df_part = pd.read_parquet(file_path)
                df_list.append(df_part)
            except Exception as e:
                raise RuntimeError(f"Error reading input file {file_path}: {e}")
        df = pd.concat(df_list, ignore_index=True)
    else:
        # Read single Parquet file
        try:
            df = pd.read_parquet(input_path)
        except Exception as e:
            raise RuntimeError(f"Error reading input file {input_path}: {e}")
    return df


def parse_args() -> Dict[str, Any]:
    import argparse
    parser = argparse.ArgumentParser(description="Process SINAN data.")
    parser.add_argument('input', type=str, help='Input Parquet file path.')
    parser.add_argument('-e', '--one-hot-encode', action='store_true', help='Encode all categorical attributes as one-hot.')
    parser.add_argument('-o', '--output', default=None, type=str, help='Output CSV file path.')
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = parse_args()

    input_path = Path(args['input'])
    one_hot = args['one_hot_encode']
    output_path = args['output']

    if not input_path.exists():
        raise FileNotFoundError(f"Input file {input_path} does not exist.")

    try:
        df = read_parquet_data(input_path)
    except Exception as e:
        raise RuntimeError(f"Error reading input data: {e}")
    
    if output_path is None:
        if input_path.is_dir():
            input_dir = input_path
            base_name = "full"
        else:
            input_dir = input_path.parent
            base_name = input_path.stem
        base_name += "-processed.csv"
        output_path = input_dir / base_name

    df_processed = process_data(df, one_hot=one_hot)
    df_processed.to_csv(output_path, encoding='utf-8', index=False)