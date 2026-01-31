import os
import sys
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from utils.helpers import get_project_root  # usa o bootstrap padrão

def save_parquet_in_chunks(df, relative_path, max_size_mb=25, compression='snappy'):
    """
    Divide a large DataFrame into multiple Parquet files, each with a size up to `max_size_mb`.
    
    Args:
        df (pd.DataFrame): The full DataFrame to split and save.
        relative_path (str): Path relative to project root, e.g., "data/staging/cnes/tbEstabelecimento".
        max_size_mb (int): Max size in MB for each Parquet file.
        compression (str): Parquet compression type (e.g., 'snappy', 'brotli', 'gzip').
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    project_root = get_project_root()
    base_path = project_root / relative_path
    os.makedirs(base_path.parent, exist_ok=True)

    partition_index = 1
    start_idx = 0
    n_rows = len(df)

    pbar = tqdm(total=n_rows, desc="Salvando arquivos Parquet", unit="linhas")

    while start_idx < n_rows:
        low = 1000
        high = n_rows - start_idx
        best_step = None

        # Busca binária para encontrar o maior chunk possível <= max_size_bytes
        while low <= high:
            mid = (low + high) // 2
            chunk_df = df.iloc[start_idx:start_idx + mid]
            temp_path = base_path.parent / "__temp.parquet"
            chunk_df.to_parquet(temp_path, compression=compression, index=False)
            size = temp_path.stat().st_size
            os.remove(temp_path)

            if size <= max_size_bytes:
                best_step = mid
                low = mid + 1
            else:
                high = mid - 1

        if best_step is None:
            raise ValueError("No chunk could be created under the size limit. Try increasing compression or reducing columns.")

        # Salva o melhor chunk encontrado
        chunk_df = df.iloc[start_idx:start_idx + best_step]
        final_path = base_path.parent / f"{base_path.name}_part_{partition_index}.parquet"
        chunk_df.to_parquet(final_path, compression=compression, index=False)
        size_mb = final_path.stat().st_size / (1024 * 1024)
        #print(f"✔️ final_path salvo com {size_mb:.2f} MB ({best_step} linhas)")
        print(f"✔️ {base_path.name}_part_{partition_index}.parquet salvo com {size_mb:.2f} MB ({best_step} linhas)")

        start_idx += best_step
        partition_index += 1
        pbar.update(best_step)

    pbar.close()
    print("✅ Todos os arquivos salvos com sucesso.")

########################################################################################################################## 
from pathlib import Path
import pandas as pd

def load_partitioned_parquet(base_path: Path, verbose: bool = True) -> pd.DataFrame:
    """
    Reads all .parquet files under a directory of subfolders (e.g., year=2020),
    forcing all columns to string to avoid schema/type conflicts (e.g., dbdate errors).

    Parameters:
    - base_path: Path to the root directory with partitioned folders
    - verbose: Whether to print status and errors

    Returns:
    - Concatenated DataFrame with all values as string
    """
    all_dfs = []
    base_path = Path(base_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Path not found: {base_path}")

    for subdir in sorted(base_path.glob("*")):
        if subdir.is_dir():
            if verbose:
                print(f"📁 Reading from {subdir.name}")
            for parquet_file in sorted(subdir.glob("*.parquet")):
                try:
                    df = pd.read_parquet(parquet_file, engine="pyarrow")
                    df = df.astype(str)  # força tudo como string
                    all_dfs.append(df)
                except Exception as e:
                    if verbose:
                        print(f"❌ Error reading {parquet_file.name}: {e}")

    if not all_dfs:
        raise ValueError("No valid Parquet files were read.")

    return pd.concat(all_dfs, ignore_index=True)
