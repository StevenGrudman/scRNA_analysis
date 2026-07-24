import anndata as ad
import numpy as np
import scanpy as sc
import pandas as pd
from pathlib import Path


### Configuration

random_seed = 123
cell_fraction = 0.8
path = Path("/home/grudmans/EC_ref/PMID_39198675")
input_path = path / "endometriumAtlasV2_cells_with_counts.h5ad"
output_path = path / "endometriumAtlasV2_cells_with_counts_80.h5ad"
stratify_columns = ["sample", "celltype"]


### Load the cell metadata without loading the full expression matrix

if not input_path.exists():
    raise FileNotFoundError(f"Input AnnData object not found: {input_path}")
if output_path.exists():
    raise FileExistsError(f"Output already exists: {output_path}")

adata_backed = ad.read_h5ad(input_path, backed="r")
missing_columns = [column for column in stratify_columns if column not in adata_backed.obs.columns]
if missing_columns:
    adata_backed.file.close()
    raise KeyError(f"Stratification columns are missing from adata.obs: {missing_columns}")

print("Input shape:", adata_backed.shape)
print("Sampling fraction:", cell_fraction)
print("Stratifying by:", stratify_columns)


### Sample approximately one quarter of cells within every sample and cell type

rng = np.random.default_rng(random_seed)
selected_positions = []
group_positions = adata_backed.obs.groupby(stratify_columns, observed=True, dropna=False).indices
for positions in group_positions.values():
    positions = np.asarray(positions, dtype=int)
    cells_to_keep = max(1, int(round(len(positions) * cell_fraction)))
    selected_positions.extend(rng.choice(positions, size=cells_to_keep, replace=False).tolist())

selected_positions = np.sort(np.asarray(selected_positions, dtype=int))
expected_cells = int(round(adata_backed.n_obs * cell_fraction))
print("Expected cells at exactly 25%:", expected_cells)
print("Selected cells after stratification:", len(selected_positions))


### Load only the selected cells into memory and save the downsized object

adata_quarter = adata_backed[selected_positions, :].to_memory()
adata_backed.file.close()

if adata_quarter.n_obs != len(selected_positions):
    raise ValueError("The downsized object does not contain the expected number of selected cells.")
if adata_quarter.n_vars == 0:
    raise ValueError("The downsized object contains no genes.")

adata_quarter.write_h5ad(output_path, compression="gzip")

input_size_gb = input_path.stat().st_size / 1024**3
output_size_gb = output_path.stat().st_size / 1024**3
print("Saved downsized AnnData:", output_path)
print("Output shape:", adata_quarter.shape)
print(f"Input file size: {input_size_gb:.2f} GB")
print(f"Output file size: {output_size_gb:.2f} GB")
print(f"Output/input disk-size ratio: {output_size_gb / input_size_gb:.3f}")


