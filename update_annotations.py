####!!!!! Dont use this Script !!!! #####


import anndata as ad
import pandas as pd
from pathlib import Path

### Configuration
study = "GSE173682"
path = Path(f"/home/grudmans/EC_ref/{study}_RAW")

full_tumor_path = path / f"{study}_adata_annotated_step3.h5ad"
query_path = path / f"{study}_tumor_all_compartments_scanvi_query.h5ad"
output_path = path / f"{study}_adata_xenium_reference_annotated.h5ad"
metadata_path = path / f"{study}_xenium_reference_cell_metadata.csv"
summary_path = path / f"{study}_xenium_reference_annotation_summary.csv"

annotation_column = "manual_annotation"
alignment_column = "normal_alignment"
malignant_alignment = "Poorly_aligned_to_normal"
final_annotation_column = "xenium_annotation"
malignant_prefix = "Malignant_"

transfer_columns = ["analysis_compartment", "predicted_normal_celltype", "nearest_normal_celltype", "normal_alignment", "median_distance_to_normal", "mean_distance_to_normal", "alignment_threshold", "nearest_normal_celltype_fraction"]


### Load data
if not full_tumor_path.exists():
    raise FileNotFoundError(f"Full tumor object not found: {full_tumor_path}")
if not query_path.exists():
    raise FileNotFoundError(f"Mapped all-compartment tumor query not found: {query_path}")

print("Loading full tumor object:", full_tumor_path)
adata = ad.read_h5ad(full_tumor_path)
print("Loading mapped all-compartment tumor query:", query_path)
tumor_query = ad.read_h5ad(query_path)

if annotation_column not in adata.obs.columns:
    raise KeyError(f"{annotation_column!r} is missing from the full tumor object.")
if "compartment" not in adata.obs.columns:
    raise KeyError("'compartment' is missing from the full tumor object.")
if alignment_column not in tumor_query.obs.columns:
    raise KeyError(f"{alignment_column!r} is missing from the mapped tumor epithelial query.")
if not adata.obs_names.is_unique:
    raise ValueError("The full tumor object contains duplicated cell IDs.")
if not tumor_query.obs_names.is_unique:
    raise ValueError("The mapped tumor epithelial query contains duplicated cell IDs.")


### Match mapped cells to the full tumor object
common_cells = adata.obs_names.intersection(tumor_query.obs_names)
if len(common_cells) == 0:
    raise ValueError("No matching cell IDs were found between the full tumor object and mapped tumor query.")

print("Full tumor cells:", adata.n_obs)
print("Mapped tumor query cells:", tumor_query.n_obs)
print("Matched tumor cells:", len(common_cells))
print("Query cells missing from full object:", tumor_query.n_obs - len(common_cells))

adata.obs["mapped_normal_reference_query"] = adata.obs_names.isin(common_cells)
adata.obs["tumor_epithelial_query"] = adata.obs["mapped_normal_reference_query"] & adata.obs["compartment"].astype(str).str.contains("epithelial",case=False,na=False)


### Transfer normal-reference mapping results
available_transfer_columns = [column for column in transfer_columns if column in tumor_query.obs.columns]
for column in available_transfer_columns:
    values = tumor_query.obs[column]
    if isinstance(values.dtype, pd.CategoricalDtype):
        values = values.astype("string")
    adata.obs[column] = values.reindex(adata.obs_names)


### Keep original labels and relabel only poorly aligned epithelial cells
adata.obs["annotation_original"] = adata.obs[annotation_column].copy()
adata.obs[final_annotation_column] = adata.obs[annotation_column].astype("string")

malignant = adata.obs["tumor_epithelial_query"] & adata.obs[alignment_column].astype("string").eq(malignant_alignment)
already_prefixed = adata.obs[final_annotation_column].str.startswith(malignant_prefix, na=False)
adata.obs.loc[malignant & ~already_prefixed, final_annotation_column] = malignant_prefix + adata.obs.loc[malignant & ~already_prefixed, final_annotation_column]

adata.obs["malignancy"] = "Not_labeled_malignant"
adata.obs.loc[malignant, "malignancy"] = "Malignant"

print("Tumor epithelial cells labeled malignant:", int(malignant.sum()))
print("Mapped epithelial cells retaining their original label:", int((adata.obs["tumor_epithelial_query"] & ~malignant).sum()))
print("All other cells retaining their original label:", int((~adata.obs["tumor_epithelial_query"]).sum()))


### Save results
summary = adata.obs.groupby([final_annotation_column, "malignancy"], observed=True, dropna=False).size().rename("n_cells").reset_index().sort_values("n_cells", ascending=False)
metadata_columns = [column for column in ["sample", "annotation_original", final_annotation_column, "malignancy", "mapped_normal_reference_query", "tumor_epithelial_query"] + available_transfer_columns if column in adata.obs.columns]
metadata = adata.obs[metadata_columns].copy()
metadata.index.name = "cell_id"

adata.write_h5ad(output_path, compression="gzip")
metadata.to_csv(metadata_path)
summary.to_csv(summary_path, index=False)

print("Saved annotated tumor object:", output_path)
print("Saved cell metadata:", metadata_path)
print("Saved annotation summary:", summary_path)
