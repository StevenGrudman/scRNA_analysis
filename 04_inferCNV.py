"""Patient-wise inferCNV analysis of all epithelial cells using same-sample non-epithelial cells as the reference."""

from pathlib import Path

import anndata as ad
import infercnvpy as cnv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


STUDY = "GSE251923"
BASE = Path(f"/home/grudmans/EC_ref/{STUDY}_RAW")
OUTPUT_DIR = BASE / "step04"
FULL_PATH = BASE / f"{STUDY}_adata_annotated_step3.h5ad"
GTF_PATH = "/home/grudmans/reference/gencode.v50.basic.annotation.gtf.gz"
MIN_REFERENCE_CELLS = 50
MIN_EPITHELIAL_CELLS = 50
WINDOW_SIZE = 100
STEP = 10


def prepare_expression(adata: ad.AnnData) -> None:
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    elif "log1p" in adata.uns:
        raise ValueError("No counts layer was found and X appears log-transformed; inferCNV should start from counts.")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)


def matrix_to_array(matrix) -> np.ndarray:
    return matrix.toarray() if sparse.issparse(matrix) else np.asarray(matrix)


def per_cell_cnv_burden(matrix) -> np.ndarray:
    return np.asarray(np.abs(matrix).mean(axis=1)).ravel()


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
adata = ad.read_h5ad(FULL_PATH)
cnv.io.genomic_position_from_gtf(GTF_PATH, adata=adata, gtf_gene_id="gene_id", adata_gene_id="gene_ids", inplace=True)
for column in adata.var.columns:
    if pd.api.types.is_string_dtype(adata.var[column].dtype):
        adata.var[column] = pd.Series(np.asarray(adata.var[column], dtype=object), index=adata.var.index, dtype=object)
adata.var_names = pd.Index(np.asarray(adata.var_names, dtype=object), dtype=object)
adata.var.index = pd.Index(np.asarray(adata.var.index, dtype=object), dtype=object)

cell_tables = []
sample_summaries = []
profile_rows = []
profile_labels = []
chromosome_positions = None

for sample in adata.obs["sample"].astype(str).unique():
    sample_mask = adata.obs["sample"].astype(str).eq(sample)
    epithelial_mask = adata.obs["compartment"].astype(str).eq("Epithelial")
    n_epithelial = int((sample_mask & epithelial_mask).sum())
    n_reference = int((sample_mask & ~epithelial_mask).sum())
    if n_epithelial < MIN_EPITHELIAL_CELLS or n_reference < MIN_REFERENCE_CELLS:
        print(f"Skipping {sample}: epithelial={n_epithelial}, non-epithelial reference={n_reference}")
        continue
    print(f"Running {sample}: {n_epithelial} epithelial cells and {n_reference} non-epithelial reference cells")
    sample_adata = adata[sample_mask].copy()
    sample_adata.obs["infercnv_group"] = pd.Categorical(np.where(sample_adata.obs["compartment"].astype(str).eq("Epithelial"), "epithelial", "reference"), categories=["reference", "epithelial"])
    prepare_expression(sample_adata)
    cnv.tl.infercnv(sample_adata, reference_key="infercnv_group", reference_cat=["reference"], window_size=WINDOW_SIZE, step=STEP)
    sample_adata.obs["cnv_burden"] = per_cell_cnv_burden(sample_adata.obsm["X_cnv"])
    cell_tables.append(pd.DataFrame({"cell_id": sample_adata.obs_names, "sample": sample, "manual_annotation": sample_adata.obs["manual_annotation"].astype(str).to_numpy(), "compartment": sample_adata.obs["compartment"].astype(str).to_numpy(), "cnv_burden": sample_adata.obs["cnv_burden"].to_numpy()}))
    epithelial_scores = sample_adata.obs.loc[sample_adata.obs["compartment"].astype(str).eq("Epithelial"), "cnv_burden"]
    reference_scores = sample_adata.obs.loc[sample_adata.obs["compartment"].astype(str).ne("Epithelial"), "cnv_burden"]
    sample_summaries.append({"sample": sample, "n_epithelial": len(epithelial_scores), "n_non_epithelial_reference": len(reference_scores), "median_epithelial_cnv_burden": epithelial_scores.median(), "median_reference_cnv_burden": reference_scores.median(), "median_burden_difference": epithelial_scores.median() - reference_scores.median()})
    epithelial = sample_adata[sample_adata.obs["compartment"].astype(str).eq("Epithelial")]
    for annotation in sorted(epithelial.obs["manual_annotation"].astype(str).unique()):
        group_matrix = epithelial.obsm["X_cnv"][epithelial.obs["manual_annotation"].astype(str).eq(annotation).to_numpy()]
        profile_rows.append(np.median(matrix_to_array(group_matrix), axis=0))
        profile_labels.append(f"{sample} | {annotation}")
    if chromosome_positions is None:
        chromosome_positions = sample_adata.uns["cnv"]["chr_pos"]

scores = pd.concat(cell_tables, ignore_index=True)
summaries = pd.DataFrame(sample_summaries)
annotation_summaries = scores.groupby(["sample", "manual_annotation", "compartment"], observed=True)["cnv_burden"].agg(n_cells="size", median_cnv_burden="median", mean_cnv_burden="mean").reset_index()
with pd.ExcelWriter(OUTPUT_DIR / f"{STUDY}_patientwise_infercnv_results.xlsx") as writer:
    summaries.to_excel(writer, sheet_name="Sample summary", index=False)
    annotation_summaries.to_excel(writer, sheet_name="Annotation summary", index=False)
    scores.to_excel(writer, sheet_name="Cell scores", index=False)

samples = summaries["sample"].astype(str).tolist()
epithelial_values = [scores.loc[(scores["sample"].astype(str).eq(sample)) & (scores["compartment"].eq("Epithelial")), "cnv_burden"].to_numpy() for sample in samples]
reference_values = [scores.loc[(scores["sample"].astype(str).eq(sample)) & (scores["compartment"].ne("Epithelial")), "cnv_burden"].to_numpy() for sample in samples]
centers = np.arange(len(samples)) * 3
fig, ax = plt.subplots(figsize=(10, 6))
epithelial_boxes = ax.boxplot(epithelial_values, positions=centers - 0.45, widths=0.8, showfliers=False, patch_artist=True)
reference_boxes = ax.boxplot(reference_values, positions=centers + 0.45, widths=0.8, showfliers=False, patch_artist=True)
for box in epithelial_boxes["boxes"]:
    box.set_facecolor("#E76F51")
for box in reference_boxes["boxes"]:
    box.set_facecolor("#4C78A8")
ax.set_xticks(centers, samples)
ax.set_xlabel("Sample")
ax.set_ylabel("Per-cell CNV burden")
ax.set_title("Epithelial CNV burden compared with same-sample reference")
ax.legend([epithelial_boxes["boxes"][0], reference_boxes["boxes"][0]], ["Epithelial", "Non-epithelial reference"], frameon=False)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f"{STUDY}_epithelial_cnv_burden.png", dpi=300, bbox_inches="tight")
plt.close(fig)

profiles = np.vstack(profile_rows)
limit = np.percentile(np.abs(profiles), 99)
fig, ax = plt.subplots(figsize=(18, max(5, 0.45 * len(profile_labels))))
image = ax.imshow(profiles, aspect="auto", interpolation="nearest", cmap="bwr", vmin=-limit, vmax=limit)
chromosomes = list(chromosome_positions)
starts = np.array(list(chromosome_positions.values()))
ends = np.r_[starts[1:], profiles.shape[1]]
centers = (starts + ends) / 2
for start in starts[1:]:
    ax.axvline(start - 0.5, color="black", linewidth=0.5)
ax.set_xticks(centers, chromosomes, rotation=90)
ax.set_yticks(np.arange(len(profile_labels)), profile_labels)
ax.set_xlabel("Chromosome")
ax.set_title("Median epithelial inferred-CNV profiles by sample and annotation")
fig.colorbar(image, ax=ax, label="Inferred CNV")
fig.tight_layout()
fig.savefig(OUTPUT_DIR / f"{STUDY}_epithelial_cnv_profiles.png", dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved one workbook and two figures to {OUTPUT_DIR}")
