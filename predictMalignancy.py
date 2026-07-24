import gc
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


### Configuration

study = "GSE251923"
normal_path = Path("/home/grudmans/EC_ref/PMID_39198675/endometriumAtlasV2_cells_with_counts_quarter.h5ad")
tumor_path = Path(f"/home/grudmans/EC_ref/{study}_RAW/{study}_adata_annotated_step3.h5ad")
output_path = Path(f"/home/grudmans/EC_ref/{study}_RAW/{study}_normal_tumor_all_compartments_scanvi.h5ad")
query_output_path = Path(f"/home/grudmans/EC_ref/{study}_RAW/{study}_tumor_all_compartments_scanvi_query.h5ad")
model_path = Path(f"/home/grudmans/EC_ref/{study}_RAW/normal_tumor_all_compartments_scanvi_model")
figure_directory = Path(f"/home/grudmans/EC_ref/{study}_RAW/normalRef")
figure_directory.mkdir(parents=True, exist_ok=True)

n_neighbors = 15
alignment_quantile = 0.99
minimum_cells_per_normal_sample = 100
random_seed = 123


def normalize_compartment(values):
    values = values.astype(str).str.strip().str.lower()
    compartments = pd.Series("Other", index=values.index, dtype="string")
    compartments.loc[values.str.contains("epith", na=False)] = "Epithelial"
    compartments.loc[values.str.contains("immune|lymph|myeloid", regex=True, na=False)] = "Immune"
    compartments.loc[values.str.contains("endothel|arterial|venous|lymphatic", regex=True, na=False)] = "Endothelial"
    compartments.loc[values.str.contains("mesench|strom|fibro|pericyte|perivascular|smooth_muscle|smc", regex=True, na=False)] = "Stromal"
    return compartments


### Load all normal and tumor compartments

adata_normal = sc.read_h5ad(normal_path)
adata_tumor = sc.read_h5ad(tumor_path)

required_normal_columns = ["sample", "celltype", "lineage"]
required_tumor_columns = ["sample", "manual_annotation", "compartment"]
missing_normal_columns = [column for column in required_normal_columns if column not in adata_normal.obs.columns]
missing_tumor_columns = [column for column in required_tumor_columns if column not in adata_tumor.obs.columns]
if missing_normal_columns:
    raise KeyError(f"Normal object is missing obs columns: {missing_normal_columns}")
if missing_tumor_columns:
    raise KeyError(f"Tumor object is missing obs columns: {missing_tumor_columns}")
if "counts" not in adata_normal.layers or "counts" not in adata_tumor.layers:
    raise KeyError("Both normal and tumor objects must contain layers['counts'].")

adata_normal.obs["analysis_compartment"] = normalize_compartment(adata_normal.obs["lineage"])
adata_tumor.obs["analysis_compartment"] = normalize_compartment(adata_tumor.obs["compartment"])

normal_sample_counts = adata_normal.obs["sample"].value_counts()
valid_normal_samples = normal_sample_counts[normal_sample_counts >= minimum_cells_per_normal_sample].index
normal_reference_full = adata_normal[adata_normal.obs["sample"].isin(valid_normal_samples)].copy()
tumor_query_full = adata_tumor.copy()

print("\nNormal compartments:")
print(normal_reference_full.obs["analysis_compartment"].value_counts())
print("\nTumor compartments:")
print(tumor_query_full.obs["analysis_compartment"].value_counts())


### Match genes and select normal-reference HVGs

normal_non_unique_var_names = normal_reference_full.var_names[normal_reference_full.var_names.duplicated(keep=False)].unique().tolist()
tumor_non_unique_var_names = tumor_query_full.var_names[tumor_query_full.var_names.duplicated(keep=False)].unique().tolist()
print("\nNon-unique normal var names:", normal_non_unique_var_names)
print("Non-unique tumor var names:", tumor_non_unique_var_names)

normal_reference_full.var_names_make_unique()
tumor_query_full.var_names_make_unique()
shared_genes = normal_reference_full.var_names.intersection(tumor_query_full.var_names)
normal_reference_full = normal_reference_full[:, shared_genes].copy()
tumor_query_full = tumor_query_full[:, shared_genes].copy()

scvi.settings.seed = random_seed
sc.pp.filter_genes(normal_reference_full,min_cells=10)
hvg_completed = False
for hvg_span in [0.5, 0.8, 1.0]:
    try:
        sc.pp.highly_variable_genes(normal_reference_full,layer="counts",flavor="seurat_v3",n_top_genes=4000,batch_key="sample",span=hvg_span,subset=False)
        print("Seurat-v3 HVG selection completed with LOESS span:", hvg_span)
        hvg_completed = True
        break
    except ValueError as error:
        if "reciprocal condition number" not in str(error):
            raise
        print("Seurat-v3 HVG LOESS failed with span", hvg_span, "and will retry with a wider span:", error)
if not hvg_completed:
    raise RuntimeError("Seurat-v3 HVG selection failed for all tested LOESS spans.")
reference_genes = normal_reference_full.var_names[normal_reference_full.var["highly_variable"]].copy()
normal_reference = normal_reference_full[:, reference_genes].copy()
del normal_reference_full
gc.collect()


### Train the all-compartment normal scVI and scANVI reference

scvi.model.SCVI.setup_anndata(normal_reference,layer="counts",batch_key="sample")
reference_model = scvi.model.SCVI(normal_reference,n_latent=30,gene_likelihood="nb")
reference_model.train()
normal_reference.obsm["X_scVI"] = reference_model.get_latent_representation()

scanvi_reference = scvi.model.SCANVI.from_scvi_model(reference_model,labels_key="celltype",unlabeled_category="Unknown")
scanvi_reference.train()
normal_reference.obsm["X_scANVI"] = scanvi_reference.get_latent_representation()


### Map every tumor compartment into the normal reference

tumor_query = tumor_query_full.copy()
tumor_query.obs["celltype"] = "Unknown"
scvi.model.SCANVI.prepare_query_anndata(tumor_query,scanvi_reference,inplace=True)
query_model = scvi.model.SCANVI.load_query_data(tumor_query,scanvi_reference)
query_model.train()
tumor_query.obsm["X_scANVI"] = query_model.get_latent_representation()
tumor_query.obs["predicted_normal_celltype"] = query_model.predict()

normal_latent = normal_reference.obsm["X_scANVI"]
tumor_latent = tumor_query.obsm["X_scANVI"]


### Calculate compartment-specific normal alignment

normal_reference.obs["median_distance_to_normal"] = np.nan
normal_reference.obs["alignment_threshold"] = np.nan
tumor_query.obs["median_distance_to_normal"] = np.nan
tumor_query.obs["mean_distance_to_normal"] = np.nan
tumor_query.obs["alignment_threshold"] = np.nan
tumor_query.obs["nearest_normal_celltype"] = pd.Series(pd.NA, index=tumor_query.obs_names, dtype="string")
tumor_query.obs["nearest_normal_celltype_fraction"] = np.nan
tumor_query.obs["normal_alignment"] = pd.Series("No_normal_compartment_reference", index=tumor_query.obs_names, dtype="string")

alignment_thresholds = {}
shared_compartments = sorted(set(normal_reference.obs["analysis_compartment"]) & set(tumor_query.obs["analysis_compartment"]))
for compartment in shared_compartments:
    normal_mask = normal_reference.obs["analysis_compartment"].eq(compartment).to_numpy()
    tumor_mask = tumor_query.obs["analysis_compartment"].eq(compartment).to_numpy()
    normal_positions = np.flatnonzero(normal_mask)
    tumor_positions = np.flatnonzero(tumor_mask)
    if len(normal_positions) < 2 or len(tumor_positions) == 0:
        print(f"Skipping {compartment}: normal cells={len(normal_positions)}, tumor cells={len(tumor_positions)}")
        continue

    normal_compartment_latent = normal_latent[normal_positions]
    tumor_compartment_latent = tumor_latent[tumor_positions]
    query_neighbors = min(n_neighbors, len(normal_positions))
    self_neighbors = min(n_neighbors + 1, len(normal_positions))

    normal_knn = NearestNeighbors(n_neighbors=query_neighbors,metric="euclidean",n_jobs=-1)
    normal_knn.fit(normal_compartment_latent)
    tumor_distances, tumor_neighbor_indices = normal_knn.kneighbors(tumor_compartment_latent)

    normal_self_knn = NearestNeighbors(n_neighbors=self_neighbors,metric="euclidean",n_jobs=-1)
    normal_self_knn.fit(normal_compartment_latent)
    normal_distances, _ = normal_self_knn.kneighbors(normal_compartment_latent)
    normal_distances = normal_distances[:, 1:]

    normal_median_distance = np.median(normal_distances, axis=1)
    tumor_median_distance = np.median(tumor_distances, axis=1)
    tumor_mean_distance = np.mean(tumor_distances, axis=1)
    alignment_threshold = np.quantile(normal_median_distance, alignment_quantile)
    alignment_thresholds[compartment] = alignment_threshold

    normal_names = normal_reference.obs_names[normal_positions]
    tumor_names = tumor_query.obs_names[tumor_positions]
    normal_reference.obs.loc[normal_names, "median_distance_to_normal"] = normal_median_distance
    normal_reference.obs.loc[normal_names, "alignment_threshold"] = alignment_threshold
    tumor_query.obs.loc[tumor_names, "median_distance_to_normal"] = tumor_median_distance
    tumor_query.obs.loc[tumor_names, "mean_distance_to_normal"] = tumor_mean_distance
    tumor_query.obs.loc[tumor_names, "alignment_threshold"] = alignment_threshold
    tumor_query.obs.loc[tumor_names, "normal_alignment"] = np.where(tumor_median_distance <= alignment_threshold, "Aligned_to_normal", "Poorly_aligned_to_normal")

    normal_celltypes = normal_reference.obs.iloc[normal_positions]["celltype"].astype(str).to_numpy()
    nearest_labels = []
    nearest_fractions = []
    for labels in normal_celltypes[tumor_neighbor_indices]:
        values, counts = np.unique(labels, return_counts=True)
        best = np.argmax(counts)
        nearest_labels.append(values[best])
        nearest_fractions.append(counts[best] / len(labels))
    tumor_query.obs.loc[tumor_names, "nearest_normal_celltype"] = nearest_labels
    tumor_query.obs.loc[tumor_names, "nearest_normal_celltype_fraction"] = nearest_fractions

    print(f"\n{compartment} 99th-percentile threshold:", alignment_threshold)

tumor_query.obs["normal_alignment"] = pd.Categorical(tumor_query.obs["normal_alignment"], categories=["Aligned_to_normal", "Poorly_aligned_to_normal", "No_normal_compartment_reference"])

alignment_summary = pd.crosstab(tumor_query.obs["analysis_compartment"],tumor_query.obs["normal_alignment"],normalize="index").round(3)
alignment_counts = pd.crosstab(tumor_query.obs["analysis_compartment"],tumor_query.obs["normal_alignment"])
print("\nTumor alignment counts by compartment:")
print(alignment_counts)
print("\nTumor alignment proportions by compartment:")
print(alignment_summary)
if {"Immune", "Epithelial"}.issubset(alignment_summary.index) and "Poorly_aligned_to_normal" in alignment_summary.columns:
    immune_poor_fraction = alignment_summary.loc["Immune", "Poorly_aligned_to_normal"]
    epithelial_poor_fraction = alignment_summary.loc["Epithelial", "Poorly_aligned_to_normal"]
    print("\nImmune poorly aligned fraction:", immune_poor_fraction)
    print("Epithelial poorly aligned fraction:", epithelial_poor_fraction)
    if immune_poor_fraction >= epithelial_poor_fraction:
        print("WARNING: Immune cells do not align better than epithelial cells; inspect batch correction, compartment labels, and the normal immune reference.")
alignment_counts_long = alignment_counts.rename_axis(index="analysis_compartment",columns="normal_alignment").stack().rename("n_cells").reset_index()
alignment_proportions_long = alignment_summary.rename_axis(index="analysis_compartment",columns="normal_alignment").stack().rename("fraction_of_compartment").reset_index()
alignment_table = alignment_counts_long.merge(alignment_proportions_long,on=["analysis_compartment", "normal_alignment"],how="left")
alignment_table.to_csv(figure_directory / f"{study}_alignment_by_compartment.csv",index=False)


### Create a shared all-compartment UMAP

normal_obs = normal_reference.obs.copy()
tumor_obs = tumor_query.obs.copy()
normal_obs.index = "Normal_" + normal_obs.index.astype(str)
normal_obs["source"] = "Normal"
normal_obs["alignment_display"] = "Normal_reference"
normal_obs["compartment_display"] = normal_obs["analysis_compartment"].astype(str)
tumor_obs.index = "Tumor_" + tumor_obs.index.astype(str)
tumor_obs["source"] = "Tumor"
tumor_obs["alignment_display"] = tumor_obs["normal_alignment"].astype(str)
tumor_obs["compartment_display"] = tumor_obs["analysis_compartment"].astype(str)

combined_latent = np.vstack([normal_latent, tumor_latent])
combined_obs = pd.concat([normal_obs, tumor_obs], axis=0)
adata_latent = ad.AnnData(X=combined_latent,obs=combined_obs)
adata_latent.obsm["X_scANVI"] = combined_latent
sc.pp.neighbors(adata_latent,use_rep="X_scANVI",n_neighbors=15)
sc.tl.umap(adata_latent,min_dist=0.3,random_state=random_seed)

normal_latent_only = adata_latent[adata_latent.obs["source"] == "Normal"].copy()
tumor_latent_only = adata_latent[adata_latent.obs["source"] == "Tumor"].copy()
xy = adata_latent.obsm["X_umap"]
xlim = (xy[:, 0].min(), xy[:, 0].max())
ylim = (xy[:, 1].min(), xy[:, 1].max())

fig, axes = plt.subplots(1,3,figsize=(36,9),sharex=True,sharey=True)
sc.pl.umap(adata_latent,color="source",size=4,frameon=False,legend_loc="right margin",legend_fontsize=10,ax=axes[0],show=False,title="Normal and tumor cells")
sc.pl.umap(adata_latent,color="alignment_display",size=4,frameon=False,legend_loc="right margin",legend_fontsize=9,ax=axes[1],show=False,title="Compartment-specific normal alignment")
sc.pl.umap(adata_latent,color="compartment_display",size=4,frameon=False,legend_loc="right margin",legend_fontsize=10,ax=axes[2],show=False,title="Normal and tumor compartments")
for axis in axes:
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
    axis.set_xlabel("UMAP1")
    axis.set_ylabel("UMAP2")
    axis.set_aspect("equal", adjustable="box")
fig.subplots_adjust(wspace=0.65)
fig.savefig(figure_directory / "normal_tumor_all_compartments_three_panel.png",dpi=300,bbox_inches="tight")
plt.close(fig)


### Plot normal labels and tumor predictions

fig, axes = plt.subplots(1,2,figsize=(30,9),sharex=True,sharey=True)
sc.pl.umap(normal_latent_only,color="celltype",size=5,frameon=False,legend_loc="right margin",legend_fontsize=7,ax=axes[0],show=False,title="Normal reference cell types")
sc.pl.umap(tumor_latent_only,color="predicted_normal_celltype",size=7,frameon=False,legend_loc="right margin",legend_fontsize=7,ax=axes[1],show=False,title="Tumor predicted normal identity")
for axis in axes:
    axis.set_xlim(xlim)
    axis.set_ylim(ylim)
fig.subplots_adjust(wspace=0.9)
fig.savefig(figure_directory / "normal_celltypes_vs_tumor_predictions.png",dpi=300,bbox_inches="tight")
plt.close(fig)


### Plot distance distributions separately for each compartment

compartments_with_thresholds = list(alignment_thresholds)
ncols = 2
nrows = int(np.ceil(len(compartments_with_thresholds) / ncols))
fig, axes = plt.subplots(nrows,ncols,figsize=(14,5 * nrows),squeeze=False)
for axis, compartment in zip(axes.ravel(), compartments_with_thresholds):
    normal_values = normal_reference.obs.loc[normal_reference.obs["analysis_compartment"].eq(compartment), "median_distance_to_normal"].dropna()
    tumor_values = tumor_query.obs.loc[tumor_query.obs["analysis_compartment"].eq(compartment), "median_distance_to_normal"].dropna()
    axis.hist(normal_values,bins=60,density=True,alpha=0.55,label="Normal")
    axis.hist(tumor_values,bins=60,density=True,alpha=0.55,label="Tumor")
    axis.axvline(alignment_thresholds[compartment],linestyle="--",linewidth=1.5,label="Normal 99th percentile")
    axis.set_title(compartment)
    axis.set_xlabel(f"Median distance to {n_neighbors} nearest normal cells")
    axis.set_ylabel("Density")
    axis.legend()
for axis in axes.ravel()[len(compartments_with_thresholds):]:
    axis.set_visible(False)
fig.tight_layout()
fig.savefig(figure_directory / "normal_tumor_distance_distributions_by_compartment.png",dpi=300,bbox_inches="tight")
plt.close(fig)


### Compare existing tumor labels with normal-reference identities

comparison = pd.crosstab(tumor_query.obs["manual_annotation"],tumor_query.obs["predicted_normal_celltype"],normalize="index")
nearest_comparison = pd.crosstab(tumor_query.obs["manual_annotation"],tumor_query.obs["nearest_normal_celltype"],normalize="index")
print("\nPredicted normal identity by manual annotation:")
print(comparison.round(3))
print("\nNearest-normal-neighbor identity by manual annotation:")
print(nearest_comparison.round(3))


### Save mapped cells, shared UMAP, and trained models

tumor_query.write_h5ad(query_output_path,compression="gzip")
for column in adata_latent.obs.columns:
    if adata_latent.obs[column].dtype == object:
        adata_latent.obs[column] = adata_latent.obs[column].astype(str)
adata_latent.write_h5ad(output_path,compression="gzip")
reference_model.save(Path(f"{model_path}_scvi"),overwrite=True)
scanvi_reference.save(Path(f"{model_path}_scanvi_reference"),overwrite=True)
query_model.save(Path(f"{model_path}_scanvi_query"),overwrite=True)

print("\nSaved outputs:")
print(query_output_path)
print(output_path)
print(figure_directory)
