import anndata as ad
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from pathlib import Path

# total_counts = total UMIs
# n_genes_by_counts = total genes

##!!! Must check and change doublet threshold !!!##
# used a theshold of 1 for GSE251923 (kept everything) and 0.9 for GSE173682

study = "GSE251923"
path = f'/home/grudmans/EC_ref/{study}_RAW'
path_save = f'/home/grudmans/EC_ref/{study}_RAW/step01'

Path(path_save).mkdir(parents=True, exist_ok=True)

### Load data

if study == "GSE251923":
    
    adata_A = sc.read_10x_mtx(f"{path}/case_A",var_names="gene_symbols", make_unique=True,prefix="GSM7990051_case_A_")
    adata_B = sc.read_10x_mtx(f"{path}/case_B",var_names="gene_symbols",make_unique=True,prefix="GSM7990052_case_B_")

    adata = ad.concat([adata_A, adata_B],label="sample",keys=["Case_A", "Case_B"],join="outer",index_unique="_",merge="same")
elif study == "GSE173682":

    adata_3533EL = sc.read_10x_mtx(f"{path}/3533EL",var_names="gene_symbols", make_unique=True,prefix="GSM5276933_")
    adata_3571DL = sc.read_10x_mtx(f"{path}/3571DL",var_names="gene_symbols",make_unique=True,prefix="GSM5276934_")
    adata_36186L = sc.read_10x_mtx(f"{path}/36186L",var_names="gene_symbols", make_unique=True,prefix="GSM5276935_")
    adata_36639L = sc.read_10x_mtx(f"{path}/36639L",var_names="gene_symbols",make_unique=True,prefix="GSM5276936_")
    adata_366C5L = sc.read_10x_mtx(f"{path}/366C5L",var_names="gene_symbols", make_unique=True,prefix="GSM5276937_")
    adata_37EACL = sc.read_10x_mtx(f"{path}/37EACL",var_names="gene_symbols",make_unique=True,prefix="GSM5276938_")

    adata = ad.concat([adata_3533EL,adata_3571DL,adata_36186L,adata_36639L,adata_366C5L,adata_37EACL],label="sample",keys=["3533EL","3571DL","36186L","36639L","366C5L","37EACL"],join="outer",index_unique="_",merge="same")
else:
    raise ValueError(f"Unknown study: {study}")
adata.layers["counts"] = adata.X.copy()

print(adata.shape)
values = adata.X.data if sp.issparse(adata.X) else adata.X.ravel()
if np.any(values < 0):
    raise ValueError("Data contains negative values.")
if not np.allclose(values, np.round(values)):
    raise ValueError("Data contains non-integer values.")
print("Minimum nonzero count:", values.min())
print("Maximum count:", values.max())


### CellBender 

# This data already filtered out empty droplets. CellBender requries them to filter out ambient RNA.

### QC

def is_outlier(adata, metric, nmads,side):
    values = np.asarray(adata.obs[metric])
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    if mad == 0:
        return median
    if side == "lower":
        return median - nmads * mad
    elif side == "upper":
        return median + nmads * mad
    raise ValueError("side must be 'lower' or 'upper'")

adata.var["mt"] = adata.var_names.str.startswith("MT-")
adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
adata.var["hb"] = adata.var_names.str.match(r"^HB(?!P)")

sc.pp.calculate_qc_metrics(adata,qc_vars=["mt", "ribo", "hb"],log1p=True,inplace=True,)
# sc.pl.violin(adata,keys=["n_genes_by_counts","total_counts","pct_counts_mt",], groupby="sample",jitter=0.4,multi_panel=True)
# sc.pl.scatter(adata,x="total_counts",y="n_genes_by_counts",    color="pct_counts_mt")
print(adata.obs.groupby("sample")[["total_counts", "n_genes_by_counts", "pct_counts_mt"]].describe())
qc_plot_metrics = ["log1p_n_genes_by_counts", "log1p_total_counts", "pct_counts_mt"]
qc_plot_sides = ["lower", "lower", "upper"]
qc_nmads = range(1, 6)
qc_thresholds = {metric: {nmad: is_outlier(adata, metric, nmad, side) for nmad in qc_nmads} for metric, side in zip(qc_plot_metrics, qc_plot_sides)}
lower_colors = plt.cm.Blues(np.linspace(0.4, 0.95, 5))
upper_colors = plt.cm.Reds(np.linspace(0.4, 0.95, 5))
fig, axes = plt.subplots(1, 3, figsize=(15,4))
axes[0].hist(adata.obs["log1p_n_genes_by_counts"], bins=300)
axes[0].set_title("log1p genes per cell")
axes[1].hist(adata.obs["log1p_total_counts"], bins=300)
axes[1].set_title("log1p UMIs per cell")
axes[2].hist(adata.obs["pct_counts_mt"], bins=300)
axes[2].set_title("% Mitochondrial RNA per cell")
for ax, metric, side in zip(axes, qc_plot_metrics, qc_plot_sides):
    colors = lower_colors if side == "lower" else upper_colors
    linestyle = ":" if side == "lower" else "--"
    for nmad, color in zip(qc_nmads, colors):
        ax.axvline(qc_thresholds[metric][nmad], color=color, linestyle=linestyle, linewidth=1.5, label=f"{nmad} MAD {side}")
    ax.legend(fontsize=7)
plt.savefig(f"{path_save}/QC_{study}.png", dpi=300, bbox_inches="tight")
plt.close()

## Filters
# GSE173682 filters 4, 3, 5 
# GSE251923 filters 2, 2, 5 

adata = adata[   ##################!!!!!!!!!!!!! MUST CHANGE !!!!!!!!!!!!!!!!!!!!!########################
(adata.obs["log1p_n_genes_by_counts"] >= qc_thresholds["log1p_n_genes_by_counts"][2]) &
(adata.obs["log1p_total_counts"] >= qc_thresholds["log1p_total_counts"][2]) &
(adata.obs["pct_counts_mt"] <= qc_thresholds["pct_counts_mt"][5])].copy()

# Copy after QC
adata_full = adata.copy()

### HVG 
adata_scvi = adata_full.copy()
sc.pp.highly_variable_genes(adata_scvi,n_top_genes=5000,flavor="seurat_v3",layer="counts",batch_key="sample", subset=False)
adata_scvi.var["highly_variable"].value_counts()
adata_scvi = adata_scvi[:, adata_scvi.var["highly_variable"]].copy()

### scVI + SOLO

# Train initial scVI model
scvi.settings.seed = 0
scvi.model.SCVI.setup_anndata(adata_scvi,layer="counts",batch_key="sample")
model = scvi.model.SCVI(adata_scvi,n_layers=2,n_latent=30,gene_likelihood="nb")
model.train(early_stopping=True)

# Store initial latent representation
adata_scvi.obsm["X_scVI_initial"] = model.get_latent_representation()
assert adata_scvi.obs_names.equals(adata_full.obs_names)
adata_full.obsm["X_scVI_initial"] = (adata_scvi.obsm["X_scVI_initial"].copy())

# Remove doublets with SOLO
# Run SOLO separately for each sample
solo_results = []
for sample in adata_scvi.obs["sample"].unique():
    print(f"Running SOLO for {sample}")
    solo = scvi.external.SOLO.from_scvi_model(model,restrict_to_batch=sample)
    solo.train()
    # Hard class labels: singlet or doublet
    solo_predictions = solo.predict(soft=False)
    # Soft class probabilities
    solo_probabilities = solo.predict(soft=True)

    sample_results = pd.DataFrame( {"solo_prediction": solo_predictions.astype(str),
            "solo_doublet_probability": solo_probabilities["doublet"],
            "solo_singlet_probability": solo_probabilities["singlet"],})

    solo_results.append(sample_results)

# Combine SOLO results for each sample
solo_results = pd.concat(solo_results)
# Align predictions with the AnnData barcodes
solo_results = solo_results.reindex(adata_scvi.obs_names)
if solo_results.isna().any().any():
    raise ValueError("Some cells are missing SOLO predictions.")

# Add predictions to adata object
adata_scvi.obs["solo_prediction"] = solo_results["solo_prediction"].astype(str)
adata_scvi.obs["solo_doublet_probability"] = solo_results["solo_doublet_probability"]
adata_scvi.obs["solo_singlet_probability"] = solo_results["solo_singlet_probability"]

# Inspect SOLO probability distributions
print(adata_scvi.obs.groupby("sample", observed=True)["solo_doublet_probability"].describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99]))
print(adata_scvi.obs.loc[adata_scvi.obs["solo_prediction"] == "doublet","solo_doublet_probability",].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]))
samples = adata_scvi.obs["sample"].unique()

thresholds = [0.6, 0.7, 0.8, 0.9]
threshold_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(thresholds)))

# Plot probability distribution
n_samples = len(samples)
if n_samples == 0:
    raise ValueError("No samples are available for the SOLO probability plot.")
ncols = min(3, n_samples)
nrows = int(np.ceil(n_samples / ncols))
fig, axes = plt.subplots(nrows,ncols,figsize=(6 * ncols, 4 * nrows),squeeze=False)
for ax, sample in zip(axes.ravel(), samples):
    probabilities = adata_scvi.obs.loc[adata_scvi.obs["sample"] == sample,"solo_doublet_probability"]
    ax.hist(probabilities, bins=100)
    for threshold, color in zip(thresholds, threshold_colors):
        ax.axvline(threshold, color=color, linestyle="--", linewidth=1.5, label=f"{threshold:g}")
    ax.set_title(sample)
    ax.set_xlabel("SOLO doublet probability")
    ax.set_ylabel("Number of cells")
    ax.legend(title="Threshold", fontsize=7, title_fontsize=8)
for ax in axes.ravel()[n_samples:]:
    ax.set_visible(False)
plt.tight_layout()
plt.savefig(f"{path_save}/SOLO_probabilities_{study}.png",dpi=300,bbox_inches="tight")
plt.close()

# Summarize predictions
solo_summary = pd.crosstab(adata_scvi.obs["sample"],adata_scvi.obs["solo_prediction"])
solo_summary["doublet_fraction"] = (solo_summary.get("doublet", 0) / solo_summary.sum(axis=1))

# Transfer SOLO results to the full-gene AnnData
for col in ["solo_prediction","solo_doublet_probability","solo_singlet_probability"]:
    adata_full.obs[col] = (adata_scvi.obs[col].reindex(adata_full.obs_names))

# Confirm that every cell received a prediction
if adata_full.obs[["solo_prediction","solo_doublet_probability","solo_singlet_probability"]].isna().any().any():
    raise ValueError("Some full-gene cells are missing SOLO results.")

# UMAP containing default singlets and predicted doublets
sc.pp.neighbors(adata_full,use_rep="X_scVI_initial",n_neighbors=15,key_added="initial_neighbors")
sc.tl.umap(adata_full,neighbors_key="initial_neighbors")
sc.pl.umap(adata_full,color=["sample","solo_doublet_probability","total_counts","n_genes_by_counts",],ncols=2,show=False)
plt.savefig(f"{path_save}/SOLO_UMAP_{study}.png",dpi=300,bbox_inches="tight")
plt.close()

# Try different SOLO thresholds
for threshold in thresholds:
    adata_full.obs[f"doublet_{threshold:g}"] = (adata_full.obs["solo_doublet_probability"] >= threshold)
sc.pl.umap(adata_full,color=[f"doublet_{threshold:g}" for threshold in thresholds],ncols=2,show=False)
plt.savefig(f"{path_save}/SOLO_thresholds_{study}.png",dpi=300,bbox_inches="tight")
plt.close()

# Save all QC-passing cells, including predicted doublets
adata_full.write(f"{path}/{study}_adata_full_with_solo_predictions.h5ad")

# Remove doublets based on default parameters
adata_singlets = adata_scvi[adata_scvi.obs["solo_prediction"] == "singlet"].copy()
adata_full_singlets = adata_full[adata_full.obs["solo_prediction"] == "singlet"].copy()

# Use a threshold to remove doublets 
# GSE173682 = 0.8
# GSE251923 = 0.9
threshold = 0.9   ##################!!!!!!!!!!!!! MUST CHANGE !!!!!!!!!!!!!!!!!!!!!########################
adata_singlets = adata_scvi[adata_scvi.obs["solo_doublet_probability"] < threshold].copy()
adata_full_singlets = adata_full[adata_full.obs["solo_doublet_probability"] < threshold].copy()

# Full-gene singlet AnnData
print("Before SOLO:", adata_scvi.n_obs)
print("After SOLO:", adata_singlets.n_obs)
print("Removed:", adata_scvi.n_obs - adata_singlets.n_obs)

# Retrain without doublets 
scvi.model.SCVI.setup_anndata(adata_singlets,layer="counts",batch_key="sample")
final_model = scvi.model.SCVI(adata_singlets,n_layers=2,n_latent=30,gene_likelihood="nb")
final_model.train(early_stopping=True)
ax = final_model.history["elbo_train"].plot(label="train")
if "elbo_validation" in final_model.history:
    final_model.history["elbo_validation"].plot(ax=ax,label="validation")
plt.yscale("log")
plt.legend()
plt.savefig(f"{path_save}/training_final_{study}.png",dpi=300,bbox_inches="tight")
plt.close()

# save model and adata
final_model.save(f"{path}/{study}_scvi_model_singlets",overwrite=True)
adata_singlets.write(f"{path}/{study}_adata_hvg.h5ad")
adata_full_singlets.write(f"{path}/{study}_adata_full.h5ad")
