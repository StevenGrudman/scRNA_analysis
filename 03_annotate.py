import anndata as ad
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import celltypist
from pathlib import Path
from celltypist import models

# Input resolution
# GSE251923 = 1.0
# GSE173682 = 0.8

res=0.8
study = "GSE173682"
path = f'/home/grudmans/EC_ref/{study}_RAW'
path_save = f'/home/grudmans/EC_ref/{study}_RAW/step03'

Path(path_save).mkdir(parents=True, exist_ok=True)

### Load Data
adata_hvg = ad.read_h5ad(f"{path}/{study}_adata_hvg.h5ad")
adata_full = ad.read_h5ad(f"{path}/{study}_adata_full.h5ad")
model = scvi.model.SCVI.load(f"{path}/{study}_scvi_model_singlets",adata=adata_hvg)

# Get latent representations
adata_hvg.obsm["X_scVI"] = model.get_latent_representation()
if not adata_hvg.obs_names.equals(adata_full.obs_names):
    raise ValueError("The HVG and full-gene objects have different cell IDs or cell ordering.")
adata_full.obsm["X_scVI"] = adata_hvg.obsm["X_scVI"].copy()

### annotate

# cluster
sc.pp.neighbors(adata_full,use_rep="X_scVI",n_neighbors=15,key_added="scvi_neighbors")
sc.tl.leiden(adata_full,resolution=res,neighbors_key="scvi_neighbors",key_added="leiden",flavor="igraph",random_state=0)
sc.tl.umap(adata_full,neighbors_key="scvi_neighbors",random_state=0)

# Normalize and log transform cells
adata_celltypist = adata_full.copy()
celltypist_values = adata_celltypist.X.data if sp.issparse(adata_celltypist.X) else np.asarray(adata_celltypist.X).ravel()
if np.any(celltypist_values < 0) or not np.allclose(celltypist_values, np.round(celltypist_values)):
    raise ValueError("adata_full.X must contain nonnegative raw counts before CellTypist normalization.")
sc.pp.normalize_total(adata_celltypist,target_sum=1e4)
sc.pp.log1p(adata_celltypist)

## See which celltypist dictionaries are avalible
# models.download_models(force_update=False)
# models.models_description()
# print(models.get_all_models())

# Use those models to label cells
model = models.Model.load(model="Immune_All_Low.pkl")
predictions = celltypist.annotate(adata_celltypist,model=model,majority_voting=True)
adata_full.obs["celltypist_immune"] = predictions.predicted_labels["majority_voting"]
model = models.Model.load(model="Human_Endometrium_Atlas.pkl")
predictions = celltypist.annotate(adata_celltypist,model=model,majority_voting=True)
adata_full.obs["celltypist_endometrium"] = predictions.predicted_labels["majority_voting"]

sc.pl.umap(adata_full,color=["celltypist_immune", "celltypist_endometrium","leiden"],legend_loc="on data",show=False)
plt.savefig(f"{path_save}/celltypist_{study}.png", dpi=300, bbox_inches="tight")
plt.close()

# Hierarchical consensus:
## Let the endometrium model determine whether the cluster is immune, epithelial, stromal, endothelial, or smooth muscle/perivascular.
## For immune clusters, use the immune model’s detailed label.
## For nonimmune clusters, use the endometrium model’s detailed label.
## Flag clusters where the models disagree at the broad-compartment level.
def shannon_entropy(labels):
    counts = labels.value_counts()
    proportions = counts / counts.sum()
    entropy = -(proportions * np.log2(proportions)).sum()
    return 0.0 if np.isclose(entropy, 0) else entropy

def immune_model_compartment(label):
    label = str(label)
    if label == "Epithelial cells": return "Epithelial"
    if label == "Endothelial cells": return "Endothelial"
    if label == "Fibroblasts": return "Stromal"
    return "Immune"

def endometrium_model_compartment(label):
    label = str(label)
    if label in {"Immune_Lymphoid", "Immune_Myeloid"}: return "Immune"
    if label in {"Arterial", "Venous", "Lymphatic"}: return "Endothelial"
    if label in {"ePV_1a", "ePV_1b", "ePV_2", "uSMCs"}: return "Perivascular/SMC"
    if label in {"eStromal", "eStromal_cycling", "eStromal_MMPs", "dStromal_early", "dStromal_mid", "dStromal_late", "Fibroblast_basalis", "HOXA13"}: return "Stromal"
    if label in {"Ciliated", "Glandular", "Glandular_secretory", "Glandular_secretory_FGF7", "SOX9_luminal", "SOX9_functionalis_I", "eHormones", "dHormones"}: return "Epithelial"
    if label == "Cycling": return "Cycling"
    return "Unknown"

unmapped_endometrium_labels = sorted(label for label in adata_full.obs["celltypist_endometrium"].dropna().astype(str).unique() if endometrium_model_compartment(label) == "Unknown")
if unmapped_endometrium_labels:
    print("WARNING: Endometrium CellTypist labels missing from the compartment mapping:", unmapped_endometrium_labels)

immune_summary_rows = []
endometrium_summary_rows = []
consensus_summary_rows = []
label_percentage_rows = []
minimum_label_fraction = 0.01
clusters = sorted(adata_full.obs["leiden"].astype(str).unique(), key=int)
for cluster in clusters:
    cluster_mask = adata_full.obs["leiden"].astype(str) == cluster

    immune_labels = adata_full.obs.loc[cluster_mask, "celltypist_immune"].dropna()
    endometrium_labels = adata_full.obs.loc[cluster_mask, "celltypist_endometrium"].dropna()

    immune_counts = immune_labels.value_counts()
    endometrium_counts = endometrium_labels.value_counts()

    if immune_counts.empty:
        raise ValueError(f"Cluster {cluster} has no Immune CellTypist predictions.")
    if endometrium_counts.empty:
        raise ValueError(f"Cluster {cluster} has no Endometrium CellTypist predictions.")

    for label, count in immune_counts.items():
        label_fraction = count / immune_counts.sum()
        if label_fraction >= minimum_label_fraction:
            label_percentage_rows.append({"Cluster": cluster, "Model": "Immune", "Label": label, "N_cells_with_label": int(count), "Percentage_of_cluster": round(label_fraction * 100, 2)})
    for label, count in endometrium_counts.items():
        label_fraction = count / endometrium_counts.sum()
        if label_fraction >= minimum_label_fraction:
            label_percentage_rows.append({"Cluster": cluster, "Model": "Endometrium", "Label": label, "N_cells_with_label": int(count), "Percentage_of_cluster": round(label_fraction * 100, 2)})

    immune_top_label = immune_counts.index[0]
    immune_top_percent = immune_counts.iloc[0] / immune_counts.sum() * 100

    endometrium_top_label = endometrium_counts.index[0]
    endometrium_top_percent = endometrium_counts.iloc[0] / endometrium_counts.sum() * 100

    immune_compartment = immune_model_compartment(immune_top_label)
    endometrium_compartment = endometrium_model_compartment(endometrium_top_label)
    immune_entropy = shannon_entropy(immune_labels)
    endometrium_entropy = shannon_entropy(endometrium_labels)
    immune_summary_rows.append({"Cluster": cluster, "N_cells": int(cluster_mask.sum()), "Immune_model_label": immune_top_label, "Immune_model_percent": round(immune_top_percent, 1), "Immune_model_entropy": round(immune_entropy, 3), "Immune_model_compartment": immune_compartment})
    endometrium_summary_rows.append({"Cluster": cluster, "N_cells": int(cluster_mask.sum()), "Endometrium_model_label": endometrium_top_label, "Endometrium_model_percent": round(endometrium_top_percent, 1), "Endometrium_model_entropy": round(endometrium_entropy, 3), "Endometrium_model_compartment": endometrium_compartment})
    consensus_compartment = endometrium_compartment

    if consensus_compartment == "Immune":
        consensus_label = immune_top_label
        consensus_percent = immune_top_percent
        consensus_entropy = immune_entropy
        label_source = "Immune model"
    else:
        consensus_label = endometrium_top_label
        consensus_percent = endometrium_top_percent
        consensus_entropy = endometrium_entropy
        label_source = "Endometrium model"

    broad_agreement = immune_compartment == endometrium_compartment

    if consensus_percent >= 80 and consensus_entropy < 1: confidence = "High"
    elif consensus_percent >= 50 and consensus_entropy < 2: confidence = "Medium"
    else: confidence = "Low"

    review_flag = "" 
    if not broad_agreement or confidence == "Low" or consensus_entropy > 2:
        review_flag = "Review"
    consensus_summary_rows.append({"Cluster": cluster, "N_cells": int(cluster_mask.sum()), "Consensus_compartment": consensus_compartment, "Consensus_label": consensus_label, "Consensus_percent": round(consensus_percent, 1), "Consensus_entropy": round(consensus_entropy, 3), "Confidence": confidence, "Label_source": label_source, "Immune_model_label": immune_top_label, "Endometrium_model_label": endometrium_top_label, "Broad_agreement": broad_agreement, "Review_flag": review_flag})

immune_summary = pd.DataFrame(immune_summary_rows).sort_values("Cluster", key=lambda column: column.astype(int))
endometrium_summary = pd.DataFrame(endometrium_summary_rows).sort_values("Cluster", key=lambda column: column.astype(int))
consensus_summary = pd.DataFrame(consensus_summary_rows)
consensus_summary = consensus_summary.sort_values("Cluster", key=lambda column: column.astype(int))
label_percentages = pd.DataFrame(label_percentage_rows, columns=["Cluster", "Model", "Label", "N_cells_with_label", "Percentage_of_cluster"]).sort_values(["Cluster", "Model", "Percentage_of_cluster"], ascending=[True, True, False], key=lambda column: column.astype(int) if column.name == "Cluster" else column)
immune_label_percentages = label_percentages.loc[label_percentages["Model"] == "Immune"].drop(columns="Model")
endometrium_label_percentages = label_percentages.loc[label_percentages["Model"] == "Endometrium"].drop(columns="Model")
print("\nIMMUNE MODEL RESULTS")
print(immune_summary.to_string(index=False))
print("\nENDOMETRIUM MODEL RESULTS")
print(endometrium_summary.to_string(index=False))
print("\nCONSENSUS RESULTS")
print(consensus_summary.to_string(index=False))
consensus_summary.to_csv(f"{path_save}/{study}_celltypist_consensus_summary.csv", index=False)
label_percentage_path = f"{path_save}/{study}_celltypist_cluster_label_percentages.xlsx"
with pd.ExcelWriter(label_percentage_path) as writer:
    immune_label_percentages.to_excel(writer, sheet_name="Immune", index=False)
    endometrium_label_percentages.to_excel(writer, sheet_name="Endometrium", index=False)
print("Saved cluster label percentages of at least 10%:", label_percentage_path)
sc.tl.rank_genes_groups(adata_celltypist,groupby="leiden", method="wilcoxon")
sc.tl.dendrogram(adata_celltypist,groupby="leiden",use_rep="X_scVI")

# Print the top 8 marker genes for the clusters flagged for review
review_clusters = consensus_summary.loc[consensus_summary["Review_flag"] == "Review", "Cluster"].astype(str).tolist()
print("\n" + "=" * 80)
print("TOP 8 MARKER GENES FOR CLUSTERS FLAGGED FOR REVIEW")
print("=" * 80)
review_markers = []
for cluster in review_clusters:
    cluster_markers = sc.get.rank_genes_groups_df(adata_celltypist, group=cluster).head(8)
    cluster_markers.insert(0, "Cluster", cluster)
    review_markers.append(cluster_markers)
    print(f"\nCluster {cluster}")
    print("-" * 40)
    print(cluster_markers[["names", "logfoldchanges", "pvals_adj"]].to_string(index=False))
if review_markers:
    review_markers = pd.concat(review_markers, ignore_index=True)
    review_markers.to_csv(f"{path_save}/{study}_review_cluster_top8_markers.csv", index=False)
else:
    print("No clusters were flagged for review; no review-marker CSV was created.")

# Dot plot of the 5 most differentially expressed genes per cluster
sc.pl.rank_genes_groups_dotplot(adata_celltypist,n_genes=5,groupby="leiden",standard_scale="var",show=False)
plt.savefig(f"{path_save}/clusterTopGenes_{study}.png", dpi=300, bbox_inches="tight")
plt.close()






### GSE173682   GSE173682   GSE173682   GSE173682   GSE173682   GSE173682   GSE173682   GSE173682   GSE173682
# ####################!!!!!!!!!!!!!!!!!!!! Must change annotations for particular study !!!!!!!!!!!!!########################################################

# def save_marker_dotplot(adata, genes, output_file):
#     available_genes = [gene for gene in genes if gene in adata.var_names]
#     missing_genes = [gene for gene in genes if gene not in adata.var_names]
#     if missing_genes:
#         print("Skipping genes absent from the dataset for", output_file, ":", missing_genes)
#     if not available_genes:
#         print("No requested genes are available; skipping", output_file)
#         return
#     sc.pl.dotplot(adata,available_genes,groupby="leiden",show=False)
#     plt.savefig(output_file,dpi=300,bbox_inches="tight")
#     plt.close()

# # =============================================================================
# # Cluster 0
# # Determine whether this is CD4, CD8, Trm, or NK contamination
# # =============================================================================
# cluster0_genes = ["CD3D", "CD3E", "TRAC", "CD4", "IL7R", "LTB", "CD8A", "CD8B", "NKG7", "GNLY", "CCL5", "GZMK", "GZMB"]
# save_marker_dotplot(adata_celltypist,cluster0_genes,f"{path_save}/cluster0Check_{study}.png")


# # =============================================================================
# # Cluster 14
# # Distinguish perivascular cells from stromal fibroblasts
# # =============================================================================
# cluster14_genes = ["RGS5", "CSPG4", "MCAM", "PDGFRB", "ACTA2", "TAGLN", "COL1A1", "COL1A2", "DCN", "LUM", "PRL", "IGFBP1", "LEFTY2", "HOXA13", "NR2F2"]
# save_marker_dotplot(adata_celltypist,cluster14_genes,f"{path_save}/cluster14Check_{study}.png")


# # =============================================================================
# # Cluster 19
# # Determine whether this is stromal, epithelial, endothelial,
# # immune, or simply a stressed/low-quality population
# # =============================================================================
# cluster19_genes = ["PAPPA", "DCN", "LUM", "COL1A1", "COL3A1", "PDGFRA", "EPCAM", "KRT8", "KRT18", "PECAM1", "VWF", "PTPRC"]
# save_marker_dotplot(adata_celltypist,cluster19_genes,f"{path_save}/cluster19Check_{study}.png")


### !!!!!!!!!!! Must change annotations for particular study !!!!!!!!!!!! ###
cluster_to_label = [
    "Tem/Trm_cytotoxic_T",  # 0
    "SOX9_luminal",         # 1
    "Memory_B",             # 2
    "Pericyte",             # 3
    "Venous",               # 4
    "eStromal",        # 5
    "uSMCs",                # 6
    "eStromal",        # 7
    "Macrophages",          # 8
    "Ciliated",             # 9
    "Regulatory_T",         # 10
    "eHormones",            # 11
    "dStromal_mid",         # 12
    "eStromal",             # 13
    "eStromal",             # 14  
    "Tem/Trm_cytotoxic_T",  # 15
    "Plasma",               # 16
    "Mast",                 # 17
    "ILC3",                 # 18
    "PAPPA_Stromal",        # 19 
    "Lymphatic",            # 20
    "Macrophages",          # 21 
]

cluster_to_compartment = {
    "0": "Immune",
    "1": "Epithelial",
    "2": "Immune",
    "3": "Stromal",
    "4": "Endothelial",
    "5": "Stromal",
    "6": "Stromal",
    "7": "Stromal",
    "8": "Immune",
    "9": "Epithelial",
    "10": "Immune",
    "11": "Epithelial",
    "12": "Stromal",
    "13": "Stromal",
    "14": "Stromal",
    "15": "Immune",
    "16": "Immune",
    "17": "Immune",
    "18": "Immune",
    "19": "Stromal",
    "20": "Endothelial",
    "21": "Immune"}

# # ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
# ### GSE173682   GSE173682   GSE173682   GSE173682   GSE173682   GSE173682   GSE173682


### !!!!!!!!!!! Must change annotations for particular study !!!!!!!!!!!! ###
### GSE251923   GSE251923   GSE251923   GSE251923   GSE251923   GSE251923   GSE251923  
# cluster_labels = [
#     "Naive_B",               # 0
#     "eStromal",              # 1
#     "uSMCs",                 # 2
#     "eHormones",             # 3
#     "Venous",                # 4
#     "Tem/Trm_cytotoxic_T",   # 5
#     "eHormones",             # 6
#     "eHormones",             # 7
#     "Cycling_Epithelial",    # 8
#     "eHormones",             # 9
#     "Macrophages",           # 10
#     "Plasma",                # 11
#     "Mast",                  # 12
#     "eHormones",             # 13
#     "Glandular_secretory",   # 14
#     "Lymphatic",             # 15
#     "CD4_T",                 # 16
# ]
# cluster_to_label = {str(index): label for index, label in enumerate(cluster_labels)}

# cluster_to_compartment = {
#     "0": "Immune",       # Naive_B
#     "1": "Stromal",      # eStromal
#     "2": "Stromal",      # uSMCs
#     "3": "Epithelial",   # eHormones
#     "4": "Endothelial",  # Venous
#     "5": "Immune",       # Tem/Trm_cytotoxic_T
#     "6": "Epithelial",   # eHormones
#     "7": "Epithelial",   # eHormones
#     "8": "Epithelial",   # Cycling_Epithelial
#     "9": "Epithelial",   # eHormones
#     "10": "Immune",      # Macrophages
#     "11": "Immune",      # Plasma_cells
#     "12": "Immune",      # Mast_cells
#     "13": "Epithelial",  # eHormones
#     "14": "Epithelial",  # Glandular_secretory
#     "15": "Endothelial", # Lymphatic
#     "16": "Immune",      # CD4_T
# }
# # ### !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! ###
### GSE251923   GSE251923   GSE251923   GSE251923   GSE251923   GSE251923   GSE251923  


if isinstance(cluster_to_label, list):
    cluster_to_label = {str(index): label for index, label in enumerate(cluster_to_label)}

observed_clusters = set(adata_full.obs["leiden"].astype(str).unique())
missing_labels = sorted(observed_clusters - set(cluster_to_label), key=int)
missing_compartments = sorted(observed_clusters - set(cluster_to_compartment), key=int)
if missing_labels or missing_compartments:
    raise ValueError(f"Manual annotation dictionaries are incomplete. Missing labels: {missing_labels}; missing compartments: {missing_compartments}")

adata_full.obs["manual_annotation"] = (adata_full.obs["leiden"].astype(str).map(cluster_to_label))
adata_full.obs["compartment"] = (adata_full.obs["leiden"].astype(str).map(cluster_to_compartment))
sc.pl.umap(adata_full,color="manual_annotation", legend_loc="on data",show=False)
plt.savefig(f"{path_save}/tumor_UMAP_{study}.png", dpi=300, bbox_inches="tight")
plt.close()

adata_full.write_h5ad(f"{path}/{study}_adata_annotated_step3.h5ad")
