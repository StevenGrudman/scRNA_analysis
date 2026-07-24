import anndata as ad
import scanpy as sc
import scvi
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

study = "GSE173682"
path = f'/home/grudmans/EC_ref/{study}_RAW'
path_save = f'/home/grudmans/EC_ref/{study}_RAW/step02'

Path(path_save).mkdir(parents=True, exist_ok=True)

### Load Data
adata_hvg = ad.read_h5ad(f"{path}/{study}_adata_hvg.h5ad")
adata_full = ad.read_h5ad(f"{path}/{study}_adata_full.h5ad")
model = scvi.model.SCVI.load(f"{path}/{study}_scvi_model_singlets",adata=adata_hvg)

# Get latent representations
adata_hvg.obsm["X_scVI"] = model.get_latent_representation()
adata_full.obsm["X_scVI"] = adata_hvg.obsm["X_scVI"].copy()


### Cluster and test many reolutions (pick best)
sc.pp.neighbors(adata_full,use_rep="X_scVI",n_neighbors=15,key_added="scvi_neighbors")
sc.tl.umap(adata_full,neighbors_key="scvi_neighbors",min_dist=0.3)
for res in [0.4, 0.6, 0.8, 1.0, 1.2]:
    key = f"leiden_{res}"
    sc.tl.leiden(adata_full,resolution=res,neighbors_key="scvi_neighbors",key_added=key,flavor="igraph",random_state=0)
    print(key,adata_full.obs[key].nunique(),"clusters")

sc.pl.umap(adata_full,color=["sample","leiden_0.4","leiden_0.6","leiden_0.8","leiden_1.0","leiden_1.2",],ncols=2,legend_loc="on data",show=False)
plt.savefig(f"{path_save}/resolution_{study}.png", dpi=300, bbox_inches="tight")
plt.close()

