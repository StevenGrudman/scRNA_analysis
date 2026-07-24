# scRNA-seq workflow

Tested on Endometrial Cancer studies GSE173682 and GSE251923:
```text
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE173682 
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE251923 
```

Normal endomtrial Sample:
```text
https://pmc.ncbi.nlm.nih.gov/articles/PMC11387200  
https://www.reproductivecellatlas.org/endometrium_reference.html  
```

## Required Python Packages 

anndata, Scanpy, scvi-tools, CellTypist, infercnvpy, pandas, NumPy, SciPy, matplotlib, igraph, and leidenalg.

## Important configuration

Each script contains a `study` or `STUDY` variable near the top. Set all four scripts to the same study before running the workflow.

Additionally, update all paths.

Several study-specific parameters must also be checked manually:

- Step 01 QC MAD thresholds and doublet threhold.
- Step 03 Leiden resolution, celltypist model, manual cluster labels, and compartment mappings.

Do not run the complete workflow without reviewing these settings.

## Directory structure

Each study directory expects 10x directories. For example:

GSE173682 had six 10x directories: `3533EL/`, `3571DL/`, `36186L/`, `36639L/`, `366C5L/`, and `37EACL/`.

GSE251923 had two 10x directories: `case_A/` and `case_B/`.

## Run order

Run the scripts sequentially:

```bash
/home/grudmans/miniconda3/envs/scanpy_env/bin/python 01_preprocessing.py
/home/grudmans/miniconda3/envs/scanpy_env/bin/python 02_cluster.py
/home/grudmans/miniconda3/envs/scanpy_env/bin/python 03_annotate.py
/home/grudmans/miniconda3/envs/scanpy_env/bin/python 04_inferCNV.py
```

## 01 — Preprocessing, quality control, scVI, and doublet detection

Script: `01_preprocessing.py`

This script loads and combines raw 10x matrices, preserves raw counts in `layers["counts"]`, calculates QC metrics, applies MAD-based filters, selects 5,000 sample-aware highly variable genes, trains scVI, runs SOLO separately for each sample, removes cells above the selected doublet threshold, and retrains the final scVI model.

The active filtering values must be verified before every run.

Main outputs:

- `{STUDY}_adata_full_with_solo_predictions.h5ad`: all QC-passing cells with SOLO predictions.
- `{STUDY}_adata_hvg.h5ad`: retained cells restricted to highly variable genes.
- `{STUDY}_adata_full.h5ad`: retained cells with the full gene set.
- `{STUDY}_scvi_model_singlets/`: final trained scVI model.
- `step01/QC_{STUDY}.png`: QC distributions and candidate MAD thresholds.
- `step01/SOLO_probabilities_{STUDY}.png`: sample-specific doublet probabilities.
- `step01/SOLO_UMAP_{STUDY}.png`: initial latent-space UMAP with QC and SOLO metrics.
- `step01/SOLO_thresholds_{STUDY}.png`: candidate doublet thresholds.
- `step01/training_final_{STUDY}.png`: final scVI training history.

## 02 — Clustering-resolution selection

Script: `02_cluster.py`

This script loads the retained AnnData objects and scVI model, transfers the latent representation to the full-gene object, constructs a neighbor graph and UMAP, and compares Leiden resolutions 0.4, 0.6, 0.8, 1.0, and 1.2.

Main output:

- `step02/resolution_{STUDY}.png`: UMAP comparison of tested Leiden resolutions.

## 03 — Cell-type annotation

Script: `03_annotate.py`

This script reconstructs clustering at the selected resolution, annotates cells with the `Immune_All_Low.pkl` and `Human_Endometrium_Atlas.pkl` (change this depending on the cancer type)  CellTypist models, summarizes both models by cluster, calculates label proportions and entropy, flags uncertain clusters, calculates marker genes, applies manually reviewed cluster labels and compartments, and saves the annotated full-gene object.

Main outputs:

- `{STUDY}_adata_annotated_step3.h5ad`: annotated full-gene object used by Step 04.
- `step03/celltypist_{STUDY}.png`: CellTypist and Leiden UMAPs.
- `step03/{STUDY}_celltypist_consensus_summary.csv`: consensus annotations and review flags.
- `step03/{STUDY}_celltypist_cluster_label_percentages.xlsx`: model-label distributions.
- `step03/{STUDY}_review_cluster_top8_markers.csv`: markers for flagged clusters, when present.
- `step03/clusterTopGenes_{STUDY}.png`: top markers by cluster.
- `step03/tumor_UMAP_{STUDY}.png`: final manual annotation UMAP.

## 04 — Patient-wise inferred CNV analysis

Script: `04_inferCNV.py`

This script loads the annotated object, adds GENCODE v50 genomic positions, runs each sample independently, treats all epithelial cells as observations, and uses all same-sample non-epithelial cells as the reference. Counts are normalized and log transformed, expression is smoothed across 100-gene windows with a step of 10 genes, and each cell's CNV burden is calculated as the mean absolute inferred-CNV value.

Required GTF:

```text
https://www.gencodegenes.org/human/
```

Main outputs:

- `step04/{STUDY}_patientwise_infercnv_results.xlsx`: sample-level epithelial-reference comparisons, annotation-level summaries, and per-cell CNV burdens.
- `step04/{STUDY}_epithelial_cnv_burden.png`: paired box plots comparing epithelial CNV burden with the same-sample non-epithelial reference.
- `step04/{STUDY}_epithelial_cnv_profiles.png`: median chromosome-ordered inferred-CNV profiles for each epithelial annotation and sample.

## Interpreting inferred CNV

Infercnvpy detects chromosome-ordered expression deviations; it does not directly measure DNA copy number. White profile regions are close to the reference, broad continuous red regions are gain-like, and broad continuous blue regions are loss-like. Narrow isolated stripes are more likely to represent individual-gene expression, lineage effects, or noise. The burden scale has no universal malignant threshold, and box-plot separation describes effect size but does not establish statistical significance.

## Additional scripts

### `predictMalignancy.py` — Map tumor cells to a normal scANVI reference

This script trains a normal endometrium scVI/scANVI reference and maps every tumor compartment into that reference. It harmonizes broad compartments, selects 4,000 highly variable genes from the normal atlas, predicts each tumor cell's closest normal identity, and measures its distance from normal cells in the same broad compartment.

For each compartment, the normal-alignment threshold is the 99th percentile of normal-to-normal median nearest-neighbor distances. Tumor cells are labeled `Aligned_to_normal`, `Poorly_aligned_to_normal`, or `No_normal_compartment_reference`. These labels describe transcriptional similarity to the atlas and do not independently prove malignancy.

Configuration includes the study, normal atlas path, 15 nearest neighbors, a 0.99 alignment quantile, and a minimum of 100 cells per retained normal sample.

Main outputs:

- `{STUDY}_tumor_all_compartments_scanvi_query.h5ad`: tumor-only expression object containing counts, scANVI coordinates, predicted normal identities, distances, and `normal_alignment`.
- `{STUDY}_normal_tumor_all_compartments_scanvi.h5ad`: combined normal and tumor latent-space object used for visualization; its `X` contains the 30-dimensional latent representation rather than gene expression.
- `normal_tumor_all_compartments_scanvi_model_scvi/`: trained normal scVI model.
- `normal_tumor_all_compartments_scanvi_model_scanvi_reference/`: trained normal scANVI reference.
- `normal_tumor_all_compartments_scanvi_model_scanvi_query/`: trained tumor query model.
- `normalRef/{STUDY}_alignment_by_compartment.csv`: alignment counts and proportions.
- `normalRef/normal_tumor_all_compartments_three_panel.png`: source, alignment, and compartment UMAPs.
- `normalRef/normal_celltypes_vs_tumor_predictions.png`: normal labels and tumor predictions.
- `normalRef/normal_tumor_distance_distributions_by_compartment.png`: normal and tumor distance distributions with thresholds.

### `build_xenium_epithelial_panel_builder_v2.py` — Rank tumor-enriched epithelial genes

This script recommends Xenium-oriented epithelial genes that are tumor enriched, consistently elevated across samples, and relatively specific to epithelial cells in both tumor studies.

It transfers `normal_alignment` from each scANVI query to the corresponding full-gene tumor object and restricts tumor expression analysis to `Poorly_aligned_to_normal` epithelial cells. Non-epithelial tumor compartments are used to measure off-target detection.

The script performs sample-level pseudobulk differential expression with PyDESeq2, calculates per-sample fold-change consistency, cell detection fractions, maximum immune/stromal/endothelial off-target detection, and epithelial specificity margins.

Current eligibility rules:

- Discovery log2FC at least 1.0.
- Replication log2FC at least 0.5.
- Discovery adjusted p-value below 0.05.
- At least 10% epithelial-cell detection in both studies.
- At least 50% of discovery samples above log2FC 0.5.
- Both replication samples above log2FC 0.5.
- Epithelial specificity margin at least 10% in discovery and 5% in replication.
- Exclusion of ribosomal, mitochondrial, hemoglobin, and selected housekeeping genes.

Candidates are ranked using the weakest performance across the two studies, prioritizing cross-study fold change, epithelial specificity, and detection. Up to 50 genes are retained.

Only output:

- `/home/grudmans/EC_ref/xenium_epithelial_panel_builder/xenium_panel_recommendations.xlsx`

Workbook sheets:

- `Recommendations`: top genes passing every consensus gate.
- `All genes`: numerical metrics for every tested gene.

Exported fold changes are ordinary fold changes rather than log2 fold changes. The sample-consistency column uses FC above 1.414 because `2^0.5` is approximately 1.414.

### `create_xenium_references.py` — Package Xenium reference archives

This script creates three independent ZIP archives: one from the normal endometrium atlas and one from each annotated tumor study. It extracts raw counts, creates unique dataset-prefixed barcodes, exports cell annotations, writes a Cell Ranger-compatible feature-barcode HDF5 matrix, verifies matrix dimensions and exact barcode agreement, and packages the matrix and annotations together.

Required metadata:

- Normal atlas: `celltype`, `lineage`, and `sample`.
- Tumor studies: `manual_annotation`, `compartment`, and `sample`.
- Every input object must contain `layers["counts"]`.

Outputs under `/home/grudmans/EC_ref/xenium_final_references/`:

- `PMID_39198675_xenium_reference.zip`
- `GSE173682_xenium_reference.zip`
- `GSE251923_xenium_reference.zip`

Each archive contains a feature-barcode HDF5 matrix and a cell-annotation CSV with barcode, cell type, compartment, source, study, sample, and original barcode.

### `downsize_h5ad.py` — Stratified normal-atlas subsampling

Downsizes an H5AD object. It opens the full normal atlas in backed mode, samples cells independently within every `sample` and `celltype` group, loads only selected cells into memory, and writes a compressed H5AD.

The current configuration retains 80% of cells with random seed 123:

- Input: `/home/grudmans/EC_ref/PMID_39198675/endometriumAtlasV2_cells_with_counts.h5ad`
- Output: `/home/grudmans/EC_ref/PMID_39198675/endometriumAtlasV2_cells_with_counts_80.h5ad`

### `update_annotations.py` — Deprecated malignancy-prefixing script

Do not use this script. The source file explicitly marks itself as deprecated.

It was designed to transfer scANVI mapping columns into the full annotated tumor object and prefix poorly aligned epithelial labels with `Malignant_`. It would save an annotated H5AD, cell-level metadata CSV, and annotation summary CSV. This logic equates poor normal-atlas alignment with malignancy, which is not sufficiently supported because poor alignment can also reflect tumor state, batch effects, missing normal subtypes, or technical differences.

