#!/usr/bin/env python3

from __future__ import annotations

import re
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


# =============================================================================
# Configuration
# =============================================================================

STUDY = "GSE173682"
VALIDATION_STUDY = "GSE251923"

NORMAL_PATH = Path("/home/grudmans/EC_ref/PMID_39198675/endometriumAtlasV2_cells_with_counts_quarter.h5ad")

TUMOR_PATH = Path(
    f"/home/grudmans/EC_ref/{STUDY}_RAW/"
    f"{STUDY}_adata_annotated_step3.h5ad"
)
TUMOR_ALIGNMENT_PATH = Path(
    f"/home/grudmans/EC_ref/{STUDY}_RAW/"
    f"{STUDY}_tumor_all_compartments_scanvi_query.h5ad"
)
VALIDATION_TUMOR_PATH = Path(
    f"/home/grudmans/EC_ref/{VALIDATION_STUDY}_RAW/"
    f"{VALIDATION_STUDY}_adata_annotated_step3.h5ad"
)
VALIDATION_TUMOR_ALIGNMENT_PATH = Path(
    f"/home/grudmans/EC_ref/{VALIDATION_STUDY}_RAW/"
    f"{VALIDATION_STUDY}_tumor_all_compartments_scanvi_query.h5ad"
)
TUMOR_ALIGNMENT_TO_ANALYZE = "Poorly_aligned_to_normal"

OUTPUT_DIRECTORY = Path(f"/home/grudmans/EC_ref/xenium_epithelial_panel_builder")

MIN_EPITHELIAL_CELLS_PER_SAMPLE = 100
MIN_TOTAL_COUNTS_FOR_DESEQ2 = 10
MIN_SAMPLES_WITH_COUNTS_FOR_DESEQ2 = 2
PATIENT_UP_LOG2FC_THRESHOLD = 0.5
MIN_DISCOVERY_LOG2FC = 1.0
MIN_REPLICATION_LOG2FC = 0.5
MAX_PADJ = 0.05
MIN_TUMOR_CELL_DETECTION = 0.10
MIN_DISCOVERY_EPITHELIAL_SPECIFICITY_MARGIN = 0.10
MIN_REPLICATION_EPITHELIAL_SPECIFICITY_MARGIN = 0.05
MIN_DISCOVERY_SAMPLE_FRACTION_UP = 0.50
MIN_VALIDATION_SAMPLE_FRACTION_UP = 1.00
TOP_CANDIDATES_TO_SAVE = 50
N_DESEQ2_CPUS = 4
RANDOM_SEED = 123

OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)


# =============================================================================
# Utility functions
# =============================================================================

def get_counts(adata: ad.AnnData):
    """Return raw counts, preferring layers['counts']."""
    matrix = adata.layers["counts"] if "counts" in adata.layers else adata.X
    if sparse.issparse(matrix):
        return matrix.tocsr()
    return np.asarray(matrix)


def first_existing_column(adata: ad.AnnData, candidates: list[str]) -> str | None:
    for column in candidates:
        if column in adata.obs.columns:
            return column
    return None


def normalize_label(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def classify_compartment(adata: ad.AnnData, source: str) -> pd.Series:
    """
    Create broad epithelial/immune/stromal/endothelial/other labels.
    Existing broad lineage columns are preferred. Manual annotations are used
    as a fallback.
    """
    broad_column = first_existing_column( adata, [ "analysis_lineage", "compartment", "reference_lineage", "lineage", ], )

    if broad_column is not None:
        raw = adata.obs[broad_column].astype(str).map(normalize_label)
    else:
        raw = pd.Series("other", index=adata.obs_names)

    output = pd.Series("other", index=adata.obs_names, dtype=object)

    output.loc[raw.str.contains("epith", na=False)] = "epithelial"
    output.loc[
        raw.str.contains( "immune|lymph|myeloid|t_cell|b_cell|macroph|mast|nk|plasma", regex=True, na=False, )
    ] = "immune"
    output.loc[
        raw.str.contains( "strom|fibro|smooth_muscle|perivascular|pericyte|smc", regex=True, na=False, )
    ] = "stromal"
    output.loc[
        raw.str.contains( "endothel|arterial|venous|lymphatic", regex=True, na=False, )
    ] = "endothelial"

    annotation_column = first_existing_column( adata, [ "manual_annotation", "celltype", "reference_celltype", "celltypist_endometrium", "celltypist_immune", ], )

    if annotation_column is not None:
        annotation = (
            adata.obs[annotation_column] .astype(str) .map(normalize_label)
        )

        unresolved = output.eq("other")
        output.loc[
            unresolved
            & annotation.str.contains( "epith|luminal|glandular|ciliated|sox9|krt5|hormone|muc5b", regex=True, na=False, )
        ] = "epithelial"

        unresolved = output.eq("other")
        output.loc[
            unresolved
            & annotation.str.contains( "cd8|cd4|treg|t_cell|b_cell|macroph|mast|plasma|nk|myeloid|lymph", regex=True, na=False, )
        ] = "immune"

        unresolved = output.eq("other")
        output.loc[
            unresolved
            & annotation.str.contains( "strom|fibro|pericyte|smooth_muscle|smc", regex=True, na=False, )
        ] = "stromal"

        unresolved = output.eq("other")
        output.loc[
            unresolved
            & annotation.str.contains( "endothel|arterial|venous|lymphatic", regex=True, na=False, )
        ] = "endothelial"

    output.name = f"{source}_broad_compartment"
    return output


def aggregate_counts_by_sample(adata: ad.AnnData, sample_column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample_names = adata.obs[sample_column].astype(str)
    unique_samples = pd.Index(sample_names.unique())
    matrix = get_counts(adata)

    rows = []
    metadata_rows = []

    for sample in unique_samples:
        mask = sample_names.eq(sample).to_numpy()
        summed = np.asarray(matrix[mask].sum(axis=0)).ravel()
        rows.append(summed)
        metadata_rows.append( {"sample": sample, "n_cells": int(mask.sum())} )

    counts = pd.DataFrame( np.vstack(rows), index=unique_samples, columns=adata.var_names, )
    metadata = pd.DataFrame(metadata_rows).set_index("sample")
    return counts, metadata


def calculate_detection_fraction(adata: ad.AnnData) -> pd.Series:
    matrix = get_counts(adata)
    if sparse.issparse(matrix):
        values = np.asarray((matrix > 0).mean(axis=0)).ravel()
    else:
        values = np.mean(matrix > 0, axis=0)
    return pd.Series(values, index=adata.var_names)


def calculate_mean_log1p_expression(adata: ad.AnnData) -> pd.Series:
    matrix = get_counts(adata)
    library_sizes = np.asarray(matrix.sum(axis=1)).ravel()
    valid = library_sizes > 0

    if valid.sum() == 0:
        return pd.Series(0.0, index=adata.var_names)

    matrix = matrix[valid]
    library_sizes = library_sizes[valid]

    if sparse.issparse(matrix):
        normalized = sparse.diags(1e4 / library_sizes) @ matrix
        normalized.data = np.log1p(normalized.data)
        means = np.asarray(normalized.mean(axis=0)).ravel()
    else:
        normalized = matrix / library_sizes[:, None] * 1e4
        means = np.log1p(normalized).mean(axis=0)

    return pd.Series(means, index=adata.var_names)


def run_pydeseq2(counts: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    """Run PyDESeq2 with compatibility for current and older APIs."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    keep = (
        (counts >= MIN_TOTAL_COUNTS_FOR_DESEQ2).sum(axis=0)
        >= MIN_SAMPLES_WITH_COUNTS_FOR_DESEQ2
    )
    filtered_counts = counts.loc[:, keep].round().astype(int)

    design_metadata = metadata[["condition"]].copy()
    design_metadata["condition"] = pd.Categorical( design_metadata["condition"], categories=["Normal", "Tumor"], )

    dds = None
    errors = []

    constructors = [
        {
            "counts": filtered_counts,
            "metadata": design_metadata,
            "design": "~condition",
            "refit_cooks": True,
            "n_cpus": N_DESEQ2_CPUS,
        },
        {
            "counts": filtered_counts,
            "metadata": design_metadata,
            "design_factors": "condition",
            "refit_cooks": True,
            "n_cpus": N_DESEQ2_CPUS,
        },
    ]

    for kwargs in constructors:
        try:
            dds = DeseqDataSet(**kwargs)
            break
        except TypeError as error:
            errors.append(str(error))

    if dds is None:
        raise RuntimeError( "Could not initialize DeseqDataSet with either supported API. " + " | ".join(errors) )

    dds.deseq2()

    stats_object = DeseqStats( dds, contrast=["condition", "Tumor", "Normal"], n_cpus=N_DESEQ2_CPUS, )
    stats_object.summary()

    result = stats_object.results_df.reset_index()
    if "gene" not in result.columns:
        result = result.rename(columns={result.columns[0]: "gene"})
    result["gene"] = result["gene"].astype(str)
    return result


def patient_robustness_metrics(cpm: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    tumor_samples = metadata.index[metadata["condition"].eq("Tumor")]
    normal_samples = metadata.index[metadata["condition"].eq("Normal")]

    normal_mean = cpm.loc[normal_samples].mean(axis=0)
    tumor_log2fc = np.log2( (cpm.loc[tumor_samples] + 0.1) .div(normal_mean + 0.1, axis=1) )

    result = pd.DataFrame({"gene": cpm.columns})
    for sample in tumor_samples:
        clean_sample = re.sub(r"^Tumor_", "", str(sample))
        result[f"patient_log2FC_{clean_sample}"] = (
            tumor_log2fc.loc[sample].to_numpy()
        )

    result["n_tumor_patients_up_0.5"] = (
        tumor_log2fc.ge(PATIENT_UP_LOG2FC_THRESHOLD) .sum(axis=0) .to_numpy()
    )
    result["fraction_tumor_patients_up_0.5"] = (
        tumor_log2fc.ge(PATIENT_UP_LOG2FC_THRESHOLD) .mean(axis=0) .to_numpy()
    )
    result["median_patient_log2FC"] = (
        tumor_log2fc.median(axis=0).to_numpy()
    )
    result["minimum_patient_log2FC"] = (
        tumor_log2fc.min(axis=0).to_numpy()
    )
    result["patient_log2FC_range"] = ( tumor_log2fc.max(axis=0) - tumor_log2fc.min(axis=0) ).to_numpy()

    return result


def gene_flags(gene: str) -> dict[str, bool]:
    gene_upper = gene.upper()
    return {
        "flag_ribosomal": bool(re.match(r"^RP[SL]\d", gene_upper)),
        "flag_mitochondrial": gene_upper.startswith("MT-"),
        "flag_hemoglobin": bool(re.match(r"^HB[ABDEGQZ]", gene_upper)),
        "flag_generic_housekeeping": gene_upper in {
            "ACTB", "ACTG1", "GAPDH", "MALAT1", "B2M", "TUBB",
            "TUBA1A", "EEF1A1", "NACA", "FTL", "FTH1", "YBX1",
        },
        "flag_lncRNA_or_uncharacterized": (
            gene_upper.endswith("-AS1")
            or gene_upper.startswith("LINC")
            or gene_upper.startswith("MIR")
            or gene_upper.startswith("AC0")
            or gene_upper.startswith("AL0")
        ),
    }


# =============================================================================
# Load full data and define compartments
# =============================================================================

normal_full = sc.read_h5ad(NORMAL_PATH)
tumor_full = sc.read_h5ad(TUMOR_PATH)
validation_tumor_full = sc.read_h5ad(VALIDATION_TUMOR_PATH)

for tumor_name, tumor_adata, alignment_path in [
    (STUDY, tumor_full, TUMOR_ALIGNMENT_PATH),
    (
        VALIDATION_STUDY,
        validation_tumor_full,
        VALIDATION_TUMOR_ALIGNMENT_PATH,
    ),
]:
    if "counts" not in tumor_adata.layers:
        raise KeyError(
            f"{tumor_name} full object is missing layers['counts']"
        )
    if not tumor_adata.obs_names.is_unique:
        raise ValueError(f"{tumor_name} full object has duplicate cell IDs")

    alignment_query = ad.read_h5ad(alignment_path, backed="r")
    if "normal_alignment" not in alignment_query.obs.columns:
        raise KeyError(
            f"{tumor_name} query object is missing obs['normal_alignment']"
        )
    if not alignment_query.obs_names.is_unique:
        raise ValueError(f"{tumor_name} query object has duplicate cell IDs")

    alignment = (
        alignment_query.obs["normal_alignment"]
        .astype("string")
        .reindex(tumor_adata.obs_names)
    )
    alignment_query.file.close()
    if alignment.isna().any():
        raise ValueError(
            f"{tumor_name}: {int(alignment.isna().sum())} full-object cells "
            "are missing from the scANVI query object"
        )
    tumor_adata.obs["normal_alignment"] = alignment.to_numpy()

normal_full.var_names_make_unique()
tumor_full.var_names_make_unique()
validation_tumor_full.var_names_make_unique()

normal_full.obs["_panel_compartment"] = classify_compartment( normal_full, "normal", )
tumor_full.obs["_panel_compartment"] = classify_compartment( tumor_full, "tumor", )
validation_tumor_full.obs["_panel_compartment"] = classify_compartment(
    validation_tumor_full, "validation_tumor",
)

print("\nNormal broad compartments:")
print(normal_full.obs["_panel_compartment"].value_counts())

print("\nTumor broad compartments:")
print(tumor_full.obs["_panel_compartment"].value_counts())

print(f"\nValidation tumor ({VALIDATION_STUDY}) broad compartments:")
print(validation_tumor_full.obs["_panel_compartment"].value_counts())

normal_epithelial = normal_full[ normal_full.obs["_panel_compartment"].eq("epithelial") ].copy()
tumor_epithelial = tumor_full[
    tumor_full.obs["_panel_compartment"].eq("epithelial")
    & tumor_full.obs["normal_alignment"]
    .astype(str)
    .eq(TUMOR_ALIGNMENT_TO_ANALYZE)
].copy()
validation_tumor_epithelial = validation_tumor_full[
    validation_tumor_full.obs["_panel_compartment"].eq("epithelial")
    & validation_tumor_full.obs["normal_alignment"]
    .astype(str)
    .eq(TUMOR_ALIGNMENT_TO_ANALYZE)
].copy()

shared_genes = normal_epithelial.var_names.intersection( tumor_epithelial.var_names )
shared_genes = shared_genes.intersection(validation_tumor_epithelial.var_names)
normal_epithelial = normal_epithelial[:, shared_genes].copy()
tumor_epithelial = tumor_epithelial[:, shared_genes].copy()
validation_tumor_epithelial = validation_tumor_epithelial[:, shared_genes].copy()

normal_sample_counts = (
    normal_epithelial.obs["sample"].astype(str).value_counts()
)
tumor_sample_counts = (
    tumor_epithelial.obs["sample"].astype(str).value_counts()
)
validation_tumor_sample_counts = (
    validation_tumor_epithelial.obs["sample"].astype(str).value_counts()
)

normal_samples_keep = normal_sample_counts[
    normal_sample_counts >= MIN_EPITHELIAL_CELLS_PER_SAMPLE
].index
tumor_samples_keep = tumor_sample_counts[
    tumor_sample_counts >= MIN_EPITHELIAL_CELLS_PER_SAMPLE
].index
validation_tumor_samples_keep = validation_tumor_sample_counts[
    validation_tumor_sample_counts >= MIN_EPITHELIAL_CELLS_PER_SAMPLE
].index

normal_epithelial = normal_epithelial[ normal_epithelial.obs["sample"] .astype(str) .isin(normal_samples_keep) ].copy()
tumor_epithelial = tumor_epithelial[ tumor_epithelial.obs["sample"] .astype(str) .isin(tumor_samples_keep) ].copy()
validation_tumor_epithelial = validation_tumor_epithelial[
    validation_tumor_epithelial.obs["sample"]
    .astype(str)
    .isin(validation_tumor_samples_keep)
].copy()

print( f"\nNormal epithelial cells retained: " f"{normal_epithelial.n_obs}" )
print(
    f"Tumor poorly aligned epithelial cells retained: "
    f"{tumor_epithelial.n_obs}"
)
print(
    f"Validation tumor poorly aligned epithelial cells retained: "
    f"{validation_tumor_epithelial.n_obs}"
)
print(f"Shared genes: {len(shared_genes)}")
print( f"Normal epithelial samples retained: " f"{normal_epithelial.obs['sample'].nunique()}" )
print(
    f"Tumor poorly aligned epithelial samples retained: "
    f"{tumor_epithelial.obs['sample'].nunique()}"
)
print(
    f"Validation tumor poorly aligned epithelial samples retained: "
    f"{validation_tumor_epithelial.obs['sample'].nunique()}"
)


# =============================================================================
# Pseudobulk counts and DE
# =============================================================================

normal_counts, normal_metadata = aggregate_counts_by_sample( normal_epithelial, "sample", )
tumor_counts, tumor_metadata = aggregate_counts_by_sample( tumor_epithelial, "sample", )
validation_tumor_counts, validation_tumor_metadata = aggregate_counts_by_sample(
    validation_tumor_epithelial, "sample",
)

normal_counts.index = "Normal_" + normal_counts.index.astype(str)
tumor_counts.index = "Tumor_" + tumor_counts.index.astype(str)
validation_tumor_counts.index = (
    "ValidationTumor_" + validation_tumor_counts.index.astype(str)
)
normal_metadata.index = normal_counts.index
tumor_metadata.index = tumor_counts.index
validation_tumor_metadata.index = validation_tumor_counts.index

normal_metadata["condition"] = "Normal"
tumor_metadata["condition"] = "Tumor"
validation_tumor_metadata["condition"] = "Tumor"

counts = pd.concat([normal_counts, tumor_counts], axis=0)
metadata = pd.concat([normal_metadata, tumor_metadata], axis=0)
validation_counts = pd.concat(
    [normal_counts, validation_tumor_counts], axis=0,
)
validation_metadata = pd.concat(
    [normal_metadata, validation_tumor_metadata], axis=0,
)

library_sizes = counts.sum(axis=1)
cpm = counts.div(library_sizes, axis=0) * 1e6
validation_library_sizes = validation_counts.sum(axis=1)
validation_cpm = validation_counts.div(
    validation_library_sizes, axis=0,
) * 1e6

try:
    deseq2_results = run_pydeseq2(counts, metadata)
except Exception as error:
    raise RuntimeError( "PyDESeq2 is installed but the analysis failed. " "The script is stopping rather than silently falling back to Welch. " f"Original error: {error}" ) from error

candidate_table = deseq2_results[
    [
        column
        for column in [
            "gene", "baseMean", "log2FoldChange", "lfcSE",
            "stat", "pvalue", "padj",
        ]
        if column in deseq2_results.columns
    ]
].copy()

try:
    validation_deseq2_results = run_pydeseq2(
        validation_counts, validation_metadata,
    )
except Exception as error:
    raise RuntimeError(
        f"PyDESeq2 validation analysis for {VALIDATION_STUDY} failed. "
        f"Original error: {error}"
    ) from error

validation_columns = [
    column
    for column in [
        "gene", "baseMean", "log2FoldChange", "lfcSE",
        "stat", "pvalue", "padj",
    ]
    if column in validation_deseq2_results.columns
]
validation_de = validation_deseq2_results[validation_columns].rename(
    columns={
        column: f"validation_{column}"
        for column in validation_columns
        if column != "gene"
    }
)
candidate_table = candidate_table.merge(validation_de, on="gene", how="left")

print(
    "\nDifferential-expression method used: PyDESeq2 for both "
    f"{STUDY} and validation study {VALIDATION_STUDY}"
)


# =============================================================================
# Patient robustness
# =============================================================================

robustness = patient_robustness_metrics(cpm, metadata)
candidate_table = candidate_table.merge( robustness, on="gene", how="left", )
validation_robustness = patient_robustness_metrics(
    validation_cpm, validation_metadata,
)
validation_robustness = validation_robustness.rename(
    columns={
        column: f"validation_{column}"
        for column in validation_robustness.columns
        if column != "gene"
    }
)
candidate_table = candidate_table.merge(
    validation_robustness, on="gene", how="left",
)


# =============================================================================
# Compartment specificity
# =============================================================================

compartment_objects: dict[str, ad.AnnData] = {
    "normal_epithelial": normal_full[
        normal_full.obs["_panel_compartment"].eq("epithelial")
    ],
    "normal_immune": normal_full[
        normal_full.obs["_panel_compartment"].eq("immune")
    ],
    "normal_stromal": normal_full[
        normal_full.obs["_panel_compartment"].eq("stromal")
    ],
    "normal_endothelial": normal_full[
        normal_full.obs["_panel_compartment"].eq("endothelial")
    ],
    "tumor_epithelial": tumor_epithelial,
    "tumor_immune": tumor_full[
        tumor_full.obs["_panel_compartment"].eq("immune")
    ],
    "tumor_stromal": tumor_full[
        tumor_full.obs["_panel_compartment"].eq("stromal")
    ],
    "tumor_endothelial": tumor_full[
        tumor_full.obs["_panel_compartment"].eq("endothelial")
    ],
    "validation_tumor_epithelial": validation_tumor_epithelial,
    "validation_tumor_immune": validation_tumor_full[
        validation_tumor_full.obs["_panel_compartment"].eq("immune")
    ],
    "validation_tumor_stromal": validation_tumor_full[
        validation_tumor_full.obs["_panel_compartment"].eq("stromal")
    ],
    "validation_tumor_endothelial": validation_tumor_full[
        validation_tumor_full.obs["_panel_compartment"].eq("endothelial")
    ],
}

specificity_table = pd.DataFrame({"gene": shared_genes})

for compartment_name, compartment_adata in compartment_objects.items():
    if compartment_adata.n_obs == 0:
        print( f"Compartment unavailable: {compartment_name}; " "columns will be NaN." )
        specificity_table[
            f"{compartment_name}_cell_fraction_detected"
        ] = np.nan
        specificity_table[
            f"{compartment_name}_mean_log1p_expression"
        ] = np.nan
        continue

    compartment_adata = compartment_adata[ :, compartment_adata.var_names.intersection(shared_genes), ].copy()

    detection = calculate_detection_fraction(compartment_adata)
    expression = calculate_mean_log1p_expression(compartment_adata)

    specificity_table[
        f"{compartment_name}_cell_fraction_detected"
    ] = detection.reindex(shared_genes).to_numpy()
    specificity_table[
        f"{compartment_name}_mean_log1p_expression"
    ] = expression.reindex(shared_genes).to_numpy()

candidate_table = candidate_table.merge( specificity_table, on="gene", how="left", )

discovery_off_target_columns = [
    "tumor_immune_cell_fraction_detected",
    "tumor_stromal_cell_fraction_detected",
    "tumor_endothelial_cell_fraction_detected",
]
validation_off_target_columns = [
    "validation_tumor_immune_cell_fraction_detected",
    "validation_tumor_stromal_cell_fraction_detected",
    "validation_tumor_endothelial_cell_fraction_detected",
]

candidate_table["discovery_maximum_off_target_detection"] = (
    candidate_table[discovery_off_target_columns].max(axis=1, skipna=True)
)
candidate_table["validation_maximum_off_target_detection"] = (
    candidate_table[validation_off_target_columns].max(axis=1, skipna=True)
)
candidate_table["discovery_epithelial_specificity_margin"] = (
    candidate_table["tumor_epithelial_cell_fraction_detected"]
    - candidate_table["discovery_maximum_off_target_detection"]
)
candidate_table["validation_epithelial_specificity_margin"] = (
    candidate_table["validation_tumor_epithelial_cell_fraction_detected"]
    - candidate_table["validation_maximum_off_target_detection"]
)
# =============================================================================
# Explicit eligibility gates and worst-study ranking
# =============================================================================

flag_table = pd.DataFrame( [ {"gene": gene, **gene_flags(gene)} for gene in candidate_table["gene"] ] )
candidate_table = candidate_table.merge( flag_table, on="gene", how="left", )

candidate_table["passes_discovery_log2fc"] = (
    candidate_table["log2FoldChange"] >= MIN_DISCOVERY_LOG2FC
)
candidate_table["passes_replication_log2fc"] = (
    candidate_table["validation_log2FoldChange"] >= MIN_REPLICATION_LOG2FC
)
candidate_table["passes_discovery_padj"] = (
    candidate_table["padj"] < MAX_PADJ
)
candidate_table["passes_discovery_detection"] = (
    candidate_table["tumor_epithelial_cell_fraction_detected"]
    >= MIN_TUMOR_CELL_DETECTION
)
candidate_table["passes_replication_detection"] = (
    candidate_table["validation_tumor_epithelial_cell_fraction_detected"]
    >= MIN_TUMOR_CELL_DETECTION
)
candidate_table["passes_discovery_sample_consistency"] = (
    candidate_table["fraction_tumor_patients_up_0.5"]
    >= MIN_DISCOVERY_SAMPLE_FRACTION_UP
)
candidate_table["passes_replication_sample_consistency"] = (
    candidate_table["validation_fraction_tumor_patients_up_0.5"]
    >= MIN_VALIDATION_SAMPLE_FRACTION_UP
)
candidate_table["passes_discovery_epithelial_specificity"] = (
    candidate_table["discovery_epithelial_specificity_margin"]
    >= MIN_DISCOVERY_EPITHELIAL_SPECIFICITY_MARGIN
)
candidate_table["passes_replication_epithelial_specificity"] = (
    candidate_table["validation_epithelial_specificity_margin"]
    >= MIN_REPLICATION_EPITHELIAL_SPECIFICITY_MARGIN
)

candidate_table["flag_exclude_technical"] = candidate_table[
    [
        "flag_ribosomal",
        "flag_mitochondrial",
        "flag_hemoglobin",
        "flag_generic_housekeeping",
    ]
].any(axis=1)
candidate_table["passes_technical_filter"] = ~candidate_table[
    "flag_exclude_technical"
]

candidate_table["passes_consensus_gates"] = candidate_table[
    [
        "passes_discovery_log2fc",
        "passes_replication_log2fc",
        "passes_discovery_padj",
        "passes_discovery_detection",
        "passes_replication_detection",
        "passes_discovery_sample_consistency",
        "passes_replication_sample_consistency",
        "passes_discovery_epithelial_specificity",
        "passes_replication_epithelial_specificity",
        "passes_technical_filter",
    ]
].all(axis=1)

candidate_table["minimum_tumor_study_log2FC"] = candidate_table[
    ["log2FoldChange", "validation_log2FoldChange"]
].min(axis=1)
candidate_table["minimum_tumor_study_detection"] = candidate_table[
    [
        "tumor_epithelial_cell_fraction_detected",
        "validation_tumor_epithelial_cell_fraction_detected",
    ]
].min(axis=1)
candidate_table["minimum_tumor_study_epithelial_specificity_margin"] = (
    candidate_table[
        [
            "discovery_epithelial_specificity_margin",
            "validation_epithelial_specificity_margin",
        ]
    ].min(axis=1)
)

primary_candidates = candidate_table.loc[
    candidate_table["passes_consensus_gates"]
].sort_values(
    [
        "minimum_tumor_study_log2FC",
        "minimum_tumor_study_epithelial_specificity_margin",
        "minimum_tumor_study_detection",
        "padj",
    ],
    ascending=[False, False, False, True],
).head(TOP_CANDIDATES_TO_SAVE).copy()
primary_candidates.insert(
    0, "rank", np.arange(1, len(primary_candidates) + 1),
)

# =============================================================================
# Single output workbook
# =============================================================================

excel_path = OUTPUT_DIRECTORY / "xenium_panel_recommendations.xlsx"

recommendation_columns = [
    "rank",
    "gene",
    "log2FoldChange",
    "validation_log2FoldChange",
    "padj",
    "validation_padj",
    "tumor_epithelial_cell_fraction_detected",
    "validation_tumor_epithelial_cell_fraction_detected",
    "fraction_tumor_patients_up_0.5",
    "validation_fraction_tumor_patients_up_0.5",
    "discovery_maximum_off_target_detection",
    "validation_maximum_off_target_detection",
    "discovery_epithelial_specificity_margin",
    "validation_epithelial_specificity_margin",
]
recommendations = primary_candidates[
    [column for column in recommendation_columns if column in primary_candidates]
].copy()

for column in [
    "log2FoldChange",
    "validation_log2FoldChange",
]:
    if column in recommendations:
        recommendations[column] = np.exp2(recommendations[column])

sample_fc_threshold = 2 ** PATIENT_UP_LOG2FC_THRESHOLD
sample_fc_threshold_label = f"{sample_fc_threshold:.3f}".rstrip("0").rstrip(".")

export_column_names = {
    "log2FoldChange": f"{STUDY}_FC",
    "validation_log2FoldChange": f"{VALIDATION_STUDY}_FC",
    "padj": f"{STUDY}_padj",
    "validation_padj": f"{VALIDATION_STUDY}_padj",
    "tumor_epithelial_cell_fraction_detected": (
        f"{STUDY}_poorly_aligned_epithelial_cell_fraction_detected"
    ),
    "validation_tumor_epithelial_cell_fraction_detected": (
        f"{VALIDATION_STUDY}_poorly_aligned_epithelial_cell_fraction_detected"
    ),
    "fraction_tumor_patients_up_0.5": (
        f"{STUDY}_poorly_aligned_sample_fraction_FC_above_"
        f"{sample_fc_threshold_label}"
    ),
    "validation_fraction_tumor_patients_up_0.5": (
        f"{VALIDATION_STUDY}_poorly_aligned_sample_fraction_FC_above_"
        f"{sample_fc_threshold_label}"
    ),
    "discovery_maximum_off_target_detection": (
        f"{STUDY}_maximum_off_target_detection"
    ),
    "validation_maximum_off_target_detection": (
        f"{VALIDATION_STUDY}_maximum_off_target_detection"
    ),
    "discovery_epithelial_specificity_margin": (
        f"{STUDY}_poorly_aligned_epithelial_specificity_margin"
    ),
    "validation_epithelial_specificity_margin": (
        f"{VALIDATION_STUDY}_poorly_aligned_epithelial_specificity_margin"
    ),
}
recommendations = recommendations.rename(columns=export_column_names)

all_gene_columns = [
    "gene",
    *recommendation_columns[2:],
]
all_genes = candidate_table.sort_values(
    ["passes_consensus_gates", "log2FoldChange", "padj"],
    ascending=[False, False, True],
)[
    [column for column in all_gene_columns if column in candidate_table]
].copy()
for column in ["log2FoldChange", "validation_log2FoldChange"]:
    if column in all_genes:
        all_genes[column] = np.exp2(all_genes[column])
all_genes = all_genes.rename(columns=export_column_names)

try:
    with pd.ExcelWriter( excel_path, engine="openpyxl", ) as writer:
        recommendations.to_excel(
            writer, sheet_name="Recommendations", index=False,
        )
        all_genes.to_excel(
            writer, sheet_name="All genes", index=False,
        )

    print(f"\nSaved Excel workbook:\n{excel_path}")

except ImportError as error:
    raise RuntimeError(
        "openpyxl is required because the recommendations workbook is "
        "the script's only output."
    ) from error


# =============================================================================
# Final summary
# =============================================================================

display_columns = [
    "rank",
    "gene",
    "log2FoldChange",
    "validation_log2FoldChange",
    "padj",
    "validation_padj",
    "tumor_epithelial_cell_fraction_detected",
    "validation_tumor_epithelial_cell_fraction_detected",
    "fraction_tumor_patients_up_0.5",
    "validation_fraction_tumor_patients_up_0.5",
]

print("\nTop consensus Xenium-oriented epithelial candidates:")
print( primary_candidates[ [ column for column in display_columns if column in primary_candidates.columns ] ].round(4) )

print(
    "\nEligibility rules:\n"
    f"Discovery ({STUDY}): log2FC >= {MIN_DISCOVERY_LOG2FC}, adjusted "
    f"p-value < {MAX_PADJ}, detection >= {MIN_TUMOR_CELL_DETECTION:.0%}, "
    f"sample consistency >= {MIN_DISCOVERY_SAMPLE_FRACTION_UP:.0%}, and "
    f"epithelial specificity >= "
    f"{MIN_DISCOVERY_EPITHELIAL_SPECIFICITY_MARGIN:.0%}.\n"
    f"Replication ({VALIDATION_STUDY}): log2FC >= "
    f"{MIN_REPLICATION_LOG2FC}, detection >= "
    f"{MIN_TUMOR_CELL_DETECTION:.0%}, both samples elevated, and "
    f"epithelial specificity >= "
    f"{MIN_REPLICATION_EPITHELIAL_SPECIFICITY_MARGIN:.0%}; replication "
    "adjusted p-value is reported but not required.\n"
    f"The workbook contains up to {TOP_CANDIDATES_TO_SAVE} consensus genes."
)

print(f"\nOnly output file:\n{excel_path}")
