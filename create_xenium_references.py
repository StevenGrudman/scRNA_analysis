from pathlib import Path
from tempfile import TemporaryDirectory
import gc
import zipfile

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse


# Create three completely separate reference archives: one for the normal atlas
# and one for each tumor study.
NORMAL_PATH = Path("/home/grudmans/EC_ref/PMID_39198675/endometriumAtlasV2_cells_with_counts_quarter.h5ad")
STUDIES = ["GSE251923", "GSE173682"]
STUDY_PATH_TEMPLATE = "/home/grudmans/EC_ref/{study}_RAW/{study}_adata_annotated_step3.h5ad"
OUTPUT_DIRECTORY = Path("/home/grudmans/EC_ref/xenium_final_references")
NORMAL_CELLTYPE_COLUMN = "celltype"
NORMAL_COMPARTMENT_COLUMN = "lineage"
STUDY_CELLTYPE_COLUMN = "manual_annotation"
STUDY_COMPARTMENT_COLUMN = "compartment"
COUNTS_LAYER = "counts"
GENOME = "GRCh38"

def clean_text(values: pd.Series, missing_value: str = "Unknown") -> pd.Series:
    cleaned = values.astype("string").fillna(missing_value).str.strip()
    return cleaned.mask(cleaned.eq(""), missing_value).astype(str)


def get_gene_ids(adata: ad.AnnData) -> pd.Series:
    if "gene_ids" in adata.var.columns:
        gene_ids = adata.var["gene_ids"].astype("string")
    else:
        gene_id_columns = [column for column in adata.var.columns if str(column).startswith("gene_ids")]
        gene_ids = adata.var[gene_id_columns].astype("string").bfill(axis=1).iloc[:, 0] if gene_id_columns else pd.Series(pd.NA, index=adata.var_names, dtype="string")
    gene_ids.index = adata.var_names.astype(str)
    gene_ids = gene_ids.mask(gene_ids.isna() | gene_ids.str.strip().eq(""), pd.Series(gene_ids.index, index=gene_ids.index))
    return gene_ids.astype(str)


def make_minimal_object(path: Path, dataset_name: str, source: str, celltype_column: str, compartment_column: str) -> tuple[ad.AnnData, pd.Series]:
    print(f"Loading {dataset_name}: {path}")
    adata = ad.read_h5ad(path)
    if COUNTS_LAYER not in adata.layers:
        raise KeyError(f"{path} does not contain the required {COUNTS_LAYER!r} layer.")
    missing_columns = [column for column in [celltype_column, compartment_column, "sample"] if column not in adata.obs.columns]
    if missing_columns:
        raise KeyError(f"{path} is missing required obs columns: {missing_columns}")
    counts = adata.layers[COUNTS_LAYER]
    counts = counts.tocsr() if sparse.issparse(counts) else sparse.csr_matrix(counts)
    original_barcodes = adata.obs_names.astype(str)
    exported_barcodes = pd.Index([f"{dataset_name}_{barcode}" for barcode in original_barcodes], name="barcode")
    if not exported_barcodes.is_unique:
        duplicated = exported_barcodes[exported_barcodes.duplicated()].unique().tolist()[:10]
        raise ValueError(f"Non-unique exported barcodes in {dataset_name}: {duplicated}")
    obs = pd.DataFrame(index=exported_barcodes)
    obs["cell_type"] = clean_text(adata.obs[celltype_column]).to_numpy()
    obs["compartment"] = clean_text(adata.obs[compartment_column]).to_numpy()
    obs["source"] = source
    obs["study"] = dataset_name
    obs["sample"] = clean_text(adata.obs["sample"]).to_numpy()
    obs["original_barcode"] = original_barcodes
    gene_ids = get_gene_ids(adata)
    minimal = ad.AnnData(X=counts, obs=obs, var=pd.DataFrame(index=pd.Index(adata.var_names.astype(str), name="gene_name")))
    del adata
    gc.collect()
    return minimal, gene_ids


def byte_array(values) -> np.ndarray:
    return np.asarray([str(value).encode("utf-8") for value in values])


def write_cellranger_h5(path: Path, counts_by_cell, barcodes: pd.Index, gene_names: pd.Index, gene_ids: np.ndarray) -> None:
    feature_by_barcode = counts_by_cell.T.tocsc()
    if feature_by_barcode.data.size and (np.nanmin(feature_by_barcode.data) < 0 or not np.allclose(feature_by_barcode.data, np.rint(feature_by_barcode.data))):
        raise ValueError("The counts layer contains negative or non-integer values.")
    feature_by_barcode.data = np.rint(feature_by_barcode.data).astype(np.int32)
    with h5py.File(path, "w") as handle:
        matrix = handle.create_group("matrix")
        matrix.create_dataset("barcodes", data=byte_array(barcodes), compression="gzip")
        matrix.create_dataset("data", data=feature_by_barcode.data, compression="gzip")
        matrix.create_dataset("indices", data=feature_by_barcode.indices.astype(np.int64), compression="gzip")
        matrix.create_dataset("indptr", data=feature_by_barcode.indptr.astype(np.int64), compression="gzip")
        matrix.create_dataset("shape", data=np.asarray(feature_by_barcode.shape, dtype=np.int64))
        features = matrix.create_group("features")
        features.create_dataset("_all_tag_keys", data=byte_array(["genome"]))
        features.create_dataset("feature_type", data=byte_array(["Gene Expression"] * len(gene_names)), compression="gzip")
        features.create_dataset("genome", data=byte_array([GENOME] * len(gene_names)), compression="gzip")
        features.create_dataset("id", data=byte_array(gene_ids), compression="gzip")
        features.create_dataset("name", data=byte_array(gene_names), compression="gzip")


def validate_outputs(h5_path: Path, annotation_path: Path, expected_cells: int, expected_genes: int) -> None:
    annotations = pd.read_csv(annotation_path)
    with h5py.File(h5_path, "r") as handle:
        shape = tuple(handle["matrix/shape"][:])
        h5_barcodes = [value.decode("utf-8") for value in handle["matrix/barcodes"][:]]
    if shape != (expected_genes, expected_cells):
        raise ValueError(f"Unexpected Cell Ranger matrix shape {shape}; expected {(expected_genes, expected_cells)}.")
    if annotations["barcode"].tolist() != h5_barcodes:
        raise ValueError("Annotation barcodes do not exactly match the Cell Ranger matrix barcodes.")


def create_reference(dataset_name: str, path: Path, source: str, celltype_column: str, compartment_column: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    reference, gene_ids = make_minimal_object(path, dataset_name, source, celltype_column, compartment_column)
    archive_path = OUTPUT_DIRECTORY / f"{dataset_name}_xenium_reference.zip"
    h5_name = f"{dataset_name}_xenium_reference_feature_bc_matrix.h5"
    annotation_name = f"{dataset_name}_xenium_reference_cell_annotations.csv"
    with TemporaryDirectory(dir=OUTPUT_DIRECTORY) as temporary_directory:
        h5_path = Path(temporary_directory) / h5_name
        annotation_path = Path(temporary_directory) / annotation_name
        annotations = reference.obs.reset_index()
        annotations.to_csv(annotation_path, index=False)
        write_cellranger_h5(h5_path, reference.X, reference.obs_names, reference.var_names, gene_ids.reindex(reference.var_names).to_numpy())
        validate_outputs(h5_path, annotation_path, reference.n_obs, reference.n_vars)
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            archive.write(h5_path, arcname=h5_name)
            archive.write(annotation_path, arcname=annotation_name)
    print(f"Saved {archive_path}")
    print(f"Reference shape for {dataset_name}: {reference.shape}")
    del reference
    gc.collect()
    return archive_path


OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
created_archives = [create_reference("PMID_39198675", NORMAL_PATH, "Normal_reference", NORMAL_CELLTYPE_COLUMN, NORMAL_COMPARTMENT_COLUMN)]
created_archives.extend([create_reference(study, Path(STUDY_PATH_TEMPLATE.format(study=study)), "Tumor_study", STUDY_CELLTYPE_COLUMN, STUDY_COMPARTMENT_COLUMN) for study in STUDIES])
print("Created reference archives:")
for created_archive in created_archives:
    print(created_archive)
