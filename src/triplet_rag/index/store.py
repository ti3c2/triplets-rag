"""FAISS index wrapper.

Stores: index.faiss + id_map.parquet (rowid -> chunk/question identifier).
We always normalize embeddings before adding so inner-product == cosine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from loguru import logger

from ..config import IndexerConfig
from ..utils.io import has_success, read_parquet, touch_success, write_parquet


@dataclass
class IndexBundle:
    index: faiss.Index
    id_map: pd.DataFrame  # columns: rowid, item_type, item_id, plus optional aux columns


def build_faiss(
    embeddings: np.ndarray,
    cfg: IndexerConfig,
) -> faiss.Index:
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    n, d = embeddings.shape
    if cfg.kind == "faiss_flat":
        index = faiss.IndexFlatIP(d) if cfg.metric == "ip" else faiss.IndexFlatL2(d)
    elif cfg.kind == "faiss_hnsw":
        if cfg.metric == "ip":
            index = faiss.IndexHNSWFlat(d, cfg.hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            index = faiss.IndexHNSWFlat(d, cfg.hnsw_m, faiss.METRIC_L2)
        index.hnsw.efConstruction = cfg.hnsw_ef_construction
        index.hnsw.efSearch = cfg.hnsw_ef_search
    else:
        raise ValueError(f"Unknown indexer kind: {cfg.kind}")
    index.add(embeddings)
    logger.info(f"Built {cfg.kind} index with {n} vectors, dim={d}")
    return index


def save_bundle(bundle: IndexBundle, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(bundle.index, str(out_dir / "index.faiss"))
    write_parquet(bundle.id_map, out_dir / "id_map.parquet")
    touch_success(
        out_dir / "_SUCCESS.json",
        {"n_vectors": bundle.index.ntotal, "n_id_map_rows": len(bundle.id_map)},
    )


def load_bundle(out_dir: Path) -> IndexBundle:
    index = faiss.read_index(str(out_dir / "index.faiss"))
    id_map = read_parquet(out_dir / "id_map.parquet")
    return IndexBundle(index=index, id_map=id_map)


def search(
    bundle: IndexBundle,
    query_embeddings: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    if query_embeddings.dtype != np.float32:
        query_embeddings = query_embeddings.astype(np.float32)
    distances, indices = bundle.index.search(query_embeddings, top_k)
    return distances, indices


def index_exists(out_dir: Path) -> bool:
    return has_success(out_dir / "_SUCCESS.json") and (out_dir / "index.faiss").exists()
