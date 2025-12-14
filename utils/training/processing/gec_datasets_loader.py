#!/usr/bin/env python3
"""
Thin wrappers around the 'gec-datasets' library to load public GEC datasets
for training mixtures with minimal local processing.

We keep this isolated from legacy loaders to preserve reproducibility.
"""

from typing import List, Dict, Optional


def _pairs_from_gec_dataset(ds) -> List[Dict]:
    pairs: List[Dict] = []
    # Prefer explicit pairs if the lib exposes them
    if hasattr(ds, 'pairs') and ds.pairs is not None:
        for i, (src, tgt) in enumerate(ds.pairs):
            pairs.append({'src_text': src, 'tgt_text': tgt, 'id': f'gec_ds_{i}'})
        return pairs
    # Common case: srcs + tgts
    if hasattr(ds, 'srcs') and hasattr(ds, 'tgts') and ds.srcs is not None and ds.tgts is not None:
        for i, (src, tgt) in enumerate(zip(ds.srcs, ds.tgts)):
            pairs.append({'src_text': src, 'tgt_text': tgt, 'id': f'gec_ds_{i}'})
        return pairs
    # Fallback for dev/test style sets: srcs + refs[0]
    if hasattr(ds, 'srcs') and hasattr(ds, 'refs') and ds.srcs is not None and ds.refs:
        ref0 = ds.refs[0]
        for i, (src, tgt) in enumerate(zip(ds.srcs, ref0)):
            pairs.append({'src_text': src, 'tgt_text': tgt, 'id': f'gec_ds_{i}'})
        return pairs
    return pairs


def _load_dataset_by_id(gec, dataset_id: str) -> List[Dict]:
    ds = gec.load(dataset_id)
    return _pairs_from_gec_dataset(ds)


def load_wilocness_train_from_gec(base_path: str) -> List[Dict]:
    from gec_datasets import GECDatasets
    gec = GECDatasets(base_path=base_path)
    return _load_dataset_by_id(gec, 'wi-locness-train')


def load_fce_train_from_gec(base_path: str) -> List[Dict]:
    from gec_datasets import GECDatasets
    gec = GECDatasets(base_path=base_path)
    return _load_dataset_by_id(gec, 'fce-train')


def load_troy_1bw_from_gec(base_path: str, split: str = 'train') -> List[Dict]:
    from gec_datasets import GECDatasets
    gec = GECDatasets(base_path=base_path)
    ds_id = 'troy-1bw-train' if split == 'train' else 'troy-1bw-dev'
    return _load_dataset_by_id(gec, ds_id)


def load_troy_blogs_from_gec(base_path: str, split: str = 'train') -> List[Dict]:
    from gec_datasets import GECDatasets
    gec = GECDatasets(base_path=base_path)
    ds_id = 'troy-blogs-train' if split == 'train' else 'troy-blogs-dev'
    return _load_dataset_by_id(gec, ds_id)


def load_lang8_train_from_gec(base_path: str) -> List[Dict]:
    """
    Load Lang-8 train via gec-datasets (requires placing licensed tarball as instructed by the package).
    """
    from gec_datasets import GECDatasets
    gec = GECDatasets(base_path=base_path)
    return _load_dataset_by_id(gec, 'lang8-train')


