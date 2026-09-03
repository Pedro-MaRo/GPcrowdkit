"""Real crowdsourcing dataset loader for AI4SkINv2-2.

AI4SkINv2-2 is a digital-pathology dataset of skin-lesion whole-slide images
(WSIs), each graded on a 6-point ordinal scale (classes ``0``-``5``) by up to
10 pathologists ("Marker_1".."Marker_10" in its label CSVs). A missing grade
is encoded either as an empty cell (the pathologist was never shown that
slide) or as ``-1`` (shown, but marked non-diagnostic); both mean "no usable
annotation" and are dropped when building the sparse crowd labels -- neither
value ever appears in the dataset's own precomputed aggregators (``MV``,
``DS``, ``GLAD``, ``MACE``), which this loader also carries through so a
GPCrowdModel can be checked against them, not just against a recomputed
majority vote.

Each WSI ships as a *bag* of patch-level feature vectors (shape ``[n_patches,
D]``, one of the ``CONCH``/``PLIP``/``UNI``/``VGG16IN`` backbones, with
``n_patches`` varying per slide) rather than gpcrowdkit's expected single
``[N, D]`` feature matrix. This loader mean-pools each bag into one vector per
slide -- the standard multiple-instance-learning reduction -- to fit the
format [CrowdLabels.from_pairs][gpcrowdkit.data.CrowdLabels.from_pairs]
expects, exactly as ``data/moons.py`` builds its own ``(Y_cr, Y_mask)`` pairs
before conversion.

The dataset's own ``train``/``val``/``test`` split already matches
gpcrowdkit's "annotated training pool" vs. "clean held-out generalisation
check" shape used by ``data/moons.py``: ``train`` and ``val`` carry per-worker
grades, ``test`` carries only the ground truth ``GT`` and no annotations at
all. ``val`` is merged into the training pool by default (see
``include_val_in_train``) since both are annotated the same way.

This dataset lives outside this repository (see ``data_root``), so no image
data or embeddings are committed here -- only the loader that adapts it.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gpcrowdkit.data import CrowdLabels

__all__ = ["AI4SkinCrowdData", "load_ai4skin_dataset", "load_config"]

_MARKER_COLUMNS = [f"Marker_{i}" for i in range(1, 11)]
_BASELINE_COLUMNS = ["MV", "DS", "GLAD", "MACE"]
_MISSING_MARKER_VALUES = {"", "-1"}  # not shown to that pathologist / marked non-diagnostic
_MISSING_BASELINE_SENTINEL = -1  # aggregator not computed for this item (e.g. too few valid annotations)


@dataclass
class AI4SkinCrowdData:
    """A loaded AI4SkINv2-2 crowdsourcing dataset, in gpcrowdkit's format.

    Mirrors [MoonsCrowdData][data.moons.MoonsCrowdData]'s shape so the same
    training/evaluation code works unchanged, plus two fields specific to a
    real (rather than synthetic) dataset: the original WSI identifiers, for
    traceability back to AI4SkINv2-2, and the dataset's own precomputed
    aggregator baselines to compare against.

    Attributes:
        X_train (np.ndarray): Mean-pooled training features. Shape ``[N, D]``.
        labels (CrowdLabels): Sparse crowd annotations over ``X_train``.
        z_train (np.ndarray): Ground-truth grade for each training item,
            hidden from the model and used only to score predictions.
        X_test (np.ndarray): Mean-pooled features for the held-out test WSIs,
            which carry no annotations at all.
        z_test (np.ndarray): Ground-truth grade for the test items.
        n_classes (int): Number of ordinal grades ``C`` (6).
        n_annotators (int): Number of pathologists ``A`` (10).
        class_names (list[str]): Human-readable class labels.
        wsi_train (np.ndarray): Original WSI identifier for each row of
            ``X_train``/``z_train``.
        wsi_test (np.ndarray): Original WSI identifier for each row of
            ``X_test``/``z_test``.
        baseline_preds (dict[str, np.ndarray]): AI4SkINv2-2's own
            precomputed aggregations of the same annotations -- majority
            vote (``"MV"``), Dawid-Skene (``"DS"``), GLAD and MACE -- aligned
            with ``X_train``/``z_train``. Missing entries (not every
            aggregator is computed for every item) are ``-1``.
    """

    X_train: np.ndarray
    labels: CrowdLabels
    z_train: np.ndarray
    X_test: np.ndarray
    z_test: np.ndarray
    n_classes: int
    n_annotators: int
    class_names: list[str]
    wsi_train: np.ndarray
    wsi_test: np.ndarray
    baseline_preds: dict[str, np.ndarray] = field(default_factory=dict)


def load_config(path: str | Path) -> dict:
    """Loads a dataset config from a YAML file.

    Args:
        path: Path to a YAML file such as ``data/ai4skin.yaml``. When the
            file has several commented-out configurations, only the first
            active one is parsed.

    Returns:
        dict: The parsed config, as passed to [load_ai4skin_dataset][data.ai4skin.load_ai4skin_dataset].
    """
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def _read_split(labels_dir: Path, split: str) -> list[dict]:
    with open(labels_dir / f"{split}.csv", newline="") as f:
        return list(csv.DictReader(f))


def _parse_marker(value: str) -> int | None:
    """Parses one ``Marker_i`` cell, returning ``None`` for a missing grade."""
    return None if value in _MISSING_MARKER_VALUES else int(value)


def _parse_baseline(value: str | None) -> int:
    """Parses one aggregator cell, using -1 where the column is absent or empty."""
    return _MISSING_BASELINE_SENTINEL if not value else int(value)


def _build_pairs(rows: list[dict]) -> list[np.ndarray]:
    """Builds the ``[S_n, 2]`` (annotator_id, grade) pairs `CrowdLabels.from_pairs` expects.

    Annotator ``a`` is the position of ``Marker_{a+1}`` in `_MARKER_COLUMNS`,
    so worker index and CSV column line up directly with no separate key
    table.
    """
    pairs = []
    for row in rows:
        row_pairs = [
            (a, grade)
            for a, col in enumerate(_MARKER_COLUMNS)
            if (grade := _parse_marker(row[col])) is not None
        ]
        pairs.append(np.array(row_pairs, dtype=int) if row_pairs else np.empty((0, 2), dtype=int))
    return pairs


def _pool_embeddings(wsi_ids: list[str], embeddings_dir: Path, backbone: str, pooling: str) -> np.ndarray:
    """Collapses each WSI's ``[n_patches, D]`` bag of patch embeddings into one ``[D]`` vector.

    ``n_patches`` varies per slide (a WSI is however many tissue patches it
    was tiled into), so this is the step that turns AI4SkINv2-2's
    multiple-instance-learning bags into the flat ``[N, D]`` matrix
    gpcrowdkit's `SVGPLatent` expects.
    """
    pool_fn = {"mean": np.mean, "max": np.max}.get(pooling)
    if pool_fn is None:
        raise ValueError(f"Unknown pooling method: {pooling!r} (expected 'mean' or 'max')")

    vectors = []
    for wsi_id in wsi_ids:
        bag = np.load(embeddings_dir / backbone / f"{wsi_id}.npy")
        vectors.append(pool_fn(bag, axis=0))
    return np.stack(vectors).astype(np.float64)


def load_ai4skin_dataset(cfg: dict) -> AI4SkinCrowdData:
    """Loads AI4SkINv2-2 as a crowdsourcing dataset in gpcrowdkit's format.

    Args:
        cfg: Dataset configuration, in the format of ``data/ai4skin.yaml``.
            Keys: ``data_root`` (path to the AI4SkINv2-2 checkout, default
            ``"../AI4SkINv2-2"``, resolved relative to the current working
            directory -- i.e. the repository root when run via
            ``./run.sh``), ``backbone`` (one of ``CONCH``, ``PLIP``, ``UNI``,
            ``VGG16IN``; default ``"CONCH"``), ``pooling`` (``"mean"`` or
            ``"max"``; default ``"mean"``), ``include_val_in_train`` (merge
            AI4SkINv2-2's annotated ``val`` split into the training pool;
            default ``True``), ``standardize`` (z-score features using
            train-split statistics, applied to both splits; default
            ``True``).

    Returns:
        AI4SkinCrowdData: Training features, sparse crowd labels, and a
        clean held-out test split, plus AI4SkINv2-2's own aggregator
        baselines.
    """
    data_root = Path(cfg.get("data_root", "../AI4SkINv2-2"))
    backbone = cfg.get("backbone", "CONCH")
    pooling = cfg.get("pooling", "mean")
    include_val_in_train = cfg.get("include_val_in_train", True)
    standardize = cfg.get("standardize", True)

    labels_dir = data_root / "labels"
    embeddings_dir = data_root / "embeddings_v3"

    train_rows = _read_split(labels_dir, "train")
    if include_val_in_train:
        train_rows = train_rows + _read_split(labels_dir, "val")
    test_rows = _read_split(labels_dir, "test")

    wsi_train = np.array([row["WSI"] for row in train_rows])
    wsi_test = np.array([row["WSI"] for row in test_rows])

    z_train = np.array([int(row["GT"]) for row in train_rows], dtype=np.int32)
    z_test = np.array([int(row["GT"]) for row in test_rows], dtype=np.int32)

    labels = CrowdLabels.from_pairs(_build_pairs(train_rows), num_items=len(train_rows))

    X_train = _pool_embeddings(list(wsi_train), embeddings_dir, backbone, pooling)
    X_test = _pool_embeddings(list(wsi_test), embeddings_dir, backbone, pooling)

    if standardize:
        mean, std = X_train.mean(axis=0), X_train.std(axis=0)
        std[std == 0] = 1.0
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    baseline_preds = {
        col: np.array([_parse_baseline(row.get(col)) for row in train_rows], dtype=np.int32)
        for col in _BASELINE_COLUMNS
    }

    return AI4SkinCrowdData(
        X_train=X_train,
        labels=labels,
        z_train=z_train,
        X_test=X_test,
        z_test=z_test,
        n_classes=labels.num_classes,
        n_annotators=labels.num_workers,
        class_names=[f"class_{i}" for i in range(labels.num_classes)],
        wsi_train=wsi_train,
        wsi_test=wsi_test,
        baseline_preds=baseline_preds,
    )
