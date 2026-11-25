"""Tests for the Stage 4 group-aware split."""
import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_prep import SPLITS, assign_splits


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    rows, gid = [], 0
    for label, ngroups in (("A", 30), ("B", 24), ("C", 90)):
        for _ in range(ngroups):
            for k in range(int(rng.integers(1, 8))):
                rows.append({"label": label, "group_id": gid, "file": f"{label}_{gid}_{k}.jpg"})
            gid += 1
    return pd.DataFrame(rows)


def test_groups_never_cross_splits():
    df = synthetic()
    a = assign_splits(df, seed=1, tries=200)
    df["s"] = df.group_id.map(a)
    assert (df.groupby("group_id").s.nunique() == 1).all()


def test_every_class_in_every_split_with_minimum_groups():
    df = synthetic()
    a = assign_splits(df, seed=1, tries=200)
    df["s"] = df.group_id.map(a)
    counts = df.groupby(["label", "s"]).group_id.nunique().unstack(fill_value=0)
    assert set(counts.columns) == set(SPLITS)
    assert (counts >= config.MIN_GROUPS_PER_SPLIT).all().all()


def test_deterministic_and_independent_of_group_numbering_and_row_order():
    df = synthetic()
    a1 = assign_splits(df, seed=7, tries=200)
    shuffled = df.sample(frac=1, random_state=3).copy()
    shuffled["group_id"] = shuffled.group_id + 1000          # renumbered groups
    a2 = assign_splits(shuffled, seed=7, tries=200)
    m1 = {g: s for g, s in a1.items()}
    m2 = {g - 1000: s for g, s in a2.items()}
    assert m1 == m2


def test_too_few_groups_is_rejected():
    df = pd.DataFrame({"label": ["A"] * 2, "group_id": [0, 1], "file": list("ab")})
    with pytest.raises(ValueError):
        assign_splits(df, tries=10)


def test_class_with_few_groups_still_gets_every_split():
    df = pd.DataFrame({"label": ["A"] * 5, "group_id": [0, 1, 2, 3, 4], "file": list("abcde")})
    a = assign_splits(df, tries=50)
    assert sorted(a.values()).count("test") >= 1 and sorted(a.values()).count("val") >= 1
    assert sorted(a.values()).count("train") >= 1


# --- checks on the real prepared data (skipped until Stage 4 has been run) ---
real = config.SPLIT_CSV.exists()


@pytest.mark.skipif(not real, reason="run python -m src.data_prep first")
def test_real_split_has_no_group_overlap_and_no_excluded_images():
    df = pd.read_csv(config.SPLIT_CSV)
    assert (df.groupby("group_id").assigned_split.nunique() == 1).all()
    ex = pd.read_csv(config.EXCLUDED_CSV)
    assert set(df.file).isdisjoint(set(ex.file))


@pytest.mark.skipif(not real, reason="run python -m src.data_prep first")
def test_real_split_files_exist_and_classes_present():
    df = pd.read_csv(config.SPLIT_CSV)
    for _, r in df.iterrows():
        assert (config.DATA_PROCESSED / r.assigned_split / r.label / r.file).is_file()
    counts = df.groupby(["label", "assigned_split"]).group_id.nunique().unstack(fill_value=0)
    assert set(counts.columns) == set(SPLITS)
    assert (counts >= 1).all().all()
