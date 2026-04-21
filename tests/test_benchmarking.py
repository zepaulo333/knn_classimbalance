"""Tests for benchmarking resume, backup, and replace-algorithm utilities."""

import pandas as pd
import pytest

from src.evaluation.benchmarking import _backup_csv, _drop_algorithm


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_csv(tmp_path):
    """CSV with two algorithms, two datasets, two rows each (8 rows total)."""
    df = pd.DataFrame({
        "algorithm": ["A", "A", "A", "A", "B", "B", "B", "B"],
        "dataset":   ["d1", "d1", "d2", "d2", "d1", "d1", "d2", "d2"],
        "fold":      [0, 1, 0, 1, 0, 1, 0, 1],
        "f1":        [0.8, 0.7, 0.9, 0.85, 0.6, 0.65, 0.7, 0.75],
    })
    p = tmp_path / "results.csv"
    df.to_csv(p, index=False)
    return p


# ── _backup_csv ───────────────────────────────────────────────────────────────

def test_backup_creates_file(sample_csv, tmp_path):
    backup = _backup_csv(sample_csv)
    assert backup is not None
    assert backup.exists()
    assert backup != sample_csv


def test_backup_content_matches_original(sample_csv):
    backup = _backup_csv(sample_csv)
    original = pd.read_csv(sample_csv)
    backed_up = pd.read_csv(backup)
    pd.testing.assert_frame_equal(original, backed_up)


def test_backup_original_unchanged(sample_csv):
    original = pd.read_csv(sample_csv)
    _backup_csv(sample_csv)
    after = pd.read_csv(sample_csv)
    pd.testing.assert_frame_equal(original, after)


def test_backup_nonexistent_returns_none(tmp_path):
    assert _backup_csv(tmp_path / "missing.csv") is None


def test_backup_empty_file_returns_none(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert _backup_csv(p) is None


def test_backup_name_contains_timestamp(sample_csv):
    backup = _backup_csv(sample_csv)
    assert "backup_" in backup.name
    assert backup.suffix == sample_csv.suffix


# ── _drop_algorithm ───────────────────────────────────────────────────────────

def test_drop_removes_target_algorithm(sample_csv):
    _drop_algorithm(sample_csv, "A")
    result = pd.read_csv(sample_csv)
    assert "A" not in result["algorithm"].values


def test_drop_preserves_other_algorithms(sample_csv):
    _drop_algorithm(sample_csv, "A")
    result = pd.read_csv(sample_csv)
    assert set(result["algorithm"].unique()) == {"B"}
    assert len(result) == 4


def test_drop_creates_backup(sample_csv, tmp_path):
    before_files = set(tmp_path.iterdir())
    _drop_algorithm(sample_csv, "A")
    after_files = set(tmp_path.iterdir())
    new_files = after_files - before_files
    assert len(new_files) == 1
    backup = next(iter(new_files))
    assert "backup_" in backup.name


def test_drop_backup_preserves_all_rows(sample_csv):
    backup = _drop_algorithm(sample_csv, "A")
    backed_up = pd.read_csv(backup)
    assert len(backed_up) == 8
    assert set(backed_up["algorithm"].unique()) == {"A", "B"}


def test_drop_algorithm_not_present_leaves_csv_intact(sample_csv):
    original = pd.read_csv(sample_csv)
    _drop_algorithm(sample_csv, "C")
    result = pd.read_csv(sample_csv)
    pd.testing.assert_frame_equal(original, result)


def test_drop_nonexistent_csv_does_not_raise(tmp_path):
    _drop_algorithm(tmp_path / "missing.csv", "A")


def test_drop_returns_backup_path(sample_csv):
    backup = _drop_algorithm(sample_csv, "A")
    assert backup is not None
    assert backup.exists()


def test_two_consecutive_drops_produce_two_backups(sample_csv, tmp_path):
    import time
    _drop_algorithm(sample_csv, "A")
    time.sleep(1)  # ensure distinct timestamps
    _drop_algorithm(sample_csv, "B")
    backups = [f for f in tmp_path.iterdir() if "backup_" in f.name]
    assert len(backups) == 2
