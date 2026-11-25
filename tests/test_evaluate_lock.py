"""The test-set lock (Stage 7)."""
from src.evaluate import already_evaluated


def test_lock_detection(tmp_path):
    lock = tmp_path / "lock.json"
    assert not already_evaluated(lock)
    lock.write_text("{}")
    assert already_evaluated(lock)
