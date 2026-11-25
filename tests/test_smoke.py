"""Smoke test: the project layout and settings are importable (Stage 2)."""
from src import config


def test_directories_exist():
    for path in (config.DATA_RAW, config.DATA_PROCESSED, config.DATA_SPLITS,
                 config.MODELS_DIR, config.REPORTS_DIR):
        assert path.is_dir(), f"missing directory: {path}"


def test_split_ratios_sum_to_one():
    assert abs(sum(config.SPLIT_RATIOS.values()) - 1.0) < 1e-9


def test_input_size_matches_resnet50():
    assert config.IMAGE_SIZE == (224, 224)
