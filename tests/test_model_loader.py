"""Tests for pipeline.model_loader.

Strategy: mock joblib.load and json.load so tests run without real
model files. Each test group covers one failure mode or happy path.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Re-import after each invalidation so lru_cache is always fresh
import pipeline.model_loader as ml
from pipeline.model_loader import (
    get_club_models,
    get_intl_models,
    invalidate_cache,
    is_club_available,
    is_intl_available,
)

_VALID_PKL = pickle.dumps(0)  # valid pickle content for dummy files


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_cache():
    """Clear lru_cache before and after every test."""
    invalidate_cache()
    yield
    invalidate_cache()


def _make_exists(present: list[str]):
    """Return an os.path.exists side_effect that only approves *present* files."""
    def _exists(path: str) -> bool:
        return any(path.endswith(f) for f in present)
    return _exists


# ── get_intl_models: happy path ─────────────────────────────────────────────

class TestGetIntlModelsHappyPath:
    def test_returns_dict_with_required_keys(self, tmp_path, monkeypatch):
        monkeypatch.setenv("DATA_DIR", str(tmp_path))
        # Create dummy required files
        required = ["dc_model.pkl", "xgb_model_29.pkl", "elo_ratings.pkl"]
        for f in required:
            (tmp_path / f).write_bytes(_VALID_PKL)

        dummy = MagicMock()
        with patch("pipeline.model_loader.joblib.load", return_value=dummy), \
             patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            models = get_intl_models()

        assert set(models.keys()) >= {"dc", "xgb29", "elo"}

    def test_optional_models_are_none_when_absent(self, tmp_path, monkeypatch):
        required = ["dc_model.pkl", "xgb_model_29.pkl", "elo_ratings.pkl"]
        for f in required:
            (tmp_path / f).write_bytes(_VALID_PKL)

        dummy = MagicMock()
        with patch("pipeline.model_loader.joblib.load", return_value=dummy), \
             patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            models = get_intl_models()

        # No optional files were created
        assert models["calibrators"] is None
        assert models["xgb30"] is None
        assert models["xgb_simple"] is None
        assert models["cal_simple"] is None

    def test_optional_models_loaded_when_present(self, tmp_path):
        all_files = [
            "dc_model.pkl", "xgb_model_29.pkl", "elo_ratings.pkl",
            "calibrators.pkl", "xgb_model_30.pkl",
            "xgb_model_simple.pkl", "calibrators_simple.pkl",
        ]
        for f in all_files:
            (tmp_path / f).write_bytes(_VALID_PKL)

        dummy = MagicMock()
        with patch("pipeline.model_loader.joblib.load", return_value=dummy), \
             patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            models = get_intl_models()

        assert models["calibrators"] is dummy
        assert models["xgb30"] is dummy

    def test_result_is_cached(self, tmp_path):
        required = ["dc_model.pkl", "xgb_model_29.pkl", "elo_ratings.pkl"]
        for f in required:
            (tmp_path / f).write_bytes(_VALID_PKL)

        load_mock = MagicMock(return_value=MagicMock())
        with patch("pipeline.model_loader.joblib.load", load_mock), \
             patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            r1 = get_intl_models()
            r2 = get_intl_models()

        assert r1 is r2
        # joblib.load called exactly 3 times (3 required files), not 6
        assert load_mock.call_count == 3


# ── get_intl_models: failure modes ─────────────────────────────────────────

class TestGetIntlModelsMissingFiles:
    @pytest.mark.parametrize("missing", [
        "dc_model.pkl",
        "xgb_model_29.pkl",
        "elo_ratings.pkl",
    ])
    def test_raises_file_not_found_for_each_required(self, tmp_path, missing):
        required = ["dc_model.pkl", "xgb_model_29.pkl", "elo_ratings.pkl"]
        for f in required:
            if f != missing:
                (tmp_path / f).write_bytes(_VALID_PKL)

        with patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            with pytest.raises(FileNotFoundError, match=missing):
                get_intl_models()

    def test_error_message_includes_training_hint(self, tmp_path):
        with patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            with pytest.raises(FileNotFoundError, match="training pipeline"):
                get_intl_models()


# ── get_club_models: happy path ─────────────────────────────────────────────

class TestGetClubModelsHappyPath:
    def test_returns_dict_with_required_keys(self, tmp_path):
        required = ["dc_model_club.pkl", "xgb_model_club.pkl", "elo_club.pkl"]
        for f in required:
            (tmp_path / f).write_bytes(_VALID_PKL)

        dummy = MagicMock()
        with patch("pipeline.model_loader.joblib.load", return_value=dummy), \
             patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            models = get_club_models()

        assert models is not None
        assert set(models.keys()) >= {"dc", "xgb", "elo"}

    def test_json_files_loaded_correctly(self, tmp_path):
        required = ["dc_model_club.pkl", "xgb_model_club.pkl", "elo_club.pkl"]
        for f in required:
            (tmp_path / f).write_bytes(_VALID_PKL)

        form_data = {"Arsenal": [[2, 1], [1, 0]]}
        xg_data   = {"Arsenal": {"xg_proxy_5": 1.8}}
        (tmp_path / "form_club.json").write_text(json.dumps(form_data))
        (tmp_path / "xg_proxy_club.json").write_text(json.dumps(xg_data))

        with patch("pipeline.model_loader.joblib.load", return_value=MagicMock()), \
             patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            models = get_club_models()

        assert models["form"] == form_data
        assert models["xg"]   == xg_data


# ── get_club_models: missing required files ─────────────────────────────────

class TestGetClubModelsMissing:
    @pytest.mark.parametrize("missing", [
        "dc_model_club.pkl",
        "xgb_model_club.pkl",
        "elo_club.pkl",
    ])
    def test_returns_none_when_required_file_absent(self, tmp_path, missing):
        required = ["dc_model_club.pkl", "xgb_model_club.pkl", "elo_club.pkl"]
        for f in required:
            if f != missing:
                (tmp_path / f).write_bytes(_VALID_PKL)

        with patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            result = get_club_models()

        assert result is None  # graceful degradation, not an exception

    def test_returns_none_when_all_club_files_absent(self, tmp_path):
        with patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            assert get_club_models() is None


# ── invalidate_cache ─────────────────────────────────────────────────────────

class TestInvalidateCache:
    def test_forces_reload_on_next_call(self, tmp_path):
        required = ["dc_model.pkl", "xgb_model_29.pkl", "elo_ratings.pkl"]
        for f in required:
            (tmp_path / f).write_bytes(_VALID_PKL)

        call_count = {"n": 0}
        def counting_load(path):
            call_count["n"] += 1
            return MagicMock()

        with patch("pipeline.model_loader.joblib.load", side_effect=counting_load), \
             patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            get_intl_models()   # load #1 (3 files)
            invalidate_cache()
            get_intl_models()   # load #2 (3 files)

        assert call_count["n"] == 6  # loaded twice, 3 files each


# ── helper functions ─────────────────────────────────────────────────────────

class TestHelpers:
    def test_is_club_available_true(self, tmp_path):
        required = ["dc_model_club.pkl", "xgb_model_club.pkl", "elo_club.pkl"]
        for f in required:
            (tmp_path / f).write_bytes(_VALID_PKL)

        with patch("pipeline.model_loader.joblib.load", return_value=MagicMock()), \
             patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            assert is_club_available() is True

    def test_is_club_available_false(self, tmp_path):
        with patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            assert is_club_available() is False

    def test_is_intl_available_false_when_files_missing(self, tmp_path):
        with patch("pipeline.model_loader.DATA_DIR", str(tmp_path)):
            invalidate_cache()
            assert is_intl_available() is False
