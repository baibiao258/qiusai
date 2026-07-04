"""Lazy model loader for all joblib / JSON artefacts.

Replaces the 12 module-level ``global _xxx = None`` variables in
daily_jczq.py with ``functools.lru_cache``-backed loaders.

Key properties
--------------
- Thread-safe single-load: lru_cache guarantees the body runs once.
- Testable: call ``invalidate_cache()`` between test cases to reset state.
- Fail-fast: missing *required* files raise FileNotFoundError immediately
  instead of surfacing as AttributeError deep in prediction code.
- Optional artefacts (calibrators, A/B model, simple model) are loaded
  only when present and returned as ``None`` otherwise.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from typing import Optional

import joblib

from config.settings import DATA_DIR

logger = logging.getLogger(__name__)

# ── 必须存在的文件 ──────────────────────────────────────────────────────────
_INTL_REQUIRED = {
    "dc":  "dc_model.pkl",
    "xgb": "xgb_model_29.pkl",
    "elo": "elo_ratings.pkl",
}

# ── 可选文件（缺失时对应 key 为 None）──────────────────────────────────────
_INTL_OPTIONAL = {
    "calibrators": "calibrators.pkl",
    "xgb30":       "xgb_model_30.pkl",
    "xgb_simple":  "xgb_model_simple.pkl",
    "cal_simple":  "calibrators_simple.pkl",
}

_CLUB_REQUIRED = {
    "dc":  "dc_model_club.pkl",
    "xgb": "xgb_model_club.pkl",
    "elo": "elo_club.pkl",
}

_CLUB_OPTIONAL = {
    "calibrators": "calibrators_club.pkl",
}

_CLUB_JSON = {
    "form":     "form_club.json",
    "xg_proxy": "xg_proxy_club.json",
    "xg_real":  "xg_real_club.json",
}


# ── 国际赛模型 ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_intl_models() -> dict:
    """Load and cache international-match models.

    Returns a dict with keys:
        dc, xgb29, elo            – always present (raises on missing)
        calibrators, xgb30,
        xgb_simple, cal_simple    – present only when file exists (else None)

    Raises
    ------
    FileNotFoundError
        If any *required* model file is absent.
    """
    models: dict = {}

    # Required
    for key, filename in _INTL_REQUIRED.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Required international model missing: {path}\n"
                "Run the training pipeline before starting predictions."
            )
        models[key] = joblib.load(path)
        logger.debug("Loaded intl model [%s] from %s", key, path)

    # Optional
    for key, filename in _INTL_OPTIONAL.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            models[key] = joblib.load(path)
            logger.debug("Loaded optional intl model [%s] from %s", key, path)
        else:
            models[key] = None
            logger.debug("Optional intl model [%s] not found at %s, skipping", key, path)

    return models


# ── 俱乐部模型 ──────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_club_models() -> Optional[dict]:
    """Load and cache club models.

    Returns ``None`` when *any* required club model file is missing
    (this is a normal condition early in the season when club data
    has not yet been built).

    Returns a dict with keys:
        dc, xgb, elo              – always present when return is not None
        calibrators               – present only when file exists (else None)
        form, xg                  – dict loaded from JSON (else None)
    """
    # Check required files first; return None silently if absent
    for filename in _CLUB_REQUIRED.values():
        if not os.path.exists(os.path.join(DATA_DIR, filename)):
            logger.info(
                "Club model file %s not found – club route disabled", filename
            )
            return None

    models: dict = {}

    for key, filename in _CLUB_REQUIRED.items():
        path = os.path.join(DATA_DIR, filename)
        models[key] = joblib.load(path)
        logger.debug("Loaded club model [%s] from %s", key, path)

    for key, filename in _CLUB_OPTIONAL.items():
        path = os.path.join(DATA_DIR, filename)
        models[key] = joblib.load(path) if os.path.exists(path) else None

    for key, filename in _CLUB_JSON.items():
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                models[key] = json.load(fh)
        else:
            models[key] = None

    return models


# ── 缓存管理 ────────────────────────────────────────────────────────────────

def invalidate_cache() -> None:
    """Clear all cached models.

    Call this in tests between cases that need different model fixtures,
    or after updating model files on disk during a long-running process.

    Example
    -------
    >>> from pipeline.model_loader import get_intl_models, invalidate_cache
    >>> invalidate_cache()
    >>> models = get_intl_models()   # reloads from disk
    """
    get_intl_models.cache_clear()
    get_club_models.cache_clear()
    logger.debug("Model loader cache invalidated")


def is_club_available() -> bool:
    """Return True if club models are loaded and ready."""
    return get_club_models() is not None


def is_intl_available() -> bool:
    """Return True if international models can be loaded without error."""
    try:
        get_intl_models()
        return True
    except FileNotFoundError:
        return False
