"""pipeline — public API surface for the JCZQ prediction system.

All seven sub-modules are re-exported here so callers never need to
remember the exact sub-module path.

Typical usage
-------------
    from pipeline import predict_match_wrapper, train, build_prediction_bundle
    from pipeline import (
        fetch_league_history, get_today_matches,
        scrape_500_odds_today, apply_euro_fallback,
        load_shared_models, load_club_models,
    )
"""

# ── probability ──────────────────────────────────────────────────────────────
from pipeline.probability import (
    poisson_pmf,
    elo_expected,
    dc_tau,
    compute_rq_probs,
    implied_probs_from_odds,
    compute_goals_distribution,
    compute_score_topn,
    compute_htft_topn_math,
    compute_dynamic_xgb_weight,
    rps_score,
    brier_decomposition_multiclass,
    quick_validate,
    format_pct,
)

# ── model_loader ─────────────────────────────────────────────────────────────
from pipeline.model_loader import (
    load_shared_models,
    load_club_models,
    get_shared_models,
    get_club_models,
)

# ── scraper ───────────────────────────────────────────────────────────────────
from pipeline.scraper import (
    scrape_500_odds_today,
    fetch_live_odds_map,
    apply_euro_fallback,
)

# ── data_loader ───────────────────────────────────────────────────────────────
from pipeline.data_loader import (
    api_get,
    fetch_league_history,
    get_today_matches,
    load_365scores_today,
    build_365_map,
)

# ── predictor ─────────────────────────────────────────────────────────────────
from pipeline.predictor import (
    predict_match_wrapper,
    predict_match_legacy,
    fallback_market_predict,
)

# ── trainer ───────────────────────────────────────────────────────────────────
from pipeline.trainer import train

# ── bundle_builder ────────────────────────────────────────────────────────────
from pipeline.bundle_builder import (
    build_prediction_bundle,
    print_match_bundle,
    record_prediction,
    compute_bet_action,
    compute_htft_topn,
    pick_best_htft,
    top_market_label,
    estimate_vote_fusion_alpha,
    ensure_log_has_source_fields,
    patch_logged_metadata,
)

__all__ = [
    # probability
    'poisson_pmf', 'elo_expected', 'dc_tau', 'compute_rq_probs',
    'implied_probs_from_odds', 'compute_goals_distribution',
    'compute_score_topn', 'compute_htft_topn_math',
    'compute_dynamic_xgb_weight', 'rps_score',
    'brier_decomposition_multiclass', 'quick_validate', 'format_pct',
    # model_loader
    'load_shared_models', 'load_club_models',
    'get_shared_models', 'get_club_models',
    # scraper
    'scrape_500_odds_today', 'fetch_live_odds_map', 'apply_euro_fallback',
    # data_loader
    'api_get', 'fetch_league_history', 'get_today_matches',
    'load_365scores_today', 'build_365_map',
    # predictor
    'predict_match_wrapper', 'predict_match_legacy', 'fallback_market_predict',
    # trainer
    'train',
    # bundle_builder
    'build_prediction_bundle', 'print_match_bundle', 'record_prediction',
    'compute_bet_action', 'compute_htft_topn', 'pick_best_htft',
    'top_market_label', 'estimate_vote_fusion_alpha',
    'ensure_log_has_source_fields', 'patch_logged_metadata',
]
