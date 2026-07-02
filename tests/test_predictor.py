"""Tests for pipeline.predictor.

All external dependencies (model artifacts, normalizers, form data) are mocked.
Zero disk I/O or real model inference.
"""
from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pipeline.predictor import (
    predict_match_wrapper,
    predict_match_legacy,
    fallback_market_predict,
    _hda_result,
    _best_score,
    _recent_form_club,
    _build_xg_feat,
    _load_h2h_gd,
    _run_simple_model,
)


# ── pure helpers ──────────────────────────────────────────────────────────────

class TestHdaResult:
    def test_home_wins(self):   assert _hda_result(0.6, 0.2, 0.2) == 'H'
    def test_draw_wins(self):   assert _hda_result(0.2, 0.6, 0.2) == 'D'
    def test_away_wins(self):   assert _hda_result(0.2, 0.2, 0.6) == 'A'
    def test_exact_tie_hd(self): assert _hda_result(0.4, 0.4, 0.2) in ('H', 'D')


class TestBestScore:
    def test_returns_ints(self):
        bh, ba = _best_score(1.5, 0.8)
        assert isinstance(bh, int) and isinstance(ba, int)

    def test_home_favoured_home_scores_more(self):
        bh, ba = _best_score(3.0, 0.5)
        assert bh > ba

    def test_zero_lambda_does_not_crash(self):
        bh, ba = _best_score(0.0, 0.0)
        assert bh == 0 and ba == 0


class TestRecentFormClub:
    def test_empty_returns_defaults(self):
        assert _recent_form_club({}, 'X') == [0.5, 0.0, 0.0, 0.0]

    def test_all_wins(self):
        form = {'A': [(2, 0)] * 5}
        r = _recent_form_club(form, 'A', 5)
        assert r[0] == pytest.approx(1.0)
        assert r[1] == pytest.approx(2.0)
        assert r[2] == pytest.approx(0.0)

    def test_truncates_to_n(self):
        form = {'A': [(1, 0)] * 20}
        with patch('pipeline.predictor._recent_form_club', wraps=_recent_form_club):
            r = _recent_form_club(form, 'A', 5)
        assert r[0] == pytest.approx(1.0)

    def test_draw_counts_half(self):
        form = {'A': [(1, 1)] * 4}
        r = _recent_form_club(form, 'A', 4)
        assert r[0] == pytest.approx(0.5)


class TestBuildXgFeat:
    def test_length_8(self):
        xg = {'France': {'xg_proxy_5': 1.5, 'xg_proxy_12': 1.3, 'xg_streak': 3, 'xg_volatility': 0.2},
              'Germany': {'xg_proxy_5': 1.2, 'xg_proxy_12': 1.1, 'xg_streak': 2, 'xg_volatility': 0.1}}
        feat = _build_xg_feat(xg, 'France', 'Germany')
        assert len(feat) == 8

    def test_none_xg_returns_zeros(self):
        feat = _build_xg_feat(None, 'X', 'Y')
        assert feat == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    def test_missing_team_returns_zeros(self):
        feat = _build_xg_feat({}, 'France', 'Germany')
        assert all(v == 0.0 for v in feat)

    def test_streak_divided_by_10(self):
        xg = {'X': {'xg_streak': 5}}
        feat = _build_xg_feat(xg, 'X', 'Y')
        assert feat[2] == pytest.approx(0.5)


class TestLoadH2hGd:
    def test_returns_zero_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.setenv('DATA_DIR', str(tmp_path))
        assert _load_h2h_gd('France', 'Germany') == 0.0

    def test_reads_correct_gd(self, tmp_path, monkeypatch):
        import json
        monkeypatch.setenv('DATA_DIR', str(tmp_path))
        cache = {'France||Germany': [10, 20, 8]}
        (tmp_path / 'h2h_cache_club.json').write_text(json.dumps(cache))
        # France is key[0] → gd = 20-8 = 12
        assert _load_h2h_gd('France', 'Germany') == 12

    def test_reverses_gd_for_away_team(self, tmp_path, monkeypatch):
        import json
        monkeypatch.setenv('DATA_DIR', str(tmp_path))
        cache = {'France||Germany': [10, 20, 8]}
        (tmp_path / 'h2h_cache_club.json').write_text(json.dumps(cache))
        assert _load_h2h_gd('Germany', 'France') == -12

    def test_returns_zero_on_exception(self):
        with patch('builtins.open', side_effect=OSError('boom')):
            assert _load_h2h_gd('X', 'Y') == 0.0


class TestRunSimpleModel:
    def test_returns_empty_when_no_model(self):
        label, conf = _run_simple_model(None, None, 0.5, [0.5]*4, [0.5]*4)
        assert label == '' and conf == 0.0

    def test_returns_prediction(self):
        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.6, 0.2, 0.2]])
        label, conf = _run_simple_model(mock_xgb, None, 0.5, [0.5]*4, [0.5]*4)
        assert label == 'H'
        assert conf == pytest.approx(0.6)

    def test_applies_calibration(self):
        mock_xgb = MagicMock()
        mock_xgb.predict_proba.return_value = np.array([[0.5, 0.3, 0.2]])
        mock_cal = {'home': MagicMock(), 'draw': MagicMock(), 'away': MagicMock()}
        mock_cal['home'].predict.return_value = [0.55]
        mock_cal['draw'].predict.return_value = [0.28]
        mock_cal['away'].predict.return_value = [0.17]
        label, conf = _run_simple_model(mock_xgb, mock_cal, 0.5, [0.5]*4, [0.5]*4)
        assert label == 'H'

    def test_returns_empty_on_exception(self):
        mock_xgb = MagicMock()
        mock_xgb.predict_proba.side_effect = RuntimeError('model error')
        label, conf = _run_simple_model(mock_xgb, None, 0.5, [0.5]*4, [0.5]*4)
        assert label == '' and conf == 0.0


# ── predict_match_legacy ──────────────────────────────────────────────────────

class TestPredictMatchLegacy:
    def _ts(self):
        return {
            'France':  {'attack': 1.5, 'defense': 0.8, 'm': 20},
            'Germany': {'attack': 1.2, 'defense': 1.0, 'm': 18},
        }

    def test_probs_sum_to_1(self):
        r = predict_match_legacy('France', 'Germany', self._ts(), 1.2, {'France': 1600, 'Germany': 1500})
        total = r['probs']['H'] + r['probs']['D'] + r['probs']['A']
        assert total == pytest.approx(1.0, abs=0.001)

    def test_returns_required_keys(self):
        r = predict_match_legacy('France', 'Germany', self._ts(), 1.2, {})
        assert {'probs', 'score', 'result', 'lambda_ft', 'model'} <= r.keys()

    def test_model_tag(self):
        r = predict_match_legacy('X', 'Y', {}, 1.2, {})
        assert r['model'] == 'legacy_poisson'

    def test_strong_home_elo_biases_toward_home(self):
        elo = {'France': 2000, 'Germany': 1000}
        r = predict_match_legacy('France', 'Germany', self._ts(), 1.2, elo)
        assert r['probs']['H'] > r['probs']['A']

    def test_missing_team_uses_defaults(self):
        r = predict_match_legacy('Unknown1', 'Unknown2', {}, 1.2, {})
        assert r['probs']['H'] + r['probs']['D'] + r['probs']['A'] == pytest.approx(1.0, abs=0.001)

    def test_lambda_clamped(self):
        ts = {'X': {'attack': 100.0, 'defense': 0.01, 'm': 5}}
        r = predict_match_legacy('X', 'Y', ts, 5.0, {})
        assert r['lambda_ft']['home'] <= 5.0


# ── fallback_market_predict ───────────────────────────────────────────────────

class TestFallbackMarketPredict:
    def _row(self, h=1.85, d=3.50, a=4.20):
        return {'odds_h': h, 'odds_d': d, 'odds_a': a}

    def test_probs_sum_to_1(self):
        r = fallback_market_predict(self._row())
        assert sum(r['probs'].values()) == pytest.approx(1.0, abs=0.001)

    def test_model_tag(self):
        assert fallback_market_predict(self._row())['model'] == 'market_fallback'

    def test_returns_required_keys(self):
        r = fallback_market_predict(self._row())
        assert {'probs', 'score', 'result', 'lambda_ft', 'model'} <= r.keys()

    def test_zero_odds_does_not_crash(self):
        r = fallback_market_predict({'odds_h': 0, 'odds_d': 0, 'odds_a': 0})
        assert r['probs']['H'] + r['probs']['D'] + r['probs']['A'] == pytest.approx(1.0, abs=0.01)

    def test_heavy_home_favourite_yields_high_H(self):
        r = fallback_market_predict(self._row(h=1.20, d=6.00, a=12.0))
        assert r['probs']['H'] > 0.6


# ── predict_match_wrapper ─────────────────────────────────────────────────────

class TestPredictMatchWrapper:
    def _mock_result(self, model='club_hybrid'):
        return {
            'probs': {'H': 0.55, 'D': 0.25, 'A': 0.20},
            'score': '2-1', 'result': 'H',
            'lambda_ft': {'home': 1.8, 'away': 0.9},
            'model': model,
        }

    def test_club_path_used_first(self):
        with patch('pipeline.predictor._try_club_predict', return_value=self._mock_result('club_hybrid')), \
             patch('pipeline.predictor._try_hybrid_predict') as mock_intl:
            r = predict_match_wrapper('France', 'Germany')
        mock_intl.assert_not_called()
        assert r['source'] == 'club'

    def test_falls_back_to_intl(self):
        with patch('pipeline.predictor._try_club_predict', return_value=None), \
             patch('pipeline.predictor._try_hybrid_predict', return_value=self._mock_result('hybrid')):
            r = predict_match_wrapper('France', 'Germany')
        assert r['source'] == 'intl'

    def test_returns_none_when_both_fail(self):
        with patch('pipeline.predictor._try_club_predict', return_value=None), \
             patch('pipeline.predictor._try_hybrid_predict', return_value=None):
            r = predict_match_wrapper('X', 'Y')
        assert r is None

    def test_365scores_adjustment_applied(self):
        base = self._mock_result()
        adjusted_probs = {'H': 0.50, 'D': 0.28, 'A': 0.22}
        mock_adj = MagicMock(return_value=adjusted_probs)
        mock_module = MagicMock()
        mock_module.adjust_with_365scores = mock_adj

        with patch('pipeline.predictor._try_club_predict', return_value=base), \
             patch.dict('sys.modules', {'scores365_adjuster': mock_module}):
            r = predict_match_wrapper('France', 'Germany')

        assert r['probs'] == adjusted_probs
        assert r.get('scores365_adjusted') is True

    def test_365scores_failure_does_not_crash(self):
        base = self._mock_result()
        with patch('pipeline.predictor._try_club_predict', return_value=base), \
             patch.dict('sys.modules', {'scores365_adjuster': None}):
            r = predict_match_wrapper('France', 'Germany')
        assert r is not None
        assert 'probs' in r
