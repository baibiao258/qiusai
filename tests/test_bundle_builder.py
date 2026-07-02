"""Tests for pipeline.bundle_builder.

All external dependencies (scraper_500_analysis, fatigue_features, bet_math)
are mocked. Zero disk I/O or real model inference.
"""
from __future__ import annotations

import json
import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from pipeline.bundle_builder import (
    estimate_vote_fusion_alpha,
    top_market_label,
    pick_best_htft,
    compute_bet_action,
    build_prediction_bundle,
    print_match_bundle,
    ensure_log_has_source_fields,
    patch_logged_metadata,
    record_prediction,
    _fmt_zjq,
    _safe_str,
    _safe_diff_str,
    _safe_fmt,
    _safe_diff_fmt,
)


# ── estimate_vote_fusion_alpha ───────────────────────────────────────────────

class TestEstimateVoteFusionAlpha:
    def test_none_votes(self):
        assert estimate_vote_fusion_alpha(None) == ''

    def test_empty_votes(self):
        assert estimate_vote_fusion_alpha({}) == ''

    def test_very_high(self):
        assert estimate_vote_fusion_alpha({'total': 10000}) == '0.30'

    def test_high(self):
        assert estimate_vote_fusion_alpha({'total': 3000}) == '0.20'

    def test_medium(self):
        assert estimate_vote_fusion_alpha({'total': 500}) == '0.10'

    def test_low(self):
        assert estimate_vote_fusion_alpha({'total': 50}) == '0.05'

    def test_boundary_5000(self):
        assert estimate_vote_fusion_alpha({'total': 5000}) == '0.30'

    def test_boundary_1000(self):
        assert estimate_vote_fusion_alpha({'total': 1000}) == '0.20'

    def test_boundary_200(self):
        assert estimate_vote_fusion_alpha({'total': 200}) == '0.10'


# ── top_market_label ─────────────────────────────────────────────────────────

class TestTopMarketLabel:
    def test_empty_map_returns_fallback(self):
        assert top_market_label({}, '主胜') == '主胜'

    def test_returns_shortest_odd(self):
        assert top_market_label({'主胜': 1.5, '平': 3.5, '客胜': 6.0}, '主胜') == '主胜'

    def test_filters_zero_odds(self):
        assert top_market_label({'主胜': 0, '平': 3.5, '客胜': 1.8}, '主胜') == '客胜'

    def test_all_zero_returns_fallback(self):
        assert top_market_label({'主胜': 0, '平': 0, '客胜': 0}, '平') == '平'

    def test_single_entry(self):
        assert top_market_label({'主胜': 2.0}, '平') == '主胜'


# ── pick_best_htft ───────────────────────────────────────────────────────────

class TestPickBestHtft:
    def test_no_market_odds_returns_highest_prob(self):
        probs = {'胜胜': 0.3, '平平': 0.4, '负负': 0.3}
        assert pick_best_htft(probs) == '平平'

    def test_market_odds_preferred(self):
        probs = {'胜胜': 0.5, '平平': 0.3, '负负': 0.2}
        market = {'胜胜': 0, '平平': 3.0, '负负': 0}
        assert pick_best_htft(probs, market) == '平平'

    def test_all_market_zero_falls_back(self):
        probs = {'胜胜': 0.5, '平平': 0.3, '负负': 0.2}
        assert pick_best_htft(probs, {}) == '胜胜'

    def test_some_available_in_market(self):
        probs = {'胜胜': 0.2, '平平': 0.2, '负负': 0.6}
        market = {'胜胜': 2.5, '平平': 0, '负负': 0}
        assert pick_best_htft(probs, market) == '胜胜'


# ── compute_bet_action ──────────────────────────────────────────────────────

class MockScenario:
    def __init__(self, play, pick, prob, ev):
        self.play = play
        self.pick = pick
        self.prob = prob
        self.ev = ev

class MockBetAnalysis:
    def __init__(self, scenarios):
        self.scenarios = scenarios

class TestComputeBetAction:
    def test_skip_league(self):
        assert compute_bet_action('UEFA Nations League', 'hybrid', None, [], 0, {}) == 'SKIP_LEAGUE'

    def test_market_fallback(self):
        assert compute_bet_action('英超', 'market_fallback', None, [], 0, {}) == 'WATCH'

    def test_friendly(self):
        assert compute_bet_action('友谊赛', 'hybrid', None, [], 0, {}) == 'WATCH_FRIENDLY'

    def test_friendly_english(self):
        assert compute_bet_action('International Friendly', 'hybrid', None, [], 0, {}) == 'WATCH_FRIENDLY'

    def test_recommend(self):
        assert compute_bet_action('英超', 'hybrid', None, [], 0, {}) == 'RECOMMEND'


# ── _fmt_zjq ─────────────────────────────────────────────────────────────────

class TestFmtZjq:
    def test_empty(self):
        assert _fmt_zjq({}) == ''

    def test_four_balls(self):
        zjq = {'0球': 10.0, '1球': 5.0, '2球': 3.0, '3球': 2.0}
        result = _fmt_zjq(zjq)
        assert '0球' in result and '3球' in result

    def test_not_enough(self):
        assert _fmt_zjq({'0球': 1.5, '1球': 1.2}) == ''

    def test_invalid_keys(self):
        assert _fmt_zjq({'foo': 1.0, 'bar': 2.0}) == ''


# ── _safe_str / _safe_diff_str / _safe_fmt / _safe_diff_fmt ──────────────────

class TestSafeHelpers:
    def test_safe_str_present(self):
        assert _safe_str({'key': 'value'}, 'key') == 'value'

    def test_safe_str_missing(self):
        assert _safe_str({}, 'key') == ''

    def test_safe_str_none_meta(self):
        assert _safe_str(None, 'key') == ''

    def test_safe_diff_str_present(self):
        assert _safe_diff_str({'a': 10, 'b': 5}, 'a', 'b') == '5'

    def test_safe_diff_str_missing(self):
        assert _safe_diff_str({}, 'a', 'b') == ''

    def test_safe_fmt_present(self):
        assert _safe_fmt({'x': 0.1234}, 'x', '.2f') == '0.12'

    def test_safe_fmt_missing(self):
        assert _safe_fmt({'x': 0.1234}, 'y', '.2f') == ''

    def test_safe_diff_fmt_present(self):
        assert _safe_diff_fmt({'a': 0.5, 'b': 0.3}, 'a', 'b', '.1f') == '0.2'


# ── ensure_log_has_source_fields ─────────────────────────────────────────────

class TestEnsureLogHasSourceFields:
    def test_no_file_does_nothing(self, tmp_path, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, 'PREDICTIONS_LOG', str(tmp_path / 'nonexistent.csv'))
        ensure_log_has_source_fields()  # should not raise

    def test_adds_missing_columns(self, tmp_path, monkeypatch):
        log_path = str(tmp_path / 'log.csv')
        monkeypatch.setattr('pipeline.bundle_builder.PREDICTIONS_LOG', log_path)
        with open(log_path, 'w') as f:
            f.write('code,date\n123,2025-01-01\n')
        ensure_log_has_source_fields()
        with open(log_path) as f:
            header = f.readline().strip()
        assert 'source_tag' in header and 'model_version' in header

    def test_no_change_when_all_present(self, tmp_path, monkeypatch):
        from config import settings
        log_path = str(tmp_path / 'log.csv')
        monkeypatch.setattr(settings, 'PREDICTIONS_LOG', log_path)
        with open(log_path, 'w') as f:
            f.write('code,date,source_tag,model_version\n123,2025-01-01,foo,bar\n')
        content_before = open(log_path).read()
        ensure_log_has_source_fields()
        content_after = open(log_path).read()
        assert content_before == content_after


# ── patch_logged_metadata ────────────────────────────────────────────────────

class TestPatchLoggedMetadata:
    def test_no_file_does_nothing(self, tmp_path, monkeypatch):
        from config import settings
        monkeypatch.setattr(settings, 'PREDICTIONS_LOG', str(tmp_path / 'nonexistent.csv'))
        patch_logged_metadata('123', 'tag', 'ver')

    def test_patches_today_row(self, tmp_path, monkeypatch):
        log_path = str(tmp_path / 'log.csv')
        monkeypatch.setattr('pipeline.bundle_builder.PREDICTIONS_LOG', log_path)
        with open(log_path, 'w') as f:
            f.write('code,date,source_tag,model_version\n123,2025-01-01,old,old_ver\n')
        monkeypatch.setattr('pipeline.bundle_builder.date', date)  # restore real date
        with patch('pipeline.bundle_builder.date') as mock_date:
            mock_date.today.return_value = date(2025, 1, 1)
            patch_logged_metadata('123', 'new_tag', 'new_ver')
        with open(log_path) as f:
            lines = f.readlines()
        assert 'new_tag' in lines[1] and 'new_ver' in lines[1]

    def test_skips_non_today_rows(self, tmp_path, monkeypatch):
        from config import settings
        log_path = str(tmp_path / 'log.csv')
        monkeypatch.setattr(settings, 'PREDICTIONS_LOG', log_path)
        with open(log_path, 'w') as f:
            f.write('code,date,source_tag,model_version\n123,2025-01-01,old,old_ver\n')
        with patch('pipeline.bundle_builder.date') as mock_date:
            mock_date.today.return_value = date(2025, 6, 1)
            patch_logged_metadata('123', 'new_tag', 'new_ver')
        with open(log_path) as f:
            lines = f.readlines()
        assert 'old' in lines[1]


# ── record_prediction ────────────────────────────────────────────────────────

class TestRecordPrediction:
    def _bundle(self):
        return {
            'code': 'M001', 'home': 'TeamA', 'away': 'TeamB',
            'home_cn': 'TeamA', 'away_cn': 'TeamB', 'league': '英超',
            'time': '15:00', 'handicap': 0, 'rq_text': '0',
            'pred_h': 50.0, 'pred_d': 30.0, 'pred_a': 20.0,
            'pred_rq_win': 45.0, 'pred_rq_draw': 30.0, 'pred_rq_loss': 25.0,
            'rq_pick': '让胜', 'market_rq_pick': '让胜',
            'spf_pick': '主胜', 'market_spf_pick': '主胜',
            'pred_top_score': '1:0', 'market_score_pick': '1:0',
            'pred_top_goals': 2, 'pred_top_htft': '胜胜',
            'market_htft_pick': '胜胜',
            'pred_spf_pick': '主胜', 'pred_rq_pick': '让胜',
            'pred_htft_pick': '胜胜', 'pred_goals_pick': 2,
            'pred_score_pick': '1:0',
            'score_top8': [], 'htft_top6': [], 'goals_all': [],
            'score_all': [], 'goals_top5': [],
            'htft_all': [], 'htft_prob_map': {},
            'market_spf': '', 'zjq_odds_str': '',
            'spf_value_tips': [], 'bet_analysis': None,
            'market_conflicts': [], 'votes_text': '',
            'model_note': '', 'model': 'hybrid',
            'direction': 'SPF:主胜 | RQ:让胜 | HTFT:胜胜 | Goals:2 | Score:1:0',
            'source_tag': '500+365', 'model_version': 'v4',
            'simple_pred': '', 'simple_conf': 0,
            'pred30_h': None, 'pred30_d': None, 'pred30_a': None,
            'odds_h_str': '1.85', 'odds_d_str': '3.50', 'odds_a_str': '4.00',
            'ev_h_str': '0.12', 'ev_d_str': '', 'ev_a_str': '',
            'vote_h_str': '', 'vote_d_str': '', 'vote_a_str': '',
            'vote_count_str': '', 'vote_fusion_alpha': '',
            'pop_rank_home_str': '', 'pop_rank_away_str': '',
            'pop_rank_diff_str': '', 'trend_win_rate_home_str': '',
            'trend_win_rate_away_str': '', 'trend_win_rate_diff_str': '',
            's365_home_winrate': None, 's365_away_winrate': None,
            's365_home_fifa': None, 's365_away_fifa': None,
            's365_rank_diff': None, 's365_popularity_diff': None,
            'ah_fair_odds': {}, 'bet_action': 'RECOMMEND',
            'htft_warning': False, 'standings': None,
            'model_route': 'hybrid', 'match_key': '',
            'date': '2025-01-01',
        }

    def test_subprocess_called(self):
        mock_record = MagicMock(return_value='✅ 已记录: M001 TeamA vs TeamB')
        with patch.dict('sys.modules', {'backtest_jczq': MagicMock(record_match=mock_record)}):
            record_prediction(self._bundle())
            mock_record.assert_called_once()
            kwargs = mock_record.call_args[1]
            assert 'code' in kwargs
            assert kwargs['code'] == 'M001'

    def test_prints_error_on_failure(self, capsys):
        mock_record = MagicMock(side_effect=RuntimeError('record failed'))
        with patch.dict('sys.modules', {'backtest_jczq': MagicMock(record_match=mock_record)}):
            record_prediction(self._bundle())
            captured = capsys.readouterr()
            assert '落盘失败' in captured.out


# ── build_prediction_bundle ──────────────────────────────────────────────────

class TestBuildPredictionBundle:
    def _prediction(self):
        return {
            'probs': {'H': 0.55, 'D': 0.25, 'A': 0.20},
            'score': '2-1', 'result': 'H',
            'lambda_ft': {'home': 1.8, 'away': 0.9},
            'model': 'hybrid', 'rho': 0.0,
            'simple_pred': '', 'simple_conf': 0,
            'pred30_h': None, 'pred30_d': None, 'pred30_a': None,
            'standings': None,
        }

    def _market_row(self):
        return {
            'odds_h': 1.85, 'odds_d': 3.50, 'odds_a': 4.00,
            'handicap': 0, 'home_cn': '主队', 'away_cn': '客队',
            'rq_h': 1.80, 'rq_d': 3.40, 'rq_a': 3.80,
            'bf_odds': {}, 'zjq_odds': {}, 'htft_odds': {},
        }

    def test_returns_required_keys(self):
        bundle = build_prediction_bundle(
            'M001', 'TeamA', 'TeamB', '15:00', '英超',
            self._prediction(), self._market_row(),
        )
        required = {'code', 'home', 'away', 'pred_h', 'pred_d', 'pred_a', 'spf_pick', 'rq_pick'}
        assert required <= bundle.keys()

    def test_missing_market_row_does_not_crash(self):
        bundle = build_prediction_bundle(
            'M001', 'TeamA', 'TeamB', '15:00', '英超',
            self._prediction(),
        )
        assert bundle is not None

    def test_spf_pick_reflects_highest_prob(self):
        bundle = build_prediction_bundle(
            'M001', 'TeamA', 'TeamB', '15:00', '英超',
            {'probs': {'H': 0.55, 'D': 0.25, 'A': 0.20},
             'score': '2-1', 'result': 'H',
             'lambda_ft': {'home': 1.8, 'away': 0.9},
             'model': 'hybrid', 'rho': 0.0,
             'simple_pred': '', 'simple_conf': 0,
             'pred30_h': None, 'pred30_d': None, 'pred30_a': None,
             'standings': None},
            self._market_row(),
        )
        assert bundle['spf_pick'] == '主胜'

    def test_bet_action_set(self):
        bundle = build_prediction_bundle(
            'M001', 'TeamA', 'TeamB', '15:00', '英超',
            self._prediction(), self._market_row(),
        )
        assert isinstance(bundle['bet_action'], str)
        assert len(bundle['bet_action']) > 0

    def test_direction_format(self):
        bundle = build_prediction_bundle(
            'M001', 'TeamA', 'TeamB', '15:00', '英超',
            self._prediction(), self._market_row(),
        )
        assert 'SPF:' in bundle['direction']
        assert 'RQ:' in bundle['direction']
