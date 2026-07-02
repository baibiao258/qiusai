"""Tests for pipeline.data_loader.

Network calls (urllib) and imports of external modules are mocked.
No real API connection or filesystem required.
"""
from __future__ import annotations

import csv
import io
import json
import os
import tempfile
import urllib.error
from unittest.mock import MagicMock, patch, call

import pytest

from pipeline.data_loader import (
    api_get,
    fetch_league_history,
    get_today_matches,
    load_365scores_today,
    build_365_map,
    _parse_365_row,
    _safe_float,
    _safe_int,
)
from config.settings import JCZQ_LEAGUES


# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_urlopen(body: dict | list):
    """Return a context-manager mock that yields JSON bytes."""
    mock_resp = MagicMock()
    mock_resp.read.return_value = json.dumps(body).encode('utf-8')
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


# ── _safe_float / _safe_int ───────────────────────────────────────────────────

class TestSafeConversions:
    def test_safe_float_normal(self):    assert _safe_float('1.5') == pytest.approx(1.5)
    def test_safe_float_empty(self):     assert _safe_float('') is None
    def test_safe_float_none(self):      assert _safe_float(None) is None
    def test_safe_float_default(self):   assert _safe_float(None, 0.0) == 0.0
    def test_safe_int_rounds(self):      assert _safe_int('3.9') == 3
    def test_safe_int_none(self):        assert _safe_int(None) is None
    def test_safe_int_default(self):     assert _safe_int('', 0) == 0


# ── _parse_365_row ────────────────────────────────────────────────────────────

class TestParse365Row:
    def _row(self, **kwargs):
        base = {
            'home': '法国', 'away': '德国', 'competition': '世界杯', 'time': '20:00',
            'vote_home': '50.0', 'vote_draw': '25.0', 'vote_away': '25.0', 'vote_count': '1000',
            'pop_rank_home': '1', 'pop_rank_away': '2',
            'fifa_rank_home': '2', 'fifa_rank_away': '4',
            'trend_home_w': '3', 'trend_home_d': '1', 'trend_home_l': '1',
            'trend_away_w': '2', 'trend_away_d': '2', 'trend_away_l': '1',
            'trend_win_rate_home': '0.60', 'trend_win_rate_away': '0.40',
        }
        base.update(kwargs)
        return base

    def test_basic_fields(self):
        g = _parse_365_row(self._row())
        assert g['home'] == '法国'
        assert g['away'] == '德国'
        assert g['competition'] == '世界杯'

    def test_votes_parsed(self):
        g = _parse_365_row(self._row())
        assert g['votes']['home'] == pytest.approx(50.0)
        assert g['votes']['total'] == 1000

    def test_trend_list_length(self):
        g = _parse_365_row(self._row())
        assert len(g['trend_home']) == 3
        assert len(g['trend_away']) == 3
        assert g['trend_home'] == [3, 1, 1]

    def test_missing_votes_returns_none(self):
        g = _parse_365_row(self._row(vote_home='', vote_draw=None, vote_away=''))
        assert g['votes']['home'] is None

    def test_fifa_rank_int(self):
        g = _parse_365_row(self._row())
        assert g['fifa_rank_home'] == 2
        assert isinstance(g['fifa_rank_home'], int)

    def test_win_rate_float(self):
        g = _parse_365_row(self._row())
        assert g['trend_win_rate_home'] == pytest.approx(0.60)


# ── api_get ───────────────────────────────────────────────────────────────────

class TestApiGet:
    def test_returns_parsed_json(self):
        body = {'matches': [{'id': 1}]}
        with patch('pipeline.data_loader.urllib.request.urlopen', return_value=_mock_urlopen(body)):
            result = api_get('/competitions/PL/matches')
        assert result == body

    def test_raises_on_http_error(self):
        with patch('pipeline.data_loader.urllib.request.urlopen',
                   side_effect=urllib.error.HTTPError(None, 403, 'Forbidden', {}, None)):
            with pytest.raises(urllib.error.HTTPError):
                api_get('/competitions/PL/matches')

    def test_url_constructed_correctly(self):
        with patch('pipeline.data_loader.urllib.request.urlopen', return_value=_mock_urlopen({})) as mock_uo:
            api_get('/competitions/PL/matches?dateFrom=2026-07-01')
        args = mock_uo.call_args[0]
        assert 'football-data.org/v4/competitions/PL/matches' in args[0].full_url


# ── fetch_league_history ──────────────────────────────────────────────────────

class TestFetchLeagueHistory:
    def _finished_match(self, home='France', away='Germany', h=2, a=1):
        return {
            'status': 'FINISHED',
            'utcDate': '2026-06-01T20:00:00Z',
            'homeTeam': {'shortName': home},
            'awayTeam': {'shortName': away},
            'score': {'fullTime': {'home': h, 'away': a}},
        }

    def test_returns_finished_matches(self):
        resp = {'matches': [self._finished_match()]}
        with patch('pipeline.data_loader.api_get', return_value=resp), \
             patch('pipeline.data_loader.time.sleep'):
            result = fetch_league_history('PL', months_back=1)
        assert len(result) == 2  # 2 segments with months_back=1
        assert result[0]['home'] == 'France'
        assert result[0]['h_score'] == 2

    def test_skips_non_finished(self):
        resp = {'matches': [
            self._finished_match(),
            {**self._finished_match(), 'status': 'SCHEDULED'},
        ]}
        with patch('pipeline.data_loader.api_get', return_value=resp), \
             patch('pipeline.data_loader.time.sleep'):
            result = fetch_league_history('PL', months_back=1)
        assert len(result) == 2  # 2 segments × 1 valid match

    def test_skips_null_score(self):
        m = self._finished_match()
        m['score']['fullTime']['home'] = None
        resp = {'matches': [m]}
        with patch('pipeline.data_loader.api_get', return_value=resp), \
             patch('pipeline.data_loader.time.sleep'):
            result = fetch_league_history('PL', months_back=1)
        assert result == []

    def test_retries_on_429(self):
        err = urllib.error.HTTPError(None, 429, 'Too Many Requests', {}, None)
        resp = {'matches': [self._finished_match()]}
        call_count = 0
        def side_effect(*a, **kw):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise err
            return resp
        with patch('pipeline.data_loader.api_get', side_effect=side_effect), \
             patch('pipeline.data_loader.time.sleep'):
            result = fetch_league_history('PL', months_back=1)
        # segment 1: retry once (2 calls), segment 2: no error (1 call) = 3 total
        assert call_count == 3

    def test_returns_empty_on_persistent_error(self):
        with patch('pipeline.data_loader.api_get', side_effect=Exception('network down')), \
             patch('pipeline.data_loader.time.sleep'):
            result = fetch_league_history('PL', months_back=1)
        assert result == []

    def test_result_dict_keys(self):
        resp = {'matches': [self._finished_match()]}
        with patch('pipeline.data_loader.api_get', return_value=resp), \
             patch('pipeline.data_loader.time.sleep'):
            result = fetch_league_history('PL', months_back=1)
        assert set(result[0].keys()) == {'date', 'home', 'away', 'h_score', 'a_score'}


# ── get_today_matches ─────────────────────────────────────────────────────────

class TestGetTodayMatches:
    def _timed_match(self, home='France', away='Germany'):
        return {
            'status': 'TIMED',
            'homeTeam': {'shortName': home},
            'awayTeam': {'shortName': away},
        }

    def test_returns_scheduled_matches(self):
        resp = {'matches': [self._timed_match()]}
        with patch('pipeline.data_loader.api_get', return_value=resp):
            result = get_today_matches()
        assert len(result) == len(JCZQ_LEAGUES)  # one per league

    def test_deduplicates_within_competition(self):
        """同一联赛内重复的 (code, home, away) 应被去重。"""
        jcq = [('PL', '英超')]  # single league
        with patch('pipeline.data_loader.JCZQ_LEAGUES', jcq), \
             patch('pipeline.data_loader.api_get', return_value={
                 'matches': [self._timed_match(), self._timed_match()]
             }):
            result = get_today_matches()
        assert len(result) == 1  # same match deduped within PL

    def test_skips_finished_matches(self):
        resp = {'matches': [
            {**self._timed_match(), 'status': 'FINISHED'},
            self._timed_match('Spain', 'Italy'),
        ]}
        with patch('pipeline.data_loader.api_get', return_value=resp):
            result = get_today_matches()
        assert all(m['status'] in ('SCHEDULED', 'TIMED') for m in result)

    def test_returns_empty_on_api_error(self):
        with patch('pipeline.data_loader.api_get', side_effect=Exception('err')):
            result = get_today_matches()
        assert result == []


# ── load_365scores_today ──────────────────────────────────────────────────────

class TestLoad365ScoresToday:
    def _write_csv(self, path: str, rows: list[dict]):
        if not rows:
            return
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    def _sample_row(self):
        return {
            'home': '法国', 'away': '德国', 'competition': '世界杯', 'time': '20:00',
            'vote_home': '50', 'vote_draw': '25', 'vote_away': '25', 'vote_count': '500',
            'pop_rank_home': '1', 'pop_rank_away': '2',
            'fifa_rank_home': '2', 'fifa_rank_away': '4',
            'trend_home_w': '3', 'trend_home_d': '1', 'trend_home_l': '1',
            'trend_away_w': '2', 'trend_away_d': '2', 'trend_away_l': '1',
            'trend_win_rate_home': '0.60', 'trend_win_rate_away': '0.40',
        }

    def test_reads_from_csv_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr('pipeline.data_loader._SCORES365_DIR', str(tmp_path))
        csv_file = tmp_path / f'{__import__("datetime").date.today().isoformat()}.csv'
        self._write_csv(str(csv_file), [self._sample_row()])
        result = load_365scores_today()
        assert len(result) == 1
        assert result[0]['home'] == '法国'

    def test_falls_back_to_live_fetch(self, tmp_path, monkeypatch):
        monkeypatch.setattr('pipeline.data_loader._SCORES365_DIR', str(tmp_path))
        mock_game = [{'home': '英格兰', 'away': '巴西'}]
        mock_module = MagicMock()
        mock_module.fetch_365scores_data.return_value = {}
        mock_module.extract_games.return_value = mock_game
        monkeypatch.setitem(__import__('sys').modules, 'fetch_365scores', mock_module)
        result = load_365scores_today()
        assert result == mock_game

    def test_returns_empty_on_all_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr('pipeline.data_loader._SCORES365_DIR', str(tmp_path))
        # Make the live-fetch import fail
        import builtins
        real_import = builtins.__import__
        def _mock_import(name, *args, **kwargs):
            if name == 'fetch_365scores':
                raise ImportError('mock fetch_365scores unavailable')
            return real_import(name, *args, **kwargs)
        monkeypatch.setattr('builtins.__import__', _mock_import)
        result = load_365scores_today()
        assert result == []


# ── build_365_map ─────────────────────────────────────────────────────────────

class TestBuild365Map:
    def test_bidirectional_lookup(self):
        games = [{'home': '法国', 'away': '德国', 'votes': {}}]
        with patch('team_name_normalizer.normalize_match_pair', side_effect=lambda h, a: (h, a)):
            mapping = build_365_map(games)
        assert ('法国', '德国') in mapping
        assert ('德国', '法国') in mapping

    def test_skips_empty_names(self):
        games = [{'home': '', 'away': '', 'votes': {}}]
        with patch('team_name_normalizer.normalize_match_pair', return_value=('', '')):
            mapping = build_365_map(games)
        assert len(mapping) == 0

    def test_skips_on_normalizer_exception(self):
        games = [{'home': '法国', 'away': '德国'}]
        with patch('team_name_normalizer.normalize_match_pair', side_effect=Exception('err')):
            mapping = build_365_map(games)
        assert len(mapping) == 0

    def test_returns_same_game_object_both_directions(self):
        g = {'home': '法国', 'away': '德国', 'votes': {'total': 100}}
        with patch('team_name_normalizer.normalize_match_pair', side_effect=lambda h, a: (h, a)):
            mapping = build_365_map([g])
        assert mapping[('法国', '德国')] is mapping[('德国', '法国')]
