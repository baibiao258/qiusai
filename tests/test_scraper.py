"""Tests for pipeline.scraper.

All network calls and subprocess calls are mocked.
No real 500.com connection required.
"""
from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch, call

import pytest

from pipeline.scraper import (
    fetch_live_odds_map,
    scrape_500_odds_today,
    apply_euro_fallback,
    _parse_row,
    _to_float,
)


# ── _to_float ────────────────────────────────────────────────────────────────

class TestToFloat:
    def test_numeric_string(self):     assert _to_float('1.85') == pytest.approx(1.85)
    def test_int_string(self):         assert _to_float('2') == 2.0
    def test_none(self):               assert _to_float(None) == 0.0
    def test_empty_string(self):       assert _to_float('') == 0.0
    def test_float_passthrough(self):  assert _to_float(1.5) == pytest.approx(1.5)


# ── _parse_row ───────────────────────────────────────────────────────────────

class TestParseRow:
    def _row(self, **kwargs):
        base = {
            'home': '法国', 'away': '德国',
            'endtime': '2026-07-03 20:00',
            'league': '世界杯',
            'handicap': 0,
            'odds': {
                'spf':  {'3': '1.85', '1': '3.50', '0': '4.20'},
                'nspf': {},
                'bf':   {'1-0': '6.00'},
                'jqs':  {'2': '2.80'},
                'bqc':  {'3-3': '3.00'},
            },
        }
        base.update(kwargs)
        return base

    def test_basic_fields_present(self):
        result = _parse_row('周四201', self._row(), None)
        assert result['code'] == '周四201'
        assert result['home_cn'] == '法国'
        assert result['away_cn'] == '德国'
        assert result['league'] == '世界杯'

    def test_no_handicap_uses_spf(self):
        result = _parse_row('周四201', self._row(), None)
        assert result['odds_h'] == pytest.approx(1.85)
        assert result['odds_d'] == pytest.approx(3.50)
        assert result['odds_a'] == pytest.approx(4.20)
        assert result['std_odds_source'] == 'spf'

    def test_handicap_with_nspf_uses_nspf(self):
        row = self._row(handicap=-1)
        row['odds']['nspf'] = {'3': '2.10', '1': '3.20', '0': '3.80'}
        row['odds']['spf']  = {'3': '1.70', '1': '3.30', '0': '4.50'}
        result = _parse_row('周四202', row, None)
        assert result['odds_h'] == pytest.approx(2.10)   # from nspf
        assert result['rq_h']   == pytest.approx(1.70)   # from spf
        assert result['std_odds_source'] == 'nspf'

    def test_nspf_empty_sets_std_to_zero(self):
        row = self._row(handicap=-1)
        row['odds']['nspf'] = {}
        result = _parse_row('周四203', row, None)
        assert result['odds_h'] == 0.0
        assert result['odds_d'] == 0.0
        assert result['odds_a'] == 0.0
        assert result['nspf_empty'] is True

    def test_nspf_empty_with_live_source_tag(self):
        row = self._row(handicap=-1)
        row['odds']['nspf'] = {}
        live_map = {'周四204': {'h': 1.90, 'd': 3.40, 'a': 4.00}}
        result = _parse_row('周四204', row, live_map)
        assert result['std_odds_source'] == 'live_euro_avg'

    def test_htft_odds_mapped_to_chinese(self):
        row = self._row()
        row['odds']['bqc'] = {'3-3': '3.00', '1-1': '5.50'}
        result = _parse_row('周四205', row, None)
        assert '胜胜' in result['htft_odds']
        assert '平平' in result['htft_odds']

    def test_bf_odds_present(self):
        result = _parse_row('周四206', self._row(), None)
        assert result['bf_odds'].get('1-0') == pytest.approx(6.00)

    def test_zjq_odds_key_format(self):
        result = _parse_row('周四207', self._row(), None)
        assert '2球' in result['zjq_odds']

    def test_home_away_bracket_stripped(self):
        row = self._row(home='[7]荷兰', away='乌兹别克[58] 单关')
        result = _parse_row('周四208', row, None)
        assert result['home_cn'] == '荷兰'
        assert result['away_cn'] == '乌兹别克'


# ── fetch_live_odds_map ───────────────────────────────────────────────────────

class TestFetchLiveOddsMap:
    def _make_html(self, odds_by_fid: dict, code_fid_pairs: list[tuple]) -> bytes:
        """Build minimal HTML that the parser can understand."""
        js = f'var liveOddsList = {json.dumps(odds_by_fid)};'
        checkboxes = ''.join(
            f'<input type="checkbox" name="check_id[]" value="{fid}" />{code}</td>'
            for code, fid in code_fid_pairs
        )
        return (js + checkboxes).encode('gbk')

    def test_returns_dict_on_success(self):
        html = self._make_html(
            {'111': {'0': ['1.85', '3.50', '4.20']}},
            [('周四201', '111')],
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = html
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch('pipeline.scraper.urllib.request.urlopen', return_value=mock_resp):
            result = fetch_live_odds_map()

        assert result is not None
        assert '周四201' in result
        assert result['周四201']['h'] == pytest.approx(1.85)

    def test_returns_none_when_js_var_missing(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b'<html>no odds here</html>'
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch('pipeline.scraper.urllib.request.urlopen', return_value=mock_resp):
            result = fetch_live_odds_map()

        assert result is None

    def test_returns_none_on_network_error(self):
        with patch('pipeline.scraper.urllib.request.urlopen', side_effect=OSError('timeout')):
            result = fetch_live_odds_map()
        assert result is None

    def test_filters_invalid_odds(self):
        """Odds with home <= 1 (e.g. 0.0) must be excluded."""
        html = self._make_html(
            {'222': {'0': ['0.0', '0.0', '0.0']}},
            [('周四202', '222')],
        )
        mock_resp = MagicMock()
        mock_resp.read.return_value = html
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)

        with patch('pipeline.scraper.urllib.request.urlopen', return_value=mock_resp):
            result = fetch_live_odds_map()

        assert not result  # empty dict or None — both falsy


# ── scrape_500_odds_today ────────────────────────────────────────────────────

class TestScrape500OddsToday:
    def _mock_proc(self, result_list: list[dict], returncode: int = 0):
        proc = MagicMock()
        proc.returncode = returncode
        proc.stdout = json.dumps({'ok': True, 'result': result_list})
        proc.stderr = ''
        return proc

    def _minimal_row(self, code: str) -> dict:
        return {
            'no': code, 'home': '法国', 'away': '德国',
            'endtime': '20:00', 'league': '世界杯', 'handicap': 0,
            'odds': {
                'spf':  {'3': '1.85', '1': '3.50', '0': '4.20'},
                'nspf': {}, 'bf': {}, 'jqs': {}, 'bqc': {},
            },
        }

    def test_returns_list_on_success(self):
        rows = [self._minimal_row('周四201'), self._minimal_row('周四202')]
        with patch('pipeline.scraper.subprocess.run', return_value=self._mock_proc(rows)), \
             patch('pipeline.scraper.fetch_live_odds_map', return_value=None):
            result = scrape_500_odds_today()
        assert len(result) == 2
        assert result[0]['code'] == '周四201'

    def test_returns_empty_on_timeout(self):
        with patch('pipeline.scraper.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('python3', 45)):
            result = scrape_500_odds_today()
        assert result == []

    def test_returns_empty_on_nonzero_exit(self):
        proc = self._mock_proc([], returncode=1)
        proc.stderr = 'something went wrong'
        with patch('pipeline.scraper.subprocess.run', return_value=proc):
            result = scrape_500_odds_today()
        assert result == []

    def test_returns_empty_on_json_error(self):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = 'not json'
        with patch('pipeline.scraper.subprocess.run', return_value=proc):
            result = scrape_500_odds_today()
        assert result == []

    def test_returns_empty_when_result_empty(self):
        proc = MagicMock()
        proc.returncode = 0
        proc.stdout = json.dumps({'ok': True, 'result': []})
        with patch('pipeline.scraper.subprocess.run', return_value=proc):
            result = scrape_500_odds_today()
        assert result == []

    def test_circuit_breaker_writes_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr('pipeline.scraper._BREAKER_LOG', str(tmp_path / '500breaker.log'))
        with patch('pipeline.scraper.subprocess.run',
                   side_effect=subprocess.TimeoutExpired('python3', 45)):
            scrape_500_odds_today()
        assert (tmp_path / '500breaker.log').exists()
        log_content = (tmp_path / '500breaker.log').read_text()
        assert '500BREAKER' in log_content

    def test_live_fallback_called_on_success(self):
        rows = [self._minimal_row('周四201')]
        with patch('pipeline.scraper.subprocess.run', return_value=self._mock_proc(rows)), \
             patch('pipeline.scraper.fetch_live_odds_map', return_value=None) as mock_live:
            scrape_500_odds_today()
        mock_live.assert_called_once()


# ── apply_euro_fallback ───────────────────────────────────────────────────────

class TestApplyEuroFallback:
    def test_no_op_when_market_row_none(self):
        bundle = {'pred_h': 50.0}
        result = apply_euro_fallback(bundle, None)
        assert result is bundle
        assert 'euro_odds_ref' not in result

    def test_no_op_when_nspf_not_empty(self):
        bundle = {}
        market_row = {'nspf_empty': False, 'current_euro_odds_500': {'home': 2.0}}
        result = apply_euro_fallback(bundle, market_row)
        assert 'euro_odds_ref' not in result

    def test_attaches_euro_ref_when_nspf_empty(self):
        bundle = {'current_euro_odds_500': {'home': 2.10, 'draw': 3.20, 'away': 3.80}}
        market_row = {'nspf_empty': True}
        result = apply_euro_fallback(bundle, market_row)
        assert result['euro_odds_ref'] == {'home': 2.10, 'draw': 3.20, 'away': 3.80}
        assert result['model_note_append'] == '+SPF未开售(仅开让球)'

    def test_no_euro_ref_when_odds_missing(self):
        bundle = {}
        market_row = {'nspf_empty': True}
        result = apply_euro_fallback(bundle, market_row)
        assert 'euro_odds_ref' not in result

    def test_no_euro_ref_when_home_le_1(self):
        bundle = {'current_euro_odds_500': {'home': 0.0}}
        market_row = {'nspf_empty': True}
        result = apply_euro_fallback(bundle, market_row)
        assert 'euro_odds_ref' not in result

    def test_returns_same_bundle_object(self):
        bundle = {}
        market_row = {'nspf_empty': False}
        assert apply_euro_fallback(bundle, market_row) is bundle
