import json

from htft_parser import (
    HTFT_KEYS,
    parse_betexplorer_htft,
    parse_vegasinsider_htft,
)


VEGAS_HTML = """
<html><body>
  <div>Some header noise</div>
  <div>爱尔兰 vs 卡塔尔</div>
  <div>HTFT odds 2.02 18.00 21.00 4.50 6.20 13.00 7.50 11.50 15.00</div>
  <div>Footer noise</div>
</body></html>
"""


BETEXPLORER_HTML = """
<html><body>
  <div>Football stats, results, tables & standings</div>
  <div>Spain vs France</div>
  <div>Market: HTFT</div>
  <div>Odds: 3.10 5.20 6.80 4.10 5.90 9.10 7.20 8.80 13.00</div>
</body></html>
"""


MISSING_MATCH_HTML = """
<html><body>
  <div>unrelated content only</div>
  <div>Odds: 1.50 2.50 3.50 4.50 5.50 6.50 7.50 8.50 9.50</div>
</body></html>
"""


def test_parse_vegasinsider_htft_success():
    res = parse_vegasinsider_htft(VEGAS_HTML, "爱尔兰 vs 卡塔尔", "爱尔兰", "卡塔尔")

    assert res.source == "vegasinsider"
    assert res.market == "htft"
    assert res.match == "爱尔兰 vs 卡塔尔"
    assert list(res.odds.keys()) == HTFT_KEYS
    assert res.odds["HH"] == 2.02
    assert res.odds["AA"] == 15.0
    assert res.confidence > 0.0
    parsed = json.loads(res.to_json())
    assert parsed["source"] == "vegasinsider"
    assert len(parsed["odds"]) == 9


def test_parse_betexplorer_htft_success():
    res = parse_betexplorer_htft(BETEXPLORER_HTML, "Spain vs France", "Spain", "France")

    assert res.source == "betexplorer"
    assert res.market == "htft"
    assert res.match == "Spain vs France"
    assert list(res.odds.keys()) == HTFT_KEYS
    assert res.odds["HH"] == 3.1
    assert res.odds["AA"] == 13.0
    assert res.confidence > 0.0


def test_parse_missing_match_returns_zero_block():
    res = parse_vegasinsider_htft(MISSING_MATCH_HTML, "Unknown vs Unknown", "Unknown", "Unknown")

    assert res.source == "vegasinsider"
    assert res.match == "Unknown vs Unknown"
    assert list(res.odds.keys()) == HTFT_KEYS
    assert all(v == 0.0 for v in res.odds.values())
    assert res.confidence == 0.05


def test_parse_output_has_required_fields():
    res = parse_betexplorer_htft(BETEXPLORER_HTML, "Spain vs France", "Spain", "France")
    parsed = json.loads(res.to_json())

    assert set(parsed.keys()) == {"source", "market", "match", "odds", "timestamp", "raw_snippet", "confidence"}
    assert parsed["market"] == "htft"
    assert isinstance(parsed["raw_snippet"], str)
    assert isinstance(parsed["confidence"], float)
