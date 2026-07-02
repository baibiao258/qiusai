#!/usr/bin/env python3
import sys, os
sys.path.insert(0, '/root')
os.chdir('/root')
from daily_jczq import _load_poisson_elo_prior, _load_shared_models, _fuzzy_team_lookup, _resolve_name, _elo_dict
from team_name_normalizer import normalize_match_pair as nmp

_load_poisson_elo_prior()
_load_shared_models()

elo_keys = list(_elo_dict.keys())
print(f"Elo字典球队数: {len(elo_keys)}")

teams = ["Côte d'Ivoire", "Australia", "USA", "United States", "Canada", "South Korea", "Türkiye", "Turkey", "Paraguay", "Bosnia & Herzegovina", "Czechia", "Czech Republic"]
for t in teams:
    resolved = _resolve_name(t)
    h, _ = nmp(t, 'x')
    fuzzy_t = _fuzzy_team_lookup(t, elo_keys)
    fuzzy_h = _fuzzy_team_lookup(h, elo_keys)
    direct = t in elo_keys or resolved in elo_keys or h in elo_keys
    
    # Check match in DC teams
    from dc_model_definition import DixonColes
    # Can't load DC directly, but we can check elo_dict
    in_elo = t in _elo_dict
    res_in_elo = resolved in _elo_dict if resolved else False
    h_in_elo = h in _elo_dict if h else False
    
    if not (in_elo or res_in_elo or h_in_elo):
        print(f"  ❌ {t:<30s}  _resolve={str(resolved):<20s}  norm={h:<20s}  fuzzy={fuzzy_h or fuzzy_t}")
    else:
        print(f"  ✅ {t:<30s}  in_elo={in_elo}  res={res_in_elo}  h={h_in_elo}")
