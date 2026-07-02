# ─── Predict Matches ───
print("\n" + "=" * 60)
print("  MATCH PREDICTIONS (Hybrid DC+XGB)")
print("=" * 60)

for home, away in MATCHES:
    print(f"\n{'─' * 55}")
    print(f"  {home} vs {away}")
    print(f"{'─' * 55}")

    home, away = normalize_match_pair(home, away)
    dh = NAME_TO_DATA.get(home, home)
    da = NAME_TO_DATA.get(away, away)

    elo_h = elo.get(dh, 1500)
    elo_a = elo.get(da, 1500)
    print(f"  Elo: {home} {elo_h:.0f} / {away} {elo_a:.0f} (diff: {elo_h - elo_a:+.0f})")

    hybrid, lam_h, lam_a, dc_p, xgb_p, odds_probs = predict_match(home, away, dc, xgb_model, elo)
    hp = hybrid  # [away, draw, home]

    print(f"\n  Expected Goals (DC): {home} {lam_h:.3f} / {away} {lam_a:.3f}")
    print(f"\n  Probability Breakdown:")
    print(f"  {'Source':<20s} {home:<10s} {'Draw':<10s} {away:<10s}")
    print(f"  {'─' * 50}")
    print(f"  {'Dixon-Coles':<20s} {dc_p[0] * 100:>7.1f}% {dc_p[1] * 100:>7.1f}% {dc_p[2] * 100:>7.1f}%")
    print(f"  {'XGBoost':<20s} {xgb_p[2] * 100:>7.1f}% {xgb_p[1] * 100:>7.1f}% {xgb_p[0] * 100:>7.1f}%")
    print(f"  {'Elo-Odds':<20s} {odds_probs[0] * 100:>7.1f}% {odds_probs[1] * 100:>7.1f}% {odds_probs[2] * 100:>7.1f}%")
    print(f"  {'Hybrid (DC×0.6+XGB×0.4)':<20s} {hp[2] * 100:>7.1f}% {hp[1] * 100:>7.1f}% {hp[0] * 100:>7.1f}%")

    if tournament_probs:
        print("  Tournament winner odds: overview-only display, excluded from single-match calibration")

    print(f"\n  --- 100K MC Simulation ---")
    hw, dr, aw, sc = simulate_match(hp, lam_h, lam_a, 100000)
    print(f"\n  Result: {home} {hw:.1%} / Draw {dr:.1%} / {away} {aw:.1%}")

    print(f"\n  Top 10 score lines:")
    for s, c in sorted(sc.items(), key=lambda x: -x[1])[:10]:
        print(f"    {s:>5s}: {c / 100000:.1%}  {'#' * int(c / 100000 * 200)}")

    bts = sum(c for k, c in sc.items() if int(k[0]) > 0 and int(k[2]) > 0)
    ov25 = sum(c for k, c in sc.items() if int(k[0]) + int(k[2]) > 2)
    un15 = sum(c for k, c in sc.items() if int(k[0]) + int(k[2]) <= 1)
    print(f"\n  Market: BTS {bts / 100000:.1%} / O2.5 {ov25 / 100000:.1%} / U1.5 {un15 / 100000:.1%}")

    best_s, best_c = sorted(sc.items(), key=lambda x: -x[1])[0]
    print(f"\n  ▶ Best Pick: {home} {best_s} ({best_c / 100000:.1%})")

    w1 = sum(c for k, c in sc.items() if int(k[0]) - int(k[2]) == 1) / 100000
    w2p = sum(c for k, c in sc.items() if int(k[0]) - int(k[2]) >= 2) / 100000
    l1 = sum(c for k, c in sc.items() if int(k[2]) - int(k[0]) == 1) / 100000
    l2p = sum(c for k, c in sc.items() if int(k[2]) - int(k[0]) >= 2) / 100000
    print(f"  {home} win 2+: {w2p:.1%} / win 1: {w1:.1%} / Draw: {dr:.1%} / {away} win 1: {l1:.1%} / win 2+: {l2p:.1%}")

print(f"\n{'=' * 60}")
print("  Done")
