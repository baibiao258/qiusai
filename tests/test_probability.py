#!/usr/bin/env python3
"""pytest 测试: pipeline/probability.py

运行:  cd /root && python -m pytest tests/test_probability.py -v
"""
import math
import numpy as np
import pytest

from pipeline.probability import (
    compute_dynamic_xgb_weight,
    compute_goals_distribution,
    compute_rq_probs,
    compute_score_topn,
    compute_htft_topn_math,
    dc_tau,
    elo_expected,
    format_pct,
    implied_probs_from_odds,
    poisson_pmf,
    quick_validate,
    rps_score,
    rq_result_label,
)

# half_full_model 位于 /root/wc_2026_upgrade/, 被 compute_htft_topn_math lazy 调用
import sys
sys.path.insert(0, '/root/wc_2026_upgrade')

# ═══════════════════════════════════════════
#  poisson_pmf
# ═══════════════════════════════════════════


class TestPoissonPmf:
    def test_sum_to_one(self):
        """Σ P(k=0..∞ | λ) ≈ 1"""
        lam = 1.5
        total = sum(poisson_pmf(k, lam) for k in range(0, 15))
        assert abs(total - 1.0) < 1e-6

    def test_k_zero(self):
        """P(k=0 | λ) = e^-λ"""
        lam = 2.0
        expected = math.exp(-2.0)
        assert abs(poisson_pmf(0, lam) - expected) < 1e-12

    def test_lambda_zero(self):
        """P(0 | λ=0) = 1, P(k>0 | λ=0) = 0"""
        assert poisson_pmf(0, 0.0) == 1.0
        assert poisson_pmf(1, 0.0) == 0.0
        assert poisson_pmf(5, 0.0) == 0.0

    def test_matches_scipy(self):
        """一致性校验: 与 scipy.stats.poisson 相同"""
        from scipy.stats import poisson as sp_poisson
        lam = 2.3
        for k in range(6):
            assert abs(poisson_pmf(k, lam) - sp_poisson.pmf(k, lam)) < 1e-15


# ═══════════════════════════════════════════
#  elo_expected
# ═══════════════════════════════════════════


class TestEloExpected:
    def test_equal_ratings(self):
        """ra == rb → 0.5"""
        assert elo_expected(1500, 1500) == 0.5

    def test_stronger_home(self):
        """ra > rb → > 0.5"""
        assert elo_expected(1600, 1400) > 0.5

    def test_weaker_home(self):
        """ra < rb → < 0.5"""
        assert elo_expected(1400, 1600) < 0.5

    def test_symmetry(self):
        """f(ra,rb) + f(rb,ra) = 1"""
        e_h = elo_expected(1500, 1700)
        e_a = elo_expected(1700, 1500)
        assert abs(e_h + e_a - 1.0) < 1e-12

    def test_extreme_difference(self):
        """极大差距收敛到 0 或 1"""
        assert elo_expected(3000, 1000) > 0.9999
        assert elo_expected(1000, 3000) < 0.0001


# ═══════════════════════════════════════════
#  dc_tau
# ═══════════════════════════════════════════


class TestDcTau:
    def test_high_scores_no_effect(self):
        """x>1 或 y>1 → 1.0"""
        assert dc_tau(2, 0, 1.5, 1.2, 0.1) == 1.0
        assert dc_tau(0, 2, 1.5, 1.2, 0.1) == 1.0
        assert dc_tau(3, 3, 1.5, 1.2, 0.1) == 1.0

    def test_00(self):
        """x=0,y=0 → 1 - rho*λh*λa"""
        expected = 1 - 0.1 * 1.5 * 1.2
        assert abs(dc_tau(0, 0, 1.5, 1.2, 0.1) - expected) < 1e-12

    def test_01(self):
        """x=0,y=1 → 1 + rho*λh"""
        expected = 1 + 0.1 * 1.5
        assert abs(dc_tau(0, 1, 1.5, 1.2, 0.1) - expected) < 1e-12

    def test_10(self):
        """x=1,y=0 → 1 + rho*λa"""
        expected = 1 + 0.1 * 1.2
        assert abs(dc_tau(1, 0, 1.5, 1.2, 0.1) - expected) < 1e-12

    def test_11(self):
        """x=1,y=1 → 1 - rho"""
        expected = 1 - 0.1
        assert abs(dc_tau(1, 1, 1.5, 1.2, 0.1) - expected) < 1e-12

    def test_rho_zero(self):
        """rho=0 时所有组合返回 1.0"""
        for x in range(3):
            for y in range(3):
                assert dc_tau(x, y, 1.5, 1.2, 0.0) == 1.0


# ═══════════════════════════════════════════
#  rq_result_label
# ═══════════════════════════════════════════


class TestRqResultLabel:
    def test_win(self):
        assert rq_result_label(-1, 2, 0) == '让胜'
        assert rq_result_label(0, 1, 0) == '让胜'
        assert rq_result_label(1, 0, 0) == '让胜'  # 受让1球, 0-0 → 让胜

    def test_draw(self):
        assert rq_result_label(-1, 2, 1) == '让平'
        assert rq_result_label(0, 1, 1) == '让平'
        assert rq_result_label(1, 0, 1) == '让平'

    def test_loss(self):
        assert rq_result_label(-1, 1, 1) == '让负'
        assert rq_result_label(0, 0, 1) == '让负'
        assert rq_result_label(1, 0, 2) == '让负'


# ═══════════════════════════════════════════
#  compute_rq_probs
# ═══════════════════════════════════════════


class TestComputeRqProbs:
    def test_sum_to_one(self):
        probs = compute_rq_probs(1.5, 1.2, handicap=0)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_sum_to_one_handicap(self):
        probs = compute_rq_probs(1.5, 1.2, handicap=-1)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_no_handicap_approx_spf(self):
        """handicap=0 时让球概率应近似 SPF"""
        probs = compute_rq_probs(1.8, 1.0, handicap=0)
        # 强队让胜 > 让平 > 让负
        assert probs['让胜'] > probs['让平']
        assert probs['让胜'] > probs['让负']

    def test_positive_handicap_helps_home(self):
        """主队受让时让胜概率更高"""
        probs1 = compute_rq_probs(0.8, 2.0, handicap=0)
        probs2 = compute_rq_probs(0.8, 2.0, handicap=1)
        assert probs2['让胜'] > probs1['让胜']

    def test_negative_handicap_hurts_home(self):
        """主队让球时让胜概率更低"""
        probs1 = compute_rq_probs(2.0, 0.8, handicap=0)
        probs2 = compute_rq_probs(2.0, 0.8, handicap=-1)
        assert probs2['让胜'] < probs1['让胜']

    def test_shrinkage_large_handicap(self):
        """大 handicap 应向 1/3 收缩 (不是精确混合但偏差缩小)"""
        probs = compute_rq_probs(2.0, 0.5, handicap=-3)
        for v in probs.values():
            assert v > 0.05


# ═══════════════════════════════════════════
#  implied_probs_from_odds
# ═══════════════════════════════════════════


class TestImpliedProbsFromOdds:
    def test_sum_to_one(self):
        probs = implied_probs_from_odds(2.0, 3.0, 4.0)
        assert abs(sum(probs.values()) - 1.0) < 1e-6

    def test_fair_odds_all_equal(self):
        """三种赔率相等 → 各 1/3"""
        probs = implied_probs_from_odds(3.0, 3.0, 3.0)
        for v in probs.values():
            assert abs(v - 1 / 3) < 1e-6

    def test_heavy_favorite(self):
        probs = implied_probs_from_odds(1.1, 8.0, 18.0)
        assert probs['H'] > 0.7
        assert probs['A'] < 0.10

    def test_vig_removal(self):
        """100% book vs 有 vig: 比例应一致"""
        without = implied_probs_from_odds(2.0, 3.0, 4.0)
        with_vig = implied_probs_from_odds(1.9, 2.9, 3.8)
        # 应有相同倾向性但 removed vig 后归一化
        ratio_without = without['H'] / without['A']
        ratio_with = with_vig['H'] / with_vig['A']
        assert abs(ratio_without - ratio_with) < 0.1

    def test_zero_odds(self):
        """无效赔率处理: 全部为 0 时不崩溃, 返回全 0"""
        probs = implied_probs_from_odds(0, 0, 0)
        for v in probs.values():
            assert v == 0.0


# ═══════════════════════════════════════════
#  compute_goals_distribution
# ═══════════════════════════════════════════


class TestComputeGoalsDistribution:
    def test_sum_to_one(self):
        """总和应接近 1.0 (Poisson 在 MAX_GOALS 处截断, 允许 ~0.5% 尾部损失)"""
        dist = compute_goals_distribution(1.5, 1.2)
        assert abs(sum(dist.values()) - 1.0) < 5e-3

    def test_keys_up_to_12(self):
        dist = compute_goals_distribution(1.5, 1.2)
        for k in range(13):
            assert str(k) in dist

    def test_low_scoring(self):
        """λ 很小 → 0球概率最大"""
        dist = compute_goals_distribution(0.3, 0.2)
        assert dist['0'] > dist['1']
        assert dist['0'] > 0.5

    def test_high_scoring(self):
        """λ 很大 → 总进球分布右偏"""
        dist = compute_goals_distribution(3.0, 2.5)
        assert dist['0'] < 0.1
        # 3球左右概率最大
        max_key = max(dist, key=lambda k: dist[k] if '球' not in k else 0)
        assert int(max_key) >= 2  # type: ignore[arg-type]

    def test_dc_rho_effect(self):
        """rho != 0 应改变分布"""
        dist_no = compute_goals_distribution(1.5, 1.2, rho=0.0)
        dist_yes = compute_goals_distribution(1.5, 1.2, rho=0.1)
        # 低比分应不同
        assert dist_no['0'] != dist_yes['0']


# ═══════════════════════════════════════════
#  compute_score_topn
# ═══════════════════════════════════════════


class TestComputeScoreTopn:
    def test_count(self):
        rows = compute_score_topn(1.5, 1.2, topn=8)
        assert len(rows) == 8

    def test_count_varied(self):
        rows = compute_score_topn(1.5, 1.2, topn=3)
        assert len(rows) == 3

    def test_sorted_descending(self):
        rows = compute_score_topn(1.5, 1.2, topn=10)
        probs = [r[1] for r in rows]
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))

    def test_tuple_structure(self):
        rows = compute_score_topn(1.5, 1.2)
        row = rows[0]
        assert len(row) == 4
        score_str, prob, hg, ag = row
        assert isinstance(score_str, str)
        assert ':' in score_str
        assert isinstance(prob, float)
        assert isinstance(hg, int)
        assert isinstance(ag, int)

    def test_sum_probabilities(self):
        """top-49 (全部 7×7 组合) 应几乎覆盖全部概率质量"""
        rows = compute_score_topn(1.5, 1.2, topn=49)
        total = sum(r[1] for r in rows)
        assert total > 0.99


# ═══════════════════════════════════════════
#  compute_htft_topn_math
# ═══════════════════════════════════════════

class TestComputeHtftTopnMath:
    """注意: 此函数有外部 import (half_full_model)，跳过如果模块不可用。"""

    def test_returns_tuple(self):
        pytest.importorskip('half_full_model')
        rows, full = compute_htft_topn_math(1.5, 1.2, topn=6)
        assert len(rows) <= 6
        assert len(full) == 9  # 9 宫格
        assert abs(sum(full.values()) - 1.0) < 1e-4

    def test_sorted(self):
        pytest.importorskip('half_full_model')
        rows, _ = compute_htft_topn_math(1.5, 1.2)
        probs = [r[1] for r in rows]
        assert all(probs[i] >= probs[i + 1] for i in range(len(probs) - 1))


# ═══════════════════════════════════════════
#  compute_dynamic_xgb_weight
# ═══════════════════════════════════════════


class TestComputeDynamicXgbWeight:
    def test_uniform_entropy(self):
        """H=D=A 均匀 → 置信度=0 → 低 xgb_w"""
        xgb_w, dc_w, conf = compute_dynamic_xgb_weight([1 / 3, 1 / 3, 1 / 3])
        assert abs(conf) < 1e-10
        # α+β*0 = 0.30, 不低于 min
        assert abs(xgb_w - 0.30) < 1e-10

    def test_certain_prediction(self):
        """某结果概率接近 1 → 高置信度 → 高 xgb_w"""
        xgb_w, dc_w, conf = compute_dynamic_xgb_weight([0.99, 0.005, 0.005])
        assert conf > 0.9
        assert xgb_w >= 0.60

    def test_bounds(self):
        """xgb_w 始终在 [0.10, 0.90]"""
        for probs in [
            [1 / 3, 1 / 3, 1 / 3],
            [0.8, 0.1, 0.1],
            [0.99, 0.01, 0.0],
            [0.5, 0.3, 0.2],
        ]:
            xgb_w, dc_w, _ = compute_dynamic_xgb_weight(probs)
            assert 0.10 <= xgb_w <= 0.90
            assert abs(xgb_w + dc_w - 1.0) < 1e-10

    def test_custom_alpha_beta(self):
        xgb_w1, _, _ = compute_dynamic_xgb_weight([0.5, 0.3, 0.2], alpha=0.5, beta=0.3)
        xgb_w2, _, _ = compute_dynamic_xgb_weight([0.5, 0.3, 0.2], alpha=0.1, beta=0.1)
        assert xgb_w1 != xgb_w2


# ═══════════════════════════════════════════
#  rps_score
# ═══════════════════════════════════════════


class TestRpsScore:
    def test_perfect_prediction(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        y_proba = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert rps_score(y_true, y_proba) == 0.0

    def test_worst_prediction(self):
        """H 实际发生但预测为 A → RPS 应 > 0 且最大值可计算"""
        y_true = np.array([[1, 0, 0]])
        y_proba = np.array([[0, 0, 1]])
        score = rps_score(y_true, y_proba)
        assert score > 0
        assert score <= 1.0


# ═══════════════════════════════════════════
#  quick_validate
# ═══════════════════════════════════════════


class TestQuickValidate:
    def test_perfect(self):
        model_probs = [
            {'H': 0.8, 'D': 0.1, 'A': 0.1},
            {'H': 0.1, 'D': 0.8, 'A': 0.1},
        ]
        actual = ['H', 'D']
        result = quick_validate(model_probs, actual)
        assert result['n'] == 2
        assert result['brier'] > 0  # 非完美 → Brier > 0
        assert 'reliability' in result
        assert 'resolution' in result
        assert 'uncertainty' in result


# ═══════════════════════════════════════════
#  format_pct
# ═══════════════════════════════════════════


class TestFormatPct:
    def test_simple(self):
        assert format_pct(0.256) == '25.6%'

    def test_zero(self):
        assert format_pct(0.0) == '0.0%'

    def test_one(self):
        assert format_pct(1.0) == '100.0%'

    def test_rounding(self):
        assert format_pct(0.6667) == '66.7%'
