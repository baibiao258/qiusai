"""
dc_model_definition.py — Dixon-Coles 模型定义
独立文件, 确保 joblib 序列化兼容.
"""
import math
import numpy as np
from datetime import datetime

MAX_GOALS = 6


class DixonColes:
    """Dixon-Coles (1997) 带时间衰减加权"""

    def __init__(self, time_decay_hl=540):
        self.teams_ = []
        self.team_idx_ = {}
        self.attack_ = None
        self.defense_ = None
        self.rho_ = 0.0
        self.gamma_ = 0.0
        self.global_avg_ = None
        self.half_life_ = time_decay_hl
        self.host_bonus_ = 0.0
        self.fitted_ = False

    def predict_lambda(self, home, away, neutral=True, host_bonus=0.0):
        if not self.fitted_:
            raise RuntimeError("Model not fitted")
        g = self.global_avg_
        h_adv = self.gamma_ if not neutral else 0.0
        h_adv += host_bonus
        h_idx = self.team_idx_.get(home)
        a_idx = self.team_idx_.get(away)
        if h_idx is None or a_idx is None:
            return None, None
        lh = g * math.exp(self.attack_[h_idx] + self.defense_[a_idx] + h_adv)
        la = g * math.exp(self.attack_[a_idx] + self.defense_[h_idx])
        lh = max(0.01, min(8.0, lh))
        la = max(0.01, min(8.0, la))
        return lh, la

    def predict_proba(self, home, away, neutral=True, host_bonus=0.0):
        lh, la = self.predict_lambda(home, away, neutral, host_bonus)
        if lh is None or la is None:
            return None
        hw = dw = aw = 0.0
        for hg in range(MAX_GOALS + 1):
            for ag in range(MAX_GOALS + 1):
                prob = self._dc_pmf(hg, ag, lh, la)
                if hg > ag: hw += prob
                elif hg == ag: dw += prob
                else: aw += prob
        total = hw + dw + aw
        return [hw / total, dw / total, aw / total]

    def _dc_pmf(self, x, y, lh, la):
        tau = 1.0
        if x == 0 and y == 0:
            tau = 1 - self.rho_ * lh * la
        elif x == 0 and y == 1:
            tau = 1 + self.rho_ * lh
        elif x == 1 and y == 0:
            tau = 1 + self.rho_ * la
        elif x == 1 and y == 1:
            tau = 1 - self.rho_
        return tau * (lh ** x * math.exp(-lh) / math.factorial(x)) * (la ** y * math.exp(-la) / math.factorial(y))

    def _weights(self, dates, cutoff=None):
        """指数时间衰减权重计算"""
        if cutoff is None:
            cutoff = datetime.now().strftime('%Y-%m-%d')
        cutoff_dt = datetime.strptime(cutoff, '%Y-%m-%d')
        days = np.array([(cutoff_dt - datetime.strptime(d, '%Y-%m-%d')).days for d in dates])
        return 0.5 ** (np.maximum(days, 0) / self.half_life_)

    def fit(self, df, cutoff=None):
        """全量 MLE 拟合 (使用 scipy.optimize 解析梯度)"""
        pass  # Body is injected by retrain_dc_model.py
