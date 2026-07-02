"""Global configuration constants for the prediction pipeline.

All hard-coded parameters previously scattered across daily_jczq.py
are centralized here.  Import from this module everywhere else.
"""
import os

# ── File paths ──
DATA_DIR = os.environ.get('DATA_DIR', '/root/data')
PREDICTIONS_LOG = os.path.join(DATA_DIR, 'predictions_log.csv')
BACKTEST_SCRIPT = os.environ.get(
    'BACKTEST_SCRIPT',
    '/root/.hermes/scripts/backtest_jczq.py',
)
MODEL_VERSION = 'daily_jczq_v4'

# ── Poisson / Dixon-Coles ──
MAX_GOALS = 6
HALF_FULL_R_HT = 0.45          # 半场/全场节奏比
HALF_FULL_MAX_HT = 8           # 半场最大进球枚举
HALF_FULL_MAX_FT = 10          # 全场最大进球枚举

# ── 动态 XGB 权重 (熵融合) ──
XGB_WEIGHT_ALPHA = 0.30
XGB_WEIGHT_BETA = 0.50
XGB_WEIGHT_MIN = 0.10
XGB_WEIGHT_MAX = 0.90

# ── 概率防御上限 ──
PROB_CAP = 0.75

# ── 竞彩足球覆盖联赛 (football-data.org codes) ──
JCZQ_LEAGUES = [
    ('PL', '英超'), ('BL1', '德甲'), ('PD', '西甲'),
    ('SA', '意甲'), ('FL1', '法甲'), ('DED', '荷甲'),
    ('PPL', '葡超'), ('ELC', '英冠'),
]

# ── HTFT 显示映射 (9 宫格) ──
HTFT_ORDER = ['胜胜', '胜平', '胜负', '平胜', '平平', '平负', '负胜', '负平', '负负']
HTFT_SHORT_MAP = {
    '胜胜': 'HH', '胜平': 'HD', '胜负': 'HA',
    '平胜': 'DH', '平平': 'DD', '平负': 'DA',
    '负胜': 'AH', '负平': 'AD', '负负': 'AA',
}
HTFT_DISPLAY_MAP = {
    'HH': '胜胜', 'HD': '胜平', 'HA': '胜负',
    'DH': '平胜', 'DD': '平平', 'DA': '平负',
    'AH': '负胜', 'AD': '负平', 'AA': '负负',
    'H/H': '胜胜', 'H/D': '胜平', 'H/A': '胜负',
    'D/H': '平胜', 'D/D': '平平', 'D/A': '平负',
    'A/H': '负胜', 'A/D': '负平', 'A/A': '负负',
}
