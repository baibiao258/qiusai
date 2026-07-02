from __future__ import annotations

import re

# Canonical English names used by the model.
# Add aliases here once; both single-match and overview scripts import this.
TEAM_ALIASES = {
    # === 官方赛程英文别名 → 模型标准名 ===
    'Korea Republic': 'South Korea',
    'Czechia': 'Czech Republic',
    'Bosnia & Herzegovina': 'Bosnia and Herzegovina',
    'Türkiye': 'Turkey',
    "Côte d'Ivoire": 'Ivory Coast',
    'IR Iran': 'Iran',
    'Cabo Verde': 'Cape Verde',
    'USA': 'United States',

    # Current June 2026 friendly fixtures
    '保加利亚': 'Bulgaria',
    '黑山': 'Montenegro',
    '挪威': 'Norway',
    '瑞典': 'Sweden',
    '土耳其': 'Turkey',
    '北马其顿': 'North Macedonia',
    '奥地利': 'Austria',
    '突尼斯': 'Tunisia',

    # Common Chinese aliases / full names
    '英格兰': 'England',
    '法国': 'France',
    '德国': 'Germany',
    '西班牙': 'Spain',
    '葡萄牙': 'Portugal',
    '巴西': 'Brazil',
    '阿根廷': 'Argentina',
    '荷兰': 'Netherlands',
    '比利时': 'Belgium',
    '意大利': 'Italy',
    '乌拉圭': 'Uruguay',
    '哥伦比亚': 'Colombia',
    '克罗地亚': 'Croatia',
    '摩洛哥': 'Morocco',
    '日本': 'Japan',
    '韩国': 'South Korea',
    '美国': 'United States',
    '墨西哥': 'Mexico',
    '加拿大': 'Canada',
    '澳大利亚': 'Australia',
    '瑞士': 'Switzerland',
    '丹麦': 'Denmark',
    '挪威': 'Norway',
    '瑞典': 'Sweden',
    '芬兰': 'Finland',
    '冰岛': 'Iceland',
    '威尔士': 'Wales',
    '苏格兰': 'Scotland',
    '塞尔维亚': 'Serbia',
    '波兰': 'Poland',
    '捷克': 'Czech Republic',
    '捷克共和国': 'Czech Republic',
    '斯洛伐克': 'Slovakia',
    '斯洛文尼亚': 'Slovenia',
    '斯洛文尼': 'Slovenia',
    '奥地利': 'Austria',
    '匈牙利': 'Hungary',
    '希腊': 'Greece',
    '罗马尼亚': 'Romania',
    '保加利亚': 'Bulgaria',
    '黑山': 'Montenegro',
    '北马其顿': 'North Macedonia',
    '阿尔巴尼亚': 'Albania',
    '波黑': 'Bosnia and Herzegovina',
    '波斯尼亚和黑塞哥维那': 'Bosnia and Herzegovina',
    '爱尔兰': 'Republic of Ireland',
    '北爱尔兰': 'Northern Ireland',
    '冰岛': 'Iceland',
    '塞浦路斯': 'Cyprus',
    '以色列': 'Israel',
    '土耳其': 'Turkey',
    '突尼斯': 'Tunisia',
    '摩尔多瓦': 'Moldova',
    '格鲁吉亚': 'Georgia',
    '亚美尼亚': 'Armenia',
    '阿塞拜疆': 'Azerbaijan',
    '哈萨克斯坦': 'Kazakhstan',
    '乌兹别克斯坦': 'Uzbekistan',
    '加纳': 'Ghana',
    '乌兹别克': 'Uzbekistan',
    '刚果(金)': 'DR Congo',
    '刚果金': 'DR Congo',
    '阿尔及利亚': 'Algeria',
    '尼日利亚': 'Nigeria',
    '卢森堡': 'Luxembourg',
    '巴勒斯坦': 'Palestine',
    '伊朗': 'Iran',
    '伊拉克': 'Iraq',
    '沙特阿拉伯': 'Saudi Arabia',
    '卡塔尔': 'Qatar',
    '阿联酋': 'United Arab Emirates',
    '阿曼': 'Oman',
    '伊拉克': 'Iraq',
    '越南': 'Vietnam',
    '泰国': 'Thailand',
    '印度尼西亚': 'Indonesia',
    '菲律宾': 'Philippines',
    '马来西亚': 'Malaysia',
    '新加坡': 'Singapore',
    '新西兰': 'New Zealand',
    '斐济': 'Fiji',
    '巴拿马': 'Panama',
    '哥斯达黎加': 'Costa Rica',
    '哥斯达': 'Costa Rica',
    '洪都拉斯': 'Honduras',
    '萨尔瓦多': 'El Salvador',
    '危地马拉': 'Guatemala',
    '牙买加': 'Jamaica',
    '特立尼达和多巴哥': 'Trinidad and Tobago',
    '多米尼加共和国': 'Dominican Republic',
    '海地': 'Haiti',
    '厄瓜多尔': 'Ecuador',
    '秘鲁': 'Peru',
    '智利': 'Chile',
    '委内瑞拉': 'Venezuela',
    '巴拉圭': 'Paraguay',
    '玻利维亚': 'Bolivia',
    '约旦': 'Jordan',
    # 2026 世界杯场次中缺失的中→英映射
    '塞内加尔': 'Senegal',
    '埃及': 'Egypt',
    '佛得角': 'Cape Verde',
    '喀麦隆': 'Cameroon',
    '科特迪瓦': 'Ivory Coast',
    '喀麦隆': 'Cameroon',
    '安哥拉': 'Angola',
    '马里': 'Mali',
    '布基纳法索': 'Burkina Faso',
    '赞比亚': 'Zambia',
    '民主刚果': 'DR Congo',
    '民主刚果(金)': 'DR Congo',
}

# Normalization helpers for noisy names like '[102] Norway' or 'Norway[43]'.
_BRACKET_ID_RE = re.compile(r"^\s*\[\d+\]\s*|\s*\[\d+\]\s*$")


def normalize_team_name(name: str) -> str:
    """Normalize Chinese/English/alias team names to canonical English names."""
    if not name:
        return name
    s = str(name).strip()
    s = _BRACKET_ID_RE.sub('', s)
    s = s.replace('（', '(').replace('）', ')')
    # remove trailing source tags like [英82]
    s = re.sub(r"\[\D*\d+\]$", '', s).strip()
    s = TEAM_ALIASES.get(s, s)
    return s


def normalize_match_pair(home: str, away: str) -> tuple[str, str]:
    return normalize_team_name(home), normalize_team_name(away)
