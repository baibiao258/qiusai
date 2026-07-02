import json
import argparse
import numpy as np


def devig(odds):
    inv = np.array([1.0 / o for o in odds], dtype=float)
    return (inv / inv.sum()).tolist()


def apply_tilt(p):
    p = np.array(p, dtype=float)
    idx = np.argsort(-p)
    p[idx[0]] += 0.03
    p[idx[1]] -= 0.01
    p[idx[2]] -= 0.02
    p = np.clip(p, 0.02, 0.96)
    p = p / p.sum()
    return p.tolist()


def main(inp, outp):
    data = json.load(open(inp, 'r', encoding='utf-8'))
    for m in data['matches']:
        p = devig(m['odds_1x2'])
        m['proba_1x2'] = apply_tilt(p)
    json.dump(data, open(outp, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(outp)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    main(a.input, a.output)
