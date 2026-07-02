import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from sklearn.isotonic import IsotonicRegression


@dataclass
class CalibratedModel:
    models: Dict[int, IsotonicRegression]
    classes_: List[int]

    def predict_proba(self, proba_raw: np.ndarray) -> np.ndarray:
        n, k = proba_raw.shape
        out = np.zeros_like(proba_raw, dtype=float)
        for j, c in enumerate(self.classes_):
            out[:, j] = self.models[c].predict(proba_raw[:, j])
        s = out.sum(axis=1, keepdims=True)
        s[s == 0] = 1.0
        return out / s


def fit_isotonic_multiclass(proba_raw: np.ndarray, y_true: np.ndarray, classes_: List[int]) -> CalibratedModel:
    models = {}
    for j, c in enumerate(classes_):
        y_bin = (y_true == c).astype(int)
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(proba_raw[:, j], y_bin)
        models[c] = ir
    return CalibratedModel(models=models, classes_=classes_)


def brier_multiclass(proba: np.ndarray, y_true: np.ndarray, classes_: List[int]) -> float:
    n, k = proba.shape
    y_one = np.zeros((n, k), dtype=float)
    idx = {c: i for i, c in enumerate(classes_)}
    for i, y in enumerate(y_true):
        y_one[i, idx[y]] = 1.0
    return float(np.mean(np.sum((proba - y_one) ** 2, axis=1)))
