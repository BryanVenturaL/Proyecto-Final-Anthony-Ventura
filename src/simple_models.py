from collections import Counter

import numpy as np
import pandas as pd


class Encoder:
    def fit(self, frame: pd.DataFrame):
        self.columns = list(frame.columns)
        self.numeric = [c for c in self.columns if pd.api.types.is_numeric_dtype(frame[c])]
        self.categorical = [c for c in self.columns if c not in self.numeric]
        self.medians = {c: float(pd.to_numeric(frame[c], errors="coerce").median()) for c in self.numeric}
        self.means, self.stds = {}, {}
        for c in self.numeric:
            values = pd.to_numeric(frame[c], errors="coerce").fillna(self.medians[c])
            self.means[c] = float(values.mean())
            self.stds[c] = float(values.std()) or 1.0
        self.levels = {
            c: sorted(frame[c].fillna("DESCONOCIDO").astype(str).unique().tolist())
            for c in self.categorical
        }
        self.feature_names = list(self.numeric)
        for c in self.categorical:
            self.feature_names.extend([f"{c}={level}" for level in self.levels[c]])
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        parts = []
        for c in self.numeric:
            values = pd.to_numeric(frame[c], errors="coerce").fillna(self.medians[c]).to_numpy(float)
            parts.append(((values - self.means[c]) / self.stds[c]).reshape(-1, 1))
        for c in self.categorical:
            values = frame[c].fillna("DESCONOCIDO").astype(str).to_numpy()
            parts.append(np.column_stack([(values == level).astype(float) for level in self.levels[c]]))
        return np.column_stack(parts)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


class MedianRegressor:
    def fit(self, x, y):
        self.value = float(np.median(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.value)


class RidgeRegressor:
    def __init__(self, alpha=10.0):
        self.alpha = alpha

    def fit(self, x, y):
        design = np.column_stack([np.ones(len(x)), x])
        penalty = np.eye(design.shape[1]) * self.alpha
        penalty[0, 0] = 0
        self.coef_ = np.linalg.pinv(design.T @ design + penalty) @ design.T @ np.asarray(y)
        return self

    def predict(self, x):
        return np.column_stack([np.ones(len(x)), x]) @ self.coef_

    @property
    def feature_importances_(self):
        return np.abs(self.coef_[1:])


class LogTargetRegressor:
    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, x, y):
        self.estimator.fit(x, np.log1p(y))
        return self

    def predict(self, x):
        return np.expm1(self.estimator.predict(x))

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_


class MajorityClassifier:
    def fit(self, x, y):
        self.classes_ = np.unique(y)
        self.value = Counter(y).most_common(1)[0][0]
        return self

    def predict(self, x):
        return np.full(len(x), self.value, dtype=object)

    def predict_proba(self, x):
        output = np.zeros((len(x), len(self.classes_)))
        output[:, list(self.classes_).index(self.value)] = 1
        return output


class _Tree:
    def __init__(self, task, max_depth, min_leaf, max_features, rng):
        self.task, self.max_depth, self.min_leaf = task, max_depth, min_leaf
        self.max_features, self.rng = max_features, rng

    def fit(self, x, y):
        self.importance = np.zeros(x.shape[1])
        self.root = self._grow(x, np.asarray(y), 0)
        return self

    def _loss(self, y):
        if len(y) == 0:
            return 0.0
        if self.task == "regression":
            return float(np.sum((y - np.mean(y)) ** 2))
        counts = np.array(list(Counter(y).values()), dtype=float)
        return float(len(y) * (1 - np.sum((counts / len(y)) ** 2)))

    def _leaf(self, y):
        value = float(np.mean(y)) if self.task == "regression" else Counter(y).most_common(1)[0][0]
        return {"value": value}

    def _grow(self, x, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_leaf or self._loss(y) == 0:
            return self._leaf(y)
        features = self.rng.choice(x.shape[1], min(self.max_features, x.shape[1]), replace=False)
        parent_loss, best = self._loss(y), None
        for feature in features:
            values = np.unique(x[:, feature])
            if len(values) < 2:
                continue
            thresholds = np.unique(np.quantile(values, np.linspace(0.1, 0.9, 9))) if len(values) > 10 else (values[:-1] + values[1:]) / 2
            for threshold in thresholds:
                left = x[:, feature] <= threshold
                if left.sum() < self.min_leaf or (~left).sum() < self.min_leaf:
                    continue
                gain = parent_loss - self._loss(y[left]) - self._loss(y[~left])
                if best is None or gain > best[0]:
                    best = (gain, feature, float(threshold), left)
        if best is None:
            return self._leaf(y)
        gain, feature, threshold, left = best
        self.importance[feature] += max(gain, 0)
        return {
            "feature": feature, "threshold": threshold,
            "left": self._grow(x[left], y[left], depth + 1),
            "right": self._grow(x[~left], y[~left], depth + 1),
        }

    def _one(self, row, node):
        while "value" not in node:
            node = node["left"] if row[node["feature"]] <= node["threshold"] else node["right"]
        return node["value"]

    def predict(self, x):
        return np.array([self._one(row, self.root) for row in x])


class RandomForest:
    def __init__(self, task, n_estimators=60, max_depth=6, min_leaf=4, random_state=42):
        self.task, self.n_estimators = task, n_estimators
        self.max_depth, self.min_leaf, self.random_state = max_depth, min_leaf, random_state

    def fit(self, x, y):
        y, rng = np.asarray(y), np.random.default_rng(self.random_state)
        self.classes_ = np.unique(y) if self.task == "classification" else None
        self.trees, importance = [], np.zeros(x.shape[1])
        for _ in range(self.n_estimators):
            idx = rng.integers(0, len(x), len(x))
            tree = _Tree(self.task, self.max_depth, self.min_leaf, max(1, int(np.sqrt(x.shape[1]))), rng).fit(x[idx], y[idx])
            self.trees.append(tree)
            importance += tree.importance
        total = importance.sum()
        self.feature_importances_ = importance / total if total else importance
        return self

    def predict(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        if self.task == "regression":
            return predictions.astype(float).mean(axis=0)
        return np.array([Counter(predictions[:, i]).most_common(1)[0][0] for i in range(len(x))])

    def predict_proba(self, x):
        predictions = np.array([tree.predict(x) for tree in self.trees])
        return np.column_stack([(predictions == cls).mean(axis=0) for cls in self.classes_])


class ModelBundle:
    def __init__(self, estimator):
        self.encoder, self.estimator = Encoder(), estimator

    def fit(self, frame, y):
        self.estimator.fit(self.encoder.fit_transform(frame), np.asarray(y))
        return self

    def predict(self, frame):
        return self.estimator.predict(self.encoder.transform(frame))

    def predict_proba(self, frame):
        return self.estimator.predict_proba(self.encoder.transform(frame))

    @property
    def classes_(self):
        return self.estimator.classes_

    @property
    def feature_names_(self):
        return self.encoder.feature_names

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_
