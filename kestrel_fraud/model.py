"""Model factories. Both are plain scikit-learn pipelines so training and inference
apply identical preprocessing."""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

RANDOM_STATE = 42


def make_logreg(C: float = 0.3, numeric=None, categorical=None) -> Pipeline:
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), categorical),
    ])
    clf = LogisticRegression(C=C, max_iter=20000, tol=1e-10, solver="lbfgs")  # tight tol: solver-build independent
    return Pipeline([("pre", pre), ("clf", clf)])


def make_hgb(learning_rate: float = 0.05, max_leaf_nodes: int = 15, min_samples_leaf: int = 40,
             l2: float = 1.0, max_iter: int = 300, numeric=None, categorical=None) -> Pipeline:
    numeric = NUMERIC_FEATURES if numeric is None else numeric
    categorical = CATEGORICAL_FEATURES if categorical is None else categorical
    pre = ColumnTransformer([
        ("num", "passthrough", numeric),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                               encoded_missing_value=np.nan), categorical),
    ])
    n_num = len(numeric)
    cat_mask = [False] * n_num + [True] * len(categorical)
    clf = HistGradientBoostingClassifier(
        learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        l2_regularization=l2, max_iter=max_iter, categorical_features=cat_mask,
        random_state=RANDOM_STATE, early_stopping=False,
    )
    return Pipeline([("pre", pre), ("clf", clf)])
