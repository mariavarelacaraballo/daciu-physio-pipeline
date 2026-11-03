"""Small-sample classification with permutation testing.

With few subjects and correlated windows, a high accuracy alone means little.
`permutation_test_score` (Ojala & Garriga, 2010) refits each model on label-shuffled
data to build the chance distribution and returns a p-value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedKFold, permutation_test_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def default_models(seed: int = 42) -> dict:
    """Scaling inside the pipeline, so the scaler is fit on training folds only."""
    return {
        "Random Forest": make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=seed)),
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", random_state=seed)),
    }


def evaluate(X, y, groups=None, models: dict | None = None, n_permutations: int = 200,
             n_splits: int = 5, seed: int = 42, n_jobs: int = -1,
             scoring: str = "balanced_accuracy") -> pd.DataFrame:
    """Score, chance-level score and permutation p-value for each model.

    Balanced accuracy by default: with unequal class sizes, plain accuracy
    rewards always predicting the majority class.

    If `groups` (e.g. subject IDs) is given, GroupKFold is used so no subject is
    in both train and test — the honest setting for generalisation to new people.
    Otherwise stratified K-fold (within-subject evaluation).

    Caveat: overlapping windows (50 %) are not independent; with plain K-fold,
    neighbours of a test window can sit in train. Compare with 0 % overlap or
    blocked CV before claiming generalisation.
    """
    models = models or default_models(seed)
    if groups is not None:
        cv = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, model in models.items():
        score, perm, p = permutation_test_score(model, X, y, groups=groups, cv=cv, scoring=scoring,
                                                n_permutations=n_permutations, n_jobs=n_jobs,
                                                random_state=seed)
        rows.append({"model": name, scoring: score, "chance": perm.mean(), "p_value": p})
    return pd.DataFrame(rows)


def feature_importance(X: pd.DataFrame, y, seed: int = 42) -> pd.Series:
    """Gini importances of a Random Forest fit on all data (descriptive only)."""
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, random_state=seed)).fit(X, y)
    imp = rf.named_steps["randomforestclassifier"].feature_importances_
    return pd.Series(imp, index=X.columns).sort_values()
