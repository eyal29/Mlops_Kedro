"""
This is a boilerplate pipeline 'training'
generated using Kedro 1.2.0
"""
import numpy as np
import pandas as pd

from collections.abc import Callable

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedKFold

from lightgbm.sklearn import LGBMClassifier

from typing import Any, TypedDict
from hyperopt import hp, tpe, fmin

import warnings
warnings.filterwarnings('ignore')

class ModelSpec(TypedDict, total=True):
    name: str
    model_class: Callable[..., Any]  # Covers LGBMClassifier()
    params: dict[str, Any]  # Accepts hp.uniform etc.
    override_schemas: dict[str, type]


# Type the MODELS list
MODELS: list[ModelSpec] = [
    {
        "name": "LightGBM",
        "model_class": LGBMClassifier,
        "params": {
            "objective": "binary",
            "verbose": -1,
            "learning_rate": hp.uniform("learning_rate", 0.001, 1),
            "num_iterations": hp.quniform("num_iterations", 100, 1000, 20),
            "max_depth": hp.quniform("max_depth", 4, 12, 6),
            "num_leaves": hp.quniform("num_leaves", 8, 128, 10),
            "colsample_bytree": hp.uniform("colsample_bytree", 0.3, 1),
            "subsample": hp.uniform("subsample", 0.5, 1),
            "min_child_samples": hp.quniform("min_child_samples", 1, 20, 10),
            "reg_alpha": hp.choice("reg_alpha", [0, 1e-1, 1, 2, 5, 10]),
            "reg_lambda": hp.choice("reg_lambda", [0, 1e-1, 1, 2, 5, 10]),
        },
        "override_schemas": {
            "num_leaves": int,
            "min_child_samples": int,
            "max_depth": int,
            "num_iterations": int,
        },
    }
]

def get_model_config(instance: BaseEstimator) -> ModelSpec:
    """Returns the configuration dictionary for the given model instance."""

    for model_spec in MODELS:
        model_cls: type = model_spec["model_class"]  # type: ignore (or guard below)
        if isinstance(model_cls, type) and isinstance(instance, model_cls):
            return model_spec
    raise ValueError(f"Unsupported model: {type(instance)}")

def train_model(
    instance: BaseEstimator,
    training_set: tuple[np.ndarray, np.ndarray],
    params: dict[str, Any] | None = None,
) -> BaseEstimator:
    """
    Trains a new instance of model with supplied training set and hyper-parameters.
    """
    model_conf = get_model_config(instance)
    params = params or {}

    override_schemas = model_conf.get("override_schemas", {})
    for p in params:
        if p in override_schemas:
            params[p] = override_schemas[p](params[p])

    model = clone(instance)
    model.set_params(**params)
    model.fit(*training_set)
    return model


def optimize_hyp(
    instance: BaseEstimator,
    dataset: tuple[pd.DataFrame, pd.Series],
    search_space: dict,
    metric: Callable[[Any, Any], float],
    max_evals: int = 40,
) -> dict:
    """
    Trains model's instances on hyper-parameters search space and returns most accurate
    hyper-parameters based on eval set.
    """
    X, y = dataset

    def objective(params):
        rep_kfold = RepeatedKFold(n_splits=4, n_repeats=1)
        scores_test = []
        for train_I, test_I in rep_kfold.split(X):
            X_fold_train = X.iloc[train_I, :]
            y_fold_train = y.iloc[train_I].values.flatten()
            X_fold_test = X.iloc[test_I, :]
            y_fold_test = y.iloc[test_I].values.flatten()
            # On entraîne une instance du modèle avec les paramètres params
            model = train_model(
                instance=instance,
                training_set=(X_fold_train, y_fold_train),
                params=params,
            )
            # On calcule le score du modèle sur le test
            scores_test.append(metric(y_fold_test, model.predict(X_fold_test)))  # type: ignore[union-attr]

        return np.mean(scores_test)

    return fmin(fn=objective, space=search_space, algo=tpe.suggest, max_evals=max_evals)

def auto_ml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    max_evals: int = 40
) -> dict[str, BaseEstimator]:
    """
    Runs training of multiple model instances and select the most accurated based on objective function.
    """
    X = pd.concat((X_train, X_test))
    y = pd.concat((y_train, y_test))

    opt_models = []
    for model_specs in MODELS:
        # Finding best hyper-parameters with bayesian optimization
        optimum_params = optimize_hyp(
            model_specs["model_class"](),
            dataset=(X, y),
            search_space=model_specs["params"],
            metric=lambda x, y: -f1_score(x, y),
            max_evals=max_evals
        )
        print("done")
        # Training the supposed best model with found hyper-parameters
        model = train_model(
            model_specs["model_class"](),
            training_set=(X_train, y_train),
            params=optimum_params,
        )
        opt_models.append(
            {
                "model": model,
                "name": model_specs["name"],
                "params": optimum_params,
                "score": f1_score(y_test, model.predict(X_test)),
            }
        )

    # In case we have multiple models
    best_model = max(opt_models, key=lambda x: x["score"])
    return dict(model=best_model)
