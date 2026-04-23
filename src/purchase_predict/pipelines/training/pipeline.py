"""
This is a boilerplate pipeline 'training'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import auto_ml


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                auto_ml,
                ["X_train", "y_train", "X_test", "y_test", "params:automl_max_evals"],
                dict(model="model"),
            )
        ]
    )
