"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline

from purchase_predict.pipelines.deployment.nodes import (
    push_to_model_registry,
    stage_model,
)


def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                push_to_model_registry,
                ["params:mlflow_model_registry", "mlflow_model_uri"],
                "mlflow_model_version",
            ),
            Node(
                stage_model,
                ["params:mlflow_model_registry", "mlflow_model_version"],
                None,
            ),
        ]
    )
