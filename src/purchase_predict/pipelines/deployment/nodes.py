"""
This is a boilerplate pipeline 'deployment'
generated using Kedro 1.2.0
"""

import os
import mlflow

from mlflow.tracking import MlflowClient


def push_to_model_registry(registry_name: str, model_uri: str) -> str:
    """
    Pushes a model's version to the specified registry.
    """
    tracking_uri = os.getenv("MLFLOW_SERVER")
    if not tracking_uri:
        raise ValueError("MLFLOW_SERVER environment variable is not set")
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()
    result = client.create_model_version(
        name=registry_name,
        source=model_uri,  # Pass directly, no runs:/ wrapping
    )

    return result.version


def stage_model(registry_name: str, version: str) -> None:
    """
    Assigns an alias (e.g., 'staging') to a model version in the registry.
    Use model URI: models://registry_name@staging for deployments.
    """
    env = os.getenv("ENV")
    if env is None:
        return

    alias = env  # 'staging' or 'production' as alias name
    client = MlflowClient()

    # Assign alias to this version (overwrites prior if same alias)
    client.set_registered_model_alias(name=registry_name, alias=alias, version=version)
