"""
This is a boilerplate pipeline 'loading'
generated using Kedro 1.2.0
"""

from kedro.pipeline import Node, Pipeline  # noqa
from .nodes import load_csv_from_bucket

def create_pipeline(**kwargs):
    return Pipeline(
        [
            Node(
                load_csv_from_bucket,
                ["params:gcp_project_id", "params:gcs_primary_folder"],
                "primary",
            )
        ]
    )
