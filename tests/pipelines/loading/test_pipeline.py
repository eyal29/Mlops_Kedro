"""
This is a boilerplate test file for pipeline 'loading'
generated using Kedro 1.2.0.
Please add your pipeline tests here.

Kedro recommends using `pytest` framework, more info about it can be found
in the official documentation:
https://docs.pytest.org/en/latest/getting-started.html
"""

import pandas as pd
from kedro.runner import SequentialRunner

from purchase_predict.pipelines.loading.pipeline import create_pipeline


def test_pipeline(catalog_test):
    runner = SequentialRunner()
    pipeline = create_pipeline()
    pipeline_output = runner.run(pipeline, catalog_test)
    df = pipeline_output["primary"].load()
    assert type(df) is pd.DataFrame
    assert df.shape[1] == 16
    assert "purchased" in df
