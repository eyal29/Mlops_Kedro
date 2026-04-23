from kedro.runner import SequentialRunner
from purchase_predict.pipelines.processing.pipeline import create_pipeline


def test_pipeline(catalog_test):
    runner = SequentialRunner()
    pipeline = create_pipeline()
    pipeline_output = runner.run(pipeline, catalog_test)
    # Load data from MemoryDataset objects
    X_train = pipeline_output["X_train"].load()
    y_train = pipeline_output["y_train"].load()
    X_test = pipeline_output["X_test"].load()
    y_test = pipeline_output["y_test"].load()

    assert X_train.shape[0] == y_train.shape[0]
    assert X_test.shape[0] == y_test.shape[0]
