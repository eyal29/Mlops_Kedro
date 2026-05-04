# """
# This is a boilerplate pipeline 'loading'
# generated using Kedro 1.2.0
# """
# import pandas as pd
# import glob

# from google.cloud import storage


# def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
#     """
#     Loads multiple CSV files from bucket (as generated from Spark).
#     """
#     storage_client = storage.Client()
#     bucket_name = bucket_path.split("/")[0]
#     folder = "/".join(bucket_path.split("/")[1:]) + "/part-"

#     for blob in storage_client.list_blobs(bucket_name, prefix=folder):
#         filename = blob.name.split("/")[-1]
#         if filename[-3:] == "csv":
#             blob.download_to_filename("/tmp/" + filename)

#     all_files = glob.glob("/tmp/*.csv")
#     li = []

#     for filename in all_files:
#         df = pd.read_csv(filename, index_col=None, sep=",")
#         li.append(df)

#     df = pd.concat(li, axis=0, ignore_index=True)
#     return df


import os
import tempfile

import pandas as pd
from google.cloud import storage


def load_csv_from_bucket(project: str, bucket_path: str) -> pd.DataFrame:
    """
    Loads multiple CSV files from bucket (as generated from Spark).
    """
    storage_client = storage.Client(project=project)
    bucket_name = bucket_path.split("/")[0]
    folder = "/".join(bucket_path.split("/")[1:]) + "/part-"

    temp_dir = tempfile.gettempdir()
    downloaded_files = []

    for blob in storage_client.list_blobs(bucket_name, prefix=folder):
        filename = blob.name.split("/")[-1]
        if filename.endswith("csv"):
            local_path = os.path.join(temp_dir, filename)
            blob.download_to_filename(local_path)
            downloaded_files.append(local_path)

    if not downloaded_files:
        raise ValueError(
            f"No CSV files found in gs://{bucket_name}/{folder}. "
            "Check gcs_primary_folder and ensure Spark output files exist (part-*.csv)."
        )

    li = []

    for filename in downloaded_files:
        df = pd.read_csv(filename, index_col=None, sep=",")
        li.append(df)

    df = pd.concat(li, axis=0, ignore_index=True)
    return df
