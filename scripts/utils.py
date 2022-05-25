
import dvc.api
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

def get_data_frame_from_dvc(path: str):
    with dvc.api.open(
        path,
        # repo=repo_path,
        mode='rb',

    ) as data:
        df_from_dvc = pd.read_csv(data)
    return df_from_dvc

def read_csv(csv_path, missing_values=[]):
    try:
        df = pd.read_csv(csv_path, na_values=missing_values)
        logging.info("the file is read")
        return df
    except FileNotFoundError:
        logging.info(f"file not found at {csv_path}")

def write_csv(df, csv_path):
    try:
        df.to_csv(csv_path, index=False)
        logging.info("writing successful")

    except Exception:
        logging.info("writing failed")

    return df
    