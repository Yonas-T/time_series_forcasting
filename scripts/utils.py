
import dvc.api as dvc
import pandas as pd
import logging
import pickle



def get_data_frame_from_dvc(path: str):
    with dvc.open(
        path,
        # repo=repo_path,
        mode='rb',

    ) as data:
        df_from_dvc = pd.read_csv(data)
    return df_from_dvc

def read_csv(csv_path, missing_values=[]):
    logging.basicConfig(level=logging.INFO)
    try:
        df = pd.read_csv(csv_path, na_values=missing_values)
        logging.info("the file is read")
        return df
    except FileNotFoundError:
        logging.info(f"file not found at {csv_path}")

def write_csv(df, csv_path):
    logging.basicConfig(level=logging.INFO)
    try:
        df.to_csv(csv_path, index=False)
        logging.info("writing successful")

    except Exception:
        logging.info("writing failed")

    return df

def read_model(file_name):
    with open(f"../models/{file_name}.pkl", "rb") as f:
        logging.info(f"Model loaded from {file_name}.pkl")
        return pickle.load(f)

def write_model(file_name, model):
    with open(f"../models/{file_name}.pkl", "wb") as f:
        logging.info(f"Model dumped to {file_name}.pkl")
        pickle.dump(model, f)
    