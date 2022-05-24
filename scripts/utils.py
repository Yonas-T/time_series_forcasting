
import dvc.api
import pandas as pd

def get_data_frame_from_dvc(path: str):
    with dvc.api.open(
        path,
        # repo=repo_path,
        mode='rb',

    ) as data:
        df_from_dvc = pd.read_csv(data)
    return df_from_dvc
    