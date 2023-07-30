from typing import Tuple

import numpy as np
import pandas as pd
import xgboost as xgb

from components.training.config import TrainingConfig
from components.utils.constants import DATA_FOLDER_PATH


class Trainer:
    def __init__(self, training_config: TrainingConfig) -> None:
        self.config = training_config
        self.models_path = DATA_FOLDER_PATH / "models" / self.config.experiment_id
        self.models_path.mkdir(parents=True, exist_ok=True)

    def load_train_val_data(self) -> Tuple:
        clean_data_path = DATA_FOLDER_PATH / "clean" / self.config.experiment_id
        x_train = pd.read_csv(clean_data_path / "x_train.csv")
        y_train = pd.read_csv(clean_data_path / "y_train.csv")
        x_val_path = clean_data_path / "x_val.csv"
        y_val_path = clean_data_path / "y_val.csv"
        if x_val_path.exists() and y_val_path.exists():
            x_val = pd.read_csv(x_val_path)
            y_val = pd.read_csv(y_val_path)
        else:
            x_val, y_val = None, None
        return x_train, y_train, x_val, y_val

    def train_model(
        self,
        x_train: pd.DataFrame,
        y_train: pd.DataFrame,
        x_val: pd.DataFrame,
        y_val: pd.DataFrame,
    ) -> xgb.XGBModel:
        params = self.config.xgboost_config.dict()
        num_class = np.unique(y_train).size
        eval_set = None
        if x_val is not None and y_val is not None:
            eval_set = [(x_val, y_val)]
        xgb_model = xgb.XGBClassifier(**params, num_class=num_class, tree_method="hist")
        xgb_model.fit(x_train, y_train, eval_set=eval_set)
        return xgb_model

    def save_model(self, model: xgb.XGBModel) -> None:
        model.save_model(self.models_path / "model.json")
