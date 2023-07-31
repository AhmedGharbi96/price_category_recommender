import pickle
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import xgboost as xgb

from components.shap.config import ShapConfig
from components.utils.constants import DATA_FOLDER_PATH


class ShapExplainer:
    def __init__(self, config: ShapConfig) -> None:
        self.config = config
        self.shap_data_folder = DATA_FOLDER_PATH / "shap" / self.config.experiment_id
        self.shap_data_folder.mkdir(exist_ok=True, parents=True)

    def load_forward_and_backward_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        clean_data_path = DATA_FOLDER_PATH / "clean" / self.config.experiment_id
        forward_dataset = pd.read_csv(clean_data_path / "x_test.csv")
        background_dataset = pd.read_csv(clean_data_path / "x_train.csv")
        return forward_dataset, background_dataset

    def load_trained_model(self) -> xgb.XGBModel:
        model_path = DATA_FOLDER_PATH / "models" / self.config.experiment_id
        model = xgb.XGBClassifier()
        model.load_model(model_path / "model.json")
        return model

    def compute_shap_values(
        self,
        model: xgb.XGBModel,
        forward_dataset: pd.DataFrame,
        background_dataset: pd.DataFrame,
    ) -> Tuple[List, np.ndarray]:
        explainer = shap.TreeExplainer(
            model, data=background_dataset, seed=self.config.random_state
        )
        data_to_explain = forward_dataset.sample(
            self.config.n_samples, random_state=self.config.random_state
        )
        shap_values = explainer.shap_values(data_to_explain)
        with open(self.shap_data_folder / "shap_values.pkl", "wb") as handle:
            pickle.dump(shap_values, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return shap_values, data_to_explain

    def generate_and_save_plots(self, shap_values: List, data_to_explain: pd.DataFrame):
        # average abs impact on output
        shap.summary_plot(shap_values, data_to_explain, max_display=80, show=False)
        plt.tight_layout()
        plt.savefig(self.shap_data_folder / "average_absolute_impact.png")
        plt.clf()
