from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from components.inference.config import InferenceConfig
from components.utils.constants import DATA_FOLDER_PATH


class Inferrer:
    def __init__(self, inference_config: InferenceConfig) -> None:
        self.config = inference_config
        self.model = self.load_model()

    def load_model(self):
        model_path = (
            DATA_FOLDER_PATH / "models" / self.config.experiment_id / "model.json"
        )
        model = xgb.XGBClassifier()
        model.load_model(model_path)
        return model

    def load_inference_data(self):
        clean_data_path = DATA_FOLDER_PATH / "clean" / self.config.experiment_id
        x_test_path = clean_data_path / "x_test.csv"
        y_test_path = clean_data_path / "y_test.csv"
        x_test = pd.read_csv(x_test_path)
        y_test = pd.read_csv(y_test_path)
        return x_test, y_test

    def predict(self, x: pd.DataFrame):
        # returns the class with the highest probability
        return self.model.predict(x)

    def predict_top_k(self, x: pd.DataFrame, k: Optional[int] = 2) -> np.ndarray:
        """returns the labels of the k classes with the highest probability.

        Args:
            x (pd.DataFrame): dataframe containing the feats.
            k (Optional[int], optional): top k classes. Defaults to 2.

        Returns:
            np.ndarray: ordered predictions according to their probabilities
        """
        assert k <= self.model.get_params()["num_class"], "k must be <= num of classes"
        arg_probs = np.argsort(self.model.predict_proba(x), axis=1)[:, ::-1]
        top_k_preds = arg_probs[:, :k]
        return top_k_preds

    def predict_top_2_for_metrics(self, x: pd.DataFrame, y: pd.DataFrame) -> np.ndarray:
        """This method is used to calculate metrics to asses the model performance.
        The model suggests two possible ranges for the client. If one of the
        suggested ranges is correct, we consider the prediction correct
        and return the correct predicted label. If none of the suggested
        margins are correct, this method returns the label with the
        highest probability.

        Args:
            x (pd.DataFrame): dataframe containing the feats.
            y (pd.DataFrame): dataframe of the ground-truths.

        Returns:
            np.ndarray: ordered predictions according to their probabilities
        """
        top_2_ranges = self.predict_top_k(x, 2)
        y_array = y.values
        if y_array.ndim == 1:
            y_array = y_array.reshape(-1, 1)
        y_pred = np.where(
            np.any(top_2_ranges == y_array, axis=1),
            np.squeeze(y_array),
            top_2_ranges[:, 0],
        )
        return y_pred
