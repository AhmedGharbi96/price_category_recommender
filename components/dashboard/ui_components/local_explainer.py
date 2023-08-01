from typing import Dict

import matplotlib

matplotlib.use("agg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import xgboost as xgb
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from PIL import Image

from components.dashboard.app_setup import app
from components.dashboard.config import DashboardConfig
from components.dashboard.utils import b64_image
from components.utils.constants import DATA_FOLDER_PATH


class LocalExplainerComponent:
    def __init__(self, config: DashboardConfig) -> None:
        self.config = config
        Path(DATA_FOLDER_PATH / "dashboard" / self.config.experiment_id).mkdir(
            exist_ok=True, parents=True
        )
        self.get_unique_labels()
        self.create_local_explainer()
        self.create_init_image()

    def create_init_image(self):
        fig = px.imshow(
            np.random.random((32, 32, 3)), title="Select a row then click the button !"
        )
        data_path = DATA_FOLDER_PATH / "dashboard" / self.config.experiment_id
        fig.write_image(data_path / "init_image.png")

    def create_local_explainer(self):
        model = xgb.XGBClassifier()
        model.load_model(
            DATA_FOLDER_PATH / "models" / self.config.experiment_id / "model.json"
        )
        background_dataset = pd.read_csv(
            DATA_FOLDER_PATH / "clean" / self.config.experiment_id / "x_train.csv"
        )
        self.explainer = shap.TreeExplainer(model, data=background_dataset)

    def get_unique_labels(self):
        y_train = pd.read_csv(
            DATA_FOLDER_PATH / "clean" / self.config.experiment_id / "y_train.csv"
        )
        self.unique_labels = np.unique(y_train)

    def build_shap_figure(self, data: Dict):
        data_path = DATA_FOLDER_PATH / "dashboard" / self.config.experiment_id
        data_path.mkdir(parents=True, exist_ok=True)
        sample_to_explain = pd.DataFrame(data, index=[0])
        sample_to_explain.drop(columns=["first_range", "second_range"], inplace=True)
        for label in self.unique_labels:
            shap.plots.bar(
                self.explainer(sample_to_explain)[:, :, label].mean(0), show=False
            )
            fig = plt.gcf()
            fig.suptitle(f"label {label}")
            fig.tight_layout()
            fig.savefig(data_path / f"label_{label}.png")
            plt.clf()
        # TODO: find a better way to convert from plt Fig to np array
        images = [
            Image.open(data_path / f"label_{label}.png") for label in self.unique_labels
        ]
        images_arr = list(map(np.array, images))
        combined_arr = np.vstack([np.hstack(images_arr[:3]), np.hstack(images_arr[3:])])
        combined_image = Image.fromarray(combined_arr)
        combined_image.save(data_path / "combined_plot.png")

    def build(self):
        data_path = DATA_FOLDER_PATH / "dashboard" / self.config.experiment_id
        return html.Div(
            style={
                "display": "flex",
                "justify-content": "center",
                "align-items": "center",
                "height": "100vh",
                "border": "5px solid black",
                # "padding": "3px",  # Add some padding around the image
            },
            children=[
                html.Img(
                    src=b64_image(data_path / "init_image.png"),
                    id="combined-image",
                    style={"height": "100%", "width": "80%", "margin": "auto"},
                )
            ],
        )

    def register_callbacks(self) -> None:
        @app.callback(
            [
                Output(component_id="combined-image", component_property="src"),
            ],
            [Input("process-button", "n_clicks")],
            [
                State("predictions-table", "selected_rows"),
                State("predictions-table", "data"),
            ],
        )
        def refresh_data(n_clicks, selected_rows, data):
            if n_clicks > 0 and selected_rows:
                selected_row_index = selected_rows[0]
                selected_row_data = data[selected_row_index]
                self.build_shap_figure(selected_row_data)
                return (
                    b64_image(
                        DATA_FOLDER_PATH
                        / "dashboard"
                        / self.config.experiment_id
                        / "combined_plot.png"
                    ),
                )
            else:
                raise PreventUpdate
