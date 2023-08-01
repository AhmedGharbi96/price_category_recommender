import pandas as pd
from dash import html

from components.dashboard.app_setup import app
from components.dashboard.builders import build_local_explainer_component, build_table
from components.dashboard.config import DashboardConfig
from components.dashboard.utils import b64_image
from components.utils.constants import DATA_FOLDER_PATH
from components.utils.get_configs import load_dashboard_config


def build_app(config: DashboardConfig):
    experiment_id = config.experiment_id
    preds_df = pd.read_csv(
        DATA_FOLDER_PATH / "inference" / experiment_id / "x_test_predictions.csv"
    )
    general_metrics = pd.read_csv(
        DATA_FOLDER_PATH
        / "inference"
        / experiment_id
        / "general_performance_metrics.csv"
    )
    cat_specific_metrics = pd.read_csv(
        DATA_FOLDER_PATH / "inference" / experiment_id / "category_specific_metrics.csv"
    )
    confusion_matrix_path = (
        DATA_FOLDER_PATH / "inference" / experiment_id / "confusion_matrix.png"
    )
    shap_avg_abs_impact = (
        DATA_FOLDER_PATH / "shap" / experiment_id / "average_absolute_impact.png"
    )
    layout = html.Div(
        [
            html.H1("Model's Predictions"),
            build_table(preds_df, "predictions-table", row_selectable="single"),
            html.Button(
                "Explain the prediction",
                id="process-button",
                n_clicks=0,
                style={
                    "font-size": "20px",
                    "padding": "10px 20px",
                    "margin-bottom": "10px",
                },
            ),
            html.H3("Impact of the features on the log odds:"),
            build_local_explainer_component(config),
            html.H1("Average impact of each feature on the model's log odds"),
            html.Div(
                children=html.Img(
                    src=b64_image(shap_avg_abs_impact),
                    id="shap-avg-impact-image",
                    style={"height": "40%", "width": "40%", "margin": "auto"},
                ),
                style={
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                },
            ),
            html.H1("General performance Metrics"),
            build_table(general_metrics, "general-metrics"),
            html.H1("Category Specific Metrics"),
            build_table(cat_specific_metrics, "specific-metrics"),
            html.H1("Confusion Matrix"),
            html.Div(
                children=html.Img(
                    src=b64_image(confusion_matrix_path),
                    id="confusion-matrix-image",
                    style={"height": "60%", "width": "60%", "margin": "auto"},
                ),
                style={
                    "display": "flex",
                    "justify-content": "center",
                    "align-items": "center",
                },
            ),
        ],
    )
    app.layout = layout


if __name__ == "__main__":
    config = load_dashboard_config()
    build_app(config)
    app.run(debug=True)
