import logging

from components.shap.config import ShapConfig
from components.shap.explainer import ShapExplainer
from components.utils.get_configs import load_shap_config


def run_shap(config: ShapConfig) -> None:
    logging.info("**** Starting Shapley Values computation workflow ****")
    explainer = ShapExplainer(config)
    model = explainer.load_trained_model()
    forward_dataset, background_dataset = explainer.load_forward_and_backward_datasets()
    shap_values, data_to_explain = explainer.compute_shap_values(
        model, forward_dataset, background_dataset
    )
    explainer.generate_and_save_plots(shap_values, data_to_explain)
    logging.info("**** SHAP workflow finished successfully. ****")

if __name__ == "__main__":
    config = load_shap_config()
    run_shap(config)
