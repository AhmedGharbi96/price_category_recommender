from components.dashboard.app import build_app
from components.dashboard.app_setup import app
from components.utils.get_configs import (load_dashboard_config,
                                          load_data_processing_config,
                                          load_fullrun_config,
                                          load_inference_config,
                                          load_shap_config,
                                          load_training_config)
from scripts.data_processing import run_data_processing
from scripts.inference import run_inference
from scripts.shap_values import run_shap
from scripts.train import run_model_training


def main():
    fullrun_config = load_fullrun_config()
    processing_config = load_data_processing_config()
    training_config = load_training_config()
    inference_config = load_inference_config()
    shap_config = load_shap_config()
    dashbord_config = load_dashboard_config()
    processing_config.experiment_id = fullrun_config.experiment_id
    training_config.experiment_id = fullrun_config.experiment_id
    inference_config.experiment_id = fullrun_config.experiment_id
    shap_config.experiment_id = fullrun_config.experiment_id
    dashbord_config.experiment_id = fullrun_config.experiment_id

    if fullrun_config.components.do_processing:
        run_data_processing(processing_config)
    if fullrun_config.components.do_training:
        run_model_training(training_config)
    if fullrun_config.components.do_inference:
        run_inference(inference_config)
    if fullrun_config.components.do_shap:
        run_shap(shap_config)
    if fullrun_config.components.do_dashboard:
        build_app(dashbord_config)
        app.run(debug=False)


if __name__ == "__main__":
    main()
