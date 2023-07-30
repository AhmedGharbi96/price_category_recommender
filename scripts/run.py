from components.utils.get_configs import (
    load_data_processing_config,
    load_fullrun_config,
    load_inference_config,
    load_training_config,
)
from scripts.data_processing import run_data_processing
from scripts.inference import run_inference
from scripts.train import run_model_training


def main():
    fullrun_config = load_fullrun_config()
    processing_config = load_data_processing_config()
    training_config = load_training_config()
    inference_config = load_inference_config()
    processing_config.experiment_id = fullrun_config.experiment_id
    training_config.experiment_id = fullrun_config.experiment_id
    inference_config.experiment_id = fullrun_config.experiment_id

    if fullrun_config.components.do_processing:
        run_data_processing(processing_config)
    if fullrun_config.components.do_training:
        run_model_training(training_config)
    if fullrun_config.components.do_inference:
        run_inference(inference_config)


if __name__ == "__main__":
    main()
