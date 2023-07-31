from pathlib import Path
from typing import Optional

from components.data_processing.config import DataProcessingConfig
from components.fullrun.config import FullRunConfig
from components.inference.config import InferenceConfig
from components.shap.config import ShapConfig
from components.training.config import TrainingConfig
from components.utils.constants import BASE_CONFIG_FOLDER
from components.utils.read_yaml import read_config


def load_data_processing_config(
    base_config_path: Optional[Path] = None,
) -> DataProcessingConfig:
    if not base_config_path:
        base_config_path = BASE_CONFIG_FOLDER
    data_processing_config = DataProcessingConfig(
        **read_config(base_config_path / "data_processing" / "config.yaml")
    )
    return data_processing_config


def load_training_config(
    base_config_path: Optional[Path] = None,
) -> TrainingConfig:
    if not base_config_path:
        base_config_path = BASE_CONFIG_FOLDER
    training_config = TrainingConfig(
        **read_config(base_config_path / "training" / "config.yaml")
    )
    return training_config


def load_fullrun_config(
    base_config_path: Optional[Path] = None,
) -> FullRunConfig:
    if not base_config_path:
        base_config_path = BASE_CONFIG_FOLDER
    fullrun_config = FullRunConfig(
        **read_config(base_config_path / "fullrun" / "config.yaml")
    )
    return fullrun_config


def load_inference_config(
    base_config_path: Optional[Path] = None,
) -> InferenceConfig:
    if not base_config_path:
        base_config_path = BASE_CONFIG_FOLDER
    inference_config = InferenceConfig(
        **read_config(base_config_path / "inference" / "config.yaml")
    )
    return inference_config


def load_shap_config(
    base_config_path: Optional[Path] = None,
) -> ShapConfig:
    if not base_config_path:
        base_config_path = BASE_CONFIG_FOLDER
    shap_config = ShapConfig(**read_config(base_config_path / "shap" / "config.yaml"))
    return shap_config
