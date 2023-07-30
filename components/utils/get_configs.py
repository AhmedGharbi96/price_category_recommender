from pathlib import Path
from typing import Optional

from components.data_processing.config import DataProcessingConfig
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
