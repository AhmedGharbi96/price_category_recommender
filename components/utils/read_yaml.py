from pathlib import Path
from typing import Dict, Optional, Union

import yaml


def read_config(cfg_path: Optional[Union[str, Path]] = None) -> Optional[Dict]:
    config = None

    if cfg_path is not None:
        with open(cfg_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    return config
