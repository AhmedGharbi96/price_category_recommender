from datetime import datetime
from typing import Optional

from pydantic import validator

from components.utils.constants import DATE_FORMAT
from components.utils.models import CustomBaseModel


class SmoteConfig(CustomBaseModel):
    use_smote: bool
    min_n_samples: Optional[int] = 5000
    k_neighbors: Optional[int] = 5


class KdeConfig(CustomBaseModel):
    # available options: 'cv_ml', 'cv_ls', 'normal_reference'
    bandwidth_estimate_method: Optional[str] = "normal_reference"


class DataProcessingConfig(CustomBaseModel):
    experiment_id: str
    random_state: Optional[int] = 1234
    raw_data_file_name: str
    use_validation_set: bool
    test_set_n_samples: Optional[int] = 10000
    validation_set_ratio: Optional[float] = 0.1
    smote_config: SmoteConfig
    kde_config: KdeConfig

    @validator("experiment_id", pre=True, always=True, allow_reuse=True)
    def set_experiment_id(cls, exp_id: str) -> str:
        if not exp_id:
            exp_id = datetime.utcnow().strftime(DATE_FORMAT)
            return exp_id
        return exp_id
