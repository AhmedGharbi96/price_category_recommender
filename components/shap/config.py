from datetime import datetime
from typing import Optional

from pydantic import validator

from components.utils.constants import DATE_FORMAT
from components.utils.models import CustomBaseModel


class ShapConfig(CustomBaseModel):
    experiment_id: str
    random_state: Optional[int] = 1234
    n_samples: int

    @validator("experiment_id", pre=True, always=True, allow_reuse=True)
    def set_experiment_id(cls, exp_id: str) -> str:
        if not exp_id:
            exp_id = datetime.utcnow().strftime(DATE_FORMAT)
            return exp_id
        return exp_id
