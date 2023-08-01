from datetime import datetime

from pydantic import validator

from components.utils.constants import DATE_FORMAT
from components.utils.models import CustomBaseModel


class DashboardConfig(CustomBaseModel):
    experiment_id: str

    @validator("experiment_id", pre=True, always=True, allow_reuse=True)
    def set_experiment_id(cls, exp_id: str) -> str:
        if not exp_id:
            exp_id = datetime.utcnow().strftime(DATE_FORMAT)
            return exp_id
        return exp_id
