from pydantic import BaseModel
from typing import Literal

class DomainLabel(str):
    pass

class Classification(BaseModel):
    domain: str
    intent: str
    product_area: str
    risk_level: Literal["low","medium","high"]
    escalation_needed: bool
    urgency: Literal["low","medium","high"]
