from enum import Enum
from typing import List

from pydantic import BaseModel


class MedicalTermType(str, Enum):
    symptom = "TRIỆU_CHỨNG"
    diagnosis = "CHẨN_ĐOÁN"
    lab_test = "XÉT_NGHIỆM"
    test_result = "KẾT_QUẢ_XÉT_NGHIỆM"
    medication = "THUỐC"


class Assertion(str, Enum):
    is_negated = "isNegated"
    is_family = "isFamily"
    is_historical = "isHistorical"


class MedicalTerm(BaseModel):
    text: str
    type: MedicalTermType
    assertions: List[Assertion]
    context: str


def entity_list_json_schema() -> dict:
    """JSON schema for a top-level array of MedicalTerm, used to constrain
    vLLM guided decoding to the exact shape the model was trained on."""
    return {"type": "array", "items": MedicalTerm.model_json_schema()}
