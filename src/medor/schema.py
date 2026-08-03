from enum import Enum
from typing import List

from pydantic import BaseModel, Field


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
    text: str = Field(description="Chuỗi trích xuất nguyên văn của thực thể, phải khớp chính xác một đoạn con trong văn bản đầu vào.")
    assertions: List[Assertion] = Field(description="Các assertion áp dụng cho thực thể (phủ định, tiền sử gia đình, tiền sử bệnh nhân); để trống nếu không có.")


def entity_list_json_schema() -> dict:
    """JSON schema for a top-level array of MedicalTerm, used to constrain
    vLLM guided decoding to the exact shape the model was trained on."""
    item_schema = MedicalTerm.model_json_schema()
    defs = item_schema.pop("$defs", None)
    schema = {"type": "array", "items": item_schema}
    if defs:
        schema["$defs"] = defs
    return schema
