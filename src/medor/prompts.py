INSTRUCTION = (
    "Trích xuất các thực thể y khoa từ văn bản và các thông tin liên quan. Trả về đúng một mảng JSON theo schema, không kèm giải thích."
)


def build_messages(input_text: str, response: str | None = None) -> list[dict]:
    messages = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": input_text},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages


def build_prompt_completion(input_text: str, response: str) -> tuple[list[dict], list[dict]]:
    """Split a training example into TRL's prompt-completion message format, so
    SFTTrainer computes loss only on `response` (the JSON) and not on the system
    instruction or the input document."""
    prompt = [
        {"role": "system", "content": INSTRUCTION},
        {"role": "user", "content": input_text},
    ]
    completion = [{"role": "assistant", "content": response}]
    return prompt, completion


TYPE_INSTRUCTION = (
    "Bạn là chuyên gia y khoa. Bạn sẽ nhận một đoạn văn bản gốc trong đó cụm từ cần "
    "phân loại được đánh dấu bằng cặp ngoặc «». Hãy xác định cụm từ đó thuộc loại thực "
    "thể y khoa nào trong các loại sau:\n"
    "- TRIỆU_CHỨNG: triệu chứng, dấu hiệu lâm sàng người bệnh gặp phải.\n"
    "- CHẨN_ĐOÁN: tên bệnh, chẩn đoán y khoa.\n"
    "- XÉT_NGHIỆM: tên xét nghiệm, thủ thuật, chỉ định cận lâm sàng.\n"
    "- KẾT_QUẢ_XÉT_NGHIỆM: kết quả hoặc chỉ số cụ thể của một xét nghiệm.\n"
    "- THUỐC: tên thuốc, hoạt chất điều trị.\n"
    "Chỉ trả lời đúng một trong các nhãn trên, không kèm giải thích."
)


def build_type_choice_messages(snippet: str, text: str) -> list[dict]:
    """Build the messages for the type-classification step: `snippet` is a
    window of the original document with the entity span marked by «», so
    the model can use surrounding context to disambiguate (the same word can
    be a symptom in one sentence and a diagnosis in another)."""
    user = f'Văn bản: {snippet}\nCụm từ cần phân loại: "{text}"\nLoại thực thể:'
    return [
        {"role": "system", "content": TYPE_INSTRUCTION},
        {"role": "user", "content": user},
    ]
