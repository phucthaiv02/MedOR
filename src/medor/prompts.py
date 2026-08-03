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
