def build_messages(input_text: str, response: str | None = None) -> list[dict]:
    messages = [
        {"role": "user", "content": input_text},
    ]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    return messages
