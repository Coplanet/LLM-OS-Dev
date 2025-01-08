import base64
import re


def to_title(text: str) -> str:
    return re.sub(r"[_]+", " ", text.title()).strip()


def binary_text2data(text: str) -> str:
    splits = text.split(";base64,")
    if len(splits) > 1:
        return "".join(splits[1:])
    return text


def binary_encode(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")


def binary2text(audio_bytes: bytes, format: str) -> str:
    return "data:" + format + ";base64," + binary_encode(audio_bytes)


def text2binary(text: str) -> bytes:
    return base64.b64decode(binary_text2data(text).encode("utf-8"))
