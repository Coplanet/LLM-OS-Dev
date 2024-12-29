import base64
import re


def to_title(text: str) -> str:
    return re.sub(r"[_]+", " ", text.title()).strip()


def audio_text2data(text: str) -> str:
    splits = text.split(";base64,")
    if len(splits) > 1:
        return "".join(splits[1:])
    return text


def audio_encode(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")


def audio2text(audio_bytes: bytes, format: str = "wav") -> str:
    return "data:audio/" + format + ";base64," + audio_encode(audio_bytes)


def text2audio(text: str) -> bytes:
    return base64.b64decode(audio_text2data(text).encode("utf-8"))
