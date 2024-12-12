import re


def to_title(text: str):
    return re.sub(r"[_]+", " ", text.title()).strip()
