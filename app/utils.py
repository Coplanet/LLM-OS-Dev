import re


def to_label(string: str):
    return re.sub(r"[_]{2,}", "_", string.strip().lower().replace(" ", "_"))
