import re

import fitz
from diskcache import Cache
from wtpsplit import SaT

sat = SaT("sat-3l")
cache = Cache("cachedir")


@cache.memoize()
def get_text(path):
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += "\n\n" + page.get_text()
    text = text.strip("\n")

    # remove all bracketed text
    text = re.sub(r"\[[^\]]*\]", "", re.sub(r"\([^)]*\)", "", text))

    return text


@cache.memoize()
def get_paragraphs(text):
    results = []
    for paragraph in sat.split(text, do_paragraph_segmentation=True):
        text = re.sub(r"[\n \t\r]+", " ", "".join(paragraph)).strip()
        if text:
            results.append(text)
    
    return results
