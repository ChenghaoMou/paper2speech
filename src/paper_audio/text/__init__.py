from diskcache import Cache
from dotenv import load_dotenv

load_dotenv()
cache = Cache("cachedir")

@cache.memoize()
def get_text(path) -> list[str]:
    import fitz

    doc = fitz.open(path)
    results = []
    for page in doc:
        for _, _, _, _, content, _, _ in page.get_text("blocks"):
            results.append(content)
    return results

if __name__ == "__main__":

    content = get_text("/Users/chenghao/Zotero/storage/NAG42KRC/Gerstgrasser et al. - 2024 - Is Model Collapse Inevitable Breaking the Curse of Recursion by Accumulating Real and Synthetic Dat.pdf")
    for block in content:
        print(block)