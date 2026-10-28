import os
import unicodedata

if not os.getenv("FTLANG_CACHE"):
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    root_dir = os.path.dirname(current_dir)
    ftlang_cache_dir = os.path.join(root_dir, 'resources', 'fasttext-langdetect')
    os.environ["FTLANG_CACHE"] = str(ftlang_cache_dir)
    # print(os.getenv("FTLANG_CACHE"))

from fast_langdetect import detect_language


def remove_invalid_surrogates(text):
    # 移除无效的 UTF-16 代理对
    return ''.join(c for c in text if not (0xD800 <= ord(c) <= 0xDFFF))


def detect_lang(text: str) -> str:

    if len(text) == 0:
        return ""

    text = text.replace("\n", "")
    text = remove_invalid_surrogates(text)

    # print(text)
    lang_upper = ""
    try:
        lang_upper = detect_language(text)
    except Exception:  # noqa: BLE001
        html_no_ctrl_chars = ''.join([l for l in text if unicodedata.category(l)[0] not in ['C', ]])
        try:
            lang_upper = detect_language(html_no_ctrl_chars)
        except Exception:  # noqa: BLE001
            lang_upper = "EN"  # 极端文本(如纯符号/SMILES)无法检测时默认英文

    try:
        lang = lang_upper.lower()
    except:  # noqa: BLE001
        lang = ""
    return lang


if __name__ == '__main__':
    print(os.getenv("FTLANG_CACHE"))
    print(detect_lang("This is a test."))
    print(detect_lang("<html>This is a test</html>"))
    print(detect_lang("这个是中文测试。"))
    print(detect_lang("<html>这个是中文测试。</html>"))
    print(detect_lang("〖\ud835\udc46\ud835〗这是个包含utf-16的中文测试"))