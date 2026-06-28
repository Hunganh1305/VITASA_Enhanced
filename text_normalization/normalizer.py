"""
Text Normalization module cho Vietnamese Targeted Aspect Sentiment Analysis (TASA).

Mục đích: chuẩn hóa văn bản mạng xã hội tiếng Việt (teencode, viết tắt,
kéo dài ký tự) trước khi đưa vào PhoBERT/ViSoBERT, nhằm giảm nhiễu do
ngôn ngữ phi chuẩn gây ra (limitation được nêu trong paper ViTASA gốc).

Thiết kế: rule-based, gồm 2 bước chính
    1. Elongation normalization: thu gọn ký tự/dấu câu lặp lại
       (vd "đẹppppp" -> "đẹp", "!!!!" -> "!")
    2. Dictionary lookup: thay thế teencode/viết tắt bằng từ chuẩn,
       dựa trên từ điển trong data/teencode_dict.json

Limitation (xem README.md):
    - Không reorder lại câu (word-level replacement, giữ nguyên thứ tự từ)
    - Không sửa lỗi chính tả ngoài từ điển (không phải spelling correction model)
    - Không phục hồi dấu (không xử lý "khong dau" -> "không dấu")
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

_DEFAULT_DICT_PATH = Path(__file__).parent / "data" / "teencode_dict.json"

# Token = chuỗi ký tự chữ/số liên tiếp (giữ Unicode tiếng Việt) HOẶC 1 ký tự không phải chữ/số/khoảng trắng
_TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)

# Thu gọn run >=3 ký tự giống nhau liên tiếp thành 1 ký tự (chữ/số)
_ELONGATION_WORD_PATTERN = re.compile(r"(\w)\1{2,}", re.UNICODE)

# Thu gọn run >=2 dấu câu lặp (!, ?, ., ,) thành 1 ký tự
_ELONGATION_PUNCT_PATTERN = re.compile(r"([!?.,])\1{1,}")

# Punctuation không cần khoảng trắng phía trước khi join lại câu
_NO_SPACE_BEFORE = set(".,!?;:)]}」")
_NO_SPACE_AFTER = set("([{「")


class TextNormalizer:
    """Rule-based normalizer cho văn bản review tiếng Việt trên mạng xã hội."""

    def __init__(self, dict_path: str | Path = _DEFAULT_DICT_PATH):
        self.dict_path = Path(dict_path)
        self.lexicon, self.phrase_lexicon = self._load_lexicon(self.dict_path)
        # Phrase patterns: nhiều từ -> dài nhất trước, tránh bị match thiếu bởi cụm con
        self._phrase_patterns = [
            (re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", re.IGNORECASE), repl)
            for phrase, repl in sorted(
                self.phrase_lexicon.items(), key=lambda kv: -len(kv[0])
            )
        ]

    @staticmethod
    def _load_lexicon(path: Path) -> tuple[dict[str, str], dict[str, str]]:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        # Bỏ các key comment (bắt đầu bằng "_"), lowercase key để lookup case-insensitive.
        # Tách riêng key 1-từ (tra theo token) và key nhiều-từ (tra theo cụm/regex).
        single_word, phrase = {}, {}
        for k, v in raw.items():
            if k.startswith("_"):
                continue
            key = k.lower()
            if " " in key:
                phrase[key] = v
            else:
                single_word[key] = v
        return single_word, phrase

    def reduce_elongation(self, text: str) -> str:
        """Thu gọn ký tự/dấu câu bị lặp lại do gõ nhấn mạnh cảm xúc."""
        text = _ELONGATION_WORD_PATTERN.sub(r"\1", text)
        text = _ELONGATION_PUNCT_PATTERN.sub(r"\1", text)
        return text

    def tokenize(self, text: str) -> list[str]:
        return _TOKEN_PATTERN.findall(text)

    def normalize_token(self, token: str) -> str:
        """Tra từ điển cho 1 token; trả về nguyên bản nếu không có trong từ điển."""
        key = token.lower()
        if key in self.lexicon:
            return self.lexicon[key]
        return token

    def replace_phrases(self, text: str) -> str:
        """Thay các cụm nhiều-từ trong từ điển trước khi tokenize từng từ
        (vd "gia ca" -> "giá cả", "mn hinh" -> "màn hình")."""
        for pattern, repl in self._phrase_patterns:
            text = pattern.sub(repl, text)
        return text

    def detokenize(self, tokens: list[str]) -> str:
        """Ghép token lại thành câu, xử lý khoảng trắng quanh dấu câu."""
        parts: list[str] = []
        for tok in tokens:
            if not tok:
                continue
            if parts and not (
                tok in _NO_SPACE_BEFORE or parts[-1][-1:] in _NO_SPACE_AFTER
            ):
                parts.append(" ")
            parts.append(tok)
        return "".join(parts)

    def normalize(self, text: str) -> str:
        """Pipeline đầy đủ: elongation -> phrase lookup -> tokenize -> dictionary lookup -> detokenize."""
        if not text or not text.strip():
            return text
        text = unicodedata.normalize("NFC", text)
        text = self.reduce_elongation(text)
        text = self.replace_phrases(text)
        tokens = self.tokenize(text)
        normalized_tokens = [self.normalize_token(tok) for tok in tokens]
        return self.detokenize(normalized_tokens)

    def __call__(self, text: str) -> str:
        return self.normalize(text)


if __name__ == "__main__":
    normalizer = TextNormalizer()
    examples = [
        "pyn trâu vcl",
        "sp ngon wa, dt pin trâu, sạc rất nhanhhhhh!!!!",
        "ks này dv tệ vl, nv ko nhiệt tình, phòng dởtệ",
        "mon an o nhahang nay ngonnnnn qá, gia ca hợp lý",
    ]
    for ex in examples:
        print(f"{ex!r}\n  -> {normalizer(ex)!r}\n")
