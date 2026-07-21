"""
Unit test cho TextNormalizer.
Chạy: pytest text_normalization/tests/test_normalizer.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from normalizer import TextNormalizer


@pytest.fixture(scope="module")
def normalizer():
    return TextNormalizer()


# ---- Elongation ----

def test_elongation_word(normalizer):
    assert normalizer.reduce_elongation("đẹppppp") == "đẹp"
    assert normalizer.reduce_elongation("nhanhhhhh") == "nhanh"
    assert normalizer.reduce_elongation("okkkk") == "ok"


def test_elongation_punctuation(normalizer):
    assert normalizer.reduce_elongation("hay quá!!!!") == "hay quá!"
    assert normalizer.reduce_elongation("thật sao???") == "thật sao?"


def test_elongation_does_not_break_normal_words(normalizer):
    # Không có ký tự lặp >=3 lần thì giữ nguyên
    assert normalizer.reduce_elongation("đẹp quá") == "đẹp quá"


# ---- Dictionary lookup (single token) ----

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("k", "không"),
        ("ko", "không"),
        ("dc", "được"),
        ("vs", "với"),
        ("pyn", "bạn"),
        ("vcl", "rất"),
        ("trâu", "bền"),
    ],
)
def test_single_token_lookup(normalizer, raw, expected):
    assert normalizer.normalize_token(raw) == expected


def test_unknown_token_unchanged(normalizer):
    assert normalizer.normalize_token("xyzkhongtontai") == "xyzkhongtontai"


# ---- Phrase lookup (multi-word) ----

def test_phrase_lookup(normalizer):
    assert normalizer.replace_phrases("gia ca hop ly") == "giá cả hop ly"
    assert normalizer.replace_phrases("mn hinh dep") == "màn hình dep"


# ---- Full pipeline trên câu thực tế (review mobile/restaurant/hotel) ----

@pytest.mark.parametrize(
    "raw,expected_contains",
    [
        ("pyn trâu vcl", ["bạn", "bền", "rất"]),
        ("sp ngon wa", ["sản phẩm", "ngon", "quá"]),
        ("dt pin trâu, sạc nhanhhhhh!!!!", ["điện thoại", "pin", "bền", "nhanh", "!"]),
        ("ks này dv tệ vl", ["khách sạn", "dịch vụ", "tệ", "rất"]),
        ("nv ko nhiệt tình", ["nhân viên", "không", "nhiệt tình"]),
    ],
)
def test_full_pipeline(normalizer, raw, expected_contains):
    result = normalizer(raw)
    for piece in expected_contains:
        assert piece in result, f"'{piece}' không xuất hiện trong kết quả: '{result}'"


def test_empty_string(normalizer):
    assert normalizer("") == ""
    assert normalizer("   ") == "   "


def test_laughter_removed(normalizer):
    # "kk", "haha" map sang rỗng -> không còn xuất hiện trong câu kết quả
    result = normalizer("ngon quá kk")
    assert "kk" not in result.split()


# ---- keep_expressive_markers (ViGoEmotions-inspired ablation option) ----

def test_keep_expressive_markers_default_false_still_removes():
    # Default (không truyền param) phải giữ đúng hành vi cũ — không phá kết
    # quả/test đã có trước đây.
    normalizer = TextNormalizer()
    assert normalizer.keep_expressive_markers is False
    result = normalizer("ngon quá kk haha")
    assert "kk" not in result.split()
    assert "haha" not in result.split()


def test_keep_expressive_markers_true_preserves_token():
    normalizer = TextNormalizer(keep_expressive_markers=True)
    result = normalizer("ngon quá kk haha")
    assert "kk" in result.split()
    assert "haha" in result.split()


def test_keep_expressive_markers_does_not_affect_other_lookups():
    # Bật keep_expressive_markers không được ảnh hưởng tới các entry khác
    # (chỉ áp dụng cho entry map sang "").
    normalizer = TextNormalizer(keep_expressive_markers=True)
    assert normalizer.normalize_token("pyn") == "bạn"
    assert normalizer.normalize_token("vcl") == "rất"
