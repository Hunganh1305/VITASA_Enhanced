"""
CLI demo cho Text Normalization Module.
Chạy: python3 demo_normalizer.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from text_normalization.normalizer import TextNormalizer

SAMPLES = [
    "pyn trâu vcl",
    "sp ngon wa, dt pin trâu, sạc rất nhanhhhhh!!!!",
    "ks này dv tệ vl, nv ko nhiệt tình, phòng dởtệ",
    "mon an o nhahang nay ngonnnnn qá, gia ca hợp lý",
    "nv phục vụ cũg dc, nhưng phòng ms dọn hum qua vẫn còn bẩn",
]


def print_header(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def run_samples(normalizer: TextNormalizer) -> None:
    print_header("CHẠY VỚI SAMPLE CÓ SẴN")
    for i, text in enumerate(SAMPLES, 1):
        print(f"\n[{i}] Input : {text}")
        print(f"    Output: {normalizer(text)}")


def run_user_input(normalizer: TextNormalizer) -> None:
    print_header("TỰ NHẬP CÂU ĐỂ TEST (gõ 'back' để quay lại menu)")
    while True:
        try:
            text = input("\n> Nhập câu cần normalize: ").strip()
        except EOFError:
            break
        if text.lower() in ("back", "exit", "quit"):
            break
        if not text:
            continue
        print(f"  Output: {normalizer(text)}")


def main() -> None:
    normalizer = TextNormalizer()
    while True:
        print_header("TEXT NORMALIZATION — DEMO CLI")
        print("1. Chạy với sample có sẵn")
        print("2. Tự nhập câu để test")
        print("0. Thoát")
        try:
            choice = input("\nChọn (0/1/2): ").strip()
        except EOFError:
            break

        if choice == "1":
            run_samples(normalizer)
        elif choice == "2":
            run_user_input(normalizer)
        elif choice == "0":
            print("Tạm biệt!")
            break
        else:
            print("Lựa chọn không hợp lệ, thử lại.")


if __name__ == "__main__":
    main()
