"""
utils.py 단위 테스트
--------------------
실행 방법: python -m unittest tests/test_utils.py -v  (프로젝트 루트에서)
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import normalize_text, normalize_date, normalize_rating, dedup_hash, detect_language, sentiment_grade


class TestUtils(unittest.TestCase):
    def test_normalize_text_collapses_whitespace(self):
        self.assertEqual(normalize_text("  안녕   하세요\n\n"), "안녕 하세요")
        self.assertEqual(normalize_text(None), "")

    def test_normalize_date_various_formats(self):
        self.assertEqual(normalize_date("2026-06-01"), "2026-06-01")
        self.assertEqual(normalize_date("2026/06/01"), "2026-06-01")
        self.assertEqual(normalize_date("2026.06.01"), "2026-06-01")
        self.assertIsNone(normalize_date("이상한날짜"))
        self.assertIsNone(normalize_date(""))

    def test_normalize_rating_range_validation(self):
        self.assertEqual(normalize_rating(5), 5)
        self.assertEqual(normalize_rating("4"), 4)
        self.assertIsNone(normalize_rating(0))     # 범위 밖
        self.assertIsNone(normalize_rating(8))     # 범위 밖 (문제기술 예시의 '8점' 케이스)
        self.assertIsNone(normalize_rating("abc"))  # 파싱 불가

    def test_dedup_hash_is_stable_and_case_insensitive(self):
        h1 = dedup_hash("정말 좋아요", "제품A")
        h2 = dedup_hash("정말 좋아요", "제품A")
        h3 = dedup_hash("정말 좋아요 ", "제품A")  # 공백 차이만 있음 -> normalize_text가 흡수
        h4 = dedup_hash("다른 리뷰", "제품A")
        self.assertEqual(h1, h2)
        self.assertEqual(h1, h3)
        self.assertNotEqual(h1, h4)

    def test_detect_language_ko_en_zh(self):
        self.assertEqual(detect_language("배송이 빨라요"), "ko")
        self.assertEqual(detect_language("Great product, fast shipping"), "en")
        self.assertEqual(detect_language("质量很好，物流也很快"), "zh")

    def test_sentiment_grade_maps_sentiment_to_3_levels(self):
        # 긍정 -> 3점(긍정), 부정 -> 1점(부정), 중립 -> 2점(중립)
        self.assertEqual(sentiment_grade("positive")["score"], 3)
        self.assertEqual(sentiment_grade("positive")["label"], "긍정")
        self.assertEqual(sentiment_grade("negative")["score"], 1)
        self.assertEqual(sentiment_grade("negative")["label"], "부정")
        self.assertEqual(sentiment_grade("neutral")["score"], 2)
        self.assertEqual(sentiment_grade("neutral")["label"], "중립")
        # confidence는 더 이상 등급 계산에 영향을 주지 않는다 (하위호환용으로만 받음)
        self.assertEqual(sentiment_grade("positive", 0.99)["score"], sentiment_grade("positive", 0.1)["score"])
        # sentiment가 없으면(미분석) 중립(2점)으로 안전하게 처리
        self.assertEqual(sentiment_grade(None, None)["score"], 2)


if __name__ == "__main__":
    unittest.main()
