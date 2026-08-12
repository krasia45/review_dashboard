"""
[팀원 기여] AI 키워드/인사이트 버그 수정 테스트
------------------------------------------
`_top_keywords()`가 "가장 최근 extract 결과"를 무조건 쓰던 예전 버그를 재현/검증한다:
성공한 AI 추출 뒤에 (일시적 오류 등으로) 규칙 기반 폴백이 나중에 저장되어도,
이미 성공했던 AI 결과를 계속 보여줘야 한다 (폴백이 성공한 결과를 가리면 안 됨).

실행 방법: python -m unittest tests/test_insights.py -v (프로젝트 루트에서)
"""
import os
import sys
import json
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import Database
from src.reporter import _top_keywords
from src.utils import now_str


class TestKeywordInsights(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = Database(os.path.join(self.tmp, "t.db"))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_top_keywords_prefers_ai_over_later_fallback(self):
        ai = {
            "positive_keywords": [{"keyword": "배송 빨라", "count": 10}],
            "negative_keywords": [{"keyword": "배송 지연", "count": 4}],
            "summary": "배송은 빠르다는 칭찬과 지연 불만이 함께 있습니다.",
            "suggestions": ["물류 점검"],
            "topic_breakdown": [{"topic": "배송", "count": 4, "examples": ["지연"]}],
        }
        # 실제 폴백 결과 형태(우리 ai_client._fallback_extract)와 동일하게 "규칙 기반" 문구 포함
        fallback = {
            "positive_keywords": [{"keyword": "좋", "count": 28}],
            "negative_keywords": [{"keyword": "늦", "count": 9}],
            "summary": "총 183건의 리뷰를 규칙 기반으로 요약했습니다.",
            "suggestions": ["API 키 설정 후 재실행하면 더 정교한 AI 인사이트를 받을 수 있습니다."],
            "topic_breakdown": [],
        }
        # AI 추출이 먼저 성공해서 저장되고, 그 다음(예: 일시적 오류로) 폴백이 나중에 저장된 상황
        self.db.insert_extraction("keyword_summary", "감정=전체", json.dumps(ai, ensure_ascii=False), now_str())
        self.db.insert_extraction("keyword_summary", "감정=전체", json.dumps(fallback, ensure_ascii=False), now_str())

        kw = _top_keywords(self.db)
        self.assertEqual(kw["positive"][0]["keyword"], "배송 빨라", "나중에 저장된 폴백이 성공한 AI 결과를 가리면 안 된다")
        self.assertIn("배송은 빠르다", kw["summary"])
        self.assertIn("AI", kw["source"])

    def test_top_keywords_falls_back_when_only_fallback_exists(self):
        fallback = {
            "positive_keywords": [{"keyword": "좋", "count": 5}],
            "negative_keywords": [],
            "summary": "총 20건의 리뷰를 규칙 기반으로 요약했습니다.",
            "suggestions": [],
            "topic_breakdown": [],
        }
        self.db.insert_extraction("keyword_summary", "감정=전체", json.dumps(fallback, ensure_ascii=False), now_str())
        kw = _top_keywords(self.db)
        self.assertEqual(kw["positive"][0]["keyword"], "좋")
        self.assertIn("폴백", kw["source"])

    def test_top_keywords_empty_when_no_extraction_yet(self):
        kw = _top_keywords(self.db)
        self.assertEqual(kw["positive"], [])
        self.assertEqual(kw["source"], "없음")


if __name__ == "__main__":
    unittest.main()
