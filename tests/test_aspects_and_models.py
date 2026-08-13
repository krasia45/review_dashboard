"""
[사용자 요청 추가 기능] 테스트
-----------------------------
1) 측면(배송/상품/응대) 만족도 분류 + 5점 만점 수치화
2) 멀티 프로바이더 AI 클라이언트(Claude/OpenAI/Gemini/fallback)가 전부 안전하게 초기화되는지
3) 모델별 비교(model_runs) — 스냅샷 저장 후 두 스냅샷을 비교했을 때 숫자가 말이 되는지

실행 방법: python -m unittest tests/test_aspects_and_models.py -v (프로젝트 루트에서)
"""
import os
import sys
import shutil
import logging
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import Database
from src.ai_client import AIClient
from src.logger_setup import setup_logger
from src import aspects, model_runs
from src import ingest, cleaner, analyzer


class TestAspectSatisfaction(unittest.TestCase):
    """측면 만족도 분류 + 5점 척도 변환 (네트워크/DB 불필요, 순수 함수 테스트)."""

    def test_aspect_score_mapping(self):
        self.assertEqual(aspects.aspect_score("positive"), 5)
        self.assertEqual(aspects.aspect_score("neutral"), 3)
        self.assertEqual(aspects.aspect_score("negative"), 1)
        self.assertIsNone(aspects.aspect_score("not_mentioned"))

    def test_average_aspect_scores_excludes_not_mentioned(self):
        # not_mentioned는 0점이 아니라 "집계에서 제외"되어야 한다 (0점 취급하면 평균이 왜곡됨).
        data = [
            {"product": "positive", "delivery": "not_mentioned", "service": "not_mentioned"},
            {"product": "negative", "delivery": "not_mentioned", "service": "not_mentioned"},
        ]
        result = aspects.average_aspect_scores(data)
        self.assertEqual(result["product"], 3.0)  # (5+1)/2
        self.assertIsNone(result["delivery"], "언급이 전혀 없으면 None이어야 한다 (0점 아님)")

    def test_infer_aspects_from_text_detects_delivery_and_service_cues(self):
        result = aspects.infer_aspects_from_text("배송이 빨라서 좋았는데 고객센터 응대는 별로였어요")
        self.assertIn(result["delivery"], ("positive", "neutral"))
        self.assertIn("product", result)
        self.assertIn("service", result)

    def test_aspects_json_roundtrip(self):
        original = {"product": "positive", "delivery": "negative", "service": "not_mentioned"}
        as_json = aspects.aspects_to_json(original)
        restored = aspects.aspects_from_json(as_json)
        self.assertEqual(restored, original)

    def test_aspects_from_json_handles_empty_or_malformed(self):
        self.assertEqual(aspects.aspects_from_json(None), aspects.empty_aspects())
        self.assertEqual(aspects.aspects_from_json("not valid json"), aspects.empty_aspects())


class TestMultiProviderAIClient(unittest.TestCase):
    """3개 provider(anthropic/openai/gemini) + fallback이 전부 예외 없이 초기화되는지 검증한다."""

    def setUp(self):
        self.logger = logging.getLogger("test_multiprovider")
        self.logger.handlers = []
        self.logger.addHandler(logging.NullHandler())

    def test_all_providers_initialize_without_crashing(self):
        for provider in ("anthropic", "openai", "gemini", "fallback", "무언가이상한값"):
            config = {"ai": {"provider": provider}}
            client = AIClient(config, self.logger)
            self.assertIn(client.provider, ("anthropic", "openai", "gemini", "fallback"))

    def test_gemini_requires_its_own_key_env(self):
        config = {"ai": {"provider": "gemini"}}
        client = AIClient(config, self.logger)
        self.assertFalse(client.available, "GEMINI_API_KEY가 없으면 비활성 상태여야 한다")
        self.assertEqual(client.gemini_api_key_env, "GEMINI_API_KEY")

    def test_model_id_fits_provider_catches_leftover_model_id(self):
        # 회귀 테스트: provider를 바꿨는데 모델 id는 예전 것이 그대로 남아있는
        # 실수를 잡아낼 수 있어야 한다.
        from src.ai_client import model_id_fits_provider
        self.assertFalse(model_id_fits_provider("openai", "claude-haiku-4-5-20251001"))
        self.assertTrue(model_id_fits_provider("anthropic", "claude-haiku-4-5-20251001"))
        self.assertTrue(model_id_fits_provider("gemini", "gemini-2.0-flash"))
        self.assertFalse(model_id_fits_provider("gemini", "gpt-4o"))

    def test_fallback_sentiment_includes_aspects(self):
        config = {"ai": {"provider": "fallback"}}
        client = AIClient(config, self.logger)
        result = client.analyze_sentiment("배송이 빨라서 좋았어요")
        self.assertIn("aspects", result)
        self.assertIn("delivery", result["aspects"])


class TestModelComparison(unittest.TestCase):
    """모델별 비교(model_runs) — 스냅샷 저장 -> 두 스냅샷 비교까지 실제 DB로 검증한다."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.config = {
            "ai": {"provider": "fallback"},
            "dedup_policy": "skip",
            "cleaning": {"min_review_length": 5},
            "storage": {"db_path": os.path.join(self.tmp_dir, "test.db")},
            "logging": {"log_dir": os.path.join(self.tmp_dir, "logs"), "level": "INFO"},
        }
        self.logger = setup_logger(self.config)
        self.db = Database(self.config["storage"]["db_path"])
        self.ai_client = AIClient(self.config, self.logger)
        csv_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_data", "reviews_sample.csv",
        )
        ingest.import_file(self.db, self.config, self.logger, csv_path)
        cleaner.clean_all(self.db, self.config, self.logger)

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_model_run_creates_snapshot_with_correct_counts(self):
        analyzer.analyze_reviews(self.db, self.ai_client, self.logger, target="unanalyzed", limit=5, show_progress=False)
        run_id = self.db.save_model_run("fallback", "규칙 기반", "테스트 스냅샷", "2026-01-01")
        run = self.db.get_model_run(run_id)
        self.assertEqual(run["analyzed_count"], 5)
        self.assertGreater(run["review_count"], 5)

    def test_compare_model_runs_reports_agreement_and_disagreements(self):
        analyzer.analyze_reviews(self.db, self.ai_client, self.logger, target="unanalyzed", limit=5, show_progress=False)
        run_a = self.db.save_model_run("fallback", "규칙 기반 A", "런 A", "2026-01-01")
        # 리뷰 5건을 더 분석해서(총 10건) 두 번째 스냅샷을 만든다 -> 두 런은 공통 5건은
        # 100% 일치(같은 폴백 로직이 같은 입력에 항상 같은 결과를 내므로)해야 하고,
        # 나머지 5건은 A에는 없고 B에만 있어서 불일치로 잡혀야 한다.
        analyzer.analyze_reviews(self.db, self.ai_client, self.logger, target="unanalyzed", limit=5, show_progress=False)
        run_b = self.db.save_model_run("fallback", "규칙 기반 B", "런 B", "2026-01-02")

        result = self.db.compare_model_runs(run_a, run_b)
        self.assertEqual(result["compared_count"], 10)
        # A에서 분석 안 됐던 5건은 B와 다르므로 불일치로 집계되어야 한다
        self.assertGreaterEqual(result["disagreement_total"], 5)
        self.assertIsNotNone(result["agreement_rate"])

    def test_compare_model_runs_raises_for_unknown_run_id(self):
        with self.assertRaises(ValueError):
            self.db.compare_model_runs(9999, 8888)

    def test_seed_snapshot_only_created_once(self):
        analyzer.analyze_reviews(self.db, self.ai_client, self.logger, target="unanalyzed", limit=3, show_progress=False)
        first = model_runs.ensure_seed_snapshot(self.db, self.config, self.logger)
        second = model_runs.ensure_seed_snapshot(self.db, self.config, self.logger)
        self.assertIsNotNone(first)
        self.assertIsNone(second, "이미 스냅샷이 있으면 시드를 또 만들면 안 된다")


if __name__ == "__main__":
    unittest.main()
