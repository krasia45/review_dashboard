"""
AI API 클라이언트 모듈
----------------------
리뷰 1) 감정 분석  2) 키워드/요약/개선제안 추출 을 수행한다.

지원 provider (config.json의 ai.provider):
  - anthropic (기본값) : Anthropic Claude 공식 REST API
  - openai             : OpenAI 공식 API (OpenAI 호환 /v1/chat/completions)
  - gemini             : Google Gemini (Generative Language API)
  - fallback           : 규칙 기반만 사용 (API 호출 없음)

- API 키/엔드포인트는 코드에 하드코딩하지 않고 config.json + 환경변수에서 읽는다.
- provider=fallback 이거나 해당 provider의 키가 없으면 규칙 기반 폴백으로 동작한다.
- 키가 있는데 호출이 실패하면(크레딧 부족 등) 감정분석은 예외를 던져
  호출부(analyzer)가 해당 건을 스킵한다 (과제 요구사항 "API 실패 시 로깅 후 스킵").
"""
import os
import re
import json
import requests
from typing import Optional

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
GEMINI_DEFAULT_BASE = "https://generativelanguage.googleapis.com/v1beta"

POSITIVE_HINTS = ["좋", "만족", "빠르", "편해", "훌륭", "추천", "예뻐", "친절", "가성비", "great", "good", "happy", "love",
                  # 중국어 긍정 키워드 (다국어 지원 보너스 — 폴백에서도 중국어 리뷰를 구분할 수 있도록)
                  "满意", "好", "快", "舒适", "舒服", "实用", "结实", "亲切", "合理", "喜欢"]
NEGATIVE_HINTS = ["불량", "늦", "실망", "안돼", "안됨", "불편", "느리", "나빠", "최악", "환불", "반품",
                  "disappoint", "defective", "bad", "slow", "broken",
                  # 중국어 부정 키워드
                  "失望", "慢", "坏", "不方便", "中断", "损坏", "划痕", "退款"]


def model_id_fits_provider(provider: str, model: Optional[str]) -> bool:
    """[품질 안전장치] provider를 바꿨는데 모델 id는 이전 provider 것 그대로 남아있는
    실수를 조기에 잡아낸다 (예: provider=openai인데 sentiment_model=claude-haiku...)."""
    m = (model or "").strip().lower()
    p = (provider or "").strip().lower()
    if not m:
        return False
    if p == "anthropic":
        return m.startswith("claude")
    if p == "openai":
        return m.startswith("gpt-") or m.startswith(("o1", "o3", "o4", "chatgpt"))
    if p == "gemini":
        return "gemini" in m
    return True

try:
    from .aspects import ASPECT_IDS, infer_aspects_from_text, normalize_aspects
except ImportError:  # pragma: no cover - 단독 스크립트 실행 대비
    from aspects import ASPECT_IDS, infer_aspects_from_text, normalize_aspects


class AIClient:
    def __init__(self, config: dict, logger):
        self.logger = logger
        ai_cfg = config.get("ai", {})

        self.provider = (ai_cfg.get("provider") or "anthropic").strip().lower()

        self.api_key_env = ai_cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        self.api_key = os.environ.get(self.api_key_env, "").strip()
        self.openai_api_key_env = ai_cfg.get("openai_api_key_env", "OPENAI_API_KEY")
        self.openai_api_key = os.environ.get(self.openai_api_key_env, "").strip()
        self.gemini_api_key_env = ai_cfg.get("gemini_api_key_env", "GEMINI_API_KEY")
        self.gemini_api_key = os.environ.get(self.gemini_api_key_env, "").strip()

        self.sentiment_model = ai_cfg.get("sentiment_model", "claude-haiku-4-5-20251001")
        self.extract_model = ai_cfg.get("extract_model", "claude-sonnet-5")
        self.max_tokens = ai_cfg.get("max_tokens", 1024)
        self.extract_max_tokens = ai_cfg.get("extract_max_tokens", max(self.max_tokens * 4, 4096))
        self.timeout = ai_cfg.get("request_timeout_sec", 30)
        self.extract_timeout = ai_cfg.get("extract_timeout_sec", max(self.timeout * 3, 120))

        self.openai_base_url = (ai_cfg.get("openai_base_url") or OPENAI_DEFAULT_BASE).rstrip("/")
        self.gemini_base_url = (ai_cfg.get("gemini_base_url") or GEMINI_DEFAULT_BASE).rstrip("/")

        if self.provider == "fallback":
            self.available = False
            self.logger.warning("AI provider=fallback — 규칙 기반 분석만 사용합니다.")
        elif self.provider == "openai":
            self.available = bool(self.openai_api_key)
            if not self.available:
                self.logger.warning(
                    f"{self.openai_api_key_env} 환경변수가 설정되지 않았습니다. "
                    "OpenAI 호출 대신 규칙 기반 폴백 분석기를 사용합니다. "
                    f"OpenAI를 쓰려면 .env 에 {self.openai_api_key_env}=... 를 넣으세요."
                )
            else:
                self.logger.info(f"AI provider=openai base_url={self.openai_base_url} model={self.sentiment_model}")
        elif self.provider == "gemini":
            self.available = bool(self.gemini_api_key)
            if not self.available:
                self.logger.warning(
                    f"{self.gemini_api_key_env} 환경변수가 설정되지 않았습니다. "
                    "Gemini 호출 대신 규칙 기반 폴백 분석기를 사용합니다. "
                    f"Gemini를 쓰려면 .env 에 {self.gemini_api_key_env}=... 를 넣으세요."
                )
            else:
                self.logger.info(f"AI provider=gemini base_url={self.gemini_base_url} model={self.sentiment_model}")
        else:
            self.provider = "anthropic"
            self.available = bool(self.api_key)
            if not self.available:
                self.logger.warning(
                    f"{self.api_key_env} 환경변수가 설정되지 않았습니다. "
                    "실제 AI 호출 대신 규칙 기반 폴백 분석기를 사용합니다. "
                    f"실제 AI 분석을 사용하려면: export {self.api_key_env}=sk-ant-xxxx "
                    "(또는 config.json 의 ai.provider 를 openai/gemini/fallback 으로 바꾸세요)."
                )

        # [품질 안전장치] provider는 바꿨는데 모델 id는 이전 provider 것 그대로 남아있는
        # 실수(예: provider=openai인데 sentiment_model=claude-haiku...)를 조기에 경고한다.
        if self.available:
            for label, model_id in (("sentiment_model", self.sentiment_model), ("extract_model", self.extract_model)):
                if not model_id_fits_provider(self.provider, model_id):
                    self.logger.warning(
                        f"ai.{label}='{model_id}' 가 provider='{self.provider}' 와 맞지 않아 보입니다 "
                        "(엔진을 바꿨는데 모델 id는 이전 것 그대로 남아있을 수 있습니다 - config.json을 확인하세요)."
                    )

    # ---------------- 내부: provider별 실제 호출 dispatch ----------------
    def _call_llm(self, model: str, system: str, user_prompt: str,
                  max_tokens: Optional[int] = None, timeout: Optional[float] = None) -> Optional[str]:
        if not self.available:
            return None
        if self.provider == "openai":
            return self._call_openai(model, system, user_prompt, max_tokens=max_tokens, timeout=timeout)
        if self.provider == "gemini":
            return self._call_gemini(model, system, user_prompt, max_tokens=max_tokens, timeout=timeout)
        return self._call_claude(model, system, user_prompt, max_tokens=max_tokens, timeout=timeout)

    def _call_claude(self, model: str, system: str, user_prompt: str,
                      max_tokens: Optional[int] = None, timeout: Optional[float] = None) -> Optional[str]:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        payload = {
            "model": model,
            "max_tokens": max_tokens or self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        try:
            resp = requests.post(ANTHROPIC_URL, headers=headers, json=payload, timeout=timeout or self.timeout)
            if resp.status_code != 200:
                self.logger.error(f"AI API 호출 실패 (status={resp.status_code}): {resp.text[:200]}")
                return None
            data = resp.json()
            if data.get("stop_reason") == "max_tokens":
                self.logger.warning(
                    f"AI 응답이 max_tokens({max_tokens or self.max_tokens}) 제한에 걸려 중간에 잘렸습니다. "
                    "JSON 파싱이 실패할 수 있습니다 (config.json의 max_tokens/extract_max_tokens를 늘려보세요)."
                )
            text_blocks = [b["text"] for b in data.get("content", []) if b.get("type") == "text"]
            return "\n".join(text_blocks).strip()
        except requests.exceptions.Timeout:
            used_timeout = timeout or self.timeout
            self.logger.error(
                f"AI API 요청이 {used_timeout}초 안에 끝나지 않아 타임아웃되었습니다 "
                "(요청이 크거나 서버가 느린 경우 흔함 - config.json의 "
                "request_timeout_sec/extract_timeout_sec를 늘려보세요)."
            )
            return None
        except requests.RequestException as e:
            self.logger.error(f"AI API 요청 중 네트워크 오류: {e}")
            return None

    def _call_openai(self, model: str, system: str, user_prompt: str,
                      max_tokens: Optional[int] = None, timeout: Optional[float] = None) -> Optional[str]:
        """OpenAI 공식 chat completions API."""
        if not self.openai_api_key:
            self.logger.error(f"openai 호출에 {self.openai_api_key_env} 가 필요합니다.")
            return None
        headers = {"content-type": "application/json", "authorization": f"Bearer {self.openai_api_key}"}
        payload = {
            "model": model,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        }
        url = f"{self.openai_base_url}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout or self.timeout)
            if resp.status_code != 200:
                self.logger.error(f"openai API 호출 실패 (status={resp.status_code}): {resp.text[:200]}")
                return None
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            finish_reason = choice.get("finish_reason")
            if finish_reason == "length":
                self.logger.warning(
                    f"AI 응답이 max_tokens({max_tokens or self.max_tokens}) 제한에 걸려 중간에 잘렸습니다. "
                    "JSON 파싱이 실패할 수 있습니다 (config.json의 max_tokens/extract_max_tokens를 늘려보세요)."
                )
            msg = choice.get("message") or {}
            content = msg.get("content")
            return str(content).strip() if content else None
        except requests.exceptions.Timeout:
            used_timeout = timeout or self.timeout
            self.logger.error(
                f"openai API 요청이 {used_timeout}초 안에 끝나지 않아 타임아웃되었습니다 "
                "(config.json의 request_timeout_sec/extract_timeout_sec를 늘려보세요)."
            )
            return None
        except requests.RequestException as e:
            self.logger.error(f"openai API 요청 중 네트워크 오류: {e}")
            return None

    def _call_gemini(self, model: str, system: str, user_prompt: str,
                      max_tokens: Optional[int] = None, timeout: Optional[float] = None) -> Optional[str]:
        """Google Generative Language generateContent (Gemini 전용 API 포맷)."""
        key = self.gemini_api_key or os.environ.get(self.gemini_api_key_env, "").strip()
        if not key:
            self.logger.error(f"Gemini 호출에 {self.gemini_api_key_env} 가 필요합니다.")
            return None
        used_max = max_tokens or self.max_tokens
        used_timeout = timeout or self.timeout
        model_id = (model or "").strip()
        if model_id.startswith("models/"):
            model_id = model_id[len("models/"):]
        url = f"{self.gemini_base_url}/models/{model_id}:generateContent"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": used_max},
        }
        try:
            resp = requests.post(url, params={"key": key}, headers={"content-type": "application/json"},
                                  json=payload, timeout=used_timeout)
            if resp.status_code != 200:
                self.logger.error(f"Gemini API 호출 실패 (status={resp.status_code}): {resp.text[:200]}")
                return None
            data = resp.json()
            candidates = data.get("candidates") or []
            if not candidates:
                self.logger.error(f"Gemini 응답에 candidates 가 없습니다: {str(data)[:200]}")
                return None
            finish_reason = (candidates[0] or {}).get("finishReason")
            if finish_reason == "MAX_TOKENS":
                self.logger.warning(
                    f"AI 응답이 max_tokens({used_max}) 제한에 걸려 중간에 잘렸습니다. "
                    "JSON 파싱이 실패할 수 있습니다 (config.json의 max_tokens/extract_max_tokens를 늘려보세요)."
                )
            parts = (((candidates[0] or {}).get("content") or {}).get("parts")) or []
            texts = [p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")]
            text = "\n".join(texts).strip()
            return text or None
        except requests.exceptions.Timeout:
            self.logger.error(f"Gemini API 요청이 {used_timeout}초 안에 끝나지 않아 타임아웃되었습니다.")
            return None
        except requests.RequestException as e:
            self.logger.error(f"Gemini API 요청 중 네트워크 오류: {e}")
            return None

    @staticmethod
    def _extract_json(text: str) -> Optional[dict]:
        if not text:
            return None
        text = text.strip()
        text = re.sub(r"^```json\s*|^```\s*|```$", "", text, flags=re.MULTILINE).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    return None
        return None

    # ---------------- 감정 분석 ----------------
    def analyze_sentiment(self, review_text: str, language: str = "ko") -> dict:
        """리뷰 1건에 대해 전체 감정 + 측면(상품/배송/응대) 만족도를 반환.

        반환: {
          'sentiment': 'positive|negative|neutral',
          'confidence': 0.0~1.0,
          'aspects': {'product'|'delivery'|'service': 'positive|negative|neutral|not_mentioned'}
        }
        """
        if not self.available:
            return self._fallback_sentiment(review_text)

        system = (
            "너는 전자상거래 고객 리뷰 감정 분석 전문가다. 주어진 리뷰(한국어 또는 영어)를 읽고 "
            "(1) 전체 감정을 positive, negative, neutral 중 하나로 분류하고 0.0~1.0 신뢰도를 매기며, "
            "(2) 만족도 측면별로 감정을 분류하라: product(상품), delivery(배송), service(응대/CS). "
            "해당 측면이 리뷰에 언급되지 않으면 not_mentioned. "
            "반드시 다른 설명 없이 JSON만 출력하라: "
            '{"sentiment":"positive|negative|neutral","confidence":0.0,'
            '"aspects":{"product":"positive|negative|neutral|not_mentioned",'
            '"delivery":"positive|negative|neutral|not_mentioned",'
            '"service":"positive|negative|neutral|not_mentioned"}}'
        )
        result = self._call_llm(self.sentiment_model, system, f"리뷰: {review_text}")
        parsed = self._extract_json(result) if result else None
        if parsed and parsed.get("sentiment") in ("positive", "negative", "neutral"):
            confidence = parsed.get("confidence", 0.75)
            try:
                confidence = float(confidence)
            except (TypeError, ValueError):
                confidence = 0.75
            aspects = normalize_aspects(parsed.get("aspects"))
            if all(aspects[a] == "not_mentioned" for a in ASPECT_IDS):
                aspects = infer_aspects_from_text(review_text)
            return {
                "sentiment": parsed["sentiment"],
                "confidence": round(max(0.0, min(1.0, confidence)), 2),
                "aspects": aspects,
            }

        if result and not parsed:
            self.logger.error(f"AI 응답을 JSON으로 파싱하지 못했습니다. 원문 일부: {result[:200]!r}")

        raise RuntimeError(
            f"AI 감정분석 API 호출에 실패했습니다 (provider={self.provider} — "
            "크레딧 부족/인증오류/네트워크 오류 등 - logs/app.log 확인)"
        )

    @staticmethod
    def _fallback_sentiment(text: str) -> dict:
        t = (text or "").lower()
        pos = sum(1 for w in POSITIVE_HINTS if w.lower() in t)
        neg = sum(1 for w in NEGATIVE_HINTS if w.lower() in t)
        if pos > neg:
            overall = {"sentiment": "positive", "confidence": round(min(0.6 + 0.1 * (pos - neg), 0.95), 2)}
        elif neg > pos:
            overall = {"sentiment": "negative", "confidence": round(min(0.6 + 0.1 * (neg - pos), 0.95), 2)}
        else:
            overall = {"sentiment": "neutral", "confidence": 0.55}
        overall["aspects"] = infer_aspects_from_text(text)
        return overall

    # ---------------- 키워드/요약 추출 ----------------
    def extract_insights(self, reviews: list, condition_desc: str) -> dict:
        """리뷰 목록을 종합하여 긍정/부정 키워드, 요약, 개선 제안을 생성."""
        if not self.available:
            self.logger.warning(f"AI provider={self.provider} 비활성 — 규칙 기반 키워드 추출로 대체합니다.")
            return self._fallback_extract(reviews)

        system = (
            "너는 커머스 VOC(고객의 소리) 분석가다. 주어진 리뷰 목록을 종합 분석하여 아래 JSON 스키마로만 답하라. "
            "다른 설명 문장은 절대 포함하지 마라.\n"
            "{\n"
            '  "positive_keywords": [{"keyword": "빠른 배송", "count": 23}, ...],\n'
            '  "negative_keywords": [{"keyword": "배송 지연", "count": 8}, ...],\n'
            '  "summary": "전체 리뷰에 대한 2~4문장 요약",\n'
            '  "suggestions": ["개선 제안1", "개선 제안2"],\n'
            '  "topic_breakdown": [{"topic": "배송", "count": 9, "examples": ["배송 지연", "오배송"]}]\n'
            "}\n"
            "positive_keywords/negative_keywords 의 keyword는 \"배송 지연\", \"품질 불량\"처럼 "
            "단어 하나가 아니라 의미가 통하는 2~3어절 구(句)로 만들고, count는 해당 키워드가 "
            "리뷰들에서 실제로 언급된(또는 그와 같은 취지의) 횟수를 세어 넣어라. 키워드는 count 내림차순으로 정렬하라.\n"
            "topic_breakdown 은 부정/긍정 리뷰를 유형별로 묶어 건수와 대표 키워드를 제공하는 항목이다.\n"
            "응답이 너무 길어지지 않도록 반드시 지켜라: positive_keywords/negative_keywords는 "
            "각각 최대 5개, topic_breakdown은 최대 5개 유형까지만, 각 유형의 examples는 최대 3개까지만 "
            "포함하라. summary는 4문장을 넘기지 마라."
        )
        joined = "\n".join(f"- ({r.get('sentiment','?')}, {r.get('rating','?')}점) {r.get('review_text','')}" for r in reviews[:200])
        user_prompt = f"[분석 조건: {condition_desc}]\n리뷰 목록:\n{joined}"

        parsed = None
        result = None
        for attempt in range(2):
            result = self._call_llm(
                self.extract_model, system, user_prompt,
                max_tokens=self.extract_max_tokens, timeout=self.extract_timeout,
            )
            parsed = self._extract_json(result) if result else None
            if parsed:
                return parsed
            if attempt == 0:
                self.logger.warning("AI 추출 첫 시도가 실패해 한 번 더 재시도합니다...")

        if result and not parsed:
            self.logger.error(f"AI 추출 응답을 JSON으로 파싱하지 못했습니다. 원문 일부: {result[:300]!r}")

        self.logger.warning(
            "AI 키워드/요약 추출 호출이 실패해 규칙 기반 결과로 대체합니다 "
            f"(provider={self.provider} — 크레딧 부족/인증오류/네트워크 오류/응답 잘림 등 - logs/app.log 확인)."
        )
        return self._fallback_extract(reviews)

    @staticmethod
    def _fallback_extract(reviews: list) -> dict:
        pos_words, neg_words = {}, {}
        for r in reviews:
            text = (r.get("review_text") or "")
            bucket = pos_words if r.get("sentiment") == "positive" else neg_words if r.get("sentiment") == "negative" else None
            if bucket is None:
                continue
            for hint in (POSITIVE_HINTS if bucket is pos_words else NEGATIVE_HINTS):
                if hint.lower() in text.lower():
                    bucket[hint] = bucket.get(hint, 0) + 1
        pos_sorted = sorted(pos_words.items(), key=lambda x: -x[1])[:5]
        neg_sorted = sorted(neg_words.items(), key=lambda x: -x[1])[:5]

        topic_map = {
            "배송": ["늦", "배송"],
            "품질": ["불량", "나빠", "최악"],
            "서비스": ["불편", "안돼", "안됨"],
            "가격/기타": ["실망", "환불", "반품"],
        }
        topic_breakdown = []
        for topic, hints in topic_map.items():
            count = sum(1 for r in reviews if r.get("sentiment") == "negative" and
                        any(h in (r.get("review_text") or "") for h in hints))
            if count:
                topic_breakdown.append({"topic": topic, "count": count, "examples": hints})

        return {
            "positive_keywords": [{"keyword": w, "count": c} for w, c in pos_sorted] or [{"keyword": "데이터 부족", "count": 0}],
            "negative_keywords": [{"keyword": w, "count": c} for w, c in neg_sorted] or [{"keyword": "데이터 부족", "count": 0}],
            "summary": f"총 {len(reviews)}건의 리뷰를 규칙 기반으로 요약했습니다. "
                       f"(실제 AI 요약을 원하면 AI 프로바이더/API 키를 설정하세요.)",
            "suggestions": ["API 키 설정 후 재실행하면 더 정교한 AI 인사이트를 받을 수 있습니다."],
            "topic_breakdown": topic_breakdown,
        }
