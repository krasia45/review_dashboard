"""
공통 유틸리티 함수 모음
----------------------
날짜 파싱, 텍스트 정규화, 해시 생성 등 여러 모듈에서 공통으로 쓰는 기능을 모아둔다.
"""
import hashlib
import re
from datetime import datetime, date
from typing import Optional


DATE_FORMATS = ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%d-%m-%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%S"]


def normalize_text(text: str) -> str:
    """공백 정리, 앞뒤 공백 제거 등 기본적인 텍스트 정규화."""
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_date(value) -> Optional[str]:
    """다양한 형식의 날짜 입력을 'YYYY-MM-DD' 문자열로 통일한다. 실패 시 None."""
    if value is None or value == "":
        return None
    if isinstance(value, (datetime, date)):
        return value.strftime("%Y-%m-%d")
    text = str(value).strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def normalize_rating(value) -> Optional[int]:
    """별점을 1~5 범위의 정수로 검증한다. 범위를 벗어나거나 파싱 불가 시 None."""
    try:
        rating = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    if 1 <= rating <= 5:
        return rating
    return None


def dedup_hash(text: str, product: Optional[str] = None) -> str:
    """중복 판정을 위한 해시. 정규화된 리뷰 텍스트 + 제품명 기준."""
    base = normalize_text(text).lower()
    if product:
        base += f"|{normalize_text(product).lower()}"
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def detect_language(text: str) -> str:
    """아주 단순한 휴리스틱 언어 감지: 한글 -> ko, (한글 없이) 한자 -> zh, 그 외 -> en
    (다국어 보너스 과제용, 한국어/영어/중국어 3개 언어 지원)."""
    t = text or ""
    if re.search(r"[가-힣]", t):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", t):
        return "zh"
    return "en"


# ── 감정 점수 (3단계) ──────────────────────────────────────────────────
# 감정(긍정/부정/중립)을 그대로 3단계 점수로 옮긴다: 부정(1)/중립(2)/긍정(3).
# 라벨을 "긍정/중립/부정"으로 통일해서, 다른 곳(감정 분류)과 같은 용어를 쓰도록 했다.
# 실제 기업 대시보드(NPS/CSAT 계열)에서도 감정 점수는 보통 3~5단계보다 훨씬
# 단순하게 "부정/중립/긍정" 3단계로 보여주는 경우가 많아, 5단계(신뢰도로 세분화)
# 대신 이 방식으로 통일했다. confidence는 여전히 별도 지표(품질 지표의 평균
# 신뢰도)로 표시되므로 정보가 사라지는 것은 아니다.
SENTIMENT_GRADES = [
    {"score": 1, "label": "부정", "color": "#DC2626"},
    {"score": 2, "label": "중립", "color": "#94A3B8"},
    {"score": 3, "label": "긍정", "color": "#16A34A"},
]
_GRADE_BY_SCORE = {g["score"]: g for g in SENTIMENT_GRADES}


def sentiment_grade(sentiment: Optional[str], confidence: Optional[float] = None, strong_threshold: float = 0.75) -> dict:
    """(sentiment) -> {"score": 1~3, "label": str, "color": str}
    confidence/strong_threshold 인자는 이전 버전과의 호환을 위해 남겨두었으나,
    3단계로 단순화하면서 등급 계산에는 더 이상 쓰이지 않는다."""
    if sentiment == "positive":
        return _GRADE_BY_SCORE[3]
    if sentiment == "negative":
        return _GRADE_BY_SCORE[1]
    return _GRADE_BY_SCORE[2]


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_str() -> str:
    return datetime.now().strftime("%Y-%m-%d")
