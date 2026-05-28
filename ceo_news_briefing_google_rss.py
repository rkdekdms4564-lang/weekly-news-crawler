import requests
import urllib.parse
import time
import random
import re
import json
import os
import html
import math
import difflib
import pandas as pd
import feedparser

from google import genai
from datetime import datetime, date, timedelta, timezone
from email.utils import parsedate_to_datetime
from newspaper import Article, Config
from bs4 import BeautifulSoup

try:
    import trafilatura
except Exception:
    trafilatura = None

try:
    from googlenewsdecoder import gnewsdecoder
except Exception:
    gnewsdecoder = None


# ==========================================
# 0. 기본 설정
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SECRET_PATH = os.path.join(BASE_DIR, "secret.txt")
PAST_REPORTS_PATH = os.path.join(BASE_DIR, "past_reports.txt")

OUTPUT_TXT = os.path.join(BASE_DIR, "CEO_Morning_Briefing.txt")
OUTPUT_SELECTED_CSV = os.path.join(BASE_DIR, "google_news_top15_raw.csv")
OUTPUT_CANDIDATES_CSV = os.path.join(BASE_DIR, "google_news_candidates_raw.csv")
OUTPUT_SKIPPED_DUP_CSV = os.path.join(BASE_DIR, "skipped_past_duplicates.csv")
OUTPUT_BODY_FAILED_CSV = os.path.join(BASE_DIR, "body_extract_failed.csv")

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_ATTEMPTS = 5

# Google News RSS에서 최근 며칠 기사까지 가져올지
RSS_RECENT_DAYS = 1

# RSS가 when:1d를 주더라도 발행시각 기준으로 한 번 더 거름
STRICT_RSS_TIME_FILTER = True
RSS_RECENCY_HOURS = 30

MAX_NEWS_PER_KEYWORD = 20
MAX_COMPRESSED_PER_KEYWORD = 8

MIN_SELECT_COUNT = 13
MAX_SELECT_COUNT = 15

# past_reports.txt에서 최근 며칠 기사를 중복 판단 기준으로 쓸지
PAST_DUP_LOOKBACK_DAYS = 7

# 본문이 이 글자 수보다 짧으면 추출 실패에 가깝다고 봄
MIN_BODY_CHARS = 250

KST = timezone(timedelta(hours=9))

WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

KOREAN_WEEKDAYS = ["월", "화", "수", "목", "금", "토", "일"]


# ==========================================
# 1. 키워드 리스트
# ==========================================

keyword_categories = {
    "자사 및 계열사 이슈": [
        "카카오", "카카오톡", "카카오모빌리티", "카카오페이", "카카오뱅크",
        "카카오엔터테인먼트", "카카오게임즈", "카카오픽코마", "카카오헬스케어",
        "카카오엔터프라이즈", "정신아", "카카오 김범수"
    ],
    "정부/국회": [
        "과학기술정보통신부", "방송미디어통신위원회", "공정거래위원회", "금융위원회",
        "금융감독원", "행정안전부", "중소벤처기업부", "과학기술정보통신위원회",
        "정무위원회", "성평등가족위원회", "문화체육관광위원회", "플랫폼 규제", "온플법",
        "스테이블코인", "국가AI컴퓨팅센터", "국가AI전략위원회", "지도반출", "딥페이크",
        "AI기본법", "AI 저작권"
    ],
    "경쟁사/해외이슈": [
        "네이버", "SKT", "KT", "LGU+", "쿠팡", "토스", "배달의민족",
        "구글", "오픈AI", "MS", "메타", "애플", "EU 규제", "트럼프 행정부",
        "중국 AI", "일본 빅테크"
    ],
    "산업동향": [
        "인공지능 산업 동향", "플랫폼 산업 동향", "IT 협단체", "플랫폼 시장점유율"
    ]
}

CATEGORY_TO_JSON_KEY = {
    "자사 및 계열사 이슈": "자사_및_계열사_이슈",
    "정부/국회": "정부_국회",
    "경쟁사/해외이슈": "경쟁사_해외이슈",
    "산업동향": "산업동향",
}

JSON_KEY_TO_DISPLAY = {
    "자사_및_계열사_이슈": "자사 및 계열사 이슈",
    "정부_국회": "정부/국회",
    "경쟁사_해외이슈": "경쟁사/해외이슈",
    "산업동향": "산업동향",
}

JSON_KEYS_ORDER = [
    "자사_및_계열사_이슈",
    "정부_국회",
    "경쟁사_해외이슈",
    "산업동향",
]

CATEGORY_MAX = {
    "자사_및_계열사_이슈": 4,
    "정부_국회": 5,
    "경쟁사_해외이슈": 4,
    "산업동향": 2,
}

KEY_ALIASES = {
    "자사_및_계열사_이슈": [
        "자사_및_계열사_이슈", "자사 및 계열사 이슈", "자사및계열사이슈", "자사", "계열사"
    ],
    "정부_국회": [
        "정부_국회", "정부/국회", "정부·국회", "정부 국회", "정부", "국회"
    ],
    "경쟁사_해외이슈": [
        "경쟁사_해외이슈", "경쟁사/해외이슈", "경쟁사·해외이슈", "경쟁사 해외이슈", "경쟁사", "해외이슈"
    ],
    "산업동향": ["산업동향", "산업 동향"],
}

all_keywords = []
keyword_to_category = {}

for category_name, keyword_list in keyword_categories.items():
    for keyword in keyword_list:
        all_keywords.append(keyword)
        keyword_to_category[keyword] = category_name


# ==========================================
# 2. 문자열/URL 유틸
# ==========================================

def clean_html_text(text):
    if text is None:
        return ""
    text = str(text)
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_url(url):
    if not url:
        return ""

    url = html.unescape(str(url)).strip()

    try:
        parsed = urllib.parse.urlparse(url)
        query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)

        tracking_keys = {
            "fbclid", "gclid", "igshid", "wbraid", "gbraid",
            "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
        }

        filtered_query_pairs = []
        for k, v in query_pairs:
            lower_k = k.lower()
            if lower_k.startswith("utm_"):
                continue
            if lower_k in tracking_keys:
                continue
            filtered_query_pairs.append((k, v))

        cleaned_query = urllib.parse.urlencode(filtered_query_pairs, doseq=True)

        cleaned = urllib.parse.urlunparse((
            parsed.scheme,
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.params,
            cleaned_query,
            "",
        ))
        return cleaned

    except Exception:
        return url


def guess_press_name_from_url(url):
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""


def clean_rss_title(title, source):
    title = clean_html_text(title)
    source = clean_html_text(source)

    if source:
        suffix = f" - {source}"
        if title.endswith(suffix):
            title = title[:-len(suffix)].strip()

    return title


def get_entry_source_title(entry):
    try:
        source = entry.get("source", {})
        if isinstance(source, dict):
            return clean_html_text(source.get("title", ""))
        return clean_html_text(getattr(source, "title", ""))
    except Exception:
        return ""


def today_report_header():
    now = datetime.now(KST).date()
    weekday = KOREAN_WEEKDAYS[now.weekday()]
    return f"[{now.month}월 {now.day}일({weekday}) 주요 이슈]"


# ==========================================
# 3. past_reports.txt 파싱 및 유사도 엔진
# ==========================================

REPORT_HEADING_RE = re.compile(r"^\s*\[(\d{1,2})월\s*(\d{1,2})일(?:\([^)]*\))?\s*주요\s*이슈\]\s*$")
CATEGORY_RE = re.compile(r"^\s*☑️\s*(.+?)\s*$")
ARTICLE_START_RE = re.compile(r"^\s*(?:\d+\ufe0f?\u20e3|🔟|\d+[.)])\s+(.+?)\s*$")
URL_RE = re.compile(r"https?://\S+")
PRESS_RE = re.compile(r"^\s*\(([^)]+)\)\s*$")

STOPWORDS = {
    "기자", "단독", "종합", "속보", "뉴스", "관련", "통해", "위해", "대한", "대해", "이번", "지난", "오는",
    "오늘", "내일", "밝힘", "계획", "추진", "진행", "시작", "개최", "도입", "확대", "강화", "나서", "나선다",
    "서비스", "시장", "사업", "정부", "업계", "기업", "플랫폼", "인공지능", "AI", "ai", "국내", "해외", "글로벌",
    "전격", "매핑", "가드레일", "방침", "전망", "가능성", "중심", "기반", "대상", "공식", "검토", "논의",
    "제공", "운영", "지원", "사용", "활용", "발표", "확인", "결과", "경우", "내용", "주요", "이슈",
}


def infer_report_date(month, day, now_date=None):
    if now_date is None:
        now_date = datetime.now(KST).date()

    inferred = date(now_date.year, month, day)

    # 1월에 전년도 12월 보고서를 읽는 경우 보정
    if inferred > now_date + timedelta(days=7):
        inferred = date(now_date.year - 1, month, day)

    return inferred


def parse_past_reports(text):
    """
    past_reports.txt 형식:
    [5월 22일(금) 주요 이슈]
    ☑️ 자사 및 계열사 이슈
    1️⃣ 기사 제목
    URL
    (언론사)
    요약문
    """
    lines = text.splitlines()
    items = []

    current_date = None
    current_category = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        heading_match = REPORT_HEADING_RE.match(line)
        if heading_match:
            month = int(heading_match.group(1))
            day = int(heading_match.group(2))
            current_date = infer_report_date(month, day)
            current_category = None
            i += 1
            continue

        category_match = CATEGORY_RE.match(line)
        if category_match:
            current_category = category_match.group(1).strip()
            i += 1
            continue

        article_match = ARTICLE_START_RE.match(line)
        if article_match and current_date and current_category:
            title = clean_html_text(article_match.group(1))
            link = ""
            press = ""
            summary_lines = []

            j = i + 1

            while j < len(lines):
                maybe = lines[j].strip()
                if URL_RE.search(maybe):
                    link = URL_RE.search(maybe).group(0).strip()
                    j += 1
                    break
                if REPORT_HEADING_RE.match(maybe) or CATEGORY_RE.match(maybe) or ARTICLE_START_RE.match(maybe):
                    break
                j += 1

            if j < len(lines):
                press_match = PRESS_RE.match(lines[j].strip())
                if press_match:
                    press = clean_html_text(press_match.group(1))
                    j += 1

            while j < len(lines):
                nxt = lines[j].strip()
                if REPORT_HEADING_RE.match(nxt) or CATEGORY_RE.match(nxt) or ARTICLE_START_RE.match(nxt):
                    break
                if nxt:
                    summary_lines.append(clean_html_text(nxt))
                j += 1

            summary = " ".join(summary_lines).strip()

            if title:
                items.append({
                    "date": current_date,
                    "category": current_category,
                    "title": title,
                    "link": normalize_url(link),
                    "press": press,
                    "summary": summary,
                    "text": f"{title} {summary}".strip(),
                })

            i = j
            continue

        i += 1

    return items


def load_past_reports():
    if not os.path.exists(PAST_REPORTS_PATH):
        print("📝 'past_reports.txt' 파일이 없습니다. 과거 중복 제거 없이 진행합니다.")
        return "", [], []

    with open(PAST_REPORTS_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()

    all_items = parse_past_reports(content)

    today = datetime.now(KST).date()
    cutoff = today - timedelta(days=PAST_DUP_LOOKBACK_DAYS)
    recent_items = [item for item in all_items if cutoff <= item["date"] <= today]

    print(f"📚 'past_reports.txt' 로드 완료: 전체 {len(all_items)}건 파싱")
    print(f"   └ 최근 {PAST_DUP_LOOKBACK_DAYS}일 중복 판단 기준 기사: {len(recent_items)}건")

    return content, all_items, recent_items


def normalize_for_similarity(text):
    text = clean_html_text(text).lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\([^)]*기자[^)]*\)", " ", text)
    text = re.sub(r"[가-힣]{2,5}\s*기자", " ", text)
    text = re.sub(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", " ", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣+]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize_for_similarity(text):
    text = normalize_for_similarity(text)
    raw_tokens = re.findall(r"[0-9a-zA-Z가-힣+]{2,}", text)

    tokens = []
    for token in raw_tokens:
        if token in STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        tokens.append(token)

    return set(tokens)


def char_ngrams(text, n=3):
    text = normalize_for_similarity(text).replace(" ", "")

    if len(text) < n:
        return {text} if text else set()

    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def sequence_ratio(a, b):
    a = normalize_for_similarity(a)
    b = normalize_for_similarity(b)

    if not a or not b:
        return 0.0

    return difflib.SequenceMatcher(None, a, b).ratio()


def article_similarity(candidate, past_item):
    candidate_title = candidate.get("기사제목") or candidate.get("title") or ""
    candidate_summary = candidate.get("본문요약") or candidate.get("summary") or candidate.get("본문전문") or ""
    candidate_link = normalize_url(candidate.get("링크") or candidate.get("link") or "")

    past_title = past_item.get("title", "")
    past_summary = past_item.get("summary", "")
    past_link = normalize_url(past_item.get("link", ""))

    if candidate_link and past_link and candidate_link == past_link:
        return {
            "is_duplicate": True,
            "reason": "same_url",
            "score": 1.0,
            "title_score": 1.0,
            "token_score": 1.0,
            "text_score": 1.0,
        }

    candidate_title_norm = normalize_for_similarity(candidate_title)
    past_title_norm = normalize_for_similarity(past_title)

    if candidate_title_norm and past_title_norm and candidate_title_norm == past_title_norm:
        return {
            "is_duplicate": True,
            "reason": "same_title",
            "score": 1.0,
            "title_score": 1.0,
            "token_score": 1.0,
            "text_score": 1.0,
        }

    title_seq = sequence_ratio(candidate_title, past_title)
    title_ngram = jaccard(char_ngrams(candidate_title), char_ngrams(past_title))
    title_score = max(title_seq, title_ngram)

    candidate_text = f"{candidate_title} {candidate_summary}".strip()
    past_text = f"{past_title} {past_summary}".strip()

    token_score = jaccard(tokenize_for_similarity(candidate_text), tokenize_for_similarity(past_text))
    text_score = jaccard(char_ngrams(candidate_text), char_ngrams(past_text))

    combined = (0.55 * title_score) + (0.25 * token_score) + (0.20 * text_score)

    is_duplicate = False
    reason = ""

    # 너무 넓게 잡으면 같은 주제의 다른 기사까지 지워지므로, 꽤 엄격하게 설정
    if title_score >= 0.78 and token_score >= 0.22:
        is_duplicate = True
        reason = "very_similar_title"
    elif title_score >= 0.68 and token_score >= 0.36:
        is_duplicate = True
        reason = "similar_title_and_tokens"
    elif combined >= 0.70 and token_score >= 0.30:
        is_duplicate = True
        reason = "similar_event"

    return {
        "is_duplicate": is_duplicate,
        "reason": reason,
        "score": round(combined, 4),
        "title_score": round(title_score, 4),
        "token_score": round(token_score, 4),
        "text_score": round(text_score, 4),
    }


def find_past_duplicate(candidate, recent_past_items):
    if not recent_past_items:
        return False, None, None

    best_item = None
    best_result = None
    best_score = -1.0

    for past_item in recent_past_items:
        result = article_similarity(candidate, past_item)
        score = result.get("score", 0.0)

        if result.get("is_duplicate"):
            return True, past_item, result

        if score > best_score:
            best_score = score
            best_item = past_item
            best_result = result

    return False, best_item, best_result


def build_recent_past_text(recent_past_items, max_chars=6000):
    if not recent_past_items:
        return "최근 7일 과거 보고서 데이터 없음."

    sorted_items = sorted(recent_past_items, key=lambda x: (x["date"], x["category"]))
    chunks = []

    for item in sorted_items:
        chunks.append(
            f"[{item['date'].month}월 {item['date'].day}일 / {item['category']}] "
            f"{item['title']}\n{item['summary']}\n"
        )

    text = "\n".join(chunks).strip()

    if len(text) > max_chars:
        text = text[-max_chars:]

    return text


# ==========================================
# 4. Gemini 유틸
# ==========================================

def gemini_generate_text(client, prompt, task_name, model=GEMINI_MODEL, max_attempts=GEMINI_MAX_ATTEMPTS):
    last_error = None

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
            )

            text = getattr(response, "text", "") or ""

            if not text.strip():
                raise ValueError("Gemini 응답이 비어 있습니다.")

            return text

        except Exception as e:
            last_error = e

            if attempt >= max_attempts:
                break

            delay = min(60, (2 ** attempt) + random.uniform(0, 3))

            print(
                f"  └ ⚠️ {task_name} 실패 "
                f"(재시도 {attempt}/{max_attempts})... "
                f"{delay:.1f}초 후 다시 요청합니다. "
                f"원인: {type(e).__name__}"
            )

            time.sleep(delay)

    raise last_error


def extract_json_object(text):
    if not text:
        raise ValueError("빈 응답입니다.")

    text = text.strip()
    text = re.sub(r"^```json", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```", "", text).strip()
    text = re.sub(r"```$", "", text).strip()

    match = re.search(r"\{[\s\S]*\}", text)

    if not match:
        raise ValueError("응답에서 JSON 객체를 찾지 못했습니다.")

    json_text = match.group(0)
    json_text = re.sub(r",\s*([\]}])", r"\1", json_text)

    return json.loads(json_text)


def normalize_selection_json(data, final_candidates):
    valid_ids = {int(a["id"]) for a in final_candidates}
    used_ids = set()
    result = {key: [] for key in JSON_KEYS_ORDER}

    for expected_key in JSON_KEYS_ORDER:
        values = None

        for alias in KEY_ALIASES.get(expected_key, [expected_key]):
            if alias in data:
                values = data.get(alias)
                break

        if values is None or not isinstance(values, list):
            values = []

        for value in values:
            try:
                art_id = int(value)
            except Exception:
                continue

            if art_id not in valid_ids:
                continue
            if art_id in used_ids:
                continue

            result[expected_key].append(art_id)
            used_ids.add(art_id)

    return enforce_selection_limits(result, final_candidates)


def enforce_selection_limits(selection, final_candidates):
    result = {key: [] for key in JSON_KEYS_ORDER}
    used = set()

    # 1차: 카테고리별 최대치까지 유지
    for key in JSON_KEYS_ORDER:
        max_count = CATEGORY_MAX.get(key, 3)
        for art_id in selection.get(key, []):
            if art_id in used:
                continue
            if len(result[key]) >= max_count:
                continue
            result[key].append(art_id)
            used.add(art_id)

    # 2차: 총량이 부족하면 후보군에서 채움
    total = sum(len(v) for v in result.values())

    if total < MIN_SELECT_COUNT:
        for article in final_candidates:
            art_id = int(article["id"])
            if art_id in used:
                continue

            json_key = article.get("JSON카테고리") or "산업동향"
            if json_key not in result:
                json_key = "산업동향"

            # 카테고리 max를 살짝 넘기는 것보다 총 13개 확보를 우선
            result[json_key].append(art_id)
            used.add(art_id)

            total = sum(len(v) for v in result.values())
            if total >= MIN_SELECT_COUNT:
                break

    # 3차: 15개 초과 시 뒤에서 자름
    while sum(len(v) for v in result.values()) > MAX_SELECT_COUNT:
        removed = False
        for key in reversed(JSON_KEYS_ORDER):
            if result[key]:
                result[key].pop()
                removed = True
                break
        if not removed:
            break

    return result


def deterministic_selection(final_candidates):
    result = {key: [] for key in JSON_KEYS_ORDER}
    used_ids = set()

    for json_key in JSON_KEYS_ORDER:
        max_count = CATEGORY_MAX.get(json_key, 3)
        category_candidates = [
            a for a in final_candidates
            if a.get("JSON카테고리") == json_key and int(a["id"]) not in used_ids
        ]

        for article in category_candidates[:max_count]:
            art_id = int(article["id"])
            result[json_key].append(art_id)
            used_ids.add(art_id)

    return enforce_selection_limits(result, final_candidates)


# ==========================================
# 5. Google News RSS 수집
# ==========================================

def parse_pubdate_to_kst(pub_date_text):
    if not pub_date_text:
        return None

    try:
        dt = parsedate_to_datetime(pub_date_text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(KST)
    except Exception:
        return None


def is_recent_pubdate(pub_date_text):
    if not STRICT_RSS_TIME_FILTER:
        return True

    dt = parse_pubdate_to_kst(pub_date_text)
    if dt is None:
        return True

    now = datetime.now(KST)
    start = now - timedelta(hours=RSS_RECENCY_HOURS)

    return start <= dt <= now + timedelta(minutes=10)


def add_article(raw_articles, seen_urls, article_id, keyword, title, link, source="", published="", summary="", collector=""):
    title = clean_html_text(title)
    link = str(link).strip() if link else ""

    if not title or not link:
        return article_id, False

    normalized = normalize_url(link)
    if not normalized:
        return article_id, False

    if normalized in seen_urls:
        return article_id, False

    seen_urls.add(normalized)

    category_name = keyword_to_category.get(keyword, "")
    json_category = CATEGORY_TO_JSON_KEY.get(category_name, "")

    if not source:
        source = guess_press_name_from_url(link)

    raw_articles.append({
        "id": article_id,
        "원카테고리": category_name,
        "JSON카테고리": json_category,
        "검색어": keyword,
        "기사제목": title,
        "언론사": clean_html_text(source),
        "게시일": clean_html_text(published),
        "본문요약": clean_html_text(summary),
        "링크": normalized,
        "수집채널": collector,
    })

    return article_id + 1, True


def collect_with_google_rss(recent_past_items):
    print("\n🚀 [STEP 1] Google News RSS로 오늘 뉴스 후보 수집 시작...")
    print("   └ 원문 URL 디코딩은 여기서 하지 않고, 최종 선별 기사에만 나중에 적용합니다.")

    session = requests.Session()
    session.headers.update(WEB_HEADERS)

    raw_articles = []
    skipped_duplicates = []
    seen_urls = set()
    article_id = 1

    for idx, keyword in enumerate(all_keywords, 1):
        collected = 0
        skipped_old = 0
        skipped_past = 0

        query = urllib.parse.quote_plus(f"{keyword} when:{RSS_RECENT_DAYS}d")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"

        try:
            req = session.get(rss_url, timeout=10)
            req.raise_for_status()
            feed = feedparser.parse(req.content)

            for entry in feed.entries[:MAX_NEWS_PER_KEYWORD]:
                google_link = entry.get("link", "")
                if not google_link:
                    continue

                source = get_entry_source_title(entry)
                title = clean_rss_title(entry.get("title", ""), source)
                summary = clean_html_text(entry.get("summary", ""))
                published = clean_html_text(entry.get("published", ""))

                if not is_recent_pubdate(published):
                    skipped_old += 1
                    continue

                # 핵심: 여기서는 Google News URL 그대로 저장. 원문 디코딩 X.
                candidate = {
                    "기사제목": title,
                    "본문요약": summary,
                    "링크": google_link,
                }

                is_dup, matched, sim = find_past_duplicate(candidate, recent_past_items)
                if is_dup:
                    skipped_past += 1
                    skipped_duplicates.append({
                        "검색어": keyword,
                        "후보제목": title,
                        "후보링크": google_link,
                        "후보언론사": source,
                        "매칭과거일자": matched.get("date") if matched else "",
                        "매칭과거제목": matched.get("title") if matched else "",
                        "매칭과거링크": matched.get("link") if matched else "",
                        "중복판정이유": sim.get("reason") if sim else "",
                        "종합점수": sim.get("score") if sim else "",
                        "제목점수": sim.get("title_score") if sim else "",
                        "토큰점수": sim.get("token_score") if sim else "",
                        "본문점수": sim.get("text_score") if sim else "",
                    })
                    continue

                article_id, added = add_article(
                    raw_articles=raw_articles,
                    seen_urls=seen_urls,
                    article_id=article_id,
                    keyword=keyword,
                    title=title,
                    link=google_link,
                    source=source,
                    published=published,
                    summary=summary,
                    collector="google_rss_fast_no_decode",
                )

                if added:
                    collected += 1

            print(
                f"▶ [GOOGLE RSS {idx}/{len(all_keywords)}] "
                f"'{keyword}' 수집 {collected}개 / 시간제외 {skipped_old}개 / 과거중복제외 {skipped_past}개        ",
                end="\r"
            )

        except Exception as e:
            print(f"\n⚠️ [GOOGLE RSS {idx}/{len(all_keywords)}] '{keyword}' 수집 실패: {e}")

        time.sleep(random.uniform(0.05, 0.15))

    print("")
    return raw_articles, skipped_duplicates


# ==========================================
# 6. 선별 기사 URL 디코딩 및 본문 추출
# ==========================================

def decode_google_news_url(google_news_url):
    """
    최종 선별된 Google News URL만 원문 URL로 변환.
    전체 후보군에는 이 함수를 절대 돌리지 않음.
    """
    if not google_news_url:
        return ""

    if "news.google." not in google_news_url:
        return google_news_url

    if gnewsdecoder is not None:
        try:
            decoded = gnewsdecoder(google_news_url, interval=1)

            if isinstance(decoded, dict):
                if decoded.get("status") and decoded.get("decoded_url"):
                    return decoded["decoded_url"]

            if isinstance(decoded, str) and decoded.startswith("http"):
                return decoded

        except Exception:
            pass

    try:
        r = requests.get(
            google_news_url,
            headers=WEB_HEADERS,
            timeout=10,
            allow_redirects=True,
        )
        if r.url:
            return r.url
    except Exception:
        pass

    return google_news_url


def remove_unwanted_tags(soup):
    for tag in soup([
        "script", "style", "noscript", "header", "footer", "nav", "aside",
        "iframe", "form", "button", "figure"
    ]):
        tag.decompose()

    return soup


def extract_text_from_bs4(html_text):
    soup = BeautifulSoup(html_text, "html.parser")
    soup = remove_unwanted_tags(soup)

    selectors = [
        "article",
        "[itemprop='articleBody']",
        "#articleBody",
        "#article_body",
        "#newsEndContents",
        "#dic_area",
        ".article_body",
        ".articleBody",
        ".article-body",
        ".article_view",
        ".article-view",
        ".news_view",
        ".news-view",
        ".news_content",
        ".news-content",
        ".view_cont",
        ".view-content",
        ".read_txt",
        ".article_txt",
        ".story-news",
        ".story-news-article",
        ".contents",
        ".content",
    ]

    candidate_texts = []

    for selector in selectors:
        for node in soup.select(selector):
            paragraphs = []
            p_tags = node.find_all(["p", "div"], recursive=True)

            if not p_tags:
                text = clean_html_text(node.get_text(" "))
                if len(text) >= 100:
                    candidate_texts.append(text)
                continue

            for p in p_tags:
                text = clean_html_text(p.get_text(" "))

                if len(text) < 25:
                    continue
                if "무단전재" in text or "재배포 금지" in text:
                    continue
                if "Copyright" in text or "저작권자" in text:
                    continue
                if re.search(r"[가-힣]{2,5}\s*기자\s*=", text):
                    continue

                paragraphs.append(text)

            joined = "\n".join(paragraphs).strip()
            if len(joined) >= 100:
                candidate_texts.append(joined)

    if candidate_texts:
        # 가장 긴 후보를 본문으로 채택
        return max(candidate_texts, key=len)

    # 최후 fallback: p 태그 전체
    paragraphs = []
    for p in soup.find_all("p"):
        text = clean_html_text(p.get_text(" "))
        if len(text) < 25:
            continue
        if "무단전재" in text or "재배포 금지" in text:
            continue
        paragraphs.append(text)

    joined = "\n".join(paragraphs).strip()
    return joined


def clean_extracted_body(text):
    text = clean_html_text(text)

    if not text:
        return ""

    # 흔한 꼬리 문구 제거
    cut_patterns = [
        r"무단전재.*$",
        r"재배포\s*금지.*$",
        r"Copyright.*$",
        r"저작권자.*$",
    ]

    for pattern in cut_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL).strip()

    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_article_body(url):
    """
    선별 기사 본문 추출.
    1) newspaper3k
    2) requests HTML + trafilatura
    3) requests HTML + BS4 selector
    4) meta description
    """
    if not url:
        return "", "none"

    # 1차: newspaper3k
    try:
        config = Config()
        config.browser_user_agent = WEB_HEADERS["User-Agent"]
        config.request_timeout = 12
        config.fetch_images = False

        article = Article(url=url, language="ko", config=config)
        article.download()
        article.parse()

        text = clean_extracted_body(article.text)

        if len(text) >= MIN_BODY_CHARS:
            return text, "newspaper3k"
    except Exception:
        pass

    html_text = ""

    # HTML 확보
    try:
        r = requests.get(
            url,
            headers=WEB_HEADERS,
            timeout=12,
            allow_redirects=True,
        )
        r.raise_for_status()

        if r.encoding is None or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding

        html_text = r.text or ""
    except Exception:
        html_text = ""

    # 2차: trafilatura
    if html_text and trafilatura is not None:
        try:
            extracted = trafilatura.extract(
                html_text,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
                favor_precision=False,
            )
            text = clean_extracted_body(extracted)
            if len(text) >= MIN_BODY_CHARS:
                return text, "trafilatura"
        except Exception:
            pass

    # 3차: BS4 selector
    if html_text:
        try:
            text = clean_extracted_body(extract_text_from_bs4(html_text))
            if len(text) >= MIN_BODY_CHARS:
                return text, "bs4_selector"
        except Exception:
            pass

    # 4차: meta description
    if html_text:
        try:
            soup = BeautifulSoup(html_text, "html.parser")
            meta_candidates = []

            for attrs in [
                {"property": "og:description"},
                {"name": "description"},
                {"name": "twitter:description"},
            ]:
                tag = soup.find("meta", attrs=attrs)
                if tag and tag.get("content"):
                    meta_candidates.append(clean_html_text(tag.get("content")))

            meta_text = max(meta_candidates, key=len) if meta_candidates else ""
            if len(meta_text) >= 50:
                return meta_text, "meta_description_short"
        except Exception:
            pass

    return "", "failed"


def process_article_for_report(article_info, json_key, recent_past_items):
    original_link = article_info.get("링크", "")

    # 핵심: 최종 선별된 기사만 여기서 디코딩
    real_url = decode_google_news_url(original_link)
    real_url = normalize_url(real_url)

    body_text, extract_method = extract_article_body(real_url)

    if not body_text or len(body_text.strip()) < 30:
        body_text = article_info.get("본문요약", "")

    if not body_text or len(body_text.strip()) < 30:
        body_text = "본문 추출 실패. 원문 링크를 확인하세요."
        extract_method = "failed"

    # 본문 추출 후 한 번 더 과거 7일과 중복 비교
    body_candidate = {
        "기사제목": article_info.get("기사제목", ""),
        "본문요약": body_text[:2000],
        "링크": real_url,
    }

    is_dup, matched, sim = find_past_duplicate(body_candidate, recent_past_items)

    if is_dup:
        skip_info = {
            "검색어": article_info.get("검색어", ""),
            "후보제목": article_info.get("기사제목", ""),
            "후보링크": real_url,
            "후보언론사": article_info.get("언론사", ""),
            "매칭과거일자": matched.get("date") if matched else "",
            "매칭과거제목": matched.get("title") if matched else "",
            "매칭과거링크": matched.get("link") if matched else "",
            "중복판정이유": f"post_body_{sim.get('reason') if sim else ''}",
            "종합점수": sim.get("score") if sim else "",
            "제목점수": sim.get("title_score") if sim else "",
            "토큰점수": sim.get("token_score") if sim else "",
            "본문점수": sim.get("text_score") if sim else "",
        }
        return None, skip_info

    report_item = {
        "카테고리": json_key,
        "카테고리명": JSON_KEY_TO_DISPLAY.get(json_key, json_key),
        "검색어": article_info.get("검색어", ""),
        "기사제목": article_info.get("기사제목", ""),
        "언론사": article_info.get("언론사", "") or guess_press_name_from_url(real_url),
        "게시일": article_info.get("게시일", ""),
        "본문요약": article_info.get("본문요약", ""),
        "본문전문": body_text.strip(),
        "본문글자수": len(body_text.strip()),
        "본문추출방식": extract_method,
        "원래RSS링크": original_link,
        "링크": real_url,
    }

    return report_item, None


# ==========================================
# 7. 최종 브리핑 fallback
# ==========================================

def fallback_summary_from_text(text):
    text = clean_html_text(text)

    if not text:
        return "본문 추출에 실패해 원문 링크 확인이 필요함."

    text = text[:500].strip()

    if text.endswith(("함.", "임.", "됨.", "함", "임", "됨")):
        return text

    text = text.rstrip(".")
    return f"{text} 등으로 보도됨."


def build_fallback_briefing(final_report_data):
    lines = [today_report_header(), ""]
    number_emojis = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]

    for json_key in JSON_KEYS_ORDER:
        category_name = JSON_KEY_TO_DISPLAY.get(json_key, json_key)
        items = [item for item in final_report_data if item.get("카테고리") == json_key]

        if not items:
            continue

        lines.append(f"☑️ {category_name}")
        lines.append("")

        for idx, item in enumerate(items, 1):
            num = number_emojis[idx - 1] if idx <= len(number_emojis) else f"{idx}."
            title = item.get("기사제목", "")
            link = item.get("링크", "")
            press = item.get("언론사", "") or guess_press_name_from_url(link)
            body = item.get("본문전문", "") or item.get("본문요약", "")

            lines.append(f"{num} {title}")
            lines.append(link)
            lines.append(f"({press})")
            lines.append(fallback_summary_from_text(body))
            lines.append("")

    return "\n".join(lines).strip()


# ==========================================
# 8. 메인 실행
# ==========================================

def main():
    total_start_time = time.time()

    # Gemini 초기화
    try:
        with open(SECRET_PATH, "r", encoding="utf-8") as f:
            google_api_key = f.read().strip()

        if not google_api_key:
            raise ValueError("secret.txt가 비어 있습니다.")

        client = genai.Client(api_key=google_api_key)

    except Exception as e:
        print("❌ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다.")
        print(f"   원인: {e}")
        return

    # past_reports 로드 및 최근 7일 메모리 생성
    past_reports_content, all_past_items, recent_past_items = load_past_reports()
    recent_past_text = build_recent_past_text(recent_past_items, max_chars=7000)

    # STEP 1: Google RSS만 사용
    raw_articles, skipped_duplicates = collect_with_google_rss(recent_past_items)

    print(f"\n  └ ✅ 총 {len(raw_articles)}개의 RSS 후보 확보 완료")
    print(f"  └ 🧹 최근 {PAST_DUP_LOOKBACK_DAYS}일 과거 보고서와 유사해 제외한 후보: {len(skipped_duplicates)}개")

    if raw_articles:
        pd.DataFrame(raw_articles).to_csv(OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 후보 기사 저장: {os.path.basename(OUTPUT_CANDIDATES_CSV)}")

    if skipped_duplicates:
        pd.DataFrame(skipped_duplicates).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 과거 중복 제외 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")

    if not raw_articles:
        print("❌ 수집된 기사가 없습니다. Google News RSS 접속 또는 키워드/기간 설정을 확인하세요.")
        return

    # STEP 2: 무료 API 보호용 후보 압축
    print("\n🧠 [STEP 2] Gemini AI가 과거 7일 중복 이슈를 제외하고 핵심 기사를 선별합니다...")

    compressed_articles = {}
    for article in raw_articles:
        keyword = article["검색어"]
        compressed_articles.setdefault(keyword, [])
        if len(compressed_articles[keyword]) < MAX_COMPRESSED_PER_KEYWORD:
            compressed_articles[keyword].append(article)

    final_candidates = [item for sublist in compressed_articles.values() for item in sublist]

    candidate_text = ""
    for article in final_candidates:
        candidate_text += (
            f"[{article['id']}] "
            f"카테고리: {article.get('원카테고리', '')} / "
            f"검색어: {article.get('검색어', '')} / "
            f"제목: {article.get('기사제목', '')} / "
            f"언론사: {article.get('언론사', '')} / "
            f"게시일: {article.get('게시일', '')} / "
            f"요약: {article.get('본문요약', '')[:180]}\n"
        )

    prompt_selection = f"""
너는 IT 대기업의 유능한 최고 비서실장이야.
아래 [최근 7일 과거 보고서 데이터]와 [오늘 뉴스 후보 리스트]를 비교해서, 오늘 보고서에 넣을 기사만 선별해.

[핵심 원칙]
1. 최근 7일 과거 보고서에 이미 정리된 사건과 실질적으로 같은 내용이면 제외해.
2. 단순 후속 기사라도 새 사실, 새 수치, 새 조치, 새 당국 발표가 없으면 제외해.
3. 오피니언, 사설, 칼럼, 전문가 기고는 제외해.
4. 단순 서비스 홍보, 이벤트성 기사, 가벼운 출시 홍보는 제외해.
5. 동일 사건에 대해 여러 기사가 있으면 대표 기사 1개만 선택해.
6. 카카오 및 계열사는 과징금, 경영진, 서비스 장애, 지배구조, 실적, 규제, 수사, 소송, 대형 제휴 등 무거운 내용 위주로 골라.
7. 타사 이름만 들어간 무관한 기사는 제외해.
8. 총 13~15개를 골라.
9. 카테고리별 기준은 자사 및 계열사 3~4개, 정부/국회 4~5개, 경쟁사/해외이슈 3~4개, 산업동향 1~2개야.

반드시 다른 설명 없이 아래 JSON 형식으로만 응답해.

{{
  "자사_및_계열사_이슈": [ID숫자들],
  "정부_국회": [ID숫자들],
  "경쟁사_해외이슈": [ID숫자들],
  "산업동향": [ID숫자들]
}}

[최근 7일 과거 보고서 데이터]
{recent_past_text}

[오늘 뉴스 후보 리스트]
{candidate_text}
"""

    try:
        selection_text = gemini_generate_text(
            client=client,
            prompt=prompt_selection,
            task_name="기사 선별",
        )
        raw_json_data = extract_json_object(selection_text)
        json_data = normalize_selection_json(raw_json_data, final_candidates)
        selected_count = sum(len(ids) for ids in json_data.values())

        if selected_count == 0:
            raise ValueError("Gemini가 선택한 기사 ID가 0개입니다.")

        print(f"  └ ✅ AI 선별 완료: {selected_count}개")

    except Exception as e:
        print(f"  └ ⚠️ AI 선별 실패. 기본 선별 로직으로 진행합니다. 원인: {e}")
        json_data = deterministic_selection(final_candidates)
        selected_count = sum(len(ids) for ids in json_data.values())
        print(f"  └ ✅ 기본 선별 완료: {selected_count}개")

    # STEP 3: 선별 기사만 원문 URL 디코딩 + 본문 추출
    print("\n🕵️‍♂️ [STEP 3] 최종 선별 기사만 원문 URL 변환 후 본문 전문을 추출합니다...")

    article_by_id = {int(article["id"]): article for article in raw_articles}
    final_report_data = []
    body_failed_rows = []
    post_body_duplicate_skips = []
    processed_ids = set()

    def try_process(art_id, json_key, reason="selected"):
        if art_id in processed_ids:
            return False

        processed_ids.add(art_id)
        article_info = article_by_id.get(int(art_id))
        if not article_info:
            return False

        report_item, skip_info = process_article_for_report(article_info, json_key, recent_past_items)

        if skip_info:
            skip_info["제외단계"] = reason
            post_body_duplicate_skips.append(skip_info)
            print(f"  └ 🧹 본문 확인 후 과거 중복 제외: {article_info.get('기사제목', '')[:30]}...")
            return False

        if not report_item:
            return False

        final_report_data.append(report_item)

        method = report_item.get("본문추출방식", "")
        body_len = report_item.get("본문글자수", 0)
        title_short = report_item.get("기사제목", "")[:32]

        if method == "failed" or body_len < MIN_BODY_CHARS:
            body_failed_rows.append(report_item)
            print(f"  └ ⚠️ 본문 추출 미흡: {title_short}... ({method}, {body_len}자)")
        else:
            print(f"  └ 📥 본문 추출 완료: {title_short}... ({method}, {body_len}자)")

        time.sleep(random.uniform(0.2, 0.5))
        return True

    for json_key in JSON_KEYS_ORDER:
        for art_id in json_data.get(json_key, []):
            try_process(int(art_id), json_key, reason="gemini_selected")

    # 본문 기반 중복으로 13개 미만이 되면 후보군에서 보충
    if len(final_report_data) < MIN_SELECT_COUNT:
        print(f"\n  └ 🔁 본문 확인 후 남은 기사가 {len(final_report_data)}개라 후보군에서 보충 선별합니다...")

        for article in final_candidates:
            if len(final_report_data) >= MIN_SELECT_COUNT:
                break

            art_id = int(article["id"])
            if art_id in processed_ids:
                continue

            json_key = article.get("JSON카테고리") or "산업동향"
            if json_key not in JSON_KEYS_ORDER:
                json_key = "산업동향"

            try_process(art_id, json_key, reason="replacement")

    if skipped_duplicates or post_body_duplicate_skips:
        all_skips = skipped_duplicates + post_body_duplicate_skips
        pd.DataFrame(all_skips).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")

    if body_failed_rows:
        pd.DataFrame(body_failed_rows).to_csv(OUTPUT_BODY_FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ ⚠️ 본문 추출 미흡 기사 저장: {os.path.basename(OUTPUT_BODY_FAILED_CSV)}")

    if not final_report_data:
        print("❌ 최종 보고서에 사용할 기사 데이터가 없습니다.")
        return

    # 15개 초과 방지
    final_report_data = final_report_data[:MAX_SELECT_COUNT]

    # STEP 4: 최종 브리핑 생성
    print("\n✍️ [STEP 4] Gemini AI가 past_reports 형식 그대로 최종 브리핑을 생성합니다...")

    final_input_text = ""
    for item in final_report_data:
        final_input_text += (
            f"[{item['카테고리명']}]\n"
            f"제목: {item['기사제목']}\n"
            f"언론사: {item['언론사']}\n"
            f"게시일: {item['게시일']}\n"
            f"링크: {item['링크']}\n"
            f"본문추출방식: {item['본문추출방식']}\n"
            f"본문:\n{item['본문전문'][:2200]}\n\n"
        )

    report_header = today_report_header()

    prompt_report = f"""
너는 최고 경영진에게 매일 아침 뉴스 브리핑을 제공하는 수석 전략가야.
아래 [최근 과거 보고서 예시]의 출력 형식과 문체를 그대로 따르고, [오늘 기사 데이터]만 사용해 오늘 보고서를 작성해.

[절대 규칙]
1. 첫 줄은 반드시 아래 제목으로 시작해.
   {report_header}
2. 카테고리 순서는 반드시 아래 순서로 작성해.
   ☑️ 자사 및 계열사 이슈
   ☑️ 정부/국회
   ☑️ 경쟁사/해외이슈
   ☑️ 산업동향
3. 각 기사는 과거 문서처럼 번호 이모지, 기사 제목, 링크, 괄호 안 언론사, 요약 1문단 순서로 작성해.
4. 네 생각, 인사이트, 전망, 대응 포인트, 의미 부여는 쓰지마.
5. 오직 기사 본문에 있는 객관적 사실만 요약해.
6. 각 기사 요약은 1문단으로만 작성해.
7. 모든 문장의 끝은 '~함', '~임', '~됨', '~계획임', '~예정임' 같은 문어체 종결로 맞춰.
8. 기사 본문에 없는 내용은 절대 추가하지마.
9. 언론사는 [오늘 기사 데이터]의 언론사 값을 우선 사용해.
10. 본문추출방식이 failed 또는 meta_description_short인 경우, 제공된 제목/요약/링크 범위 안에서만 보수적으로 작성해.

[최근 과거 보고서 예시]
{recent_past_text}

[오늘 기사 데이터]
{final_input_text}
"""

    try:
        final_briefing_text = gemini_generate_text(
            client=client,
            prompt=prompt_report,
            task_name="최종 브리핑 생성",
        )
        print("  └ ✅ 최종 브리핑 생성 완료")

    except Exception as e:
        print(f"  └ ⚠️ Gemini 최종 브리핑 생성 실패. 기본 보고서로 대체합니다. 원인: {e}")
        final_briefing_text = build_fallback_briefing(final_report_data)

    # STEP 5: 저장 및 출력
    print("\n" + "=" * 60)
    print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
    print("=" * 60)
    print(final_briefing_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_briefing_text)

    pd.DataFrame(final_report_data).to_csv(OUTPUT_SELECTED_CSV, index=False, encoding="utf-8-sig")

    total_duration = time.time() - total_start_time

    print("\n" + "=" * 60)
    print(f"💾 시스템 자동화 작업 완료! (총 소요 시간: {total_duration / 60:.2f}분)")
    print(f"- '{os.path.basename(OUTPUT_TXT)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_SELECTED_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_CANDIDATES_CSV)}' 저장 완료")

    if skipped_duplicates or post_body_duplicate_skips:
        print(f"- '{os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}' 저장 완료")

    if body_failed_rows:
        print(f"- '{os.path.basename(OUTPUT_BODY_FAILED_CSV)}' 저장 완료")

    print("=" * 60)


if __name__ == "__main__":
    main()
