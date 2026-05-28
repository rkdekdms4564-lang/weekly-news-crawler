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
import traceback
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
OUTPUT_RANKED_CSV = os.path.join(BASE_DIR, "google_news_ranked_candidates.csv")
OUTPUT_SKIPPED_DUP_CSV = os.path.join(BASE_DIR, "skipped_past_duplicates.csv")
OUTPUT_BODY_FAILED_CSV = os.path.join(BASE_DIR, "body_extract_failed.csv")
OUTPUT_RUN_LOG_CSV = os.path.join(BASE_DIR, "run_quality_log.csv")

# Gemini 무료 티어에서 429가 자주 나면 아래 모델을 flash-lite 계열로 바꿔도 됨.
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_ATTEMPTS = 2
ENABLE_GEMINI_SELECTION = True
ENABLE_GEMINI_REPORT = True

# Google News RSS 수집 범위
RSS_RECENT_DAYS = 1
STRICT_RSS_TIME_FILTER = True
RSS_RECENCY_HOURS = 36

# 핵심 변경점: 키워드당 최대 200개까지 시도.
# Google News RSS가 실제로 200개를 항상 주지는 않음. query variant를 여러 개 돌려 최대한 확보함.
MAX_RSS_ITEMS_PER_KEYWORD = 200
RSS_FEED_ITEM_LIMIT_PER_QUERY = 100
RSS_QUERY_VARIANTS = ["plain", "exact"]

# Gemini에 전부 넣으면 prompt가 터지므로, 수집은 크게 하고 선별 후보는 랭킹으로 압축함.
MAX_CANDIDATES_FOR_GEMINI = 200
CATEGORY_POOL_LIMIT_FOR_GEMINI = {
    "자사_및_계열사_이슈": 70,
    "정부_국회": 80,
    "경쟁사_해외이슈": 55,
    "산업동향": 25,
}

MIN_SELECT_COUNT = 13
MAX_SELECT_COUNT = 15

# past_reports.txt에서 최근 며칠 기사를 중복 판단 기준으로 쓸지
PAST_DUP_LOOKBACK_DAYS = 7

# 본문 품질 기준. 이보다 짧으면 최종 브리핑 원문으로 쓰지 않고 후보 교체함.
MIN_GOOD_BODY_CHARS = 700
MIN_ACCEPT_BODY_CHARS = 450
ALLOW_SHORT_BODY_IN_REPORT = False

# 최종 요약 prompt에 기사당 본문 몇 자까지 넣을지
MAX_BODY_CHARS_FOR_PROMPT = 2800

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

CATEGORY_TARGET = {
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


def url_domain(url):
    try:
        return urllib.parse.urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def guess_press_name_from_url(url):
    return url_domain(url)


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


def title_fingerprint(title):
    text = normalize_for_similarity(title)
    text = re.sub(r"\b(단독|종합|속보|영상|포토)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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
    "대표", "회장", "위원장", "부총리", "장관", "부처", "관계자", "전문가", "이용자", "사용자", "모델",
}

ENTITY_PATTERNS = {
    "카카오": r"카카오|카톡|Kakao",
    "카카오톡": r"카카오톡|카톡|KakaoTalk",
    "카카오모빌리티": r"카카오모빌리티|카카오\s*T",
    "카카오페이": r"카카오페이",
    "카카오뱅크": r"카카오뱅크",
    "카카오게임즈": r"카카오게임즈",
    "카카오엔터": r"카카오엔터|카카오엔터테인먼트",
    "네이버": r"네이버|NAVER",
    "두나무": r"두나무|업비트|Dunamu|Upbit",
    "SKT": r"SKT|SK텔레콤|에스케이텔레콤",
    "KT": r"\bKT\b|케이티",
    "LGU+": r"LGU\+|LG\s*U\+|LG유플러스|엘지유플러스",
    "쿠팡": r"쿠팡|Coupang",
    "토스": r"토스|비바리퍼블리카",
    "배달의민족": r"배달의민족|배민|우아한형제들",
    "구글": r"구글|Google|제미나이|Gemini",
    "오픈AI": r"오픈AI|OpenAI|챗GPT|ChatGPT",
    "앤트로픽": r"앤트로픽|Anthropic|클로드|Claude|미토스",
    "메타": r"메타|Meta|인스타그램|페이스북|왓츠앱",
    "애플": r"애플|Apple|아이폰|iPhone",
    "공정위": r"공정거래위원회|공정위",
    "금융위": r"금융위원회|금융위",
    "금감원": r"금융감독원|금감원",
    "과기정통부": r"과학기술정보통신부|과기정통부|과기부",
    "방미통위": r"방송미디어통신위원회|방미통위",
    "행안부": r"행정안전부|행안부",
}

EVENT_PATTERNS = {
    "labor_dispute": r"노조|파업|쟁의|임단협|임금교섭|성과급|RSU|조정|지노위|노동위|쟁의권|결의대회",
    "dunamu_stake": r"두나무|업비트|지분|처분|매각|주식교환|스테이블코인.*하나|하나금융",
    "mobility_sale_ipo": r"카카오모빌리티.*(매각|상장|IPO|나스닥|TPG|칼라일|우버)|우버.*카카오모빌리티",
    "kakao_pay_alipay": r"카카오페이.*(알리페이|개인정보|542억|과징금|행정소송|수사)",
    "kim_beomsoo_trial": r"김범수|SM|시세조종|항소심",
    "k_moonshot": r"K[- ]?문샷|문샷",
    "aidc_datacenter": r"AIDC|AI\s*데이터센터|국가AI컴퓨팅센터|GPU|NPU|AI컴퓨팅센터|AI 고속도로",
    "stablecoin": r"스테이블코인|디지털자산기본법|가상자산|특금법|FIU|CBDC|예금토큰",
    "map_export": r"지도\s*반출|고정밀\s*지도|구글맵|정밀지도",
    "ai_basic_law": r"AI\s*기본법|인공지능기본법|고영향\s*AI|투명성\s*표시",
    "deepfake_youth": r"딥페이크|디지털성범죄|청소년|SNS.*금지|유해정보",
    "phishing_security": r"피싱|사칭|악성코드|해킹|보안|침해사고|제로트러스트|유심|개인정보\s*유출",
    "antitrust_platform": r"공정위|공정거래위원회|온플법|플랫폼\s*규제|최혜대우|과징금|담합|다크패턴|인앱결제",
    "earnings": r"실적|영업이익|순이익|매출|적자|흑자|어닝",
    "ai_agent": r"에이전틱|AI\s*에이전트|AI탭|AI\s*브리핑|카나나|하이퍼클로바|제미나이|클로드",
    "search_ads": r"검색|광고|애드테크|AI\s*광고|광고시장|브랜드메시지",
    "government_ai": r"국가AI전략위|AI\s*전략위|AI\s*국민비서|공공\s*AI|AI\s*정부|행정\s*AI",
    "copyright_ai": r"저작권|AI\s*학습|무단\s*학습|콘텐츠\s*학습|TDM",
}

STRONG_EVENT_TAGS = {
    "labor_dispute", "dunamu_stake", "mobility_sale_ipo", "kakao_pay_alipay", "kim_beomsoo_trial",
    "k_moonshot", "aidc_datacenter", "stablecoin", "map_export", "ai_basic_law", "deepfake_youth",
    "phishing_security", "antitrust_platform", "government_ai", "copyright_ai",
}

IMPORTANT_KEYWORDS = [
    "과징금", "행정소송", "수사", "압수수색", "재판", "항소심", "판결", "제재", "조사", "공정위", "금감원",
    "금융위", "개보위", "방미통위", "과기정통부", "국회", "법안", "시행령", "입법예고", "본회의", "상임위",
    "파업", "쟁의", "노조", "성과급", "장애", "먹통", "개인정보", "유출", "해킹", "피싱", "악성코드",
    "매각", "인수", "합병", "상장", "IPO", "지분", "경영권", "실적", "영업이익", "순이익", "적자",
    "스테이블코인", "디지털자산", "AI기본법", "데이터센터", "GPU", "NPU", "AIDC", "K-문샷",
]

LOW_VALUE_TITLE_PATTERNS = [
    r"특강", r"교육\s*실시", r"마케팅\s*교육", r"시민\s*파워셀러", r"모집\s*시작", r"이벤트", r"기획전",
    r"할인", r"쿠폰", r"혜택", r"오픈\s*기념", r"브랜드\s*대상", r"수상", r"캠페인", r"체험단",
]

HIGH_VALUE_PRESS = {
    "연합뉴스": 11, "전자신문": 10, "조선일보": 10, "한국경제": 9, "서울경제": 9, "이데일리": 9,
    "뉴스1": 8, "뉴시스": 8, "머니투데이": 8, "지디넷코리아": 8, "디지털데일리": 8,
    "아이뉴스24": 7, "매일경제": 7, "파이낸셜뉴스": 7, "아시아경제": 7, "헤럴드경제": 7,
    "노컷뉴스": 6, "SBS Biz": 6, "SBS": 6, "JTBC": 6, "KBS": 6,
}


def infer_report_date(month, day, now_date=None):
    if now_date is None:
        now_date = datetime.now(KST).date()

    inferred = date(now_date.year, month, day)

    if inferred > now_date + timedelta(days=7):
        inferred = date(now_date.year - 1, month, day)

    return inferred


def parse_past_reports(text):
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
                item_text = f"{title} {summary}".strip()
                items.append({
                    "date": current_date,
                    "category": current_category,
                    "title": title,
                    "link": normalize_url(link),
                    "press": press,
                    "summary": summary,
                    "text": item_text,
                    "entities": detect_entities(item_text),
                    "event_tags": detect_event_tags(item_text),
                    "issue_terms": tokenize_for_similarity(item_text),
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


def detect_entities(text):
    found = set()
    normalized = clean_html_text(text)
    for key, pattern in ENTITY_PATTERNS.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            found.add(key)
    return found


def detect_event_tags(text):
    found = set()
    normalized = clean_html_text(text)
    for key, pattern in EVENT_PATTERNS.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            found.add(key)
    return found


def issue_key(candidate):
    text = f"{candidate.get('기사제목') or candidate.get('title') or ''} {candidate.get('본문요약') or candidate.get('summary') or candidate.get('본문전문') or ''}"
    entities = detect_entities(text)
    tags = detect_event_tags(text)
    strong_tags = tags & STRONG_EVENT_TAGS
    return entities, strong_tags


def article_similarity(candidate, past_item):
    candidate_title = candidate.get("기사제목") or candidate.get("title") or ""
    candidate_summary = candidate.get("본문요약") or candidate.get("summary") or candidate.get("본문전문") or ""
    candidate_link = normalize_url(candidate.get("링크") or candidate.get("link") or "")

    past_title = past_item.get("title", "")
    past_summary = past_item.get("summary", "")
    past_link = normalize_url(past_item.get("link", ""))

    if candidate_link and past_link and candidate_link == past_link:
        return {
            "is_duplicate": True, "reason": "same_url", "score": 1.0,
            "title_score": 1.0, "token_score": 1.0, "text_score": 1.0,
            "shared_entities": "", "shared_tags": "",
        }

    candidate_title_norm = normalize_for_similarity(candidate_title)
    past_title_norm = normalize_for_similarity(past_title)

    if candidate_title_norm and past_title_norm and candidate_title_norm == past_title_norm:
        return {
            "is_duplicate": True, "reason": "same_title", "score": 1.0,
            "title_score": 1.0, "token_score": 1.0, "text_score": 1.0,
            "shared_entities": "", "shared_tags": "",
        }

    title_seq = sequence_ratio(candidate_title, past_title)
    title_ngram = jaccard(char_ngrams(candidate_title), char_ngrams(past_title))
    title_score = max(title_seq, title_ngram)

    candidate_text = f"{candidate_title} {candidate_summary}".strip()
    past_text = f"{past_title} {past_summary}".strip()

    candidate_tokens = tokenize_for_similarity(candidate_text)
    past_tokens = past_item.get("issue_terms") or tokenize_for_similarity(past_text)

    token_score = jaccard(candidate_tokens, past_tokens)
    text_score = jaccard(char_ngrams(candidate_text), char_ngrams(past_text))

    candidate_entities = detect_entities(candidate_text)
    past_entities = past_item.get("entities") or detect_entities(past_text)
    shared_entities = candidate_entities & past_entities

    candidate_tags = detect_event_tags(candidate_text)
    past_tags = past_item.get("event_tags") or detect_event_tags(past_text)
    shared_tags = (candidate_tags & past_tags) & STRONG_EVENT_TAGS

    combined = (0.48 * title_score) + (0.30 * token_score) + (0.22 * text_score)

    is_duplicate = False
    reason = ""

    # 1) 같은 강한 사건 태그 + 같은 주요 주체면, 제목이 다소 달라도 같은 이슈로 봄.
    if shared_entities and shared_tags:
        is_duplicate = True
        reason = "same_issue_cluster"

    # 2) 제목과 토큰이 모두 비슷하면 중복.
    elif title_score >= 0.70 and token_score >= 0.18:
        is_duplicate = True
        reason = "very_similar_title"

    elif title_score >= 0.62 and token_score >= 0.28:
        is_duplicate = True
        reason = "similar_title_and_tokens"

    elif combined >= 0.56 and token_score >= 0.24 and shared_entities:
        is_duplicate = True
        reason = "similar_event"

    return {
        "is_duplicate": is_duplicate,
        "reason": reason,
        "score": round(combined, 4),
        "title_score": round(title_score, 4),
        "token_score": round(token_score, 4),
        "text_score": round(text_score, 4),
        "shared_entities": ",".join(sorted(shared_entities)),
        "shared_tags": ",".join(sorted(shared_tags)),
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


def build_recent_past_text(recent_past_items, max_chars=7000):
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

def is_quota_error(exc):
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "quota" in text.lower()


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

            # 무료 quota 초과는 재시도해도 거의 소용없으므로 즉시 fallback.
            if is_quota_error(e):
                raise e

            if attempt >= max_attempts:
                break

            delay = min(20, (2 ** attempt) + random.uniform(0, 2))

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


def normalize_selection_json(data, ranked_candidates):
    valid_ids = {int(a["id"]) for a in ranked_candidates}
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

    return enforce_selection_limits(result, ranked_candidates)


def enforce_selection_limits(selection, ranked_candidates):
    result = {key: [] for key in JSON_KEYS_ORDER}
    used = set()

    for key in JSON_KEYS_ORDER:
        max_count = CATEGORY_TARGET.get(key, 3)
        for art_id in selection.get(key, []):
            if art_id in used:
                continue
            if len(result[key]) >= max_count:
                continue
            result[key].append(art_id)
            used.add(art_id)

    total = sum(len(v) for v in result.values())

    if total < MIN_SELECT_COUNT:
        for article in ranked_candidates:
            art_id = int(article["id"])
            if art_id in used:
                continue

            json_key = article.get("JSON카테고리") or "산업동향"
            if json_key not in result:
                json_key = "산업동향"

            result[json_key].append(art_id)
            used.add(art_id)

            total = sum(len(v) for v in result.values())
            if total >= MIN_SELECT_COUNT:
                break

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


def deterministic_selection(ranked_candidates):
    result = {key: [] for key in JSON_KEYS_ORDER}
    used_ids = set()

    for json_key in JSON_KEYS_ORDER:
        max_count = CATEGORY_TARGET.get(json_key, 3)
        category_candidates = [
            a for a in ranked_candidates
            if a.get("JSON카테고리") == json_key and int(a["id"]) not in used_ids
        ]

        for article in category_candidates[:max_count]:
            art_id = int(article["id"])
            result[json_key].append(art_id)
            used_ids.add(art_id)

    return enforce_selection_limits(result, ranked_candidates)


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


def build_rss_queries(keyword):
    queries = []
    base = keyword.strip()

    for variant in RSS_QUERY_VARIANTS:
        if variant == "plain":
            q = f"{base} when:{RSS_RECENT_DAYS}d"
        elif variant == "exact":
            q = f"\"{base}\" when:{RSS_RECENT_DAYS}d"
        else:
            continue

        if q not in queries:
            queries.append(q)

    return queries


def add_article(raw_articles, seen_keys, article_id, keyword, title, link, source="", published="", summary="", collector=""):
    title = clean_html_text(title)
    link = str(link).strip() if link else ""

    if not title or not link:
        return article_id, False

    normalized = normalize_url(link)
    if not normalized:
        return article_id, False

    # RSS 단계에서는 Google News URL이 서로 달라도 제목이 같으면 거의 같은 기사로 봄.
    fp = title_fingerprint(title)
    dedup_key = f"{fp}::{source}"
    url_key = normalized

    if url_key in seen_keys or dedup_key in seen_keys:
        return article_id, False

    seen_keys.add(url_key)
    seen_keys.add(dedup_key)

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
    print(f"   └ 키워드당 최대 {MAX_RSS_ITEMS_PER_KEYWORD}개까지 시도합니다.")
    print("   └ 원문 URL 디코딩은 여기서 하지 않고, 최종 후보 본문 추출 단계에서만 적용합니다.")

    session = requests.Session()
    session.headers.update(WEB_HEADERS)

    raw_articles = []
    skipped_duplicates = []
    seen_keys = set()
    article_id = 1

    for idx, keyword in enumerate(all_keywords, 1):
        collected = 0
        skipped_old = 0
        skipped_past = 0
        skipped_seen = 0

        queries = build_rss_queries(keyword)

        for query_text in queries:
            if collected >= MAX_RSS_ITEMS_PER_KEYWORD:
                break

            encoded_query = urllib.parse.quote_plus(query_text)
            rss_url = (
                f"https://news.google.com/rss/search?q={encoded_query}"
                f"&hl=ko&gl=KR&ceid=KR:ko&num={RSS_FEED_ITEM_LIMIT_PER_QUERY}"
            )

            try:
                req = session.get(rss_url, timeout=12)
                req.raise_for_status()
                feed = feedparser.parse(req.content)

                for entry in feed.entries[:RSS_FEED_ITEM_LIMIT_PER_QUERY]:
                    if collected >= MAX_RSS_ITEMS_PER_KEYWORD:
                        break

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
                            "공유주체": sim.get("shared_entities") if sim else "",
                            "공유사건태그": sim.get("shared_tags") if sim else "",
                            "종합점수": sim.get("score") if sim else "",
                            "제목점수": sim.get("title_score") if sim else "",
                            "토큰점수": sim.get("token_score") if sim else "",
                            "본문점수": sim.get("text_score") if sim else "",
                        })
                        continue

                    before = article_id
                    article_id, added = add_article(
                        raw_articles=raw_articles,
                        seen_keys=seen_keys,
                        article_id=article_id,
                        keyword=keyword,
                        title=title,
                        link=google_link,
                        source=source,
                        published=published,
                        summary=summary,
                        collector=f"google_rss_{query_text}",
                    )

                    if added:
                        collected += 1
                    else:
                        skipped_seen += 1

            except Exception as e:
                print(f"\n⚠️ [GOOGLE RSS {idx}/{len(all_keywords)}] '{keyword}' 쿼리 실패: {query_text} / {e}")

            time.sleep(random.uniform(0.03, 0.10))

        print(
            f"▶ [GOOGLE RSS {idx}/{len(all_keywords)}] "
            f"'{keyword}' 수집 {collected}개 / 시간제외 {skipped_old}개 / "
            f"과거중복제외 {skipped_past}개 / 중복URL제외 {skipped_seen}개        ",
            end="\r"
        )

    print("")
    return raw_articles, skipped_duplicates


# ==========================================
# 6. 후보 랭킹 및 압축
# ==========================================

def press_score(source):
    source = clean_html_text(source)
    if not source:
        return 0

    for key, score in HIGH_VALUE_PRESS.items():
        if key.lower() in source.lower():
            return score

    # 도메인 언론사/블로그형 매체는 낮은 점수
    if ".com" in source or ".co.kr" in source or ".kr" in source:
        return 2

    return 4


def recency_score(published):
    dt = parse_pubdate_to_kst(published)
    if not dt:
        return 5

    now = datetime.now(KST)
    hours = max(0.0, (now - dt).total_seconds() / 3600)

    if hours <= 6:
        return 12
    if hours <= 12:
        return 10
    if hours <= 24:
        return 8
    if hours <= 36:
        return 5
    return 1


def rank_score_article(article):
    title = article.get("기사제목", "")
    summary = article.get("본문요약", "")
    text = f"{title} {summary}"
    json_key = article.get("JSON카테고리", "")

    score = 0.0
    score += recency_score(article.get("게시일", ""))
    score += press_score(article.get("언론사", ""))

    # 중요한 키워드가 있으면 가산
    for kw in IMPORTANT_KEYWORDS:
        if kw in text:
            score += 4

    # 사건 태그가 있으면 가산
    event_tags = detect_event_tags(text)
    score += min(16, len(event_tags) * 4)

    # 자사/정부/경쟁사별 약간의 가중치
    if json_key == "자사_및_계열사_이슈":
        score += 8
    elif json_key == "정부_국회":
        score += 6
    elif json_key == "경쟁사_해외이슈":
        score += 4
    elif json_key == "산업동향":
        score += 2

    # 너무 가벼운 기사 감점
    for pattern in LOW_VALUE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            score -= 20

    # 제목만 너무 짧거나 맥락 없는 기사 감점
    if len(title) < 12:
        score -= 5

    # Google RSS summary가 제목+언론사만 들어오는 경우가 많으므로 summary 길이는 큰 가중치로 쓰지 않음.
    article["랭킹점수"] = round(score, 3)
    article["사건태그"] = ",".join(sorted(event_tags))
    article["주요주체"] = ",".join(sorted(detect_entities(text)))
    return score


def rank_and_trim_candidates(raw_articles):
    for article in raw_articles:
        rank_score_article(article)

    ranked_all = sorted(raw_articles, key=lambda x: (x.get("랭킹점수", 0), x.get("게시일", "")), reverse=True)

    # Gemini 후보는 카테고리별로 압축하되, 전체 후보 CSV는 전부 저장.
    selected_for_gemini = []
    used_ids = set()

    for json_key in JSON_KEYS_ORDER:
        limit = CATEGORY_POOL_LIMIT_FOR_GEMINI.get(json_key, 50)
        bucket = [a for a in ranked_all if a.get("JSON카테고리") == json_key]
        for article in bucket[:limit]:
            if len(selected_for_gemini) >= MAX_CANDIDATES_FOR_GEMINI:
                break
            aid = int(article["id"])
            if aid in used_ids:
                continue
            selected_for_gemini.append(article)
            used_ids.add(aid)

    if len(selected_for_gemini) < MAX_CANDIDATES_FOR_GEMINI:
        for article in ranked_all:
            if len(selected_for_gemini) >= MAX_CANDIDATES_FOR_GEMINI:
                break
            aid = int(article["id"])
            if aid in used_ids:
                continue
            selected_for_gemini.append(article)
            used_ids.add(aid)

    selected_for_gemini = sorted(selected_for_gemini, key=lambda x: x.get("랭킹점수", 0), reverse=True)
    return ranked_all, selected_for_gemini


# ==========================================
# 7. 선별 기사 URL 디코딩 및 본문 추출
# ==========================================

def decode_google_news_url(google_news_url):
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


def candidate_urls_for_extraction(url):
    """
    일부 언론사는 일반 URL보다 AMP/출력 URL에서 본문이 더 잘 추출됩니다.
    최종 선별 기사 13~15개에만 적용하므로 추가 요청 비용을 감수합니다.
    """
    url = normalize_url(url)
    if not url:
        return []

    urls = [url]
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower().replace("www.", "")

    def add(u):
        u = normalize_url(u)
        if u and u not in urls:
            urls.append(u)

    if "chosun.com" in domain and "outputType=amp" not in url:
        sep = "&" if parsed.query else "?"
        add(url + sep + "outputType=amp")

    if "yna.co.kr" in domain and "/amp/" in url:
        add(url.replace("/amp/view/", "/view/"))

    if "news.einfomax.co.kr" in domain and "articleViewAmp" in url:
        add(url.replace("articleViewAmp", "articleView"))

    if "v.daum.net" in domain and "output=amp" not in url:
        sep = "&" if parsed.query else "?"
        add(url + sep + "output=amp")

    if "news1.kr" in domain and "/amp/" not in url:
        add(url.rstrip("/") + "/amp")

    return urls


def remove_unwanted_tags(soup):
    for tag in soup([
        "script", "style", "noscript", "header", "footer", "nav", "aside",
        "iframe", "form", "button", "figure", "svg", "canvas"
    ]):
        tag.decompose()

    return soup


def extract_from_json_ld(soup):
    texts = []
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = script.string or script.get_text(" ") or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        stack = data if isinstance(data, list) else [data]
        while stack:
            obj = stack.pop(0)
            if isinstance(obj, list):
                stack.extend(obj)
                continue
            if not isinstance(obj, dict):
                continue

            body = obj.get("articleBody") or obj.get("description")
            if body:
                body = clean_html_text(body)
                if len(body) >= 120:
                    texts.append(body)

            graph = obj.get("@graph")
            if isinstance(graph, list):
                stack.extend(graph)

    return max(texts, key=len) if texts else ""


def clean_paragraphs(paragraphs):
    cleaned = []
    seen = set()

    for text in paragraphs:
        text = clean_html_text(text)
        if len(text) < 25:
            continue
        if re.search(r"무단전재|재배포\s*금지|Copyright|저작권자|구독|좋아요|팔로우", text, flags=re.IGNORECASE):
            continue
        if re.fullmatch(r"[가-힣]{2,5}\s*기자", text):
            continue
        if text in seen:
            continue
        seen.add(text)
        cleaned.append(text)

    return cleaned


def extract_text_from_bs4(html_text):
    soup = BeautifulSoup(html_text, "html.parser")

    json_ld_text = extract_from_json_ld(soup)

    soup = remove_unwanted_tags(soup)

    selectors = [
        "article",
        "[itemprop='articleBody']",
        "#articleBody", "#article_body", "#articeBody", "#article-view-content-div",
        "#newsEndContents", "#dic_area", "#CmAdContent",
        ".article_body", ".articleBody", ".article-body", ".article_body_wrap",
        ".article_view", ".article-view", ".article_view_content", ".article-content",
        ".news_view", ".news-view", ".news_content", ".news-content", ".newsct_article",
        ".view_cont", ".view-content", ".view_body", ".read_txt", ".article_txt",
        ".story-news", ".story-news-article", ".articleCont", ".article_body_view",
        ".contents", ".content", ".view",
    ]

    candidate_texts = []

    if json_ld_text:
        candidate_texts.append(json_ld_text)

    for selector in selectors:
        for node in soup.select(selector):
            paragraphs = []
            p_tags = node.find_all(["p", "div", "span"], recursive=True)

            if not p_tags:
                text = clean_html_text(node.get_text(" "))
                if len(text) >= 120:
                    candidate_texts.append(text)
                continue

            for p in p_tags:
                text = clean_html_text(p.get_text(" "))
                paragraphs.append(text)

            joined = "\n".join(clean_paragraphs(paragraphs)).strip()
            if len(joined) >= 120:
                candidate_texts.append(joined)

    # 최후 fallback: p 태그 전체
    paragraphs = [p.get_text(" ") for p in soup.find_all("p")]
    joined = "\n".join(clean_paragraphs(paragraphs)).strip()
    if len(joined) >= 120:
        candidate_texts.append(joined)

    if not candidate_texts:
        return ""

    # 가장 긴 후보를 기본으로 하되, 지나치게 메뉴/댓글까지 붙은 것은 clean_extracted_body가 잘라냄.
    return max(candidate_texts, key=len)


def cut_after_noise_markers(text):
    markers = [
        "관련 키워드", "관련기사", "관련 기사", "많이 본 뉴스", "함께 보면 좋은", "당신이 좋아할 만한",
        "오늘의 주요뉴스", "주요 뉴스", "인기뉴스", "추천뉴스", "기자의 다른기사", "다른기사 보기",
        "카카오톡에 공유", "페이스북에 공유", "트위터에 공유", "구독신청",
    ]

    best = text
    for marker in markers:
        idx = best.find(marker)
        if idx >= 0 and idx > 250:
            best = best[:idx].strip()
    return best


def clean_extracted_body(text):
    text = clean_html_text(text)

    if not text:
        return ""

    text = cut_after_noise_markers(text)

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


def get_html(url):
    try:
        r = requests.get(
            url,
            headers=WEB_HEADERS,
            timeout=15,
            allow_redirects=True,
        )
        r.raise_for_status()

        if r.encoding is None or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding

        return r.text or "", normalize_url(r.url)
    except Exception:
        return "", url


def extract_meta_description(html_text):
    if not html_text:
        return ""
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

        return max(meta_candidates, key=len) if meta_candidates else ""
    except Exception:
        return ""


def extract_article_body(url):
    if not url:
        return "", "none", url

    all_candidates = []
    last_url = url
    last_html = ""

    for try_url in candidate_urls_for_extraction(url):
        final_url = try_url

        # HTML을 먼저 확보해 trafilatura와 selector가 같은 소스를 쓰도록 함.
        html_text, fetched_url = get_html(try_url)
        last_html = html_text or last_html
        if fetched_url:
            final_url = fetched_url
            last_url = fetched_url

        extraction_candidates = []

        # 1차: trafilatura. 언론사 본문 추출에 강함.
        if html_text and trafilatura is not None:
            try:
                extracted = trafilatura.extract(
                    html_text,
                    url=final_url,
                    include_comments=False,
                    include_tables=False,
                    include_images=False,
                    no_fallback=False,
                    favor_precision=False,
                    target_language="ko",
                )
                text = clean_extracted_body(extracted)
                if text:
                    extraction_candidates.append((text, "trafilatura", final_url))
            except Exception:
                pass

        # 2차: BS4 selector + JSON-LD.
        if html_text:
            try:
                text = clean_extracted_body(extract_text_from_bs4(html_text))
                if text:
                    extraction_candidates.append((text, "bs4_selector", final_url))
            except Exception:
                pass

        # 3차: newspaper3k.
        try:
            config = Config()
            config.browser_user_agent = WEB_HEADERS["User-Agent"]
            config.request_timeout = 15
            config.fetch_images = False

            article = Article(url=final_url, language="ko", config=config)
            article.download()
            article.parse()

            text = clean_extracted_body(article.text)
            if text:
                extraction_candidates.append((text, "newspaper3k", final_url))
        except Exception:
            pass

        for text, method, used_url in extraction_candidates:
            score, reason = body_quality_score("", text, method)
            all_candidates.append((score, len(text), text, method, reason, used_url))

    if all_candidates:
        all_candidates.sort(reverse=True, key=lambda x: (x[0], x[1]))
        score, length, text, method, reason, used_url = all_candidates[0]
        return text, method, used_url or last_url

    # 4차: meta description은 본문이 아니므로 실패로 돌려보내되, 진단용으로 method 표시.
    meta_text = clean_extracted_body(extract_meta_description(last_html))
    if meta_text:
        return meta_text, "meta_description_short", last_url

    return "", "failed", last_url


def split_sentences(text):
    text = clean_html_text(text)
    if not text:
        return []

    # 한국어 기사 문장 기준 대략 분리
    pieces = re.split(r"(?<=[.!?다요임음됨함])\s+", text)
    sentences = []
    for p in pieces:
        p = clean_html_text(p)
        if len(p) < 20:
            continue
        if re.search(r"관련 키워드|관련 기사|무단전재|재배포 금지|Copyright", p, flags=re.IGNORECASE):
            continue
        sentences.append(p)
    return sentences


def body_quality_score(title, body, method):
    body = clean_html_text(body)
    title = clean_html_text(title)

    if not body:
        return 0, "empty"

    length = len(body)
    score = min(50, length / 30)

    if method in {"trafilatura", "bs4_selector", "newspaper3k"}:
        score += 20
    if method == "meta_description_short":
        score -= 30
    if method == "failed":
        score -= 50

    sentences = split_sentences(body)
    if len(sentences) >= 3:
        score += 15
    elif len(sentences) == 2:
        score += 5
    else:
        score -= 15

    if title:
        start = body[:max(20, len(title) + 10)]
        title_ratio = difflib.SequenceMatcher(None, normalize_for_similarity(title), normalize_for_similarity(start)).ratio()
        if title_ratio > 0.88 and length < 600:
            score -= 30

    bad_markers = [
        "이 누리집은 대한민국 공식 전자정부", "내려받기", "바로보기 내려받기", "본문 추출 실패",
        "관련 키워드", "관련 기사", "배달 업계 1위", "뉴스레터", "로그인",
    ]
    for marker in bad_markers:
        if marker in body:
            score -= 12

    if length >= MIN_GOOD_BODY_CHARS:
        score += 20
    elif length >= MIN_ACCEPT_BODY_CHARS:
        score += 5
    else:
        score -= 25

    reason = "good"
    if method == "meta_description_short":
        reason = "meta_description_not_full_body"
    elif length < MIN_ACCEPT_BODY_CHARS:
        reason = "too_short"
    elif len(sentences) < 2:
        reason = "not_enough_sentences"
    elif score < 45:
        reason = "low_quality"

    return round(score, 2), reason


def is_body_good(title, body, method):
    score, reason = body_quality_score(title, body, method)

    if method in {"failed", "meta_description_short"}:
        return False, reason, score

    if len(clean_html_text(body)) >= MIN_GOOD_BODY_CHARS and score >= 45:
        return True, "good", score

    if ALLOW_SHORT_BODY_IN_REPORT and len(clean_html_text(body)) >= MIN_ACCEPT_BODY_CHARS and score >= 40:
        return True, "acceptable_short", score

    return False, reason, score


def process_article_for_report(article_info, json_key, recent_past_items):
    original_link = article_info.get("링크", "")

    real_url = decode_google_news_url(original_link)
    real_url = normalize_url(real_url)

    body_text, extract_method, fetched_url = extract_article_body(real_url)
    if fetched_url:
        real_url = normalize_url(fetched_url)

    body_text = body_text.strip()
    good, quality_reason, quality_score = is_body_good(article_info.get("기사제목", ""), body_text, extract_method)

    # 본문 추출 후 한 번 더 과거 7일과 중복 비교. 이때는 본문 일부를 같이 사용.
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
            "공유주체": sim.get("shared_entities") if sim else "",
            "공유사건태그": sim.get("shared_tags") if sim else "",
            "종합점수": sim.get("score") if sim else "",
            "제목점수": sim.get("title_score") if sim else "",
            "토큰점수": sim.get("token_score") if sim else "",
            "본문점수": sim.get("text_score") if sim else "",
        }
        return None, skip_info, None

    report_item = {
        "카테고리": json_key,
        "카테고리명": JSON_KEY_TO_DISPLAY.get(json_key, json_key),
        "검색어": article_info.get("검색어", ""),
        "기사제목": article_info.get("기사제목", ""),
        "언론사": article_info.get("언론사", "") or guess_press_name_from_url(real_url),
        "게시일": article_info.get("게시일", ""),
        "RSS요약": article_info.get("본문요약", ""),
        "본문전문": body_text,
        "본문글자수": len(body_text),
        "본문추출방식": extract_method,
        "본문품질점수": quality_score,
        "본문품질사유": quality_reason,
        "본문사용가능": good,
        "원래RSS링크": original_link,
        "링크": real_url,
        "랭킹점수": article_info.get("랭킹점수", ""),
        "사건태그": article_info.get("사건태그", ""),
    }

    if not good:
        return None, None, report_item

    return report_item, None, None


# ==========================================
# 8. 최종 브리핑 fallback 요약
# ==========================================

def convert_to_report_ending(sentence):
    s = clean_html_text(sentence).rstrip(". ")

    replacements = [
        (r"했다$", "했음"), (r"하였다$", "했음"), (r"밝혔다$", "밝힘"), (r"전했다$", "전함"),
        (r"설명했다$", "설명함"), (r"강조했다$", "강조함"), (r"말했다$", "밝힘"),
        (r"나섰다$", "나섰음"), (r"착수했다$", "착수함"), (r"시작했다$", "시작함"),
        (r"진행했다$", "진행함"), (r"개최했다$", "개최함"), (r"체결했다$", "체결함"),
        (r"공개했다$", "공개함"), (r"도입했다$", "도입함"), (r"추진한다$", "추진함"),
        (r"예정이다$", "예정임"), (r"계획이다$", "계획임"), (r"전망이다$", "전망임"),
        (r"상황이다$", "상황임"), (r"상태다$", "상태임"), (r"수준이다$", "수준임"),
        (r"됐다$", "됐음"), (r"되었다$", "됐음"), (r"된다$", "됨"),
        (r"있다$", "있음"), (r"없다$", "없음"),
        (r"이다$", "임"), (r"다$", "음"),
    ]

    for pattern, repl in replacements:
        if re.search(pattern, s):
            s = re.sub(pattern, repl, s)
            break
    else:
        if not s.endswith(("함", "임", "됨", "음", "계획임", "예정임")):
            s += "임"

    return s + "."


def summarize_body_locally(title, body, max_chars=330):
    title_terms = tokenize_for_similarity(title)
    sentences = split_sentences(body)

    if not sentences:
        return "본문 추출 품질이 낮아 원문 링크 확인이 필요함."

    scored = []
    for idx, s in enumerate(sentences[:12]):
        tokens = tokenize_for_similarity(s)
        score = 0
        score += max(0, 8 - idx)  # 앞 문장 우선
        score += len(tokens & title_terms) * 3
        for kw in IMPORTANT_KEYWORDS:
            if kw in s:
                score += 3
        if re.search(r"\d", s):
            score += 2
        scored.append((score, idx, s))

    scored.sort(key=lambda x: (-x[0], x[1]))
    chosen = sorted(scored[:2], key=lambda x: x[1])

    summary_sentences = []
    total = 0
    for _, _, s in chosen:
        converted = convert_to_report_ending(s)
        if total + len(converted) > max_chars and summary_sentences:
            continue
        summary_sentences.append(converted)
        total += len(converted)

    if not summary_sentences:
        summary_sentences.append(convert_to_report_ending(sentences[0][:max_chars]))

    return " ".join(summary_sentences).strip()


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
            body = item.get("본문전문", "")

            lines.append(f"{num} {title}")
            lines.append(link)
            lines.append(f"({press})")
            lines.append(summarize_body_locally(title, body))
            lines.append("")

    return "\n".join(lines).strip()


# ==========================================
# 9. 메인 실행
# ==========================================

def main():
    total_start_time = time.time()
    run_log = []

    # Gemini 초기화
    client = None
    if ENABLE_GEMINI_SELECTION or ENABLE_GEMINI_REPORT:
        try:
            with open(SECRET_PATH, "r", encoding="utf-8") as f:
                google_api_key = f.read().strip()

            if not google_api_key:
                raise ValueError("secret.txt가 비어 있습니다.")

            client = genai.Client(api_key=google_api_key)

        except Exception as e:
            print("⚠️ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다. Gemini 없이 로컬 선별/요약으로 진행합니다.")
            print(f"   원인: {e}")
            client = None

    # past_reports 로드 및 최근 7일 메모리 생성
    past_reports_content, all_past_items, recent_past_items = load_past_reports()
    recent_past_text = build_recent_past_text(recent_past_items, max_chars=9000)

    # STEP 1: Google RSS만 사용
    raw_articles, skipped_duplicates = collect_with_google_rss(recent_past_items)

    print(f"\n  └ ✅ 총 {len(raw_articles)}개의 RSS 후보 확보 완료")
    print(f"  └ 🧹 최근 {PAST_DUP_LOOKBACK_DAYS}일 과거 보고서와 유사해 제외한 후보: {len(skipped_duplicates)}개")

    if not raw_articles:
        print("❌ 수집된 기사가 없습니다. Google News RSS 접속 또는 키워드/기간 설정을 확인하세요.")
        return

    # STEP 1.5: 랭킹 및 Gemini 후보 압축
    ranked_all, ranked_candidates = rank_and_trim_candidates(raw_articles)

    pd.DataFrame(ranked_all).to_csv(OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(ranked_candidates).to_csv(OUTPUT_RANKED_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 전체 후보 기사 저장: {os.path.basename(OUTPUT_CANDIDATES_CSV)}")
    print(f"  └ 💾 AI/본문 추출 후보 랭킹 저장: {os.path.basename(OUTPUT_RANKED_CSV)} ({len(ranked_candidates)}개)")

    if skipped_duplicates:
        pd.DataFrame(skipped_duplicates).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 과거 중복 제외 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")

    # STEP 2: Gemini AI 선별. 실패하면 로컬 랭킹 선별.
    print("\n🧠 [STEP 2] 후보 랭킹 + Gemini가 과거 7일 중복 이슈를 제외하고 핵심 기사를 선별합니다...")
    print(f"   └ RSS 전체 {len(raw_articles)}개 중 랭킹 상위 {len(ranked_candidates)}개만 Gemini 후보로 전달합니다.")

    candidate_text = ""
    for article in ranked_candidates:
        candidate_text += (
            f"[{article['id']}] "
            f"점수: {article.get('랭킹점수', '')} / "
            f"카테고리: {article.get('원카테고리', '')} / "
            f"검색어: {article.get('검색어', '')} / "
            f"제목: {article.get('기사제목', '')} / "
            f"언론사: {article.get('언론사', '')} / "
            f"게시일: {article.get('게시일', '')} / "
            f"사건태그: {article.get('사건태그', '')}\n"
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
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 선별 비활성화 또는 client 없음")

        selection_text = gemini_generate_text(
            client=client,
            prompt=prompt_selection,
            task_name="기사 선별",
        )
        raw_json_data = extract_json_object(selection_text)
        json_data = normalize_selection_json(raw_json_data, ranked_candidates)
        selected_count = sum(len(ids) for ids in json_data.values())

        if selected_count == 0:
            raise ValueError("Gemini가 선택한 기사 ID가 0개입니다.")

        print(f"  └ ✅ AI 선별 완료: {selected_count}개")

    except Exception as e:
        print(f"  └ ⚠️ AI 선별 실패. 로컬 랭킹 선별로 진행합니다. 원인: {e}")
        json_data = deterministic_selection(ranked_candidates)
        selected_count = sum(len(ids) for ids in json_data.values())
        print(f"  └ ✅ 로컬 랭킹 선별 완료: {selected_count}개")

    # STEP 3: 선별 기사만 원문 URL 디코딩 + 본문 추출.
    # 본문 품질이 낮으면 최종 기사에서 제외하고 랭킹 후보에서 자동 보충.
    print("\n🕵️‍♂️ [STEP 3] 선별/보충 후보만 원문 URL 변환 후 본문 전문을 추출합니다...")
    print(f"   └ 본문 {MIN_GOOD_BODY_CHARS}자 이상 + 품질 통과 기사만 최종 보고서에 사용합니다.")

    article_by_id = {int(article["id"]): article for article in raw_articles}
    final_report_data = []
    body_failed_rows = []
    post_body_duplicate_skips = []
    processed_ids = set()

    def category_count(json_key):
        return len([x for x in final_report_data if x.get("카테고리") == json_key])

    def total_count():
        return len(final_report_data)

    def try_process(art_id, json_key, reason="selected"):
        if art_id in processed_ids:
            return False

        processed_ids.add(art_id)
        article_info = article_by_id.get(int(art_id))
        if not article_info:
            return False

        report_item, skip_info, failed_item = process_article_for_report(article_info, json_key, recent_past_items)

        if skip_info:
            skip_info["제외단계"] = reason
            post_body_duplicate_skips.append(skip_info)
            print(f"  └ 🧹 본문 확인 후 과거 중복 제외: {article_info.get('기사제목', '')[:40]}...")
            return False

        if failed_item:
            body_failed_rows.append(failed_item)
            print(
                f"  └ ⚠️ 본문 품질 미달로 교체: {failed_item.get('기사제목', '')[:40]}... "
                f"({failed_item.get('본문추출방식')}, {failed_item.get('본문글자수')}자, "
                f"{failed_item.get('본문품질사유')})"
            )
            return False

        if not report_item:
            return False

        final_report_data.append(report_item)
        print(
            f"  └ 📥 본문 추출 완료: {report_item.get('기사제목', '')[:40]}... "
            f"({report_item.get('본문추출방식')}, {report_item.get('본문글자수')}자, "
            f"품질 {report_item.get('본문품질점수')})"
        )
        time.sleep(random.uniform(0.15, 0.35))
        return True

    # 3-1. Gemini/로컬이 고른 기사 먼저 처리
    for json_key in JSON_KEYS_ORDER:
        for art_id in json_data.get(json_key, []):
            if total_count() >= MAX_SELECT_COUNT:
                break
            try_process(int(art_id), json_key, reason="selected")

    # 3-2. 카테고리별 목표치가 부족하면 같은 카테고리 랭킹 후보에서 보충
    print("\n  └ 🔁 본문 미달/과거중복 제외분을 랭킹 후보에서 자동 보충합니다...")

    for json_key in JSON_KEYS_ORDER:
        target = CATEGORY_TARGET.get(json_key, 3)
        bucket = [a for a in ranked_all if a.get("JSON카테고리") == json_key]

        for article in bucket:
            if category_count(json_key) >= target:
                break
            if total_count() >= MAX_SELECT_COUNT:
                break

            art_id = int(article["id"])
            if art_id in processed_ids:
                continue

            try_process(art_id, json_key, reason="category_replacement")

    # 3-3. 그래도 총 13개 미만이면 전체 랭킹 후보에서 보충
    if total_count() < MIN_SELECT_COUNT:
        print(f"\n  └ 🔁 아직 {total_count()}개라 전체 랭킹 후보에서 추가 보충합니다...")
        for article in ranked_all:
            if total_count() >= MIN_SELECT_COUNT:
                break
            if total_count() >= MAX_SELECT_COUNT:
                break

            art_id = int(article["id"])
            if art_id in processed_ids:
                continue

            json_key = article.get("JSON카테고리") or "산업동향"
            if json_key not in JSON_KEYS_ORDER:
                json_key = "산업동향"

            try_process(art_id, json_key, reason="global_replacement")

    if skipped_duplicates or post_body_duplicate_skips:
        all_skips = skipped_duplicates + post_body_duplicate_skips
        pd.DataFrame(all_skips).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 과거 중복 제외 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")

    if body_failed_rows:
        pd.DataFrame(body_failed_rows).to_csv(OUTPUT_BODY_FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ ⚠️ 본문 품질 미달 기사 저장: {os.path.basename(OUTPUT_BODY_FAILED_CSV)}")

    if not final_report_data:
        print("❌ 최종 보고서에 사용할 본문 품질 통과 기사가 없습니다.")
        return

    final_report_data = final_report_data[:MAX_SELECT_COUNT]

    # STEP 4: 최종 브리핑 생성
    print("\n✍️ [STEP 4] 본문 전문 기반으로 past_reports 형식의 최종 브리핑을 생성합니다...")

    final_input_text = ""
    for item in final_report_data:
        final_input_text += (
            f"[{item['카테고리명']}]\n"
            f"제목: {item['기사제목']}\n"
            f"언론사: {item['언론사']}\n"
            f"게시일: {item['게시일']}\n"
            f"링크: {item['링크']}\n"
            f"본문글자수: {item['본문글자수']}\n"
            f"본문:\n{item['본문전문'][:MAX_BODY_CHARS_FOR_PROMPT]}\n\n"
        )

    report_header = today_report_header()

    prompt_report = f"""
너는 최고 경영진에게 매일 아침 뉴스 브리핑을 제공하는 수석 전략가야.
아래 [최근 과거 보고서 예시]의 출력 형식과 문체를 그대로 따르고, [오늘 기사 데이터]의 본문만 사용해 오늘 보고서를 작성해.

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

[최근 과거 보고서 예시]
{recent_past_text}

[오늘 기사 데이터]
{final_input_text}
"""

    try:
        if not client or not ENABLE_GEMINI_REPORT:
            raise RuntimeError("Gemini 최종 브리핑 비활성화 또는 client 없음")

        final_briefing_text = gemini_generate_text(
            client=client,
            prompt=prompt_report,
            task_name="최종 브리핑 생성",
        )
        print("  └ ✅ Gemini 최종 브리핑 생성 완료")

    except Exception as e:
        print(f"  └ ⚠️ Gemini 최종 브리핑 생성 실패. 로컬 본문 요약 보고서로 대체합니다. 원인: {e}")
        final_briefing_text = build_fallback_briefing(final_report_data)

    # STEP 5: 저장 및 출력
    print("\n" + "=" * 60)
    print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
    print("=" * 60)
    print(final_briefing_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_briefing_text)

    pd.DataFrame(final_report_data).to_csv(OUTPUT_SELECTED_CSV, index=False, encoding="utf-8-sig")

    run_log.append({
        "전체_RSS_후보": len(raw_articles),
        "Gemini_후보": len(ranked_candidates),
        "과거중복_제외": len(skipped_duplicates) + len(post_body_duplicate_skips),
        "본문품질미달_교체": len(body_failed_rows),
        "최종기사수": len(final_report_data),
        "실행분": round((time.time() - total_start_time) / 60, 2),
    })
    pd.DataFrame(run_log).to_csv(OUTPUT_RUN_LOG_CSV, index=False, encoding="utf-8-sig")

    total_duration = time.time() - total_start_time

    print("\n" + "=" * 60)
    print(f"💾 시스템 자동화 작업 완료! (총 소요 시간: {total_duration / 60:.2f}분)")
    print(f"- '{os.path.basename(OUTPUT_TXT)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_SELECTED_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_CANDIDATES_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_RANKED_CSV)}' 저장 완료")

    if skipped_duplicates or post_body_duplicate_skips:
        print(f"- '{os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}' 저장 완료")

    if body_failed_rows:
        print(f"- '{os.path.basename(OUTPUT_BODY_FAILED_CSV)}' 저장 완료")

    print(f"- '{os.path.basename(OUTPUT_RUN_LOG_CSV)}' 저장 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
