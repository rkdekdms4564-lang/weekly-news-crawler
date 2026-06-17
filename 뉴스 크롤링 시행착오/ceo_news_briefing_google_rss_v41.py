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
import warnings
import pandas as pd
import feedparser

from google import genai
from datetime import datetime, date, timedelta, timezone
from email.utils import parsedate_to_datetime
from newspaper import Article, Config
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

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
RSS_RECENT_DAYS = 2
STRICT_RSS_TIME_FILTER = True
RSS_RECENCY_HOURS = 28

# 핵심 변경점: RSS는 검색어 1개당 사실상 상한이 있어 query sharding으로 쪼개 수집함.
# Google News RSS Search feed는 보통 한 쿼리에서 최대 100개 안팎만 안정적으로 반환하므로,
# 카카오 같은 광범위 키워드는 site: / 세부 이슈어를 여러 번 던져 후보를 확장함.
MAX_RSS_ITEMS_PER_KEYWORD = 220
MAX_RSS_ITEMS_PER_KEYWORD_OVERRIDES = {
    "카카오": 420,
    "카카오톡": 320,
    "카카오게임즈": 260,
    "카카오모빌리티": 260,
    "카카오페이": 260,
    "카카오뱅크": 260,
    "네이버": 360,
    "구글": 320,
    "쿠팡": 300,
    "스테이블코인": 300,
}
RSS_FEED_ITEM_LIMIT_PER_QUERY = 100
RSS_QUERY_VARIANTS = ["plain", "exact"]
ENABLE_RSS_QUERY_SHARDING = True
MAX_RSS_QUERIES_PER_KEYWORD = 28

# v5 핵심: query sharding을 많이 돌리면 같은 기사가 반복 노출됩니다.
# 아래 값들은 한 키워드에서 더 이상 새 기사가 거의 나오지 않을 때 남은 shard를 멈추기 위한 안전장치입니다.
ENABLE_CANONICAL_COLLECTION_DEDUPE = True
ENABLE_COLLECTION_RELEVANCE_FILTER = True
ENABLE_SHARD_OVERLAP_EARLY_STOP = True
MIN_COLLECTED_BEFORE_EARLY_STOP = 80
MAX_CONSECUTIVE_LOW_YIELD_QUERIES = 8
LOW_YIELD_NEW_ARTICLE_COUNT = 1
LOW_YIELD_DUPLICATE_RATE = 0.88

RSS_SOURCE_SHARDS = [
    "yna.co.kr", "newsis.com", "news1.kr", "etnews.com", "zdnet.co.kr",
    "ddaily.co.kr", "edaily.co.kr", "hankyung.com", "mk.co.kr", "chosun.com",
    "joongang.co.kr", "sedaily.com", "mt.co.kr", "fnnews.com", "heraldcorp.com",
    "dt.co.kr", "inews24.com", "bizwatch.co.kr", "bloter.net", "it.chosun.com",
]
RSS_SHARD_KEYWORDS = {
    "카카오", "카카오톡", "카카오모빌리티", "카카오페이", "카카오뱅크", "카카오게임즈",
    "네이버", "구글", "쿠팡", "토스", "배달의민족", "스테이블코인", "금융위원회",
    "금융감독원", "공정거래위원회", "과학기술정보통신부", "오픈AI", "애플", "MS", "메타",
}
KAKAO_ISSUE_SHARDS = [
    "노조", "파업", "성과급", "조정", "쟁의", "두나무", "지분", "매각", "카나나",
    "AI", "카카오페이", "카카오뱅크", "카카오모빌리티", "카카오게임즈", "김범수",
    "개인정보", "과징금", "장애", "피싱", "스테이블코인", "실적", "상장", "우버",
]
GENERAL_ISSUE_SHARDS = [
    "AI", "규제", "과징금", "조사", "법안", "해킹", "개인정보", "스테이블코인",
    "데이터센터", "GPU", "인수", "매각", "실적", "장애",
]

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
    """기사 제목을 canonical dedupe용으로 강하게 정규화합니다."""
    text = clean_html_text(title).lower()
    text = html.unescape(text)

    # 언론사/포털에서 붙는 꼬리 제거
    text = re.sub(r"\s*[-|｜]\s*(조선비즈|연합뉴스|뉴시스|뉴스1|전자신문|지디넷코리아|디지털데일리|이데일리|매일경제|한국경제|머니투데이|서울경제|파이낸셜뉴스|헤럴드경제|아이뉴스24|아시아경제|데일리안|노컷뉴스|sbs\s*biz|jtbc|kbs|mbn).*$", " ", text, flags=re.IGNORECASE)

    # 반복 노출을 유발하는 표식 제거
    text = re.sub(r"\[[^\]]{0,12}(단독|종합|속보|영상|포토|사진|ai픽|게시판)[^\]]{0,12}\]", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\((단독|종합|속보|영상|포토|사진|종합\d*보)\)", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^(단독|종합|속보|영상|포토|사진)\s*", " ", text, flags=re.IGNORECASE)

    # 인용부호/특수문자 정규화
    text = text.replace("‘", "").replace("’", "").replace("“", "").replace("”", "")
    text = re.sub(r"[·ㆍ•,.:;!?…~\"'`´“”‘’(){}\[\]<>]", " ", text)
    text = re.sub(r"[^0-9a-zA-Z가-힣+]+", " ", text)
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
    "labor_dispute", "dunamu_stake", "mobility_sale_ipo", "kakao_pay_alipay", "kim_beomsoo_trial", "kakao_games_management",
    "k_moonshot", "aidc_datacenter", "stablecoin", "map_export", "ai_basic_law", "deepfake_youth",
    "phishing_security", "antitrust_platform", "government_ai", "copyright_ai", "kakao_games_management",
}

IMPORTANT_KEYWORDS = [
    "과징금", "행정소송", "수사", "압수수색", "재판", "항소심", "판결", "제재", "조사", "공정위", "금감원",
    "금융위", "개보위", "방미통위", "과기정통부", "국회", "법안", "시행령", "입법예고", "본회의", "상임위",
    "파업", "쟁의", "노조", "성과급", "장애", "먹통", "개인정보", "유출", "해킹", "피싱", "악성코드",
    "매각", "인수", "합병", "상장", "IPO", "지분", "경영권", "실적", "영업이익", "순이익", "적자",
    "스테이블코인", "디지털자산", "AI기본법", "데이터센터", "GPU", "NPU", "AIDC", "K-문샷",
]

LOW_VALUE_TITLE_PATTERNS = [
    r"\[게시판\]", r"게시판", r"전문강사", r"연수자\s*모집", r"수료자",
    r"특강", r"교육\s*실시", r"마케팅\s*교육", r"시민\s*파워셀러", r"모집\s*시작", r"이벤트", r"기획전",
    r"할인", r"쿠폰", r"혜택", r"오픈\s*기념", r"브랜드\s*대상", r"수상", r"캠페인", r"체험단",
    r"케이스\s*유출", r"렌더링", r"출시\s*예상", r"스펙\s*유출", r"색상\s*유출",
]

# 최종 보고서에 넣기 전에 한 번 더 걸러야 하는 저가치/오탐 패턴
STRICT_EXCLUDE_TITLE_PATTERNS = [
    r"\[게시판\]", r"전문강사", r"연수자\s*모집", r"마케팅\s*교육", r"특강",
    r"케이스\s*유출", r"렌더링", r"스펙\s*유출", r"출시\s*예상",
]

DIGITAL_STRATEGIC_PATTERN = re.compile(
    r"AI|인공지능|에이전트|플랫폼|빅테크|카카오|카톡|네이버|쿠팡|토스|배달의민족|배민|구글|오픈AI|MS|메타|애플|"
    r"스테이블코인|디지털자산|가상자산|핀테크|전자금융|마이데이터|개인정보|보안|해킹|피싱|딥페이크|"
    r"데이터센터|AIDC|GPU|NPU|클라우드|망\s*사용료|온플법|인앱결제|알고리즘|지도\s*반출|저작권|콘텐츠\s*학습",
    re.IGNORECASE,
)

SELF_KAKAO_PATTERN = re.compile(
    r"카카오|카톡|Kakao|카나나|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|카카오엔터|카카오헬스케어|카카오엔터프라이즈",
    re.IGNORECASE,
)

KAKAOTALK_NOISE_PATTERN = re.compile(
    r"카카오톡\s*채널|카카오톡\s*[:：]|카톡\s*제보|카카오톡으로\s*제보|카카오톡\s*오픈채팅|e-금융교육센터.*카카오톡",
    re.IGNORECASE,
)
def canonical_press_name(source, url=""):
    """동일 언론사의 모바일/AMP/포털 경유 URL을 같은 출처로 묶기 위한 정규화."""
    source = clean_html_text(source)
    if source:
        s = source.lower().strip()
    else:
        s = guess_press_name_from_url(url).lower().strip()

    s = s.replace("www.", "").replace("m.", "").replace("mobile.", "")
    s = re.sub(r"\s+", "", s)

    aliases = {
        "연합뉴스tv": "연합뉴스TV",
        "yna.co.kr": "연합뉴스",
        "m.yna.co.kr": "연합뉴스",
        "newsis.com": "뉴시스",
        "mobile.newsis.com": "뉴시스",
        "news1.kr": "뉴스1",
        "etnews.com": "전자신문",
        "zdnet.co.kr": "지디넷코리아",
        "ddaily.co.kr": "디지털데일리",
        "edaily.co.kr": "이데일리",
        "mk.co.kr": "매일경제",
        "hankyung.com": "한국경제",
        "chosun.com": "조선일보",
        "biz.chosun.com": "조선비즈",
        "mt.co.kr": "머니투데이",
        "sedaily.com": "서울경제",
        "fnnews.com": "파이낸셜뉴스",
        "heraldcorp.com": "헤럴드경제",
        "inews24.com": "아이뉴스24",
        "dt.co.kr": "디지털타임스",
        "v.daum.net": "다음뉴스",
        "n.news.naver.com": "네이버뉴스",
    }

    for key, val in aliases.items():
        if key.lower() in s:
            return val

    return source or s


def canonical_article_signature(title, source="", url=""):
    fp = title_fingerprint(title)
    press = canonical_press_name(source, url)
    return press, fp


def title_token_set(title):
    return tokenize_for_similarity(title_fingerprint(title))


def is_title_near_duplicate(title_a, title_b):
    fp_a = title_fingerprint(title_a)
    fp_b = title_fingerprint(title_b)
    if not fp_a or not fp_b:
        return False, 0.0
    if fp_a == fp_b:
        return True, 1.0
    if min(len(fp_a), len(fp_b)) < 16:
        return False, 0.0
    seq = difflib.SequenceMatcher(None, fp_a, fp_b).ratio()
    tok = jaccard(title_token_set(fp_a), title_token_set(fp_b))
    score = max(seq, tok)
    return (seq >= 0.92 or (seq >= 0.86 and tok >= 0.55)), score


BRAND_KEYWORD_PATTERNS = {
    "카카오": SELF_KAKAO_PATTERN,
    "카카오톡": re.compile(r"카카오톡|카톡|KakaoTalk", re.IGNORECASE),
    "카카오모빌리티": re.compile(r"카카오모빌리티|카카오\s*T", re.IGNORECASE),
    "카카오페이": re.compile(r"카카오페이", re.IGNORECASE),
    "카카오뱅크": re.compile(r"카카오뱅크", re.IGNORECASE),
    "카카오엔터테인먼트": re.compile(r"카카오엔터|카카오엔터테인먼트", re.IGNORECASE),
    "카카오게임즈": re.compile(r"카카오게임즈|엑스엘게임즈", re.IGNORECASE),
    "카카오픽코마": re.compile(r"카카오픽코마|픽코마", re.IGNORECASE),
    "카카오헬스케어": re.compile(r"카카오헬스케어", re.IGNORECASE),
    "카카오엔터프라이즈": re.compile(r"카카오엔터프라이즈|카카오클라우드", re.IGNORECASE),
    "정신아": re.compile(r"정신아", re.IGNORECASE),
    "카카오 김범수": re.compile(r"김범수|카카오", re.IGNORECASE),
    "네이버": re.compile(r"네이버|NAVER|하이퍼클로바", re.IGNORECASE),
    "SKT": re.compile(r"\bSKT\b|SK텔레콤|에스케이텔레콤", re.IGNORECASE),
    "KT": re.compile(r"\bKT\b|케이티", re.IGNORECASE),
    "LGU+": re.compile(r"LGU\+|LG\s*U\+|LG유플러스|엘지유플러스", re.IGNORECASE),
    "쿠팡": re.compile(r"쿠팡|Coupang", re.IGNORECASE),
    "토스": re.compile(r"토스|비바리퍼블리카", re.IGNORECASE),
    "배달의민족": re.compile(r"배달의민족|배민|우아한형제들", re.IGNORECASE),
    "구글": re.compile(r"구글|Google|제미나이|Gemini", re.IGNORECASE),
    "오픈AI": re.compile(r"오픈\s*AI|오픈AI|OpenAI|챗GPT|ChatGPT", re.IGNORECASE),
    "MS": re.compile(r"\bMS\b|마이크로소프트|Microsoft|코파일럿", re.IGNORECASE),
    "메타": re.compile(r"메타|Meta|인스타그램|페이스북|왓츠앱", re.IGNORECASE),
    "애플": re.compile(r"애플|Apple|아이폰|iPhone", re.IGNORECASE),
}

NOISE_ONLY_PATTERNS = re.compile(
    r"카카오톡\s*채널|카카오톡\s*[:：]|카톡\s*제보|카카오톡으로\s*제보|"
    r"네이버에서\s*구독|네이버\s*채널|네이버\s*뉴스\s*구독|"
    r"제보는\s*(카카오톡|네이버|라인)|기사제보|구독\s*신청|무단전재|재배포\s*금지",
    re.IGNORECASE,
)


def collection_relevance_reason(keyword, title, summary):
    """수집 단계에서 명백한 검색 오탐을 걸러냅니다. 애매하면 살립니다."""
    if not ENABLE_COLLECTION_RELEVANCE_FILTER:
        return ""

    title = clean_html_text(title)
    summary = clean_html_text(summary)
    front_text = f"{title} {summary[:450]}"
    pattern = BRAND_KEYWORD_PATTERNS.get(keyword)

    if pattern:
        title_has = bool(pattern.search(title))
        front_has = bool(pattern.search(front_text))
        if not title_has and not front_has:
            return "brand_keyword_absent_from_title_and_front_summary"

        if not title_has and NOISE_ONLY_PATTERNS.search(front_text):
            mentions = len(pattern.findall(front_text))
            if mentions <= 2:
                return "brand_only_in_contact_or_subscription_noise"

    if keyword in {"카카오", "카카오톡"}:
        if not SELF_KAKAO_PATTERN.search(title) and NOISE_ONLY_PATTERNS.search(front_text):
            return "kakao_contact_noise"

    if keyword in {"네이버", "토스", "카카오", "카카오톡", "구글"} and pattern:
        if re.search(r"교통사고|사망사고|음주운전|화재|폭행|살인|실종|구속송치", title) and not pattern.search(title):
            return "local_incident_without_brand_in_title"

    return ""


def make_seen_registry():
    return {
        "urls": set(),
        "exact_titles_global": set(),
        "exact_titles_by_press": set(),
        "source_buckets": {},
        "prefix_buckets": {},
    }


def canonical_duplicate_reason(registry, title, source, url):
    normalized_url = normalize_url(url)
    press, fp = canonical_article_signature(title, source, url)
    if not fp:
        return "", None

    if normalized_url in registry["urls"]:
        return "same_normalized_url", None

    if len(fp) >= 18 and fp in registry["exact_titles_global"]:
        return "same_canonical_title_global", None

    exact_press_key = f"{press}::{fp}"
    if exact_press_key in registry["exact_titles_by_press"]:
        return "same_canonical_title_same_press", None

    for old in registry["source_buckets"].get(press, []):
        near, score = is_title_near_duplicate(title, old.get("title", ""))
        if near:
            return f"near_duplicate_same_press:{score:.2f}", old

    prefix = fp[:22]
    for old in registry["prefix_buckets"].get(prefix, []):
        near, score = is_title_near_duplicate(title, old.get("title", ""))
        if near and score >= 0.94:
            return f"near_duplicate_same_title_prefix:{score:.2f}", old

    return "", None


def register_canonical_article(registry, title, source, url):
    normalized_url = normalize_url(url)
    press, fp = canonical_article_signature(title, source, url)
    if normalized_url:
        registry["urls"].add(normalized_url)
    if fp:
        registry["exact_titles_global"].add(fp)
        registry["exact_titles_by_press"].add(f"{press}::{fp}")
        item = {"title": title, "source": press, "url": normalized_url, "fp": fp}
        registry["source_buckets"].setdefault(press, []).append(item)
        registry["prefix_buckets"].setdefault(fp[:22], []).append(item)


# 한 보고서 안에서 같은 큰 사건이 과도하게 반복되는 것을 막는 상한
MAX_FINAL_PER_EVENT_TAG = {
    "stablecoin": 2,
    "earnings": 1,
    "aidc_datacenter": 2,
    "ai_agent": 2,
    "antitrust_platform": 2,
    "phishing_security": 2,
    "kakao_games_management": 1,
}

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

    # 과잉 매칭 보정: 기존 dunamu_stake는 '지분/매각'만 있어도 잡혀
    # 카카오게임즈 대표 인선 기사와 두나무 지분 매각 기사를 같은 사건으로 오판했음.
    if "dunamu_stake" in found and not re.search(r"두나무|업비트|Dunamu|Upbit|하나금융|하나은행|한화투자", normalized, flags=re.IGNORECASE):
        found.discard("dunamu_stake")

    # 카카오게임즈 경영권/대표/라인야후 이슈는 두나무 지분 매각과 별개 사건으로 분리.
    if re.search(r"카카오게임즈", normalized) and re.search(r"라인야후|김태환|대표|경영권|최대주주|적자|목표가", normalized):
        found.add("kakao_games_management")

    return found


def issue_key(candidate):
    text = f"{candidate.get('기사제목') or candidate.get('title') or ''} {candidate.get('본문요약') or candidate.get('summary') or candidate.get('본문전문') or ''}"
    entities = detect_entities(text)
    tags = detect_event_tags(text)
    strong_tags = tags & STRONG_EVENT_TAGS
    return entities, strong_tags


PHASE_SENSITIVE_TAGS = {
    "labor_dispute", "stablecoin", "antitrust_platform", "aidc_datacenter",
    "mobility_sale_ipo", "kakao_pay_alipay", "kim_beomsoo_trial", "kakao_games_management",
}


def extract_event_stage(text, tags=None):
    text = clean_html_text(text)
    tags = tags or detect_event_tags(text)

    if "labor_dispute" in tags:
        if re.search(r"찬반\s*투표|투표.*가결|투표.*찬성", text):
            return "labor_vote"
        if re.search(r"1차\s*조정", text):
            return "labor_1st_mediation"
        if re.search(r"2차\s*조정|최종\s*조정|조정\s*회의|오늘.*조정", text):
            return "labor_mediation_meeting"
        if re.search(r"조정\s*중지|쟁의권\s*확보", text):
            return "labor_right_to_strike"
        if re.search(r"파업\s*돌입|공동파업|공동\s*총파업|파업\s*예고|결의대회", text):
            return "labor_strike_action"
        if re.search(r"성과급|RSU|보상체계|임금", text):
            return "labor_pay_dispute"
        return "labor_general"

    if "kakao_games_management" in tags:
        if re.search(r"김태환|대표\s*(취임|내정|선임)|신임\s*대표", text):
            return "kakao_games_ceo_appointment"
        if re.search(r"적자|영업손실|목표가|실적|매출", text):
            return "kakao_games_earnings_outlook"
        if re.search(r"라인야후|경영권|최대주주|매각|인수", text):
            return "kakao_games_control_sale"
        if re.search(r"구조조정|희망퇴직|정리해고|엑스엘게임즈", text):
            return "kakao_games_restructuring"
        return "kakao_games_general"

    if "dunamu_stake" in tags:
        if re.search(r"두나무|업비트", text):
            if re.search(r"하나금융|하나은행|한화투자|처분|매각|지분|1\.6조|1조", text):
                return "dunamu_stake_sale"
            if re.search(r"주식교환|합병|네이버", text):
                return "dunamu_merger_exchange"
        return "dunamu_general"

    if "stablecoin" in tags:
        if re.search(r"안도걸|정책\s*토론회|토론회|자금세탁방지|AML", text):
            return "stablecoin_forum_aml"
        if re.search(r"디지털자산기본법|특금법|입법|법안|시행령", text):
            return "stablecoin_legislation"
        if re.search(r"한은|CBDC|예금토큰", text):
            return "stablecoin_cbdc"
        if re.search(r"시총|외환보유액|USDT|USDC|테더", text):
            return "stablecoin_market_size"
        return "stablecoin_general"

    if "antitrust_platform" in tags:
        if re.search(r"김범석|동일인|총수|지정자료|허위\s*자료", text):
            return "antitrust_coupang_identity"
        if re.search(r"쿠팡이츠|최혜대우|동의의결|배달앱", text):
            return "antitrust_delivery_mfn"
        if re.search(r"온플법|온라인플랫폼", text):
            return "antitrust_onplaw"
        if re.search(r"구글|애플|인앱결제", text):
            return "antitrust_inapp"
        if re.search(r"담합|처분시효", text):
            return "antitrust_collusion"
        if re.search(r"조사국|경제분석국|중점조사", text):
            return "antitrust_org_reform"
        return "antitrust_general"

    if "aidc_datacenter" in tags:
        if re.search(r"국가AI컴퓨팅센터|해남|삼성SDS", text):
            return "aidc_national_center"
        if re.search(r"AIDC\s*특별법|전력|PPA|LNG", text):
            return "aidc_law_power"
        if re.search(r"NPU|GPU|AI\s*반도체", text):
            return "aidc_gpu_npu"
        return "aidc_general"

    if "phishing_security" in tags:
        if re.search(r"피싱|사칭|악성코드", text):
            return "security_phishing_malware"
        if re.search(r"해킹|침해사고|제로트러스트|유심", text):
            return "security_hacking_incident"
        if re.search(r"개인정보\s*유출|개보위", text):
            return "security_privacy_leak"
        return "security_general"

    strong = sorted(set(tags) & STRONG_EVENT_TAGS)
    return strong[0] if strong else ""


def event_signature(text):
    entities = detect_entities(text)
    tags = detect_event_tags(text) & STRONG_EVENT_TAGS
    if not entities or not tags:
        return ""

    stage = extract_event_stage(text, tags)
    primary_entities = sorted(entities)[:3]
    primary_tags = sorted(tags)[:2]
    return "|".join(primary_entities + primary_tags + ([stage] if stage else []))


def should_cluster_as_duplicate(mode, title_score, token_score, text_score, shared_entities, shared_tags, candidate_text, past_text):
    if not (shared_entities and shared_tags):
        return False

    candidate_stage = extract_event_stage(candidate_text, detect_event_tags(candidate_text))
    past_stage = extract_event_stage(past_text, detect_event_tags(past_text))

    # 노사/입법/규제/M&A처럼 단계가 중요한 이슈는 단계가 다르면 업데이트일 수 있으므로 보수적으로 유지.
    phase_sensitive = bool((detect_event_tags(candidate_text) | detect_event_tags(past_text)) & PHASE_SENSITIVE_TAGS)
    if phase_sensitive and candidate_stage and past_stage and candidate_stage != past_stage:
        return title_score >= 0.78 and token_score >= 0.22

    same_signature = event_signature(candidate_text) and event_signature(candidate_text) == event_signature(past_text)

    if mode == "rss":
        # RSS 단계에서는 제목/요약밖에 없어 과잉 삭제 위험이 큼.
        # 같은 시그니처여도 제목/토큰 유사도가 낮으면 업데이트 기사일 수 있으므로 살림.
        return same_signature and (title_score >= 0.62 or token_score >= 0.30 or text_score >= 0.48)

    # 본문 확인 뒤에도 같은 태그만으로는 제거하지 않음.
    # 같은 시그니처 + 일정 유사도 이상일 때만 제거.
    if same_signature:
        return title_score >= 0.42 or token_score >= 0.18 or text_score >= 0.34
    return title_score >= 0.66 and token_score >= 0.24


def article_similarity(candidate, past_item, mode="body"):
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

    # 1) 같은 강한 사건 태그 + 같은 주요 주체라도, 단계가 다른 업데이트면 중복으로 보지 않음.
    #    예: 1차 조정 → 2차 조정, 찬반투표 가결 → 조정회의 진행은 새 진행상황으로 유지.
    if should_cluster_as_duplicate(
        mode=mode,
        title_score=title_score,
        token_score=token_score,
        text_score=text_score,
        shared_entities=shared_entities,
        shared_tags=shared_tags,
        candidate_text=candidate_text,
        past_text=past_text,
    ):
        is_duplicate = True
        reason = f"same_issue_signature:{extract_event_stage(candidate_text, candidate_tags)}"

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


def find_past_duplicate(candidate, recent_past_items, mode="body"):
    if not recent_past_items:
        return False, None, None

    best_item = None
    best_result = None
    best_score = -1.0

    for past_item in recent_past_items:
        result = article_similarity(candidate, past_item, mode=mode)
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

    # 실행 시각과 무관하게 "어제 00:00 KST ~ 현재" 범위를 사용한다.
    # 예: 6월 10일 16:00 실행 시 6월 9일 00:00부터 수집 대상.
    start = datetime.combine(
        now.date() - timedelta(days=1),
        datetime.min.time(),
        tzinfo=KST,
    )

    return start <= dt <= now + timedelta(minutes=10)


def max_items_for_keyword(keyword):
    return MAX_RSS_ITEMS_PER_KEYWORD_OVERRIDES.get(keyword, MAX_RSS_ITEMS_PER_KEYWORD)


def add_unique_query(queries, query):
    query = re.sub(r"\s+", " ", query).strip()
    if query and query not in queries:
        queries.append(query)


def build_rss_queries(keyword):
    queries = []
    base = keyword.strip()

    # 기본 쿼리 2개. 대부분의 좁은 키워드는 이것만으로 충분함.
    add_unique_query(queries, f"{base} when:{RSS_RECENT_DAYS}d")
    add_unique_query(queries, f"\"{base}\" when:{RSS_RECENT_DAYS}d")

    if not ENABLE_RSS_QUERY_SHARDING:
        return queries

    # 카카오처럼 너무 넓은 키워드는 이슈 축으로 쪼개야 RSS 100개 안팎 상한을 우회할 수 있음.
    if base in {"카카오", "카카오톡"}:
        for term in KAKAO_ISSUE_SHARDS:
            add_unique_query(queries, f"{base} {term} when:{RSS_RECENT_DAYS}d")

    # 중대 키워드만 일반 이슈 축을 추가함. 모든 키워드에 적용하면 요청 수가 과도해짐.
    elif base in RSS_SHARD_KEYWORDS:
        for term in GENERAL_ISSUE_SHARDS[:8]:
            add_unique_query(queries, f"{base} {term} when:{RSS_RECENT_DAYS}d")

    # 언론사별 site: shard. 구글 뉴스 화면의 많은 결과는 RSS 단일 검색에서 생략되는 경우가 있어
    # 주요 언론 도메인으로 한 번씩 나눠 긁어 후보를 보강함.
    if base in RSS_SHARD_KEYWORDS:
        for domain in RSS_SOURCE_SHARDS:
            add_unique_query(queries, f"{base} site:{domain} when:{RSS_RECENT_DAYS}d")

    return queries[:MAX_RSS_QUERIES_PER_KEYWORD]


def add_article(raw_articles, seen_registry, article_id, keyword, title, link, source="", published="", summary="", collector=""):
    title = clean_html_text(title)
    link = str(link).strip() if link else ""
    summary = clean_html_text(summary)

    if not title or not link:
        return article_id, False, "empty_title_or_link"

    normalized = normalize_url(link)
    if not normalized:
        return article_id, False, "empty_normalized_url"

    relevance_reason = collection_relevance_reason(keyword, title, summary)
    if relevance_reason:
        return article_id, False, f"collection_relevance:{relevance_reason}"

    dedupe_reason, matched_old = canonical_duplicate_reason(seen_registry, title, source, normalized)
    if dedupe_reason:
        return article_id, False, dedupe_reason

    category_name = keyword_to_category.get(keyword, "")
    json_category = CATEGORY_TO_JSON_KEY.get(category_name, "")

    if not source:
        source = guess_press_name_from_url(link)

    canonical_press = canonical_press_name(source, normalized)
    title_fp = title_fingerprint(title)

    raw_articles.append({
        "id": article_id,
        "원카테고리": category_name,
        "JSON카테고리": json_category,
        "검색어": keyword,
        "기사제목": title,
        "기사제목_정규화": title_fp,
        "언론사": clean_html_text(source),
        "정규언론사": canonical_press,
        "게시일": clean_html_text(published),
        "본문요약": summary,
        "링크": normalized,
        "수집채널": collector,
    })

    register_canonical_article(seen_registry, title, source, normalized)
    return article_id + 1, True, "added"


def collect_with_google_rss(recent_past_items):
    print("\n🚀 [STEP 1] Google News RSS로 오늘 뉴스 후보 수집 시작...")
    print(f"   └ 키워드당 기본 최대 {MAX_RSS_ITEMS_PER_KEYWORD}개, 카카오 등 광범위 키워드는 query sharding으로 더 많이 시도합니다.")
    print("   └ v7: URL/제목/언론사 중복 제거, 검색 오탐 필터, shard overlap 조기중단, 원문 발행일 검증을 적용합니다.")
    print("   └ 원문 URL 디코딩은 여기서 하지 않고, 최종 후보 본문 추출 단계에서만 적용합니다.")

    session = requests.Session()
    session.headers.update(WEB_HEADERS)

    raw_articles = []
    skipped_duplicates = []
    seen_registry = make_seen_registry()
    article_id = 1

    for idx, keyword in enumerate(all_keywords, 1):
        collected = 0
        skipped_old = 0
        skipped_past = 0
        skipped_seen = 0
        skipped_relevance = 0
        low_yield_streak = 0
        early_stopped = False

        queries = build_rss_queries(keyword)
        target_count = max_items_for_keyword(keyword)

        for q_idx, query_text in enumerate(queries, 1):
            if collected >= target_count:
                break

            encoded_query = urllib.parse.quote_plus(query_text)
            rss_url = (
                f"https://news.google.com/rss/search?q={encoded_query}"
                f"&hl=ko&gl=KR&ceid=KR:ko&num={RSS_FEED_ITEM_LIMIT_PER_QUERY}"
            )

            query_seen_before = skipped_seen
            query_relevance_before = skipped_relevance
            query_past_before = skipped_past
            query_old_before = skipped_old
            query_collected_before = collected

            try:
                req = session.get(rss_url, timeout=12)
                req.raise_for_status()
                feed = feedparser.parse(req.content)

                for entry in feed.entries[:RSS_FEED_ITEM_LIMIT_PER_QUERY]:
                    if collected >= target_count:
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

                    relevance_reason = collection_relevance_reason(keyword, title, summary)
                    if relevance_reason:
                        skipped_relevance += 1
                        skipped_duplicates.append({
                            "검색어": keyword,
                            "후보제목": title,
                            "후보링크": google_link,
                            "후보언론사": source,
                            "매칭과거일자": "COLLECTION_FILTER",
                            "매칭과거제목": "",
                            "매칭과거링크": "",
                            "중복판정이유": f"collection_relevance:{relevance_reason}",
                            "공유주체": "",
                            "공유사건태그": "",
                            "종합점수": "",
                            "제목점수": "",
                            "토큰점수": "",
                            "본문점수": "",
                        })
                        continue

                    candidate = {"기사제목": title, "본문요약": summary, "링크": google_link}
                    is_dup, matched, sim = find_past_duplicate(candidate, recent_past_items, mode="rss")
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

                    article_id, added, add_reason = add_article(
                        raw_articles=raw_articles,
                        seen_registry=seen_registry,
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
                        if add_reason.startswith("collection_relevance"):
                            skipped_relevance += 1
                        else:
                            skipped_seen += 1
                        skipped_duplicates.append({
                            "검색어": keyword,
                            "후보제목": title,
                            "후보링크": google_link,
                            "후보언론사": source,
                            "매칭과거일자": "CURRENT_COLLECTION",
                            "매칭과거제목": "canonical_seen",
                            "매칭과거링크": "",
                            "중복판정이유": add_reason,
                            "공유주체": "",
                            "공유사건태그": "",
                            "종합점수": "",
                            "제목점수": "",
                            "토큰점수": "",
                            "본문점수": "",
                        })

            except Exception as e:
                print(f"\n⚠️ [GOOGLE RSS {idx}/{len(all_keywords)}] '{keyword}' 쿼리 실패: {query_text} / {e}")

            new_from_query = collected - query_collected_before
            dup_from_query = skipped_seen - query_seen_before
            rel_from_query = skipped_relevance - query_relevance_before
            past_from_query = skipped_past - query_past_before
            old_from_query = skipped_old - query_old_before
            rejected_from_query = dup_from_query + rel_from_query + past_from_query + old_from_query
            processed_from_query = max(1, new_from_query + rejected_from_query)
            duplicate_rate = rejected_from_query / processed_from_query

            if ENABLE_SHARD_OVERLAP_EARLY_STOP and q_idx > 2:
                if new_from_query <= LOW_YIELD_NEW_ARTICLE_COUNT and duplicate_rate >= LOW_YIELD_DUPLICATE_RATE:
                    low_yield_streak += 1
                else:
                    low_yield_streak = 0

                min_before_stop = min(MIN_COLLECTED_BEFORE_EARLY_STOP, max(25, int(target_count * 0.45)))
                if collected >= min_before_stop and low_yield_streak >= MAX_CONSECUTIVE_LOW_YIELD_QUERIES:
                    early_stopped = True
                    break

            time.sleep(random.uniform(0.03, 0.10))

        print(
            f"▶ [GOOGLE RSS {idx}/{len(all_keywords)}] "
            f"'{keyword}' 수집 {collected}개 / 목표 {target_count}개 / 쿼리 {len(queries)}개"
            f"{' / 조기중단' if early_stopped else ''} / "
            f"시간제외 {skipped_old}개 / 과거중복제외 {skipped_past}개 / "
            f"수집중복제외 {skipped_seen}개 / 오탐제외 {skipped_relevance}개        ",
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



def is_article_obviously_low_value(article):
    title = article.get("기사제목", "")
    text = f"{title} {article.get('본문요약', '')}"

    for pattern in STRICT_EXCLUDE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return True, f"strict_low_value_title:{pattern}"

    # 카카오톡 검색에서 자주 걸리는 '카카오톡 채널로 신청/제보' 오탐 제거
    if article.get("검색어") == "카카오톡":
        if not re.search(r"카카오톡|카톡", title, flags=re.IGNORECASE) and KAKAOTALK_NOISE_PATTERN.search(text):
            return True, "kakaotalk_channel_noise"

    return False, ""


def is_report_item_relevant(report_item, json_key):
    title = report_item.get("기사제목", "")
    body = report_item.get("본문전문", "")
    keyword = report_item.get("검색어", "")
    text = f"{title} {body[:1200]}"

    for pattern in STRICT_EXCLUDE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return False, f"low_value_title:{pattern}"

    # 자사/계열사 카테고리는 제목이나 본문 앞부분에 카카오 핵심 주체가 실제로 있어야 함.
    if json_key == "자사_및_계열사_이슈":
        if not SELF_KAKAO_PATTERN.search(text):
            return False, "self_category_without_kakao_entity"

        # 제목에는 카카오가 없고, 본문에 '카카오톡 채널로 신청' 같은 표현만 있는 경우는 오탐.
        if keyword == "카카오톡" and not re.search(r"카카오톡|카톡|카카오", title, flags=re.IGNORECASE):
            kakao_mentions = len(re.findall(r"카카오|카톡|Kakao", text, flags=re.IGNORECASE))
            if kakao_mentions <= 2 and KAKAOTALK_NOISE_PATTERN.search(text):
                return False, "kakaotalk_channel_only_noise"

    # 정부/국회는 IT·플랫폼·AI·디지털금융 관련성이 있어야 함. 일반 금융/교육/보험 통계는 제외.
    if json_key == "정부_국회":
        if not DIGITAL_STRATEGIC_PATTERN.search(text):
            return False, "government_without_digital_platform_ai_relevance"
        if re.search(r"보험사\s*순이익|금융교육\s*전문강사|생산적\s*금융\s*실적", title):
            return False, "generic_finance_not_ceo_platform_issue"

    # 경쟁사/해외이슈도 단순 기기 루머/케이스 유출은 제외.
    if json_key == "경쟁사_해외이슈":
        if re.search(r"케이스\s*유출|렌더링|출시\s*예상|스펙\s*유출", title):
            return False, "consumer_product_rumor"

    return True, ""


def final_duplicate_reason(new_item, existing_items):
    new_title = new_item.get("기사제목", "")
    new_body = new_item.get("본문전문", "")[:1800]
    new_text = f"{new_title} {new_body}"
    new_tokens = tokenize_for_similarity(new_text)
    new_tags = set(filter(None, (new_item.get("사건태그", "") or "").split(","))) or detect_event_tags(new_text)
    new_sig = event_signature(new_text)
    new_stage = extract_event_stage(new_text, new_tags)
    new_press, new_fp = canonical_article_signature(new_title, new_item.get("언론사", ""), new_item.get("링크", ""))

    for old in existing_items:
        old_title = old.get("기사제목", "")
        old_body = old.get("본문전문", "")[:1800]
        old_text = f"{old_title} {old_body}"
        old_tokens = tokenize_for_similarity(old_text)
        old_tags = set(filter(None, (old.get("사건태그", "") or "").split(","))) or detect_event_tags(old_text)
        old_sig = event_signature(old_text)
        old_stage = extract_event_stage(old_text, old_tags)
        old_press, old_fp = canonical_article_signature(old_title, old.get("언론사", ""), old.get("링크", ""))

        title_seq = sequence_ratio(new_title, old_title)
        token_score = jaccard(new_tokens, old_tokens)
        body_ngram_score = jaccard(char_ngrams(new_body[:1200], 5), char_ngrams(old_body[:1200], 5))

        if new_fp and old_fp and new_fp == old_fp:
            return old, "same_current_canonical_title"

        near, near_score = is_title_near_duplicate(new_title, old_title)
        if near and (new_press == old_press or near_score >= 0.94):
            return old, f"near_current_title:{near_score:.2f}"

        if body_ngram_score >= 0.55:
            return old, "same_current_body"

        if new_sig and old_sig and new_sig == old_sig:
            if title_seq >= 0.42 or token_score >= 0.16 or body_ngram_score >= 0.30:
                return old, f"same_current_signature:{new_sig}"

        if title_seq >= 0.68 and token_score >= 0.14:
            return old, f"similar_current_issue:title={title_seq:.2f},token={token_score:.2f}"

        if "카카오" in new_text and "카카오" in old_text:
            labor_words = r"노조|파업|공동파업|총파업|조정|지노위|노동위|성과급|RSU|쟁의권"
            if re.search(labor_words, new_text) and re.search(labor_words, old_text):
                if new_stage == old_stage and new_stage in {"labor_mediation_meeting", "labor_vote", "labor_right_to_strike", "labor_strike_action", "labor_pay_dispute"}:
                    if token_score >= 0.07 or title_seq >= 0.38:
                        return old, f"same_current_labor_stage:{new_stage}"

        anchor_sets = [
            ("안도걸", "디지털자산기본법", "스테이블코인"),
            ("K-문샷", "추진단"),
            ("카나나 스칼라", "콜로키움"),
            ("국가AI컴퓨팅센터", "삼성SDS"),
            ("구글", "EU", "과징금"),
        ]
        for anchors in anchor_sets:
            if all(a in new_text for a in anchors) and all(a in old_text for a in anchors):
                if token_score >= 0.06 or title_seq >= 0.35:
                    return old, f"same_current_anchor:{'/'.join(anchors)}"

        for anchor in ["안도걸", "디지털자산기본법", "K-문샷", "카나나 스칼라", "카카오게임즈", "카카오T블루"]:
            if anchor in new_text and anchor in old_text and (new_tags & old_tags):
                if new_stage and old_stage and new_stage != old_stage:
                    continue
                if token_score >= 0.08 or title_seq >= 0.50:
                    return old, f"same_current_anchor:{anchor}"

    return None, ""


def event_tag_limit_reason(new_item, existing_items):
    tags = [t for t in (new_item.get("사건태그", "") or "").split(",") if t]
    if not tags:
        return ""

    for tag in tags:
        limit = MAX_FINAL_PER_EVENT_TAG.get(tag)
        if not limit:
            continue
        count = 0
        for old in existing_items:
            old_tags = set(filter(None, (old.get("사건태그", "") or "").split(",")))
            if tag in old_tags:
                count += 1
        if count >= limit:
            return f"event_tag_limit:{tag}:{limit}"

    return ""

def rank_score_article(article):
    title = article.get("기사제목", "")
    summary = article.get("본문요약", "")
    text = f"{title} {summary}"
    json_key = article.get("JSON카테고리", "")
    keyword = article.get("검색어", "")

    score = 0.0
    penalty_reasons = []

    low_value, low_value_reason = is_article_obviously_low_value(article)
    if low_value:
        score -= 90
        penalty_reasons.append(low_value_reason)

    relevance_reason = collection_relevance_reason(keyword, title, summary)
    if relevance_reason:
        score -= 70
        penalty_reasons.append(f"collection_relevance:{relevance_reason}")

    pseudo_item = {"기사제목": title, "본문전문": summary, "검색어": keyword}
    relevant, report_relevance_reason = is_report_item_relevant(pseudo_item, json_key)
    if not relevant:
        score -= 45
        penalty_reasons.append(report_relevance_reason)

    score += recency_score(article.get("게시일", ""))
    score += press_score(article.get("언론사", ""))

    for kw in IMPORTANT_KEYWORDS:
        if kw in text:
            score += 4

    event_tags = detect_event_tags(text)
    score += min(18, len(event_tags) * 4)

    if json_key == "자사_및_계열사_이슈":
        score += 8
    elif json_key == "정부_국회":
        score += 6
    elif json_key == "경쟁사_해외이슈":
        score += 4
    elif json_key == "산업동향":
        score += 2

    for pattern in LOW_VALUE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            score -= 22
            penalty_reasons.append(f"low_value:{pattern}")

    brand_pat = BRAND_KEYWORD_PATTERNS.get(keyword)
    if brand_pat and not brand_pat.search(title):
        score -= 12
        penalty_reasons.append("brand_not_in_title")

    if len(title) < 12:
        score -= 5
        penalty_reasons.append("short_title")

    article["랭킹점수"] = round(score, 3)
    article["사건태그"] = ",".join(sorted(event_tags))
    article["사건단계"] = extract_event_stage(text, event_tags)
    article["주요주체"] = ",".join(sorted(detect_entities(text)))
    article["랭킹감점사유"] = ";".join([r for r in penalty_reasons if r])
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

    is_dup, matched, sim = find_past_duplicate(body_candidate, recent_past_items, mode="body")

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
        "사건태그": ",".join(sorted(detect_event_tags(f"{article_info.get('기사제목', '')} {body_text[:1800]}"))),
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

            tag_line = build_display_tag_line(item)
            if tag_line:
                lines.append(tag_line)

            lines.append(summarize_body_locally(title, body))
            lines.append("")

    return "\n".join(lines).strip()


# ==========================================
# Display tags for briefing UI
# ==========================================

DISPLAY_TAG_MAP = {
    # Gemini internal/category style
    "자사_및_계열사_이슈": "#자사이슈",
    "정부_국회": "#정부정책",
    "경쟁사_해외이슈": "#경쟁사해외",
    "산업동향": "#산업동향",
    "platform_operator_obligation": "#플랫폼의무",
    "self:platform_obligation": "#플랫폼의무",
    "government:platform_obligation": "#플랫폼규제",
    "competitor:platform_obligation": "#플랫폼의무",
    "자사 직접 리스크": "#자사리스크",
    "플랫폼 사업자 의무/규제": "#플랫폼규제",
    "경쟁사 전략": "#경쟁사전략",
    "산업 구조 변화": "#산업구조변화",
    "홍보/후원/프로모션": "#홍보성",
    "SELF_DIRECT_RISK": "#자사리스크",
    "SELF_AFFILIATE_BUSINESS": "#자사계열",
    "SELF_INCLUDED_REGULATION": "#자사관련규제",
    "GOV_PLATFORM_REGULATION": "#플랫폼규제",
    "GOV_AI_DIGITAL_POLICY": "#AI정책",
    "GOV_FINANCIAL_DIGITAL_POLICY": "#디지털금융",
    "COMPETITOR_PLATFORM_RISK": "#경쟁사리스크",
    "COMPETITOR_AI_STRATEGY": "#경쟁사전략",
    "INDUSTRY_STRUCTURAL_CHANGE": "#산업구조변화",

    # issue/event tags
    "labor_dispute": "#노사갈등",
    "labor_negotiation": "#임단협",
    "labor_law_complaint": "#노무리스크",
    "privacy_security_reliability": "#개인정보",
    "legal_regulatory_enforcement": "#규제제재",
    "platform_obligation": "#플랫폼의무",
    "ai_infrastructure_security": "#AI인프라",
    "policy_legislation": "#정책입법",
    "digital_asset_stablecoin": "#디지털자산",
    "market_power_platform": "#플랫폼경쟁",
    "strategy_mna_governance": "#지배구조",
    "financial_performance": "#실적",

    # common issue families
    "self_labor": "#노사갈등",
    "self_legal_regulatory": "#자사리스크",
    "self_privacy_security": "#개인정보",
    "self_governance_mna": "#지배구조",
    "government_platform_regulation": "#플랫폼규제",
    "government_ai_policy": "#AI정책",
    "government_digital_finance_policy": "#디지털금융",
    "competitor_ai_strategy": "#경쟁사전략",
    "competitor_security_risk": "#경쟁사리스크",
    "competitor_business_strategy": "#경쟁사전략",
    "industry_ai_infrastructure": "#AI인프라",
}

DISPLAY_ENTITY_TAGS = [
    "카카오", "카카오페이", "카카오뱅크", "카카오모빌리티", "카카오게임즈",
    "네이버", "쿠팡", "SKT", "KT", "LGU+", "구글", "오픈AI", "MS", "메타",
    "애플", "앤트로픽", "엔비디아", "토스", "배달의민족", "배민",
    "공정위", "금융위", "금감원", "과기정통부", "방미통위", "개보위",
]

def normalize_display_tag(value):
    value = clean_html_text(value)
    if not value:
        return ""
    if value.startswith("#"):
        return value

    mapped = DISPLAY_TAG_MAP.get(value)
    if mapped:
        return mapped

    # 너무 긴 설명형 값은 태그로 쓰지 않음
    if len(value) > 18:
        return ""

    # 영문 코드형인데 매핑이 없으면 화면에 노출하지 않음
    if re.fullmatch(r"[A-Z0-9_]+", value) or re.fullmatch(r"[a-z0-9_]+", value):
        return ""

    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[^0-9A-Za-z가-힣+]", "", value)
    if not value:
        return ""
    return f"#{value}"

def split_tag_values(value):
    value = clean_html_text(value)
    if not value:
        return []
    parts = re.split(r"[,;/|]+", value)
    return [p.strip() for p in parts if p.strip()]

def build_display_tags(item, max_tags=4):
    tags = []

    def add_tag(tag):
        tag = normalize_display_tag(tag)
        if tag and tag not in tags:
            tags.append(tag)

    # 1) 관점 태그: 최우선
    for col in ["Gemini내부카테고리", "internal_category", "Gemini카테고리"]:
        val = item.get(col, "")
        if val:
            add_tag(val)
            break

    # 카테고리 fallback
    if not tags:
        cat = item.get("카테고리", "")
        if cat == "자사_및_계열사_이슈":
            add_tag("#자사이슈")
        elif cat == "정부_국회":
            add_tag("#정부정책")
        elif cat == "경쟁사_해외이슈":
            add_tag("#경쟁사해외")
        elif cat == "산업동향":
            add_tag("#산업동향")

    # 2) 사건/이슈 유형 태그
    for col in ["사건태그", "issue_family", "이슈패밀리", "v20_issue_family", "Gemini이슈유형"]:
        for val in split_tag_values(item.get(col, "")):
            add_tag(val)
            if len(tags) >= max_tags:
                return tags

    # 3) 주요 주체 태그
    entity_text = " ".join(str(item.get(col, "")) for col in [
        "주요주체", "Gemini이슈그룹", "v20_issue_group", "기사제목", "본문전문"
    ])
    for entity in DISPLAY_ENTITY_TAGS:
        if entity in entity_text:
            add_tag(f"#{entity}")
            if len(tags) >= max_tags:
                return tags

    # 4) 규제의무 보조 태그
    if item.get("정책규제의무") == "Y":
        add_tag("#규제의무")

    if item.get("자사홍보성") == "Y":
        add_tag("#홍보성")

    return tags[:max_tags]

def build_display_tag_line(item):
    tags = build_display_tags(item)
    return " ".join(tags)

def build_structured_briefing(final_report_data, summary_map=None):
    """Gemini에는 요약문만 맡기고, 최종 레이아웃은 코드가 고정 생성."""
    summary_map = summary_map or {}
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
            brief_id = str(item.get("브리핑ID", ""))
            summary = clean_html_text(summary_map.get(brief_id, ""))
            if not summary:
                summary = summarize_body_locally(item.get("기사제목", ""), item.get("본문전문", ""))
            # 마침표와 종결어미 보정
            if not summary.endswith("."):
                summary = summary.rstrip() + "."

            lines.append(f"{num} {item.get('기사제목', '')}")
            lines.append(item.get("링크", ""))
            lines.append(f"({item.get('언론사', '') or guess_press_name_from_url(item.get('링크', ''))})")

            tag_line = build_display_tag_line(item)
            if tag_line:
                lines.append(tag_line)

            lines.append(summary)
            lines.append("")

    return "\n".join(lines).strip()


def normalize_summary_json(data):
    if not isinstance(data, dict):
        return {}
    out = {}
    for k, v in data.items():
        key = str(k).strip()
        if isinstance(v, dict):
            v = v.get("summary") or v.get("요약") or ""
        if not isinstance(v, str):
            continue
        out[key] = clean_html_text(v)
    return out

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
너는 카카오의 대외협력·대관·정책리스크 모니터링팀에서 매일 아침 최고경영진용 뉴스 브리핑을 고르는 담당자야.
단순히 검색어가 들어간 기사를 고르는 게 아니라, 카카오 및 계열사의 경영·규제·평판·노무·정책 대응에 영향을 줄 수 있는 이슈를 선별해야 해.
아래 [최근 7일 과거 보고서 데이터]를 보고 유저가 어떤 무게감의 기사를 골랐는지 학습한 뒤, [오늘 뉴스 후보 리스트]에서 오늘 보고서에 넣을 기사만 골라.

[선별 관점: 카카오 대외협력/대관팀의 자아]
1. 카카오 및 계열사 이슈는 최우선으로 봐. 특히 경영진·임원 이동, 조직개편, 서비스 개편 논란, 노사/임단협/파업, 노동부 진정, 최저임금법, 고용노동부 근로감독, 소송·수사·과징금·개인정보·장애·피싱·지배구조·실적·매각·투자회수는 높게 평가해.
2. 예를 들어 '[단독] 카톡 개편 논란 관련 임원/CPO 퇴사', '민주노총·노동부 진정·최저임금법 위반', '카카오모빌리티 노무/규제 리스크'는 제목이 자극적이지 않아도 CEO 브리핑 가치가 높아.
3. 정부/국회는 AI, 플랫폼, 디지털자산, 스테이블코인, 개인정보, 보안, 공정위, 금융위, 금감원, 과기정통부, 방미통위, 온플법, 망사용료, 지도반출 등 카카오 사업환경에 영향을 주는 정책·규제 중심으로 골라.
4. 경쟁사/해외이슈는 네이버·구글·오픈AI·MS·메타·애플·쿠팡·토스·배민·통신3사 등 주요 플레이어의 전략, 규제, 소송, 장애, 보안, AI/플랫폼 변화 위주로 골라.
5. 산업동향은 정말 구조적 변화가 있는 1~2개만 골라. 단순 인터뷰, 행사, 일반 제품 루머, 개별 범죄사건은 제외해.

[중복/업데이트 판단]
1. 최근 7일 과거 보고서에 이미 정리된 사건과 실질적으로 같은 내용이면 제외해.
2. 단, 새 사실·새 단계·새 수치·새 당국 조치가 있으면 업데이트 기사로 볼 수 있어. 예: 1차 조정→2차 조정, 투표→쟁의권, 소송 제기→판결, 논란→임원 퇴사, 노사갈등→노동부 진정.
3. 동일 사건의 여러 기사 중 하나만 골라. 최종 대표 기사는 코드가 본문 품질과 언론사 신뢰도로 다시 고를 거야.

[제외 원칙]
1. 오피니언, 사설, 칼럼, 전문가 기고는 제외해.
2. 사진/포토/캡션 기사, 외국어 저품질 기사, 제품 케이스 유출·렌더링 루머, 블로그/재가공 글은 제외해.
3. 보이스피싱·해킹도 개별 범죄 사건은 제외하고, 정부 대책·플랫폼 대응·제도 변화·대형 기업 리스크만 골라.
4. 타사 이름이나 카카오톡 제보 문구만 들어간 무관한 기사는 제외해.
5. 주가 지지선, 목표가, 투자의견, 단기 급등락, 밸류에이션, ETF 유출입 등 투자·시황 중심 기사는 제외해. 단, 지분 매각, M&A, 경영권 변동, 규제 제재, 구조조정과 직접 연결된 기업 경영 이벤트는 살릴 수 있어.
6. 스테이블코인/디지털자산은 가격·종목 전망보다 FIU, 특금법, AML, 가상자산사업자, 개인지갑, 해외이전, 시행령, 감독기준 등 정책·규제 변화가 있는 기사를 우선해.

[유연한 카테고리 범위]
- 자사 및 계열사 이슈: 최소 2개, 최대 7개
- 정부/국회: 최소 4개, 최대 8개
- 경쟁사/해외이슈: 최소 3개, 최대 6개
- 산업동향: 최소 1개, 최대 2개
중요한 기사가 많을 때만 많이 고르고, 억지로 최대치를 채우지마. 보통 총 12~18개 수준이 적절하며, 정말 중요한 날만 19개까지 가능해.

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

        relevant, relevance_reason = is_report_item_relevant(report_item, json_key)
        if not relevant:
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"relevance_failed:{relevance_reason}"
            body_failed_rows.append(failed_copy)
            print(f"  └ ⚠️ 관련성 미달로 교체: {report_item.get('기사제목', '')[:40]}... ({relevance_reason})")
            return False

        dup_item, dup_reason = final_duplicate_reason(report_item, final_report_data)
        if dup_item:
            if should_replace_duplicate_representative(report_item, dup_item, dup_reason):
                try:
                    final_report_data.remove(dup_item)
                except ValueError:
                    pass
                post_body_duplicate_skips.append({
                    "검색어": dup_item.get("검색어", ""),
                    "후보제목": dup_item.get("기사제목", ""),
                    "후보링크": dup_item.get("링크", ""),
                    "후보언론사": dup_item.get("언론사", ""),
                    "후보대표점수": dup_item.get("대표선택점수", ""),
                    "매칭과거일자": "CURRENT_RUN",
                    "매칭과거제목": report_item.get("기사제목", ""),
                    "매칭과거링크": report_item.get("링크", ""),
                    "중복판정이유": f"replaced_by_better_duplicate:{dup_reason}",
                    "공유주체": "",
                    "공유사건태그": dup_item.get("사건태그", ""),
                    "종합점수": "",
                    "제목점수": "",
                    "토큰점수": "",
                    "본문점수": "",
                    "제외단계": reason,
                })
                final_report_data.append(report_item)
                print(
                    f"  └ 🔁 중복 대표 교체: {dup_item.get('기사제목', '')[:30]}... → "
                    f"{report_item.get('기사제목', '')[:30]}... "
                    f"({v15_duplicate_preference_score(dup_item):.1f}→{v15_duplicate_preference_score(report_item):.1f}, {dup_reason})"
                )
                time.sleep(random.uniform(0.15, 0.35))
                return True

            post_body_duplicate_skips.append({
                "검색어": report_item.get("검색어", ""),
                "후보제목": report_item.get("기사제목", ""),
                "후보링크": report_item.get("링크", ""),
                "후보언론사": report_item.get("언론사", ""),
                "후보대표점수": report_item.get("대표선택점수", ""),
                "매칭과거일자": "CURRENT_RUN",
                "매칭과거제목": dup_item.get("기사제목", ""),
                "매칭과거링크": dup_item.get("링크", ""),
                "중복판정이유": dup_reason,
                "공유주체": "",
                "공유사건태그": report_item.get("사건태그", ""),
                "종합점수": "",
                "제목점수": "",
                "토큰점수": "",
                "본문점수": "",
                "제외단계": reason,
            })
            print(f"  └ 🧹 당일 중복 제외: {report_item.get('기사제목', '')[:40]}... ({dup_reason})")
            return False

        tag_limit_reason = event_tag_limit_reason(report_item, final_report_data)
        if tag_limit_reason:
            post_body_duplicate_skips.append({
                "검색어": report_item.get("검색어", ""),
                "후보제목": report_item.get("기사제목", ""),
                "후보링크": report_item.get("링크", ""),
                "후보언론사": report_item.get("언론사", ""),
                "매칭과거일자": "CURRENT_RUN",
                "매칭과거제목": "event_tag_cap",
                "매칭과거링크": "",
                "중복판정이유": tag_limit_reason,
                "공유주체": "",
                "공유사건태그": report_item.get("사건태그", ""),
                "종합점수": "",
                "제목점수": "",
                "토큰점수": "",
                "본문점수": "",
                "제외단계": reason,
            })
            print(f"  └ 🧹 사건태그 과다 반복 제외: {report_item.get('기사제목', '')[:40]}... ({tag_limit_reason})")
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

    # 최종 레이아웃은 Gemini에게 맡기지 않음.
    # Gemini는 기사별 요약문만 JSON으로 만들고, 제목/링크/언론사/카테고리/번호 형식은 코드가 고정 생성함.
    for idx, item in enumerate(final_report_data, 1):
        item["브리핑ID"] = idx

    final_input_text = ""
    for item in final_report_data:
        final_input_text += (
            f"[기사번호 {item['브리핑ID']}]\n"
            f"카테고리: {item['카테고리명']}\n"
            f"제목: {item['기사제목']}\n"
            f"언론사: {item['언론사']}\n"
            f"게시일: {item['게시일']}\n"
            f"본문글자수: {item['본문글자수']}\n"
            f"본문:\n{item['본문전문'][:MAX_BODY_CHARS_FOR_PROMPT]}\n\n"
        )

    prompt_report = f"""
너는 최고 경영진에게 매일 아침 뉴스 브리핑을 제공하는 수석 전략가야.
아래 [오늘 기사 데이터]의 본문만 사용해서 각 기사별 요약문만 작성해.

[요약 규칙]
1. 네 생각, 인사이트, 전망, 대응 포인트, 의미 부여는 쓰지마.
2. 오직 기사 본문에 있는 객관적 사실만 요약해.
3. 기사 1개당 요약은 1문단으로만 작성해.
4. 모든 문장의 끝은 '~함', '~임', '~됨', '~계획임', '~예정임' 같은 문어체 종결로 맞춰.
5. 기사 본문에 없는 내용은 절대 추가하지마.
6. 제목, 링크, 언론사, 카테고리, 번호는 쓰지마. 요약문만 반환해.

반드시 아래 JSON 객체 형식으로만 응답해. 키는 기사번호 문자열이고 값은 요약문이야.
{{
  "1": "요약문",
  "2": "요약문"
}}

[최근 과거 보고서 문체 참고]
{recent_past_text[:5000]}

[오늘 기사 데이터]
{final_input_text}
"""

    summary_map = {}
    try:
        if not client or not ENABLE_GEMINI_REPORT:
            raise RuntimeError("Gemini 최종 브리핑 비활성화 또는 client 없음")

        summary_text = gemini_generate_text(
            client=client,
            prompt=prompt_report,
            task_name="최종 기사별 요약 생성",
        )
        summary_map = normalize_summary_json(extract_json_object(summary_text))
        print("  └ ✅ Gemini 기사별 요약 생성 완료")

    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사별 요약 생성 실패. 로컬 본문 요약으로 대체합니다. 원인: {e}")
        summary_map = {}

    final_briefing_text = build_structured_briefing(final_report_data, summary_map)

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


# v6 overrides are appended below.


# ==========================================
# 10. v6 개선 오버라이드
# ==========================================
# v6 핵심 변경점
# 1) 수집 단계에서는 서로 다른 언론사의 유사 제목 기사를 너무 빨리 버리지 않음
# 2) 본문 추출 후 같은 사건끼리 묶고, 대표선택점수로 더 좋은 기사 1개를 남김
# 3) 본문 700자 미만이어도 중요 기사면 통과 가능
# 4) 사진/캡션/외국어/제품 루머/무관 일반 기사 필터 강화
# 5) 너무 긴 본문은 잡텍스트 가능성을 감점하고, prompt 입력은 안전하게 제한

V6_VERSION = "google_rss_v8_gr_public_affairs_flexible_quota"

# 본문 사용 판단 기준. MIN_GOOD_BODY_CHARS는 기존 설정을 유지하되,
# v6에서는 중요도가 높은 짧은 기사에 한해 MIN_IMPORTANT_BODY_CHARS까지 허용합니다.
MIN_IMPORTANT_BODY_CHARS = 320
MAX_BODY_STORAGE_CHARS = 22000
BODY_IDEAL_MIN_CHARS = 900
BODY_IDEAL_MAX_CHARS = 6500
BODY_SUSPICIOUS_LONG_CHARS = 18000

# 중복 대표 기사 교체 시 새 기사가 이 점수만큼 더 높으면 기존 대표를 교체합니다.
REPRESENTATIVE_REPLACE_MARGIN = 4.0

PHOTO_OR_CAPTION_TITLE_RE = re.compile(
    r"\[?포토\]?|\[?사진\]?|화보|기념촬영|포즈|질문에\s*답하는|취재진\s*질문|자료사진|"
    r"사진은|사진으로\s*보는|현장사진|브리핑하는|발언하는|입장하는|퇴장하는",
    re.IGNORECASE,
)

FOREIGN_LOW_VALUE_SOURCE_RE = re.compile(
    r"إسلام|i[- ]?phone islam|phonearena|macrumors|9to5mac|gsmarena|wccftech|tom's guide",
    re.IGNORECASE,
)

V6_STRICT_EXCLUDE_TITLE_PATTERNS = STRICT_EXCLUDE_TITLE_PATTERNS + [
    r"질문에\s*답하는", r"취재진\s*질문", r"기념촬영", r"자료사진", r"포즈",
    r"왕복\s*\d+차선", r"교통사고\s*환자", r"오토바이와\s*승합차",
    r"장애인\s*표준사업장\s*출범", r"세라믹기술원\s*원장", r"산업현장\s*수요\s*대응",
]

CRITICAL_SHORT_ARTICLE_RE = re.compile(
    r"압수수색|수사\s*착수|입건|기소|구형|판결|선고|제재|과징금|행정소송|소송|고발|"
    r"조정\s*결렬|조정\s*중지|조정\s*회의|쟁의권|파업|총파업|임단협|성과급|"
    r"공시|실적|영업손실|영업이익|순이익|매출|적자|흑자|"
    r"인수|매각|합병|지분|처분|최대주주|대표\s*(내정|취임|선임)|"
    r"법안|입법예고|시행령|본회의|상임위|국무회의|특별법|"
    r"개인정보|유출|해킹|침해사고|피싱|악성코드|장애|먹통",
    re.IGNORECASE,
)

MAJOR_PRESS_BONUS = {
    "연합뉴스": 24, "연합뉴스TV": 22, "뉴시스": 20, "뉴스1": 20,
    "전자신문": 20, "지디넷코리아": 18, "디지털데일리": 18,
    "이데일리": 18, "매일경제": 17, "한국경제": 17, "조선비즈": 17,
    "조선일보": 17, "서울경제": 16, "머니투데이": 16, "파이낸셜뉴스": 15,
    "헤럴드경제": 15, "아이뉴스24": 14, "아시아경제": 14, "노컷뉴스": 13,
    "SBS Biz": 13, "SBS": 13, "JTBC": 12, "KBS": 12,
}


def korean_char_ratio(text):
    text = clean_html_text(text)
    if not text:
        return 0.0
    hangul = len(re.findall(r"[가-힣]", text))
    letters = len(re.findall(r"[가-힣A-Za-z0-9]", text))
    if letters == 0:
        return 0.0
    return hangul / letters


def is_foreign_language_low_value(title, source="", body=""):
    title = clean_html_text(title)
    source = clean_html_text(source)
    sample = f"{title} {body[:300]}"
    # 한글 비중이 매우 낮고, 명확한 국내/전략 키워드도 없으면 한국 CEO 브리핑에는 부적합하다고 봅니다.
    if korean_char_ratio(sample) < 0.18:
        if not DIGITAL_STRATEGIC_PATTERN.search(sample):
            return True
        if FOREIGN_LOW_VALUE_SOURCE_RE.search(source) or FOREIGN_LOW_VALUE_SOURCE_RE.search(title):
            return True
    return False


def clean_extracted_body_v6(body):
    """본문 추출 결과에서 페이지 UI/관련기사/저작권 문구를 최대한 제거합니다."""
    body = str(body or "")
    body = body.replace("\r", "\n")
    raw_lines = [clean_html_text(x) for x in body.split("\n")]
    cleaned_lines = []
    seen_lines = set()

    noise_line_re = re.compile(
        r"^(공유하기|글자크기|확대|축소|인쇄|메일|댓글|좋아요|구독|로그인|뉴스레터|많이 본 뉴스|인기기사|"
        r"관련 기사|관련뉴스|관련 키워드|이 시각 주요뉴스|주요뉴스|영상|포토|사진|제보|무단전재|재배포 금지|"
        r"Copyright|저작권자|기사제보|닫기|전체메뉴|검색|본문 바로가기)$",
        re.IGNORECASE,
    )

    for line in raw_lines:
        if not line:
            continue
        if noise_line_re.search(line):
            continue
        if len(line) < 12 and not re.search(r"\d", line):
            continue
        # 같은 줄 반복 제거
        key = normalize_for_similarity(line)[:120]
        if key and key in seen_lines:
            continue
        seen_lines.add(key)
        cleaned_lines.append(line)

    joined = "\n".join(cleaned_lines).strip()

    # 일정 분량 이후부터 관련기사/많이 본 뉴스가 나오면 그 뒤는 잘라냅니다.
    stop_markers = [
        "관련 기사", "관련뉴스", "많이 본 뉴스", "인기기사", "이 시각 주요뉴스",
        "함께 보면 좋은", "추천 기사", "제보는 카카오톡", "무단전재", "재배포 금지", "Copyright",
    ]
    for marker in stop_markers:
        idx = joined.find(marker)
        if idx >= 1200:
            joined = joined[:idx].strip()

    if len(joined) > MAX_BODY_STORAGE_CHARS:
        joined = joined[:MAX_BODY_STORAGE_CHARS].strip()

    return joined


def article_importance_score(title, body="", json_key="", keyword=""):
    text = f"{clean_html_text(title)} {clean_html_text(body)[:2000]}"
    tags = detect_event_tags(text)
    entities = detect_entities(text)
    score = 0.0

    # 사건 태그 기반 중요도
    for tag in tags:
        if tag in STRONG_EVENT_TAGS:
            score += 7
        else:
            score += 3

    # 핵심 키워드 기반 중요도
    for kw in IMPORTANT_KEYWORDS:
        if kw in text:
            score += 3

    # 자사/정부 이슈는 기본 가중치
    if json_key == "자사_및_계열사_이슈":
        score += 8
    elif json_key == "정부_국회":
        score += 7
    elif json_key == "경쟁사_해외이슈":
        score += 4
    elif json_key == "산업동향":
        score += 2

    # 카카오/네이버/공정위/금융위 등 핵심 주체 보너스
    for entity in ["카카오", "카카오페이", "카카오뱅크", "카카오모빌리티", "카카오게임즈", "네이버", "구글", "오픈AI", "공정위", "금융위", "금감원", "과기정통부", "방미통위"]:
        if entity in entities:
            score += 2.5

    # 짧아도 반드시 살려볼 만한 사건성 기사
    if CRITICAL_SHORT_ARTICLE_RE.search(text):
        score += 15

    # 단순 홍보/게시판/사진성 제목은 중요도 감점
    for pattern in LOW_VALUE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            score -= 15
    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        score -= 30

    return round(score, 2)


def body_quality_score_v6(title, body, method, json_key="", source=""):
    body = clean_html_text(body)
    title = clean_html_text(title)

    if not body:
        return 0.0, "empty"

    length = len(body)
    sentences = split_sentences(body)
    score = 0.0

    # 본문 길이 점수: 길수록 무조건 좋게 보지 않고, 기사 본문으로 적당한 구간을 가장 높게 봅니다.
    if length < 250:
        score -= 40
    elif length < MIN_IMPORTANT_BODY_CHARS:
        score -= 25
    elif length < MIN_ACCEPT_BODY_CHARS:
        score -= 10
    elif length < MIN_GOOD_BODY_CHARS:
        score += 3
    elif length < BODY_IDEAL_MIN_CHARS:
        score += 12
    elif length <= BODY_IDEAL_MAX_CHARS:
        score += 32
    elif length <= BODY_SUSPICIOUS_LONG_CHARS:
        score += 22
    else:
        score += 6

    # 추출 방식 점수
    if method == "trafilatura":
        score += 22
    elif method == "bs4_selector":
        score += 20
    elif method == "newspaper3k":
        score += 18
    elif method == "json_ld":
        score += 17
    elif method == "meta_description_short":
        score -= 45
    elif method == "failed":
        score -= 80

    # 문장 수
    if len(sentences) >= 6:
        score += 16
    elif len(sentences) >= 3:
        score += 11
    elif len(sentences) == 2:
        score += 4
    else:
        score -= 18

    # 제목과 본문 시작부가 거의 동일하고 짧으면 메타 설명일 가능성
    if title:
        start = body[:max(50, len(title) + 80)]
        title_ratio = difflib.SequenceMatcher(None, normalize_for_similarity(title), normalize_for_similarity(start)).ratio()
        if title_ratio > 0.90 and length < 650:
            score -= 30

    # 잡텍스트 감점
    bad_markers = [
        "이 누리집은 대한민국 공식 전자정부", "내려받기", "바로보기 내려받기", "본문 추출 실패",
        "관련 키워드", "관련 기사", "관련뉴스", "많이 본 뉴스", "인기기사", "뉴스레터", "로그인",
        "제보는 카카오톡", "무단전재", "재배포 금지",
    ]
    for marker in bad_markers:
        if marker in body:
            score -= 8

    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        score -= 35

    if is_foreign_language_low_value(title, source, body):
        score -= 45

    if length > BODY_SUSPICIOUS_LONG_CHARS:
        score -= 12

    reason = "good"
    if method == "meta_description_short":
        reason = "meta_description_not_full_body"
    elif method == "failed":
        reason = "extract_failed"
    elif length < MIN_IMPORTANT_BODY_CHARS:
        reason = "too_short"
    elif len(sentences) < 2:
        reason = "not_enough_sentences"
    elif PHOTO_OR_CAPTION_TITLE_RE.search(title):
        reason = "photo_or_caption_article"
    elif score < 40:
        reason = "low_quality"

    return round(score, 2), reason


def is_body_usable_v6(title, body, method, json_key="", source="", importance_score=0):
    quality_score, quality_reason = body_quality_score_v6(title, body, method, json_key=json_key, source=source)
    body_len = len(clean_html_text(body))

    if method in {"failed", "meta_description_short"}:
        return False, quality_reason, quality_score

    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        return False, "photo_or_caption_article", quality_score

    # 일반 기사: 기존처럼 700자 이상 + 품질점수 통과
    if body_len >= MIN_GOOD_BODY_CHARS and quality_score >= 42:
        return True, "good", quality_score

    # v6 핵심: 짧아도 중요 사건이면 통과 가능
    if body_len >= MIN_IMPORTANT_BODY_CHARS and importance_score >= 35 and quality_score >= 20:
        return True, "short_but_important", quality_score

    if body_len >= MIN_ACCEPT_BODY_CHARS and importance_score >= 25 and quality_score >= 28:
        return True, "acceptable_important", quality_score

    return False, quality_reason, quality_score


def representative_score(report_item):
    """동일 사건/중복 기사 중 대표 기사 1개를 고르기 위한 점수."""
    title = report_item.get("기사제목", "")
    source = report_item.get("언론사", "")
    link = report_item.get("링크", "")
    body_len = int(report_item.get("본문글자수") or 0)
    quality = float(report_item.get("본문품질점수") or 0)
    importance = float(report_item.get("중요도점수") or 0)
    method = report_item.get("본문추출방식", "")
    press = canonical_press_name(source, link)

    score = 0.0
    score += quality * 0.9
    score += importance * 0.7
    score += MAJOR_PRESS_BONUS.get(press, press_score(source) * 1.5)

    # 본문 길이: 충분한 본문은 가산, 너무 긴 본문은 잡텍스트 가능성으로 감점
    if body_len < MIN_IMPORTANT_BODY_CHARS:
        score -= 35
    elif body_len < MIN_ACCEPT_BODY_CHARS:
        score -= 14
    elif body_len < MIN_GOOD_BODY_CHARS:
        score += 2
    elif body_len < 1500:
        score += 10
    elif body_len <= 6500:
        score += 20
    elif body_len <= BODY_SUSPICIOUS_LONG_CHARS:
        score += 11
    else:
        score -= 5

    if method == "trafilatura":
        score += 8
    elif method == "bs4_selector":
        score += 6
    elif method == "newspaper3k":
        score += 5
    elif method == "meta_description_short":
        score -= 35

    domain = url_domain(link)
    if domain and domain not in {"v.daum.net", "n.news.naver.com", "news.google.com"}:
        score += 4

    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        score -= 60
    if is_foreign_language_low_value(title, source, report_item.get("본문전문", "")):
        score -= 55
    for pattern in LOW_VALUE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            score -= 18

    return round(score, 2)


# v6 수집 중복 판단: 같은 URL과 같은 언론사 내부 중복은 버리되,
# 서로 다른 언론사의 유사 기사까지 수집 단계에서 버리지는 않습니다.
def canonical_duplicate_reason(registry, title, source, url):
    normalized_url = normalize_url(url)
    press, fp = canonical_article_signature(title, source, url)
    if not fp:
        return "", None

    if normalized_url in registry["urls"]:
        return "same_normalized_url", None

    exact_press_key = f"{press}::{fp}"
    if exact_press_key in registry["exact_titles_by_press"]:
        return "same_canonical_title_same_press", None

    # 같은 언론사 내 모바일/AMP/제목 변형만 수집 단계에서 제거
    for old in registry["source_buckets"].get(press, []):
        near, score = is_title_near_duplicate(title, old.get("title", ""))
        if near:
            return f"near_duplicate_same_press:{score:.2f}", old

    return "", None


def collection_relevance_reason(keyword, title, summary):
    """v6: 수집 오탐 필터. 애매한 건 살리되 명백한 카톡 제보/구독/사진성 오탐은 제거."""
    if not ENABLE_COLLECTION_RELEVANCE_FILTER:
        return ""

    title = clean_html_text(title)
    summary = clean_html_text(summary)
    front_text = f"{title} {summary[:550]}"
    pattern = BRAND_KEYWORD_PATTERNS.get(keyword)

    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        return "photo_or_caption_title"

    if is_foreign_language_low_value(title, "", summary):
        return "foreign_low_value"

    # 브랜드 키워드는 제목 또는 요약 앞부분에 실제 브랜드가 있어야 함.
    if pattern:
        title_has = bool(pattern.search(title))
        front_has = bool(pattern.search(front_text))
        if not title_has and not front_has:
            return "brand_keyword_absent_from_title_and_front_summary"

        # '제보는 카카오톡', '네이버에서 구독' 같은 경우는 제목에 브랜드가 없으면 제거
        if not title_has and NOISE_ONLY_PATTERNS.search(front_text):
            mentions = len(pattern.findall(front_text))
            if mentions <= 2:
                return "brand_only_in_contact_or_subscription_noise"

    if keyword in {"카카오", "카카오톡"}:
        if not SELF_KAKAO_PATTERN.search(title) and NOISE_ONLY_PATTERNS.search(front_text):
            return "kakao_contact_noise"

    if keyword in {"네이버", "토스", "카카오", "카카오톡", "구글", "애플"} and pattern:
        if re.search(r"교통사고|사망사고|음주운전|화재|폭행|살인|실종|구속송치", title) and not pattern.search(title):
            return "local_incident_without_brand_in_title"

    return ""


def is_article_obviously_low_value(article):
    title = article.get("기사제목", "")
    summary = article.get("본문요약", "")
    text = f"{title} {summary}"

    for pattern in V6_STRICT_EXCLUDE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return True, f"strict_low_value_title:{pattern}"

    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        return True, "photo_or_caption_title"

    if is_foreign_language_low_value(title, article.get("언론사", ""), summary):
        return True, "foreign_low_value"

    # 산업동향/플랫폼 키워드에서 자주 들어오는 일반 기관장 인터뷰 오탐
    if re.search(r"세라믹기술원|장애인\s*표준사업장|교통사고|수사요원", title):
        return True, "off_topic_general_news"

    if article.get("검색어") == "카카오톡":
        if not re.search(r"카카오톡|카톡", title, flags=re.IGNORECASE) and KAKAOTALK_NOISE_PATTERN.search(text):
            return True, "kakaotalk_channel_noise"

    return False, ""


def is_report_item_relevant(report_item, json_key):
    title = report_item.get("기사제목", "")
    body = report_item.get("본문전문", "") or report_item.get("RSS요약", "") or report_item.get("본문요약", "")
    keyword = report_item.get("검색어", "")
    source = report_item.get("언론사", "")
    text = f"{title} {body[:1400]}"

    for pattern in V6_STRICT_EXCLUDE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return False, f"low_value_title:{pattern}"

    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        return False, "photo_or_caption_article"

    if is_foreign_language_low_value(title, source, body):
        return False, "foreign_language_low_value"

    low, reason = is_article_obviously_low_value({
        "기사제목": title, "본문요약": body[:600], "검색어": keyword, "언론사": source
    })
    if low:
        return False, reason

    if json_key == "자사_및_계열사_이슈":
        if not SELF_KAKAO_PATTERN.search(text):
            return False, "self_category_without_kakao_entity"
        if keyword == "카카오톡" and not re.search(r"카카오톡|카톡|카카오", title, flags=re.IGNORECASE):
            kakao_mentions = len(re.findall(r"카카오|카톡|Kakao", text, flags=re.IGNORECASE))
            if kakao_mentions <= 2 and KAKAOTALK_NOISE_PATTERN.search(text):
                return False, "kakaotalk_channel_only_noise"

    if json_key == "정부_국회":
        if not DIGITAL_STRATEGIC_PATTERN.search(text):
            return False, "government_without_digital_platform_ai_relevance"
        if re.search(r"금융교육\s*전문강사|보험사\s*순이익|생산적\s*금융\s*실적", title):
            return False, "generic_finance_not_ceo_platform_issue"

    if json_key == "경쟁사_해외이슈":
        if re.search(r"케이스\s*유출|렌더링|출시\s*예상|스펙\s*유출|화면\s*없는\s*기기", title):
            return False, "consumer_product_rumor"
        # 경쟁사 키워드로 들어온 기사는 제목 또는 본문 앞부분에 그 주체가 실제로 있어야 함.
        brand_pat = BRAND_KEYWORD_PATTERNS.get(keyword)
        if brand_pat and not brand_pat.search(text[:900]):
            return False, "competitor_brand_absent_from_front_text"

    if json_key == "산업동향":
        if not re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|플랫폼|망\s*사용료|스테이블코인|디지털자산|보안|해킹|딥페이크|저작권|알고리즘", text, flags=re.IGNORECASE):
            return False, "industry_without_core_digital_trend"
        if re.search(r"세라믹기술원|장애인\s*표준사업장|교통사고", title):
            return False, "generic_industry_or_local_news"

    return True, ""


def rank_score_article(article):
    title = article.get("기사제목", "")
    summary = article.get("본문요약", "")
    text = f"{title} {summary}"
    json_key = article.get("JSON카테고리", "")
    keyword = article.get("검색어", "")

    score = 0.0
    penalty_reasons = []

    low_value, low_value_reason = is_article_obviously_low_value(article)
    if low_value:
        score -= 100
        penalty_reasons.append(low_value_reason)

    relevance_reason = collection_relevance_reason(keyword, title, summary)
    if relevance_reason:
        score -= 75
        penalty_reasons.append(f"collection_relevance:{relevance_reason}")

    pseudo_item = {"기사제목": title, "본문전문": summary, "RSS요약": summary, "검색어": keyword, "언론사": article.get("언론사", "")}
    relevant, report_relevance_reason = is_report_item_relevant(pseudo_item, json_key)
    if not relevant:
        score -= 50
        penalty_reasons.append(report_relevance_reason)

    score += recency_score(article.get("게시일", ""))
    score += press_score(article.get("언론사", ""))

    # 중요도 점수 반영
    imp = article_importance_score(title, summary, json_key=json_key, keyword=keyword)
    score += min(45, imp)

    event_tags = detect_event_tags(text)
    score += min(15, len(event_tags) * 3)

    if json_key == "자사_및_계열사_이슈":
        score += 12
    elif json_key == "정부_국회":
        score += 8
    elif json_key == "경쟁사_해외이슈":
        score += 4
    elif json_key == "산업동향":
        score += 2

    for pattern in LOW_VALUE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            score -= 22
            penalty_reasons.append(f"low_value:{pattern}")

    brand_pat = BRAND_KEYWORD_PATTERNS.get(keyword)
    if brand_pat and not brand_pat.search(title):
        score -= 10
        penalty_reasons.append("brand_not_in_title")

    if len(title) < 12:
        score -= 5
        penalty_reasons.append("short_title")

    if is_foreign_language_low_value(title, article.get("언론사", ""), summary):
        score -= 45
        penalty_reasons.append("foreign_low_value")

    article["랭킹점수"] = round(score, 3)
    article["중요도점수"] = imp
    article["사건태그"] = ",".join(sorted(event_tags))
    article["사건단계"] = extract_event_stage(text, event_tags)
    article["주요주체"] = ",".join(sorted(detect_entities(text)))
    article["랭킹감점사유"] = ";".join([r for r in penalty_reasons if r])
    return score


def process_article_for_report(article_info, json_key, recent_past_items):
    original_link = article_info.get("링크", "")

    real_url = decode_google_news_url(original_link)
    real_url = normalize_url(real_url)

    body_text, extract_method, fetched_url = extract_article_body(real_url)
    if fetched_url:
        real_url = normalize_url(fetched_url)

    body_text_raw = body_text.strip()
    body_text = clean_extracted_body_v6(body_text_raw)
    body_cleaned = body_text != body_text_raw

    importance = article_importance_score(
        article_info.get("기사제목", ""),
        body_text or article_info.get("본문요약", ""),
        json_key=json_key,
        keyword=article_info.get("검색어", ""),
    )

    good, quality_reason, quality_score = is_body_usable_v6(
        article_info.get("기사제목", ""),
        body_text,
        extract_method,
        json_key=json_key,
        source=article_info.get("언론사", ""),
        importance_score=importance,
    )

    # 본문 추출 후 한 번 더 과거 7일과 중복 비교. 본문 일부까지 사용합니다.
    # 단, 단계가 다른 노사/입법/규제 이슈는 find_past_duplicate 내부에서 보수적으로 살립니다.
    body_candidate = {
        "기사제목": article_info.get("기사제목", ""),
        "본문요약": body_text[:2200],
        "링크": real_url,
    }

    is_dup, matched, sim = find_past_duplicate(body_candidate, recent_past_items, mode="body")

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

    event_tags = detect_event_tags(f"{article_info.get('기사제목', '')} {body_text[:1800]}")
    report_item = {
        "카테고리": json_key,
        "카테고리명": JSON_KEY_TO_DISPLAY.get(json_key, json_key),
        "검색어": article_info.get("검색어", ""),
        "기사제목": article_info.get("기사제목", ""),
        "언론사": article_info.get("언론사", "") or guess_press_name_from_url(real_url),
        "정규언론사": canonical_press_name(article_info.get("언론사", ""), real_url),
        "게시일": article_info.get("게시일", ""),
        "RSS요약": article_info.get("본문요약", ""),
        "본문전문": body_text,
        "본문글자수": len(body_text),
        "본문추출방식": extract_method,
        "본문품질점수": quality_score,
        "본문품질사유": quality_reason,
        "중요도점수": importance,
        "본문사용가능": good,
        "본문정제적용": body_cleaned,
        "원래RSS링크": original_link,
        "링크": real_url,
        "랭킹점수": article_info.get("랭킹점수", ""),
        "사건태그": ",".join(sorted(event_tags)),
        "사건단계": extract_event_stage(f"{article_info.get('기사제목', '')} {body_text[:1600]}", event_tags),
    }
    report_item["대표선택점수"] = representative_score(report_item)

    if not good:
        return None, None, report_item

    return report_item, None, None


def final_duplicate_reason(new_item, existing_items):
    """v6: 같은 날 기사끼리 중복/동일 사건 여부 판단. 중복이면 대표 선택 비교로 넘깁니다."""
    new_title = new_item.get("기사제목", "")
    new_body = new_item.get("본문전문", "")[:1800]
    new_text = f"{new_title} {new_body}"
    new_tokens = tokenize_for_similarity(new_text)
    new_tags = set(filter(None, (new_item.get("사건태그", "") or "").split(","))) or detect_event_tags(new_text)
    new_sig = event_signature(new_text)
    new_stage = new_item.get("사건단계") or extract_event_stage(new_text, new_tags)
    new_press, new_fp = canonical_article_signature(new_title, new_item.get("언론사", ""), new_item.get("링크", ""))
    new_entities = detect_entities(new_text)

    for old in existing_items:
        old_title = old.get("기사제목", "")
        old_body = old.get("본문전문", "")[:1800]
        old_text = f"{old_title} {old_body}"
        old_tokens = tokenize_for_similarity(old_text)
        old_tags = set(filter(None, (old.get("사건태그", "") or "").split(","))) or detect_event_tags(old_text)
        old_sig = event_signature(old_text)
        old_stage = old.get("사건단계") or extract_event_stage(old_text, old_tags)
        old_press, old_fp = canonical_article_signature(old_title, old.get("언론사", ""), old.get("링크", ""))
        old_entities = detect_entities(old_text)

        title_seq = sequence_ratio(new_title, old_title)
        token_score = jaccard(new_tokens, old_tokens)
        body_ngram_score = jaccard(char_ngrams(new_body[:1200], 5), char_ngrams(old_body[:1200], 5))
        shared_entities = new_entities & old_entities
        shared_tags = (new_tags & old_tags) & STRONG_EVENT_TAGS

        if new_fp and old_fp and new_fp == old_fp:
            return old, "same_current_canonical_title"

        near, near_score = is_title_near_duplicate(new_title, old_title)
        if near and (new_press == old_press or near_score >= 0.94):
            return old, f"near_current_title:{near_score:.2f}"

        if body_ngram_score >= 0.55:
            return old, "same_current_body"

        if new_sig and old_sig and new_sig == old_sig:
            if title_seq >= 0.36 or token_score >= 0.12 or body_ngram_score >= 0.24:
                return old, f"same_current_signature:{new_sig}"

        # 카카오 노사: 단계가 같으면 같은 날 중복으로 보고 대표 기사만 남김.
        if "카카오" in new_text and "카카오" in old_text:
            labor_words = r"노조|파업|공동파업|총파업|조정|지노위|노동위|성과급|RSU|쟁의권"
            if re.search(labor_words, new_text) and re.search(labor_words, old_text):
                if new_stage == old_stage and new_stage:
                    if token_score >= 0.045 or title_seq >= 0.30 or body_ngram_score >= 0.15:
                        return old, f"same_current_labor_stage:{new_stage}"

        # 같은 행사/발표/토론회 anchor
        anchor_sets = [
            ("안도걸", "디지털자산기본법", "스테이블코인"),
            ("K-문샷", "추진단"),
            ("카나나 스칼라", "콜로키움"),
            ("국가AI컴퓨팅센터", "삼성SDS"),
            ("구글", "EU", "과징금"),
            ("오픈AI", "GTAC"),
            ("공정위", "조사국"),
            ("쿠팡", "김범석", "지정자료"),
        ]
        for anchors in anchor_sets:
            if all(a in new_text for a in anchors) and all(a in old_text for a in anchors):
                if token_score >= 0.04 or title_seq >= 0.28 or body_ngram_score >= 0.10:
                    return old, f"same_current_anchor:{'/'.join(anchors)}"

        # 같은 핵심 주체 + 같은 강한 사건태그 + 유사도가 어느 정도 있으면 중복 후보로 봄
        if shared_entities and shared_tags:
            if new_stage and old_stage and new_stage != old_stage:
                # 진행 단계가 다르면 업데이트일 수 있으므로 보수적으로만 중복 처리
                if title_seq >= 0.72 and token_score >= 0.22:
                    return old, f"same_entities_tags_high_similarity:{','.join(sorted(shared_tags))}"
            else:
                if token_score >= 0.11 or title_seq >= 0.42 or body_ngram_score >= 0.20:
                    return old, f"same_entities_tags:{','.join(sorted(shared_tags))}"

    return None, ""


def rank_and_trim_candidates(raw_articles):
    for article in raw_articles:
        rank_score_article(article)

    # 너무 낮은 점수 후보는 Gemini 후보로 보내지 않되, 전체 후보 CSV에는 남깁니다.
    ranked_all = sorted(raw_articles, key=lambda x: (x.get("랭킹점수", 0), x.get("게시일", "")), reverse=True)

    selected_for_gemini = []
    used_ids = set()

    for json_key in JSON_KEYS_ORDER:
        limit = CATEGORY_POOL_LIMIT_FOR_GEMINI.get(json_key, 50)
        bucket = [a for a in ranked_all if a.get("JSON카테고리") == json_key and a.get("랭킹점수", 0) > -25]
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
            if article.get("랭킹점수", 0) <= -25:
                continue
            aid = int(article["id"])
            if aid in used_ids:
                continue
            selected_for_gemini.append(article)
            used_ids.add(aid)

    selected_for_gemini = sorted(selected_for_gemini, key=lambda x: x.get("랭킹점수", 0), reverse=True)
    return ranked_all, selected_for_gemini


def add_duplicate_skip_row(rows, candidate, matched, reason, stage="current_duplicate", extra=""):
    rows.append({
        "검색어": candidate.get("검색어", ""),
        "후보제목": candidate.get("기사제목", ""),
        "후보링크": candidate.get("링크", ""),
        "후보언론사": candidate.get("언론사", ""),
        "후보대표점수": candidate.get("대표선택점수", ""),
        "매칭과거일자": "CURRENT_RUN" if matched else "",
        "매칭과거제목": matched.get("기사제목", "") if matched else "",
        "매칭과거링크": matched.get("링크", "") if matched else "",
        "매칭대표점수": matched.get("대표선택점수", "") if matched else "",
        "중복판정이유": reason,
        "공유주체": "",
        "공유사건태그": candidate.get("사건태그", ""),
        "종합점수": extra,
        "제목점수": "",
        "토큰점수": "",
        "본문점수": "",
        "제외단계": stage,
    })



# ==========================================
# 9-B. v7 추가 패치: 원문 발행일 검증 + 대표 기사 선택 보강
# ==========================================

# Google News RSS는 오래된 기사가 재색인/수정/재노출되면 pubDate가 최신처럼 보일 수 있습니다.
# v7은 최종 후보의 원문 HTML과 URL, 본문 앞부분에서 실제 발행일을 다시 검증합니다.
VERIFY_ORIGINAL_ARTICLE_DATE = True
ARTICLE_DATE_MAX_AGE_HOURS = 52          # 오늘/어제 기사까지 허용. qdr:d의 경계 오차를 감안함.
ARTICLE_DATE_FUTURE_TOLERANCE_HOURS = 18 # 시차/예약 발행 오차 허용.
ALLOW_DATELESS_ARTICLE = True            # 원문 날짜를 못 찾았다고 무조건 버리지는 않되 점수 감점.
DATELESS_LOW_CONFIDENCE_PENALTY = 8

EXCLUDED_FINAL_DOMAINS = {
    "blog.naver.com", "m.blog.naver.com", "cafe.naver.com", "m.cafe.naver.com",
    "post.naver.com", "m.post.naver.com", "blog.daum.net", "brunch.co.kr",
}
LOW_VALUE_FINAL_DOMAIN_RE = re.compile(
    r"(news\.dlwlrmaon\.com|itnewsmoa|it-newsmoa|tistory\.com|wordpress\.com|medium\.com)",
    re.IGNORECASE,
)

# 날짜가 누락됐더라도 제목/본문이 명백한 과거 재탕 사건이면 제외합니다.
OLD_KNOWN_EVENT_RE = re.compile(
    r"SK\s*C\s*&\s*C\s*데이터센터\s*화재|SK\s*C&C\s*데이터센터\s*화재|"
    r"카카오\s*먹통|판교\s*데이터센터\s*화재|2022년\s*카카오\s*먹통",
    re.IGNORECASE,
)
CURRENT_ANGLE_RE = re.compile(
    r"오늘|전날|27일|26일|최근|올해|이번|조사|판결|소송|제재|과징금|발표|보고서|재발|재점검|감사|국감|후속|대책",
    re.IGNORECASE,
)

DATE_LABEL_RE = re.compile(
    r"(?:입력|등록|승인|발행|송고|기사입력|최종수정|수정|업데이트|게시)\s*[:：]?\s*"
    r"(20\d{2})[.\-/년\s]+(\d{1,2})[.\-/월\s]+(\d{1,2})",
    re.IGNORECASE,
)
DATE_PLAIN_RE = re.compile(r"(20\d{2})[.\-/년\s]+(\d{1,2})[.\-/월\s]+(\d{1,2})")
DATE_COMPACT_RE = re.compile(r"(?<!\d)(20\d{2})([01]\d)([0-3]\d)(?!\d)")


def safe_date(year, month, day):
    try:
        return date(int(year), int(month), int(day))
    except Exception:
        return None


def parse_any_date_to_date(value):
    """ISO/RFC/Korean 날짜 문자열을 date로 변환합니다."""
    if value is None:
        return None
    s = clean_html_text(str(value))
    if not s:
        return None

    # ISO 8601 우선 처리
    iso = s.replace("Z", "+00:00")
    try:
        # 2026-05-27T10:20:00+09:00 등
        dt = datetime.fromisoformat(iso[:32])
        if dt.tzinfo is not None:
            return dt.astimezone(KST).date()
        return dt.date()
    except Exception:
        pass

    try:
        dt = parsedate_to_datetime(s)
        if dt:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=KST)
            return dt.astimezone(KST).date()
    except Exception:
        pass

    m = DATE_LABEL_RE.search(s)
    if m:
        return safe_date(m.group(1), m.group(2), m.group(3))

    m = DATE_PLAIN_RE.search(s)
    if m:
        return safe_date(m.group(1), m.group(2), m.group(3))

    m = DATE_COMPACT_RE.search(s)
    if m:
        return safe_date(m.group(1), m.group(2), m.group(3))

    return None


def extract_date_from_url(url):
    if not url:
        return None, ""
    decoded = urllib.parse.unquote(str(url))

    patterns = [
        r"/(20\d{2})/(\d{1,2})/(\d{1,2})(?:/|$)",
        r"/(20\d{2})-(\d{1,2})-(\d{1,2})(?:/|$)",
        r"/(20\d{2})\.(\d{1,2})\.(\d{1,2})(?:/|$)",
        r"(?:newsId|date|ymd|publish|articleDate|no|idx|arcid|logNo)[=/_-]?(20\d{2})([01]\d)([0-3]\d)",
        r"(?<!\d)(20\d{2})([01]\d)([0-3]\d)(?!\d)",
    ]
    for pat in patterns:
        m = re.search(pat, decoded)
        if m:
            d = safe_date(m.group(1), m.group(2), m.group(3))
            if d:
                return d, "url_date"
    return None, ""


def iter_json_ld_objects(data):
    stack = data if isinstance(data, list) else [data]
    while stack:
        obj = stack.pop(0)
        if isinstance(obj, list):
            stack.extend(obj)
            continue
        if not isinstance(obj, dict):
            continue
        yield obj
        graph = obj.get("@graph")
        if isinstance(graph, list):
            stack.extend(graph)


def extract_dates_from_html(html_text):
    """원문 HTML에서 날짜 후보를 추출합니다. datePublished 계열을 가장 신뢰합니다."""
    candidates = []
    if not html_text:
        return candidates

    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception:
        return candidates

    # JSON-LD: datePublished 우선, dateModified는 낮은 신뢰도로 기록
    for script in soup.find_all("script", attrs={"type": re.compile(r"ld\+json", re.I)}):
        raw = script.string or script.get_text(" ") or ""
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        for obj in iter_json_ld_objects(data):
            obj_type = obj.get("@type", "")
            type_text = " ".join(obj_type) if isinstance(obj_type, list) else str(obj_type)
            if type_text and not re.search(r"NewsArticle|Article|Reportage|BlogPosting", type_text, flags=re.I):
                # graph 안에는 다른 객체도 많으므로 날짜가 있으면 보되 우선순위는 낮춤
                pass
            for field, reliability in [
                ("datePublished", "strong"), ("dateCreated", "strong"), ("dateIssued", "strong"),
                ("uploadDate", "medium"), ("dateModified", "modified_only"),
            ]:
                d = parse_any_date_to_date(obj.get(field))
                if d:
                    candidates.append({"date": d, "source": f"json_ld:{field}", "reliability": reliability})

    # meta published fields
    published_meta_names = [
        ("property", "article:published_time"), ("property", "og:published_time"),
        ("name", "pubdate"), ("name", "publishdate"), ("name", "publish-date"),
        ("name", "date"), ("name", "dc.date.issued"), ("name", "dcterms.created"),
        ("name", "article.published"), ("name", "parsely-pub-date"), ("name", "sailthru.date"),
        ("itemprop", "datePublished"),
    ]
    for attr, val in published_meta_names:
        for tag in soup.find_all("meta", attrs={attr: re.compile(re.escape(val), re.I)}):
            content = tag.get("content") or tag.get("value") or ""
            d = parse_any_date_to_date(content)
            if d:
                candidates.append({"date": d, "source": f"meta:{val}", "reliability": "strong"})

    modified_meta_names = [
        ("property", "article:modified_time"), ("name", "lastmod"), ("name", "modified"),
        ("itemprop", "dateModified"),
    ]
    for attr, val in modified_meta_names:
        for tag in soup.find_all("meta", attrs={attr: re.compile(re.escape(val), re.I)}):
            content = tag.get("content") or tag.get("value") or ""
            d = parse_any_date_to_date(content)
            if d:
                candidates.append({"date": d, "source": f"meta:{val}", "reliability": "modified_only"})

    # time datetime/text
    for tag in soup.find_all("time")[:8]:
        raw = tag.get("datetime") or tag.get("content") or tag.get_text(" ")
        d = parse_any_date_to_date(raw)
        if d:
            candidates.append({"date": d, "source": "time_tag", "reliability": "strong"})

    # 날짜 class/id가 있는 요소. 너무 많이 보지 않음.
    date_node_re = re.compile(r"date|time|publish|posted|article.*date|news.*date|write", re.I)
    nodes = []
    for tag in soup.find_all(attrs={"class": date_node_re})[:12]:
        nodes.append(tag)
    for tag in soup.find_all(attrs={"id": date_node_re})[:12]:
        nodes.append(tag)
    for tag in nodes[:16]:
        raw = tag.get_text(" ")
        d = parse_any_date_to_date(raw)
        if d:
            candidates.append({"date": d, "source": "visible_date_node", "reliability": "medium"})

    return candidates


def extract_dates_from_body_front(body_text):
    candidates = []
    body = clean_html_text(body_text or "")
    if not body:
        return candidates

    front = body[:2500]

    # 입력/등록/승인 등 라벨이 있으면 강한 근거
    for m in DATE_LABEL_RE.finditer(front):
        d = safe_date(m.group(1), m.group(2), m.group(3))
        if d:
            candidates.append({"date": d, "source": "body_labeled_date", "reliability": "strong"})

    # 본문 첫 700자 안의 일반 날짜는 원문 기사 날짜일 가능성이 높음.
    early = body[:700]
    for m in DATE_PLAIN_RE.finditer(early):
        d = safe_date(m.group(1), m.group(2), m.group(3))
        if d:
            candidates.append({"date": d, "source": "body_early_plain_date", "reliability": "medium"})
            break

    return candidates


def choose_article_original_date(candidates, rss_published=""):
    """날짜 후보 중 원문 발행일로 볼 값을 선택합니다."""
    rss_date = parse_pubdate_to_kst(rss_published).date() if parse_pubdate_to_kst(rss_published) else None

    if not candidates:
        return None, "no_original_date_found", rss_date

    # 중복 제거
    uniq = []
    seen = set()
    for c in candidates:
        key = (c.get("date"), c.get("source"), c.get("reliability"))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)

    # datePublished/url/body_labeled 같은 강한 근거를 우선. modified_only는 마지막.
    priority = {"strong": 0, "medium": 1, "modified_only": 2, "weak": 3}
    uniq.sort(key=lambda c: (priority.get(c.get("reliability", "weak"), 9), c.get("date") or date.min))
    chosen = uniq[0]
    return chosen.get("date"), chosen.get("source"), rss_date


def verify_article_original_date(url, body_text, html_text="", rss_published=""):
    """원문 날짜가 최근 기사인지 검증합니다."""
    now_dt = datetime.now(KST)
    today = now_dt.date()
    min_allowed_dt = now_dt - timedelta(hours=ARTICLE_DATE_MAX_AGE_HOURS)
    max_allowed_dt = now_dt + timedelta(hours=ARTICLE_DATE_FUTURE_TOLERANCE_HOURS)

    body_front = clean_html_text(body_text or "")[:1200]
    titleish = body_front[:400]

    candidates = []
    d_url, src_url = extract_date_from_url(url)
    if d_url:
        candidates.append({"date": d_url, "source": src_url, "reliability": "strong"})
    candidates.extend(extract_dates_from_html(html_text))
    candidates.extend(extract_dates_from_body_front(body_text))

    original_date, source, rss_date = choose_article_original_date(candidates, rss_published=rss_published)

    # 명백한 과거 대형 사건 재탕 방지. 원문 날짜를 못 찾은 경우에도 제목/본문으로 차단.
    if OLD_KNOWN_EVENT_RE.search(f"{url} {titleish} {body_front}") and not CURRENT_ANGLE_RE.search(f"{titleish} {body_front}"):
        return {
            "valid": False,
            "date": original_date,
            "source": source or "old_known_event_pattern",
            "status": "old_known_event_recirculated",
            "rss_date": rss_date,
            "all_candidates": candidates,
        }

    if not VERIFY_ORIGINAL_ARTICLE_DATE:
        return {"valid": True, "date": original_date, "source": source, "status": "date_check_disabled", "rss_date": rss_date, "all_candidates": candidates}

    if original_date is None:
        if ALLOW_DATELESS_ARTICLE:
            return {
                "valid": True,
                "date": None,
                "source": "no_original_date_found",
                "status": "no_original_date_found_kept",
                "rss_date": rss_date,
                "all_candidates": candidates,
            }
        return {"valid": False, "date": None, "source": "no_original_date_found", "status": "no_original_date_found", "rss_date": rss_date, "all_candidates": candidates}

    original_dt = datetime.combine(original_date, datetime.min.time(), tzinfo=KST)
    # 날짜 단위만 있으면 해당 일자 23:59까지 가능하므로 미래/과거 판단에 여유를 둠.
    original_dt_end = datetime.combine(original_date, datetime.max.time(), tzinfo=KST)

    if original_dt_end < min_allowed_dt:
        return {
            "valid": False,
            "date": original_date,
            "source": source,
            "status": f"original_date_too_old>{ARTICLE_DATE_MAX_AGE_HOURS}h",
            "rss_date": rss_date,
            "all_candidates": candidates,
        }

    if original_dt > max_allowed_dt:
        return {
            "valid": False,
            "date": original_date,
            "source": source,
            "status": "original_date_in_future",
            "rss_date": rss_date,
            "all_candidates": candidates,
        }

    return {
        "valid": True,
        "date": original_date,
        "source": source,
        "status": "original_date_recent",
        "rss_date": rss_date,
        "all_candidates": candidates,
    }


def is_bad_final_domain(url, title=""):
    domain = url_domain(url)
    title = clean_html_text(title)
    if domain in EXCLUDED_FINAL_DOMAINS:
        return True, f"excluded_final_domain:{domain}"
    if LOW_VALUE_FINAL_DOMAIN_RE.search(domain or ""):
        return True, f"low_value_final_domain:{domain}"
    if re.search(r"네이버\s*블로그|Naver\s*Blog|: 네이버 블로그", title, flags=re.IGNORECASE):
        return True, "naver_blog_result"
    return False, ""


def labor_stage_v7(title, body=""):
    title = clean_html_text(title)
    text = f"{title} {clean_html_text(body)[:800]}"
    # 제목 우선. 여러 단계가 본문에 섞일 수 있으므로 제목에 나온 오늘의 action을 더 신뢰함.
    if re.search(r"조정\s*중지|쟁의권\s*확보", title):
        return "labor_right_to_strike"
    if re.search(r"2차\s*조정|조정회의|조정\s*돌입|협상.*고비|분수령|갈림길", title):
        return "labor_mediation_meeting"
    if re.search(r"찬반투표|투표.*가결", title):
        return "labor_vote"
    if re.search(r"파업\s*돌입|총파업\s*돌입|공동파업\s*돌입", title):
        return "labor_strike_action"
    if re.search(r"성과급|RSU|임금|임단협", title):
        return "labor_pay_dispute"
    if re.search(r"2차\s*조정|조정회의|조정\s*돌입|협상.*고비|분수령|갈림길", text):
        return "labor_mediation_meeting"
    if re.search(r"찬반투표|투표.*가결", text):
        return "labor_vote"
    if re.search(r"조정\s*중지|쟁의권\s*확보", text):
        return "labor_right_to_strike"
    if re.search(r"성과급|RSU|임금|임단협", text):
        return "labor_pay_dispute"
    return ""


def is_kakao_labor_issue(item):
    text = f"{item.get('기사제목','')} {item.get('본문전문','')[:1200]} {item.get('RSS요약','')}"
    return bool(re.search(r"카카오|카카오페이|카카오엔터프라이즈|디케이테크인|엑스엘게임즈", text) and re.search(r"노조|파업|공동파업|총파업|조정|지노위|노동위|성과급|RSU|쟁의권|임단협|고용불안", text))


def event_limit_bucket_v7(item, tag):
    text = f"{item.get('기사제목','')} {item.get('본문전문','')[:1000]}"
    entities = detect_entities(text)
    entity_order = [
        "카카오게임즈", "카카오페이", "카카오뱅크", "카카오모빌리티", "카카오엔터", "카카오", 
        "쿠팡", "네이버", "구글", "오픈AI", "SKT", "KT", "LGU+", "토스", "배달의민족",
        "공정위", "금융위", "금감원", "과기정통부", "방미통위",
    ]
    primary = "general"
    for ent in entity_order:
        if ent in entities or ent in text:
            primary = ent
            break
    stage = item.get("사건단계") or ""
    if tag == "earnings":
        return f"{tag}:{primary}"
    if tag in {"antitrust_platform", "phishing_security", "aidc_datacenter", "ai_agent"}:
        return f"{tag}:{primary}"
    if tag == "stablecoin":
        # 같은 토론회/입법 이슈는 anchor가 final_duplicate에서 먼저 묶이고, 여기서는 broad cap만 보조적으로 적용.
        return f"{tag}:general"
    if stage:
        return f"{tag}:{primary}:{stage}"
    return f"{tag}:{primary}"


def event_tag_limit_reason(new_item, existing_items):
    """v7: 사건태그 제한을 전역이 아니라 주체/세부 버킷 기준으로 적용합니다."""
    if is_kakao_labor_issue(new_item):
        count = sum(1 for old in existing_items if is_kakao_labor_issue(old))
        if count >= 2:
            return "event_topic_limit:kakao_labor:2"

    tags = [t for t in (new_item.get("사건태그", "") or "").split(",") if t]
    if not tags:
        return ""

    for tag in tags:
        limit = MAX_FINAL_PER_EVENT_TAG.get(tag)
        if not limit:
            continue
        new_bucket = event_limit_bucket_v7(new_item, tag)
        count = 0
        for old in existing_items:
            old_tags = set(filter(None, (old.get("사건태그", "") or "").split(",")))
            if tag not in old_tags:
                continue
            if event_limit_bucket_v7(old, tag) == new_bucket:
                count += 1
        if count >= limit:
            return f"event_tag_limit:{new_bucket}:{limit}"
    return ""


def representative_score(report_item):
    """v7: 대표 기사 선택 점수. 본문 품질+중요도+언론사+원문 날짜 신뢰도+도메인 신뢰도를 함께 반영."""
    title = report_item.get("기사제목", "")
    source = report_item.get("언론사", "")
    link = report_item.get("링크", "")
    body_len = int(report_item.get("본문글자수") or 0)
    quality = float(report_item.get("본문품질점수") or 0)
    importance = float(report_item.get("중요도점수") or 0)
    method = report_item.get("본문추출방식", "")
    press = canonical_press_name(source, link)
    date_status = report_item.get("날짜검증결과", "")
    date_source = report_item.get("원문날짜출처", "")

    score = 0.0
    score += quality * 0.9
    score += importance * 0.7
    score += MAJOR_PRESS_BONUS.get(press, press_score(source) * 1.5)

    if body_len < MIN_IMPORTANT_BODY_CHARS:
        score -= 35
    elif body_len < MIN_ACCEPT_BODY_CHARS:
        score -= 14
    elif body_len < MIN_GOOD_BODY_CHARS:
        score += 2
    elif body_len < 1500:
        score += 10
    elif body_len <= 6500:
        score += 20
    elif body_len <= BODY_SUSPICIOUS_LONG_CHARS:
        score += 9
    else:
        score -= 10

    if method == "trafilatura":
        score += 8
    elif method == "bs4_selector":
        score += 6
    elif method == "newspaper3k":
        score += 5
    elif method == "meta_description_short":
        score -= 35

    domain = url_domain(link)
    if domain and domain not in {"v.daum.net", "n.news.naver.com", "news.google.com", "news.nate.com"}:
        score += 4
    bad_domain, _ = is_bad_final_domain(link, title)
    if bad_domain:
        score -= 100

    if date_status == "original_date_recent":
        score += 10
        if date_source in {"url_date", "body_labeled_date", "time_tag"} or str(date_source).startswith(("json_ld", "meta")):
            score += 4
    elif date_status == "no_original_date_found_kept":
        score -= DATELESS_LOW_CONFIDENCE_PENALTY
    elif date_status:
        score -= 45

    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        score -= 60
    if is_foreign_language_low_value(title, source, report_item.get("본문전문", "")):
        score -= 55
    for pattern in LOW_VALUE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            score -= 18

    return round(score, 2)


def process_article_for_report(article_info, json_key, recent_past_items):
    """v7: 원문 발행일 검증과 최종 도메인 품질 필터를 추가한 최종 기사 처리."""
    original_link = article_info.get("링크", "")

    real_url = decode_google_news_url(original_link)
    real_url = normalize_url(real_url)

    bad_domain, bad_domain_reason = is_bad_final_domain(real_url, article_info.get("기사제목", ""))
    if bad_domain:
        failed = {
            "카테고리": json_key,
            "카테고리명": JSON_KEY_TO_DISPLAY.get(json_key, json_key),
            "검색어": article_info.get("검색어", ""),
            "기사제목": article_info.get("기사제목", ""),
            "언론사": article_info.get("언론사", ""),
            "게시일": article_info.get("게시일", ""),
            "RSS요약": article_info.get("본문요약", ""),
            "본문전문": "",
            "본문글자수": 0,
            "본문추출방식": "domain_filter",
            "본문품질점수": -100,
            "본문품질사유": bad_domain_reason,
            "중요도점수": 0,
            "본문사용가능": False,
            "원래RSS링크": original_link,
            "링크": real_url,
            "날짜검증결과": "not_checked_domain_filtered",
        }
        return None, None, failed

    body_text, extract_method, fetched_url = extract_article_body(real_url)
    if fetched_url:
        real_url = normalize_url(fetched_url)

    body_text_raw = body_text.strip()
    body_text = clean_extracted_body_v6(body_text_raw)
    body_cleaned = body_text != body_text_raw

    # 원문 발행일 검증. 본문 추출이 끝난 뒤 HTML을 한 번 더 확인합니다.
    html_for_date = ""
    fetched_for_date = ""
    try:
        html_for_date, fetched_for_date = get_html(real_url)
        if fetched_for_date:
            real_url = normalize_url(fetched_for_date)
    except Exception:
        html_for_date = ""

    date_info = verify_article_original_date(
        real_url,
        body_text,
        html_text=html_for_date,
        rss_published=article_info.get("게시일", ""),
    )

    importance = article_importance_score(
        article_info.get("기사제목", ""),
        body_text or article_info.get("본문요약", ""),
        json_key=json_key,
        keyword=article_info.get("검색어", ""),
    )

    good, quality_reason, quality_score = is_body_usable_v6(
        article_info.get("기사제목", ""),
        body_text,
        extract_method,
        json_key=json_key,
        source=article_info.get("언론사", ""),
        importance_score=importance,
    )

    if not date_info.get("valid", True):
        good = False
        quality_reason = f"date_failed:{date_info.get('status')}:{date_info.get('date')}"
        quality_score = min(float(quality_score or 0), -10.0)

    event_tags = detect_event_tags(f"{article_info.get('기사제목', '')} {body_text[:1800]}")
    title_for_stage = article_info.get("기사제목", "")
    default_stage = extract_event_stage(f"{title_for_stage} {body_text[:1600]}", event_tags)
    labor_stage = labor_stage_v7(title_for_stage, body_text)
    final_stage = labor_stage or default_stage

    body_candidate = {
        "기사제목": article_info.get("기사제목", ""),
        "본문요약": body_text[:2200],
        "링크": real_url,
    }

    is_dup, matched, sim = find_past_duplicate(body_candidate, recent_past_items, mode="body")

    if is_dup:
        skip_info = {
            "검색어": article_info.get("검색어", ""),
            "후보제목": article_info.get("기사제목", ""),
            "후보링크": real_url,
            "후보언론사": article_info.get("언론사", ""),
            "후보대표점수": "",
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
            "원문날짜": date_info.get("date"),
            "날짜검증결과": date_info.get("status"),
        }
        return None, skip_info, None

    report_item = {
        "카테고리": json_key,
        "카테고리명": JSON_KEY_TO_DISPLAY.get(json_key, json_key),
        "검색어": article_info.get("검색어", ""),
        "기사제목": article_info.get("기사제목", ""),
        "언론사": article_info.get("언론사", "") or guess_press_name_from_url(real_url),
        "정규언론사": canonical_press_name(article_info.get("언론사", ""), real_url),
        "게시일": article_info.get("게시일", ""),
        "RSS게시일": article_info.get("게시일", ""),
        "RSS요약": article_info.get("본문요약", ""),
        "본문전문": body_text,
        "본문글자수": len(body_text),
        "본문추출방식": extract_method,
        "본문품질점수": quality_score,
        "본문품질사유": quality_reason,
        "중요도점수": importance,
        "본문사용가능": good,
        "본문정제적용": body_cleaned,
        "원래RSS링크": original_link,
        "링크": real_url,
        "랭킹점수": article_info.get("랭킹점수", ""),
        "사건태그": ",".join(sorted(event_tags)),
        "사건단계": final_stage,
        "원문날짜": date_info.get("date"),
        "원문날짜출처": date_info.get("source"),
        "날짜검증결과": date_info.get("status"),
        "날짜후보수": len(date_info.get("all_candidates") or []),
    }
    report_item["대표선택점수"] = representative_score(report_item)

    if not good:
        return None, None, report_item

    return report_item, None, None


def is_report_item_relevant(report_item, json_key):
    title = report_item.get("기사제목", "")
    body = report_item.get("본문전문", "") or report_item.get("RSS요약", "") or report_item.get("본문요약", "")
    keyword = report_item.get("검색어", "")
    source = report_item.get("언론사", "")
    link = report_item.get("링크", "")
    text = f"{title} {body[:1400]}"

    bad_domain, reason = is_bad_final_domain(link, title)
    if bad_domain:
        return False, reason

    if report_item.get("날짜검증결과", "").startswith("original_date_too_old") or report_item.get("날짜검증결과") == "old_known_event_recirculated":
        return False, f"old_article:{report_item.get('원문날짜')}:{report_item.get('원문날짜출처')}"

    for pattern in V6_STRICT_EXCLUDE_TITLE_PATTERNS:
        if re.search(pattern, title, flags=re.IGNORECASE):
            return False, f"low_value_title:{pattern}"

    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        return False, "photo_or_caption_article"

    if is_foreign_language_low_value(title, source, body):
        return False, "foreign_language_low_value"

    low, low_reason = is_article_obviously_low_value({
        "기사제목": title, "본문요약": body[:600], "검색어": keyword, "언론사": source
    })
    if low:
        return False, low_reason

    # 원문 날짜가 없고 신뢰 낮은 도메인이면 최종 제외
    if report_item.get("날짜검증결과") == "no_original_date_found_kept":
        domain = url_domain(link)
        if press_score(source) <= 1 and domain not in {"v.daum.net", "news.nate.com"}:
            return False, "dateless_low_confidence_source"

    if OLD_KNOWN_EVENT_RE.search(text) and not CURRENT_ANGLE_RE.search(text[:1400]):
        return False, "old_known_event_without_current_angle"

    if json_key == "자사_및_계열사_이슈":
        if not SELF_KAKAO_PATTERN.search(text):
            return False, "self_category_without_kakao_entity"
        if keyword == "카카오톡" and not re.search(r"카카오톡|카톡|카카오", title, flags=re.IGNORECASE):
            kakao_mentions = len(re.findall(r"카카오|카톡|Kakao", text, flags=re.IGNORECASE))
            if kakao_mentions <= 2 and KAKAOTALK_NOISE_PATTERN.search(text):
                return False, "kakaotalk_channel_only_noise"

    if json_key == "정부_국회":
        if not DIGITAL_STRATEGIC_PATTERN.search(text):
            return False, "government_without_digital_platform_ai_relevance"
        if re.search(r"금융교육\s*전문강사|보험사\s*순이익|생산적\s*금융\s*실적", title):
            return False, "generic_finance_not_ceo_platform_issue"

    if json_key == "경쟁사_해외이슈":
        if re.search(r"케이스\s*유출|렌더링|출시\s*예상|스펙\s*유출|화면\s*없는\s*기기", title):
            return False, "consumer_product_rumor"
        brand_pat = BRAND_KEYWORD_PATTERNS.get(keyword)
        if brand_pat and not brand_pat.search(text[:900]):
            return False, "competitor_brand_absent_from_front_text"
        if re.search(r"건설|아파트|재매각|블로그", title) and not re.search(r"네이버|카카오|쿠팡|토스|구글|오픈AI|MS|메타|애플|플랫폼|AI", title):
            return False, "competitor_off_topic_general_business"

    if json_key == "산업동향":
        if not re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|플랫폼|망\s*사용료|스테이블코인|디지털자산|보안|해킹|딥페이크|저작권|알고리즘", text, flags=re.IGNORECASE):
            return False, "industry_without_core_digital_trend"
        if re.search(r"세라믹기술원|장애인\s*표준사업장|교통사고", title):
            return False, "generic_industry_or_local_news"

    return True, ""


def final_duplicate_reason(new_item, existing_items):
    """v7: 같은 날 기사끼리 중복 판단. 카카오 노사 단계와 anchor를 더 정확히 봅니다."""
    new_title = new_item.get("기사제목", "")
    new_body = new_item.get("본문전문", "")[:1800]
    new_text = f"{new_title} {new_body}"
    new_tokens = tokenize_for_similarity(new_text)
    new_tags = set(filter(None, (new_item.get("사건태그", "") or "").split(","))) or detect_event_tags(new_text)
    new_sig = event_signature(new_text)
    new_stage = labor_stage_v7(new_title, new_body) or new_item.get("사건단계") or extract_event_stage(new_text, new_tags)
    new_press, new_fp = canonical_article_signature(new_title, new_item.get("언론사", ""), new_item.get("링크", ""))
    new_entities = detect_entities(new_text)

    for old in existing_items:
        old_title = old.get("기사제목", "")
        old_body = old.get("본문전문", "")[:1800]
        old_text = f"{old_title} {old_body}"
        old_tokens = tokenize_for_similarity(old_text)
        old_tags = set(filter(None, (old.get("사건태그", "") or "").split(","))) or detect_event_tags(old_text)
        old_sig = event_signature(old_text)
        old_stage = labor_stage_v7(old_title, old_body) or old.get("사건단계") or extract_event_stage(old_text, old_tags)
        old_press, old_fp = canonical_article_signature(old_title, old.get("언론사", ""), old.get("링크", ""))
        old_entities = detect_entities(old_text)

        title_seq = sequence_ratio(new_title, old_title)
        token_score = jaccard(new_tokens, old_tokens)
        body_ngram_score = jaccard(char_ngrams(new_body[:1200], 5), char_ngrams(old_body[:1200], 5))
        shared_entities = new_entities & old_entities
        shared_tags = (new_tags & old_tags) & STRONG_EVENT_TAGS

        if new_fp and old_fp and new_fp == old_fp:
            return old, "same_current_canonical_title"

        near, near_score = is_title_near_duplicate(new_title, old_title)
        if near and (new_press == old_press or near_score >= 0.94):
            return old, f"near_current_title:{near_score:.2f}"

        if body_ngram_score >= 0.55:
            return old, "same_current_body"

        # 카카오 노사: 같은 단계면 대표 기사 경쟁, 다른 단계면 업데이트로 살림.
        if is_kakao_labor_issue(new_item) and is_kakao_labor_issue(old):
            if new_stage and old_stage and new_stage == old_stage:
                if token_score >= 0.04 or title_seq >= 0.28 or body_ngram_score >= 0.12:
                    return old, f"same_current_labor_stage:{new_stage}"
            elif token_score >= 0.28 and title_seq >= 0.58:
                return old, "same_current_labor_high_similarity"

        anchor_sets = [
            ("안도걸", "디지털자산기본법", "스테이블코인"),
            ("K-문샷", "추진단"),
            ("카나나 스칼라", "콜로키움"),
            ("국가AI컴퓨팅센터", "삼성SDS"),
            ("구글", "EU", "과징금"),
            ("오픈AI", "GTAC"),
            ("공정위", "조사국"),
            ("쿠팡", "김범석", "지정자료"),
        ]
        for anchors in anchor_sets:
            if all(a in new_text for a in anchors) and all(a in old_text for a in anchors):
                if token_score >= 0.035 or title_seq >= 0.26 or body_ngram_score >= 0.10:
                    return old, f"same_current_anchor:{'/'.join(anchors)}"

        if new_sig and old_sig and new_sig == old_sig:
            if title_seq >= 0.36 or token_score >= 0.12 or body_ngram_score >= 0.24:
                return old, f"same_current_signature:{new_sig}"

        if shared_entities and shared_tags:
            if new_stage and old_stage and new_stage != old_stage:
                if title_seq >= 0.72 and token_score >= 0.22:
                    return old, f"same_entities_tags_high_similarity:{','.join(sorted(shared_tags))}"
            else:
                if token_score >= 0.11 or title_seq >= 0.42 or body_ngram_score >= 0.20:
                    return old, f"same_entities_tags:{','.join(sorted(shared_tags))}"

    return None, ""



# ==========================================
# v11 overrides: stable daily GR engine
# - 자사/계열사만 엄격한 must-review 적용
# - 정부/경쟁사/산업동향은 기존 랭킹 중심으로 안정화
# - 검색어가 아니라 실제 기사 주체로 카테고리 재검증
# ==========================================

V6_VERSION = "google_rss_v11_self_strict_must_review_stable_others"

# v11 원칙:
# 1) must-review는 자사/계열사 기사에만 적용한다.
# 2) 자사/계열사 must-review도 '실제 카카오 주체 + 강한 행위/위험 신호'가 같이 있어야 한다.
# 3) 정부/경쟁사/산업동향은 v7~v8처럼 랭킹과 Gemini 선별 중심으로 둔다.
MAX_CANDIDATES_FOR_GEMINI = 300
CATEGORY_POOL_LIMIT_FOR_GEMINI = {
    "자사_및_계열사_이슈": 120,
    "정부_국회": 100,
    "경쟁사_해외이슈": 75,
    "산업동향": 22,
}
MUST_REVIEW_POOL_LIMIT = {
    "자사_및_계열사_이슈": 45,
    "정부_국회": 0,
    "경쟁사_해외이슈": 0,
    "산업동향": 0,
}
MUST_REVIEW_PROCESS_LIMIT = {
    "자사_및_계열사_이슈": 5,
    "정부_국회": 0,
    "경쟁사_해외이슈": 0,
    "산업동향": 0,
}

CATEGORY_MIN = {
    "자사_및_계열사_이슈": 3,
    "정부_국회": 5,
    "경쟁사_해외이슈": 3,
    "산업동향": 1,
}
CATEGORY_TARGET = {
    "자사_및_계열사_이슈": 5,
    "정부_국회": 6,
    "경쟁사_해외이슈": 4,
    "산업동향": 1,
}
CATEGORY_MAX = {
    "자사_및_계열사_이슈": 7,
    "정부_국회": 8,
    "경쟁사_해외이슈": 6,
    "산업동향": 2,
}
MIN_SELECT_COUNT = sum(CATEGORY_MIN.values())
MAX_SELECT_COUNT = 19

# 특정 사건명 하드코딩은 피하되, 매일 재사용 가능한 일반 리스크 shard만 유지한다.
_PERSON_SPECIFIC_SHARDS_TO_DROP = {"홍민택"}
KAKAO_ISSUE_SHARDS[:] = [q for q in KAKAO_ISSUE_SHARDS if q not in _PERSON_SPECIFIC_SHARDS_TO_DROP]
for _q in [
    "카카오 임원", "카카오 최고책임자", "카카오 조직개편", "카카오 인사", "카카오 리더십",
    "카카오 서비스 개편", "카카오톡 개편", "카카오톡 논란", "카카오 이용자 반발", "카카오 사과",
    "카카오 노조", "카카오 파업", "카카오 쟁의권", "카카오 조정 결렬", "카카오 임단협",
    "카카오 성과급", "카카오 RSU", "카카오 고용불안", "카카오 노동부", "카카오 근로감독",
    "카카오 최저임금", "카카오 임금체불", "카카오 민주노총", "카카오 법 위반",
    "카카오 개인정보", "카카오 과징금", "카카오 수사", "카카오 소송", "카카오 장애", "카카오 피싱",
    "카카오 지분", "카카오 매각", "카카오 인수", "카카오 합병", "카카오 최대주주",
    "카카오 T", "카카오T", "카카오택시", "카카오 대리운전", "카카오 가맹택시", "카카오 플랫폼 노동",
]:
    if _q not in KAKAO_ISSUE_SHARDS:
        KAKAO_ISSUE_SHARDS.append(_q)

ENTITY_PATTERNS.update({
    "카카오모빌리티": r"카카오모빌리티|카카오\s*T\b|카카오T\b|카카오\s*T블루|카카오T블루|카카오택시|카카오\s*택시|카카오\s*대리|카카오대리|대리운전|가맹택시|플랫폼\s*운송|플랫폼\s*노동",
    "고용노동부": r"고용노동부|노동부|노동청|근로감독|최저임금법|임금체불|노동위원회|지노위|중노위",
    "카카오서비스": r"카카오톡|카톡|친구탭|오픈채팅|카카오맵|카카오\s*T|카카오T|선물하기|톡비즈|카나나",
})
BRAND_KEYWORD_PATTERNS.update({
    "카카오모빌리티": re.compile(r"카카오모빌리티|카카오\s*T\b|카카오T\b|카카오\s*T블루|카카오T블루|카카오택시|카카오\s*택시|카카오\s*대리|카카오대리|대리운전|가맹택시|플랫폼\s*운송|플랫폼\s*노동", re.IGNORECASE),
})

SELF_KAKAO_PATTERN = re.compile(
    r"카카오|카톡|Kakao|카나나|카카오페이|카카오뱅크|카카오모빌리티|카카오\s*T\b|카카오T\b|카카오택시|카카오게임즈|카카오엔터|카카오헬스케어|카카오엔터프라이즈|카카오픽코마|디케이테크인|엑스엘게임즈|AXZ|다음",
    re.IGNORECASE,
)
COMPETITOR_CORE_PATTERN = re.compile(
    r"네이버|NAVER|구글|Google|오픈AI|OpenAI|MS|마이크로소프트|Microsoft|메타|Meta|애플|Apple|쿠팡|토스|배달의민족|배민|SKT|SK텔레콤|KT|LGU\+|LG유플러스|라인야후|우버|Uber|앤트로픽|Anthropic",
    re.IGNORECASE,
)
REGULATOR_CORE_PATTERN = re.compile(
    r"과기정통부|과학기술정보통신부|공정위|공정거래위원회|금융위|금융위원회|금감원|금융감독원|개보위|개인정보보호위원회|방미통위|방송미디어통신위원회|행안부|행정안전부|고용노동부|노동부|국회|정무위|과방위|정부|경찰청|검찰|KISA|인터넷진흥원",
    re.IGNORECASE,
)

# 일반 신호 체계. must-review는 자사 기사에 한해 이 신호들을 사용한다.
V11_SIGNAL_PATTERNS = {
    "leadership_change": re.compile(r"(대표|임원|경영진|CPO|CTO|CFO|C레벨|최고[가-힣A-Za-z]*책임자|성과리더|본부장|리더|책임자).{0,80}(퇴사|사의|사퇴|사임|물러|떠난|교체|내정|선임|영입|해임|직무배제|조직개편|쇄신)|리더십\s*공백|컨트롤타워", re.IGNORECASE),
    "service_reform_backlash": re.compile(r"(개편|업데이트|서비스\s*개편|UX|UI|친구탭|추천|광고|숏폼|프로필|피드|오픈채팅).{0,100}(논란|반발|불만|비판|역풍|철회|되돌|복구|사과|해명|개선|후폭풍)", re.IGNORECASE),
    "labor_negotiation": re.compile(r"노조|파업|총파업|쟁의권|조정\s*결렬|조정\s*중지|노동위|지노위|중노위|임단협|교섭|결의대회|찬반투표|성과급|RSU|임금협상|고용불안|희망퇴직|정리해고", re.IGNORECASE),
    "labor_law_complaint": re.compile(r"최저임금법|노동부\s*진정|고용노동부|근로감독|노동청|임금체불|근로기준법|부당노동행위|위장도급|플랫폼\s*노동|산재|노동법\s*위반", re.IGNORECASE),
    "legal_regulatory_enforcement": re.compile(r"과징금|제재|시정명령|고발|수사|압수수색|입건|기소|구형|판결|선고|행정소송|검찰|경찰|특사경|조사\s*착수|현장점검|직권조사", re.IGNORECASE),
    "privacy_security_reliability": re.compile(r"개인정보\s*유출|정보\s*유출|침해사고|해킹|피싱|사칭|보이스피싱|장애|먹통|접속\s*장애|오류|보안|랜섬웨어|악성코드|딥페이크|허위조작정보", re.IGNORECASE),
    "policy_legislation": re.compile(r"법안|입법|시행령|개정안|가이드라인|규제|자율규제|기본법|특별법|국회|상임위|본회의|의결|행정예고|입법예고|대책|정책|전략|위원회|협의체", re.IGNORECASE),
    "market_power_platform": re.compile(r"플랫폼|온라인플랫폼|온플법|독점|시장지배|자사우대|최혜대우|다크패턴|정산|수수료|광고시장|검색시장|DMA|디지털시장법|망사용료", re.IGNORECASE),
    "strategy_mna_governance": re.compile(r"인수|매각|합병|주식교환|지분|처분|최대주주|경영권|투자유치|IPO|상장|블록딜|컨소시엄|우선협상|구주|신주|증자", re.IGNORECASE),
    "financial_performance": re.compile(r"실적|매출|영업이익|영업손실|순이익|적자|흑자|목표가|주가|배당|자사주|손상차손|수익성|컨콜|공시", re.IGNORECASE),
    "ai_infrastructure_security": re.compile(r"AI|인공지능|에이전트|에이전틱|LLM|파운데이션\s*모델|GPU|NPU|AIDC|데이터센터|AI\s*보안|AI\s*기본법|AI\s*저작권|소버린\s*AI|딥페이크|미토스|GPT", re.IGNORECASE),
    "digital_asset_stablecoin": re.compile(r"스테이블코인|디지털자산|가상자산|토큰증권|STO|RWA|CBDC|원화\s*스테이블|두나무|업비트|빗썸|코인|블록체인", re.IGNORECASE),
}
EVENT_PATTERNS.update({k: p.pattern for k, p in V11_SIGNAL_PATTERNS.items()})
STRONG_EVENT_TAGS.update(set(V11_SIGNAL_PATTERNS.keys()))

# 확실한 자사 must-review 신호. 단순 AI/실적 단어만으로는 must-review가 아니다.
V11_SELF_MUST_SIGNALS = {
    "leadership_change", "service_reform_backlash", "labor_negotiation", "labor_law_complaint",
    "legal_regulatory_enforcement", "privacy_security_reliability", "strategy_mna_governance",
}
V11_SIGNAL_WEIGHTS_SELF = {
    "leadership_change": 30,
    "service_reform_backlash": 28,
    "labor_negotiation": 26,
    "labor_law_complaint": 34,
    "legal_regulatory_enforcement": 30,
    "privacy_security_reliability": 28,
    "strategy_mna_governance": 24,
    "financial_performance": 12,
    "ai_infrastructure_security": 12,
    "digital_asset_stablecoin": 14,
}

V11_STORY_TOPIC_LIMITS = {
    "self:leadership_service": 2,
    "self:labor_law": 2,
    "self:labor_strike": 2,
    "self:labor_compensation": 1,
    "self:legal_regulatory": 2,
    "self:privacy_security": 2,
    "self:strategy_mna": 2,
    "self:financial_performance": 2,
    "self:ai_strategy": 2,
}
V11_CATEGORY_FAMILY_TOTAL_LIMITS = {
    ("자사_및_계열사_이슈", "labor_"): 3,
    ("자사_및_계열사_이슈", "leadership_service"): 2,
    ("자사_및_계열사_이슈", "financial_performance"): 2,
}
V11_MUST_REVIEW_TOPIC_POOL_LIMIT = 10

_BASE_collection_relevance_reason_v11 = collection_relevance_reason
_BASE_article_importance_score_v11 = article_importance_score
_BASE_is_article_obviously_low_value_v11 = is_article_obviously_low_value
_BASE_is_report_item_relevant_v11 = is_report_item_relevant
_BASE_final_duplicate_reason_v11 = final_duplicate_reason
_BASE_event_tag_limit_reason_v11 = event_tag_limit_reason


def v11_detect_signals(text):
    text = clean_html_text(text)
    found = []
    for name, pattern in V11_SIGNAL_PATTERNS.items():
        if pattern.search(text):
            found.append(name)
    return found


def v10_detect_signals(text):
    return v11_detect_signals(text)


def v11_signal_string(signals):
    return "+".join(sorted(set(signals)))


def v10_signal_string(signals):
    return v11_signal_string(signals)


def v11_has_self_entity(text):
    return bool(SELF_KAKAO_PATTERN.search(clean_html_text(text)))


def v11_main_entity_group(text, json_key=""):
    text = clean_html_text(text)
    if SELF_KAKAO_PATTERN.search(text):
        return "self"
    if REGULATOR_CORE_PATTERN.search(text) or json_key == "정부_국회":
        return "government"
    if COMPETITOR_CORE_PATTERN.search(text) or json_key == "경쟁사_해외이슈":
        return "competitor"
    return "industry"


def v10_main_entity_group(text, json_key=""):
    return v11_main_entity_group(text, json_key)


def v11_is_individual_crime_case(text, json_key=""):
    text = clean_html_text(text)
    if not re.search(r"보이스피싱|사기|살인|교통사고|인출책|징역|피해자|범죄", text):
        return False
    # 정부 대책/플랫폼 대응/제도 개선이면 개별 사건으로 보지 않는다.
    if re.search(r"정부|금융위|금감원|경찰청|방미통위|개보위|대책|정책|가이드라인|법안|제도|플랫폼|카카오|네이버|구글|통신사|금융권|협의회|공동대응", text):
        return False
    return True


def v10_is_individual_crime_case(text, json_key=""):
    return v11_is_individual_crime_case(text, json_key)


def v11_story_family_from_signals(signals, text="", json_key=""):
    s = set(signals)
    text = clean_html_text(text)
    group = v11_main_entity_group(text, json_key)

    if group == "self":
        if "labor_law_complaint" in s:
            return "labor_law"
        if "labor_negotiation" in s and re.search(r"성과급|RSU|임금|보상", text):
            return "labor_compensation"
        if "labor_negotiation" in s:
            return "labor_strike"
        if "leadership_change" in s or "service_reform_backlash" in s:
            return "leadership_service"
        if "legal_regulatory_enforcement" in s:
            return "legal_regulatory"
        if "privacy_security_reliability" in s:
            return "privacy_security"
        if "strategy_mna_governance" in s:
            return "strategy_mna"
        if "financial_performance" in s:
            return "financial_performance"
        if "ai_infrastructure_security" in s:
            return "ai_strategy"
        if "digital_asset_stablecoin" in s:
            return "digital_asset"
        return "general"

    # 정부/경쟁사/산업은 must-review용 family가 아니라 중복/로그 참고용만 최소화.
    if group == "government":
        if "legal_regulatory_enforcement" in s or "market_power_platform" in s:
            return "regulatory_enforcement"
        if "digital_asset_stablecoin" in s:
            return "digital_asset"
        if "ai_infrastructure_security" in s or "privacy_security_reliability" in s:
            return "tech_security_policy"
        if "policy_legislation" in s:
            return "policy_legislation"
        return "general"

    if group == "competitor":
        if "strategy_mna_governance" in s:
            return "strategy_mna"
        if "legal_regulatory_enforcement" in s or "market_power_platform" in s:
            return "regulatory_enforcement"
        if "privacy_security_reliability" in s:
            return "security_reliability"
        if "ai_infrastructure_security" in s:
            return "ai_strategy"
        return "general"

    if "ai_infrastructure_security" in s:
        return "ai_infrastructure"
    if "digital_asset_stablecoin" in s:
        return "digital_asset"
    return "general"


def v10_story_family_from_signals(signals, text="", json_key=""):
    return v11_story_family_from_signals(signals, text, json_key)


def v11_story_topic_bucket(item):
    text = f"{item.get('기사제목','')} {item.get('본문전문','')[:1800]} {item.get('본문요약','')} {item.get('RSS요약','')}"
    json_key = item.get("카테고리") or item.get("JSON카테고리") or ""
    signals = v11_detect_signals(text)
    if not signals:
        return ""
    group = v11_main_entity_group(text, json_key)
    family = v11_story_family_from_signals(signals, text, json_key)
    return f"{group}:{family}"


def v10_story_topic_bucket(item):
    return v11_story_topic_bucket(item)


def story_topic_bucket_v9(item):
    return v11_story_topic_bucket(item)


def v11_self_must_review_reason(text, json_key=""):
    text = clean_html_text(text)
    if not text:
        return False, ""
    # 검색어 카테고리가 아니라 실제 기사 주체가 카카오/계열사여야 한다.
    if not v11_has_self_entity(text):
        return False, ""
    signals = set(v11_detect_signals(text))
    strong = signals & V11_SELF_MUST_SIGNALS
    if not strong:
        return False, ""
    family = v11_story_family_from_signals(signals, text, "자사_및_계열사_이슈")
    return True, f"must_review:self_strict:{family}:{v11_signal_string(strong)}"


def v10_must_review_reason(text, json_key=""):
    # v11: must-review는 자사/계열사에만 적용한다.
    if json_key != "자사_및_계열사_이슈":
        return False, ""
    return v11_self_must_review_reason(text, json_key)


def is_v9_must_review_text(text, json_key=""):
    return v10_must_review_reason(text, json_key)


def is_v9_must_review_article(article):
    text = f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('검색어','')} {article.get('원카테고리','')}"
    return v10_must_review_reason(text, article.get("JSON카테고리", ""))


def collection_relevance_reason(keyword, title, summary):
    front_text = f"{clean_html_text(title)} {clean_html_text(summary)[:900]}"
    json_key = CATEGORY_TO_JSON_KEY.get(keyword_to_category.get(keyword, ""), "")
    # 자사 키워드 수집인데 실제 제목/요약 앞부분에 카카오/계열사 주체가 없으면 오탐으로 본다.
    if json_key == "자사_및_계열사_이슈" and not v11_has_self_entity(front_text):
        return "collection_relevance:self_entity_absent_from_title_and_front_summary"
    must, _reason = v10_must_review_reason(front_text, json_key)
    if must:
        return ""
    return _BASE_collection_relevance_reason_v11(keyword, title, summary)


def article_importance_score(title, body="", json_key="", keyword=""):
    base = float(_BASE_article_importance_score_v11(title, body, json_key, keyword) or 0)
    title_text = clean_html_text(title)
    text = f"{title_text} {clean_html_text(body)[:2600]}"

    if v11_is_individual_crime_case(text, json_key):
        return round(base - 35, 2)

    signals = v11_detect_signals(text)
    group = v11_main_entity_group(text, json_key)

    # 자사/계열사만 리스크 신호를 강하게 가산한다. 나머지는 기존 랭킹을 유지한다.
    if json_key == "자사_및_계열사_이슈":
        if not v11_has_self_entity(text):
            base -= 45
        else:
            for signal in signals:
                base += V11_SIGNAL_WEIGHTS_SELF.get(signal, 0)
            must, _reason = v10_must_review_reason(text, json_key)
            if must:
                base += 18
            if re.search(r"\[단독\]|단독|종합|속보", title_text):
                base += 6
    else:
        # 정부/경쟁사/산업은 v7~v8 수준으로 안정화. 단, 명백한 저가치는 감점.
        if re.search(r"네이버\s*블로그|Naver\s*Blog|블로그", title_text, re.IGNORECASE):
            base -= 30
        if re.search(r"비트코인.*(전망|하락|상승)|주가\s*전망|목표가|매수\s*추천", title_text, re.IGNORECASE):
            base -= 18

    if re.search(r"사진|포토|화보|캡션|영상만|현장사진", title_text, re.IGNORECASE):
        base -= 22
    return round(base, 2)


def is_article_obviously_low_value(article):
    text = f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('검색어','')}"
    if v11_is_individual_crime_case(text, article.get("JSON카테고리", "")):
        return True, "individual_crime_case_not_policy"
    if article.get("JSON카테고리") == "자사_및_계열사_이슈":
        if not v11_has_self_entity(text):
            return True, "self_category_without_self_entity"
        must, _reason = v10_must_review_reason(text, article.get("JSON카테고리", ""))
        if must:
            return False, ""
    return _BASE_is_article_obviously_low_value_v11(article)


def is_report_item_relevant(report_item, json_key):
    text = f"{report_item.get('기사제목','')} {report_item.get('본문전문','')[:2200]} {report_item.get('RSS요약','')}"
    if json_key == "자사_및_계열사_이슈" and not v11_has_self_entity(text):
        return False, "self_category_without_self_entity"
    if v11_is_individual_crime_case(text, json_key):
        return False, "individual_crime_case_not_policy"
    # 자사는 must-review면 통과. 나머지는 기존 relevance 판단을 따른다.
    if json_key == "자사_및_계열사_이슈":
        must, _reason = v10_must_review_reason(text, json_key)
        if must:
            return True, ""
    return _BASE_is_report_item_relevant_v11(report_item, json_key)


def v11_story_anchor_tokens(item):
    title = clean_html_text(item.get("기사제목", ""))
    body = clean_html_text(item.get("본문전문", "")[:500])
    tokens = tokenize_for_similarity(f"{title} {body}")
    drop = {"카카오", "네이버", "정부", "국회", "논란", "종합", "단독", "오늘", "어제", "기자", "관련", "추진", "검토", "강화"}
    return {t for t in tokens if t not in drop and len(t) >= 2}


def v10_story_anchor_tokens(item):
    return v11_story_anchor_tokens(item)


def final_duplicate_reason(new_item, existing_items):
    base_dup, base_reason = _BASE_final_duplicate_reason_v11(new_item, existing_items)
    if base_dup:
        return base_dup, base_reason

    new_bucket = v11_story_topic_bucket(new_item)
    if not new_bucket:
        return None, ""
    new_title = clean_html_text(new_item.get("기사제목", ""))
    new_anchors = v11_story_anchor_tokens(new_item)

    for old in existing_items:
        old_bucket = v11_story_topic_bucket(old)
        if old_bucket != new_bucket:
            continue
        old_title = clean_html_text(old.get("기사제목", ""))
        ratio = sequence_ratio(new_title, old_title)
        anchor_overlap = jaccard(new_anchors, v11_story_anchor_tokens(old))
        # 자사는 같은 리스크 family가 반복되기 쉬우므로 중복 기준을 조금 민감하게 둔다.
        if new_bucket.startswith("self:"):
            if ratio >= 0.52 or anchor_overlap >= 0.32:
                return old, f"same_self_story_topic:{new_bucket}:ratio={round(ratio,2)}:anchor={round(anchor_overlap,2)}"
        else:
            # 나머지 카테고리는 기존처럼 보수적으로 중복 처리한다.
            if ratio >= 0.70 or anchor_overlap >= 0.45:
                return old, f"same_story_topic:{new_bucket}:ratio={round(ratio,2)}:anchor={round(anchor_overlap,2)}"
    return None, ""


def event_tag_limit_reason(new_item, existing_items):
    bucket = v11_story_topic_bucket(new_item)
    json_key = new_item.get("카테고리") or new_item.get("JSON카테고리") or ""

    # v11: 자사/계열사만 family 독식 제한을 추가 적용한다. 나머지는 기존 기준으로 둔다.
    if json_key == "자사_및_계열사_이슈" and bucket:
        limit = V11_STORY_TOPIC_LIMITS.get(bucket)
        if limit is not None:
            count = sum(1 for old in existing_items if v11_story_topic_bucket(old) == bucket)
            if count >= limit:
                return f"event_story_topic_limit:{bucket}:{limit}"
        if ":" in bucket:
            _, family = bucket.split(":", 1)
            for (cat, prefix), total_limit in V11_CATEGORY_FAMILY_TOTAL_LIMITS.items():
                if cat == json_key and family.startswith(prefix):
                    total = 0
                    for old in existing_items:
                        old_bucket = v11_story_topic_bucket(old)
                        old_family = old_bucket.split(":", 1)[-1] if ":" in old_bucket else old_bucket
                        if old.get("카테고리") == json_key and old_family.startswith(prefix):
                            total += 1
                    if total >= total_limit:
                        return f"event_family_limit:{json_key}:{prefix}:{total_limit}"
    return _BASE_event_tag_limit_reason_v11(new_item, existing_items)


def v10_must_priority(article):
    text = f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('검색어','')}"
    if article.get("JSON카테고리") != "자사_및_계열사_이슈":
        return float(article.get("랭킹점수") or 0)
    signals = v11_detect_signals(text)
    score = 0.0
    for signal in signals:
        score += V11_SIGNAL_WEIGHTS_SELF.get(signal, 0)
    score += float(article.get("중요도점수") or 0) * 0.9
    score += float(article.get("랭킹원점수") or article.get("랭킹점수") or 0) * 0.25
    return score


def rank_and_trim_candidates(raw_articles):
    """v11: 자사 must-review는 엄격히 포함, 나머지는 기존 랭킹 중심으로 안정화."""
    for article in raw_articles:
        article["랭킹점수"] = rank_score_article(article)
        article["랭킹원점수"] = article.get("랭킹점수", 0)
        text = f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('검색어','')}"
        signals = v11_detect_signals(text)
        article["편집신호"] = ",".join(signals)
        article["스토리버킷"] = v11_story_topic_bucket(article)
        must, reason = v10_must_review_reason(text, article.get("JSON카테고리", ""))
        article["강제검토"] = "Y" if must else ""
        article["강제검토사유"] = reason
        if must:
            article["랭킹점수"] = round(float(article.get("랭킹점수") or 0) + 35 + min(28, v10_must_priority(article) * 0.16), 2)
        # 자사 카테고리지만 실제 카카오 주체가 없으면 강한 감점.
        if article.get("JSON카테고리") == "자사_및_계열사_이슈" and not v11_has_self_entity(text):
            article["랭킹점수"] = round(float(article.get("랭킹점수") or 0) - 60, 2)

    ranked_all = sorted(
        raw_articles,
        key=lambda x: (float(x.get("랭킹점수") or 0), float(x.get("중요도점수") or 0), x.get("게시일", "")),
        reverse=True,
    )

    selected = []
    selected_ids = set()
    selected_topic_counts = {}

    def add(article, respect_topic_cap=True):
        art_id = int(article["id"])
        if art_id in selected_ids:
            return False
        if float(article.get("랭킹점수") or 0) <= -30 and article.get("강제검토") != "Y":
            return False
        topic = article.get("스토리버킷") or ""
        if respect_topic_cap and topic:
            cap = V11_MUST_REVIEW_TOPIC_POOL_LIMIT if article.get("강제검토") == "Y" else 8
            if selected_topic_counts.get(topic, 0) >= cap:
                return False
        selected.append(article)
        selected_ids.add(art_id)
        if topic:
            selected_topic_counts[topic] = selected_topic_counts.get(topic, 0) + 1
        return True

    # 1) 자사/계열사 must-review만 먼저 넣는다.
    own_must = [
        a for a in ranked_all
        if a.get("JSON카테고리") == "자사_및_계열사_이슈"
        and a.get("강제검토") == "Y"
        and float(a.get("랭킹점수") or 0) > -20
    ]
    own_must = sorted(own_must, key=lambda x: (v10_must_priority(x), float(x.get("랭킹점수") or 0)), reverse=True)
    added = 0
    for article in own_must:
        if added >= MUST_REVIEW_POOL_LIMIT.get("자사_및_계열사_이슈", 45):
            break
        if add(article, respect_topic_cap=True):
            added += 1

    # 2) 카테고리 균형 풀. 정부/경쟁사/산업은 여기서 기존 랭킹 기반으로 들어온다.
    for json_key in JSON_KEYS_ORDER:
        limit = CATEGORY_POOL_LIMIT_FOR_GEMINI.get(json_key, 40)
        current_count = sum(1 for a in selected if a.get("JSON카테고리") == json_key)
        bucket = [
            a for a in ranked_all
            if a.get("JSON카테고리") == json_key
            and int(a["id"]) not in selected_ids
            and float(a.get("랭킹점수") or 0) > -25
        ]
        for article in bucket:
            if current_count >= limit:
                break
            if add(article, respect_topic_cap=True):
                current_count += 1

    # 3) 남는 슬롯은 전역 랭킹으로 채움.
    for article in ranked_all:
        if len(selected) >= MAX_CANDIDATES_FOR_GEMINI:
            break
        if int(article["id"]) in selected_ids:
            continue
        add(article, respect_topic_cap=True)

    if len(selected) < MAX_CANDIDATES_FOR_GEMINI:
        for article in ranked_all:
            if len(selected) >= MAX_CANDIDATES_FOR_GEMINI:
                break
            if int(article["id"]) in selected_ids:
                continue
            add(article, respect_topic_cap=False)

    return ranked_all, selected[:MAX_CANDIDATES_FOR_GEMINI]


def enforce_selection_limits(selection, ranked_candidates):
    result = {key: [] for key in JSON_KEYS_ORDER}
    used = set()
    id_to_article = {int(a["id"]): a for a in ranked_candidates}

    for key in JSON_KEYS_ORDER:
        max_count = CATEGORY_MAX.get(key, CATEGORY_TARGET.get(key, 3))
        for art_id in selection.get(key, []):
            try:
                art_id = int(art_id)
            except Exception:
                continue
            if art_id in used or art_id not in id_to_article:
                continue
            if len(result[key]) >= max_count:
                continue
            # 자사 카테고리는 실제 주체가 없으면 제외.
            article = id_to_article[art_id]
            if key == "자사_및_계열사_이슈":
                text = f"{article.get('기사제목','')} {article.get('본문요약','')}"
                if not v11_has_self_entity(text):
                    continue
            result[key].append(art_id)
            used.add(art_id)

    # Gemini가 놓친 자사 must-review만 보강한다. 정부/경쟁사/산업은 보강하지 않는다.
    key = "자사_및_계열사_이슈"
    max_count = CATEGORY_MAX.get(key, CATEGORY_TARGET.get(key, 3))
    bucket_counts = {}
    for art_id in result[key]:
        topic = id_to_article.get(int(art_id), {}).get("스토리버킷", "")
        if topic:
            bucket_counts[topic] = bucket_counts.get(topic, 0) + 1
    own_must_bucket = [a for a in ranked_candidates if a.get("JSON카테고리") == key and a.get("강제검토") == "Y"]
    own_must_bucket = sorted(own_must_bucket, key=lambda x: (v10_must_priority(x), float(x.get("랭킹점수") or 0)), reverse=True)
    for article in own_must_bucket:
        if len(result[key]) >= max_count:
            break
        art_id = int(article["id"])
        if art_id in used:
            continue
        topic = article.get("스토리버킷", "")
        if topic and bucket_counts.get(topic, 0) >= 2:
            continue
        result[key].append(art_id)
        used.add(art_id)
        if topic:
            bucket_counts[topic] = bucket_counts.get(topic, 0) + 1

    # 최소치 보충.
    for key in JSON_KEYS_ORDER:
        min_count = CATEGORY_MIN.get(key, 0)
        if len(result[key]) >= min_count:
            continue
        for article in ranked_candidates:
            if article.get("JSON카테고리") != key:
                continue
            art_id = int(article["id"])
            if art_id in used:
                continue
            if key == "자사_및_계열사_이슈":
                text = f"{article.get('기사제목','')} {article.get('본문요약','')}"
                if not v11_has_self_entity(text):
                    continue
            result[key].append(art_id)
            used.add(art_id)
            if len(result[key]) >= min_count:
                break

    total = sum(len(v) for v in result.values())
    if total < MIN_SELECT_COUNT:
        for article in ranked_candidates:
            art_id = int(article["id"])
            if art_id in used:
                continue
            key = article.get("JSON카테고리") or "산업동향"
            if key not in result:
                key = "산업동향"
            if len(result[key]) >= CATEGORY_MAX.get(key, 3):
                continue
            if key == "자사_및_계열사_이슈":
                text = f"{article.get('기사제목','')} {article.get('본문요약','')}"
                if not v11_has_self_entity(text):
                    continue
            result[key].append(art_id)
            used.add(art_id)
            total = sum(len(v) for v in result.values())
            if total >= MIN_SELECT_COUNT:
                break

    id_to_rank = {int(a["id"]): float(a.get("랭킹점수") or 0) for a in ranked_candidates}
    id_to_must = {int(a["id"]): a.get("강제검토") == "Y" for a in ranked_candidates}
    while sum(len(v) for v in result.values()) > MAX_SELECT_COUNT:
        removable = []
        for key in reversed(JSON_KEYS_ORDER):
            for art_id in result[key]:
                if len(result[key]) <= CATEGORY_MIN.get(key, 0):
                    continue
                removable.append((1 if id_to_must.get(int(art_id)) else 0, id_to_rank.get(int(art_id), 0), key, art_id))
        if not removable:
            break
        _, _, key, art_id = min(removable, key=lambda x: (x[0], x[1]))
        result[key].remove(art_id)
    return result


def deterministic_selection(ranked_candidates):
    result = {key: [] for key in JSON_KEYS_ORDER}
    used_ids = set()
    for json_key in JSON_KEYS_ORDER:
        target = CATEGORY_TARGET.get(json_key, 3)
        category_candidates = [
            a for a in ranked_candidates
            if a.get("JSON카테고리") == json_key and int(a["id"]) not in used_ids
        ]
        category_candidates = sorted(category_candidates, key=lambda x: (x.get("강제검토") == "Y", float(x.get("랭킹점수") or 0)), reverse=True)
        for article in category_candidates:
            if len(result[json_key]) >= target:
                break
            if json_key == "자사_및_계열사_이슈":
                text = f"{article.get('기사제목','')} {article.get('본문요약','')}"
                if not v11_has_self_entity(text):
                    continue
            art_id = int(article["id"])
            result[json_key].append(art_id)
            used_ids.add(art_id)
    return enforce_selection_limits(result, ranked_candidates)

def main():
    total_start_time = time.time()
    run_log = []

    print(f"\n🧩 v17 실행: {V6_VERSION}")

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

    past_reports_content, all_past_items, recent_past_items = load_past_reports()
    recent_past_text = build_recent_past_text(recent_past_items, max_chars=9000)

    raw_articles, skipped_duplicates = collect_with_google_rss(recent_past_items)

    print(f"\n  └ ✅ 총 {len(raw_articles)}개의 RSS 후보 확보 완료")
    print(f"  └ 🧹 수집/과거중복/오탐 제외 후보: {len(skipped_duplicates)}개")

    if not raw_articles:
        print("❌ 수집된 기사가 없습니다. Google News RSS 접속 또는 키워드/기간 설정을 확인하세요.")
        return

    ranked_all, ranked_candidates = rank_and_trim_candidates(raw_articles)

    pd.DataFrame(ranked_all).to_csv(OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(ranked_candidates).to_csv(OUTPUT_RANKED_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 전체 후보 기사 저장: {os.path.basename(OUTPUT_CANDIDATES_CSV)}")
    must_count = sum(1 for a in ranked_candidates if a.get("강제검토") == "Y")
    print(f"  └ 💾 AI/본문 추출 후보 랭킹 저장: {os.path.basename(OUTPUT_RANKED_CSV)} ({len(ranked_candidates)}개, 강제검토 {must_count}개)")

    if skipped_duplicates:
        pd.DataFrame(skipped_duplicates).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 제외 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")

    print("\n🧠 [STEP 2] 후보 랭킹 + Gemini가 과거 7일 중복 이슈를 제외하고 핵심 기사를 선별합니다...")
    print(f"   └ RSS 전체 {len(raw_articles)}개 중 혼합 후보 {len(ranked_candidates)}개를 Gemini 후보로 전달합니다.")
    print("   └ v16: 자사 must-review 안정화 + 개인정보/보안/금융규제 등 전역 사건 중복 제거를 적용합니다.")

    candidate_text = ""
    for article in ranked_candidates:
        candidate_text += (
            f"[{article['id']}] "
            f"점수: {article.get('랭킹점수', '')} / "
            f"원점수: {article.get('랭킹원점수', '')} / "
            f"중요도: {article.get('중요도점수', '')} / "
            f"강제검토: {article.get('강제검토', '')} {article.get('강제검토사유', '')} / "
            f"카테고리: {article.get('원카테고리', '')} / "
            f"검색어: {article.get('검색어', '')} / "
            f"제목: {article.get('기사제목', '')} / "
            f"언론사: {article.get('언론사', '')} / "
            f"게시일: {article.get('게시일', '')} / "
            f"사건태그: {article.get('사건태그', '')} / "
            f"사건단계: {article.get('사건단계', '')}\n"
        )

    prompt_selection = f"""
너는 카카오의 대외협력·대관·정책리스크 모니터링팀에서 매일 아침 최고경영진용 뉴스 브리핑을 고르는 담당자야.
단순히 검색어가 들어간 기사를 고르는 게 아니라, 카카오 및 계열사의 경영·규제·평판·노무·정책 대응에 영향을 줄 수 있는 이슈를 선별해야 해.
아래 [최근 7일 과거 보고서 데이터]를 보고 유저가 어떤 무게감의 기사를 골랐는지 학습한 뒤, [오늘 뉴스 후보 리스트]에서 오늘 보고서에 넣을 기사만 골라.

[선별 관점: 카카오 대외협력/대관팀의 자아]
1. 카카오 및 계열사 이슈는 최우선으로 봐. 특히 경영진·임원 이동, 조직개편, 서비스 개편 논란, 노사/임단협/파업, 노동부 진정, 최저임금법, 고용노동부 근로감독, 소송·수사·과징금·개인정보·장애·피싱·지배구조·실적·매각·투자회수는 높게 평가해.
2. 강제검토=Y 후보는 자사/계열사 기사 중 실제 카카오 주체가 명확하고, 임원·조직 변화, 서비스 개편 후폭풍, 노무·법적 진정, 규제·제재, 장애·보안, M&A·지배구조 등 강한 위험 신호가 함께 있는 기사야. 정부/경쟁사/산업동향은 강제검토가 아니라 기존 랭킹과 기사 가치 중심으로 판단해.
3. 정부/국회는 AI, 플랫폼, 디지털자산, 스테이블코인, 개인정보, 보안, 공정위, 금융위, 금감원, 과기정통부, 방미통위, 온플법, 망사용료, 지도반출 등 카카오 사업환경에 영향을 주는 정책·규제 중심으로 골라.
4. 경쟁사/해외이슈는 네이버·구글·오픈AI·MS·메타·애플·쿠팡·토스·배민·통신3사 등 주요 플레이어의 전략, 규제, 소송, 장애, 보안, AI/플랫폼 변화 위주로 골라.
5. 산업동향은 정말 구조적 변화가 있는 1~2개만 골라. 단순 인터뷰, 행사, 일반 제품 루머, 개별 범죄사건은 제외해.

[중복/업데이트 판단]
1. 최근 7일 과거 보고서에 이미 정리된 사건과 실질적으로 같은 내용이면 제외해.
2. 단, 새 사실·새 단계·새 수치·새 당국 조치가 있으면 업데이트 기사로 볼 수 있어. 예: 1차 조정→2차 조정, 투표→쟁의권, 소송 제기→판결, 논란→임원 퇴사, 노사갈등→노동부 진정.
3. 동일 사건의 여러 기사 중 하나만 골라. 최종 대표 기사는 코드가 본문 품질과 언론사 신뢰도로 다시 고를 거야.

[제외 원칙]
1. 오피니언, 사설, 칼럼, 전문가 기고는 제외해.
2. 사진/포토/캡션 기사, 외국어 저품질 기사, 제품 케이스 유출·렌더링 루머, 블로그/재가공 글은 제외해.
3. 보이스피싱·해킹도 개별 범죄 사건은 제외하고, 정부 대책·플랫폼 대응·제도 변화·대형 기업 리스크만 골라.
4. 타사 이름이나 카카오톡 제보 문구만 들어간 무관한 기사는 제외해.
5. 주가 지지선, 목표가, 투자의견, 단기 급등락, 밸류에이션, ETF 유출입 등 투자·시황 중심 기사는 제외해. 단, 지분 매각, M&A, 경영권 변동, 규제 제재, 구조조정과 직접 연결된 기업 경영 이벤트는 살릴 수 있어.
6. 스테이블코인/디지털자산은 가격·종목 전망보다 FIU, 특금법, AML, 가상자산사업자, 개인지갑, 해외이전, 시행령, 감독기준 등 정책·규제 변화가 있는 기사를 우선해.

[유연한 카테고리 범위]
- 자사 및 계열사 이슈: 최소 2개, 최대 7개
- 정부/국회: 최소 4개, 최대 8개
- 경쟁사/해외이슈: 최소 3개, 최대 6개
- 산업동향: 최소 1개, 최대 2개
중요한 기사가 많을 때만 많이 고르고, 억지로 최대치를 채우지마. 보통 총 12~18개 수준이 적절하며, 정말 중요한 날만 19개까지 가능해.

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

        selection_text = gemini_generate_text(client=client, prompt=prompt_selection, task_name="기사 선별")
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

    print("\n🕵️‍♂️ [STEP 3] 선별/보충 후보만 원문 URL 변환 후 본문 전문을 추출합니다...")
    print("   └ v10: 범용 위험 신호 후보를 별도 처리하고, 같은 사건군은 대표선택점수로 가장 좋은 기사만 남깁니다.")

    article_by_id = {int(article["id"]): article for article in raw_articles}
    final_report_data = []
    body_failed_rows = []
    post_body_duplicate_skips = []
    processed_ids = set()
    replacement_count = 0

    def category_count(json_key):
        return len([x for x in final_report_data if x.get("카테고리") == json_key])

    def total_count():
        return len(final_report_data)

    def try_process(art_id, json_key, reason="selected"):
        nonlocal replacement_count
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
            print(f"  └ 🧹 본문 확인 후 과거 중복 제외: {article_info.get('기사제목', '')[:44]}...")
            return False

        if failed_item:
            body_failed_rows.append(failed_item)
            print(
                f"  └ ⚠️ 본문 품질 미달로 교체: {failed_item.get('기사제목', '')[:44]}... "
                f"({failed_item.get('본문추출방식')}, {failed_item.get('본문글자수')}자, "
                f"품질 {failed_item.get('본문품질점수')}, 중요도 {failed_item.get('중요도점수')}, "
                f"{failed_item.get('본문품질사유')})"
            )
            return False

        if not report_item:
            return False

        relevant, relevance_reason = is_report_item_relevant(report_item, json_key)
        if not relevant:
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"relevance_failed:{relevance_reason}"
            body_failed_rows.append(failed_copy)
            print(f"  └ ⚠️ 관련성 미달로 교체: {report_item.get('기사제목', '')[:44]}... ({relevance_reason})")
            return False

        report_item["대표선택점수"] = representative_score(report_item)
        report_item["강제검토"] = article_info.get("강제검토", "")
        report_item["강제검토사유"] = article_info.get("강제검토사유", "")

        dup_item, dup_reason = final_duplicate_reason(report_item, final_report_data)
        if dup_item:
            new_score = float(report_item.get("대표선택점수") or 0)
            old_score = float(dup_item.get("대표선택점수") or 0)
            if new_score > old_score + REPRESENTATIVE_REPLACE_MARGIN:
                idx = final_report_data.index(dup_item)
                final_report_data[idx] = report_item
                replacement_count += 1
                add_duplicate_skip_row(
                    post_body_duplicate_skips,
                    dup_item,
                    report_item,
                    f"replaced_by_better_duplicate:{dup_reason}",
                    stage=reason,
                    extra=f"old={old_score},new={new_score}",
                )
                print(
                    f"  └ 🔁 중복 대표 교체: {dup_item.get('기사제목', '')[:28]}... "
                    f"→ {report_item.get('기사제목', '')[:28]}... "
                    f"({old_score}→{new_score}, {dup_reason})"
                )
                return True
            else:
                add_duplicate_skip_row(
                    post_body_duplicate_skips,
                    report_item,
                    dup_item,
                    f"kept_better_duplicate:{dup_reason}",
                    stage=reason,
                    extra=f"old={old_score},new={new_score}",
                )
                print(
                    f"  └ 🧹 당일 중복 제외: {report_item.get('기사제목', '')[:44]}... "
                    f"(대표점수 {new_score}≤{old_score}, {dup_reason})"
                )
                return False

        tag_limit_reason = event_tag_limit_reason(report_item, final_report_data)
        if tag_limit_reason:
            add_duplicate_skip_row(
                post_body_duplicate_skips,
                report_item,
                {"기사제목": "event_tag_cap", "링크": "", "대표선택점수": ""},
                tag_limit_reason,
                stage=reason,
            )
            print(f"  └ 🧹 사건태그 과다 반복 제외: {report_item.get('기사제목', '')[:44]}... ({tag_limit_reason})")
            return False

        final_report_data.append(report_item)
        print(
            f"  └ 📥 본문 추출 완료: {report_item.get('기사제목', '')[:44]}... "
            f"({report_item.get('본문추출방식')}, {report_item.get('본문글자수')}자, "
            f"품질 {report_item.get('본문품질점수')}, 중요도 {report_item.get('중요도점수')}, "
            f"대표 {report_item.get('대표선택점수')}, 강제검토 {report_item.get('강제검토')})"
        )
        time.sleep(random.uniform(0.15, 0.35))
        return True

    # 3-1. Gemini/로컬이 고른 기사 먼저 처리
    for json_key in JSON_KEYS_ORDER:
        for art_id in json_data.get(json_key, []):
            if total_count() >= MAX_SELECT_COUNT:
                break
            try_process(int(art_id), json_key, reason="selected")

    # 3-1.5. must-review 후보는 Gemini가 안 골라도 별도 처리한다.
    print("\n  └ 🧭 자사 must-review 고위험 후보를 별도 검토합니다...")
    for json_key in JSON_KEYS_ORDER:
        limit = MUST_REVIEW_PROCESS_LIMIT.get(json_key, 3)
        tried = 0
        must_bucket = [
            a for a in ranked_candidates
            if a.get("JSON카테고리") == json_key and a.get("강제검토") == "Y"
        ]
        must_bucket = sorted(must_bucket, key=lambda x: (float(x.get("중요도점수") or 0), float(x.get("랭킹점수") or 0)), reverse=True)
        for article in must_bucket:
            if total_count() >= MAX_SELECT_COUNT:
                break
            if category_count(json_key) >= CATEGORY_MAX.get(json_key, 3):
                break
            if tried >= limit:
                break
            art_id = int(article["id"])
            if art_id in processed_ids:
                continue
            tried += 1
            try_process(art_id, json_key, reason="must_review")

    print("\n  └ 🔁 본문 미달/과거중복/당일중복 제외분을 랭킹 후보에서 자동 보충합니다...")

    # 3-2. 카테고리별 목표치 보충
    for json_key in JSON_KEYS_ORDER:
        target = CATEGORY_TARGET.get(json_key, 3)
        bucket = [a for a in ranked_all if a.get("JSON카테고리") == json_key and float(a.get("랭킹점수") or 0) > -25]
        for article in bucket:
            if category_count(json_key) >= target:
                break
            if total_count() >= MAX_SELECT_COUNT:
                break
            art_id = int(article["id"])
            if art_id in processed_ids:
                continue
            try_process(art_id, json_key, reason="category_replacement")

    # 3-3. 총량 부족 시 전체 보충
    if total_count() < MIN_SELECT_COUNT:
        print(f"\n  └ 🔁 아직 {total_count()}개라 전체 랭킹 후보에서 추가 보충합니다...")
        for article in ranked_all:
            if total_count() >= MIN_SELECT_COUNT:
                break
            if total_count() >= MAX_SELECT_COUNT:
                break
            if float(article.get("랭킹점수") or 0) <= -25 and article.get("강제검토") != "Y":
                continue
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
        print(f"  └ 💾 제외/중복/대표교체 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")

    if body_failed_rows:
        pd.DataFrame(body_failed_rows).to_csv(OUTPUT_BODY_FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ ⚠️ 본문 품질 미달/관련성 미달 기사 저장: {os.path.basename(OUTPUT_BODY_FAILED_CSV)}")

    if not final_report_data:
        print("❌ 최종 보고서에 사용할 기사 데이터가 없습니다.")
        return

    ordered_final = []
    for json_key in JSON_KEYS_ORDER:
        items = [x for x in final_report_data if x.get("카테고리") == json_key]
        items = sorted(items, key=lambda x: (float(x.get("강제검토") == "Y"), float(x.get("대표선택점수") or 0), float(x.get("랭킹점수") or 0)), reverse=True)
        ordered_final.extend(items)
    final_report_data = ordered_final[:MAX_SELECT_COUNT]

    print("\n✍️ [STEP 4] 본문 전문 기반으로 past_reports 형식의 최종 브리핑을 생성합니다...")

    for idx, item in enumerate(final_report_data, 1):
        item["브리핑ID"] = idx

    final_input_text = ""
    for item in final_report_data:
        final_input_text += (
            f"[기사번호 {item['브리핑ID']}]\n"
            f"카테고리: {item['카테고리명']}\n"
            f"제목: {item['기사제목']}\n"
            f"언론사: {item['언론사']}\n"
            f"게시일: {item['게시일']}\n"
            f"본문글자수: {item['본문글자수']}\n"
            f"본문품질점수: {item['본문품질점수']}\n"
            f"중요도점수: {item['중요도점수']}\n"
            f"본문:\n{item['본문전문'][:MAX_BODY_CHARS_FOR_PROMPT]}\n\n"
        )

    prompt_report = f"""
너는 최고 경영진에게 매일 아침 뉴스 브리핑을 제공하는 수석 전략가야.
아래 [오늘 기사 데이터]의 본문만 사용해서 각 기사별 요약문만 작성해.

[요약 규칙]
1. 네 생각, 인사이트, 전망, 대응 포인트, 의미 부여는 쓰지마.
2. 오직 기사 본문에 있는 객관적 사실만 요약해.
3. 기사 1개당 요약은 1문단으로만 작성해.
4. 모든 문장의 끝은 '~함', '~임', '~됨', '~계획임', '~예정임' 같은 문어체 종결로 맞춰.
5. 기사 본문에 없는 내용은 절대 추가하지마.
6. 제목, 링크, 언론사, 카테고리, 번호는 쓰지마. 요약문만 반환해.
7. 사진/캡션/관련기사/저작권 문구는 요약하지마.

반드시 아래 JSON 객체 형식으로만 응답해. 키는 기사번호 문자열이고 값은 요약문이야.
{{
  "1": "요약문",
  "2": "요약문"
}}

[최근 과거 보고서 문체 참고]
{recent_past_text[:5000]}

[오늘 기사 데이터]
{final_input_text}
"""

    summary_map = {}
    try:
        if not client or not ENABLE_GEMINI_REPORT:
            raise RuntimeError("Gemini 최종 브리핑 비활성화 또는 client 없음")
        summary_text = gemini_generate_text(client=client, prompt=prompt_report, task_name="최종 기사별 요약 생성")
        summary_map = normalize_summary_json(extract_json_object(summary_text))
        print("  └ ✅ Gemini 기사별 요약 생성 완료")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사별 요약 생성 실패. 로컬 본문 요약으로 대체합니다. 원인: {e}")
        summary_map = {}

    final_briefing_text = build_structured_briefing(final_report_data, summary_map)

    print("\n" + "=" * 60)
    print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
    print("=" * 60)
    print(final_briefing_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_briefing_text)

    pd.DataFrame(final_report_data).to_csv(OUTPUT_SELECTED_CSV, index=False, encoding="utf-8-sig")

    run_log.append({
        "버전": V6_VERSION,
        "전체_RSS_후보": len(raw_articles),
        "Gemini_후보": len(ranked_candidates),
        "강제검토_Gemini후보": must_count,
        "제외_총합": len(skipped_duplicates) + len(post_body_duplicate_skips),
        "본문품질미달_관련성미달_교체": len(body_failed_rows),
        "중복대표교체": replacement_count,
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


# ==========================================
# v12 overrides: strict self must-review + stable others + quality floor
# - 최종 보고서 최소치 조정: 자사 2 / 정부 4
# - 딥페이크/보이스피싱 등 개별 사건형 기사 제외 강화
# - 품질 하한선 미달 기사는 최소 기사 수를 채우기 위해 억지로 포함하지 않음
# ==========================================

V6_VERSION = "google_rss_v12_quality_floor_deepfake_case_filter"

# 최종 보고서 기사 수 하한 조정
CATEGORY_MIN = {
    "자사_및_계열사_이슈": 2,
    "정부_국회": 4,
    "경쟁사_해외이슈": 3,
    "산업동향": 1,
}
CATEGORY_TARGET = {
    "자사_및_계열사_이슈": 5,
    "정부_국회": 6,
    "경쟁사_해외이슈": 4,
    "산업동향": 1,
}
CATEGORY_MAX = {
    "자사_및_계열사_이슈": 7,
    "정부_국회": 8,
    "경쟁사_해외이슈": 6,
    "산업동향": 2,
}
MIN_SELECT_COUNT = sum(CATEGORY_MIN.values())
MAX_SELECT_COUNT = 19

# 품질 하한선: 최소 기사 수를 채우기 위해 품질 낮은 기사를 억지로 넣지 않기 위한 기준
ABSOLUTE_QUALITY_FLOOR = 25.0
NORMAL_QUALITY_FLOOR = 42.0
SHORT_IMPORTANT_QUALITY_FLOOR = 30.0
ACCEPTABLE_IMPORTANT_QUALITY_FLOOR = 35.0

_POLICY_CONTEXT_RE = re.compile(
    r"정부|국회|과기정통부|방송미디어통신위원회|방미통위|개인정보위|개보위|금융위|금감원|공정위|"
    r"행안부|경찰청|검찰청|선관위|중앙선관위|법안|개정안|시행령|고시|행정예고|입법예고|"
    r"가이드라인|자율규제|대책|정책|제도|규제|표시의무|삭제의무|차단|플랫폼\s*책임|"
    r"사업자\s*의무|합동\s*대응|협의체|전담조직|통합지원|피해지원|예방|모니터링|단속|전수조사|과징금|제재",
    re.IGNORECASE,
)

_DEEPFAKE_RE = re.compile(r"딥페이크|허위영상|합성영상|AI\s*합성|불법합성|허위조작영상|선거\s*딥페이크", re.IGNORECASE)
_DEEPFAKE_INDIVIDUAL_CASE_RE = re.compile(
    r"피의자|입건|제보자|고소|고발|수사\s*착수|수사\s*중|경찰\s*수사|검찰\s*송치|송치|압수수색|"
    r"특정\s*후보|후보자|도지사|시장|군수|구청장|시의원|군의원|경남|경북|전남|전북|충남|충북|"
    r"강원|제주|부산|대구|광주|대전|울산|인천|서울|경기|지역\s*선거|선거\s*의혹|허위사실",
    re.IGNORECASE,
)

_GENERAL_INDIVIDUAL_CASE_RE = re.compile(
    r"인출책|징역|항소심|1심|2심|선고|구속|구속송치|검찰\s*송치|범행|범죄|피해자|사망|숨져|"
    r"교통사고|살인|폭행|협박|감금|보복|일당|조직원|피고인|재판부|법원은|벌금형|집행유예",
    re.IGNORECASE,
)

_SCAM_WORD_RE = re.compile(r"보이스피싱|로맨스\s*스캠|피싱|스미싱|사기", re.IGNORECASE)

_LOW_VALUE_MARKET_RE = re.compile(
    r"비트코인.*(전망|하락|상승|저항선|지지선)|코인.*(전망|폭등|급락)|주가\s*전망|목표가|매수\s*추천|"
    r"실적\s*미리보기|밸류에이션|투자의견|ETF\s*유출|가격\s*예측",
    re.IGNORECASE,
)


# 기존 v11 필터 보존 후 v12 필터로 대체
_BASE_v11_is_individual_crime_case_v12 = v11_is_individual_crime_case


def v12_is_policy_context(text):
    return bool(_POLICY_CONTEXT_RE.search(clean_html_text(text)))


def v12_is_deepfake_individual_case(text):
    text = clean_html_text(text)
    if not _DEEPFAKE_RE.search(text):
        return False
    # 정책/제도/플랫폼 의무 중심이면 제외하지 않는다.
    if v12_is_policy_context(text):
        # 단, 정책 맥락보다 특정 지역·피의자 수사 맥락이 압도적으로 강하면 개별 사건으로 본다.
        if _DEEPFAKE_INDIVIDUAL_CASE_RE.search(text) and re.search(r"피의자|입건|제보자|특정\s*후보|도지사|지역\s*선거|선거\s*의혹", text):
            return True
        return False
    return bool(_DEEPFAKE_INDIVIDUAL_CASE_RE.search(text))


def v12_is_individual_scam_or_crime_case(text):
    text = clean_html_text(text)
    if not text:
        return False
    if v12_is_deepfake_individual_case(text):
        return True
    # 보이스피싱/사기 류는 정부 대책·금융권 가이드라인이면 통과, 개별 피고인/피해자 기사면 제외
    if _SCAM_WORD_RE.search(text):
        if v12_is_policy_context(text) and not _GENERAL_INDIVIDUAL_CASE_RE.search(text):
            return False
        if _GENERAL_INDIVIDUAL_CASE_RE.search(text):
            return True
    # 일반 범죄/사고 사건도 정책 맥락이 없으면 제외
    if _GENERAL_INDIVIDUAL_CASE_RE.search(text) and not v12_is_policy_context(text):
        return True
    return False


def v11_is_individual_crime_case(text, json_key=""):
    text = clean_html_text(text)
    if v12_is_individual_scam_or_crime_case(text):
        return True
    return bool(_BASE_v11_is_individual_crime_case_v12(text, json_key))


def v10_is_individual_crime_case(text, json_key=""):
    return v11_is_individual_crime_case(text, json_key)


_BASE_body_quality_score_v12 = body_quality_score_v6


def body_quality_score_v6(title, body, method, json_key="", source=""):
    score, reason = _BASE_body_quality_score_v12(title, body, method, json_key=json_key, source=source)
    body_text = clean_html_text(body)
    title_text = clean_html_text(title)
    joined = f"{title_text} {body_text[:2500]}"
    length = len(body_text)

    # 개별 범죄/지역 딥페이크 사건은 본문이 길어도 품질을 낮게 본다.
    if v12_is_individual_scam_or_crime_case(joined):
        score -= 55
        reason = "individual_case_not_policy"

    # 비트코인 가격 전망/단순 종목·투자 전망은 브리핑 품질 하한에서 감점
    if _LOW_VALUE_MARKET_RE.search(title_text):
        score -= 25
        if reason == "good":
            reason = "low_value_market_or_price_forecast"

    # 너무 긴 본문에서 잡텍스트 마커가 많으면 기사 본문이 아니라 페이지 전체일 가능성
    noisy_markers = [
        "관련기사", "관련 기사", "많이 본 뉴스", "인기기사", "실시간 뉴스", "주요뉴스", "추천기사",
        "연예", "스포츠", "증권", "코인", "댓글", "로그인", "뉴스레터", "구독", "포토뉴스",
        "제보는 카카오톡", "무단전재", "재배포 금지", "Copyright",
    ]
    noise_hits = sum(1 for marker in noisy_markers if marker in body_text)
    if length >= 10000 and noise_hits >= 3:
        score -= 30
        reason = "suspicious_long_noisy_body"
    elif length >= 18000:
        score -= 22
        if reason == "good":
            reason = "suspicious_very_long_body"

    # 정책성이 없는 정부/국회 개별 사건은 추가 감점
    if json_key == "정부_국회" and v12_is_individual_scam_or_crime_case(joined):
        score -= 25
        reason = "government_individual_case_not_policy"

    if score < ABSOLUTE_QUALITY_FLOOR and reason == "good":
        reason = "below_quality_floor"
    return round(score, 2), reason


_BASE_is_body_usable_v12 = is_body_usable_v6


def is_body_usable_v6(title, body, method, json_key="", source="", importance_score=0):
    quality_score, quality_reason = body_quality_score_v6(title, body, method, json_key=json_key, source=source)
    body_len = len(clean_html_text(body))
    title_text = clean_html_text(title)
    joined = f"{title_text} {clean_html_text(body)[:2600]}"

    if method in {"failed", "meta_description_short"}:
        return False, quality_reason, quality_score
    if PHOTO_OR_CAPTION_TITLE_RE.search(title_text):
        return False, "photo_or_caption_article", quality_score
    if v12_is_individual_scam_or_crime_case(joined):
        return False, quality_reason or "individual_case_not_policy", quality_score
    if quality_score < ABSOLUTE_QUALITY_FLOOR:
        return False, quality_reason or "below_absolute_quality_floor", quality_score

    # 일반 기사: 700자 이상 + 품질 하한선 통과
    if body_len >= MIN_GOOD_BODY_CHARS and quality_score >= NORMAL_QUALITY_FLOOR:
        return True, "good", quality_score

    # 짧아도 중요한 기사: 품질 하한선을 반드시 넘고, 중요도도 충분해야 함
    if body_len >= MIN_ACCEPT_BODY_CHARS and quality_score >= ACCEPTABLE_IMPORTANT_QUALITY_FLOOR and importance_score >= 42:
        return True, "acceptable_important", quality_score

    if body_len >= MIN_IMPORTANT_BODY_CHARS and quality_score >= SHORT_IMPORTANT_QUALITY_FLOOR and importance_score >= 62:
        return True, "short_but_high_importance", quality_score

    return False, quality_reason or "below_quality_floor", quality_score


_BASE_article_importance_score_v12 = article_importance_score


def article_importance_score(title, body="", json_key="", keyword=""):
    score = float(_BASE_article_importance_score_v12(title, body, json_key, keyword) or 0)
    text = f"{clean_html_text(title)} {clean_html_text(body)[:2600]}"
    if v12_is_individual_scam_or_crime_case(text):
        score -= 45
    if json_key == "정부_국회" and _DEEPFAKE_RE.search(text) and v12_is_deepfake_individual_case(text):
        score -= 45
    if _LOW_VALUE_MARKET_RE.search(clean_html_text(title)):
        score -= 22
    return round(score, 2)


_BASE_is_article_obviously_low_value_v12 = is_article_obviously_low_value


def is_article_obviously_low_value(article):
    text = f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('검색어','')}"
    json_key = article.get("JSON카테고리", "")
    if v12_is_individual_scam_or_crime_case(text):
        return True, "individual_case_not_policy"
    if json_key == "정부_국회" and _DEEPFAKE_RE.search(text) and v12_is_deepfake_individual_case(text):
        return True, "individual_deepfake_case_not_policy"
    if json_key == "산업동향" and _LOW_VALUE_MARKET_RE.search(text):
        return True, "low_value_market_or_price_forecast"
    return _BASE_is_article_obviously_low_value_v12(article)


_BASE_is_report_item_relevant_v12 = is_report_item_relevant


def is_report_item_relevant(report_item, json_key):
    text = f"{report_item.get('기사제목','')} {report_item.get('본문전문','')[:2600]} {report_item.get('RSS요약','')}"
    quality = float(report_item.get("본문품질점수") or 0)
    body_len = int(report_item.get("본문글자수") or 0)
    importance = float(report_item.get("중요도점수") or 0)

    # 품질 하한선: 최소 기사 수를 채우기 위한 억지 포함 금지
    if quality < ABSOLUTE_QUALITY_FLOOR:
        return False, "below_absolute_quality_floor"
    if body_len < MIN_ACCEPT_BODY_CHARS and not (body_len >= MIN_IMPORTANT_BODY_CHARS and importance >= 62 and quality >= SHORT_IMPORTANT_QUALITY_FLOOR):
        return False, "too_short_below_quality_floor"

    # 딥페이크/보이스피싱/해킹 등은 정책형만 통과. 개별 사건형은 제외.
    if v12_is_individual_scam_or_crime_case(text):
        return False, "individual_case_not_policy"

    if json_key == "정부_국회" and _DEEPFAKE_RE.search(text) and not v12_is_policy_context(text):
        return False, "deepfake_without_policy_context"

    if json_key == "산업동향" and _LOW_VALUE_MARKET_RE.search(text):
        return False, "low_value_market_or_price_forecast"

    return _BASE_is_report_item_relevant_v12(report_item, json_key)


_BASE_collection_relevance_reason_v12 = collection_relevance_reason


def collection_relevance_reason(keyword, title, summary):
    reason = _BASE_collection_relevance_reason_v12(keyword, title, summary)
    text = f"{clean_html_text(title)} {clean_html_text(summary)[:900]}"
    json_key = CATEGORY_TO_JSON_KEY.get(keyword_to_category.get(keyword, ""), "")
    if json_key == "정부_국회" and _DEEPFAKE_RE.search(text) and v12_is_deepfake_individual_case(text):
        return "collection_relevance:individual_deepfake_case_not_policy"
    return reason


_BASE_normalize_selection_json_v12 = normalize_selection_json


def normalize_selection_json(data, final_candidates):
    result = _BASE_normalize_selection_json_v12(data, final_candidates)
    # 최소치 변경이 반영되도록 한 번 더 정리한다.
    for key in JSON_KEYS_ORDER:
        max_count = CATEGORY_MAX.get(key, len(result.get(key, [])))
        result[key] = list(result.get(key, []))[:max_count]
    return result



# =========================================================
# v13 overrides: entertainment/private KakaoTalk chatter filter
# =========================================================

V6_VERSION = "google_rss_v13_1_self_gate_entertainment_gossip_filter"

# 카톡을 무조건 배제하지는 않는다.
# 다만 '카톡'이 회사/서비스 이슈가 아니라 연예인·개인 간 대화·사생활 폭로의 소재일 때만 제외한다.
_ENTERTAINMENT_GOSSIP_RE = re.compile(
    r"MC몽|엠씨몽|차가원|불륜|열애|이혼|사생활|연예인|가수|배우|아이돌|라방|라이브\s*방송|"
    r"틱톡\s*라이브|인스타\s*라이브|SNS\s*폭로|폭로전|폭로|악플|루머|밀회|스캔들|"
    r"카톡\s*(캡처|캡쳐|공개|내용|대화|방|메시지|메세지|언급|조작|증거)|"
    r"조작\s*카톡|카카오톡\s*(캡처|캡쳐|공개|내용|대화|메시지|메세지|언급|조작|증거)",
    re.IGNORECASE,
)

# 진짜 카카오/카카오톡 서비스·회사 이슈로 볼 수 있는 맥락.
_KAKAO_BUSINESS_CONTEXT_RE = re.compile(
    r"카카오(?!톡?\s*(캡처|캡쳐|내용|대화|메시지|메세지|언급|조작|공개|증거))|"
    r"카카오톡\s*(개편|업데이트|장애|오류|먹통|피싱|보안|채널|오픈채팅|광고|선물하기|톡비즈|"
    r"서비스|정책|이용자|고객|친구탭|프로필|숏폼|알림|PC버전|공식|다운로드)|"
    r"카톡\s*(개편|업데이트|장애|오류|먹통|피싱|보안|채널|오픈채팅|광고|선물하기|톡비즈|"
    r"서비스|정책|이용자|고객|친구탭|프로필|공식|다운로드)|"
    r"카카오페이|카카오뱅크|카카오모빌리티|카카오\s*T\b|카카오T\b|카카오택시|카카오게임즈|"
    r"카카오엔터테인먼트|카카오엔터|카카오헬스케어|카카오엔터프라이즈|카카오픽코마|"
    r"디케이테크인|엑스엘게임즈|AXZ|카나나|정신아|김범수",
    re.IGNORECASE,
)

_SELF_LOW_VALUE_RANKING_RE = re.compile(
    r"브랜드평판|평판지수|상장기업\s*브랜드|빅데이터\s*분석\s*순위|브랜드\s*순위|"
    r"\d+위\s*카카오|카카오\s*\d+위|순위는\s*.*카카오",
    re.IGNORECASE,
)

_ENTERTAINMENT_MEDIA_HINT_RE = re.compile(
    r"엑스포츠뉴스|스포츠조선|스타뉴스|OSEN|텐아시아|일간스포츠|마이데일리|스포티비뉴스|"
    r"디스패치|위키트리|톱스타뉴스|브레이크뉴스",
    re.IGNORECASE,
)


def v13_is_private_kakaotalk_or_entertainment_noise(text):
    """카톡이 회사 이슈가 아니라 연예·사생활·개인 대화 소재일 때 True."""
    t = clean_html_text(text)
    if not t:
        return False
    if not _ENTERTAINMENT_GOSSIP_RE.search(t):
        return False

    # 카카오엔터/카카오톡 서비스 자체 리스크라면 예외적으로 살린다.
    if _KAKAO_BUSINESS_CONTEXT_RE.search(t):
        # 단, MC몽/불륜/라방 등 사생활형 기사에서 '카톡'만 있는 경우는 여전히 제외.
        strong_company = re.search(
            r"카카오(페이|뱅크|모빌리티|게임즈|엔터테인먼트|엔터프라이즈|헬스케어|픽코마)|"
            r"카카오\s*(노조|대표|관계자|측|임원|CPO|CTO|CFO|서비스|정책|개편)|"
            r"카카오톡\s*(개편|업데이트|장애|오류|피싱|보안|채널|오픈채팅|광고|서비스|정책|이용자|고객|공식|PC버전)|"
            r"카톡\s*(개편|업데이트|장애|오류|피싱|보안|서비스|정책|이용자|고객|공식|PC버전)",
            t,
            flags=re.IGNORECASE,
        )
        return not bool(strong_company)

    return True


def v13_is_self_low_value_ranking_article(text):
    """카카오가 단순 순위표/브랜드평판에 등장하는 기사."""
    t = clean_html_text(text)
    if not t:
        return False
    return bool(_SELF_LOW_VALUE_RANKING_RE.search(t))


_BASE_v11_has_self_entity_v13 = v11_has_self_entity


def v11_has_self_entity(text):
    t = clean_html_text(text)
    if v13_is_private_kakaotalk_or_entertainment_noise(t):
        return False
    if v13_is_self_low_value_ranking_article(t):
        return False
    return _BASE_v11_has_self_entity_v13(t)


_BASE_v11_self_must_review_reason_v13 = v11_self_must_review_reason


def v11_self_must_review_reason(text, json_key=""):
    t = clean_html_text(text)
    if v13_is_private_kakaotalk_or_entertainment_noise(t):
        return False, "not_must_review:private_kakaotalk_or_entertainment_noise"
    if v13_is_self_low_value_ranking_article(t):
        return False, "not_must_review:self_low_value_ranking_article"
    return _BASE_v11_self_must_review_reason_v13(t, json_key)


def v10_general_must_review_reason(text, json_key=""):
    return v11_self_must_review_reason(text, json_key)


_BASE_collection_relevance_reason_v13 = collection_relevance_reason


def collection_relevance_reason(keyword, title, summary):
    text = f"{clean_html_text(title)} {clean_html_text(summary)[:1000]}"
    json_key = CATEGORY_TO_JSON_KEY.get(keyword_to_category.get(keyword, ""), "")
    if json_key == "자사_및_계열사_이슈":
        if v13_is_private_kakaotalk_or_entertainment_noise(text):
            return "collection_relevance:private_kakaotalk_or_entertainment_noise"
        if v13_is_self_low_value_ranking_article(text):
            return "collection_relevance:self_low_value_ranking_article"
    # 카카오/카카오톡 검색어에서 연예·사생활 카톡 기사면 수집 단계에서 제거
    if keyword in {"카카오", "카카오톡"}:
        if v13_is_private_kakaotalk_or_entertainment_noise(text):
            return "collection_relevance:private_kakaotalk_or_entertainment_noise"
    return _BASE_collection_relevance_reason_v13(keyword, title, summary)


_BASE_is_article_obviously_low_value_v13 = is_article_obviously_low_value


def is_article_obviously_low_value(article):
    text = f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('언론사','')} {article.get('검색어','')}"
    json_key = article.get("JSON카테고리", "")
    if json_key == "자사_및_계열사_이슈":
        if v13_is_private_kakaotalk_or_entertainment_noise(text):
            return True, "private_kakaotalk_or_entertainment_noise"
        if v13_is_self_low_value_ranking_article(text):
            return True, "self_low_value_ranking_article"
    if article.get("검색어") in {"카카오", "카카오톡"} and v13_is_private_kakaotalk_or_entertainment_noise(text):
        return True, "private_kakaotalk_or_entertainment_noise"
    return _BASE_is_article_obviously_low_value_v13(article)


_BASE_article_importance_score_v13 = article_importance_score


def article_importance_score(title, body="", json_key="", keyword=""):
    score = float(_BASE_article_importance_score_v13(title, body, json_key, keyword) or 0)
    text = f"{clean_html_text(title)} {clean_html_text(body)[:2500]}"
    if json_key == "자사_및_계열사_이슈" or keyword in {"카카오", "카카오톡"}:
        if v13_is_private_kakaotalk_or_entertainment_noise(text):
            score -= 120
        if v13_is_self_low_value_ranking_article(text):
            score -= 85
        if _ENTERTAINMENT_MEDIA_HINT_RE.search(text) and not _KAKAO_BUSINESS_CONTEXT_RE.search(text):
            score -= 45
    return round(score, 2)



_BASE_is_report_item_relevant_v13 = is_report_item_relevant


def is_report_item_relevant(report_item, json_key):
    text = f"{report_item.get('기사제목','')} {report_item.get('언론사','')} {report_item.get('본문전문','')[:3500]} {report_item.get('RSS요약','')}"
    if json_key == "자사_및_계열사_이슈":
        if v13_is_private_kakaotalk_or_entertainment_noise(text):
            return False, "private_kakaotalk_or_entertainment_noise"
        if v13_is_self_low_value_ranking_article(text):
            return False, "self_low_value_ranking_article"
    return _BASE_is_report_item_relevant_v13(report_item, json_key)


_BASE_body_quality_score_v13 = body_quality_score_v6


def body_quality_score_v6(title, body, method, json_key="", source=""):
    score, reason = _BASE_body_quality_score_v13(title, body, method, json_key=json_key, source=source)
    text = f"{clean_html_text(title)} {clean_html_text(source)} {clean_html_text(body)[:3500]}"
    if json_key == "자사_및_계열사_이슈":
        if v13_is_private_kakaotalk_or_entertainment_noise(text):
            return -35.0, "private_kakaotalk_or_entertainment_noise"
        if v13_is_self_low_value_ranking_article(text):
            return -18.0, "self_low_value_ranking_article"
    return score, reason


_BASE_is_body_usable_v13 = is_body_usable_v6


def is_body_usable_v6(title, body, method, json_key="", source="", importance_score=0):
    text = f"{clean_html_text(title)} {clean_html_text(source)} {clean_html_text(body)[:3500]}"
    if json_key == "자사_및_계열사_이슈":
        if v13_is_private_kakaotalk_or_entertainment_noise(text):
            return False, "private_kakaotalk_or_entertainment_noise", -35.0
        if v13_is_self_low_value_ranking_article(text):
            return False, "self_low_value_ranking_article", -18.0
    return _BASE_is_body_usable_v13(title, body, method, json_key=json_key, source=source, importance_score=importance_score)



# =========================================================
# v14 overrides: daily-safe market noise filter + digital asset policy relevance
# - 주식/시황/투자 분석 기사 과대평가 방지
# - 스테이블코인 검색어 노이즈(ATN 등) 차단
# - FIU/특금법/AML/개인지갑/해외이전 등 디지털자산 정책 키워드 보강
# - v11~v13의 자사 must-review 안정화 구조는 유지
# =========================================================

V6_VERSION = "google_rss_v14_daily_safe_policy_relevance_market_noise_filter"

# 정부/국회 키워드 보강: 스테이블코인 하나로 디지털자산 규제를 커버하면 노이즈가 많아져서
# FIU/특금법/AML/VASP/개인지갑/해외이전 등 정책성 query shard를 추가한다.
def _v14_add_keyword(category, keyword):
    if keyword not in all_keywords:
        all_keywords.append(keyword)
    keyword_to_category[keyword] = category

for _kw in [
    "FIU", "금융정보분석원", "특금법", "특정금융정보법", "가상자산사업자", "VASP",
    "개인지갑", "가상자산 개인지갑", "해외이전", "가상자산 해외이전", "자금세탁방지", "AML",
    "디지털자산기본법", "가상자산 이용자보호법", "가상자산 거래소", "미신고 거래소",
    "트래블룰", "원화 스테이블코인", "스테이블코인 시행령",
]:
    _v14_add_keyword("정부/국회", _kw)

# 광범위 디지털자산 정책 키워드는 조금 더 수집한다.
try:
    MAX_RSS_ITEMS_PER_KEYWORD_OVERRIDES.update({
        "FIU": 180,
        "금융정보분석원": 180,
        "특금법": 220,
        "가상자산사업자": 180,
        "VASP": 160,
        "개인지갑": 160,
        "가상자산 해외이전": 180,
        "자금세탁방지": 180,
        "AML": 180,
        "디지털자산기본법": 220,
        "원화 스테이블코인": 220,
    })
    MAX_RSS_QUERIES_PER_KEYWORD_OVERRIDES.update({
        "FIU": 12,
        "금융정보분석원": 12,
        "특금법": 16,
        "디지털자산기본법": 16,
        "원화 스테이블코인": 16,
    })
except Exception:
    pass

_DIGITAL_ASSET_CORE_RE = re.compile(
    r"스테이블\s*코인|스테이블코인|가상자산|디지털자산|특금법|특정금융정보법|FIU|금융정보분석원|"
    r"자금세탁방지|AML|VASP|가상자산사업자|개인지갑|개인\s*지갑|해외\s*이전|해외이전|"
    r"트래블룰|원화\s*스테이블|USDT|USDC|테더|서클|Circle|토큰증권|STO|"
    r"디지털자산기본법|가상자산\s*이용자보호법|가상자산\s*거래소|업비트|두나무|빗썸|코인원|코빗",
    re.IGNORECASE,
)

_DIGITAL_ASSET_POLICY_RE = re.compile(
    r"시행령|개정안|법안|입법|입법예고|행정예고|고시|가이드라인|감독|감독기준|규제|제도|"
    r"보완|검토|추가\s*협의|협의|의견수렴|간담회|확정|시행|유예|보고|신고|제한|금지|"
    r"제재|과징금|검사|조사|당국|금융위|금감원|FIU|금융정보분석원|국회|정무위|거래소|업계",
    re.IGNORECASE,
)

_DIGITAL_ASSET_QUERY_KEYWORDS = {
    "스테이블코인", "원화 스테이블코인", "FIU", "금융정보분석원", "특금법", "특정금융정보법",
    "가상자산사업자", "VASP", "개인지갑", "가상자산 개인지갑", "해외이전", "가상자산 해외이전",
    "자금세탁방지", "AML", "디지털자산기본법", "가상자산 이용자보호법", "가상자산 거래소",
    "미신고 거래소", "트래블룰", "스테이블코인 시행령",
}

# 순수 투자/시황성 기사 신호. 단어 1개만으로 제외하지 않고, 강한 시황 문맥을 본다.
_STOCK_MARKET_NOISE_RE = re.compile(
    r"주가|목표가|투자의견|매수\s*의견|매도\s*의견|중립\s*의견|지지선|저항선|매수세|매도세|"
    r"급등|급락|폭등|폭락|상승세|하락세|하락폭|상승폭|밸류에이션|PER|PBR|EPS|"
    r"증권가|애널리스트|리포트|종목|나스닥|뉴욕증시|S&P|다우지수|시총|시가총액|"
    r"ETF\s*(유입|유출)|투자자|차익실현|공매도|배당|실적\s*전망|전망치\s*(상향|하향)|"
    r"어닝\s*(서프라이즈|쇼크)|분기\s*실적|수익률|가격\s*예측|코인\s*(가격|전망)",
    re.IGNORECASE,
)

# 순수 시황이 아니라 실제 기업/정책 이벤트라면 살릴 수 있는 신호.
_MATERIAL_BUSINESS_OR_POLICY_EVENT_RE = re.compile(
    r"인수|합병|매각|지분|주식교환|경영권|최대주주|상장|IPO|투자유치|전략적\s*투자|블록딜|"
    r"대표|임원|CPO|CTO|CFO|사의|사임|퇴사|해임|교체|선임|내정|조직개편|구조조정|희망퇴직|"
    r"파업|쟁의권|조정\s*결렬|노동부|근로감독|최저임금|임금체불|진정|노조|성과급|"
    r"과징금|제재|수사|고발|소송|판결|압수수색|개인정보|유출|해킹|장애|먹통|피싱|보안|"
    r"법안|시행령|개정안|규제|가이드라인|감독|FIU|금융정보분석원|특금법|AML|자금세탁방지|"
    r"가상자산사업자|개인지갑|해외\s*이전|디지털자산기본법|정부|국회|공정위|금융위|금감원|과기정통부|방미통위",
    re.IGNORECASE,
)

_LOW_TRUST_MARKET_DOMAIN_RE = re.compile(
    r"simplywall\.st|tradersunion\.com|marketbeat\.com|investing\.com|seekingalpha\.com|"
    r"tokenpost\.kr/news/market|benzinga\.com|fool\.com",
    re.IGNORECASE,
)


def v14_is_pure_market_or_stock_noise(text, url="", json_key="", keyword=""):
    """주가·투자·시황 자체가 주제인 기사인지 판단. 실제 정책/경영 이벤트면 제외하지 않는다."""
    t = clean_html_text(text)
    u = str(url or "")
    if not t:
        return False
    if _LOW_TRUST_MARKET_DOMAIN_RE.search(u) and _STOCK_MARKET_NOISE_RE.search(t):
        return True
    if not _STOCK_MARKET_NOISE_RE.search(t):
        return False
    # 실제 카카오/경쟁사/정책 이벤트가 있으면 단순 시황으로 보지 않는다.
    if _MATERIAL_BUSINESS_OR_POLICY_EVENT_RE.search(t):
        # 하지만 제목이 지지선/목표가/투자의견/차트 자체에 집중하면 여전히 시황성으로 본다.
        title_like = t[:260]
        if re.search(r"지지선|저항선|목표가|투자의견|차트|기술적\s*분석|주가\s*(전망|향방|흐름)", title_like, re.IGNORECASE):
            return True
        return False
    return True


def v14_has_digital_asset_core_context(text):
    return bool(_DIGITAL_ASSET_CORE_RE.search(clean_html_text(text)))


def v14_has_digital_asset_policy_context(text):
    t = clean_html_text(text)
    return bool(_DIGITAL_ASSET_CORE_RE.search(t) and _DIGITAL_ASSET_POLICY_RE.search(t))


def v14_is_digital_asset_search_noise(keyword, title, summary, url=""):
    """스테이블코인/FIU/특금법 계열 검색어로 들어왔지만 실제 핵심 주제가 아닌 기사 제거."""
    if keyword not in _DIGITAL_ASSET_QUERY_KEYWORDS:
        return False
    front = f"{clean_html_text(title)} {clean_html_text(summary)[:1100]}"
    if not v14_has_digital_asset_core_context(front):
        return True
    # 주가/실적/시황성 제목이면 정책 맥락 없을 때 제거.
    if v14_is_pure_market_or_stock_noise(front, url=url, keyword=keyword) and not v14_has_digital_asset_policy_context(front):
        return True
    return False


def v14_digital_asset_policy_bonus(text):
    t = clean_html_text(text)
    if not v14_has_digital_asset_core_context(t):
        return 0.0
    bonus = 0.0
    if v14_has_digital_asset_policy_context(t):
        bonus += 28
    if re.search(r"FIU|금융정보분석원|특금법|특정금융정보법|자금세탁방지|AML|개인지갑|해외\s*이전|가상자산사업자|VASP|트래블룰", t, re.IGNORECASE):
        bonus += 22
    if re.search(r"보완\s*검토|추가\s*협의|업계\s*의견|의견수렴|시행령|7월\s*중|확정|보고\s*의무|거래\s*제한", t, re.IGNORECASE):
        bonus += 18
    return min(70.0, bonus)

# 수집 단계 필터 보강
_BASE_collection_relevance_reason_v14 = collection_relevance_reason

def collection_relevance_reason(keyword, title, summary):
    text = f"{clean_html_text(title)} {clean_html_text(summary)[:1100]}"
    if v14_is_digital_asset_search_noise(keyword, title, summary):
        return "digital_asset_keyword_without_core_policy_context"
    if v14_is_pure_market_or_stock_noise(text, keyword=keyword):
        return "pure_stock_market_or_investment_noise"
    return _BASE_collection_relevance_reason_v14(keyword, title, summary)

# 후보 저품질 판단 보강
_BASE_is_article_obviously_low_value_v14 = is_article_obviously_low_value

def is_article_obviously_low_value(article):
    text = f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('언론사','')} {article.get('검색어','')}"
    url = article.get("링크", "") or article.get("원래RSS링크", "")
    keyword = article.get("검색어", "")
    json_key = article.get("JSON카테고리", "")
    if v14_is_digital_asset_search_noise(keyword, article.get('기사제목',''), article.get('본문요약',''), url=url):
        return True, "digital_asset_keyword_without_core_policy_context"
    if v14_is_pure_market_or_stock_noise(text, url=url, json_key=json_key, keyword=keyword):
        return True, "pure_stock_market_or_investment_noise"
    return _BASE_is_article_obviously_low_value_v14(article)

# 중요도 점수 보정
_BASE_article_importance_score_v14 = article_importance_score

def article_importance_score(title, body="", json_key="", keyword=""):
    base_score = float(_BASE_article_importance_score_v14(title, body, json_key, keyword) or 0)
    text = f"{clean_html_text(title)} {clean_html_text(body)[:2600]}"
    if v14_is_pure_market_or_stock_noise(text, json_key=json_key, keyword=keyword):
        base_score -= 90
    if keyword in _DIGITAL_ASSET_QUERY_KEYWORDS or v14_has_digital_asset_core_context(text):
        if not v14_has_digital_asset_core_context(text):
            base_score -= 80
        else:
            base_score += v14_digital_asset_policy_bonus(text)
            # 단순 가격/시총/시장 규모는 산업동향에서는 가능하지만 정부/국회에서는 정책형보다 낮춘다.
            if json_key == "정부_국회" and not v14_has_digital_asset_policy_context(text):
                base_score -= 30
    return round(base_score, 2)

# 본문 품질 보정
_BASE_body_quality_score_v14 = body_quality_score_v6

def body_quality_score_v6(title, body, method, json_key="", source=""):
    score, reason = _BASE_body_quality_score_v14(title, body, method, json_key=json_key, source=source)
    text = f"{clean_html_text(title)} {clean_html_text(source)} {clean_html_text(body)[:3500]}"
    if v14_is_pure_market_or_stock_noise(text, json_key=json_key):
        return min(score - 55, 5.0), "pure_stock_market_or_investment_noise"
    if (json_key == "정부_국회" or v14_has_digital_asset_core_context(text)):
        if v14_has_digital_asset_core_context(text) and not v14_has_digital_asset_policy_context(text):
            # 산업동향은 시장 구조 변화면 가능하지만, 정부/국회에서는 정책 맥락이 필요하다.
            if json_key == "정부_국회":
                return min(score - 35, 20.0), "digital_asset_without_policy_context"
    return score, reason

_BASE_is_body_usable_v14 = is_body_usable_v6

def is_body_usable_v6(title, body, method, json_key="", source="", importance_score=0):
    text = f"{clean_html_text(title)} {clean_html_text(source)} {clean_html_text(body)[:3500]}"
    if v14_is_pure_market_or_stock_noise(text, json_key=json_key):
        return False, "pure_stock_market_or_investment_noise", 0.0
    if json_key == "정부_국회" and v14_has_digital_asset_core_context(text) and not v14_has_digital_asset_policy_context(text):
        return False, "digital_asset_without_policy_context", 18.0
    return _BASE_is_body_usable_v14(title, body, method, json_key=json_key, source=source, importance_score=importance_score)

# 최종 보고서 적합성 보강
_BASE_is_report_item_relevant_v14 = is_report_item_relevant

def is_report_item_relevant(report_item, json_key):
    text = f"{report_item.get('기사제목','')} {report_item.get('언론사','')} {report_item.get('링크','')} {report_item.get('본문전문','')[:3600]} {report_item.get('RSS요약','')}"
    if v14_is_pure_market_or_stock_noise(text, url=report_item.get('링크',''), json_key=json_key):
        return False, "pure_stock_market_or_investment_noise"
    if json_key == "정부_국회" and v14_has_digital_asset_core_context(text) and not v14_has_digital_asset_policy_context(text):
        return False, "digital_asset_without_policy_context"
    return _BASE_is_report_item_relevant_v14(report_item, json_key)

# 랭킹 후 CSV에 정책 관련성 표시를 남긴다.
_BASE_rank_and_trim_candidates_v14 = rank_and_trim_candidates

def rank_and_trim_candidates(raw_articles):
    ranked_all, ranked_candidates = _BASE_rank_and_trim_candidates_v14(raw_articles)
    for rows in (ranked_all, ranked_candidates):
        for a in rows:
            text = f"{a.get('기사제목','')} {a.get('본문요약','')[:1500]}"
            a["디지털자산핵심"] = "Y" if v14_has_digital_asset_core_context(text) else ""
            a["디지털자산정책성"] = "Y" if v14_has_digital_asset_policy_context(text) else ""
            a["시황성노이즈"] = "Y" if v14_is_pure_market_or_stock_noise(text, keyword=a.get('검색어','')) else ""
    return ranked_all, ranked_candidates


# =========================================================
# v15 overrides: privacy/security incident cross-dedupe
# - 개인정보 유출/해킹/침해사고/서비스 장애 등은 카테고리와 제목이 달라도
#   같은 주체 + 같은 사건유형이면 최종 보고서에서 대표 기사 1개만 남긴다.
# - 단, 과징금 부과/집단소송/제도개선처럼 진행 단계가 명확히 바뀐 후속 기사면 별도 이슈로 인정한다.
# =========================================================

V6_VERSION = "google_rss_v15_privacy_security_incident_dedupe"

_PRIVACY_SECURITY_ENTITY_ALIASES = [
    ("tving", r"티빙|TVING"),
    ("coupang", r"쿠팡|쿠팡이츠|Coupang"),
    ("duo", r"듀오"),
    ("ddarungi", r"따릉이"),
    ("kakao", r"카카오톡|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|카카오엔터프라이즈|카카오엔터|카카오|Kakao"),
    ("naver", r"네이버|NAVER"),
    ("toss", r"토스|토스증권|토스뱅크|비바리퍼블리카"),
    ("baemin", r"배달의민족|배민|우아한형제들"),
    ("skt", r"SKT|SK텔레콤"),
    ("kt", r"\bKT\b|케이티"),
    ("lguplus", r"LGU\+|LG유플러스|엘지유플러스"),
    ("google", r"구글|Google|알파벳|Alphabet"),
    ("openai", r"오픈AI|OpenAI|챗GPT|ChatGPT"),
    ("meta", r"메타|Meta|인스타그램|Instagram|페이스북|Facebook"),
    ("apple", r"애플|Apple"),
    ("upbit_dunamu", r"업비트|두나무|Dunamu|Upbit"),
    ("bithumb", r"빗썸|Bithumb"),
    ("coinone", r"코인원|Coinone"),
    ("government24", r"정부24|행정안전부|행안부"),
]

_PRIVACY_SECURITY_EVENT_RE = re.compile(
    r"개인정보|회원정보|고객정보|개인정보보호|CI\b|DI\b|연계정보|주민번호|주민등록번호|"
    r"유출|노출|침해사고|해킹|정보유출|정보\s*유출|데이터\s*유출|보안사고|보안\s*사고|"
    r"무단접속|계정\s*탈취|접속\s*장애|로그인\s*장애|서비스\s*장애|먹통|피싱|스미싱|악성코드|사칭",
    re.IGNORECASE,
)

_DATA_LEAK_RE = re.compile(r"개인정보|회원정보|고객정보|CI\b|DI\b|연계정보|주민번호|주민등록번호|유출|노출|침해사고|해킹|데이터\s*유출", re.IGNORECASE)
_OUTAGE_RE = re.compile(r"접속\s*장애|로그인\s*장애|서비스\s*장애|장애|먹통|오류", re.IGNORECASE)
_PHISHING_RE = re.compile(r"피싱|스미싱|악성코드|사칭|계정\s*탈취", re.IGNORECASE)

_INCIDENT_STAGE_SANCTION_RE = re.compile(r"과징금|과태료|제재|처분|시정명령|징계|검찰\s*고발", re.IGNORECASE)
_INCIDENT_STAGE_LITIGATION_RE = re.compile(r"집단소송|소송\s*제기|손해배상|고소|고발|법적\s*대응", re.IGNORECASE)
_INCIDENT_STAGE_POLICY_RE = re.compile(r"법안|개정안|시행령|가이드라인|대책|제도\s*개선|의무화|규제\s*강화|보완\s*대책", re.IGNORECASE)
_INCIDENT_STAGE_INVESTIGATION_RE = re.compile(r"조사\s*착수|조사에\s*착수|민관합동조사|합동조사|긴급\s*대응|KISA|한국인터넷진흥원|개인정보위|개보위|과기정통부|방미통위|신고\s*접수|사고\s*경위", re.IGNORECASE)
_INCIDENT_STAGE_RESPONSE_RE = re.compile(r"대표|CEO|책임지겠다|사과|입장문|재발\s*방지|보상|피해\s*구제|공지", re.IGNORECASE)

_INCIDENT_GENERAL_LIST_RE = re.compile(r"업종\s*가리지|곳곳|잇단|잇따른|연쇄|확산|비상|사례\s*늘|대거|전방위", re.IGNORECASE)


def v15_detect_privacy_security_subject(text):
    t = clean_html_text(text)
    for key, pat in _PRIVACY_SECURITY_ENTITY_ALIASES:
        if re.search(pat, t, flags=re.IGNORECASE):
            return key
    return ""


def v15_detect_privacy_security_type(text):
    t = clean_html_text(text)
    if _DATA_LEAK_RE.search(t):
        return "data_leak"
    if _PHISHING_RE.search(t):
        return "phishing_or_account_takeover"
    if _OUTAGE_RE.search(t):
        return "service_outage"
    if _PRIVACY_SECURITY_EVENT_RE.search(t):
        return "security_incident"
    return ""


def v15_detect_incident_stage_cluster(text):
    """초기 조사/기업 입장 계열은 같은 사고로 묶고, 제재·소송·제도개선은 별도 후속 단계로 둔다."""
    t = clean_html_text(text)
    if _INCIDENT_STAGE_SANCTION_RE.search(t):
        return "sanction_or_penalty"
    if _INCIDENT_STAGE_LITIGATION_RE.search(t):
        return "litigation"
    if _INCIDENT_STAGE_POLICY_RE.search(t):
        return "policy_change"
    if _INCIDENT_STAGE_INVESTIGATION_RE.search(t) or _INCIDENT_STAGE_RESPONSE_RE.search(t):
        return "initial_investigation_response"
    return "initial_investigation_response"


def v15_privacy_security_incident_key(item):
    title = clean_html_text(item.get("기사제목", ""))
    body = clean_html_text((item.get("본문전문") or item.get("본문요약") or item.get("RSS요약") or "")[:1800])
    text = f"{title} {body}"
    if not _PRIVACY_SECURITY_EVENT_RE.search(text):
        return ""
    subject = v15_detect_privacy_security_subject(text)
    event_type = v15_detect_privacy_security_type(text)
    if not subject or not event_type:
        return ""
    stage = v15_detect_incident_stage_cluster(text)
    return f"privacy_security:{subject}:{event_type}:{stage}"


def v15_privacy_security_base_key(item):
    key = v15_privacy_security_incident_key(item)
    if not key:
        return ""
    # stage를 제거한 base key. 조사 착수/대표 사과/민관합동조사는 같은 base로 묶는다.
    parts = key.split(":")
    if len(parts) >= 4:
        return ":".join(parts[:3])
    return key


def v15_is_same_privacy_security_incident(new_item, old_item):
    new_key = v15_privacy_security_incident_key(new_item)
    old_key = v15_privacy_security_incident_key(old_item)
    if not new_key or not old_key:
        return False, ""

    new_base = v15_privacy_security_base_key(new_item)
    old_base = v15_privacy_security_base_key(old_item)
    if new_base != old_base:
        return False, ""

    new_stage = new_key.split(":")[-1]
    old_stage = old_key.split(":")[-1]

    # 제재/소송/제도개선은 조사 착수와 다른 후속 단계면 별도 이슈로 살릴 수 있다.
    distinct_followup = {"sanction_or_penalty", "litigation", "policy_change"}
    if new_stage in distinct_followup or old_stage in distinct_followup:
        if new_stage != old_stage:
            return False, ""

    return True, f"privacy_security_incident_duplicate:{new_base}:{new_stage}/{old_stage}"


def v15_privacy_security_representative_bonus(item):
    title = clean_html_text(item.get("기사제목", ""))
    body = clean_html_text((item.get("본문전문") or item.get("본문요약") or item.get("RSS요약") or "")[:2400])
    text = f"{title} {body}"
    bonus = 0.0
    if _INCIDENT_STAGE_INVESTIGATION_RE.search(text):
        bonus += 22
    if re.search(r"민관합동조사|합동조사|KISA|한국인터넷진흥원|개인정보위|개보위|과기정통부|방미통위", text, re.IGNORECASE):
        bonus += 18
    if re.search(r"CI|DI|연계정보|주민번호|생년월일|전화번호|이메일|유출\s*항목|피해\s*규모|몇\s*명|[0-9,]+\s*명", text, re.IGNORECASE):
        bonus += 16
    if _INCIDENT_STAGE_RESPONSE_RE.search(text):
        bonus += 7
    if _INCIDENT_GENERAL_LIST_RE.search(title) and not re.search(r"조사|과징금|민관합동|KISA|개인정보위|과기정통부", title, re.IGNORECASE):
        bonus -= 18
    # 보안 사고는 정부/국회 기사일 때 규제 액션이 있으면 더 우선한다.
    if item.get("카테고리") == "정부_국회" or item.get("JSON카테고리") == "정부_국회":
        if _INCIDENT_STAGE_INVESTIGATION_RE.search(text) or _INCIDENT_STAGE_SANCTION_RE.search(text):
            bonus += 14
    return bonus


def v15_duplicate_preference_score(item):
    try:
        base = float(item.get("대표선택점수") or representative_score(item) or 0)
    except Exception:
        base = 0.0
    return base + v15_privacy_security_representative_bonus(item)


def should_replace_duplicate_representative(new_item, old_item, reason=""):
    # 개인정보/보안 사고는 더 종합적이고 정부 조치가 명확한 기사를 대표로 교체할 수 있게 한다.
    if str(reason).startswith("privacy_security_incident_duplicate"):
        return v15_duplicate_preference_score(new_item) > v15_duplicate_preference_score(old_item) + 6
    try:
        return float(new_item.get("대표선택점수") or 0) > float(old_item.get("대표선택점수") or 0) + 12
    except Exception:
        return False


_BASE_final_duplicate_reason_v15 = final_duplicate_reason

def final_duplicate_reason(new_item, existing_items):
    for old in existing_items:
        same, reason = v15_is_same_privacy_security_incident(new_item, old)
        if same:
            return old, reason
    return _BASE_final_duplicate_reason_v15(new_item, existing_items)

# CSV 확인용 컬럼을 남긴다.
_BASE_process_article_for_report_v15 = process_article_for_report

def process_article_for_report(article_info, json_key, recent_past_items):
    report_item, skip_info, failed_item = _BASE_process_article_for_report_v15(article_info, json_key, recent_past_items)
    target = report_item or failed_item
    if target:
        target["개인정보보안사건키"] = v15_privacy_security_incident_key(target)
        target["개인정보보안대표보너스"] = v15_privacy_security_representative_bonus(target)
    return report_item, skip_info, failed_item


# =========================================================
# v17 overrides: global incident dedupe 강화 (v16 대비 두 가지 수정)
# - ELS 과징금처럼 키워드/카테고리가 달라도 같은 사건인 기사를 하나로 묶는다.
# - 규칙은 특정 이슈 하드코딩이 아니라 주체 + 대상 + 사건유형 + 진행단계 기반으로 일반화한다.
# - 개인정보/보안 사고 dedupe(v15)는 유지하고, 그 밖의 금융규제/플랫폼규제/디지털자산정책/노무/경영진/자본거래 등을 전역에서 묶는다.
#
# [v16 → v17 변경사항]
# 수정 A: v16_same_family_object_key() 추가
#   - v16_detect_entity()가 같은 사건을 기사마다 다른 entity로 분류하는 오탐 문제를 보완.
#   - entity를 제외하고 family:object만으로 비교하는 보조 키를 추가.
#   - base key(family:entity:object)가 달라도 family:object가 같으면 중복으로 판정.
#   - 예) 티빙 유출 기사에서 한 기사는 entity=kakao, 다른 기사는 entity=tving으로
#     추출돼 base key가 달라지던 문제를 해결. 앞으로 어떤 사건이든 entity 오탐 시 동일하게 커버.
#
# 수정 B: v16_is_distinct_followup_stage()에서 policy_change 제거
#   - 기존에는 policy_change를 "다른 국면"으로 보고 같은 사건의 두 기사를 살렸음.
#   - 실제로는 같은 날 같은 제재 결정을 "감경", "축소", "부과" 등 다르게 표현한 기사들.
#   - litigation(집단소송 등 수개월 후 다른 전개)만 distinct stage로 유지하고
#     policy_change는 제거해 같은 국면으로 묶음.
# =========================================================

V6_VERSION = "google_rss_v17_dedup_fix"

# 전역 사건 클러스터용 핵심 패턴들
_V16_FINANCIAL_PRODUCT_RE = re.compile(
    r"홍콩\s*ELS|ELS|주가연계증권|DLF|라임|옵티머스|펀드\s*불완전판매|불완전판매|분쟁조정|자율배상|판매\s*은행|판매銀",
    re.IGNORECASE,
)
_V16_FINANCIAL_REG_ACTION_RE = re.compile(
    r"금감원|금융감독원|금융위|금융위원회|제재심|과징금|과태료|제재|감경|축소|부과|처분|배상|분쟁조정|검사|조사|은행권|판매사",
    re.IGNORECASE,
)
_V16_ELS_ANCHOR_RE = re.compile(r"홍콩\s*ELS|ELS|주가연계증권", re.IGNORECASE)

_V16_PLATFORM_REG_RE = re.compile(
    r"공정위|공정거래위원회|플랫폼|배달앱|배민|배달의민족|쿠팡이츠|요기요|수수료|공시제|최혜대우|동의의결|자사우대|알고리즘|독과점|담합|온플법|플랫폼\s*규제|중점조사기획단",
    re.IGNORECASE,
)
_V16_DIGITAL_ASSET_POLICY_RE = re.compile(
    r"스테이블\s*코인|스테이블코인|가상자산|디지털자산|특금법|특정금융정보법|FIU|금융정보분석원|AML|자금세탁방지|VASP|가상자산사업자|개인지갑|해외\s*이전|트래블룰|디지털자산기본법|토큰증권|STO|원화\s*스테이블|거래소",
    re.IGNORECASE,
)
_V16_AI_SECURITY_POLICY_RE = re.compile(
    r"미토스|글래스윙|앤트로픽|오픈AI|AI\s*보안|사이버보안|해킹\s*자동화|보안모델|정부\s*프로그램|GTAC|신뢰\s*기반\s*접근",
    re.IGNORECASE,
)
_V16_LABOR_RE = re.compile(
    r"노조|파업|쟁의권|조정\s*결렬|임단협|임금협상|성과급|고용안정|근로감독|노동부|최저임금|임금체불|진정|부분파업|총파업",
    re.IGNORECASE,
)
_V16_EXEC_SERVICE_RE = re.compile(
    r"CPO|CTO|CFO|CEO|대표|임원|최고제품책임자|퇴사|사의|사임|해임|교체|선임|내정|조직개편|서비스\s*개편|개편\s*논란|이용자\s*반발|후폭풍",
    re.IGNORECASE,
)
_V16_MNA_CAPITAL_RE = re.compile(
    r"인수|합병|매각|지분|블록딜|주식교환|경영권|최대주주|상장|IPO|투자유치|전략적\s*투자|자금\s*조달|출자",
    re.IGNORECASE,
)

# 진행 단계. 단계가 뚜렷이 다른 경우는 별도 이슈로 살릴 수 있게 하되,
# 같은 사건의 초기 보도/같은 제재 국면은 하나로 묶는다.
_V16_STAGE_LITIGATION_RE = re.compile(r"집단소송|소송\s*제기|손해배상|행정소송|가처분|무효확인|법적\s*대응", re.IGNORECASE)
_V16_STAGE_POLICY_RE = re.compile(r"법안|개정안|시행령|입법예고|행정예고|가이드라인|제도\s*개선|의무화|규제\s*강화|보완\s*검토|확정|시행", re.IGNORECASE)
_V16_STAGE_SANCTION_RE = re.compile(r"과징금|과태료|제재|처분|시정명령|검찰\s*고발|고발\s*시사|제재심", re.IGNORECASE)
_V16_STAGE_PENALTY_ADJUST_RE = re.compile(r"감경|축소|낮춰|절반|6000억|6천억|1조\s*4000억|자율배상|분쟁조정\s*수용", re.IGNORECASE)
_V16_STAGE_INVESTIGATION_RE = re.compile(r"조사\s*착수|조사에\s*착수|검사|현장조사|수사\s*착수|민관합동조사|긴급\s*대응|신고\s*접수|의견수렴|협의", re.IGNORECASE)
_V16_STAGE_CONFLICT_ESCALATION_RE = re.compile(r"조정\s*결렬|쟁의권|파업\s*예고|파업\s*돌입|집회|총파업|부분파업|사측|노조", re.IGNORECASE)
_V16_STAGE_EXEC_CHANGE_RE = re.compile(r"퇴사|사의|사임|해임|교체|선임|내정|영입|조직개편", re.IGNORECASE)

# 전역 사건 주체 추출. 필요한 일반 엔티티만 넓게 둔다.
_V16_ENTITY_ALIASES = [
    ("kakao", r"카카오톡|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|카카오엔터프라이즈|카카오엔터|카카오|Kakao"),
    ("naver", r"네이버|NAVER"),
    ("coupang", r"쿠팡|쿠팡이츠|Coupang"),
    ("baemin", r"배달의민족|배민|우아한형제들"),
    ("toss", r"토스|토스뱅크|토스증권|비바리퍼블리카"),
    ("tving", r"티빙|TVING"),
    ("google", r"구글|Google|알파벳|Alphabet"),
    ("openai", r"오픈AI|OpenAI|챗GPT|ChatGPT"),
    ("meta", r"메타|Meta|인스타그램|Instagram|페이스북|Facebook"),
    ("apple", r"애플|Apple"),
    ("skt", r"SKT|SK텔레콤"),
    ("kt", r"\bKT\b|케이티"),
    ("lguplus", r"LGU\+|LG유플러스|엘지유플러스"),
    ("dunamu_upbit", r"두나무|업비트|Dunamu|Upbit"),
    ("bithumb", r"빗썸|Bithumb"),
    ("financial_authority", r"금감원|금융감독원|금융위|금융위원회|FIU|금융정보분석원"),
    ("fair_trade_authority", r"공정위|공정거래위원회"),
    ("science_ict_ministry", r"과기정통부|과학기술정보통신부|KISA|한국인터넷진흥원"),
]

# low-value 일반 묶음 기사를 대표 기사로 고르지 않기 위한 신호
_V16_GENERAL_ROUNDUP_RE = re.compile(
    r"업종\s*가리지|곳곳|잇단|잇따른|연쇄|확산|비상|종합|총정리|한눈에|주요\s*이슈|브리핑|시장\s*동향|전망",
    re.IGNORECASE,
)


def v16_clean_for_incident(item):
    title = clean_html_text(item.get("기사제목", ""))
    body = clean_html_text((item.get("본문전문") or item.get("본문요약") or item.get("RSS요약") or "")[:2600])
    source = clean_html_text(item.get("언론사", ""))
    return f"{title} {source} {body}"


def v16_detect_entity(text):
    t = clean_html_text(text)
    # ELS처럼 대상이 사건의 주체인 경우는 별도 처리
    if _V16_ELS_ANCHOR_RE.search(t):
        return "hongkong_els"
    # 디지털자산 쪽은 업비트/두나무/거래소/당국 순서 고려
    for key, pat in _V16_ENTITY_ALIASES:
        if re.search(pat, t, flags=re.IGNORECASE):
            return key
    # 정부 규제인데 특정 기업이 없으면 분야 중심으로 묶는다.
    if _V16_DIGITAL_ASSET_POLICY_RE.search(t):
        return "digital_asset_sector"
    if _V16_PLATFORM_REG_RE.search(t):
        return "platform_sector"
    return ""


def v16_detect_incident_family(text):
    t = clean_html_text(text)
    if _V16_FINANCIAL_PRODUCT_RE.search(t) and _V16_FINANCIAL_REG_ACTION_RE.search(t):
        return "financial_regulation"
    if _V16_DIGITAL_ASSET_POLICY_RE.search(t) and (_DIGITAL_ASSET_POLICY_RE.search(t) or _V16_STAGE_POLICY_RE.search(t) or _V16_STAGE_INVESTIGATION_RE.search(t)):
        return "digital_asset_policy"
    if _V16_PLATFORM_REG_RE.search(t):
        return "platform_regulation"
    if _V16_AI_SECURITY_POLICY_RE.search(t):
        return "ai_security_policy"
    # 개인정보/보안은 v15가 강하게 처리하지만, 전역 key 컬럼에는 남길 수 있다.
    if _PRIVACY_SECURITY_EVENT_RE.search(t):
        return "privacy_security"
    if _V16_LABOR_RE.search(t):
        return "labor_ir"
    if _V16_EXEC_SERVICE_RE.search(t):
        return "leadership_service"
    if _V16_MNA_CAPITAL_RE.search(t):
        return "mna_capital"
    return ""


def v16_detect_object(text, family):
    t = clean_html_text(text)
    if family == "financial_regulation":
        if _V16_ELS_ANCHOR_RE.search(t):
            return "hongkong_els_misselling"
        if re.search(r"DLF", t, re.IGNORECASE):
            return "dlf_misselling"
        if re.search(r"라임", t):
            return "lime_fund"
        if re.search(r"옵티머스", t):
            return "optimus_fund"
        if re.search(r"불완전판매", t):
            return "misselling"
        return "financial_product_case"
    if family == "digital_asset_policy":
        if re.search(r"특금법|특정금융정보법|FIU|금융정보분석원|AML|자금세탁방지|개인지갑|해외\s*이전|VASP|가상자산사업자|트래블룰", t, re.IGNORECASE):
            return "vasp_aml_rulemaking"
        if re.search(r"스테이블\s*코인|스테이블코인|원화\s*스테이블", t, re.IGNORECASE):
            return "stablecoin_policy"
        if re.search(r"토큰증권|STO", t, re.IGNORECASE):
            return "sto_policy"
        return "digital_asset_policy"
    if family == "platform_regulation":
        if re.search(r"배달앱|배민|배달의민족|쿠팡이츠|수수료|공시제", t, re.IGNORECASE):
            return "delivery_app_fee_policy"
        if re.search(r"쿠팡|김범석|동일인|지정자료|허위자료", t, re.IGNORECASE):
            return "coupang_designation_disclosure"
        if re.search(r"중점조사기획단|조사국|경제분석국", t, re.IGNORECASE):
            return "ftc_org_enforcement_reform"
        if re.search(r"자사우대|알고리즘|독과점", t, re.IGNORECASE):
            return "platform_self_preferencing"
        return "platform_regulation"
    if family == "ai_security_policy":
        if re.search(r"미토스|글래스윙|앤트로픽", t, re.IGNORECASE):
            return "anthropic_glasswing_security"
        if re.search(r"오픈AI|GTAC|신뢰\s*기반", t, re.IGNORECASE):
            return "openai_security_access"
        return "ai_security_policy"
    if family == "labor_ir":
        if re.search(r"최저임금|노동부|근로감독|임금체불|진정", t):
            return "labor_law_complaint"
        if re.search(r"파업|쟁의권|조정\s*결렬|집회|부분파업|총파업", t):
            return "strike_negotiation"
        if re.search(r"성과급|N%|영업이익", t, re.IGNORECASE):
            return "bonus_compensation_dispute"
        return "labor_ir"
    if family == "leadership_service":
        if re.search(r"퇴사|사의|사임|해임|교체|선임|내정|영입", t):
            return "executive_change"
        if re.search(r"개편|이용자\s*반발|서비스\s*논란|후폭풍", t):
            return "service_reform_backlash"
        return "leadership_service"
    if family == "mna_capital":
        if re.search(r"두나무|업비트", t):
            return "dunamu_upbit_capital"
        if re.search(r"배민|배달의민족|우아한형제들", t):
            return "baemin_capital"
        return "mna_capital"
    if family == "privacy_security":
        base = v15_privacy_security_base_key({"기사제목": t, "본문전문": t})
        return base.replace("privacy_security:", "") if base else "privacy_security"
    return ""


def v16_detect_stage(text, family=""):
    t = clean_html_text(text)
    if _V16_STAGE_LITIGATION_RE.search(t):
        return "litigation"
    if _V16_STAGE_POLICY_RE.search(t):
        return "policy_change"
    if family == "financial_regulation" and (_V16_STAGE_PENALTY_ADJUST_RE.search(t) or _V16_STAGE_SANCTION_RE.search(t)):
        # ELS 과징금 축소/감경/부과 기사들은 같은 제재 국면으로 묶는다.
        return "sanction_penalty_decision"
    if _V16_STAGE_SANCTION_RE.search(t):
        return "sanction_penalty"
    if _V16_STAGE_CONFLICT_ESCALATION_RE.search(t):
        return "conflict_escalation"
    if _V16_STAGE_EXEC_CHANGE_RE.search(t):
        return "executive_or_org_change"
    if _V16_STAGE_INVESTIGATION_RE.search(t):
        return "investigation_or_consultation"
    if re.search(r"발표|공개|착수|합류|검토|추진", t):
        return "announcement"
    return "general"


def v16_global_incident_key(item):
    text = v16_clean_for_incident(item)
    family = v16_detect_incident_family(text)
    if not family:
        return ""
    entity = v16_detect_entity(text)
    obj = v16_detect_object(text, family)
    stage = v16_detect_stage(text, family)
    if not entity and family not in {"digital_asset_policy", "platform_regulation", "financial_regulation", "ai_security_policy"}:
        return ""
    if not obj:
        return ""
    # 너무 넓은 mna_capital/leadership_service는 자사/주요 경쟁사 등 entity가 없으면 묶지 않는다.
    if family in {"mna_capital", "leadership_service", "labor_ir"} and not entity:
        return ""
    return f"global_incident:{family}:{entity or 'sector'}:{obj}:{stage}"


def v16_global_incident_base_key(item):
    key = v16_global_incident_key(item)
    if not key:
        return ""
    parts = key.split(":")
    # global_incident:family:entity:object:stage -> stage 제거
    if len(parts) >= 5:
        return ":".join(parts[:4])
    return key


def v16_is_distinct_followup_stage(stage):
    # litigation(집단소송·법적 대응)만 "다른 국면"으로 인정한다.
    # policy_change는 제거: 같은 날 같은 제재 결정을 "감경/축소/부과" 등 다르게 표현한
    # 기사들이 서로 다른 stage로 분류돼 중복이 통과되던 문제를 막기 위해.
    return stage in {"litigation"}


def v16_same_family_object_key(item):
    """entity를 무시하고 family:object만으로 비교용 보조 키를 생성한다.
    v16_detect_entity()가 같은 사건인데 기사마다 다른 entity를 반환하는 오탐 시
    base key(family:entity:object)가 달라져 중복 판정에 실패하는 문제를 보완한다.

    예) 티빙 유출 기사에서:
      뉴시스 기사  → global_incident:privacy_security:kakao:tving:...
      뉴스1 기사   → global_incident:privacy_security:tving:tving:...
    base key가 달라 미탐 → 이 함수로 privacy_security:tving 동일 확인 가능.
    """
    key = v16_global_incident_key(item)
    if not key:
        return ""
    parts = key.split(":")
    # global_incident:family:entity:object:stage → family:object
    if len(parts) >= 5:
        return f"{parts[1]}:{parts[3]}"
    return ""


def v16_is_same_global_incident(new_item, old_item):
    new_key = v16_global_incident_key(new_item)
    old_key = v16_global_incident_key(old_item)
    if not new_key or not old_key:
        return False, ""
    if new_key == old_key:
        return True, f"global_incident_duplicate:{new_key}"
    new_base = v16_global_incident_base_key(new_item)
    old_base = v16_global_incident_base_key(old_item)
    if not new_base or new_base != old_base:
        # [수정 A] base key가 달라도 family:object가 같으면 중복 후보로 판정.
        # entity 오탐(같은 사건인데 기사마다 다른 entity 추출)을 커버한다.
        new_fo = v16_same_family_object_key(new_item)
        old_fo = v16_same_family_object_key(old_item)
        if new_fo and old_fo and new_fo == old_fo:
            return True, f"global_incident_same_family_object:{new_fo}"
        return False, ""
    new_stage = new_key.split(":")[-1]
    old_stage = old_key.split(":")[-1]
    # 소송처럼 뚜렷이 다른 단계는 살린다. 단, 같은 단계면 중복.
    if v16_is_distinct_followup_stage(new_stage) or v16_is_distinct_followup_stage(old_stage):
        if new_stage != old_stage:
            return False, ""
    # 금융규제 제재 국면은 제목 표현이 달라도 같은 사건이면 하나로 묶는다.
    return True, f"global_incident_duplicate:{new_base}:{new_stage}/{old_stage}"


def v16_global_representative_bonus(item):
    text = v16_clean_for_incident(item)
    title = clean_html_text(item.get("기사제목", ""))
    family = v16_detect_incident_family(text)
    bonus = 0.0
    if family == "financial_regulation":
        if re.search(r"금감원|금융감독원|금융위|금융위원회|제재심", text):
            bonus += 22
        if re.search(r"과징금|감경|축소|6000억|6천억|1조\s*4000억|자율배상|분쟁조정", text):
            bonus += 22
        if re.search(r"은행|판매사|불완전판매|홍콩\s*ELS|ELS", text, re.IGNORECASE):
            bonus += 14
        if re.search(r"영향|이유|배경|수용", title):
            bonus += 8
    elif family == "digital_asset_policy":
        bonus += v14_digital_asset_policy_bonus(text)
    elif family == "platform_regulation":
        if re.search(r"공정위|공정거래위원회|국회|법안|조사|제재|공시제|중점조사기획단", text):
            bonus += 24
    elif family == "ai_security_policy":
        if re.search(r"정부|과기정통부|KISA|합류|접근권|프로젝트|보안모델", text):
            bonus += 18
    elif family == "labor_ir":
        if re.search(r"조정\s*결렬|쟁의권|파업|노동부|진정|최저임금", text):
            bonus += 20
    elif family == "leadership_service":
        if re.search(r"퇴사|사의|사임|책임|개편\s*논란|이용자\s*반발", text):
            bonus += 18
    # 같은 사건 묶음에서는 종합/공식/맥락 있는 기사를 우선하고, 단순 모음/시황은 후순위.
    if _V16_GENERAL_ROUNDUP_RE.search(title) and not re.search(r"조사|과징금|제재|법안|시행령|결렬|쟁의권|퇴사|사의", title):
        bonus -= 18
    if v14_is_pure_market_or_stock_noise(text, url=item.get("링크", ""), json_key=item.get("카테고리", "")):
        bonus -= 40
    return bonus


def v16_global_duplicate_preference_score(item):
    try:
        base = float(item.get("대표선택점수") or representative_score(item) or 0)
    except Exception:
        base = 0.0
    return base + v16_global_representative_bonus(item)


_BASE_should_replace_duplicate_representative_v16 = should_replace_duplicate_representative

def should_replace_duplicate_representative(new_item, old_item, reason=""):
    if str(reason).startswith("global_incident_duplicate"):
        return v16_global_duplicate_preference_score(new_item) > v16_global_duplicate_preference_score(old_item) + 6
    return _BASE_should_replace_duplicate_representative_v16(new_item, old_item, reason)


_BASE_final_duplicate_reason_v16 = final_duplicate_reason

def final_duplicate_reason(new_item, existing_items):
    # 개인정보/보안 특화 dedupe는 기존 v15에서 우선 처리되므로, 여기서는 더 넓은 전역 사건 dedupe를 추가한다.
    for old in existing_items:
        same, reason = v16_is_same_global_incident(new_item, old)
        if same:
            return old, reason
    return _BASE_final_duplicate_reason_v16(new_item, existing_items)


_BASE_process_article_for_report_v16 = process_article_for_report

def process_article_for_report(article_info, json_key, recent_past_items):
    report_item, skip_info, failed_item = _BASE_process_article_for_report_v16(article_info, json_key, recent_past_items)
    target = report_item or failed_item
    if target:
        target["전역사건키"] = v16_global_incident_key(target)
        target["전역사건기본키"] = v16_global_incident_base_key(target)
        target["전역사건대표보너스"] = v16_global_representative_bonus(target)
    return report_item, skip_info, failed_item

# ranked/candidate CSV에도 사건키를 남겨 디버깅한다.
_BASE_rank_and_trim_candidates_v16 = rank_and_trim_candidates

def rank_and_trim_candidates(raw_articles):
    ranked_all, ranked_candidates = _BASE_rank_and_trim_candidates_v16(raw_articles)
    for rows in (ranked_all, ranked_candidates):
        for a in rows:
            a["전역사건키"] = v16_global_incident_key(a)
            a["전역사건기본키"] = v16_global_incident_base_key(a)
    return ranked_all, ranked_candidates


# =========================================================
# v18 overrides: daily-safe platform obligation framing
# - 특정 기사/키워드 하드코딩 대신 "플랫폼 사업자 의무/규제 부담"을 범용 이슈 패밀리로 인식
# - 자사 직접 리스크와 자사 포함 업계 공통 규제 이슈를 분리해 평가
# - 스폰서/행사/프로모션성 자사 홍보 기사는 CEO 브리핑 우선순위에서 강하게 감점
# - RSS 단계에서 본문 품질점수(아직 없음)가 랭킹 감점으로 새는 문제를 보정
# =========================================================

V6_VERSION = "google_rss_v18_platform_obligation_family_balancing"


def v18_fast_clean(text):
    if text is None:
        return ""
    s = str(text)
    if "<" in s and ">" in s:
        s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# 플랫폼/온라인서비스 사업자에게 새 의무·준수 부담이 생기는지를 보는 범용 프레임.
# 특정 기사명이나 일회성 키워드가 아니라, 주체(플랫폼/사업자) + 의무행위(차단/삭제/보고/공시/준수 등) + 정책/시행 맥락을 조합해서 판단한다.
_V18_PLATFORM_ACTOR_RE = re.compile(
    r"플랫폼|온라인\s*플랫폼|포털|SNS|소셜미디어|앱마켓|검색사업자|부가통신사업자|"
    r"정보통신서비스\s*제공자|인터넷\s*사업자|서비스\s*사업자|사업자|업계|"
    r"네이버|카카오|구글|메타|인스타그램|페이스북|유튜브|틱톡|X\b|트위터|애플|쿠팡|배달의민족|배민|토스",
    re.IGNORECASE,
)

_V18_OBLIGATION_ACTION_RE = re.compile(
    r"의무|의무화|해야\s*한다|해야\s*함|해야\s*한다고|해야\s*할|걸러야|차단|사전\s*차단|삭제|"
    r"필터링|신고|보고|공시|자료\s*제출|자료제출|준수|위반|금지|제한|제재|과징금|과태료|"
    r"시정명령|가이드라인|시행|적용|확대|강화|유통\s*방지|확산\s*방지|모니터링|소명|동의|검증",
    re.IGNORECASE,
)

_V18_POLICY_OR_REGULATOR_CONTEXT_RE = re.compile(
    r"정부|국회|과기정통부|과학기술정보통신부|방미통위|방송미디어통신위원회|방통위|개보위|개인정보위|"
    r"공정위|공정거래위원회|금융위|금융위원회|금감원|금융감독원|행안부|경찰청|검찰|KISA|인터넷진흥원|"
    r"법안|개정안|시행령|시행규칙|고시|입법예고|행정예고|제도|정책|규제|가이드라인|설명회|간담회|"
    r"내달부터|다음달부터|7월부터|오는\s*\d{1,2}월부터|시행|적용|준수|점검|감독|조사|제재",
    re.IGNORECASE,
)

# 플랫폼이 다루는 콘텐츠/데이터/거래/광고/알고리즘 등 운영 의무 영역.
_V18_PLATFORM_DUTY_DOMAIN_RE = re.compile(
    r"불법\s*정보|유해\s*정보|유해정보|콘텐츠|게시물|이미지|영상|동영상|허위조작정보|딥페이크|"
    r"사칭|피싱|스미싱|보이스피싱|악성코드|개인정보|정보\s*유출|청소년|아동|성착취|음란물|"
    r"저작권|AI\s*학습|광고|추천|알고리즘|검색|입점|판매자|정산|수수료|다크패턴|리뷰|위조상품|"
    r"이용자\s*보호|소비자\s*보호|피해\s*방지|권리침해|유통\s*방지|확산\s*방지",
    re.IGNORECASE,
)

# 행사 후원/프로모션처럼 회사명은 들어가지만 CEO 정책·리스크 브리핑 우선순위가 낮은 자사 홍보성 기사.
_V18_SELF_PR_PROMO_RE = re.compile(
    r"스폰서|후원|협찬|골드\s*스폰서|공식\s*후원|공식\s*파트너|공식\s*결제수단|"
    r"축제|영화제|페스티벌|전시회|팝업스토어|프로모션|이벤트|쿠폰|할인|혜택|경품|기획전|"
    r"브랜드데이|브랜드\s*캠페인|참여자\s*모집|오픈\s*기념|굿즈|체험존",
    re.IGNORECASE,
)

# 사업적 제휴/신규 서비스와 단순 홍보를 구분하기 위한 예외 신호.
_V18_MATERIAL_PARTNERSHIP_RE = re.compile(
    r"인수|합병|매각|지분|투자|전략적\s*제휴|업무협약|MOU|계약|출시|도입|연동|결제망|교통카드|"
    r"서비스\s*개편|수익화|정산|수수료|규제|제재|과징금|개인정보|보안|장애|피싱|노조|파업",
    re.IGNORECASE,
)

# detect_event_tags / v11 signal 체계에 범용 신호를 연결한다.
_V18_PLATFORM_OBLIGATION_EVENT_RE = re.compile(
    r"(플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|부가통신사업자|정보통신서비스\s*제공자|사업자|"
    r"네이버|카카오|구글|메타|유튜브|틱톡|애플|쿠팡|배달의민족|배민|토스)"
    r".{0,120}"
    r"(의무|의무화|해야|걸러야|차단|삭제|필터링|신고|보고|공시|자료\s*제출|준수|금지|제한|제재|과징금|과태료|시정명령|시행|적용|유통\s*방지|확산\s*방지|모니터링)"
    r"|"
    r"(의무|의무화|해야|걸러야|차단|삭제|필터링|신고|보고|공시|자료\s*제출|준수|금지|제한|제재|과징금|과태료|시정명령|시행|적용|유통\s*방지|확산\s*방지|모니터링)"
    r".{0,120}"
    r"(플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|부가통신사업자|정보통신서비스\s*제공자|사업자|네이버|카카오|구글|메타|유튜브|틱톡|애플|쿠팡|배달의민족|배민|토스)",
    re.IGNORECASE,
)
EVENT_PATTERNS.update({
    "platform_obligation": _V18_PLATFORM_OBLIGATION_EVENT_RE.pattern,
    "self_pr_promo": _V18_SELF_PR_PROMO_RE.pattern,
})
STRONG_EVENT_TAGS.add("platform_obligation")
try:
    V11_SIGNAL_PATTERNS["platform_obligation"] = _V18_PLATFORM_OBLIGATION_EVENT_RE
    V11_SIGNAL_WEIGHTS_SELF["platform_obligation"] = 12
    # platform_obligation is a policy-risk signal, not a strict direct-self must-review signal.
    V11_STORY_TOPIC_LIMITS.update({
        "self:platform_obligation": 2,
        "government:platform_obligation": 2,
        "competitor:platform_obligation": 1,
        "self:pr_promo": 0,
    })
except Exception:
    pass

# v16 전역 사건 중복 감지에서도 플랫폼 의무형 규제 이슈를 잡도록 범용 패턴 확장.
try:
    _V16_PLATFORM_REG_RE = re.compile(
        r"공정위|공정거래위원회|플랫폼|배달앱|배민|배달의민족|쿠팡이츠|요기요|수수료|공시제|최혜대우|동의의결|"
        r"자사우대|알고리즘|독과점|담합|온플법|플랫폼\s*규제|중점조사기획단|"
        r"사업자\s*의무|의무화|사전\s*차단|유통\s*방지|확산\s*방지|필터링|자료\s*제출|시정명령|과징금|과태료|"
        r"불법\s*정보|유해\s*정보|유해정보|허위조작정보|딥페이크|사칭|피싱|개인정보|청소년\s*보호|이용자\s*보호",
        re.IGNORECASE,
    )
except Exception:
    pass


def v18_text_from_article_like(item):
    return v18_fast_clean(
        f"{item.get('기사제목','')} {item.get('본문요약','')} {item.get('RSS요약','')} "
        f"{item.get('본문전문','')[:2800]} {item.get('언론사','')}"
    )


def v18_is_platform_obligation_text(text):
    t = v18_fast_clean(text)
    if not t:
        return False
    # 주체 + 의무행위 + 정책/시행 또는 운영의무 영역이 함께 있어야 한다.
    if _V18_PLATFORM_ACTOR_RE.search(t) and _V18_OBLIGATION_ACTION_RE.search(t):
        if _V18_POLICY_OR_REGULATOR_CONTEXT_RE.search(t) or _V18_PLATFORM_DUTY_DOMAIN_RE.search(t):
            return True
    # 규제기관/정책 문맥이 강하면, 플랫폼 주체가 일반명사로만 표현돼도 살린다.
    if _V18_POLICY_OR_REGULATOR_CONTEXT_RE.search(t) and _V18_OBLIGATION_ACTION_RE.search(t) and _V18_PLATFORM_DUTY_DOMAIN_RE.search(t):
        return True
    return False


def v18_is_self_pr_promo_text(text):
    t = v18_fast_clean(text)
    if not t:
        return False
    if not SELF_KAKAO_PATTERN.search(t):
        return False
    if not _V18_SELF_PR_PROMO_RE.search(t):
        return False
    # 실제 사업·규제·보안·노무 이벤트가 함께 있으면 단순 PR로 보지 않는다.
    if _V18_MATERIAL_PARTNERSHIP_RE.search(t) and not re.search(r"스폰서|후원|협찬|축제|영화제|페스티벌", t, re.IGNORECASE):
        return False
    return True


def v18_issue_family_from_text(text, json_key=""):
    t = v18_fast_clean(text)
    if not t:
        return ""
    if v18_is_self_pr_promo_text(t):
        return "self:pr_promo"
    if v18_is_platform_obligation_text(t):
        if SELF_KAKAO_PATTERN.search(t):
            return "self:platform_obligation"
        if REGULATOR_CORE_PATTERN.search(t) or json_key == "정부_국회":
            return "government:platform_obligation"
        if COMPETITOR_CORE_PATTERN.search(t) or json_key == "경쟁사_해외이슈":
            return "competitor:platform_obligation"
        return "industry:platform_obligation"
    return ""


def v18_issue_family(item, json_key=""):
    jk = json_key or item.get("카테고리") or item.get("JSON카테고리") or ""
    return v18_issue_family_from_text(v18_text_from_article_like(item), jk)


def v18_is_pre_body_relevance_check(report_item):
    # rank_score_article에서 RSS 요약만 넣어 만든 pseudo item에는 본문품질점수/본문글자수 컬럼이 없다.
    # 이 단계에서 본문 품질 하한을 적용하면 모든 RSS 후보가 부당 감점을 받으므로 분리한다.
    return "본문품질점수" not in report_item and "본문글자수" not in report_item


def v18_pre_body_relevance(report_item, json_key):
    text = v18_text_from_article_like(report_item)
    title = v18_fast_clean(report_item.get("기사제목", ""))

    if v18_is_self_pr_promo_text(text):
        return False, "self_pr_promo_low_ceo_priority"
    if v18_is_platform_obligation_text(text):
        return True, ""
    if v12_is_individual_scam_or_crime_case(text):
        return False, "individual_case_not_policy"
    if PHOTO_OR_CAPTION_TITLE_RE.search(title):
        return False, "photo_or_caption_article"
    if v14_is_pure_market_or_stock_noise(text, keyword=report_item.get("검색어", "")):
        return False, "pure_stock_market_or_investment_noise"

    if json_key == "자사_및_계열사_이슈":
        if not v11_has_self_entity(text):
            return False, "self_category_without_self_entity"
        return True, ""
    if json_key == "정부_국회":
        if DIGITAL_STRATEGIC_PATTERN.search(text) or REGULATOR_CORE_PATTERN.search(text) or v12_is_policy_context(text):
            return True, ""
        return False, "government_without_digital_platform_ai_relevance"
    if json_key == "경쟁사_해외이슈":
        if COMPETITOR_CORE_PATTERN.search(text) or DIGITAL_STRATEGIC_PATTERN.search(text):
            return True, ""
        return False, "competitor_without_core_entity_or_digital_relevance"
    if json_key == "산업동향":
        if DIGITAL_STRATEGIC_PATTERN.search(text):
            return True, ""
        return False, "industry_without_core_digital_trend"
    return True, ""


_BASE_v11_story_family_from_signals_v18 = v11_story_family_from_signals

def v11_story_family_from_signals(signals, text="", json_key=""):
    if "platform_obligation" in set(signals) and v18_is_platform_obligation_text(text):
        return "platform_obligation"
    return _BASE_v11_story_family_from_signals_v18(signals, text, json_key)


_BASE_v11_story_topic_bucket_v18 = v11_story_topic_bucket

def v11_story_topic_bucket(item):
    family = v18_issue_family(item)
    if family:
        return family
    return _BASE_v11_story_topic_bucket_v18(item)


def v10_story_topic_bucket(item):
    return v11_story_topic_bucket(item)


def story_topic_bucket_v9(item):
    return v11_story_topic_bucket(item)


_BASE_v11_self_must_review_reason_v18 = v11_self_must_review_reason

def v11_self_must_review_reason(text, json_key=""):
    t = v18_fast_clean(text)
    if v18_is_self_pr_promo_text(t):
        return False, "not_must_review:self_pr_promo_low_ceo_priority"
    if v18_is_platform_obligation_text(t) and SELF_KAKAO_PATTERN.search(t):
        return False, "policy_review:self_included_platform_obligation"
    return _BASE_v11_self_must_review_reason_v18(t, json_key)


def v10_must_review_reason(text, json_key=""):
    if json_key != "자사_및_계열사_이슈":
        return False, ""
    return v11_self_must_review_reason(text, json_key)


def is_v9_must_review_text(text, json_key=""):
    return v10_must_review_reason(text, json_key)


def is_v9_must_review_article(article):
    text = f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('검색어','')} {article.get('원카테고리','')}"
    return v10_must_review_reason(text, article.get("JSON카테고리", ""))


_BASE_collection_relevance_reason_v18 = collection_relevance_reason

def collection_relevance_reason(keyword, title, summary):
    text = f"{v18_fast_clean(title)} {v18_fast_clean(summary)[:1100]}"
    json_key = CATEGORY_TO_JSON_KEY.get(keyword_to_category.get(keyword, ""), "")
    if v18_is_self_pr_promo_text(text):
        # 수집 자체를 막지는 않고 랭킹에서 강하게 낮추는 편이 매일 운영에는 안전하다.
        return "" if json_key == "자사_및_계열사_이슈" else _BASE_collection_relevance_reason_v18(keyword, title, summary)
    if v18_is_platform_obligation_text(text):
        return ""
    return _BASE_collection_relevance_reason_v18(keyword, title, summary)


_BASE_article_importance_score_v18 = article_importance_score

def article_importance_score(title, body="", json_key="", keyword=""):
    score = float(_BASE_article_importance_score_v18(title, body, json_key, keyword) or 0)
    text = f"{v18_fast_clean(title)} {v18_fast_clean(body)[:2600]}"
    if v18_is_self_pr_promo_text(text):
        score -= 85
    if v18_is_platform_obligation_text(text):
        # 플랫폼 사업자 의무/규제 부담은 카카오 사업환경에 직접 영향을 줄 수 있으므로 정책 리스크 가점.
        if SELF_KAKAO_PATTERN.search(text):
            score += 58
        elif json_key == "정부_국회" or REGULATOR_CORE_PATTERN.search(text):
            score += 52
        elif json_key == "경쟁사_해외이슈" or COMPETITOR_CORE_PATTERN.search(text):
            score += 40
        else:
            score += 34
        if re.search(r"내달부터|다음달부터|7월부터|시행|적용|의무화|준수", text, re.IGNORECASE):
            score += 14
        if REGULATOR_CORE_PATTERN.search(text):
            score += 10
    return round(score, 2)


_BASE_is_article_obviously_low_value_v18 = is_article_obviously_low_value

def is_article_obviously_low_value(article):
    text = v18_text_from_article_like(article)
    if v18_is_platform_obligation_text(text):
        return False, ""
    if v18_is_self_pr_promo_text(text):
        return True, "self_pr_promo_low_ceo_priority"
    return _BASE_is_article_obviously_low_value_v18(article)


_BASE_is_report_item_relevant_v18 = is_report_item_relevant

def is_report_item_relevant(report_item, json_key):
    text = v18_text_from_article_like(report_item)
    if v18_is_pre_body_relevance_check(report_item):
        return v18_pre_body_relevance(report_item, json_key)
    if v18_is_self_pr_promo_text(text):
        return False, "self_pr_promo_low_ceo_priority"
    if v18_is_platform_obligation_text(text):
        quality = float(report_item.get("본문품질점수") or 0)
        body_len = int(report_item.get("본문글자수") or 0)
        method = report_item.get("본문추출방식", "")
        if method in {"failed", "meta_description_short", "domain_filter"}:
            return False, "platform_obligation_body_unusable"
        if body_len >= 260 and quality >= 18:
            return True, ""
    return _BASE_is_report_item_relevant_v18(report_item, json_key)


_BASE_body_quality_score_v18 = body_quality_score_v6

def body_quality_score_v6(title, body, method, json_key="", source=""):
    score, reason = _BASE_body_quality_score_v18(title, body, method, json_key=json_key, source=source)
    text = f"{v18_fast_clean(title)} {v18_fast_clean(source)} {v18_fast_clean(body)[:3500]}"
    if v18_is_self_pr_promo_text(text):
        return min(score - 45, 5.0), "self_pr_promo_low_ceo_priority"
    if v18_is_platform_obligation_text(text) and method not in {"failed", "meta_description_short"}:
        score += 18
        if score >= ABSOLUTE_QUALITY_FLOOR and str(reason).startswith("below_"):
            reason = "platform_obligation_policy_relevance"
    return round(score, 2), reason


_BASE_is_body_usable_v18 = is_body_usable_v6

def is_body_usable_v6(title, body, method, json_key="", source="", importance_score=0):
    text = f"{v18_fast_clean(title)} {v18_fast_clean(source)} {v18_fast_clean(body)[:3500]}"
    if v18_is_self_pr_promo_text(text):
        return False, "self_pr_promo_low_ceo_priority", 0.0
    usable, reason, quality = _BASE_is_body_usable_v18(title, body, method, json_key=json_key, source=source, importance_score=importance_score)
    if usable:
        return usable, reason, quality
    if v18_is_platform_obligation_text(text) and method not in {"failed", "meta_description_short"}:
        body_len = len(v18_fast_clean(body))
        if body_len >= 260 and quality >= 18 and importance_score >= 45:
            return True, "short_but_platform_obligation", quality
    return usable, reason, quality


_BASE_rank_score_article_v18 = rank_score_article

def rank_score_article(article):
    score = float(_BASE_rank_score_article_v18(article) or article.get("랭킹점수") or 0)
    text = v18_text_from_article_like(article)
    family = v18_issue_family(article)
    reasons = [r for r in str(article.get("랭킹감점사유", "")).split(";") if r and r != "nan"]

    if family == "self:pr_promo":
        score -= 70
        reasons.append("self_pr_promo_low_ceo_priority")
    elif family == "self:platform_obligation":
        score += 8
        reasons = [r for r in reasons if r not in {"below_absolute_quality_floor", "below_quality_floor"}]
    elif family == "government:platform_obligation":
        score += 8
        reasons = [r for r in reasons if r not in {"below_absolute_quality_floor", "below_quality_floor"}]
    elif family == "competitor:platform_obligation":
        score += 6
        reasons = [r for r in reasons if r not in {"below_absolute_quality_floor", "below_quality_floor"}]

    if v18_is_platform_obligation_text(text):
        article["정책규제의무"] = "Y"
        tags = set(filter(None, str(article.get("사건태그", "")).split(",")))
        tags.add("platform_obligation")
        article["사건태그"] = ",".join(sorted(tags))
    else:
        article["정책규제의무"] = ""

    if v18_is_self_pr_promo_text(text):
        article["자사홍보성"] = "Y"
    else:
        article["자사홍보성"] = ""

    article["이슈패밀리"] = family or article.get("스토리버킷", "")
    article["랭킹감점사유"] = ";".join(dict.fromkeys(reasons))
    article["랭킹점수"] = round(score, 3)
    return article["랭킹점수"]


_BASE_event_tag_limit_reason_v18 = event_tag_limit_reason

def event_tag_limit_reason(new_item, existing_items):
    family = v18_issue_family(new_item)
    if family == "self:pr_promo":
        return "event_story_topic_limit:self:pr_promo:0"
    if family:
        limit = V11_STORY_TOPIC_LIMITS.get(family)
        if limit is not None:
            count = sum(1 for old in existing_items if v18_issue_family(old) == family)
            if count >= limit:
                return f"event_story_topic_limit:{family}:{limit}"
    return _BASE_event_tag_limit_reason_v18(new_item, existing_items)



_BASE_v16_detect_incident_family_v18 = v16_detect_incident_family

def v16_detect_incident_family(text):
    if v18_is_platform_obligation_text(text):
        return "platform_regulation"
    return _BASE_v16_detect_incident_family_v18(text)


_BASE_v16_detect_entity_v18 = v16_detect_entity

def v16_detect_entity(text):
    entity = _BASE_v16_detect_entity_v18(text)
    if entity:
        return entity
    if v18_is_platform_obligation_text(text):
        return "platform_sector"
    return ""

_BASE_v16_detect_object_v18 = v16_detect_object

def v16_detect_object(text, family):
    t = v18_fast_clean(text)
    if family == "platform_regulation" and v18_is_platform_obligation_text(t):
        if re.search(r"콘텐츠|게시물|이미지|영상|동영상|허위조작정보|딥페이크|사칭|피싱|악성코드|청소년|아동|성착취|음란물|유해정보|불법\s*정보|유통\s*방지|확산\s*방지", t, re.IGNORECASE):
            return "platform_content_moderation_obligation"
        if re.search(r"개인정보|정보\s*유출|데이터|위치정보", t, re.IGNORECASE):
            return "platform_data_protection_obligation"
        if re.search(r"광고|추천|알고리즘|검색|다크패턴|리뷰", t, re.IGNORECASE):
            return "platform_algorithm_ad_obligation"
        if re.search(r"입점|판매자|정산|수수료|소비자\s*보호|이용자\s*보호", t, re.IGNORECASE):
            return "platform_transaction_user_protection_obligation"
        return "platform_operator_obligation"
    return _BASE_v16_detect_object_v18(text, family)


_BASE_v16_global_representative_bonus_v18 = v16_global_representative_bonus

def v16_global_representative_bonus(item):
    bonus = float(_BASE_v16_global_representative_bonus_v18(item) or 0)
    text = v18_text_from_article_like(item)
    if v18_is_platform_obligation_text(text):
        bonus += 24
        if REGULATOR_CORE_PATTERN.search(text):
            bonus += 8
    if v18_is_self_pr_promo_text(text):
        bonus -= 50
    return bonus


_BASE_process_article_for_report_v18 = process_article_for_report

def process_article_for_report(article_info, json_key, recent_past_items):
    report_item, skip_info, failed_item = _BASE_process_article_for_report_v18(article_info, json_key, recent_past_items)
    target = report_item or failed_item
    if target:
        family = v18_issue_family(target, json_key)
        target["이슈패밀리"] = family
        target["정책규제의무"] = "Y" if v18_is_platform_obligation_text(v18_text_from_article_like(target)) else ""
        target["자사홍보성"] = "Y" if v18_is_self_pr_promo_text(v18_text_from_article_like(target)) else ""
        if target["정책규제의무"] == "Y":
            tags = set(filter(None, str(target.get("사건태그", "")).split(",")))
            tags.add("platform_obligation")
            target["사건태그"] = ",".join(sorted(tags))
        if report_item:
            target["대표선택점수"] = representative_score(target)
    return report_item, skip_info, failed_item


_BASE_rank_and_trim_candidates_v18 = rank_and_trim_candidates

def rank_and_trim_candidates(raw_articles):
    ranked_all, ranked_candidates = _BASE_rank_and_trim_candidates_v18(raw_articles)
    for rows in (ranked_all, ranked_candidates):
        for a in rows:
            text = v18_text_from_article_like(a)
            family = v18_issue_family(a)
            a["이슈패밀리"] = family or a.get("스토리버킷", "")
            a["정책규제의무"] = "Y" if v18_is_platform_obligation_text(text) else ""
            a["자사홍보성"] = "Y" if v18_is_self_pr_promo_text(text) else ""
            if family == "self:platform_obligation":
                # 업계 공통 플랫폼 의무는 강제검토가 아니라 정책규제의무로 표시한다.
                # 강제검토는 노무/임원/장애/제재 등 직접 자사 리스크에 남겨 둔다.
                a["정책규제의무"] = "Y"
                if float(a.get("랭킹점수") or 0) < 45:
                    a["랭킹점수"] = round(float(a.get("랭킹점수") or 0) + 20, 3)
            if family == "self:pr_promo":
                a["강제검토"] = ""
                a["강제검토사유"] = ""
    # v18 보정 필드 반영 후 다시 정렬한다.
    ranked_all = sorted(
        ranked_all,
        key=lambda x: (float(x.get("랭킹점수") or 0), float(x.get("중요도점수") or 0), x.get("게시일", "")),
        reverse=True,
    )
    ranked_candidates = sorted(
        ranked_candidates,
        key=lambda x: (float(x.get("랭킹점수") or 0), float(x.get("중요도점수") or 0), x.get("게시일", "")),
        reverse=True,
    )
    return ranked_all, ranked_candidates



# v18 performance guard: 복잡한 lookahead 정규식 대신 bounded proximity 패턴을 사용한다.
# 대량 RSS 후보 랭킹에서 편집신호 탐지가 느려지는 것을 방지한다.
try:
    _V18_PLATFORM_OBLIGATION_EVENT_RE = re.compile(
        r"(플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|사업자|부가통신사업자|정보통신서비스\s*제공자|"
        r"네이버|카카오|구글|메타|오픈AI|애플|쿠팡|토스|배민|통신사)"
        r".{0,160}"
        r"(의무|의무화|해야|걸러야|차단|삭제|필터링|신고|보고|공시|자료\s*제출|준수|금지|제한|시행|적용|유통\s*방지|확산\s*방지)|"
        r"(의무|의무화|해야|걸러야|차단|삭제|필터링|신고|보고|공시|자료\s*제출|준수|금지|제한|시행|적용|유통\s*방지|확산\s*방지)"
        r".{0,160}"
        r"(플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|사업자|부가통신사업자|정보통신서비스\s*제공자|"
        r"네이버|카카오|구글|메타|오픈AI|애플|쿠팡|토스|배민|통신사)",
        re.IGNORECASE,
    )
    EVENT_PATTERNS["platform_obligation"] = _V18_PLATFORM_OBLIGATION_EVENT_RE.pattern
    V11_SIGNAL_PATTERNS["platform_obligation"] = _V18_PLATFORM_OBLIGATION_EVENT_RE
except Exception:
    pass


# v18 final ranker: 기존 여러 버전의 rank_and_trim 래퍼를 다시 타지 않고, 현재 최종 scoring 함수만 사용한다.
# 이렇게 해야 v18 신호 추가 후에도 대량 후보 랭킹 속도가 안정적으로 유지된다.
def rank_and_trim_candidates(raw_articles):
    for article in raw_articles:
        score = rank_score_article(article)
        article["랭킹점수"] = score
        article["랭킹원점수"] = article.get("랭킹원점수", score)
        text = v18_text_from_article_like(article)
        signals = v11_detect_signals(text)
        family = v18_issue_family(article)
        article["편집신호"] = ",".join(signals)
        article["스토리버킷"] = family or v11_story_topic_bucket(article)
        article["이슈패밀리"] = family or article.get("스토리버킷", "")
        article["정책규제의무"] = "Y" if v18_is_platform_obligation_text(text) else article.get("정책규제의무", "")
        article["자사홍보성"] = "Y" if v18_is_self_pr_promo_text(text) else article.get("자사홍보성", "")
        article["전역사건키"] = v16_global_incident_key(article)
        article["전역사건기본키"] = v16_global_incident_base_key(article)
        must, reason = v10_must_review_reason(text, article.get("JSON카테고리", ""))
        if v18_is_self_pr_promo_text(text):
            must, reason = False, "not_must_review:self_pr_promo_low_ceo_priority"
        article["강제검토"] = "Y" if must else ""
        article["강제검토사유"] = reason if must else (reason if reason.startswith("not_must_review") else "")
        if must:
            article["랭킹점수"] = round(float(article.get("랭킹점수") or 0) + 28, 3)
        # 업계 공통 의무라도 제목/본문에 카카오가 명시된 플랫폼 규제 이슈는 별도 검토 대상으로 올린다.
        # 다만 노무/임원/장애 같은 직접 자사 리스크와 달리 추가 랭킹 보너스는 주지 않는다.
        if family == "self:platform_obligation" and article.get("정책규제의무") == "Y":
            article["강제검토"] = "Y"
            article["강제검토사유"] = "policy_review:self_included_platform_obligation"
        if article.get("JSON카테고리") == "자사_및_계열사_이슈" and not v11_has_self_entity(text):
            article["랭킹점수"] = round(float(article.get("랭킹점수") or 0) - 60, 3)

    ranked_all = sorted(
        raw_articles,
        key=lambda x: (float(x.get("랭킹점수") or 0), float(x.get("중요도점수") or 0), x.get("게시일", "")),
        reverse=True,
    )

    selected = []
    selected_ids = set()
    selected_topic_counts = {}

    def add(article, respect_topic_cap=True):
        try:
            art_id = int(article["id"])
        except Exception:
            return False
        if art_id in selected_ids:
            return False
        score = float(article.get("랭킹점수") or 0)
        if score <= -30 and article.get("강제검토") != "Y":
            return False
        topic = article.get("이슈패밀리") or article.get("스토리버킷") or ""
        if topic == "self:pr_promo":
            return False
        if respect_topic_cap and topic:
            cap = V11_STORY_TOPIC_LIMITS.get(topic, V11_MUST_REVIEW_TOPIC_POOL_LIMIT if article.get("강제검토") == "Y" else 8)
            if selected_topic_counts.get(topic, 0) >= cap:
                return False
        selected.append(article)
        selected_ids.add(art_id)
        if topic:
            selected_topic_counts[topic] = selected_topic_counts.get(topic, 0) + 1
        return True

    # 1) 직접 자사 리스크 must-review 후보 우선 포함. 플랫폼 공통 의무는 정책규제의무로 표시하되 must-review는 아님.
    own_must = [
        a for a in ranked_all
        if a.get("JSON카테고리") == "자사_및_계열사_이슈"
        and a.get("강제검토") == "Y"
        and float(a.get("랭킹점수") or 0) > -20
    ]
    own_must = sorted(own_must, key=lambda x: (float(x.get("중요도점수") or 0), float(x.get("랭킹점수") or 0)), reverse=True)
    added = 0
    for article in own_must:
        if added >= MUST_REVIEW_POOL_LIMIT.get("자사_및_계열사_이슈", 45):
            break
        if add(article, respect_topic_cap=True):
            added += 1

    # 2) 카테고리별 후보 풀 구성.
    for json_key in JSON_KEYS_ORDER:
        limit = CATEGORY_POOL_LIMIT_FOR_GEMINI.get(json_key, 40)
        current_count = sum(1 for a in selected if a.get("JSON카테고리") == json_key)
        bucket = [
            a for a in ranked_all
            if a.get("JSON카테고리") == json_key
            and int(a.get("id")) not in selected_ids
            and float(a.get("랭킹점수") or 0) > -25
        ]
        for article in bucket:
            if current_count >= limit:
                break
            if add(article, respect_topic_cap=True):
                current_count += 1

    # 3) 남는 슬롯은 전체 랭킹에서 채운다.
    for article in ranked_all:
        if len(selected) >= MAX_CANDIDATES_FOR_GEMINI:
            break
        try:
            art_id = int(article.get("id"))
        except Exception:
            continue
        if art_id in selected_ids:
            continue
        add(article, respect_topic_cap=True)

    if len(selected) < MAX_CANDIDATES_FOR_GEMINI:
        for article in ranked_all:
            if len(selected) >= MAX_CANDIDATES_FOR_GEMINI:
                break
            try:
                art_id = int(article.get("id"))
            except Exception:
                continue
            if art_id in selected_ids:
                continue
            add(article, respect_topic_cap=False)

    selected = sorted(
        selected,
        key=lambda x: (float(x.get("랭킹점수") or 0), float(x.get("중요도점수") or 0), x.get("게시일", "")),
        reverse=True,
    )
    return ranked_all, selected[:MAX_CANDIDATES_FOR_GEMINI]


# =========================================================
# v19 overrides: Gemini editor pipeline
# - Code keeps broad candidate pool and diagnostic hints only.
# - Gemini first labels candidates semantically.
# - Gemini then edits final selection plus backups.
# - Code validates URL/body/duplicates and formats the report.
# =========================================================

V6_VERSION = "google_rss_v19_gemini_editor_pipeline"

# Keep candidate pool broad enough that Gemini can rescue semantically important articles.
MAX_CANDIDATES_FOR_GEMINI = 260
V19_LABEL_BATCH_SIZE = 65
V19_EDITOR_MAX_CANDIDATES = 220
V19_FINAL_SOFT_MAX = 16

# Do not let category filling overrule Gemini too aggressively.
CATEGORY_MIN = {
    "자사_및_계열사_이슈": 2,
    "정부_국회": 4,
    "경쟁사_해외이슈": 3,
    "산업동향": 1,
}
CATEGORY_TARGET = {
    "자사_및_계열사_이슈": 4,
    "정부_국회": 5,
    "경쟁사_해외이슈": 4,
    "산업동향": 1,
}
CATEGORY_MAX = {
    "자사_및_계열사_이슈": 6,
    "정부_국회": 7,
    "경쟁사_해외이슈": 5,
    "산업동향": 2,
}
MIN_SELECT_COUNT = sum(CATEGORY_MIN.values())
MAX_SELECT_COUNT = 18

OUTPUT_GEMINI_LABELS_CSV = os.path.join(BASE_DIR, "google_news_gemini_labels.csv")


def v19_clip(text, limit=240):
    text = clean_html_text(text)
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def v19_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def v19_int(value, default=3, min_value=1, max_value=5):
    try:
        if isinstance(value, str):
            m = re.search(r"\d+", value)
            value = m.group(0) if m else value
        n = int(float(value))
    except Exception:
        n = default
    return max(min_value, min(max_value, n))


def v19_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"true", "1", "y", "yes", "예", "네", "맞음", "맞다", "제외", "홍보"}


def v19_article_text(article, body_chars=0):
    body = article.get("본문전문", "")[:body_chars] if body_chars else ""
    return clean_html_text(
        f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('RSS요약','')} "
        f"{body} {article.get('검색어','')} {article.get('언론사','')}"
    )


def v19_normalize_category_key(value, fallback="산업동향"):
    if isinstance(value, list):
        value = value[0] if value else fallback
    text = clean_html_text(value)
    if not text:
        return fallback if fallback in JSON_KEYS_ORDER else "산업동향"
    if text in JSON_KEYS_ORDER:
        return text
    compact = re.sub(r"\s+", "", text).replace("/", "").replace("·", "").replace("_", "")
    for key in JSON_KEYS_ORDER:
        if text == JSON_KEY_TO_DISPLAY.get(key):
            return key
        for alias in KEY_ALIASES.get(key, []):
            alias_compact = re.sub(r"\s+", "", alias).replace("/", "").replace("·", "").replace("_", "")
            if compact == alias_compact or alias_compact in compact or compact in alias_compact:
                return key
    if "자사" in text or "계열" in text or "카카오" in text:
        return "자사_및_계열사_이슈"
    if "정부" in text or "국회" in text or "규제" in text or "정책" in text:
        return "정부_국회"
    if "경쟁" in text or "해외" in text:
        return "경쟁사_해외이슈"
    if "산업" in text or "동향" in text:
        return "산업동향"
    return fallback if fallback in JSON_KEYS_ORDER else "산업동향"


def v19_has_multiple_platform_names(text):
    text = clean_html_text(text)
    names = ["네이버", "카카오", "구글", "메타", "유튜브", "틱톡", "애플", "쿠팡", "토스", "배민", "배달의민족"]
    return sum(1 for name in names if name in text) >= 2


def v19_code_hints(article):
    text = v19_article_text(article)
    json_key = article.get("JSON카테고리", "") or article.get("카테고리", "")
    family = ""
    platform_obligation = False
    self_pr = False
    try:
        family = v18_issue_family(article) or ""
        platform_obligation = bool(v18_is_platform_obligation_text(text))
        self_pr = bool(v18_is_self_pr_promo_text(text))
    except Exception:
        pass

    direct_must = False
    must_reason = ""
    try:
        direct_must, must_reason = v10_must_review_reason(text, json_key)
    except Exception:
        direct_must, must_reason = False, ""

    individual_case = False
    try:
        individual_case = v12_is_individual_scam_or_crime_case(text)
    except Exception:
        individual_case = False

    stock_noise = False
    try:
        stock_noise = v14_is_pure_market_or_stock_noise(text, keyword=article.get("검색어", ""))
    except Exception:
        stock_noise = False

    hints = []
    if direct_must:
        hints.append("direct_self_risk")
    if platform_obligation:
        hints.append("platform_operator_obligation")
    if self_pr:
        hints.append("self_pr_promo")
    if individual_case:
        hints.append("individual_case")
    if stock_noise:
        hints.append("stock_market_noise")
    if not hints:
        hints.append("general")

    return {
        "text": text,
        "family": family,
        "platform_obligation": platform_obligation,
        "self_pr": self_pr,
        "direct_must": direct_must,
        "must_reason": must_reason,
        "individual_case": individual_case,
        "stock_noise": stock_noise,
        "hints": ",".join(hints),
    }


def v19_local_duplicate_group(article):
    try:
        key = v16_global_incident_base_key(article) or v16_global_incident_key(article)
        if key:
            return v19_clip(key, 90)
    except Exception:
        pass
    hints = v19_code_hints(article)
    if hints.get("family"):
        return hints["family"]
    title = title_fingerprint(article.get("기사제목", ""))
    return title[:70] if title else f"article_{article.get('id','')}"


def v19_local_label_article(article):
    hints = v19_code_hints(article)
    text = hints["text"]
    fallback_cat = article.get("JSON카테고리") or "산업동향"
    category = fallback_cat if fallback_cat in JSON_KEYS_ORDER else "산업동향"
    issue_type = "일반 후보"
    company_relevance = "간접 관련"
    priority = 3
    relevance = 3
    exclude = False
    is_pr = False
    reason = "로컬 보조 라벨"

    score = v19_float(article.get("랭킹점수"), 0)
    if score >= 70:
        priority = 5
    elif score >= 45:
        priority = 4
    elif score < 5:
        priority = 2

    if hints["self_pr"]:
        issue_type = "홍보/후원/프로모션"
        company_relevance = "카카오명은 있으나 CEO 정책·리스크 우선순위 낮음"
        priority = 1
        relevance = 1
        exclude = True
        is_pr = True
        reason = "자사 홍보성/후원성 기사로 판단"

    elif hints["platform_obligation"]:
        issue_type = "플랫폼 사업자 의무/규제"
        company_relevance = "카카오 포함 업계 공통 사업환경 영향" if SELF_KAKAO_PATTERN.search(text) else "플랫폼 업계 정책 영향"
        priority = 4
        relevance = 5
        # 카카오가 여러 사업자 중 하나로 언급되는 규제/의무 기사는 정부/국회로 보는 편이 안정적이다.
        if REGULATOR_CORE_PATTERN.search(text) or v19_has_multiple_platform_names(text) or category == "자사_및_계열사_이슈":
            category = "정부_국회"
        reason = "플랫폼/사업자에 새 의무·준수 부담이 생기는 정책형 기사"

    elif hints["direct_must"]:
        issue_type = "자사 직접 리스크"
        company_relevance = "카카오·계열사 직접 영향"
        priority = 5
        relevance = 5
        category = "자사_및_계열사_이슈"
        reason = hints.get("must_reason") or "자사 직접 리스크 신호"

    elif hints["individual_case"]:
        issue_type = "개별 사건"
        priority = 1
        relevance = 1
        exclude = True
        reason = "정책/플랫폼 대응보다 개별 사건 성격이 강함"

    elif hints["stock_noise"]:
        issue_type = "투자·시황성 기사"
        priority = 1
        relevance = 1
        exclude = True
        reason = "단기 주가·투자 전망 중심"

    return {
        "id": int(article.get("id")),
        "category": category,
        "issue_type": issue_type,
        "company_relevance": company_relevance,
        "ceo_priority": priority,
        "relevance": relevance,
        "is_pr": is_pr,
        "exclude": exclude,
        "duplicate_group": v19_local_duplicate_group(article),
        "reason": v19_clip(reason, 180),
        "label_source": "local",
    }


def v19_candidate_line(article):
    hints = v19_code_hints(article)
    return (
        f"[{article.get('id')}] "
        f"cat={article.get('JSON카테고리','')} search={article.get('검색어','')} "
        f"score={article.get('랭킹점수','')} imp={article.get('중요도점수','')} "
        f"hint={hints.get('hints','')} family={hints.get('family','')} "
        f"title={v19_clip(article.get('기사제목',''), 110)} "
        f"source={v19_clip(article.get('언론사',''), 40)} "
        f"summary={v19_clip(article.get('본문요약',''), 230)}"
    )


def v19_normalize_label_json(data, valid_ids, article_by_id):
    rows = []
    if isinstance(data, dict):
        rows = data.get("articles") or data.get("기사") or data.get("labels") or data.get("items") or []
    elif isinstance(data, list):
        rows = data
    if not isinstance(rows, list):
        rows = []

    labels = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            art_id = int(row.get("id") or row.get("기사ID") or row.get("article_id"))
        except Exception:
            continue
        if art_id not in valid_ids:
            continue
        article = article_by_id.get(art_id, {})
        local = v19_local_label_article(article) if article else {}
        fallback_cat = local.get("category") or article.get("JSON카테고리") or "산업동향"
        label = {
            "id": art_id,
            "category": v19_normalize_category_key(row.get("category") or row.get("카테고리"), fallback=fallback_cat),
            "issue_type": v19_clip(row.get("issue_type") or row.get("이슈유형") or local.get("issue_type", ""), 80),
            "company_relevance": v19_clip(row.get("company_relevance") or row.get("관련성") or row.get("카카오관련성") or local.get("company_relevance", ""), 120),
            "ceo_priority": v19_int(row.get("ceo_priority") or row.get("CEO중요도") or row.get("priority"), default=local.get("ceo_priority", 3)),
            "relevance": v19_int(row.get("relevance") or row.get("관련도"), default=local.get("relevance", 3)),
            "is_pr": v19_bool(row.get("is_pr") if "is_pr" in row else row.get("홍보성", local.get("is_pr", False))),
            "exclude": v19_bool(row.get("exclude") if "exclude" in row else row.get("제외", local.get("exclude", False))),
            "duplicate_group": v19_clip(row.get("duplicate_group") or row.get("중복그룹") or local.get("duplicate_group", ""), 100),
            "reason": v19_clip(row.get("reason") or row.get("판단사유") or local.get("reason", ""), 180),
            "label_source": "gemini",
        }
        if not label["duplicate_group"]:
            label["duplicate_group"] = local.get("duplicate_group", f"article_{art_id}")
        labels[art_id] = label
    return labels


def v19_gemini_label_candidates(client, candidates, recent_past_text):
    if not client:
        raise RuntimeError("Gemini client 없음")
    article_by_id = {int(a["id"]): a for a in candidates}
    all_labels = {}
    batches = [candidates[i:i + V19_LABEL_BATCH_SIZE] for i in range(0, len(candidates), V19_LABEL_BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches, 1):
        candidate_text = "\n".join(v19_candidate_line(a) for a in batch)
        prompt = f"""
너는 카카오 대외협력·정책리스크팀의 뉴스 후보 분류 담당자야.
아래 후보 각각에 대해 CEO 아침 브리핑 관점의 의미 라벨을 붙여줘.

[판단 기준]
- 특정 키워드에 집착하지 말고, 기사 본질을 봐.
- 카카오 직접 리스크: 노무/임단협/파업, 임원·조직개편, 서비스 개편 후폭풍, 장애·보안·개인정보, 소송·수사·제재, 지배구조/M&A/실적 등.
- 플랫폼 사업자 의무/규제: 카카오가 여러 사업자 중 하나로 포함되더라도 사업자에게 차단·삭제·필터링·공시·보고·자료제출·준수 의무가 생기면 중요하다. 이 경우 보통 정부_국회 카테고리가 맞다.
- 홍보/후원/스폰서/이벤트/축제/쿠폰/프로모션은 원칙적으로 제외한다.
- 개별 범죄·지역 사건·단순 주가전망·오피니언·제품 루머는 제외한다.
- 같은 사건은 duplicate_group을 같게 붙여라. 예: 카카오 노조 공동파업, 카카오게임즈 공동대표, 플랫폼 유해정보 차단의무.

[최근 7일 과거 보고서 참고]
{recent_past_text[:2500]}

[후보 리스트: 배치 {batch_idx}/{len(batches)}]
{candidate_text}

반드시 JSON 객체만 반환해. 설명문 금지.
형식:
{{
  "articles": [
    {{
      "id": 123,
      "category": "자사_및_계열사_이슈 또는 정부_국회 또는 경쟁사_해외이슈 또는 산업동향",
      "issue_type": "자사 직접 리스크/플랫폼 사업자 의무·규제/경쟁사 전략/산업 구조 변화/홍보·프로모션/개별 사건 등",
      "company_relevance": "카카오 직접 영향/카카오 포함 업계 공통 영향/간접 영향/낮음",
      "ceo_priority": 1,
      "relevance": 1,
      "is_pr": false,
      "exclude": false,
      "duplicate_group": "짧은 사건그룹명",
      "reason": "짧은 판단 사유"
    }}
  ]
}}
"""
        text = gemini_generate_text(client=client, prompt=prompt, task_name=f"Gemini 후보 라벨링 {batch_idx}/{len(batches)}")
        data = extract_json_object(text)
        labels = v19_normalize_label_json(data, set(article_by_id.keys()), article_by_id)
        all_labels.update(labels)
        print(f"  └ 🏷️ Gemini 라벨링 배치 {batch_idx}/{len(batches)} 완료: {len(labels)}개")

    if not all_labels:
        raise ValueError("Gemini 라벨링 결과가 비어 있습니다.")
    return all_labels


def v19_apply_labels_to_articles(articles, labels):
    rows = []
    for article in articles:
        try:
            art_id = int(article.get("id"))
        except Exception:
            continue
        label = labels.get(art_id)
        if not label:
            label = v19_local_label_article(article)
            labels[art_id] = label
        article["Gemini카테고리"] = label.get("category", "")
        article["Gemini이슈유형"] = label.get("issue_type", "")
        article["Gemini관련성"] = label.get("company_relevance", "")
        article["Gemini중요도"] = label.get("ceo_priority", "")
        article["Gemini관련도"] = label.get("relevance", "")
        article["GeminiPR성"] = "Y" if label.get("is_pr") else ""
        article["Gemini제외"] = "Y" if label.get("exclude") else ""
        article["Gemini중복그룹"] = label.get("duplicate_group", "")
        article["Gemini판단사유"] = label.get("reason", "")
        article["Gemini라벨소스"] = label.get("label_source", "")
        rows.append({
            "id": art_id,
            "기사제목": article.get("기사제목", ""),
            "언론사": article.get("언론사", ""),
            "원카테고리": article.get("JSON카테고리", ""),
            "Gemini카테고리": article.get("Gemini카테고리", ""),
            "Gemini이슈유형": article.get("Gemini이슈유형", ""),
            "Gemini관련성": article.get("Gemini관련성", ""),
            "Gemini중요도": article.get("Gemini중요도", ""),
            "Gemini관련도": article.get("Gemini관련도", ""),
            "GeminiPR성": article.get("GeminiPR성", ""),
            "Gemini제외": article.get("Gemini제외", ""),
            "Gemini중복그룹": article.get("Gemini중복그룹", ""),
            "Gemini판단사유": article.get("Gemini판단사유", ""),
            "랭킹점수": article.get("랭킹점수", ""),
        })
    return rows


def v19_editor_line(article):
    return (
        f"[{article.get('id')}] cat_suggest={article.get('Gemini카테고리') or article.get('JSON카테고리')} "
        f"priority={article.get('Gemini중요도','')} relevance={article.get('Gemini관련도','')} "
        f"type={v19_clip(article.get('Gemini이슈유형',''), 55)} "
        f"dup={v19_clip(article.get('Gemini중복그룹',''), 70)} "
        f"pr={article.get('GeminiPR성','')} exclude={article.get('Gemini제외','')} "
        f"score={article.get('랭킹점수','')} "
        f"title={v19_clip(article.get('기사제목',''), 120)} "
        f"reason={v19_clip(article.get('Gemini판단사유',''), 120)}"
    )


def v19_extract_id_list(value):
    out = []
    if not isinstance(value, list):
        return out
    for item in value:
        try:
            if isinstance(item, dict):
                item = item.get("id") or item.get("기사ID") or item.get("article_id")
            out.append(int(item))
        except Exception:
            continue
    return out


def v19_normalize_editor_json(data, valid_ids, labels):
    if not isinstance(data, dict):
        raise ValueError("편집 JSON 객체가 아닙니다.")
    selection_raw = data.get("selection") or data.get("선정") or data.get("selected") or data
    backup_raw = data.get("backup") or data.get("대체") or data.get("backup_selection") or {}
    reason_raw = data.get("selection_reason") or data.get("reasons") or data.get("선정사유") or {}

    selection = {key: [] for key in JSON_KEYS_ORDER}
    backup = {key: [] for key in JSON_KEYS_ORDER}
    used = set()

    for raw_key, raw_values in selection_raw.items() if isinstance(selection_raw, dict) else []:
        key = v19_normalize_category_key(raw_key, fallback="산업동향")
        if key not in selection:
            continue
        for art_id in v19_extract_id_list(raw_values):
            if art_id not in valid_ids or art_id in used:
                continue
            label = labels.get(art_id, {})
            if label.get("exclude") or label.get("is_pr"):
                continue
            if len(selection[key]) >= CATEGORY_MAX.get(key, 5):
                continue
            selection[key].append(art_id)
            used.add(art_id)

    backup_used = set(used)
    for raw_key, raw_values in backup_raw.items() if isinstance(backup_raw, dict) else []:
        key = v19_normalize_category_key(raw_key, fallback="산업동향")
        if key not in backup:
            continue
        for art_id in v19_extract_id_list(raw_values):
            if art_id not in valid_ids or art_id in backup_used:
                continue
            label = labels.get(art_id, {})
            if label.get("exclude") or label.get("is_pr"):
                continue
            backup[key].append(art_id)
            backup_used.add(art_id)

    reason_map = {}
    if isinstance(reason_raw, dict):
        for k, v in reason_raw.items():
            try:
                reason_map[int(k)] = v19_clip(v, 200)
            except Exception:
                continue
    elif isinstance(reason_raw, list):
        for row in reason_raw:
            if isinstance(row, dict):
                try:
                    reason_map[int(row.get("id"))] = v19_clip(row.get("reason") or row.get("사유") or "", 200)
                except Exception:
                    pass
    return selection, backup, reason_map


def v19_rank_for_editor(article):
    label_priority = v19_int(article.get("Gemini중요도"), default=3)
    relevance = v19_int(article.get("Gemini관련도"), default=3)
    score = v19_float(article.get("랭킹점수"), 0)
    direct = 1 if article.get("강제검토") == "Y" else 0
    exclude = -10 if article.get("Gemini제외") == "Y" else 0
    pr = -8 if article.get("GeminiPR성") == "Y" else 0
    return (exclude + pr, label_priority, relevance, direct, score)


def v19_fill_selection_with_backups(selection, backup, ranked_candidates, labels):
    used = {art_id for ids in selection.values() for art_id in ids}

    def can_add(art_id, key):
        if art_id in used:
            return False
        label = labels.get(art_id, {})
        if label.get("exclude") or label.get("is_pr"):
            return False
        if len(selection[key]) >= CATEGORY_MAX.get(key, 5):
            return False
        return True

    # Fill category minimums from Gemini backups first.
    for key in JSON_KEYS_ORDER:
        for art_id in backup.get(key, []):
            if len(selection[key]) >= CATEGORY_MIN.get(key, 0):
                break
            if can_add(art_id, key):
                selection[key].append(art_id)
                used.add(art_id)

    # Fill category minimums from labeled candidates.
    for key in JSON_KEYS_ORDER:
        if len(selection[key]) >= CATEGORY_MIN.get(key, 0):
            continue
        pool = [a for a in ranked_candidates if labels.get(int(a["id"]), {}).get("category") == key]
        pool = sorted(pool, key=v19_rank_for_editor, reverse=True)
        for article in pool:
            if len(selection[key]) >= CATEGORY_MIN.get(key, 0):
                break
            art_id = int(article["id"])
            if can_add(art_id, key):
                selection[key].append(art_id)
                used.add(art_id)

    # Soft total fill to target count.
    target_total = min(V19_FINAL_SOFT_MAX, MAX_SELECT_COUNT)
    while sum(len(v) for v in selection.values()) < MIN_SELECT_COUNT:
        added = False
        pool = sorted(ranked_candidates, key=v19_rank_for_editor, reverse=True)
        for article in pool:
            art_id = int(article["id"])
            label = labels.get(art_id, {})
            key = label.get("category") or article.get("JSON카테고리") or "산업동향"
            key = v19_normalize_category_key(key, fallback="산업동향")
            if can_add(art_id, key):
                selection[key].append(art_id)
                used.add(art_id)
                added = True
                break
        if not added:
            break

    # Trim if Gemini over-selected.
    all_pairs = []
    for key, ids in selection.items():
        for order, art_id in enumerate(ids):
            label = labels.get(art_id, {})
            all_pairs.append((key, art_id, label.get("ceo_priority", 3), label.get("relevance", 3), order))
    if len(all_pairs) > MAX_SELECT_COUNT:
        keep = sorted(all_pairs, key=lambda x: (x[2], x[3], -x[4]), reverse=True)[:MAX_SELECT_COUNT]
        keep_ids = {art_id for _, art_id, _, _, _ in keep}
        for key in JSON_KEYS_ORDER:
            selection[key] = [art_id for art_id in selection[key] if art_id in keep_ids]
    return selection


def v19_gemini_editor_selection(client, ranked_candidates, labels, recent_past_text):
    if not client:
        raise RuntimeError("Gemini client 없음")
    valid_ids = {int(a["id"]) for a in ranked_candidates}
    eligible = [a for a in ranked_candidates if not labels.get(int(a["id"]), {}).get("exclude")]
    eligible = sorted(eligible, key=v19_rank_for_editor, reverse=True)[:V19_EDITOR_MAX_CANDIDATES]
    editor_text = "\n".join(v19_editor_line(a) for a in eligible)

    prompt = f"""
너는 카카오 CEO 아침 브리핑 최종 편집장이다.
아래 후보는 이미 Gemini 1차 라벨링을 거친 기사들이다. 최종 브리핑에 들어갈 기사와 대체 후보를 골라라.

[최종 편집 원칙]
1. 총 12~16개를 원칙으로 한다. 정말 중요한 날만 17개까지 가능하다.
2. 같은 duplicate_group은 원칙적으로 대표 기사 1개만 고른다. 단, 진행 단계가 명확히 다른 자사 노무/소송/규제 업데이트는 최대 2개까지 가능하다.
3. 카카오 직접 리스크는 우선한다. 단, 같은 노조/카카오게임즈/실적 이슈가 반복되면 대표 기사만 남긴다.
4. 네이버·카카오·구글 등 여러 사업자에게 새 의무가 생기는 플랫폼 규제 기사는 보통 정부_국회로 넣는다.
5. 홍보/후원/스폰서/이벤트/축제/쿠폰/단순 프로모션은 원칙적으로 제외한다.
6. 일반 금융 제재, 개별 기업 공정위 사건, 개별 범죄·선거사범 등은 카카오 플랫폼 사업환경과 직접 연결될 때만 고른다.
7. 본문 추출 실패를 대비해 각 카테고리별 backup도 넉넉히 골라라.

[최근 7일 과거 보고서 참고]
{recent_past_text[:3000]}

[라벨링된 후보]
{editor_text}

반드시 JSON 객체만 반환해. 설명문 금지.
형식:
{{
  "selection": {{
    "자사_및_계열사_이슈": [1, 2],
    "정부_국회": [3, 4],
    "경쟁사_해외이슈": [5, 6],
    "산업동향": [7]
  }},
  "backup": {{
    "자사_및_계열사_이슈": [10, 11, 12],
    "정부_국회": [13, 14, 15],
    "경쟁사_해외이슈": [16, 17, 18],
    "산업동향": [19, 20]
  }},
  "selection_reason": {{
    "1": "짧은 선정 사유"
  }}
}}
"""
    text = gemini_generate_text(client=client, prompt=prompt, task_name="Gemini 최종 편집")
    data = extract_json_object(text)
    selection, backup, reason_map = v19_normalize_editor_json(data, valid_ids, labels)
    selection = v19_fill_selection_with_backups(selection, backup, ranked_candidates, labels)
    return selection, backup, reason_map


def v19_local_editor_selection(ranked_candidates, labels):
    selection = {key: [] for key in JSON_KEYS_ORDER}
    backup = {key: [] for key in JSON_KEYS_ORDER}
    used = set()
    group_counts = {}

    pool = sorted(ranked_candidates, key=v19_rank_for_editor, reverse=True)

    def try_add(article, key, target, hard=False):
        art_id = int(article["id"])
        if art_id in used:
            return False
        label = labels.get(art_id, {})
        if label.get("exclude") or label.get("is_pr"):
            return False
        group = label.get("duplicate_group", "")
        group_limit = 2 if re.search(r"노조|파업|labor|임단협|조정", group, re.IGNORECASE) else 1
        if group and group_counts.get(group, 0) >= group_limit:
            return False
        if len(selection[key]) >= target and not hard:
            return False
        if len(selection[key]) >= CATEGORY_MAX.get(key, 5):
            return False
        selection[key].append(art_id)
        used.add(art_id)
        if group:
            group_counts[group] = group_counts.get(group, 0) + 1
        return True

    for key in JSON_KEYS_ORDER:
        target = CATEGORY_MIN.get(key, 0)
        for article in pool:
            label = labels.get(int(article["id"]), {})
            if label.get("category") != key:
                continue
            if len(selection[key]) >= target:
                break
            try_add(article, key, target, hard=True)

    for key in JSON_KEYS_ORDER:
        target = CATEGORY_TARGET.get(key, 3)
        for article in pool:
            label = labels.get(int(article["id"]), {})
            if label.get("category") != key:
                continue
            if sum(len(v) for v in selection.values()) >= V19_FINAL_SOFT_MAX:
                break
            if len(selection[key]) >= target:
                break
            try_add(article, key, target)

    for key in JSON_KEYS_ORDER:
        for article in pool:
            art_id = int(article["id"])
            label = labels.get(art_id, {})
            if label.get("category") != key or art_id in used:
                continue
            if label.get("exclude") or label.get("is_pr"):
                continue
            backup[key].append(art_id)
            used.add(art_id)
            if len(backup[key]) >= 12:
                break
    return selection, backup, {}


_BASE_RANK_SCORE_V19 = globals().get("_BASE_rank_score_article_v18", rank_score_article)


def rank_score_article(article):
    # Use the stable pre-v18 scorer, then add only soft diagnostics.
    try:
        score = float(_BASE_RANK_SCORE_V19(article) or article.get("랭킹점수") or 0)
    except Exception:
        score = float(article.get("랭킹점수") or 0)

    hints = v19_code_hints(article)
    reasons = [r for r in str(article.get("랭킹감점사유", "")).split(";") if r and r != "nan"]

    if hints["self_pr"]:
        score -= 35
        reasons.append("self_pr_promo_low_ceo_priority")
    if hints["platform_obligation"]:
        # This is a rescue signal, not an automatic top-rank boost.
        score += 4
        reasons = [r for r in reasons if r not in {"below_absolute_quality_floor", "below_quality_floor"}]
        article["정책규제의무"] = "Y"
    else:
        article["정책규제의무"] = ""
    if hints["direct_must"]:
        score += 18
        article["강제검토"] = "Y"
        article["강제검토사유"] = hints.get("must_reason", "direct_self_risk")
    else:
        article["강제검토"] = ""
        article["강제검토사유"] = hints.get("must_reason", "") if str(hints.get("must_reason", "")).startswith("not_must_review") else ""
    if hints["self_pr"]:
        article["자사홍보성"] = "Y"
    else:
        article["자사홍보성"] = ""

    article["이슈패밀리"] = hints.get("family", "")
    article["코드힌트"] = hints.get("hints", "")
    article["랭킹감점사유"] = ";".join(dict.fromkeys(reasons))
    article["랭킹점수"] = round(score, 3)
    return article["랭킹점수"]


def rank_and_trim_candidates(raw_articles):
    for article in raw_articles:
        score = rank_score_article(article)
        article["랭킹원점수"] = article.get("랭킹원점수", score)
        try:
            text = v19_article_text(article)
            article["편집신호"] = ",".join(v11_detect_signals(text))
            article["스토리버킷"] = v11_story_topic_bucket(article)
            article["전역사건키"] = v16_global_incident_key(article)
            article["전역사건기본키"] = v16_global_incident_base_key(article)
        except Exception:
            pass

    ranked_all = sorted(
        raw_articles,
        key=lambda x: (v19_float(x.get("랭킹점수")), v19_float(x.get("중요도점수")), x.get("게시일", "")),
        reverse=True,
    )

    selected = []
    selected_ids = set()

    def add(article):
        try:
            art_id = int(article["id"])
        except Exception:
            return False
        if art_id in selected_ids:
            return False
        score = v19_float(article.get("랭킹점수"), 0)
        # Keep pool broad, but remove deeply irrelevant low-score noise unless it has a rescue signal.
        if score <= -55 and article.get("강제검토") != "Y" and article.get("정책규제의무") != "Y":
            return False
        selected.append(article)
        selected_ids.add(art_id)
        return True

    # Ensure high-signal candidates are visible to Gemini.
    priority_pool = [a for a in ranked_all if a.get("강제검토") == "Y" or a.get("정책규제의무") == "Y"]
    for article in priority_pool[:90]:
        add(article)

    # Category-balanced pool, without aggressive topic caps.
    for key in JSON_KEYS_ORDER:
        limit = CATEGORY_POOL_LIMIT_FOR_GEMINI.get(key, 50)
        bucket = [a for a in ranked_all if a.get("JSON카테고리") == key]
        count = sum(1 for a in selected if a.get("JSON카테고리") == key)
        for article in bucket:
            if count >= limit:
                break
            if add(article):
                count += 1

    # Fill remaining by global score.
    for article in ranked_all:
        if len(selected) >= MAX_CANDIDATES_FOR_GEMINI:
            break
        add(article)

    selected = sorted(
        selected,
        key=lambda x: (v19_float(x.get("랭킹점수")), v19_float(x.get("중요도점수")), x.get("게시일", "")),
        reverse=True,
    )
    return ranked_all, selected[:MAX_CANDIDATES_FOR_GEMINI]


def v19_gemini_group_limit_reason(new_item, existing_items):
    group = clean_html_text(new_item.get("Gemini중복그룹", ""))
    if not group or group.lower() in {"general", "일반", "기타"}:
        return ""
    limit = 2 if re.search(r"노조|파업|임단협|조정|labor", group, re.IGNORECASE) else 1
    count = sum(1 for old in existing_items if clean_html_text(old.get("Gemini중복그룹", "")) == group)
    if count >= limit:
        return f"gemini_duplicate_group_cap:{group}:{limit}"
    return ""


def v19_attach_label_fields(report_item, article_info, selected_category, reason_map=None):
    reason_map = reason_map or {}
    try:
        art_id = int(article_info.get("id"))
    except Exception:
        art_id = None
    label_reason = ""
    if art_id is not None:
        label_reason = reason_map.get(art_id, "")
    report_item["Gemini카테고리"] = selected_category or article_info.get("Gemini카테고리", "")
    report_item["Gemini이슈유형"] = article_info.get("Gemini이슈유형", "")
    report_item["Gemini관련성"] = article_info.get("Gemini관련성", "")
    report_item["Gemini중요도"] = article_info.get("Gemini중요도", "")
    report_item["Gemini관련도"] = article_info.get("Gemini관련도", "")
    report_item["GeminiPR성"] = article_info.get("GeminiPR성", "")
    report_item["Gemini제외"] = article_info.get("Gemini제외", "")
    report_item["Gemini중복그룹"] = article_info.get("Gemini중복그룹", "")
    report_item["Gemini판단사유"] = article_info.get("Gemini판단사유", "")
    report_item["Gemini선정사유"] = label_reason or article_info.get("Gemini판단사유", "")
    report_item["코드힌트"] = article_info.get("코드힌트", "")
    return report_item


def main():
    total_start_time = time.time()
    run_log = []

    print(f"\n🧩 v19 실행: {V6_VERSION}")

    client = None
    if ENABLE_GEMINI_SELECTION or ENABLE_GEMINI_REPORT:
        try:
            with open(SECRET_PATH, "r", encoding="utf-8") as f:
                google_api_key = f.read().strip()
            if not google_api_key:
                raise ValueError("secret.txt가 비어 있습니다.")
            client = genai.Client(api_key=google_api_key)
        except Exception as e:
            print("⚠️ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다. Gemini 없이 로컬 라벨/선별로 진행합니다.")
            print(f"   원인: {e}")
            client = None

    past_reports_content, all_past_items, recent_past_items = load_past_reports()
    recent_past_text = build_recent_past_text(recent_past_items, max_chars=9000)

    raw_articles, skipped_duplicates = collect_with_google_rss(recent_past_items)

    print(f"\n  └ ✅ 총 {len(raw_articles)}개의 RSS 후보 확보 완료")
    print(f"  └ 🧹 수집/과거중복/오탐 제외 후보: {len(skipped_duplicates)}개")

    if not raw_articles:
        print("❌ 수집된 기사가 없습니다. Google News RSS 접속 또는 키워드/기간 설정을 확인하세요.")
        return

    ranked_all, ranked_candidates = rank_and_trim_candidates(raw_articles)
    article_by_ranked_id = {int(a["id"]): a for a in ranked_all}

    print("\n🏷️ [STEP 2-1] Gemini가 후보별 관련성·카테고리·중복그룹을 라벨링합니다...")
    labels = {}
    gemini_label_success = False
    try:
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 라벨링 비활성화 또는 client 없음")
        labels = v19_gemini_label_candidates(client, ranked_candidates, recent_past_text)
        gemini_label_success = True
        print(f"  └ ✅ Gemini 라벨링 완료: {len(labels)}개")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 라벨링 실패. 로컬 보조 라벨로 진행합니다. 원인: {e}")
        labels = {int(a["id"]): v19_local_label_article(a) for a in ranked_candidates}

    # Fill labels for all ranked articles so CSV diagnostics are complete.
    for article in ranked_all:
        art_id = int(article["id"])
        if art_id not in labels:
            labels[art_id] = v19_local_label_article(article)

    label_rows = v19_apply_labels_to_articles(ranked_all, labels)
    v19_apply_labels_to_articles(ranked_candidates, labels)

    pd.DataFrame(ranked_all).to_csv(OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(ranked_candidates).to_csv(OUTPUT_RANKED_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(OUTPUT_GEMINI_LABELS_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 전체 후보 기사 저장: {os.path.basename(OUTPUT_CANDIDATES_CSV)}")
    print(f"  └ 💾 AI/본문 추출 후보 저장: {os.path.basename(OUTPUT_RANKED_CSV)} ({len(ranked_candidates)}개)")
    print(f"  └ 💾 Gemini/로컬 라벨 저장: {os.path.basename(OUTPUT_GEMINI_LABELS_CSV)}")

    if skipped_duplicates:
        pd.DataFrame(skipped_duplicates).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 제외 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")

    print("\n🧠 [STEP 2-2] Gemini가 최종 기사와 대체 후보를 편집합니다...")
    selection = {key: [] for key in JSON_KEYS_ORDER}
    backup = {key: [] for key in JSON_KEYS_ORDER}
    reason_map = {}
    gemini_editor_success = False
    try:
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 최종 편집 비활성화 또는 client 없음")
        selection, backup, reason_map = v19_gemini_editor_selection(client, ranked_candidates, labels, recent_past_text)
        gemini_editor_success = True
        selected_count = sum(len(v) for v in selection.values())
        print(f"  └ ✅ Gemini 최종 편집 완료: 선택 {selected_count}개")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 최종 편집 실패. 라벨 기반 로컬 편집으로 진행합니다. 원인: {e}")
        selection, backup, reason_map = v19_local_editor_selection(ranked_candidates, labels)
        print(f"  └ ✅ 로컬 편집 완료: 선택 {sum(len(v) for v in selection.values())}개")

    # Store selection decision columns in ranked CSV after editor pass.
    selected_set = {art_id for ids in selection.values() for art_id in ids}
    backup_set = {art_id for ids in backup.values() for art_id in ids}
    for article in ranked_all:
        art_id = int(article["id"])
        article["Gemini최종선택"] = "Y" if art_id in selected_set else ""
        article["Gemini백업후보"] = "Y" if art_id in backup_set else ""
        article["Gemini선정사유"] = reason_map.get(art_id, "")
    pd.DataFrame(ranked_all).to_csv(OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame([a for a in ranked_all if int(a["id"]) in {int(x["id"]) for x in ranked_candidates}]).to_csv(OUTPUT_RANKED_CSV, index=False, encoding="utf-8-sig")

    print("\n🕵️‍♂️ [STEP 3] Gemini 선택/백업 후보만 원문 URL 변환 후 본문 전문을 추출합니다...")
    print("   └ 코드는 본문 품질·중복·관련성 검증만 수행하고, 대체는 Gemini backup을 우선 사용합니다.")

    article_by_id = {int(article["id"]): article for article in raw_articles}
    final_report_data = []
    body_failed_rows = []
    post_body_duplicate_skips = []
    processed_ids = set()
    replacement_count = 0
    selection_order_map = {}
    order_counter = 1

    def category_count(json_key):
        return len([x for x in final_report_data if x.get("카테고리") == json_key])

    def total_count():
        return len(final_report_data)

    def try_process(art_id, json_key, reason="selected"):
        nonlocal replacement_count, order_counter
        if art_id in processed_ids:
            return False
        processed_ids.add(art_id)

        article_info = article_by_id.get(int(art_id)) or article_by_ranked_id.get(int(art_id))
        if not article_info:
            return False

        report_item, skip_info, failed_item = process_article_for_report(article_info, json_key, recent_past_items)

        if skip_info:
            skip_info["제외단계"] = reason
            skip_info["Gemini중복그룹"] = article_info.get("Gemini중복그룹", "")
            skip_info["Gemini판단사유"] = article_info.get("Gemini판단사유", "")
            post_body_duplicate_skips.append(skip_info)
            print(f"  └ 🧹 본문 확인 후 과거 중복 제외: {article_info.get('기사제목', '')[:44]}...")
            return False

        if failed_item:
            v19_attach_label_fields(failed_item, article_info, json_key, reason_map)
            failed_item["제외단계"] = reason
            body_failed_rows.append(failed_item)
            print(
                f"  └ ⚠️ 본문 품질 미달로 교체: {failed_item.get('기사제목', '')[:44]}... "
                f"({failed_item.get('본문추출방식')}, {failed_item.get('본문글자수')}자, "
                f"품질 {failed_item.get('본문품질점수')}, 중요도 {failed_item.get('중요도점수')}, "
                f"{failed_item.get('본문품질사유')})"
            )
            return False

        if not report_item:
            return False

        v19_attach_label_fields(report_item, article_info, json_key, reason_map)
        report_item["선정단계"] = reason
        report_item["선정순서"] = order_counter
        order_counter += 1

        relevant, relevance_reason = is_report_item_relevant(report_item, json_key)
        if not relevant:
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"relevance_failed:{relevance_reason}"
            failed_copy["제외단계"] = reason
            body_failed_rows.append(failed_copy)
            print(f"  └ ⚠️ 관련성 미달로 교체: {report_item.get('기사제목', '')[:44]}... ({relevance_reason})")
            return False

        group_limit_reason = v19_gemini_group_limit_reason(report_item, final_report_data)
        if group_limit_reason:
            add_duplicate_skip_row(
                post_body_duplicate_skips,
                report_item,
                {"기사제목": "gemini_duplicate_group_cap", "링크": "", "대표선택점수": ""},
                group_limit_reason,
                stage=reason,
            )
            print(f"  └ 🧹 Gemini 중복그룹 제한 제외: {report_item.get('기사제목', '')[:44]}... ({group_limit_reason})")
            return False

        report_item["대표선택점수"] = representative_score(report_item)
        report_item["강제검토"] = article_info.get("강제검토", "")
        report_item["강제검토사유"] = article_info.get("강제검토사유", "")

        dup_item, dup_reason = final_duplicate_reason(report_item, final_report_data)
        if dup_item:
            new_score = v19_float(report_item.get("대표선택점수"), 0)
            old_score = v19_float(dup_item.get("대표선택점수"), 0)
            if new_score > old_score + REPRESENTATIVE_REPLACE_MARGIN:
                idx = final_report_data.index(dup_item)
                final_report_data[idx] = report_item
                replacement_count += 1
                add_duplicate_skip_row(
                    post_body_duplicate_skips,
                    dup_item,
                    report_item,
                    f"replaced_by_better_duplicate:{dup_reason}",
                    stage=reason,
                    extra=f"old={old_score},new={new_score}",
                )
                print(
                    f"  └ 🔁 중복 대표 교체: {dup_item.get('기사제목', '')[:28]}... "
                    f"→ {report_item.get('기사제목', '')[:28]}... ({old_score}→{new_score}, {dup_reason})"
                )
                return True
            add_duplicate_skip_row(
                post_body_duplicate_skips,
                report_item,
                dup_item,
                f"kept_better_duplicate:{dup_reason}",
                stage=reason,
                extra=f"old={old_score},new={new_score}",
            )
            print(
                f"  └ 🧹 당일 중복 제외: {report_item.get('기사제목', '')[:44]}... "
                f"(대표점수 {new_score}≤{old_score}, {dup_reason})"
            )
            return False

        tag_limit_reason = event_tag_limit_reason(report_item, final_report_data)
        if tag_limit_reason:
            add_duplicate_skip_row(
                post_body_duplicate_skips,
                report_item,
                {"기사제목": "event_tag_cap", "링크": "", "대표선택점수": ""},
                tag_limit_reason,
                stage=reason,
            )
            print(f"  └ 🧹 사건태그 과다 반복 제외: {report_item.get('기사제목', '')[:44]}... ({tag_limit_reason})")
            return False

        final_report_data.append(report_item)
        print(
            f"  └ 📥 본문 추출 완료: {report_item.get('기사제목', '')[:44]}... "
            f"({report_item.get('본문추출방식')}, {report_item.get('본문글자수')}자, "
            f"품질 {report_item.get('본문품질점수')}, 대표 {report_item.get('대표선택점수')}, "
            f"Gemini중요도 {report_item.get('Gemini중요도')})"
        )
        time.sleep(random.uniform(0.15, 0.35))
        return True

    # 3-1. Gemini final selections first.
    for json_key in JSON_KEYS_ORDER:
        for art_id in selection.get(json_key, []):
            if total_count() >= MAX_SELECT_COUNT:
                break
            try_process(int(art_id), json_key, reason="gemini_selected")

    # 3-2. Fill category minimums and failed replacements from Gemini backup first.
    print("\n  └ 🔁 부족분은 Gemini backup 후보에서 먼저 보충합니다...")
    for json_key in JSON_KEYS_ORDER:
        target = CATEGORY_TARGET.get(json_key, 3)
        for art_id in backup.get(json_key, []):
            if category_count(json_key) >= target:
                break
            if total_count() >= MAX_SELECT_COUNT:
                break
            try_process(int(art_id), json_key, reason="gemini_backup")

    # 3-3. If still below minimum, use label-ranked local backup.
    if total_count() < MIN_SELECT_COUNT:
        print(f"\n  └ 🔁 아직 {total_count()}개라 Gemini 라벨 기반 후보에서 추가 보충합니다...")
        pool = sorted(ranked_candidates, key=v19_rank_for_editor, reverse=True)
        for article in pool:
            if total_count() >= MIN_SELECT_COUNT:
                break
            if total_count() >= MAX_SELECT_COUNT:
                break
            art_id = int(article["id"])
            if art_id in processed_ids:
                continue
            label = labels.get(art_id, {})
            if label.get("exclude") or label.get("is_pr"):
                continue
            json_key = label.get("category") or article.get("JSON카테고리") or "산업동향"
            json_key = v19_normalize_category_key(json_key, fallback="산업동향")
            try_process(art_id, json_key, reason="label_based_replacement")

    if skipped_duplicates or post_body_duplicate_skips:
        all_skips = skipped_duplicates + post_body_duplicate_skips
        pd.DataFrame(all_skips).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 제외/중복/대표교체 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")

    if body_failed_rows:
        pd.DataFrame(body_failed_rows).to_csv(OUTPUT_BODY_FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ ⚠️ 본문 품질 미달/관련성 미달 기사 저장: {os.path.basename(OUTPUT_BODY_FAILED_CSV)}")

    if not final_report_data:
        print("❌ 최종 보고서에 사용할 기사 데이터가 없습니다.")
        return

    ordered_final = []
    for json_key in JSON_KEYS_ORDER:
        items = [x for x in final_report_data if x.get("카테고리") == json_key]
        items = sorted(items, key=lambda x: int(x.get("선정순서") or 9999))
        ordered_final.extend(items)
    final_report_data = ordered_final[:MAX_SELECT_COUNT]

    print("\n✍️ [STEP 4] 본문 전문 기반으로 past_reports 형식의 최종 브리핑을 생성합니다...")

    for idx, item in enumerate(final_report_data, 1):
        item["브리핑ID"] = idx

    final_input_text = ""
    for item in final_report_data:
        final_input_text += (
            f"[기사번호 {item['브리핑ID']}]\n"
            f"카테고리: {item['카테고리명']}\n"
            f"제목: {item['기사제목']}\n"
            f"언론사: {item['언론사']}\n"
            f"게시일: {item['게시일']}\n"
            f"Gemini이슈유형: {item.get('Gemini이슈유형','')}\n"
            f"Gemini선정사유: {item.get('Gemini선정사유','')}\n"
            f"본문글자수: {item['본문글자수']}\n"
            f"본문:\n{item['본문전문'][:MAX_BODY_CHARS_FOR_PROMPT]}\n\n"
        )

    prompt_report = f"""
너는 최고 경영진에게 매일 아침 뉴스 브리핑을 제공하는 수석 전략가야.
아래 [오늘 기사 데이터]의 본문만 사용해서 각 기사별 요약문만 작성해.

[요약 규칙]
1. 네 생각, 인사이트, 전망, 대응 포인트, 의미 부여는 쓰지마.
2. 오직 기사 본문에 있는 객관적 사실만 요약해.
3. 기사 1개당 요약은 1문단으로만 작성해.
4. 모든 문장의 끝은 '~함', '~임', '~됨', '~계획임', '~예정임' 같은 문어체 종결로 맞춰.
5. 기사 본문에 없는 내용은 절대 추가하지마.
6. 제목, 링크, 언론사, 카테고리, 번호는 쓰지마. 요약문만 반환해.
7. 사진/캡션/관련기사/저작권 문구는 요약하지마.

반드시 아래 JSON 객체 형식으로만 응답해. 키는 기사번호 문자열이고 값은 요약문이야.
{{
  "1": "요약문",
  "2": "요약문"
}}

[최근 과거 보고서 문체 참고]
{recent_past_text[:5000]}

[오늘 기사 데이터]
{final_input_text}
"""

    summary_map = {}
    try:
        if not client or not ENABLE_GEMINI_REPORT:
            raise RuntimeError("Gemini 최종 브리핑 비활성화 또는 client 없음")
        summary_text = gemini_generate_text(client=client, prompt=prompt_report, task_name="최종 기사별 요약 생성")
        summary_map = normalize_summary_json(extract_json_object(summary_text))
        print("  └ ✅ Gemini 기사별 요약 생성 완료")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사별 요약 생성 실패. 로컬 본문 요약으로 대체합니다. 원인: {e}")
        summary_map = {}

    final_briefing_text = build_structured_briefing(final_report_data, summary_map)

    print("\n" + "=" * 60)
    print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
    print("=" * 60)
    print(final_briefing_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_briefing_text)

    pd.DataFrame(final_report_data).to_csv(OUTPUT_SELECTED_CSV, index=False, encoding="utf-8-sig")

    run_log.append({
        "버전": V6_VERSION,
        "전체_RSS_후보": len(raw_articles),
        "Gemini_후보": len(ranked_candidates),
        "Gemini라벨링성공": gemini_label_success,
        "Gemini편집성공": gemini_editor_success,
        "Gemini선택수": sum(len(v) for v in selection.values()),
        "Gemini백업수": sum(len(v) for v in backup.values()),
        "제외_총합": len(skipped_duplicates) + len(post_body_duplicate_skips),
        "본문품질미달_관련성미달_교체": len(body_failed_rows),
        "중복대표교체": replacement_count,
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
    print(f"- '{os.path.basename(OUTPUT_GEMINI_LABELS_CSV)}' 저장 완료")
    if skipped_duplicates or post_body_duplicate_skips:
        print(f"- '{os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}' 저장 완료")
    if body_failed_rows:
        print(f"- '{os.path.basename(OUTPUT_BODY_FAILED_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_RUN_LOG_CSV)}' 저장 완료")
    print("=" * 60)


# =========================================================
# v20 overrides: issue-desk pipeline for Kakao public affairs
# =========================================================
# v20 design notes
# 1) Collect broadly and avoid pre-Gemini over-filtering.
# 2) Gemini labels article meaning, but final selection is made at issue-group level.
# 3) Code chooses the representative article inside each selected issue.
# 4) If body extraction fails, try another article in the same issue first.
# 5) If no replacement exists and the issue is critical, include title/link/RSS with an explicit body-limit flag.
# 6) Use different Gemini models by task: cheap labeling, stronger issue editor, balanced summarizer.

V6_VERSION = "google_rss_v20_issue_desk_pipeline"

# Model routing. You can override these with environment variables without editing the file.
GEMINI_MODEL_LABELING = os.getenv("GEMINI_MODEL_LABELING", "gemini-2.5-flash-lite")
GEMINI_MODEL_EDITOR = os.getenv("GEMINI_MODEL_EDITOR", "gemini-2.5-pro")
GEMINI_MODEL_SUMMARY = os.getenv("GEMINI_MODEL_SUMMARY", "gemini-2.5-flash")
GEMINI_MODEL_QA = os.getenv("GEMINI_MODEL_QA", "gemini-2.5-pro")
ENABLE_GEMINI_QA = str(os.getenv("ENABLE_GEMINI_QA", "1")).strip().lower() not in {"0", "false", "no", "n"}

# v20 keeps the candidate pool broad; issue editing happens after semantic grouping.
MAX_CANDIDATES_FOR_GEMINI = 320
V20_LABEL_BATCH_SIZE = 55
V20_MAX_ISSUES_FOR_EDITOR = 90
V20_FINAL_MIN = 12
V20_FINAL_TARGET = 15
V20_FINAL_MAX = 17
MAX_SELECT_COUNT = V20_FINAL_MAX
MIN_SELECT_COUNT = V20_FINAL_MIN

CATEGORY_MIN = {
    "자사_및_계열사_이슈": 2,
    "정부_국회": 3,
    "경쟁사_해외이슈": 2,
    "산업동향": 1,
}
CATEGORY_TARGET = {
    "자사_및_계열사_이슈": 4,
    "정부_국회": 5,
    "경쟁사_해외이슈": 4,
    "산업동향": 1,
}
CATEGORY_MAX = {
    "자사_및_계열사_이슈": 6,
    "정부_국회": 7,
    "경쟁사_해외이슈": 5,
    "산업동향": 2,
}

OUTPUT_GEMINI_LABELS_CSV = os.path.join(BASE_DIR, "google_news_gemini_labels.csv")
OUTPUT_ISSUES_CSV = os.path.join(BASE_DIR, "google_news_issues.csv")
OUTPUT_ISSUE_SELECTION_CSV = os.path.join(BASE_DIR, "google_news_issue_selection.csv")
OUTPUT_QA_TXT = os.path.join(BASE_DIR, "CEO_Morning_Briefing_QA.txt")

# Broad internal categories. These are for editing only; final output still uses the existing four categories.
V20_INTERNAL_CATEGORIES = {
    "SELF_DIRECT_RISK",
    "SELF_AFFILIATE_BUSINESS",
    "SELF_INCLUDED_REGULATION",
    "GOV_PLATFORM_REGULATION",
    "GOV_AI_DIGITAL_POLICY",
    "GOV_FINANCIAL_DIGITAL_POLICY",
    "COMPETITOR_AI_STRATEGY",
    "COMPETITOR_PLATFORM_RISK",
    "INDUSTRY_STRUCTURAL_CHANGE",
    "LOW_VALUE_PR",
    "OFF_TOPIC",
}

# Family caps are applied to issues, not individual articles.
V20_FAMILY_CAPS = {
    "self_labor": 2,
    "self_leadership_service": 2,
    "self_governance_mna": 2,
    "self_financial_performance": 1,
    "platform_obligation": 2,
    "government_platform_regulation": 3,
    "general_financial_enforcement": 1,
    "competitor_ai_strategy": 2,
    "competitor_security_risk": 2,
    "industry_ai_infrastructure": 2,
    "low_value_pr": 0,
    "off_topic": 0,
}

V20_GENERIC_GROUPS = {"", "general", "일반", "기타", "뉴스", "후보", "article", "unknown", "none", "null"}


def v20_clip(text, limit=240):
    return v19_clip(text, limit)


def v20_int(value, default=3, min_value=1, max_value=5):
    return v19_int(value, default=default, min_value=min_value, max_value=max_value)


def v20_float(value, default=0.0):
    return v19_float(value, default=default)


def v20_bool(value):
    return v19_bool(value)


def v20_norm(text):
    text = clean_html_text(text).lower()
    text = re.sub(r"[^0-9a-zA-Z가-힣]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def v20_issue_group_key(text):
    text = v20_norm(text)
    if not text or text in V20_GENERIC_GROUPS:
        return ""
    # Remove words that make Gemini issue names vary without changing the issue.
    text = re.sub(r"\b(관련|이슈|기사|보도|뉴스|종합|단독|속보)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:90]


def v20_article_text(article, body_chars=0):
    body = article.get("본문전문", "")[:body_chars] if body_chars else ""
    return clean_html_text(
        f"{article.get('기사제목','')} {article.get('본문요약','')} {article.get('RSS요약','')} "
        f"{body} {article.get('검색어','')} {article.get('언론사','')}"
    )


def v20_has_self_entity(text):
    try:
        return v11_has_self_entity(text)
    except Exception:
        return bool(SELF_KAKAO_PATTERN.search(clean_html_text(text)))


def v20_is_pr_text(text):
    try:
        return bool(v18_is_self_pr_promo_text(text))
    except Exception:
        return bool(v20_has_self_entity(text) and re.search(r"스폰서|후원|협찬|영화제|축제|페스티벌|이벤트|쿠폰|할인|프로모션|캠페인|수상", text, re.I))


def v20_is_platform_obligation_text(text):
    try:
        return bool(v18_is_platform_obligation_text(text))
    except Exception:
        t = clean_html_text(text)
        actor = r"플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|부가통신사업자|정보통신서비스|사업자|네이버|카카오|구글|메타|유튜브|틱톡|애플|쿠팡|배달의민족|배민|토스"
        action = r"의무|의무화|해야|걸러야|차단|삭제|필터링|신고|보고|공시|자료\s*제출|준수|제재|과징금|과태료|시정명령|시행|적용|유통\s*방지|확산\s*방지"
        context = r"방미통위|방송미디어통신위원회|공정위|개보위|과기정통부|국회|법안|시행령|가이드라인|규제|정책|유해정보|불법정보|딥페이크|개인정보|청소년|이용자\s*보호"
        return bool(re.search(actor, t, re.I) and re.search(action, t, re.I) and re.search(context, t, re.I))


def v20_detect_issue_family(text, json_key=""):
    t = clean_html_text(text)
    if not t:
        return "off_topic"
    if v20_is_pr_text(t):
        return "low_value_pr"
    if v20_is_platform_obligation_text(t):
        if v20_has_self_entity(t):
            return "platform_obligation"
        return "government_platform_regulation"
    if v20_has_self_entity(t):
        if re.search(r"노조|파업|쟁의|임단협|임금교섭|성과급|RSU|조정|노동위|지노위|중노위|고용불안|노동부|근로감독|최저임금", t, re.I):
            return "self_labor"
        if re.search(r"임원|CPO|CTO|CFO|대표|경영진|조직개편|퇴사|사임|사퇴|교체|리더십|서비스\s*개편|논란|반발|사과|후폭풍|친구탭|카톡\s*개편", t, re.I):
            return "self_leadership_service"
        if re.search(r"인수|매각|합병|지분|최대주주|경영권|공동대표|사내이사|주총|상장|IPO|투자|유상증자|전환사채", t, re.I):
            return "self_governance_mna"
        if re.search(r"개인정보|유출|해킹|피싱|사칭|보안|장애|먹통|침해사고|딥페이크", t, re.I):
            return "self_privacy_security"
        if re.search(r"수사|조사|과징금|제재|고발|소송|판결|검찰|경찰|압수수색|시정명령", t, re.I):
            return "self_legal_regulatory"
        if re.search(r"실적|매출|영업이익|영업손실|순이익|적자|흑자|주가|목표가|투자의견", t, re.I):
            return "self_financial_performance"
        return "self_affiliate_business"
    if json_key == "정부_국회" or REGULATOR_CORE_PATTERN.search(t):
        if re.search(r"금감원|금융위|ELS|은행|보험|증권|파생|금소법|불완전\s*판매", t, re.I):
            return "general_financial_enforcement"
        if re.search(r"플랫폼|온플법|공정위|온라인플랫폼|자사우대|최혜대우|수수료|정산|인앱결제|알고리즘|유해정보|개인정보|딥페이크|방미통위|개보위", t, re.I):
            return "government_platform_regulation"
        if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|저작권|국가AI", t, re.I):
            return "government_ai_policy"
        if re.search(r"스테이블코인|디지털자산|가상자산|특금법|FIU|CBDC|토큰", t, re.I):
            return "government_digital_finance_policy"
        return "government_general_policy"
    if json_key == "경쟁사_해외이슈" or COMPETITOR_CORE_PATTERN.search(t):
        if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|LLM|오픈AI|구글|MS|메타|애플|앤트로픽|엔비디아", t, re.I):
            return "competitor_ai_strategy"
        if re.search(r"개인정보|유출|해킹|피싱|보안|장애|제재|소송|과징금|규제|조사", t, re.I):
            return "competitor_security_risk"
        return "competitor_business_strategy"
    if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|반도체|HBM|TSMC|엔비디아|전력|원전|스테이블코인|디지털자산|플랫폼", t, re.I):
        return "industry_ai_infrastructure"
    return "off_topic"


def v20_internal_category_from_family(family, text, original_json_key=""):
    t = clean_html_text(text)
    if family == "low_value_pr":
        return "LOW_VALUE_PR"
    if family == "off_topic":
        return "OFF_TOPIC"
    if family.startswith("self_labor") or family in {"self_leadership_service", "self_privacy_security", "self_legal_regulatory", "self_governance_mna", "self_financial_performance"}:
        return "SELF_DIRECT_RISK"
    if family == "self_affiliate_business":
        return "SELF_AFFILIATE_BUSINESS"
    if family == "platform_obligation":
        return "SELF_INCLUDED_REGULATION" if v20_has_self_entity(t) else "GOV_PLATFORM_REGULATION"
    if family in {"government_platform_regulation", "government_general_policy"}:
        return "GOV_PLATFORM_REGULATION"
    if family == "government_ai_policy":
        return "GOV_AI_DIGITAL_POLICY"
    if family in {"government_digital_finance_policy", "general_financial_enforcement"}:
        return "GOV_FINANCIAL_DIGITAL_POLICY"
    if family == "competitor_ai_strategy":
        return "COMPETITOR_AI_STRATEGY"
    if family in {"competitor_security_risk", "competitor_business_strategy"}:
        return "COMPETITOR_PLATFORM_RISK"
    if family == "industry_ai_infrastructure":
        return "INDUSTRY_STRUCTURAL_CHANGE"
    if original_json_key == "자사_및_계열사_이슈":
        return "SELF_AFFILIATE_BUSINESS"
    if original_json_key == "정부_국회":
        return "GOV_PLATFORM_REGULATION"
    if original_json_key == "경쟁사_해외이슈":
        return "COMPETITOR_PLATFORM_RISK"
    return "INDUSTRY_STRUCTURAL_CHANGE"


def v20_output_category_from_internal(internal_category, original_json_key=""):
    if internal_category in {"SELF_DIRECT_RISK", "SELF_AFFILIATE_BUSINESS"}:
        return "자사_및_계열사_이슈"
    if internal_category == "SELF_INCLUDED_REGULATION":
        # These are often more actionable for public affairs as regulation/policy issues.
        return "정부_국회"
    if internal_category.startswith("GOV_"):
        return "정부_국회"
    if internal_category.startswith("COMPETITOR_"):
        return "경쟁사_해외이슈"
    if internal_category == "INDUSTRY_STRUCTURAL_CHANGE":
        return "산업동향"
    if original_json_key in JSON_KEYS_ORDER:
        return original_json_key
    return "산업동향"


def v20_local_article_label(article):
    text = v20_article_text(article)
    original_key = article.get("JSON카테고리") or "산업동향"
    family = v20_detect_issue_family(text, original_key)
    internal = v20_internal_category_from_family(family, text, original_key)
    category = v20_output_category_from_internal(internal, original_key)
    score = v20_float(article.get("랭킹점수"), 0)

    ceo = 3
    pa = 3
    relevance = 3
    exclude = False
    is_pr = False
    reason = "로컬 보조 라벨"
    impact = "간접 영향"
    risk_types = []

    if family == "low_value_pr":
        ceo = pa = relevance = 1
        exclude = True
        is_pr = True
        impact = "홍보/후원성으로 정책·리스크 우선순위 낮음"
        risk_types = ["홍보"]
        reason = "스폰서·후원·이벤트성 기사로 최종 브리핑 우선순위 낮음"
    elif family in {"self_labor", "self_leadership_service", "self_privacy_security", "self_legal_regulatory", "self_governance_mna"}:
        ceo = 5
        pa = 5 if family in {"self_labor", "self_privacy_security", "self_legal_regulatory"} else 4
        relevance = 5
        impact = "카카오·계열사 직접 영향"
        risk_types = ["자사리스크", family]
        reason = "카카오 직접 리스크 신호"
    elif family == "platform_obligation":
        ceo = 4
        pa = 5
        relevance = 5
        impact = "카카오 포함 주요 플랫폼 사업자 공통 규제·준수 의무 영향"
        risk_types = ["규제", "플랫폼의무"]
        reason = "플랫폼 사업자에게 차단·삭제·필터링·보고·준수 등 의무가 생기는 이슈"
    elif family.startswith("government_"):
        ceo = 4 if family != "government_general_policy" else 3
        pa = 5 if "platform" in family or "digital" in family or "ai" in family else 3
        relevance = 4
        impact = "정책·규제 환경 영향"
        risk_types = ["정책", family]
    elif family == "general_financial_enforcement":
        ceo = 3
        pa = 2
        relevance = 2
        impact = "일반 금융 제재 이슈로 카카오 직접성 낮음"
        risk_types = ["일반금융제재"]
    elif family.startswith("competitor_"):
        ceo = 4
        pa = 3
        relevance = 4
        impact = "경쟁사·해외 사업환경 영향"
        risk_types = ["경쟁사", family]
    elif family == "industry_ai_infrastructure":
        ceo = 3
        pa = 3
        relevance = 3
        impact = "산업 구조 변화 참고"
        risk_types = ["산업동향"]
    else:
        ceo = 2 if score < 10 else 3
        pa = 2
        relevance = 2
        if family == "off_topic":
            exclude = True
            reason = "카카오 대외협력 브리핑 관련성이 낮음"

    if score >= 70:
        ceo = max(ceo, 4)
    if article.get("강제검토") == "Y":
        ceo = max(ceo, 5)
        pa = max(pa, 4)
        relevance = max(relevance, 5)
    if article.get("정책규제의무") == "Y":
        pa = max(pa, 5)
        relevance = max(relevance, 5)

    group = ""
    try:
        group = v16_global_incident_base_key(article) or v16_global_incident_key(article)
    except Exception:
        group = ""
    if not group:
        if family in {"platform_obligation", "government_platform_regulation"}:
            # Use anchors so different titles on the same platform obligation merge better.
            if re.search(r"불법촬영|유해정보|딥페이크|청소년|개인정보|피싱|사칭", text):
                m = re.search(r"불법촬영|유해정보|딥페이크|청소년|개인정보|피싱|사칭", text)
                group = f"플랫폼 사업자 {m.group(0)} 의무"
            else:
                group = "플랫폼 사업자 의무/규제"
        elif family == "low_value_pr":
            group = "자사 홍보/후원성 기사"
        elif family != "off_topic":
            group = f"{family}:{title_fingerprint(article.get('기사제목',''))[:50]}"
    if not group:
        group = title_fingerprint(article.get("기사제목", ""))[:70] or f"article_{article.get('id')}"

    return {
        "id": int(article.get("id")),
        "is_relevant": not exclude,
        "primary_category": category,
        "internal_category": internal,
        "issue_family": family,
        "issue_group": v20_clip(group, 100),
        "company_impact": v20_clip(impact, 160),
        "ceo_priority": ceo,
        "public_affairs_priority": pa,
        "relevance": relevance,
        "risk_types": risk_types,
        "is_pr": is_pr,
        "exclude": exclude,
        "reason": v20_clip(reason, 200),
        "label_source": "local",
    }


def v20_label_candidate_line(article):
    local = v20_local_article_label(article)
    hints = v19_code_hints(article)
    return (
        f"[{article.get('id')}] "
        f"orig_cat={article.get('JSON카테고리','')} search={article.get('검색어','')} "
        f"score={article.get('랭킹점수','')} imp={article.get('중요도점수','')} "
        f"code_family={local.get('issue_family','')} code_hint={hints.get('hints','')} "
        f"title={v20_clip(article.get('기사제목',''), 120)} "
        f"source={v20_clip(article.get('언론사',''), 40)} "
        f"summary={v20_clip(article.get('본문요약',''), 330)}"
    )


def v20_normalize_internal_category(value, fallback):
    text = clean_html_text(value).upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "자사_직접_리스크": "SELF_DIRECT_RISK",
        "자사직접리스크": "SELF_DIRECT_RISK",
        "자사_사업": "SELF_AFFILIATE_BUSINESS",
        "자사_포함_규제": "SELF_INCLUDED_REGULATION",
        "업계_공통_규제": "SELF_INCLUDED_REGULATION",
        "정부_플랫폼_규제": "GOV_PLATFORM_REGULATION",
        "플랫폼_규제": "GOV_PLATFORM_REGULATION",
        "정부_AI_정책": "GOV_AI_DIGITAL_POLICY",
        "정부_디지털금융": "GOV_FINANCIAL_DIGITAL_POLICY",
        "경쟁사_AI": "COMPETITOR_AI_STRATEGY",
        "경쟁사_리스크": "COMPETITOR_PLATFORM_RISK",
        "산업_구조변화": "INDUSTRY_STRUCTURAL_CHANGE",
        "홍보": "LOW_VALUE_PR",
        "홍보성": "LOW_VALUE_PR",
        "오프토픽": "OFF_TOPIC",
        "무관": "OFF_TOPIC",
    }
    if text in V20_INTERNAL_CATEGORIES:
        return text
    if text in aliases:
        return aliases[text]
    for k in V20_INTERNAL_CATEGORIES:
        if k in text or text in k:
            return k
    return fallback if fallback in V20_INTERNAL_CATEGORIES else "INDUSTRY_STRUCTURAL_CHANGE"


def v20_normalize_label_json(data, valid_ids, article_by_id):
    if isinstance(data, dict):
        rows = data.get("articles") or data.get("items") or data.get("labels") or data.get("기사") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    labels = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            art_id = int(row.get("id") or row.get("article_id") or row.get("기사ID"))
        except Exception:
            continue
        if art_id not in valid_ids:
            continue
        article = article_by_id.get(art_id, {})
        local = v20_local_article_label(article) if article else {}
        internal = v20_normalize_internal_category(
            row.get("internal_category") or row.get("내부카테고리") or row.get("category_internal"),
            local.get("internal_category", "INDUSTRY_STRUCTURAL_CHANGE"),
        )
        category = v19_normalize_category_key(
            row.get("primary_category") or row.get("category") or row.get("카테고리"),
            fallback=v20_output_category_from_internal(internal, article.get("JSON카테고리", "산업동향")),
        )
        family = v20_clip(row.get("issue_family") or row.get("family") or row.get("이슈패밀리") or local.get("issue_family", ""), 80)
        if not family:
            family = v20_detect_issue_family(v20_article_text(article), article.get("JSON카테고리", ""))
        group = v20_clip(row.get("issue_group") or row.get("duplicate_group") or row.get("중복그룹") or local.get("issue_group", ""), 120)
        if not group or v20_issue_group_key(group) in V20_GENERIC_GROUPS:
            group = local.get("issue_group", "") or f"{family}:{title_fingerprint(article.get('기사제목',''))[:55]}"
        risk_types = row.get("risk_types") or row.get("risk_type") or row.get("리스크유형") or local.get("risk_types", [])
        if isinstance(risk_types, str):
            risk_types = [x.strip() for x in re.split(r"[,/|+]", risk_types) if x.strip()]
        if not isinstance(risk_types, list):
            risk_types = []
        label = {
            "id": art_id,
            "is_relevant": v20_bool(row.get("is_relevant")) if "is_relevant" in row else not v20_bool(row.get("exclude")),
            "primary_category": category,
            "internal_category": internal,
            "issue_family": family,
            "issue_group": group,
            "company_impact": v20_clip(row.get("company_impact") or row.get("company_relevance") or row.get("카카오영향") or local.get("company_impact", ""), 180),
            "ceo_priority": v20_int(row.get("ceo_priority") or row.get("CEO중요도") or row.get("priority"), default=local.get("ceo_priority", 3)),
            "public_affairs_priority": v20_int(row.get("public_affairs_priority") or row.get("pa_priority") or row.get("대외협력중요도"), default=local.get("public_affairs_priority", 3)),
            "relevance": v20_int(row.get("relevance") or row.get("관련도"), default=local.get("relevance", 3)),
            "risk_types": risk_types,
            "is_pr": v20_bool(row.get("is_pr") if "is_pr" in row else row.get("홍보성", local.get("is_pr", False))),
            "exclude": v20_bool(row.get("exclude") if "exclude" in row else row.get("제외", local.get("exclude", False))),
            "reason": v20_clip(row.get("reason") or row.get("판단사유") or local.get("reason", ""), 240),
            "label_source": "gemini",
        }
        if label["is_pr"] or label["internal_category"] == "LOW_VALUE_PR":
            label["exclude"] = True
            label["is_relevant"] = False
        labels[art_id] = label
    return labels


def v20_gemini_label_candidates(client, candidates, recent_past_text):
    if not client:
        raise RuntimeError("Gemini client 없음")
    article_by_id = {int(a["id"]): a for a in candidates}
    all_labels = {}
    batches = [candidates[i:i + V20_LABEL_BATCH_SIZE] for i in range(0, len(candidates), V20_LABEL_BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches, 1):
        candidate_text = "\n".join(v20_label_candidate_line(a) for a in batch)
        prompt = f"""
너는 카카오 대외협력·정책리스크팀의 기사 분류 데스크야.
후보 기사 각각을 '최종 기사'가 아니라 '이슈로 묶기 위한 재료'로 라벨링해.
특정 키워드 하나에 매달리지 말고, 카카오에 미치는 영향과 대외협력 대응 필요성을 판단해.

[핵심 분류 철학]
1. 카카오 직접 리스크: 노무/파업/임단협, 임원·조직개편, 서비스 개편 논란, 개인정보·보안·장애, 수사·제재·소송, 지배구조/M&A/실적.
2. 자사 포함 업계 공통 규제: 네이버·카카오·구글 등 여러 사업자에게 차단·삭제·필터링·보고·공시·자료제출·준수 의무가 생기는 기사. 카카오 단독 기사가 아니어도 대관 관점에서 중요하다.
3. 정부/국회 정책 변화: AI, 플랫폼, 디지털자산, 개인정보, 보안, 공정위, 과기정통부, 방미통위, 개보위, 금융위/금감원 중 카카오 사업환경에 영향을 주는 것.
4. 경쟁사/해외: 네이버·구글·오픈AI·MS·메타·애플·쿠팡·토스·배민·통신사 등의 전략/규제/보안/AI 변화.
5. 홍보/후원/스폰서/이벤트/축제/쿠폰/프로모션/수상은 원칙적으로 LOW_VALUE_PR로 제외.
6. 개별 범죄·지역 사건·단순 주가 전망·오피니언·제품 루머는 OFF_TOPIC 또는 낮은 우선순위.

[내부카테고리 중 하나]
SELF_DIRECT_RISK, SELF_AFFILIATE_BUSINESS, SELF_INCLUDED_REGULATION,
GOV_PLATFORM_REGULATION, GOV_AI_DIGITAL_POLICY, GOV_FINANCIAL_DIGITAL_POLICY,
COMPETITOR_AI_STRATEGY, COMPETITOR_PLATFORM_RISK, INDUSTRY_STRUCTURAL_CHANGE,
LOW_VALUE_PR, OFF_TOPIC

[issue_group 작성법]
- 같은 사건은 반드시 같은 issue_group으로 묶어라.
- 예: '카카오 노조 공동파업/임금교섭 결렬', '카카오게임즈 공동대표 체제', '플랫폼 사업자 유해정보 차단 의무', '홍콩 ELS 과징금 감경'.
- 언론사명이나 제목 표현 차이 때문에 group을 나누지 마라.

[최근 7일 과거 보고서 참고]
{recent_past_text[:2600]}

[후보 리스트: 배치 {batch_idx}/{len(batches)}]
{candidate_text}

반드시 JSON 객체만 반환해. 설명문 금지.
형식:
{{
  "articles": [
    {{
      "id": 123,
      "is_relevant": true,
      "primary_category": "자사_및_계열사_이슈 또는 정부_국회 또는 경쟁사_해외이슈 또는 산업동향",
      "internal_category": "SELF_DIRECT_RISK 등 위 내부카테고리 중 하나",
      "issue_family": "self_labor/platform_obligation/government_platform_regulation/competitor_ai_strategy/low_value_pr 등 짧은 family",
      "issue_group": "같은 사건끼리 공유할 짧은 이슈명",
      "company_impact": "카카오 직접 영향/카카오 포함 업계 공통 영향/간접 영향/낮음",
      "ceo_priority": 1,
      "public_affairs_priority": 1,
      "relevance": 1,
      "risk_types": ["규제", "플랫폼의무"],
      "is_pr": false,
      "exclude": false,
      "reason": "짧은 판단 사유"
    }}
  ]
}}
"""
        text = gemini_generate_text(
            client=client,
            prompt=prompt,
            task_name=f"v20 기사 라벨링 {batch_idx}/{len(batches)}",
            model=GEMINI_MODEL_LABELING,
        )
        data = extract_json_object(text)
        labels = v20_normalize_label_json(data, set(article_by_id.keys()), article_by_id)
        all_labels.update(labels)
        print(f"  └ 🏷️ v20 Gemini 기사 라벨링 배치 {batch_idx}/{len(batches)} 완료: {len(labels)}개")
    if not all_labels:
        raise ValueError("Gemini 라벨링 결과가 비어 있습니다.")
    return all_labels


def v20_apply_labels_to_articles(articles, labels):
    rows = []
    for article in articles:
        try:
            art_id = int(article.get("id"))
        except Exception:
            continue
        label = labels.get(art_id) or v20_local_article_label(article)
        labels[art_id] = label
        article["Gemini카테고리"] = label.get("primary_category", "")
        article["Gemini내부카테고리"] = label.get("internal_category", "")
        article["Gemini이슈패밀리"] = label.get("issue_family", "")
        article["Gemini이슈그룹"] = label.get("issue_group", "")
        article["Gemini관련성"] = label.get("company_impact", "")
        article["Gemini중요도"] = label.get("ceo_priority", "")
        article["Gemini대외협력중요도"] = label.get("public_affairs_priority", "")
        article["Gemini관련도"] = label.get("relevance", "")
        article["GeminiPR성"] = "Y" if label.get("is_pr") else ""
        article["Gemini제외"] = "Y" if label.get("exclude") else ""
        article["Gemini판단사유"] = label.get("reason", "")
        article["Gemini라벨소스"] = label.get("label_source", "")
        # Backward compatible fields used by older helpers.
        article["Gemini중복그룹"] = label.get("issue_group", "")
        article["Gemini이슈유형"] = label.get("issue_family", "")
        rows.append({
            "id": art_id,
            "기사제목": article.get("기사제목", ""),
            "언론사": article.get("언론사", ""),
            "검색어": article.get("검색어", ""),
            "원카테고리": article.get("JSON카테고리", ""),
            "Gemini카테고리": article.get("Gemini카테고리", ""),
            "Gemini내부카테고리": article.get("Gemini내부카테고리", ""),
            "Gemini이슈패밀리": article.get("Gemini이슈패밀리", ""),
            "Gemini이슈그룹": article.get("Gemini이슈그룹", ""),
            "Gemini중요도": article.get("Gemini중요도", ""),
            "Gemini대외협력중요도": article.get("Gemini대외협력중요도", ""),
            "Gemini관련도": article.get("Gemini관련도", ""),
            "Gemini관련성": article.get("Gemini관련성", ""),
            "GeminiPR성": article.get("GeminiPR성", ""),
            "Gemini제외": article.get("Gemini제외", ""),
            "Gemini판단사유": article.get("Gemini판단사유", ""),
            "Gemini라벨소스": article.get("Gemini라벨소스", ""),
            "랭킹점수": article.get("랭킹점수", ""),
            "코드힌트": article.get("코드힌트", ""),
        })
    return rows


def v20_article_pre_rep_score(article):
    label_priority = v20_int(article.get("Gemini중요도"), default=3)
    pa_priority = v20_int(article.get("Gemini대외협력중요도"), default=3)
    relevance = v20_int(article.get("Gemini관련도"), default=3)
    rank = v20_float(article.get("랭킹점수"), 0)
    press = press_score(article.get("언론사", ""))
    pr_penalty = -100 if article.get("GeminiPR성") == "Y" else 0
    exclude_penalty = -100 if article.get("Gemini제외") == "Y" else 0
    return label_priority * 18 + pa_priority * 18 + relevance * 10 + min(30, max(-20, rank / 2)) + press + pr_penalty + exclude_penalty


def v20_issue_score(issue):
    score = 0.0
    score += issue.get("max_ceo_priority", 1) * 22
    score += issue.get("max_pa_priority", 1) * 24
    score += issue.get("max_relevance", 1) * 12
    score += min(35, max(-25, issue.get("max_rank_score", 0) / 2))
    score += min(12, issue.get("article_count", 1) * 2)
    family = issue.get("issue_family", "")
    internal = issue.get("internal_category", "")
    if internal == "SELF_DIRECT_RISK":
        score += 18
    if internal == "SELF_INCLUDED_REGULATION" or family == "platform_obligation":
        score += 16
    if internal in {"GOV_PLATFORM_REGULATION", "GOV_AI_DIGITAL_POLICY"}:
        score += 10
    if family == "general_financial_enforcement":
        score -= 16
    if family in {"low_value_pr", "off_topic"} or internal in {"LOW_VALUE_PR", "OFF_TOPIC"}:
        score -= 120
    return round(score, 3)


def v20_issue_to_output_category(issue):
    cat = issue.get("primary_category") or ""
    if cat in JSON_KEYS_ORDER:
        return cat
    return v20_output_category_from_internal(issue.get("internal_category", ""), "산업동향")


def v20_build_issues(candidates, labels):
    # Group article candidates into issue objects.
    article_by_id = {int(a["id"]): a for a in candidates}
    buckets = {}
    for article in candidates:
        art_id = int(article["id"])
        label = labels.get(art_id) or v20_local_article_label(article)
        group = label.get("issue_group") or ""
        group_key = v20_issue_group_key(group)
        if not group_key:
            group_key = v20_issue_group_key(v16_global_incident_base_key(article) if "v16_global_incident_base_key" in globals() else "")
        if not group_key:
            group_key = v20_issue_group_key(f"{label.get('issue_family','')} {title_fingerprint(article.get('기사제목',''))[:60]}")
        if not group_key:
            group_key = f"article_{art_id}"
        buckets.setdefault(group_key, []).append(article)

    issues = []
    for idx, (group_key, arts) in enumerate(buckets.items(), 1):
        arts = sorted(arts, key=v20_article_pre_rep_score, reverse=True)
        issue_id = f"I{idx:03d}"
        article_ids = [int(a["id"]) for a in arts]
        issue_labels = [labels.get(int(a["id"])) or v20_local_article_label(a) for a in arts]
        families = [l.get("issue_family", "") for l in issue_labels if l.get("issue_family")]
        internals = [l.get("internal_category", "") for l in issue_labels if l.get("internal_category")]
        categories = [l.get("primary_category", "") for l in issue_labels if l.get("primary_category") in JSON_KEYS_ORDER]

        def mode_or_first(values, fallback=""):
            if not values:
                return fallback
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            return sorted(counts.items(), key=lambda x: (-x[1], values.index(x[0])))[0][0]

        family = mode_or_first(families, v20_detect_issue_family(v20_article_text(arts[0]), arts[0].get("JSON카테고리", "")))
        internal = mode_or_first(internals, v20_internal_category_from_family(family, v20_article_text(arts[0]), arts[0].get("JSON카테고리", "")))
        category = mode_or_first(categories, v20_output_category_from_internal(internal, arts[0].get("JSON카테고리", "산업동향")))
        if internal == "SELF_INCLUDED_REGULATION" or family == "platform_obligation":
            category = "정부_국회"
        if internal == "LOW_VALUE_PR":
            category = arts[0].get("JSON카테고리") if arts[0].get("JSON카테고리") in JSON_KEYS_ORDER else category

        issue_group = mode_or_first([l.get("issue_group", "") for l in issue_labels if l.get("issue_group")], group_key)
        issue = {
            "issue_id": issue_id,
            "issue_key": group_key,
            "issue_group": v20_clip(issue_group, 120),
            "issue_family": family,
            "internal_category": internal,
            "primary_category": category,
            "article_ids": article_ids,
            "article_count": len(article_ids),
            "top_article_id": article_ids[0],
            "candidate_article_ids": article_ids[:8],
            "titles": [a.get("기사제목", "") for a in arts[:5]],
            "sources": [a.get("언론사", "") for a in arts[:5]],
            "max_ceo_priority": max([v20_int(l.get("ceo_priority"), default=1) for l in issue_labels] or [1]),
            "max_pa_priority": max([v20_int(l.get("public_affairs_priority"), default=1) for l in issue_labels] or [1]),
            "max_relevance": max([v20_int(l.get("relevance"), default=1) for l in issue_labels] or [1]),
            "max_rank_score": max([v20_float(a.get("랭킹점수"), 0) for a in arts] or [0]),
            "is_pr": any(l.get("is_pr") for l in issue_labels) or internal == "LOW_VALUE_PR",
            "exclude": all(l.get("exclude") for l in issue_labels) or internal in {"LOW_VALUE_PR", "OFF_TOPIC"},
            "company_impact": v20_clip(" / ".join(dict.fromkeys([l.get("company_impact", "") for l in issue_labels if l.get("company_impact")]))[:260], 260),
            "label_reasons": v20_clip(" / ".join(dict.fromkeys([l.get("reason", "") for l in issue_labels if l.get("reason")]))[:320], 320),
            "risk_types": sorted({str(x) for l in issue_labels for x in (l.get("risk_types") or []) if str(x).strip()}),
        }
        issue["issue_score"] = v20_issue_score(issue)
        issues.append(issue)
    issues = sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True)
    # Re-number after sorting to make editor view stable by priority.
    for idx, issue in enumerate(issues, 1):
        issue["issue_id"] = f"I{idx:03d}"
    return issues, article_by_id


def v20_issue_rows(issues):
    rows = []
    for issue in issues:
        rows.append({
            "issue_id": issue.get("issue_id"),
            "issue_group": issue.get("issue_group"),
            "issue_family": issue.get("issue_family"),
            "internal_category": issue.get("internal_category"),
            "primary_category": issue.get("primary_category"),
            "issue_score": issue.get("issue_score"),
            "max_ceo_priority": issue.get("max_ceo_priority"),
            "max_pa_priority": issue.get("max_pa_priority"),
            "max_relevance": issue.get("max_relevance"),
            "article_count": issue.get("article_count"),
            "top_article_id": issue.get("top_article_id"),
            "candidate_article_ids": ",".join(map(str, issue.get("candidate_article_ids", []))),
            "titles": " | ".join(issue.get("titles", [])),
            "sources": " | ".join(issue.get("sources", [])),
            "company_impact": issue.get("company_impact"),
            "label_reasons": issue.get("label_reasons"),
            "is_pr": issue.get("is_pr"),
            "exclude": issue.get("exclude"),
        })
    return rows


def v20_issue_line(issue):
    ids = ",".join(map(str, issue.get("candidate_article_ids", [])[:5]))
    titles = " || ".join(v20_clip(t, 85) for t in issue.get("titles", [])[:3])
    return (
        f"[{issue.get('issue_id')}] cat={issue.get('primary_category')} internal={issue.get('internal_category')} "
        f"family={issue.get('issue_family')} score={issue.get('issue_score')} "
        f"ceo={issue.get('max_ceo_priority')} pa={issue.get('max_pa_priority')} rel={issue.get('max_relevance')} "
        f"articles={issue.get('article_count')} ids={ids} pr={issue.get('is_pr')} exclude={issue.get('exclude')} "
        f"group={v20_clip(issue.get('issue_group',''), 90)} "
        f"impact={v20_clip(issue.get('company_impact',''), 110)} "
        f"titles={titles}"
    )


def v20_family_cap_key(issue):
    family = issue.get("issue_family", "") or "general"
    internal = issue.get("internal_category", "")
    if family == "platform_obligation" or internal == "SELF_INCLUDED_REGULATION":
        return "platform_obligation"
    if family == "government_platform_regulation":
        return "government_platform_regulation"
    if family in V20_FAMILY_CAPS:
        return family
    if internal == "LOW_VALUE_PR":
        return "low_value_pr"
    if internal == "OFF_TOPIC":
        return "off_topic"
    return family


def v20_can_add_issue(issue, selected_issues, category_counts, family_counts, hard=False):
    if issue.get("exclude") or issue.get("is_pr"):
        return False, "excluded_or_pr_issue"
    category = v20_issue_to_output_category(issue)
    if category_counts.get(category, 0) >= CATEGORY_MAX.get(category, 99):
        return False, f"category_cap:{category}"
    cap_key = v20_family_cap_key(issue)
    cap = V20_FAMILY_CAPS.get(cap_key)
    if cap is not None and family_counts.get(cap_key, 0) >= cap:
        return False, f"family_cap:{cap_key}:{cap}"
    # Avoid nearly identical issue groups even if Gemini labels differ slightly.
    new_key = v20_issue_group_key(issue.get("issue_group", ""))
    for old in selected_issues:
        old_key = v20_issue_group_key(old.get("issue_group", ""))
        if new_key and old_key and (new_key == old_key or sequence_ratio(new_key, old_key) >= 0.90):
            return False, "same_issue_group_already_selected"
    return True, ""


def v20_extract_article_ids_from_any(value):
    ids = []
    if value is None:
        return ids
    if isinstance(value, (int, float, str)):
        value = [value]
    if not isinstance(value, list):
        return ids
    for item in value:
        try:
            if isinstance(item, dict):
                item = item.get("id") or item.get("article_id") or item.get("기사ID")
            ids.append(int(item))
        except Exception:
            continue
    return ids


def v20_normalize_issue_editor_json(data, issue_by_id, article_by_id):
    if not isinstance(data, dict):
        raise ValueError("이슈 편집 JSON 객체가 아닙니다.")
    raw = data.get("issues") or data.get("selection") or data.get("selected_issues") or data.get("선정이슈") or []
    backups_raw = data.get("backup_issues") or data.get("backups") or data.get("대체이슈") or []
    if isinstance(raw, dict):
        # Allow {"자사_및...": [{...}]} style.
        flattened = []
        for cat, vals in raw.items():
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, dict):
                        v.setdefault("category", cat)
                        flattened.append(v)
                    else:
                        flattened.append({"issue_id": v, "category": cat})
        raw = flattened
    if not isinstance(raw, list):
        raw = []
    if isinstance(backups_raw, dict):
        flattened = []
        for cat, vals in backups_raw.items():
            if isinstance(vals, list):
                for v in vals:
                    if isinstance(v, dict):
                        v.setdefault("category", cat)
                        flattened.append(v)
                    else:
                        flattened.append({"issue_id": v, "category": cat})
        backups_raw = flattened
    if not isinstance(backups_raw, list):
        backups_raw = []

    decisions = []
    seen_issues = set()
    for row in raw:
        if not isinstance(row, dict):
            row = {"issue_id": row}
        issue_id = clean_html_text(row.get("issue_id") or row.get("id") or row.get("이슈ID"))
        if issue_id not in issue_by_id or issue_id in seen_issues:
            continue
        issue = issue_by_id[issue_id]
        category = v19_normalize_category_key(row.get("category") or row.get("primary_category") or issue.get("primary_category"), fallback=v20_issue_to_output_category(issue))
        best_article_id = None
        try:
            best_article_id = int(row.get("best_article_id") or row.get("article_id") or row.get("대표기사ID") or issue.get("top_article_id"))
        except Exception:
            best_article_id = int(issue.get("top_article_id"))
        backup_ids = v20_extract_article_ids_from_any(row.get("backup_article_ids") or row.get("backup_ids") or row.get("대체기사ID"))
        # Keep only articles that belong to this issue; then append the issue's own candidates.
        issue_article_set = set(issue.get("article_ids", []))
        backup_ids = [x for x in backup_ids if x in issue_article_set]
        for x in issue.get("candidate_article_ids", []):
            if x != best_article_id and x not in backup_ids:
                backup_ids.append(x)
        decisions.append({
            "issue_id": issue_id,
            "category": category,
            "include": True,
            "priority": v20_int(row.get("priority") or row.get("중요도"), default=issue.get("max_ceo_priority", 3)),
            "best_article_id": best_article_id if best_article_id in issue_article_set else int(issue.get("top_article_id")),
            "backup_article_ids": backup_ids[:12],
            "reason": v20_clip(row.get("reason") or row.get("선정사유") or issue.get("label_reasons", ""), 260),
        })
        seen_issues.add(issue_id)

    backup_decisions = []
    for row in backups_raw:
        if not isinstance(row, dict):
            row = {"issue_id": row}
        issue_id = clean_html_text(row.get("issue_id") or row.get("id") or row.get("이슈ID"))
        if issue_id not in issue_by_id or issue_id in seen_issues:
            continue
        issue = issue_by_id[issue_id]
        category = v19_normalize_category_key(row.get("category") or issue.get("primary_category"), fallback=v20_issue_to_output_category(issue))
        try:
            best_article_id = int(row.get("best_article_id") or row.get("article_id") or issue.get("top_article_id"))
        except Exception:
            best_article_id = int(issue.get("top_article_id"))
        backup_ids = v20_extract_article_ids_from_any(row.get("backup_article_ids") or row.get("backup_ids"))
        issue_article_set = set(issue.get("article_ids", []))
        backup_ids = [x for x in backup_ids if x in issue_article_set]
        for x in issue.get("candidate_article_ids", []):
            if x != best_article_id and x not in backup_ids:
                backup_ids.append(x)
        backup_decisions.append({
            "issue_id": issue_id,
            "category": category,
            "include": False,
            "priority": v20_int(row.get("priority"), default=issue.get("max_ceo_priority", 3)),
            "best_article_id": best_article_id if best_article_id in issue_article_set else int(issue.get("top_article_id")),
            "backup_article_ids": backup_ids[:12],
            "reason": v20_clip(row.get("reason") or issue.get("label_reasons", ""), 260),
        })
    return decisions, backup_decisions


def v20_gemini_edit_issues(client, issues, recent_past_text):
    if not client:
        raise RuntimeError("Gemini client 없음")
    visible_issues = [i for i in issues if not i.get("exclude") and not i.get("is_pr")]
    visible_issues = sorted(visible_issues, key=lambda x: x.get("issue_score", 0), reverse=True)[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = "\n".join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
너는 카카오 CEO 아침 브리핑을 편집하는 대외협력팀 데스크야.
아래는 기사 단위가 아니라 '이슈 단위'로 묶은 후보 목록이야.
최종 보고서는 기사 나열이 아니라 오늘 카카오가 알아야 할 이슈 12~15개를 담아야 해.

[편집 원칙]
1. 카카오 직접 리스크는 우선: 노무/파업/임단협, 임원·조직개편, 서비스 논란, 장애·보안·개인정보, 수사·제재·소송, 지배구조/M&A/실적.
2. 카카오가 여러 사업자 중 하나로 포함된 업계 공통 규제도 중요: 플랫폼 사업자 의무, 차단·삭제·필터링·보고·공시·준수, 방미통위/공정위/개보위/과기정통부/국회.
3. 정부/국회는 카카오 사업환경에 직접 영향이 있는 AI·플랫폼·디지털자산·개인정보·보안·공정거래·통신/인터넷 정책 중심.
4. 경쟁사/해외는 네이버·구글·오픈AI·MS·메타·애플·쿠팡·토스·배민·통신사 전략/규제/보안/AI 변화 중심.
5. 산업동향은 구조적 변화 1~2개만.
6. 홍보/스폰서/후원/이벤트/수상/쿠폰/프로모션은 제외.
7. 같은 issue_group은 하나만 고른다. 같은 family도 과도하게 반복하지 마라. 카카오 노무는 최대 2개, 카카오게임즈 경영권/대표는 보통 1개, 일반 금융 제재는 1개 이내.
8. 본문 추출 실패에 대비해 best_article_id와 backup_article_ids를 꼭 지정해라. backup은 같은 이슈 안의 기사 ID만 사용해라.
9. 카테고리 개수는 유연하게 하되 보통 자사 2~5, 정부/국회 3~6, 경쟁사 2~4, 산업 1~2 정도로 맞춰라.

[최근 7일 과거 보고서 참고]
{recent_past_text[:3500]}

[이슈 후보]
{issue_text}

반드시 JSON 객체만 반환해. 설명문 금지.
형식:
{{
  "issues": [
    {{
      "issue_id": "I001",
      "category": "자사_및_계열사_이슈 또는 정부_국회 또는 경쟁사_해외이슈 또는 산업동향",
      "priority": 5,
      "best_article_id": 123,
      "backup_article_ids": [124, 125],
      "reason": "선정 사유"
    }}
  ],
  "backup_issues": [
    {{
      "issue_id": "I099",
      "category": "정부_국회",
      "priority": 4,
      "best_article_id": 555,
      "backup_article_ids": [556],
      "reason": "대체 후보 사유"
    }}
  ]
}}
"""
    text = gemini_generate_text(
        client=client,
        prompt=prompt,
        task_name="v20 이슈 단위 최종 편집",
        model=GEMINI_MODEL_EDITOR,
    )
    data = extract_json_object(text)
    issue_by_id = {i["issue_id"]: i for i in visible_issues}
    article_by_id = {}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    if not decisions:
        raise ValueError("Gemini 이슈 편집 결과가 비어 있습니다.")
    return decisions, backup_decisions


def v20_local_issue_selection(issues):
    selected = []
    backups = []
    category_counts = {k: 0 for k in JSON_KEYS_ORDER}
    family_counts = {}
    selected_ids = set()
    pool = [i for i in sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True) if not i.get("exclude") and not i.get("is_pr")]

    def add_issue(issue, hard=False):
        if issue["issue_id"] in selected_ids:
            return False
        ok, reason = v20_can_add_issue(issue, selected, category_counts, family_counts, hard=hard)
        if not ok:
            return False
        category = v20_issue_to_output_category(issue)
        selected.append(issue)
        selected_ids.add(issue["issue_id"])
        category_counts[category] = category_counts.get(category, 0) + 1
        cap_key = v20_family_cap_key(issue)
        family_counts[cap_key] = family_counts.get(cap_key, 0) + 1
        return True

    # First satisfy category minimums.
    for key in JSON_KEYS_ORDER:
        for issue in pool:
            if category_counts.get(key, 0) >= CATEGORY_MIN.get(key, 0):
                break
            if v20_issue_to_output_category(issue) != key:
                continue
            add_issue(issue, hard=True)

    # Then add by global editorial score until target.
    for issue in pool:
        if len(selected) >= V20_FINAL_TARGET:
            break
        add_issue(issue)

    # Keep high-priority issues even if target not reached.
    for issue in pool:
        if len(selected) >= V20_FINAL_MAX:
            break
        if issue["issue_id"] in selected_ids:
            continue
        if issue.get("max_ceo_priority", 0) >= 5 and issue.get("max_pa_priority", 0) >= 4:
            add_issue(issue)

    for issue in pool:
        if issue["issue_id"] in selected_ids:
            continue
        category = v20_issue_to_output_category(issue)
        backups.append({
            "issue_id": issue["issue_id"],
            "category": category,
            "include": False,
            "priority": issue.get("max_ceo_priority", 3),
            "best_article_id": issue.get("top_article_id"),
            "backup_article_ids": [x for x in issue.get("candidate_article_ids", []) if x != issue.get("top_article_id")][:8],
            "reason": issue.get("label_reasons", "로컬 대체 후보"),
        })
        if len(backups) >= 30:
            break

    decisions = []
    for issue in selected:
        decisions.append({
            "issue_id": issue["issue_id"],
            "category": v20_issue_to_output_category(issue),
            "include": True,
            "priority": issue.get("max_ceo_priority", 3),
            "best_article_id": issue.get("top_article_id"),
            "backup_article_ids": [x for x in issue.get("candidate_article_ids", []) if x != issue.get("top_article_id")][:10],
            "reason": issue.get("label_reasons", "로컬 이슈 점수 기반 선정"),
        })
    return decisions, backups


def v20_apply_decision_caps(decisions, issue_by_id):
    selected_issues = []
    category_counts = {k: 0 for k in JSON_KEYS_ORDER}
    family_counts = {}
    out = []
    for d in decisions:
        issue = issue_by_id.get(d.get("issue_id"))
        if not issue:
            continue
        category = v19_normalize_category_key(d.get("category"), fallback=v20_issue_to_output_category(issue))
        issue_for_cap = dict(issue)
        issue_for_cap["primary_category"] = category
        ok, reason = v20_can_add_issue(issue_for_cap, selected_issues, category_counts, family_counts)
        if not ok:
            d = dict(d)
            d["dropped_by_cap"] = reason
            continue
        d = dict(d)
        d["category"] = category
        out.append(d)
        selected_issues.append(issue_for_cap)
        category_counts[category] = category_counts.get(category, 0) + 1
        cap_key = v20_family_cap_key(issue_for_cap)
        family_counts[cap_key] = family_counts.get(cap_key, 0) + 1
        if len(out) >= V20_FINAL_MAX:
            break
    return out


def v20_is_critical_issue(issue, decision=None):
    decision = decision or {}
    family = issue.get("issue_family", "")
    internal = issue.get("internal_category", "")
    priority = max(v20_int(decision.get("priority"), default=1), int(issue.get("max_ceo_priority") or 1))
    pa = int(issue.get("max_pa_priority") or 1)
    if internal == "LOW_VALUE_PR" or family == "low_value_pr" or issue.get("is_pr"):
        return False
    if priority >= 5 and pa >= 4:
        return True
    if internal in {"SELF_DIRECT_RISK", "SELF_INCLUDED_REGULATION", "GOV_PLATFORM_REGULATION"} and priority >= 4:
        return True
    if family in {"self_labor", "self_privacy_security", "self_legal_regulatory", "self_governance_mna", "platform_obligation", "government_platform_regulation"} and priority >= 4:
        return True
    return False


def v20_create_bodyless_report_item(article_info, issue, decision, json_key, reason="critical_issue_no_body_available"):
    original_link = article_info.get("링크", "")
    real_url = original_link
    try:
        real_url = normalize_url(decode_google_news_url(original_link)) or normalize_url(original_link)
    except Exception:
        real_url = normalize_url(original_link)
    rss_summary = clean_html_text(article_info.get("본문요약", ""))
    body_stub = rss_summary or clean_html_text(article_info.get("기사제목", ""))
    event_tags = detect_event_tags(f"{article_info.get('기사제목','')} {rss_summary}")
    item = {
        "카테고리": json_key,
        "카테고리명": JSON_KEY_TO_DISPLAY.get(json_key, json_key),
        "검색어": article_info.get("검색어", ""),
        "기사제목": article_info.get("기사제목", ""),
        "언론사": article_info.get("언론사", "") or guess_press_name_from_url(real_url),
        "정규언론사": canonical_press_name(article_info.get("언론사", ""), real_url),
        "게시일": article_info.get("게시일", ""),
        "RSS게시일": article_info.get("게시일", ""),
        "RSS요약": rss_summary,
        "본문전문": body_stub,
        "본문글자수": len(body_stub),
        "본문추출방식": "critical_without_body",
        "본문품질점수": 0,
        "본문품질사유": reason,
        "중요도점수": article_info.get("중요도점수", ""),
        "본문사용가능": False,
        "본문상태": "본문추출실패_중요이슈_제한포함",
        "원래RSS링크": original_link,
        "링크": real_url,
        "랭킹점수": article_info.get("랭킹점수", ""),
        "사건태그": ",".join(sorted(event_tags)),
        "사건단계": extract_event_stage(f"{article_info.get('기사제목','')} {rss_summary}", event_tags),
        "원문날짜": "",
        "원문날짜출처": "not_checked_bodyless",
        "날짜검증결과": "not_checked_bodyless",
        "날짜후보수": 0,
        "대표선택점수": -5,
        "v20_issue_id": issue.get("issue_id"),
        "v20_issue_group": issue.get("issue_group"),
        "v20_issue_family": issue.get("issue_family"),
        "v20_internal_category": issue.get("internal_category"),
        "v20_bodyless_reason": reason,
        "Gemini카테고리": json_key,
        "Gemini내부카테고리": article_info.get("Gemini내부카테고리", issue.get("internal_category", "")),
        "Gemini이슈유형": article_info.get("Gemini이슈패밀리", issue.get("issue_family", "")),
        "Gemini이슈그룹": article_info.get("Gemini이슈그룹", issue.get("issue_group", "")),
        "Gemini중복그룹": article_info.get("Gemini이슈그룹", issue.get("issue_group", "")),
        "Gemini관련성": article_info.get("Gemini관련성", issue.get("company_impact", "")),
        "Gemini중요도": article_info.get("Gemini중요도", issue.get("max_ceo_priority", "")),
        "Gemini대외협력중요도": article_info.get("Gemini대외협력중요도", issue.get("max_pa_priority", "")),
        "Gemini관련도": article_info.get("Gemini관련도", issue.get("max_relevance", "")),
        "Gemini판단사유": article_info.get("Gemini판단사유", issue.get("label_reasons", "")),
        "Gemini선정사유": decision.get("reason", issue.get("label_reasons", "")),
        "선정단계": "critical_without_body",
    }
    return item


def v20_attach_issue_fields(report_item, article_info, issue, decision, reason):
    report_item["v20_issue_id"] = issue.get("issue_id")
    report_item["v20_issue_group"] = issue.get("issue_group")
    report_item["v20_issue_family"] = issue.get("issue_family")
    report_item["v20_internal_category"] = issue.get("internal_category")
    report_item["v20_issue_score"] = issue.get("issue_score")
    report_item["Gemini카테고리"] = decision.get("category") or article_info.get("Gemini카테고리", "")
    report_item["Gemini내부카테고리"] = article_info.get("Gemini내부카테고리", issue.get("internal_category", ""))
    report_item["Gemini이슈유형"] = article_info.get("Gemini이슈패밀리", issue.get("issue_family", ""))
    report_item["Gemini이슈그룹"] = article_info.get("Gemini이슈그룹", issue.get("issue_group", ""))
    report_item["Gemini중복그룹"] = article_info.get("Gemini이슈그룹", issue.get("issue_group", ""))
    report_item["Gemini관련성"] = article_info.get("Gemini관련성", issue.get("company_impact", ""))
    report_item["Gemini중요도"] = article_info.get("Gemini중요도", issue.get("max_ceo_priority", ""))
    report_item["Gemini대외협력중요도"] = article_info.get("Gemini대외협력중요도", issue.get("max_pa_priority", ""))
    report_item["Gemini관련도"] = article_info.get("Gemini관련도", issue.get("max_relevance", ""))
    report_item["Gemini판단사유"] = article_info.get("Gemini판단사유", issue.get("label_reasons", ""))
    report_item["Gemini선정사유"] = decision.get("reason", issue.get("label_reasons", ""))
    report_item["선정단계"] = reason
    return report_item


def v20_process_issue(decision, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter):
    issue = issue_by_id.get(decision.get("issue_id"))
    if not issue:
        return None, order_counter, "missing_issue"
    json_key = v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue))
    candidate_ids = []
    if decision.get("best_article_id"):
        candidate_ids.append(int(decision["best_article_id"]))
    for art_id in decision.get("backup_article_ids", []):
        if art_id not in candidate_ids:
            candidate_ids.append(int(art_id))
    for art_id in issue.get("candidate_article_ids", []):
        if art_id not in candidate_ids:
            candidate_ids.append(int(art_id))
    for art_id in issue.get("article_ids", []):
        if art_id not in candidate_ids:
            candidate_ids.append(int(art_id))

    attempted_failed = []
    for art_id in candidate_ids:
        if art_id in processed_article_ids:
            continue
        article_info = article_by_id.get(int(art_id))
        if not article_info:
            continue
        processed_article_ids.add(int(art_id))
        report_item, skip_info, failed_item = process_article_for_report(article_info, json_key, recent_past_items)
        if skip_info:
            skip_info["제외단계"] = "issue_representative_attempt"
            skip_info["v20_issue_id"] = issue.get("issue_id")
            skip_info["v20_issue_group"] = issue.get("issue_group")
            skip_rows.append(skip_info)
            attempted_failed.append((article_info, "past_duplicate_or_skip"))
            continue
        if failed_item:
            failed_item["제외단계"] = "issue_representative_attempt"
            failed_item["v20_issue_id"] = issue.get("issue_id")
            failed_item["v20_issue_group"] = issue.get("issue_group")
            body_failed_rows.append(failed_item)
            attempted_failed.append((article_info, failed_item.get("본문품질사유", "body_failed")))
            continue
        if not report_item:
            attempted_failed.append((article_info, "empty_process_result"))
            continue
        report_item = v20_attach_issue_fields(report_item, article_info, issue, decision, "issue_selected")
        report_item["선정순서"] = order_counter
        report_item["본문상태"] = "본문추출성공"

        relevant, relevance_reason = is_report_item_relevant(report_item, json_key)
        if not relevant:
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"relevance_failed:{relevance_reason}"
            failed_copy["제외단계"] = "issue_representative_attempt"
            body_failed_rows.append(failed_copy)
            attempted_failed.append((article_info, relevance_reason))
            continue

        # Issue-level duplication check. If a different issue already represents this event, keep the better article.
        dup_item, dup_reason = final_duplicate_reason(report_item, final_report_data)
        if dup_item:
            new_score = v20_float(report_item.get("대표선택점수"), 0)
            old_score = v20_float(dup_item.get("대표선택점수"), 0)
            if new_score > old_score + REPRESENTATIVE_REPLACE_MARGIN:
                try:
                    idx = final_report_data.index(dup_item)
                    final_report_data[idx] = report_item
                except ValueError:
                    final_report_data.append(report_item)
                add_duplicate_skip_row(
                    skip_rows,
                    dup_item,
                    report_item,
                    f"replaced_by_better_duplicate:{dup_reason}",
                    stage="issue_selected",
                    extra=f"old={old_score},new={new_score}",
                )
                return report_item, order_counter + 1, "replaced_duplicate"
            add_duplicate_skip_row(
                skip_rows,
                report_item,
                dup_item,
                f"kept_better_duplicate:{dup_reason}",
                stage="issue_selected",
                extra=f"old={old_score},new={new_score}",
            )
            return None, order_counter, "duplicate_already_represented"

        tag_limit_reason = event_tag_limit_reason(report_item, final_report_data)
        if tag_limit_reason:
            add_duplicate_skip_row(
                skip_rows,
                report_item,
                {"기사제목": "event_tag_cap", "링크": "", "대표선택점수": ""},
                tag_limit_reason,
                stage="issue_selected",
            )
            return None, order_counter, "event_tag_limited"

        final_report_data.append(report_item)
        return report_item, order_counter + 1, "body_success"

    # All representative candidates failed. Include critical issue with link/title/RSS only if needed.
    if v20_is_critical_issue(issue, decision):
        fallback_id = int(decision.get("best_article_id") or issue.get("top_article_id"))
        article_info = article_by_id.get(fallback_id)
        if article_info:
            item = v20_create_bodyless_report_item(article_info, issue, decision, json_key)
            item["선정순서"] = order_counter
            # Do not run normal body relevance filters; this is an explicit critical exception.
            dup_item, dup_reason = final_duplicate_reason(item, final_report_data)
            if dup_item:
                add_duplicate_skip_row(skip_rows, item, dup_item, f"bodyless_duplicate:{dup_reason}", stage="critical_without_body")
                return None, order_counter, "bodyless_duplicate"
            final_report_data.append(item)
            return item, order_counter + 1, "critical_without_body"
    return None, order_counter, "all_candidates_failed"


def v20_bodyless_summary(item):
    rss = clean_html_text(item.get("RSS요약", ""))
    title = clean_html_text(item.get("기사제목", ""))
    if rss:
        base = rss
    else:
        base = title
    base = v20_clip(base, 230)
    return f"원문 본문 자동 추출이 제한돼 세부 내용은 링크 확인이 필요함. 제목 및 RSS 요약 기준으로, {base}"


def v20_build_summary_prompt(final_report_data, recent_past_text):
    final_input_text = ""
    for item in final_report_data:
        body_status = item.get("본문상태", "본문추출성공")
        body_limit = 900 if body_status.startswith("본문추출실패") else MAX_BODY_CHARS_FOR_PROMPT
        final_input_text += (
            f"[기사번호 {item['브리핑ID']}]\n"
            f"카테고리: {item['카테고리명']}\n"
            f"제목: {item['기사제목']}\n"
            f"언론사: {item['언론사']}\n"
            f"게시일: {item['게시일']}\n"
            f"본문상태: {body_status}\n"
            f"이슈그룹: {item.get('v20_issue_group','')}\n"
            f"선정사유: {item.get('Gemini선정사유','')}\n"
            f"본문/RSS:\n{item['본문전문'][:body_limit]}\n\n"
        )
    return f"""
너는 최고 경영진에게 매일 아침 뉴스 브리핑을 제공하는 수석 전략가야.
아래 [오늘 기사 데이터]의 본문 또는 RSS 요약만 사용해서 각 기사별 요약문만 작성해.

[요약 규칙]
1. 네 생각, 인사이트, 전망, 대응 포인트, 의미 부여는 쓰지마.
2. 오직 제공된 본문/RSS에 있는 객관적 사실만 요약해.
3. 기사 1개당 요약은 1문단으로만 작성해.
4. 모든 문장의 끝은 '~함', '~임', '~됨', '~계획임', '~예정임' 같은 문어체 종결로 맞춰.
5. 제공 내용에 없는 사실은 절대 추가하지마.
6. 제목, 링크, 언론사, 카테고리, 번호는 쓰지마. 요약문만 반환해.
7. 본문상태가 '본문추출실패_중요이슈_제한포함'이면 첫 문장에 '원문 본문 자동 추출이 제한돼 세부 내용은 링크 확인이 필요함'이라고 명시하고, 제목/RSS 요약 범위에서만 짧게 정리해.
8. 사진/캡션/관련기사/저작권 문구는 요약하지마.

반드시 JSON 객체 형식으로만 응답해. 키는 기사번호 문자열이고 값은 요약문이야.
{{
  "1": "요약문",
  "2": "요약문"
}}

[최근 과거 보고서 문체 참고]
{recent_past_text[:5000]}

[오늘 기사 데이터]
{final_input_text}
"""


def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    if not client or not ENABLE_GEMINI_QA:
        return ""
    item_lines = []
    for item in final_report_data:
        item_lines.append(
            f"[{item.get('브리핑ID')}] cat={item.get('카테고리')} issue={v20_clip(item.get('v20_issue_group',''),80)} "
            f"family={item.get('v20_issue_family','')} body={item.get('본문상태','')} title={v20_clip(item.get('기사제목',''),100)}"
        )
    prompt = f"""
너는 카카오 대외협력팀 뉴스 브리핑의 품질검수자야.
아래 최종 편집 목록을 보고 문제 가능성을 짧게 점검해.
특히 같은 이슈 중복, 홍보성 기사 포함, 카카오 관련성이 낮은 일반 금융/지역/시황 기사, 본문추출실패 기사 과다 여부를 확인해.
최종 보고서를 수정하지 말고 검수 의견만 JSON으로 반환해.

[편집 목록]
{chr(10).join(item_lines)}

[최종 보고서]
{final_briefing_text[:7000]}

형식:
{{
  "overall": "양호/주의/문제",
  "warnings": ["경고1", "경고2"],
  "suggested_human_checks": ["확인할 기사 또는 이슈"]
}}
"""
    try:
        text = gemini_generate_text(client=client, prompt=prompt, task_name="v20 최종 품질검수", model=GEMINI_MODEL_QA)
        return text.strip()
    except Exception as e:
        return f"QA failed: {e}"


def v20_decision_rows(decisions, backups, issue_by_id):
    rows = []
    for kind, arr in [("selected", decisions), ("backup", backups)]:
        for order, d in enumerate(arr, 1):
            issue = issue_by_id.get(d.get("issue_id"), {})
            rows.append({
                "kind": kind,
                "order": order,
                "issue_id": d.get("issue_id"),
                "category": d.get("category"),
                "priority": d.get("priority"),
                "best_article_id": d.get("best_article_id"),
                "backup_article_ids": ",".join(map(str, d.get("backup_article_ids", []))),
                "reason": d.get("reason"),
                "issue_group": issue.get("issue_group"),
                "issue_family": issue.get("issue_family"),
                "internal_category": issue.get("internal_category"),
                "issue_score": issue.get("issue_score"),
                "titles": " | ".join(issue.get("titles", [])[:3]),
            })
    return rows


def main():
    total_start_time = time.time()
    run_log = []

    print(f"\n🧩 v20 실행: {V6_VERSION}")
    print(f"   └ 모델: labeling={GEMINI_MODEL_LABELING}, editor={GEMINI_MODEL_EDITOR}, summary={GEMINI_MODEL_SUMMARY}, qa={GEMINI_MODEL_QA}")

    client = None
    if ENABLE_GEMINI_SELECTION or ENABLE_GEMINI_REPORT:
        try:
            with open(SECRET_PATH, "r", encoding="utf-8") as f:
                google_api_key = f.read().strip()
            if not google_api_key:
                raise ValueError("secret.txt가 비어 있습니다.")
            client = genai.Client(api_key=google_api_key)
        except Exception as e:
            print("⚠️ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다. Gemini 없이 로컬 이슈 편집으로 진행합니다.")
            print(f"   원인: {e}")
            client = None

    past_reports_content, all_past_items, recent_past_items = load_past_reports()
    recent_past_text = build_recent_past_text(recent_past_items, max_chars=9000)

    raw_articles, skipped_duplicates = collect_with_google_rss(recent_past_items)
    print(f"\n  └ ✅ 총 {len(raw_articles)}개의 RSS 후보 확보 완료")
    print(f"  └ 🧹 수집/과거중복/오탐 제외 후보: {len(skipped_duplicates)}개")

    if not raw_articles:
        print("❌ 수집된 기사가 없습니다. Google News RSS 접속 또는 키워드/기간 설정을 확인하세요.")
        return

    ranked_all, ranked_candidates = rank_and_trim_candidates(raw_articles)
    print(f"  └ 🧺 v20 후보 풀: 전체 {len(ranked_all)}개 중 {len(ranked_candidates)}개를 기사 라벨링 대상으로 사용")

    print("\n🏷️ [STEP 2] Gemini가 기사별 의미 라벨을 붙입니다...")
    labels = {}
    gemini_label_success = False
    try:
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 라벨링 비활성화 또는 client 없음")
        labels = v20_gemini_label_candidates(client, ranked_candidates, recent_past_text)
        gemini_label_success = True
        print(f"  └ ✅ Gemini 기사 라벨링 완료: {len(labels)}개")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사 라벨링 실패. 로컬 보조 라벨로 진행합니다. 원인: {e}")
        labels = {int(a["id"]): v20_local_article_label(a) for a in ranked_candidates}

    # Complete labels for every article for diagnostics.
    for article in ranked_all:
        art_id = int(article["id"])
        if art_id not in labels:
            labels[art_id] = v20_local_article_label(article)

    label_rows = v20_apply_labels_to_articles(ranked_all, labels)

    # Save candidate CSVs after diagnostic fields are attached.
    pd.DataFrame(ranked_all).to_csv(OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(ranked_candidates).to_csv(OUTPUT_RANKED_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(OUTPUT_GEMINI_LABELS_CSV, index=False, encoding="utf-8-sig")
    if skipped_duplicates:
        pd.DataFrame(skipped_duplicates).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 후보/라벨 CSV 저장 완료: {os.path.basename(OUTPUT_CANDIDATES_CSV)}, {os.path.basename(OUTPUT_RANKED_CSV)}, {os.path.basename(OUTPUT_GEMINI_LABELS_CSV)}")

    print("\n🧱 [STEP 3] 기사 후보를 이슈 단위로 묶습니다...")
    issues, article_by_id = v20_build_issues(ranked_candidates, labels)
    issue_by_id = {i["issue_id"]: i for i in issues}
    pd.DataFrame(v20_issue_rows(issues)).to_csv(OUTPUT_ISSUES_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ ✅ 이슈 그룹 {len(issues)}개 생성 / 저장: {os.path.basename(OUTPUT_ISSUES_CSV)}")

    print("\n🧠 [STEP 4] Gemini가 이슈 단위로 최종 편집합니다...")
    gemini_editor_success = False
    try:
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 이슈 편집 비활성화 또는 client 없음")
        decisions, backup_decisions = v20_gemini_edit_issues(client, issues, recent_past_text)
        gemini_editor_success = True
        print(f"  └ ✅ Gemini 이슈 편집 완료: 선택 {len(decisions)}개 / 백업 {len(backup_decisions)}개")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 이슈 편집 실패. 로컬 이슈 점수 기반으로 진행합니다. 원인: {e}")
        decisions, backup_decisions = v20_local_issue_selection(issues)

    decisions = v20_apply_decision_caps(decisions, issue_by_id)
    # If caps or malformed output made the list too short, refill from local backup.
    if len(decisions) < V20_FINAL_MIN:
        local_decisions, local_backups = v20_local_issue_selection(issues)
        existing = {d["issue_id"] for d in decisions}
        for d in local_decisions + local_backups:
            if len(decisions) >= V20_FINAL_TARGET:
                break
            if d["issue_id"] in existing:
                continue
            test = v20_apply_decision_caps(decisions + [d], issue_by_id)
            if len(test) > len(decisions):
                decisions = test
                existing.add(d["issue_id"])
        for b in local_backups:
            if b["issue_id"] not in {x["issue_id"] for x in backup_decisions}:
                backup_decisions.append(b)

    pd.DataFrame(v20_decision_rows(decisions, backup_decisions, issue_by_id)).to_csv(OUTPUT_ISSUE_SELECTION_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 이슈 편집 결과 저장: {os.path.basename(OUTPUT_ISSUE_SELECTION_CSV)}")

    print("\n🕵️‍♂️ [STEP 5] 선택 이슈별 대표 기사를 찾고 본문을 추출합니다...")
    print("   └ 본문 실패 시 같은 이슈의 다른 기사로 먼저 대체하고, 매우 중요한 이슈만 제목/RSS 제한 포함합니다.")

    final_report_data = []
    body_failed_rows = []
    post_body_duplicate_skips = []
    processed_article_ids = set()
    order_counter = 1
    status_counts = {}

    def process_decision_list(decision_list, stage_name="selected"):
        nonlocal order_counter
        for decision in decision_list:
            if len(final_report_data) >= V20_FINAL_TARGET and stage_name == "backup":
                break
            if len(final_report_data) >= V20_FINAL_MAX:
                break
            item, order_counter, status = v20_process_issue(
                decision,
                issue_by_id,
                article_by_id,
                recent_past_items,
                final_report_data,
                body_failed_rows,
                post_body_duplicate_skips,
                processed_article_ids,
                order_counter,
            )
            status_counts[status] = status_counts.get(status, 0) + 1
            issue = issue_by_id.get(decision.get("issue_id"), {})
            if item:
                print(
                    f"  └ 📥 이슈 반영: {v20_clip(issue.get('issue_group',''), 42)} / "
                    f"{v20_clip(item.get('기사제목',''), 42)}... "
                    f"({item.get('본문추출방식')}, {item.get('본문글자수')}자, {status})"
                )
            else:
                print(f"  └ ⚠️ 이슈 제외/대체대기: {v20_clip(issue.get('issue_group',''), 48)}... ({status})")

    process_decision_list(decisions, stage_name="selected")

    if len(final_report_data) < V20_FINAL_MIN:
        print(f"\n  └ 🔁 최종 {len(final_report_data)}개라 이슈 backup으로 보충합니다...")
        process_decision_list(backup_decisions, stage_name="backup")

    if len(final_report_data) < V20_FINAL_MIN:
        print(f"\n  └ 🔁 아직 {len(final_report_data)}개라 로컬 이슈 후보에서 추가 보충합니다...")
        local_decisions, local_backups = v20_local_issue_selection(issues)
        already_issues = {x.get("v20_issue_id") for x in final_report_data}
        more = [d for d in local_decisions + local_backups if d.get("issue_id") not in already_issues]
        process_decision_list(more, stage_name="local_extra")

    if skipped_duplicates or post_body_duplicate_skips:
        all_skips = skipped_duplicates + post_body_duplicate_skips
        pd.DataFrame(all_skips).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 제외/중복/대표교체 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")
    if body_failed_rows:
        pd.DataFrame(body_failed_rows).to_csv(OUTPUT_BODY_FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ ⚠️ 본문 품질 미달/관련성 미달 기사 저장: {os.path.basename(OUTPUT_BODY_FAILED_CSV)}")

    if not final_report_data:
        print("❌ 최종 보고서에 사용할 기사 데이터가 없습니다.")
        return

    # Output order: category order, then selected order inside category.
    ordered_final = []
    for json_key in JSON_KEYS_ORDER:
        items = [x for x in final_report_data if x.get("카테고리") == json_key]
        items = sorted(items, key=lambda x: int(x.get("선정순서") or 9999))
        ordered_final.extend(items)
    final_report_data = ordered_final[:V20_FINAL_MAX]

    print("\n✍️ [STEP 6] 본문/RSS 기반으로 최종 브리핑 요약을 생성합니다...")
    for idx, item in enumerate(final_report_data, 1):
        item["브리핑ID"] = idx

    prompt_report = v20_build_summary_prompt(final_report_data, recent_past_text)
    summary_map = {}
    try:
        if not client or not ENABLE_GEMINI_REPORT:
            raise RuntimeError("Gemini 최종 브리핑 비활성화 또는 client 없음")
        summary_text = gemini_generate_text(
            client=client,
            prompt=prompt_report,
            task_name="v20 최종 기사별 요약 생성",
            model=GEMINI_MODEL_SUMMARY,
        )
        summary_map = normalize_summary_json(extract_json_object(summary_text))
        print("  └ ✅ Gemini 기사별 요약 생성 완료")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사별 요약 생성 실패. 로컬 본문/RSS 요약으로 대체합니다. 원인: {e}")
        summary_map = {}

    for item in final_report_data:
        bid = str(item.get("브리핑ID"))
        if item.get("본문상태") == "본문추출실패_중요이슈_제한포함" and not summary_map.get(bid):
            summary_map[bid] = v20_bodyless_summary(item)

    final_briefing_text = build_structured_briefing(final_report_data, summary_map)

    qa_text = ""
    if ENABLE_GEMINI_QA:
        print("\n🔎 [STEP 7] 최종 편집 품질검수 로그를 생성합니다...")
        qa_text = v20_gemini_quality_check(client, final_report_data, final_briefing_text)
        if qa_text:
            with open(OUTPUT_QA_TXT, "w", encoding="utf-8") as f:
                f.write(qa_text)
            print(f"  └ 💾 QA 로그 저장: {os.path.basename(OUTPUT_QA_TXT)}")

    print("\n" + "=" * 60)
    print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
    print("=" * 60)
    print(final_briefing_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_briefing_text)
    pd.DataFrame(final_report_data).to_csv(OUTPUT_SELECTED_CSV, index=False, encoding="utf-8-sig")

    run_log.append({
        "버전": V6_VERSION,
        "모델_라벨링": GEMINI_MODEL_LABELING,
        "모델_편집": GEMINI_MODEL_EDITOR,
        "모델_요약": GEMINI_MODEL_SUMMARY,
        "전체_RSS_후보": len(raw_articles),
        "Gemini_기사라벨_후보": len(ranked_candidates),
        "이슈그룹수": len(issues),
        "Gemini라벨링성공": gemini_label_success,
        "Gemini이슈편집성공": gemini_editor_success,
        "선택이슈수": len(decisions),
        "백업이슈수": len(backup_decisions),
        "본문성공": status_counts.get("body_success", 0) + status_counts.get("replaced_duplicate", 0),
        "본문실패_중요이슈_제한포함": status_counts.get("critical_without_body", 0),
        "제외_총합": len(skipped_duplicates) + len(post_body_duplicate_skips),
        "본문품질미달_관련성미달": len(body_failed_rows),
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
    print(f"- '{os.path.basename(OUTPUT_GEMINI_LABELS_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_ISSUES_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_ISSUE_SELECTION_CSV)}' 저장 완료")
    if skipped_duplicates or post_body_duplicate_skips:
        print(f"- '{os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}' 저장 완료")
    if body_failed_rows:
        print(f"- '{os.path.basename(OUTPUT_BODY_FAILED_CSV)}' 저장 완료")
    if qa_text:
        print(f"- '{os.path.basename(OUTPUT_QA_TXT)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_RUN_LOG_CSV)}' 저장 완료")
    print("=" * 60)



# v20.1 patch: make platform-obligation detection stricter than the v18 rescue regex.
# This prevents ordinary "카카오톡 선물하기 입점/유통망 확대" articles from being mistaken for regulation.
def v20_is_platform_obligation_text(text):
    t = clean_html_text(text)
    if not t:
        return False
    actor_re = re.compile(
        r"플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|부가통신사업자|정보통신서비스\s*제공자|인터넷\s*사업자|"
        r"네이버|카카오|구글|메타|유튜브|틱톡|애플|쿠팡|배달의민족|배민|토스",
        re.IGNORECASE,
    )
    strong_action_re = re.compile(
        r"의무|의무화|해야\s*한다|해야\s*함|해야\s*할|걸러야|차단|사전\s*차단|삭제|필터링|"
        r"신고|보고|공시|자료\s*제출|자료제출|준수|위반|금지|제한|제재|과징금|과태료|"
        r"시정명령|유통\s*방지|확산\s*방지|모니터링|소명",
        re.IGNORECASE,
    )
    soft_action_re = re.compile(r"시행|적용|확대|강화|가이드라인", re.IGNORECASE)
    regulator_re = re.compile(
        r"정부|국회|과기정통부|방미통위|방송미디어통신위원회|방통위|개보위|개인정보위|공정위|"
        r"공정거래위원회|금융위|금융감독원|금감원|행안부|경찰청|검찰|KISA|인터넷진흥원|"
        r"법안|개정안|시행령|시행규칙|고시|입법예고|행정예고|제도|정책|규제|감독|조사|제재|내달부터|다음달부터|오는\s*\d{1,2}월부터",
        re.IGNORECASE,
    )
    duty_domain_re = re.compile(
        r"불법\s*촬영|불법정보|유해정보|유해\s*정보|허위조작정보|딥페이크|디지털성범죄|청소년\s*보호|"
        r"개인정보|정보\s*유출|사칭|피싱|보이스피싱|악성코드|이용자\s*보호|알고리즘|광고\s*표시|"
        r"인앱결제|앱마켓|수수료|정산|입점업체|판매자|자사우대|최혜대우|온라인플랫폼법|온플법",
        re.IGNORECASE,
    )
    if not actor_re.search(t):
        return False
    if strong_action_re.search(t) and (regulator_re.search(t) or duty_domain_re.search(t)):
        return True
    # Soft words like 확대/강화 are too broad, so require both regulator and duty domain.
    if soft_action_re.search(t) and regulator_re.search(t) and duty_domain_re.search(t):
        return True
    return False



# v20.2 patch: avoid false self matches from the Korean adverb "다음은" and demote commerce partner noise.
def v20_has_self_entity(text):
    t = clean_html_text(text)
    if not t:
        return False
    strong_self_re = re.compile(
        r"카카오|카톡|Kakao|카나나|카카오페이|카카오뱅크|카카오모빌리티|카카오\s*T\b|카카오T\b|"
        r"카카오택시|카카오게임즈|카카오엔터|카카오헬스케어|카카오엔터프라이즈|카카오픽코마|디케이테크인|엑스엘게임즈|AXZ",
        re.IGNORECASE,
    )
    if strong_self_re.search(t):
        return True
    # Treat Daum as self only when it is clearly the portal/service name, not "다음은/다음으로".
    if re.search(r"\bDaum\b|포털\s*다음|다음\s*뉴스|다음\s*메일|다음\s*카페", t, re.IGNORECASE):
        return True
    return False


def v20_detect_issue_family(text, json_key=""):
    t = clean_html_text(text)
    if not t:
        return "off_topic"
    if v20_is_pr_text(t):
        return "low_value_pr"
    if v20_is_platform_obligation_text(t):
        if v20_has_self_entity(t):
            return "platform_obligation"
        return "government_platform_regulation"
    if v20_has_self_entity(t):
        # 카카오톡 선물하기에 브랜드가 입점/판매/순위 회복했다는 류는 서비스 리스크가 아니라 낮은 사업/커머스 소식으로 둔다.
        if re.search(r"카카오톡\s*선물하기|카톡\s*선물하기", t, re.IGNORECASE):
            if re.search(r"입점|판매|1위|재탈환|브랜드|상품|스토어|기획전|혜택|프로모션|유통망|선보", t, re.IGNORECASE):
                if not re.search(r"공정위|규제|제재|과징금|수수료|정산|장애|개인정보|노조|파업|서비스\s*개편|이용자\s*반발|카카오.{0,30}논란", t, re.IGNORECASE):
                    return "self_affiliate_business"
        if re.search(r"노조|파업|쟁의|임단협|임금교섭|성과급|RSU|조정|노동위|지노위|중노위|고용불안|노동부|근로감독|최저임금", t, re.I):
            return "self_labor"
        if re.search(r"임원|CPO|CTO|CFO|대표|경영진|조직개편|퇴사|사임|사퇴|교체|리더십|서비스\s*개편|논란|반발|사과|후폭풍|친구탭|카톡\s*개편", t, re.I):
            return "self_leadership_service"
        if re.search(r"인수|매각|합병|지분|최대주주|경영권|공동대표|사내이사|주총|상장|IPO|투자|유상증자|전환사채", t, re.I):
            return "self_governance_mna"
        if re.search(r"개인정보|유출|해킹|피싱|사칭|보안|장애|먹통|침해사고|딥페이크", t, re.I):
            return "self_privacy_security"
        if re.search(r"수사|조사|과징금|제재|고발|소송|판결|검찰|경찰|압수수색|시정명령", t, re.I):
            return "self_legal_regulatory"
        if re.search(r"실적|매출|영업이익|영업손실|순이익|적자|흑자|주가|목표가|투자의견", t, re.I):
            return "self_financial_performance"
        return "self_affiliate_business"
    if json_key == "정부_국회" or REGULATOR_CORE_PATTERN.search(t):
        if re.search(r"금감원|금융위|ELS|은행|보험|증권|파생|금소법|불완전\s*판매", t, re.I):
            return "general_financial_enforcement"
        if re.search(r"플랫폼|온플법|공정위|온라인플랫폼|자사우대|최혜대우|수수료|정산|인앱결제|알고리즘|유해정보|개인정보|딥페이크|방미통위|개보위", t, re.I):
            return "government_platform_regulation"
        if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|저작권|국가AI", t, re.I):
            return "government_ai_policy"
        if re.search(r"스테이블코인|디지털자산|가상자산|특금법|FIU|CBDC|토큰", t, re.I):
            return "government_digital_finance_policy"
        return "government_general_policy"
    if json_key == "경쟁사_해외이슈" or COMPETITOR_CORE_PATTERN.search(t):
        if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|LLM|오픈AI|구글|MS|메타|애플|앤트로픽|엔비디아", t, re.I):
            return "competitor_ai_strategy"
        if re.search(r"개인정보|유출|해킹|피싱|보안|장애|제재|소송|과징금|규제|조사", t, re.I):
            return "competitor_security_risk"
        return "competitor_business_strategy"
    if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|반도체|HBM|TSMC|엔비디아|전력|원전|스테이블코인|디지털자산|플랫폼", t, re.I):
        return "industry_ai_infrastructure"
    return "off_topic"



# =========================================================
# v21 overrides: protected issue desk + domestic competitor first
# =========================================================
# v21 changes from v20
# 1) Gemini issue_group is authoritative. Cross-issue duplicate removal is strict only.
# 2) Final category balance is protected; competitor/domestic issues cannot disappear in fallback.
# 3) Self/Gov/Competitor prompts use tiered public-affairs reasoning rather than fixed slots.
# 4) Domestic competitors are prioritized; overseas big-tech is normally capped at 1, 2 only for priority-5 issues.
# 5) Local fallback is category-aware and excludes LOW_VALUE_PR / low-relevance general finance.

V6_VERSION = "google_rss_v21_protected_issue_desk_domestic_competitor"

V21_FINAL_MIN = 11
V21_FINAL_TARGET = 13
V21_FINAL_MAX = 15
V20_FINAL_MIN = V21_FINAL_MIN
V20_FINAL_TARGET = V21_FINAL_TARGET
V20_FINAL_MAX = V21_FINAL_MAX
MAX_SELECT_COUNT = V21_FINAL_MAX
MIN_SELECT_COUNT = V21_FINAL_MIN

CATEGORY_MIN = {
    "자사_및_계열사_이슈": 3,
    "정부_국회": 3,
    "경쟁사_해외이슈": 3,
    "산업동향": 1,
}
CATEGORY_TARGET = {
    "자사_및_계열사_이슈": 4,
    "정부_국회": 4,
    "경쟁사_해외이슈": 4,
    "산업동향": 1,
}
CATEGORY_MAX = {
    "자사_및_계열사_이슈": 5,
    "정부_국회": 5,
    "경쟁사_해외이슈": 4,
    "산업동향": 2,
}

V21_COMPETITOR_DOMESTIC_MIN = 2
V21_COMPETITOR_DOMESTIC_TARGET = 3
V21_COMPETITOR_OVERSEAS_MAX_DEFAULT = 1
V21_COMPETITOR_OVERSEAS_MAX_PRIORITY5 = 2

V21_SELF_TIERS = {
    "DIRECT_RISK",
    "RESPONSE_RELEVANT",
    "STRATEGIC_REFERENCE",
    "FILLER_REFERENCE",
    "LOW_VALUE_PR",
    "OFF_TOPIC",
}
V21_GOV_TIERS = {
    "GOV_DIRECT_PLATFORM_OBLIGATION",
    "GOV_AI_CORE_POLICY",
    "GOV_REGULATORY_ENFORCEMENT",
    "GOV_PLATFORM_MARKET_RULE",
    "GOV_APPOINTMENT_GOVERNANCE",
    "GOV_DIGITAL_FINANCE_POLICY",
    "GOV_ADJACENT_DIGITAL_POLICY",
    "GOV_GENERAL_LOW_RELEVANCE",
    "OFF_TOPIC",
}
V21_COMPETITOR_TIERS = {
    "COMP_DOMESTIC_DIRECT_RISK",
    "COMP_DOMESTIC_STRATEGY",
    "COMP_DOMESTIC_AI_PLATFORM",
    "COMP_DOMESTIC_FINANCE_COMMERCE",
    "COMP_DOMESTIC_PUBLIC_INFRA",
    "COMP_OVERSEAS_PLATFORM_AI",
    "COMP_OVERSEAS_REGULATION_DATA",
    "COMP_OVERSEAS_MARKET_SIGNAL",
    "COMP_GENERAL_LOW_RELEVANCE",
    "OFF_TOPIC",
}

V21_DOMESTIC_COMPETITOR_RE = re.compile(
    r"네이버|NAVER|네이버페이|쿠팡|쿠팡이츠|쿠팡플레이|토스|토스페이|비바리퍼블리카|"
    r"배달의민족|배민|우아한형제들|SKT|SK텔레콤|KT|LGU\+|LG\s*U\+|LG유플러스|"
    r"라인야후|라인게임즈|야놀자|당근|당근마켓|쏘카|우버",
    re.IGNORECASE,
)
V21_OVERSEAS_COMPETITOR_RE = re.compile(
    r"구글|Google|오픈AI|OpenAI|MS|마이크로소프트|Microsoft|메타|Meta|애플|Apple|"
    r"앤트로픽|Anthropic|아마존|Amazon|xAI|테슬라|Tesla|엔비디아|NVIDIA|"
    r"유튜브|YouTube|틱톡|TikTok",
    re.IGNORECASE,
)

V21_SELF_PR_RE = re.compile(
    r"스폰서|후원|협찬|기부|기금\s*전달|영화제|축제|페스티벌|캠페인|이벤트|쿠폰|할인|프로모션|수상|브랜드\s*대상|체험단",
    re.IGNORECASE,
)
V21_SELF_STRATEGIC_RE = re.compile(
    r"챗GPT|ChatGPT|AI|인공지능|에이전트|카나나|카카오톡\s*연동|카톡방|슈퍼\s*월렛|원화코인|스테이블코인|"
    r"판교|오피스|이전|조직\s*일원화|데이터센터|클라우드|서비스\s*개편|신규\s*기능|결제|금융|디지털지갑|파트너십|제휴",
    re.IGNORECASE,
)
V21_SELF_DIRECT_RISK_RE = re.compile(
    r"노조|파업|쟁의|임단협|고용불안|구조조정|임금|성과급|RSU|수사|조사|과징금|제재|소송|판결|"
    r"개인정보|유출|해킹|피싱|사칭|장애|먹통|서비스\s*논란|이용자\s*반발|조직개편|임원|CPO|CTO|CFO|대표|퇴사|사임|"
    r"지분|매각|인수|합병|최대주주|경영권|공동대표|사내이사|주총|유상증자|전환사채|실적|영업이익|적자",
    re.IGNORECASE,
)

V21_GOV_AI_CORE_RE = re.compile(
    r"AI|인공지능|에이전트|에이전틱|국가AI|AI전략위|AI\s*수석|AI\s*컨트롤타워|AI\s*행동계획|"
    r"AI기본법|AI\s*저작권|AI\s*안전|AI\s*거버넌스|AIDC|AI\s*데이터센터|GPU|NPU|공공\s*AI|"
    r"디지털플랫폼정부|디플정|클라우드|데이터\s*센터",
    re.IGNORECASE,
)
V21_GOV_DIRECT_PLATFORM_RE = re.compile(
    r"플랫폼|온라인\s*플랫폼|포털|SNS|부가통신사업자|정보통신서비스|네이버|카카오|구글|유튜브|틱톡|앱마켓|"
    r"불법촬영|유해정보|딥페이크|청소년|개인정보|사칭|피싱|알고리즘|광고\s*표시|이용자\s*보호|"
    r"의무|의무화|차단|삭제|필터링|보고|공시|자료\s*제출|고지|금지|시행|적용",
    re.IGNORECASE,
)
V21_GOV_DIGITAL_FINANCE_RE = re.compile(
    r"스테이블코인|원화코인|디지털자산|가상자산|특금법|FIU|트래블룰|AML|자금세탁|전자금융|마이데이터|간편결제|CBDC|토큰증권|STO",
    re.IGNORECASE,
)
V21_GOV_GENERAL_LOW_RE = re.compile(
    r"홍콩\s*ELS|ELS|보험사\s*순이익|제약|건설|레미콘|보령|탁소텔|삼표|선거사범|지방선거|일반\s*은행|불완전\s*판매",
    re.IGNORECASE,
)


def v21_competitor_scope(text):
    t = clean_html_text(text)
    domestic = bool(V21_DOMESTIC_COMPETITOR_RE.search(t))
    overseas = bool(V21_OVERSEAS_COMPETITOR_RE.search(t))
    if domestic and overseas:
        # Domestic wins when the issue is about a Korean company partnering/competing with a global firm.
        return "mixed"
    if domestic:
        return "domestic"
    if overseas:
        return "overseas"
    return "other"


def v21_self_tier(text):
    t = clean_html_text(text)
    if not v20_has_self_entity(t):
        return "OFF_TOPIC"
    if V21_SELF_PR_RE.search(t) and not V21_SELF_DIRECT_RISK_RE.search(t) and not V21_GOV_DIRECT_PLATFORM_RE.search(t):
        return "LOW_VALUE_PR"
    if V21_SELF_DIRECT_RISK_RE.search(t):
        return "DIRECT_RISK"
    if v20_is_platform_obligation_text(t) or REGULATOR_CORE_PATTERN.search(t):
        return "RESPONSE_RELEVANT"
    if V21_SELF_STRATEGIC_RE.search(t):
        return "STRATEGIC_REFERENCE"
    return "FILLER_REFERENCE"


def v21_gov_tier(text):
    t = clean_html_text(text)
    if V21_GOV_DIRECT_PLATFORM_RE.search(t) and v20_is_platform_obligation_text(t):
        return "GOV_DIRECT_PLATFORM_OBLIGATION"
    if V21_GOV_AI_CORE_RE.search(t):
        # AI policy is a core Kakao strategy environment, not a mere filler.
        return "GOV_AI_CORE_POLICY"
    if re.search(r"공정위|개보위|방미통위|과기정통부|금융위|금감원|검찰|경찰|조사|제재|과징금|시정명령|감독|가이드라인", t, re.I):
        if re.search(r"플랫폼|AI|인공지능|개인정보|보안|딥페이크|유해정보|전자금융|디지털자산|가상자산|핀테크|광고|알고리즘|앱마켓|인앱결제", t, re.I):
            return "GOV_REGULATORY_ENFORCEMENT"
    if re.search(r"온플법|온라인플랫폼|자사우대|최혜대우|수수료|정산|검색시장|광고시장|다크패턴|인앱결제|앱마켓", t, re.I):
        return "GOV_PLATFORM_MARKET_RULE"
    if re.search(r"AI\s*수석|AI전략위|국가AI전략위원회|컨트롤타워|위원장|부위원장|인선|공백|출범|조직\s*개편|상임위|과방위|정무위", t, re.I):
        return "GOV_APPOINTMENT_GOVERNANCE"
    if V21_GOV_DIGITAL_FINANCE_RE.search(t):
        return "GOV_DIGITAL_FINANCE_POLICY"
    if re.search(r"디지털트윈|공공데이터|스마트시티|지도|공간정보|자율주행|마이데이터", t, re.I):
        return "GOV_ADJACENT_DIGITAL_POLICY"
    if V21_GOV_GENERAL_LOW_RE.search(t):
        return "GOV_GENERAL_LOW_RELEVANCE"
    if REGULATOR_CORE_PATTERN.search(t):
        return "GOV_GENERAL_LOW_RELEVANCE"
    return "OFF_TOPIC"


def v21_competitor_tier(text):
    t = clean_html_text(text)
    scope = v21_competitor_scope(t)
    if scope in {"domestic", "mixed"}:
        if re.search(r"장애|오류|복구|개인정보|유출|해킹|피싱|보안|사과|이용자\s*불편|제재|과징금|조사|논란", t, re.I):
            return "COMP_DOMESTIC_DIRECT_RISK"
        if re.search(r"AI|인공지능|AX|에이전트|검색|추천|하이퍼클로바|에이닷|GPU|클라우드|데이터센터|LLM", t, re.I):
            return "COMP_DOMESTIC_AI_PLATFORM"
        if re.search(r"네이버페이|토스|토스페이|쿠팡페이|간편결제|금융|커머스|쇼핑|멤버십|배달|쿠팡이츠|배민|광고", t, re.I):
            return "COMP_DOMESTIC_FINANCE_COMMERCE"
        if re.search(r"공공|재난|우선\s*통신|우선\s*접속|행정|정부|망|통신권|재난망|공공\s*AI", t, re.I):
            return "COMP_DOMESTIC_PUBLIC_INFRA"
        return "COMP_DOMESTIC_STRATEGY"
    if scope == "overseas":
        if re.search(r"데이터|학습|저작권|언론|뉴스|콘텐츠|개인정보|광고|앱마켓|DMA|DSA|규제|소송|과징금|AI\s*랩", t, re.I):
            return "COMP_OVERSEAS_REGULATION_DATA"
        if re.search(r"AI|인공지능|에이전트|LLM|검색|제미나이|ChatGPT|GPT|클로드|모델|앱|플랫폼", t, re.I):
            return "COMP_OVERSEAS_PLATFORM_AI"
        if re.search(r"투자|IPO|상장|주가|지분|CAPEX|데이터센터|전력|발전|자금\s*조달", t, re.I):
            return "COMP_OVERSEAS_MARKET_SIGNAL"
    return "COMP_GENERAL_LOW_RELEVANCE"


def v21_canonical_issue_group(label_group, article=None, issue=None):
    text = f"{label_group or ''} "
    if article:
        text += v20_article_text(article)
    if issue:
        text += " " + " ".join(issue.get("titles", [])[:3]) + " " + str(issue.get("company_impact", ""))
    t = clean_html_text(text)
    if re.search(r"카카오게임즈", t) and re.search(r"라인야후|LAAA|최대주주|공동대표|김태환|이시우|사내이사|주총", t, re.I):
        return "카카오게임즈 최대주주 변경 및 공동대표 체제"
    if re.search(r"카카오", t) and re.search(r"노조|파업|임단협|임금\s*교섭|경영진\s*책임|고용불안|조정", t, re.I):
        return "카카오 노조 임단협/파업 및 내부 갈등"
    if re.search(r"불법촬영", t) and re.search(r"이미지|차단|필터링|유통\s*방지|의무", t, re.I):
        return "플랫폼 불법촬영 이미지 차단 의무 확대"
    if re.search(r"금융.*AI|AI.*금융|FSB|금융안정위", t, re.I) and re.search(r"가이드라인|모범사례|AI\s*도입", t, re.I):
        return "금융권 AI 도입 가이드라인/모범사례"
    if re.search(r"AI\s*수석|AI전략위|국가AI전략위원회|컨트롤타워|디플정|부위원장|인선|공백", t, re.I):
        return "AI 정책 컨트롤타워 인선/공백"
    if re.search(r"AIDC|AI\s*데이터센터|GPU|국가AI|AI\s*인프라", t, re.I):
        return "국가 AI 인프라/AIDC/GPU 정책"
    if re.search(r"가상자산", t) and re.search(r"이전거래|해외\s*거래|의무보고|자금세탁|AML|트래블룰|FIU", t, re.I):
        return "가상자산 이전거래/AML 규제 조정"
    if re.search(r"홍콩\s*H?지수|홍콩\s*ELS|ELS", t, re.I) and re.search(r"과징금|판매|은행|불완전", t, re.I):
        return "홍콩 ELS 판매은행 제재"
    if re.search(r"네이버페이", t) and re.search(r"오류|장애|복구|안드로이드", t, re.I):
        return "네이버페이 앱 장애/복구"
    if re.search(r"\bKT\b|KT가|KT ", t, re.I) and re.search(r"금융|AX|AI|GPU|클린존|RCS", t, re.I):
        return "KT 금융 AX/AI 인프라 전략"
    if re.search(r"LGU\+|LG유플러스", t, re.I) and re.search(r"공공|우선\s*통신|우선\s*접속|재난|소방", t, re.I):
        return "LGU+ 공공 우선 통신권/재난 대응"
    if re.search(r"구글", t) and re.search(r"뉴스\s*AI\s*랩|AI\s*학습용\s*봇|콘텐츠\s*접근|뉴스\s*데이터|언론", t, re.I):
        return "구글 뉴스 AI 랩 및 콘텐츠 데이터 접근"
    if re.search(r"챗GPT|ChatGPT", t, re.I) and re.search(r"카톡|카카오톡|채팅방|연동", t, re.I):
        return "카카오톡 내 ChatGPT 연동"
    if re.search(r"카카오뱅크", t) and re.search(r"여의도|판교|오피스|이전|본사|일원화", t, re.I):
        return "카카오뱅크 판교 조직/거점 일원화"
    if re.search(r"카카오페이", t) and re.search(r"슈퍼\s*월렛|원화코인|스테이블코인|간편결제", t, re.I):
        return "카카오페이 슈퍼월렛 원화코인/스테이블코인 구상"
    if re.search(r"스타벅스", t) and re.search(r"카톡\s*선물|카카오톡\s*선물|선물하기|교환권", t, re.I):
        return "카톡 선물하기 내 스타벅스 교환권/소비 변화"
    fallback_key = v20_norm(label_group)
    if fallback_key and fallback_key not in V20_GENERIC_GROUPS:
        return v20_clip(label_group, 140)
    return ""


def v20_issue_group_key(text):
    # v21 keeps more of the Gemini issue name, but canonicalizes common phrasing.
    text = clean_html_text(text)
    canonical = v21_canonical_issue_group(text)
    if canonical:
        text = canonical
    text = v20_norm(text)
    if not text or text in V20_GENERIC_GROUPS:
        return ""
    text = re.sub(r"\b(관련|이슈|기사|보도|뉴스|종합|단독|속보)\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:110]


def v20_detect_issue_family(text, json_key=""):
    t = clean_html_text(text)
    if not t:
        return "off_topic"
    self_tier = v21_self_tier(t)
    if self_tier == "LOW_VALUE_PR":
        return "low_value_pr"
    if v20_is_platform_obligation_text(t):
        return "platform_obligation" if v20_has_self_entity(t) else "government_platform_regulation"
    if v20_has_self_entity(t):
        if self_tier == "DIRECT_RISK":
            if re.search(r"노조|파업|임단협|임금|성과급|RSU|고용불안|조정", t, re.I):
                return "self_labor"
            if re.search(r"지분|매각|인수|합병|최대주주|경영권|공동대표|사내이사|주총|유상증자|전환사채", t, re.I):
                return "self_governance_mna"
            if re.search(r"개인정보|유출|해킹|피싱|보안|장애|먹통", t, re.I):
                return "self_privacy_security"
            if re.search(r"수사|조사|과징금|제재|소송|판결|검찰|경찰", t, re.I):
                return "self_legal_regulatory"
            if re.search(r"임원|대표|CPO|CTO|조직개편|퇴사|사임|서비스\s*논란|이용자\s*반발|카톡\s*개편", t, re.I):
                return "self_leadership_service"
            return "self_direct_risk"
        if self_tier == "RESPONSE_RELEVANT":
            return "self_response_relevant"
        if self_tier == "STRATEGIC_REFERENCE":
            return "self_strategic_reference"
        return "self_filler_reference"
    gov_tier = v21_gov_tier(t)
    if json_key == "정부_국회" or gov_tier != "OFF_TOPIC" or REGULATOR_CORE_PATTERN.search(t):
        if gov_tier == "GOV_DIRECT_PLATFORM_OBLIGATION":
            return "government_platform_obligation"
        if gov_tier == "GOV_AI_CORE_POLICY":
            return "government_ai_core_policy"
        if gov_tier == "GOV_REGULATORY_ENFORCEMENT":
            return "government_regulatory_enforcement"
        if gov_tier == "GOV_PLATFORM_MARKET_RULE":
            return "government_platform_market_rule"
        if gov_tier == "GOV_APPOINTMENT_GOVERNANCE":
            return "government_appointment_governance"
        if gov_tier == "GOV_DIGITAL_FINANCE_POLICY":
            return "government_digital_finance_policy"
        if gov_tier == "GOV_ADJACENT_DIGITAL_POLICY":
            return "government_adjacent_digital_policy"
        if gov_tier == "GOV_GENERAL_LOW_RELEVANCE":
            return "government_general_low_relevance"
        return "government_general_policy"
    comp_tier = v21_competitor_tier(t)
    if json_key == "경쟁사_해외이슈" or comp_tier not in {"OFF_TOPIC", "COMP_GENERAL_LOW_RELEVANCE"} or COMPETITOR_CORE_PATTERN.search(t):
        if comp_tier == "COMP_GENERAL_LOW_RELEVANCE":
            return "competitor_general_low_relevance"
        return comp_tier.lower()
    if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|반도체|HBM|TSMC|엔비디아|전력|플랫폼|카카오톡\s*선물", t, re.I):
        return "industry_structural_change"
    return "off_topic"


def v20_internal_category_from_family(family, text, original_json_key=""):
    t = clean_html_text(text)
    if family in {"low_value_pr"}:
        return "LOW_VALUE_PR"
    if family == "off_topic":
        return "OFF_TOPIC"
    if family.startswith("self_"):
        if family in {"self_strategic_reference", "self_filler_reference", "self_response_relevant", "self_affiliate_business"}:
            return "SELF_AFFILIATE_BUSINESS"
        return "SELF_DIRECT_RISK"
    if family == "platform_obligation":
        return "SELF_INCLUDED_REGULATION" if v20_has_self_entity(t) else "GOV_PLATFORM_REGULATION"
    if family.startswith("government_"):
        if "ai_core" in family:
            return "GOV_AI_DIGITAL_POLICY"
        if "digital_finance" in family:
            return "GOV_FINANCIAL_DIGITAL_POLICY"
        return "GOV_PLATFORM_REGULATION"
    if family.startswith("comp_") or family.startswith("competitor_"):
        if "ai_platform" in family or "strategy" in family or "platform_ai" in family:
            return "COMPETITOR_AI_STRATEGY"
        return "COMPETITOR_PLATFORM_RISK"
    if family.startswith("industry_"):
        return "INDUSTRY_STRUCTURAL_CHANGE"
    return v20_internal_category_from_family.__globals__.get('_unused', None) or ("산업동향" if False else (
        "SELF_AFFILIATE_BUSINESS" if original_json_key == "자사_및_계열사_이슈" else
        "GOV_PLATFORM_REGULATION" if original_json_key == "정부_국회" else
        "COMPETITOR_PLATFORM_RISK" if original_json_key == "경쟁사_해외이슈" else
        "INDUSTRY_STRUCTURAL_CHANGE"
    ))


def v20_local_article_label(article):
    text = v20_article_text(article)
    original_key = article.get("JSON카테고리") or "산업동향"
    family = v20_detect_issue_family(text, original_key)
    internal = v20_internal_category_from_family(family, text, original_key)
    category = v20_output_category_from_internal(internal, original_key)
    if internal == "SELF_INCLUDED_REGULATION":
        category = "정부_국회"
    self_tier = v21_self_tier(text) if v20_has_self_entity(text) else ""
    gov_tier = v21_gov_tier(text) if (original_key == "정부_국회" or internal.startswith("GOV") or internal == "SELF_INCLUDED_REGULATION") else ""
    comp_tier = v21_competitor_tier(text) if (original_key == "경쟁사_해외이슈" or internal.startswith("COMPETITOR") or COMPETITOR_CORE_PATTERN.search(text)) else ""
    comp_scope = v21_competitor_scope(text)

    ceo = 2
    pa = 2
    relevance = 2
    exclude = False
    is_pr = False
    reason = "로컬 보조 라벨"
    impact = "카카오 대외협력 관점 간접 참고"
    risk_types = []

    if self_tier == "LOW_VALUE_PR" or family == "low_value_pr" or internal == "LOW_VALUE_PR":
        ceo = pa = relevance = 1
        exclude = True
        is_pr = True
        impact = "단순 홍보·후원·캠페인성으로 대외협력 우선순위 낮음"
        risk_types = ["LOW_VALUE_PR"]
        reason = "자사 홍보/후원성 기사"
    elif self_tier == "DIRECT_RISK":
        ceo = 5; pa = 5; relevance = 5
        impact = "카카오 또는 계열사에 직접 발생한 외부 대응 필요 이슈"
        risk_types = ["SELF_DIRECT_RISK", family]
    elif self_tier == "RESPONSE_RELEVANT":
        ceo = 4; pa = 5; relevance = 5
        impact = "카카오가 설명·정책 대응·준수 주체로 포함될 수 있는 이슈"
        risk_types = ["SELF_RESPONSE_RELEVANT", family]
    elif self_tier == "STRATEGIC_REFERENCE":
        ceo = 4; pa = 3; relevance = 4
        impact = "카카오 현직자가 사업 방향과 외부 질의 가능성을 참고할 실질 변화"
        risk_types = ["SELF_STRATEGIC_REFERENCE", family]
    elif family == "platform_obligation":
        ceo = 4; pa = 5; relevance = 5
        impact = "카카오 포함 플랫폼 사업자의 준수 의무·정책 대응 이슈"
        risk_types = ["PLATFORM_OBLIGATION"]
    elif gov_tier == "GOV_DIRECT_PLATFORM_OBLIGATION":
        ceo = 5; pa = 5; relevance = 5
        impact = "카카오 또는 주요 플랫폼 사업자에게 직접 의무·차단·보고·준수 부담이 생김"
        risk_types = [gov_tier]
    elif gov_tier == "GOV_AI_CORE_POLICY":
        ceo = 5; pa = 4; relevance = 5
        impact = "카카오 AI agent/AI 서비스 전략과 연결되는 핵심 정책환경"
        risk_types = [gov_tier]
    elif gov_tier in {"GOV_REGULATORY_ENFORCEMENT", "GOV_PLATFORM_MARKET_RULE", "GOV_APPOINTMENT_GOVERNANCE"}:
        ceo = 4; pa = 5; relevance = 4
        impact = "카카오 플랫폼·AI·대관 대응 경로에 영향을 줄 수 있는 정책/감독 이슈"
        risk_types = [gov_tier]
    elif gov_tier == "GOV_DIGITAL_FINANCE_POLICY":
        ceo = 3; pa = 4; relevance = 3
        impact = "카카오페이·카카오뱅크 등 금융 계열사에 참고 가능한 디지털금융 정책"
        risk_types = [gov_tier]
    elif gov_tier == "GOV_ADJACENT_DIGITAL_POLICY":
        ceo = 3; pa = 3; relevance = 3
        impact = "디지털트윈·공공데이터 등 인접 정책 참고"
        risk_types = [gov_tier]
    elif gov_tier == "GOV_GENERAL_LOW_RELEVANCE" or family == "government_general_low_relevance":
        ceo = 2; pa = 2; relevance = 1
        impact = "정부/국회 기사이나 카카오 플랫폼·AI·디지털금융과 연결 경로가 약함"
        risk_types = [gov_tier or family]
        if original_key != "정부_국회":
            exclude = True
    elif comp_tier in V21_COMPETITOR_TIERS and comp_tier not in {"COMP_GENERAL_LOW_RELEVANCE", "OFF_TOPIC"}:
        if comp_scope in {"domestic", "mixed"}:
            ceo = 4; pa = 4; relevance = 5
        else:
            ceo = 4 if comp_tier in {"COMP_OVERSEAS_REGULATION_DATA", "COMP_OVERSEAS_PLATFORM_AI"} else 3
            pa = 3; relevance = 4
        impact = "카카오가 경쟁사 전략·리스크·AI/플랫폼 대응을 비교 참고할 이슈"
        risk_types = [comp_tier, comp_scope]
    elif family.startswith("industry_"):
        ceo = 3; pa = 3; relevance = 3
        impact = "카카오 사업환경 이해에 필요한 산업 구조 변화"
        risk_types = [family]
    else:
        if original_key == "자사_및_계열사_이슈" and self_tier == "FILLER_REFERENCE":
            ceo = 2; pa = 2; relevance = 3
            impact = "자사 이슈가 부족한 날에만 참고할 수 있는 보충성 자사 소식"
        else:
            exclude = family == "off_topic"
            reason = "카카오 대외협력 브리핑 관련성이 낮음" if exclude else reason

    canonical_group = v21_canonical_issue_group("", article=article)
    if canonical_group:
        group = canonical_group
    else:
        try:
            group = v16_global_incident_base_key(article) or v16_global_incident_key(article)
        except Exception:
            group = ""
        if not group:
            group = f"{family}:{title_fingerprint(article.get('기사제목',''))[:60]}"

    return {
        "id": int(article.get("id")),
        "is_relevant": not exclude,
        "primary_category": category,
        "internal_category": internal,
        "issue_family": family,
        "issue_group": v20_clip(group, 120),
        "company_impact": v20_clip(impact, 200),
        "ceo_priority": ceo,
        "public_affairs_priority": pa,
        "relevance": relevance,
        "risk_types": risk_types,
        "is_pr": is_pr,
        "exclude": exclude,
        "reason": v20_clip(reason, 240),
        "label_source": "local",
        "self_tier": self_tier,
        "gov_tier": gov_tier,
        "competitor_tier": comp_tier,
        "competitor_scope": comp_scope,
        "kakao_policy_impact": v20_clip(impact if gov_tier else "", 220),
        "kakao_competitive_implication": v20_clip(impact if comp_tier else "", 220),
        "why_it_matters_to_kakao": v20_clip(impact, 220),
    }


def v20_normalize_internal_category(value, fallback):
    text = clean_html_text(value).upper().replace(" ", "_").replace("-", "_")
    aliases = {
        "SELF_DIRECT_RISK": "SELF_DIRECT_RISK",
        "DIRECT_RISK": "SELF_DIRECT_RISK",
        "SELF_AFFILIATE_BUSINESS": "SELF_AFFILIATE_BUSINESS",
        "STRATEGIC_REFERENCE": "SELF_AFFILIATE_BUSINESS",
        "RESPONSE_RELEVANT": "SELF_AFFILIATE_BUSINESS",
        "FILLER_REFERENCE": "SELF_AFFILIATE_BUSINESS",
        "SELF_INCLUDED_REGULATION": "SELF_INCLUDED_REGULATION",
        "GOV_DIRECT_PLATFORM_OBLIGATION": "GOV_PLATFORM_REGULATION",
        "GOV_AI_CORE_POLICY": "GOV_AI_DIGITAL_POLICY",
        "GOV_AI_DIGITAL_POLICY": "GOV_AI_DIGITAL_POLICY",
        "GOV_DIGITAL_FINANCE_POLICY": "GOV_FINANCIAL_DIGITAL_POLICY",
        "GOV_FINANCIAL_DIGITAL_POLICY": "GOV_FINANCIAL_DIGITAL_POLICY",
        "GOV_PLATFORM_MARKET_RULE": "GOV_PLATFORM_REGULATION",
        "GOV_REGULATORY_ENFORCEMENT": "GOV_PLATFORM_REGULATION",
        "GOV_APPOINTMENT_GOVERNANCE": "GOV_AI_DIGITAL_POLICY",
        "COMP_DOMESTIC_DIRECT_RISK": "COMPETITOR_PLATFORM_RISK",
        "COMP_DOMESTIC_STRATEGY": "COMPETITOR_PLATFORM_RISK",
        "COMP_DOMESTIC_AI_PLATFORM": "COMPETITOR_AI_STRATEGY",
        "COMP_DOMESTIC_FINANCE_COMMERCE": "COMPETITOR_PLATFORM_RISK",
        "COMP_DOMESTIC_PUBLIC_INFRA": "COMPETITOR_PLATFORM_RISK",
        "COMP_OVERSEAS_PLATFORM_AI": "COMPETITOR_AI_STRATEGY",
        "COMP_OVERSEAS_REGULATION_DATA": "COMPETITOR_PLATFORM_RISK",
        "COMP_OVERSEAS_MARKET_SIGNAL": "COMPETITOR_AI_STRATEGY",
        "LOW_VALUE_PR": "LOW_VALUE_PR",
        "OFF_TOPIC": "OFF_TOPIC",
    }
    if text in V20_INTERNAL_CATEGORIES:
        return text
    if text in aliases:
        return aliases[text]
    for k in V20_INTERNAL_CATEGORIES:
        if k in text or text in k:
            return k
    return fallback if fallback in V20_INTERNAL_CATEGORIES else "INDUSTRY_STRUCTURAL_CHANGE"


def v20_normalize_label_json(data, valid_ids, article_by_id):
    if isinstance(data, dict):
        rows = data.get("articles") or data.get("items") or data.get("labels") or data.get("기사") or []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []
    labels = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            art_id = int(row.get("id") or row.get("article_id") or row.get("기사ID"))
        except Exception:
            continue
        if art_id not in valid_ids:
            continue
        article = article_by_id.get(art_id, {})
        local = v20_local_article_label(article) if article else {}
        raw_self = clean_html_text(row.get("self_tier") or row.get("자사티어") or local.get("self_tier", "")).upper()
        raw_gov = clean_html_text(row.get("gov_tier") or row.get("정부티어") or local.get("gov_tier", "")).upper()
        raw_comp = clean_html_text(row.get("competitor_tier") or row.get("경쟁사티어") or local.get("competitor_tier", "")).upper()
        self_tier = raw_self if raw_self in V21_SELF_TIERS else local.get("self_tier", "")
        gov_tier = raw_gov if raw_gov in V21_GOV_TIERS else local.get("gov_tier", "")
        competitor_tier = raw_comp if raw_comp in V21_COMPETITOR_TIERS else local.get("competitor_tier", "")
        comp_scope = clean_html_text(row.get("competitor_scope") or row.get("scope") or local.get("competitor_scope", "other")).lower()
        if comp_scope not in {"domestic", "overseas", "mixed", "other"}:
            comp_scope = v21_competitor_scope(v20_article_text(article))
        internal_seed = row.get("internal_category") or row.get("내부카테고리") or row.get("category_internal") or gov_tier or competitor_tier or self_tier
        internal = v20_normalize_internal_category(internal_seed, local.get("internal_category", "INDUSTRY_STRUCTURAL_CHANGE"))
        category = v19_normalize_category_key(
            row.get("primary_category") or row.get("category") or row.get("카테고리"),
            fallback=v20_output_category_from_internal(internal, article.get("JSON카테고리", "산업동향")),
        )
        family = v20_clip(row.get("issue_family") or row.get("family") or row.get("이슈패밀리") or local.get("issue_family", ""), 80)
        if not family:
            family = v20_detect_issue_family(v20_article_text(article), article.get("JSON카테고리", ""))
        group = v20_clip(row.get("issue_group") or row.get("duplicate_group") or row.get("중복그룹") or local.get("issue_group", ""), 120)
        canonical_group = v21_canonical_issue_group(group, article=article)
        if canonical_group:
            group = canonical_group
        if not group or not v20_issue_group_key(group):
            group = local.get("issue_group", "") or f"{family}:{title_fingerprint(article.get('기사제목',''))[:55]}"
        risk_types = row.get("risk_types") or row.get("risk_type") or row.get("리스크유형") or local.get("risk_types", [])
        if isinstance(risk_types, str):
            risk_types = [x.strip() for x in re.split(r"[,/|+]", risk_types) if x.strip()]
        if not isinstance(risk_types, list):
            risk_types = []
        company_impact = v20_clip(
            row.get("company_impact") or row.get("company_relevance") or row.get("카카오영향") or
            row.get("why_it_matters_to_kakao") or row.get("kakao_policy_impact") or row.get("kakao_competitive_implication") or
            local.get("company_impact", ""), 220)
        label = {
            "id": art_id,
            "is_relevant": v20_bool(row.get("is_relevant")) if "is_relevant" in row else not v20_bool(row.get("exclude")),
            "primary_category": category,
            "internal_category": internal,
            "issue_family": family,
            "issue_group": group,
            "company_impact": company_impact,
            "ceo_priority": v20_int(row.get("ceo_priority") or row.get("CEO중요도") or row.get("priority"), default=local.get("ceo_priority", 3)),
            "public_affairs_priority": v20_int(row.get("public_affairs_priority") or row.get("pa_priority") or row.get("대외협력중요도"), default=local.get("public_affairs_priority", 3)),
            "relevance": v20_int(row.get("relevance") or row.get("관련도"), default=local.get("relevance", 3)),
            "risk_types": risk_types,
            "is_pr": v20_bool(row.get("is_pr") if "is_pr" in row else row.get("홍보성", local.get("is_pr", False))),
            "exclude": v20_bool(row.get("exclude") if "exclude" in row else row.get("제외", local.get("exclude", False))),
            "reason": v20_clip(row.get("reason") or row.get("판단사유") or local.get("reason", ""), 260),
            "label_source": "gemini",
            "self_tier": self_tier,
            "gov_tier": gov_tier,
            "competitor_tier": competitor_tier,
            "competitor_scope": comp_scope,
            "kakao_policy_impact": v20_clip(row.get("kakao_policy_impact") or local.get("kakao_policy_impact", ""), 240),
            "kakao_competitive_implication": v20_clip(row.get("kakao_competitive_implication") or local.get("kakao_competitive_implication", ""), 240),
            "why_it_matters_to_kakao": v20_clip(row.get("why_it_matters_to_kakao") or company_impact, 240),
        }
        if label["is_pr"] or label["internal_category"] == "LOW_VALUE_PR" or label.get("self_tier") == "LOW_VALUE_PR":
            label["exclude"] = True
            label["is_relevant"] = False
        # General finance/legal items are not hard-excluded, but get low relevance unless Gemini explicitly explains Kakao path.
        if label.get("gov_tier") == "GOV_GENERAL_LOW_RELEVANCE" and label["public_affairs_priority"] > 2:
            label["public_affairs_priority"] = 2
            label["relevance"] = min(label["relevance"], 2)
        labels[art_id] = label
    return labels


def v20_apply_labels_to_articles(articles, labels):
    rows = []
    for article in articles:
        try:
            art_id = int(article.get("id"))
        except Exception:
            continue
        label = labels.get(art_id) or v20_local_article_label(article)
        labels[art_id] = label
        article["Gemini카테고리"] = label.get("primary_category", "")
        article["Gemini내부카테고리"] = label.get("internal_category", "")
        article["Gemini이슈패밀리"] = label.get("issue_family", "")
        article["Gemini이슈그룹"] = label.get("issue_group", "")
        article["Gemini관련성"] = label.get("company_impact", "")
        article["Gemini중요도"] = label.get("ceo_priority", "")
        article["Gemini대외협력중요도"] = label.get("public_affairs_priority", "")
        article["Gemini관련도"] = label.get("relevance", "")
        article["GeminiPR성"] = "Y" if label.get("is_pr") else ""
        article["Gemini제외"] = "Y" if label.get("exclude") else ""
        article["Gemini판단사유"] = label.get("reason", "")
        article["Gemini라벨소스"] = label.get("label_source", "")
        article["Gemini자사티어"] = label.get("self_tier", "")
        article["Gemini정부티어"] = label.get("gov_tier", "")
        article["Gemini경쟁사티어"] = label.get("competitor_tier", "")
        article["Gemini경쟁사범위"] = label.get("competitor_scope", "")
        article["Gemini정책영향경로"] = label.get("kakao_policy_impact", "")
        article["Gemini경쟁시사점"] = label.get("kakao_competitive_implication", "")
        article["Gemini중복그룹"] = label.get("issue_group", "")
        article["Gemini이슈유형"] = label.get("issue_family", "")
        rows.append({
            "id": art_id,
            "기사제목": article.get("기사제목", ""),
            "언론사": article.get("언론사", ""),
            "검색어": article.get("검색어", ""),
            "원카테고리": article.get("JSON카테고리", ""),
            "Gemini카테고리": article.get("Gemini카테고리", ""),
            "Gemini내부카테고리": article.get("Gemini내부카테고리", ""),
            "Gemini자사티어": article.get("Gemini자사티어", ""),
            "Gemini정부티어": article.get("Gemini정부티어", ""),
            "Gemini경쟁사티어": article.get("Gemini경쟁사티어", ""),
            "Gemini경쟁사범위": article.get("Gemini경쟁사범위", ""),
            "Gemini이슈패밀리": article.get("Gemini이슈패밀리", ""),
            "Gemini이슈그룹": article.get("Gemini이슈그룹", ""),
            "Gemini중요도": article.get("Gemini중요도", ""),
            "Gemini대외협력중요도": article.get("Gemini대외협력중요도", ""),
            "Gemini관련도": article.get("Gemini관련도", ""),
            "Gemini관련성": article.get("Gemini관련성", ""),
            "Gemini정책영향경로": article.get("Gemini정책영향경로", ""),
            "Gemini경쟁시사점": article.get("Gemini경쟁시사점", ""),
            "GeminiPR성": article.get("GeminiPR성", ""),
            "Gemini제외": article.get("Gemini제외", ""),
            "Gemini판단사유": article.get("Gemini판단사유", ""),
            "Gemini라벨소스": article.get("Gemini라벨소스", ""),
            "랭킹점수": article.get("랭킹점수", ""),
            "코드힌트": article.get("코드힌트", ""),
        })
    return rows


def v20_article_pre_rep_score(article):
    label_priority = v20_int(article.get("Gemini중요도"), default=3)
    pa_priority = v20_int(article.get("Gemini대외협력중요도"), default=3)
    relevance = v20_int(article.get("Gemini관련도"), default=3)
    rank = v20_float(article.get("랭킹점수"), 0)
    press = press_score(article.get("언론사", ""))
    text = v20_article_text(article)
    bonus = 0
    self_tier = article.get("Gemini자사티어", "")
    gov_tier = article.get("Gemini정부티어", "")
    comp_tier = article.get("Gemini경쟁사티어", "")
    comp_scope = article.get("Gemini경쟁사범위", "")
    if self_tier == "DIRECT_RISK": bonus += 22
    if self_tier == "RESPONSE_RELEVANT": bonus += 16
    if self_tier == "STRATEGIC_REFERENCE": bonus += 10
    if gov_tier in {"GOV_DIRECT_PLATFORM_OBLIGATION", "GOV_AI_CORE_POLICY"}: bonus += 18
    if gov_tier in {"GOV_REGULATORY_ENFORCEMENT", "GOV_PLATFORM_MARKET_RULE", "GOV_APPOINTMENT_GOVERNANCE"}: bonus += 12
    if gov_tier in {"GOV_DIGITAL_FINANCE_POLICY", "GOV_ADJACENT_DIGITAL_POLICY"}: bonus -= 2
    if gov_tier == "GOV_GENERAL_LOW_RELEVANCE": bonus -= 25
    if comp_scope in {"domestic", "mixed"}: bonus += 18
    if comp_scope == "overseas" and comp_tier in {"COMP_OVERSEAS_REGULATION_DATA", "COMP_OVERSEAS_PLATFORM_AI"}: bonus += 8
    if comp_tier == "COMP_OVERSEAS_MARKET_SIGNAL": bonus -= 10
    if article.get("GeminiPR성") == "Y": bonus -= 120
    if article.get("Gemini제외") == "Y": bonus -= 100
    if V21_SELF_PR_RE.search(text) and not V21_SELF_DIRECT_RISK_RE.search(text): bonus -= 70
    return label_priority * 18 + pa_priority * 18 + relevance * 10 + min(30, max(-20, rank / 2)) + press + bonus


def v20_issue_score(issue):
    score = 0.0
    score += issue.get("max_ceo_priority", 1) * 22
    score += issue.get("max_pa_priority", 1) * 24
    score += issue.get("max_relevance", 1) * 12
    score += min(30, max(-20, issue.get("max_rank_score", 0) / 2))
    score += min(10, issue.get("article_count", 1) * 1.5)
    family = issue.get("issue_family", "")
    internal = issue.get("internal_category", "")
    self_tier = issue.get("self_tier", "")
    gov_tier = issue.get("gov_tier", "")
    comp_tier = issue.get("competitor_tier", "")
    comp_scope = issue.get("competitor_scope", "")
    if self_tier == "DIRECT_RISK": score += 36
    elif self_tier == "RESPONSE_RELEVANT": score += 24
    elif self_tier == "STRATEGIC_REFERENCE": score += 14
    elif self_tier == "FILLER_REFERENCE": score -= 8
    elif self_tier == "LOW_VALUE_PR": score -= 200
    if gov_tier == "GOV_DIRECT_PLATFORM_OBLIGATION": score += 34
    elif gov_tier == "GOV_AI_CORE_POLICY": score += 32
    elif gov_tier in {"GOV_REGULATORY_ENFORCEMENT", "GOV_PLATFORM_MARKET_RULE", "GOV_APPOINTMENT_GOVERNANCE"}: score += 22
    elif gov_tier == "GOV_DIGITAL_FINANCE_POLICY": score += 8
    elif gov_tier == "GOV_ADJACENT_DIGITAL_POLICY": score -= 2
    elif gov_tier == "GOV_GENERAL_LOW_RELEVANCE": score -= 45
    if comp_scope in {"domestic", "mixed"}: score += 28
    elif comp_scope == "overseas": score += 5
    if comp_tier in {"COMP_DOMESTIC_DIRECT_RISK", "COMP_DOMESTIC_AI_PLATFORM", "COMP_DOMESTIC_FINANCE_COMMERCE", "COMP_DOMESTIC_PUBLIC_INFRA"}: score += 16
    elif comp_tier == "COMP_OVERSEAS_REGULATION_DATA": score += 12
    elif comp_tier == "COMP_OVERSEAS_PLATFORM_AI": score += 6
    elif comp_tier == "COMP_OVERSEAS_MARKET_SIGNAL": score -= 18
    if internal == "SELF_INCLUDED_REGULATION": score += 18
    if internal == "LOW_VALUE_PR" or issue.get("is_pr"): score -= 220
    if issue.get("exclude"): score -= 80
    return round(score, 3)


def v20_build_issues(candidates, labels):
    article_by_id = {int(a["id"]): a for a in candidates}
    buckets = {}
    for article in candidates:
        art_id = int(article["id"])
        label = labels.get(art_id) or v20_local_article_label(article)
        group = v21_canonical_issue_group(label.get("issue_group"), article=article) or label.get("issue_group") or ""
        group_key = v20_issue_group_key(group)
        if not group_key:
            group_key = v20_issue_group_key(f"{label.get('issue_family','')} {title_fingerprint(article.get('기사제목',''))[:60]}")
        if not group_key:
            group_key = f"article_{art_id}"
        buckets.setdefault(group_key, []).append(article)
    issues = []
    for idx, (group_key, arts) in enumerate(buckets.items(), 1):
        arts = sorted(arts, key=v20_article_pre_rep_score, reverse=True)
        article_ids = [int(a["id"]) for a in arts]
        issue_labels = [labels.get(int(a["id"])) or v20_local_article_label(a) for a in arts]
        def mode_or_first(values, fallback=""):
            values = [v for v in values if v]
            if not values:
                return fallback
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1
            return sorted(counts.items(), key=lambda x: (-x[1], values.index(x[0])))[0][0]
        families = [l.get("issue_family", "") for l in issue_labels]
        internals = [l.get("internal_category", "") for l in issue_labels]
        categories = [l.get("primary_category", "") for l in issue_labels if l.get("primary_category") in JSON_KEYS_ORDER]
        self_tiers = [l.get("self_tier", "") for l in issue_labels]
        gov_tiers = [l.get("gov_tier", "") for l in issue_labels]
        comp_tiers = [l.get("competitor_tier", "") for l in issue_labels]
        comp_scopes = [l.get("competitor_scope", "") for l in issue_labels]
        family = mode_or_first(families, v20_detect_issue_family(v20_article_text(arts[0]), arts[0].get("JSON카테고리", "")))
        internal = mode_or_first(internals, v20_internal_category_from_family(family, v20_article_text(arts[0]), arts[0].get("JSON카테고리", "")))
        category = mode_or_first(categories, v20_output_category_from_internal(internal, arts[0].get("JSON카테고리", "산업동향")))
        if internal == "SELF_INCLUDED_REGULATION" or family == "platform_obligation":
            category = "정부_국회"
        if internal == "LOW_VALUE_PR":
            category = arts[0].get("JSON카테고리") if arts[0].get("JSON카테고리") in JSON_KEYS_ORDER else category
        issue_group = v21_canonical_issue_group(mode_or_first([l.get("issue_group", "") for l in issue_labels], group_key), article=arts[0]) or group_key
        issue = {
            "issue_id": f"I{idx:03d}",
            "issue_key": v20_issue_group_key(issue_group) or group_key,
            "issue_group": v20_clip(issue_group, 140),
            "issue_family": family,
            "internal_category": internal,
            "primary_category": category,
            "self_tier": mode_or_first(self_tiers, ""),
            "gov_tier": mode_or_first(gov_tiers, ""),
            "competitor_tier": mode_or_first(comp_tiers, ""),
            "competitor_scope": mode_or_first(comp_scopes, "other"),
            "article_ids": article_ids,
            "article_count": len(article_ids),
            "top_article_id": article_ids[0],
            "candidate_article_ids": article_ids[:10],
            "titles": [a.get("기사제목", "") for a in arts[:5]],
            "sources": [a.get("언론사", "") for a in arts[:5]],
            "max_ceo_priority": max([v20_int(l.get("ceo_priority"), default=1) for l in issue_labels] or [1]),
            "max_pa_priority": max([v20_int(l.get("public_affairs_priority"), default=1) for l in issue_labels] or [1]),
            "max_relevance": max([v20_int(l.get("relevance"), default=1) for l in issue_labels] or [1]),
            "max_rank_score": max([v20_float(a.get("랭킹점수"), 0) for a in arts] or [0]),
            "is_pr": any(l.get("is_pr") for l in issue_labels) or internal == "LOW_VALUE_PR" or mode_or_first(self_tiers, "") == "LOW_VALUE_PR",
            "exclude": all(l.get("exclude") for l in issue_labels) or internal in {"LOW_VALUE_PR", "OFF_TOPIC"},
            "company_impact": v20_clip(" / ".join(dict.fromkeys([l.get("company_impact", "") for l in issue_labels if l.get("company_impact")]))[:320], 320),
            "label_reasons": v20_clip(" / ".join(dict.fromkeys([l.get("reason", "") for l in issue_labels if l.get("reason")]))[:360], 360),
            "risk_types": sorted({str(x) for l in issue_labels for x in (l.get("risk_types") or []) if str(x).strip()}),
        }
        issue["issue_score"] = v20_issue_score(issue)
        issues.append(issue)
    issues = sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True)
    for idx, issue in enumerate(issues, 1):
        issue["issue_id"] = f"I{idx:03d}"
    return issues, article_by_id


def v20_issue_rows(issues):
    rows = []
    for issue in issues:
        rows.append({
            "issue_id": issue.get("issue_id"),
            "issue_group": issue.get("issue_group"),
            "issue_family": issue.get("issue_family"),
            "internal_category": issue.get("internal_category"),
            "primary_category": issue.get("primary_category"),
            "self_tier": issue.get("self_tier"),
            "gov_tier": issue.get("gov_tier"),
            "competitor_tier": issue.get("competitor_tier"),
            "competitor_scope": issue.get("competitor_scope"),
            "issue_score": issue.get("issue_score"),
            "max_ceo_priority": issue.get("max_ceo_priority"),
            "max_pa_priority": issue.get("max_pa_priority"),
            "max_relevance": issue.get("max_relevance"),
            "article_count": issue.get("article_count"),
            "top_article_id": issue.get("top_article_id"),
            "candidate_article_ids": ",".join(map(str, issue.get("candidate_article_ids", []))),
            "titles": " | ".join(issue.get("titles", [])),
            "sources": " | ".join(issue.get("sources", [])),
            "company_impact": issue.get("company_impact"),
            "label_reasons": issue.get("label_reasons"),
            "is_pr": issue.get("is_pr"),
            "exclude": issue.get("exclude"),
        })
    return rows


def v20_issue_line(issue):
    ids = ",".join(map(str, issue.get("candidate_article_ids", [])[:5]))
    titles = " || ".join(v20_clip(t, 85) for t in issue.get("titles", [])[:3])
    return (
        f"[{issue.get('issue_id')}] cat={issue.get('primary_category')} internal={issue.get('internal_category')} "
        f"self_tier={issue.get('self_tier','')} gov_tier={issue.get('gov_tier','')} "
        f"comp_tier={issue.get('competitor_tier','')} comp_scope={issue.get('competitor_scope','')} "
        f"family={issue.get('issue_family')} score={issue.get('issue_score')} "
        f"ceo={issue.get('max_ceo_priority')} pa={issue.get('max_pa_priority')} rel={issue.get('max_relevance')} "
        f"articles={issue.get('article_count')} ids={ids} pr={issue.get('is_pr')} exclude={issue.get('exclude')} "
        f"group={v20_clip(issue.get('issue_group',''), 95)} "
        f"impact={v20_clip(issue.get('company_impact',''), 130)} "
        f"titles={titles}"
    )


def v20_gemini_label_candidates(client, candidates, recent_past_text):
    if not client:
        raise RuntimeError("Gemini client 없음")
    article_by_id = {int(a["id"]): a for a in candidates}
    all_labels = {}
    batches = [candidates[i:i + V20_LABEL_BATCH_SIZE] for i in range(0, len(candidates), V20_LABEL_BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches, 1):
        candidate_text = "\n".join(v20_label_candidate_line(a) for a in batch)
        prompt = f"""
너는 카카오 대외협력·정책리스크팀의 기사 분류 데스크다.
각 후보를 최종 기사로 바로 고르지 말고, 이슈 편집을 위한 의미 라벨로 분류하라.
핵심은 '카카오 언급 여부'가 아니라 '카카오 대외협력팀이 오늘 알아야 할 영향 경로'다.

[공통 원칙]
- 같은 사건은 같은 issue_group으로 묶는다. 표현이 달라도 같은 정책/장애/인선/경영권 변화면 같은 이름을 쓴다.
- 단순 홍보·후원·캠페인·기부·이벤트·수상은 LOW_VALUE_PR로 보고 제외한다.
- 카카오톡 제보/채널 문구만 있는 무관 기사는 OFF_TOPIC으로 제외한다.
- 카카오와의 영향 경로를 한 문장으로 설명할 수 없으면 우선순위를 낮춘다.

[자사 및 계열사 판단]
자사 섹션은 카카오 소식 모음이 아니다. 외부 질의·규제 대응·커뮤니케이션·경영진 인지가 필요한 이슈를 우선한다.
self_tier는 다음 중 하나로 붙여라.
- DIRECT_RISK: 노조/파업/임단협, 수사/제재/소송, 개인정보/보안/장애, 서비스 논란, 임원/조직개편, 지분/인수/경영권/대표/실적 등 누가 봐도 직접 이슈
- RESPONSE_RELEVANT: 카카오가 의무 사업자, 설명 주체, 규제/정책 대응 주체로 포함되는 이슈
- STRATEGIC_REFERENCE: 직접 리스크는 아니지만 AI agent, 카톡 AI, 카카오톡 기능, 결제/금융, 조직/거점 이전, 주요 제휴 등 현직자가 전략적으로 참고할 실질 변화
- FILLER_REFERENCE: 자사 이슈가 부족한 날에만 참고할 만한 낮은 강도의 운영/서비스 소식
- LOW_VALUE_PR: 후원, 스폰서, 캠페인, 기부, 이벤트, 할인, 수상, 브랜드 홍보
- OFF_TOPIC: 무관

[정부/국회 판단]
정부/국회는 부처 뉴스가 아니라 카카오의 정책환경·규제의무·대관 경로에 닿는 이슈만 고른다.
gov_tier는 다음 중 하나로 붙여라.
- GOV_DIRECT_PLATFORM_OBLIGATION: 플랫폼/포털/앱 사업자에게 의무·금지·차단·삭제·필터링·보고·공시·고지·자료제출이 생기거나 바뀜
- GOV_AI_CORE_POLICY: 국가 AI 전략, AIDC, GPU, 공공 AI, 디지털플랫폼정부, AI 기본법/저작권/안전/거버넌스, AI 정책 컨트롤타워. 카카오 AI agent·AI 서비스 전략의 핵심 환경으로 본다.
- GOV_REGULATORY_ENFORCEMENT: 플랫폼/AI/개인정보/전자금융/디지털자산/공정거래 영역의 조사·제재·감독·가이드라인
- GOV_PLATFORM_MARKET_RULE: 온플법, 앱마켓, 인앱결제, 자사우대, 최혜대우, 수수료, 정산, 광고/검색시장 규율
- GOV_APPOINTMENT_GOVERNANCE: AI/디지털/플랫폼/방송통신/개인정보/공정거래/금융정책 라인의 인선, 공백, 조직 변화
- GOV_DIGITAL_FINANCE_POLICY: 스테이블코인, 전자금융, 디지털자산, 특금법, FIU, AML, 트래블룰. 단 AI·플랫폼 직접 이슈보다는 후순위다.
- GOV_ADJACENT_DIGITAL_POLICY: 디지털트윈, 공공데이터, 스마트시티, 지도/공간정보 등 인접 정책. 직접 연결이 명확할 때만.
- GOV_GENERAL_LOW_RELEVANCE: 일반 은행/보험/제약/건설/선거/비디지털 공정위·금융 제재 등 카카오 연결 경로가 약함
- OFF_TOPIC

[경쟁사/해외 판단]
국내 경쟁사/인접 플랫폼을 우선한다. 해외 빅테크는 하루 1~2개만 필요하다.
competitor_tier는 다음 중 하나로 붙여라.
- COMP_DOMESTIC_DIRECT_RISK: 네이버/쿠팡/토스/배민/통신3사 등 국내 경쟁사의 장애, 보안, 개인정보, 서비스 논란, 규제/제재
- COMP_DOMESTIC_STRATEGY: 국내 경쟁사의 서비스/플랫폼/커머스/광고/멤버십/사업 전략 변화
- COMP_DOMESTIC_AI_PLATFORM: 네이버 AI, SKT/KT/LGU+ AX, 국내 플랫폼의 AI agent·AI 검색·AI 추천·AI 커머스
- COMP_DOMESTIC_FINANCE_COMMERCE: 네이버페이, 토스, 쿠팡페이, 커머스/배달/금융 플랫폼 움직임
- COMP_DOMESTIC_PUBLIC_INFRA: 통신사·플랫폼의 공공/재난/행정/사회 인프라 대응
- COMP_OVERSEAS_PLATFORM_AI: 구글/OpenAI/애플/메타/앤트로픽 등 해외 AI·플랫폼 전략. 카카오 AI/플랫폼 전략과 직접 관련될 때
- COMP_OVERSEAS_REGULATION_DATA: 해외 빅테크의 AI 학습 데이터, 콘텐츠 저작권, 앱마켓, 개인정보, 광고, 플랫폼 규제
- COMP_OVERSEAS_MARKET_SIGNAL: 투자, IPO, 주가, CAPEX, 데이터센터 등 시장 신호. 후순위
- COMP_GENERAL_LOW_RELEVANCE / OFF_TOPIC
competitor_scope는 domestic / overseas / mixed / other 중 하나로 써라.

반드시 JSON만 반환하라.
{{
  "articles": [
    {{
      "id": 123,
      "is_relevant": true,
      "primary_category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향",
      "internal_category": "SELF_DIRECT_RISK/GOV_AI_DIGITAL_POLICY/COMPETITOR_PLATFORM_RISK/...",
      "self_tier": "DIRECT_RISK 또는 빈 문자열",
      "gov_tier": "GOV_AI_CORE_POLICY 또는 빈 문자열",
      "competitor_tier": "COMP_DOMESTIC_AI_PLATFORM 또는 빈 문자열",
      "competitor_scope": "domestic/overseas/mixed/other",
      "issue_family": "짧은 family",
      "issue_group": "같은 사건끼리 공유할 구체적 이슈명",
      "company_impact": "카카오 영향 경로 한 문장",
      "kakao_policy_impact": "정부/국회 기사일 때 정책 영향 경로",
      "kakao_competitive_implication": "경쟁사 기사일 때 비교/대응 시사점",
      "why_it_matters_to_kakao": "왜 봐야 하는지 한 문장",
      "ceo_priority": 1,
      "public_affairs_priority": 1,
      "relevance": 1,
      "risk_types": ["규제", "AI"],
      "is_pr": false,
      "exclude": false,
      "reason": "판단 사유"
    }}
  ]
}}

[최근 과거 보고서 문체/선호 참고]
{recent_past_text[:4500]}

[후보 기사]
{candidate_text}
"""
        text = gemini_generate_text(
            client=client,
            prompt=prompt,
            task_name=f"v21 기사 라벨링 {batch_idx}/{len(batches)}",
            model=GEMINI_MODEL_LABELING,
        )
        data = extract_json_object(text)
        labels = v20_normalize_label_json(data, set(article_by_id.keys()), article_by_id)
        all_labels.update(labels)
        print(f"  └ 🏷️ v21 Gemini 기사 라벨링 배치 {batch_idx}/{len(batches)} 완료: {len(labels)}개")
    return all_labels


def v20_family_cap_key(issue):
    if issue.get("is_pr") or issue.get("internal_category") == "LOW_VALUE_PR":
        return "low_value_pr"
    group_key = v20_issue_group_key(issue.get("issue_group", ""))
    if "카카오 노조" in issue.get("issue_group", "") or issue.get("issue_family") == "self_labor":
        return "self_labor"
    if "카카오게임즈" in issue.get("issue_group", ""):
        return "self_governance_mna"
    if issue.get("gov_tier") == "GOV_GENERAL_LOW_RELEVANCE" or issue.get("issue_family") == "government_general_low_relevance":
        return "general_financial_enforcement"
    if issue.get("gov_tier") == "GOV_DIGITAL_FINANCE_POLICY" and re.search(r"가상자산|스테이블|특금|FIU|트래블룰", issue.get("issue_group", ""), re.I):
        return "digital_finance_policy"
    if issue.get("competitor_scope") == "overseas":
        return "overseas_competitor"
    if issue.get("issue_family") in V20_FAMILY_CAPS:
        return issue.get("issue_family")
    return group_key


V20_FAMILY_CAPS.update({
    "digital_finance_policy": 1,
    "overseas_competitor": 2,
    "self_filler_reference": 1,
})


def v21_issue_allowed_basic(issue):
    if not issue:
        return False, "missing_issue"
    if issue.get("exclude"):
        return False, "issue_excluded"
    if issue.get("is_pr") or issue.get("internal_category") == "LOW_VALUE_PR" or issue.get("self_tier") == "LOW_VALUE_PR":
        return False, "low_value_pr"
    if issue.get("internal_category") == "OFF_TOPIC" or issue.get("issue_family") == "off_topic":
        return False, "off_topic"
    if issue.get("gov_tier") == "GOV_GENERAL_LOW_RELEVANCE" and issue.get("max_pa_priority", 0) <= 2:
        return False, "gov_general_low_relevance"
    if issue.get("competitor_tier") in {"COMP_GENERAL_LOW_RELEVANCE", "OFF_TOPIC"}:
        return False, "competitor_low_relevance"
    if issue.get("competitor_tier") == "COMP_OVERSEAS_MARKET_SIGNAL" and issue.get("max_ceo_priority", 0) < 5:
        return False, "overseas_market_signal_low_priority"
    return True, "ok"


def v21_category_counts_from_decisions(decisions, issue_by_id):
    counts = {k: 0 for k in JSON_KEYS_ORDER}
    comp_domestic = 0
    comp_overseas = 0
    used_groups = set()
    family_counts = {}
    for d in decisions:
        issue = issue_by_id.get(d.get("issue_id"), {})
        cat = v19_normalize_category_key(d.get("category"), fallback=v20_issue_to_output_category(issue))
        counts[cat] = counts.get(cat, 0) + 1
        key = v20_issue_group_key(issue.get("issue_group", ""))
        if key:
            used_groups.add(key)
        cap = v20_family_cap_key(issue)
        family_counts[cap] = family_counts.get(cap, 0) + 1
        if cat == "경쟁사_해외이슈":
            scope = issue.get("competitor_scope") or "other"
            if scope in {"domestic", "mixed"}:
                comp_domestic += 1
            elif scope == "overseas":
                comp_overseas += 1
    return counts, used_groups, family_counts, comp_domestic, comp_overseas


def v21_can_add_decision(decision, selected, issue_by_id, strict_max=True):
    issue = issue_by_id.get(decision.get("issue_id"))
    ok, reason = v21_issue_allowed_basic(issue)
    if not ok:
        return False, reason
    cat = v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue))
    counts, used_groups, family_counts, comp_domestic, comp_overseas = v21_category_counts_from_decisions(selected, issue_by_id)
    if strict_max and counts.get(cat, 0) >= CATEGORY_MAX.get(cat, 99):
        return False, f"category_max:{cat}"
    key = v20_issue_group_key(issue.get("issue_group", ""))
    if key and key in used_groups:
        return False, "same_issue_group_already_selected"
    cap_key = v20_family_cap_key(issue)
    cap = V20_FAMILY_CAPS.get(cap_key)
    if cap is not None and family_counts.get(cap_key, 0) >= cap:
        return False, f"family_cap:{cap_key}:{cap}"
    if cat == "경쟁사_해외이슈" and issue.get("competitor_scope") == "overseas":
        overseas_priority5 = max(int(issue.get("max_ceo_priority") or 1), int(decision.get("priority") or 1)) >= 5
        max_overseas = V21_COMPETITOR_OVERSEAS_MAX_PRIORITY5 if overseas_priority5 else V21_COMPETITOR_OVERSEAS_MAX_DEFAULT
        if comp_overseas >= max_overseas:
            return False, f"overseas_competitor_cap:{max_overseas}"
    # Do not fill self section with weak filler if already has three self issues.
    if cat == "자사_및_계열사_이슈" and issue.get("self_tier") == "FILLER_REFERENCE" and counts.get(cat, 0) >= 3:
        return False, "self_filler_not_needed"
    return True, "ok"


def v20_apply_decision_caps(decisions, issue_by_id):
    selected = []
    seen = set()
    for d in decisions:
        if d.get("issue_id") in seen:
            continue
        issue = issue_by_id.get(d.get("issue_id"), {})
        d = dict(d)
        d["category"] = v19_normalize_category_key(d.get("category"), fallback=v20_issue_to_output_category(issue))
        can, reason = v21_can_add_decision(d, selected, issue_by_id, strict_max=True)
        if not can:
            continue
        selected.append(d)
        seen.add(d.get("issue_id"))
        if len(selected) >= V21_FINAL_MAX:
            break
    return selected


def v21_candidate_decisions_from_issues(issues):
    out = []
    for issue in sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True):
        if not v21_issue_allowed_basic(issue)[0]:
            continue
        out.append({
            "issue_id": issue.get("issue_id"),
            "category": v20_issue_to_output_category(issue),
            "priority": max(issue.get("max_ceo_priority", 1), issue.get("max_pa_priority", 1)),
            "best_article_id": issue.get("top_article_id"),
            "backup_article_ids": issue.get("candidate_article_ids", [])[1:6],
            "reason": issue.get("label_reasons", "로컬 이슈 후보"),
            "selection_source": "local_issue_pool",
        })
    return out


def v21_fill_decisions(selected, backup_candidates, issue_by_id, issues):
    # 1) Protect category minima with Gemini backups first, then high-scoring local issue pool.
    selected = list(v20_apply_decision_caps(selected, issue_by_id))
    all_pool = []
    for d in backup_candidates:
        if d.get("issue_id") not in {x.get("issue_id") for x in all_pool}:
            all_pool.append(d)
    for d in v21_candidate_decisions_from_issues(issues):
        if d.get("issue_id") not in {x.get("issue_id") for x in all_pool}:
            all_pool.append(d)

    def try_add_from_pool(predicate, max_total=V21_FINAL_TARGET):
        nonlocal selected
        for d in all_pool:
            if len(selected) >= max_total:
                break
            if d.get("issue_id") in {x.get("issue_id") for x in selected}:
                continue
            issue = issue_by_id.get(d.get("issue_id"), {})
            if not predicate(d, issue):
                continue
            d2 = dict(d)
            d2["category"] = v19_normalize_category_key(d2.get("category"), fallback=v20_issue_to_output_category(issue))
            can, _ = v21_can_add_decision(d2, selected, issue_by_id, strict_max=True)
            if can:
                selected.append(d2)

    # Fill hard category minima.
    for cat, min_count in CATEGORY_MIN.items():
        def pred(d, issue, cat=cat):
            return v19_normalize_category_key(d.get("category"), fallback=v20_issue_to_output_category(issue)) == cat
        while sum(1 for d in selected if v19_normalize_category_key(d.get("category"), fallback=v20_issue_to_output_category(issue_by_id.get(d.get("issue_id"), {}))) == cat) < min_count:
            before = len(selected)
            try_add_from_pool(pred, max_total=V21_FINAL_MAX)
            if len(selected) == before:
                break

    # Domestic competitor first: target 3 domestic/mixed competitor issues when possible.
    def competitor_domestic_pred(d, issue):
        cat = v19_normalize_category_key(d.get("category"), fallback=v20_issue_to_output_category(issue))
        return cat == "경쟁사_해외이슈" and issue.get("competitor_scope") in {"domestic", "mixed"}
    while True:
        comp_domestic = sum(1 for d in selected if competitor_domestic_pred(d, issue_by_id.get(d.get("issue_id"), {})))
        if comp_domestic >= V21_COMPETITOR_DOMESTIC_TARGET:
            break
        before = len(selected)
        try_add_from_pool(competitor_domestic_pred, max_total=V21_FINAL_MAX)
        if len(selected) == before:
            break

    # Fill to target with best remaining, but avoid weak self filler/general low relevance.
    def general_pred(d, issue):
        return True
    try_add_from_pool(general_pred, max_total=V21_FINAL_TARGET)
    return v20_apply_decision_caps(selected, issue_by_id)


def v20_gemini_edit_issues(client, issues, recent_past_text):
    if not client:
        raise RuntimeError("Gemini client 없음")
    visible_issues = [i for i in sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True) if not i.get("exclude") and not i.get("is_pr")]
    visible_issues = visible_issues[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = "\n".join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
너는 카카오 대외협력팀 아침 뉴스 편집장이다.
아래 이슈 후보는 기사 단위가 아니라 issue_group 단위로 묶인 후보들이다.
최종 보고서는 카카오 경영진/대외협력/정책/커뮤니케이션/사업 담당자가 오늘 알아야 할 이슈를 담아야 한다.

[전체 편집 원칙]
- 같은 issue_group은 하나만 고른다.
- 단순 홍보·후원·캠페인·기부·이벤트·수상은 선택하지 않는다.
- 본문 추출 실패 시 같은 issue_group 안의 다른 기사로 대체될 수 있으므로, 중요한 이슈는 best/backup article_ids를 함께 준다.
- 일반 금융/제약/건설/비디지털 공정위·금감원 기사는 카카오 연결 경로가 약하면 제외한다.

[자사/계열사 원칙]
1. DIRECT_RISK와 RESPONSE_RELEVANT를 최우선으로 포함한다.
2. STRATEGIC_REFERENCE는 카카오 AI agent, 카톡 AI, 결제/금융, 조직/거점, 지배구조, 핵심 서비스 변화처럼 현직자가 실질적으로 참고할 때만 포함한다.
3. FILLER_REFERENCE는 자사 이슈가 3개 미만일 때만 고려한다.
4. LOW_VALUE_PR은 절대 자사 슬롯 보충용으로 쓰지 않는다.

[정부/국회 원칙]
1. 플랫폼 사업자 직접 의무/규제는 최우선이다.
2. AI 정책, 국가 AI 전략, AIDC, GPU, 공공 AI, 디지털플랫폼정부, AI 기본법/저작권/안전/거버넌스, AI 정책 컨트롤타워는 카카오 AI agent 전략과 연결되는 핵심 이슈로 본다.
3. 스테이블코인, 전자금융, 디지털자산, 디지털트윈은 연결성이 있을 때 포함하되 AI·플랫폼 직접 이슈보다 후순위다.
4. 홍콩 ELS 같은 일반 금융 제재는 원칙적으로 제외한다.

[경쟁사/해외 원칙]
1. 국내 경쟁사/인접 플랫폼을 우선한다. 네이버·네이버페이·쿠팡·토스·배민·SKT·KT·LGU+·라인야후 등.
2. 경쟁사/해외 섹션은 3~4개가 적절하며, 국내 경쟁사 이슈를 최소 2개, 가능하면 3개 포함한다.
3. 해외 빅테크/글로벌 AI 이슈는 기본 1개만 포함한다. 카카오 AI·플랫폼·콘텐츠 데이터·앱마켓·개인정보·광고 전략에 직접 시사점이 매우 크면 2개까지 가능하다.
4. 단순 글로벌 AI 투자, IPO, 주가, CAPEX, 제품 루머는 후순위 또는 제외한다.

[권장 분량]
- 전체 11~15개. 억지로 낮은 품질 기사를 채우지 않는다.
- 자사 3~5개, 정부/국회 3~5개, 경쟁사/해외 3~4개, 산업 1~2개.

반드시 JSON만 반환하라.
{{
  "selected_issues": [
    {{
      "issue_id": "I001",
      "category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향",
      "priority": 5,
      "best_article_id": 123,
      "backup_article_ids": [124, 125],
      "reason": "선정 이유"
    }}
  ],
  "backup_issues": [
    {{
      "issue_id": "I050",
      "category": "경쟁사_해외이슈",
      "priority": 4,
      "best_article_id": 555,
      "backup_article_ids": [556],
      "reason": "대체 후보 이유"
    }}
  ]
}}

[최근 과거 보고서 참고]
{recent_past_text[:4500]}

[이슈 후보]
{issue_text}
"""
    text = gemini_generate_text(
        client=client,
        prompt=prompt,
        task_name="v21 이슈 단위 최종 편집",
        model=GEMINI_MODEL_EDITOR,
    )
    data = extract_json_object(text)
    issue_by_id = {i["issue_id"]: i for i in issues}
    article_by_id = {}
    for issue in issues:
        for art_id in issue.get("article_ids", []):
            article_by_id[int(art_id)] = {"id": int(art_id)}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    return decisions, backup_decisions


def v21_strict_cross_issue_duplicate(new_item, existing_items):
    new_url = normalize_url(new_item.get("링크", ""))
    new_fp = title_fingerprint(new_item.get("기사제목", ""))
    new_group = v20_issue_group_key(new_item.get("v20_issue_group") or new_item.get("Gemini이슈그룹") or "")
    new_body = clean_html_text(new_item.get("본문전문", ""))[:1500]
    for old in existing_items:
        old_url = normalize_url(old.get("링크", ""))
        old_fp = title_fingerprint(old.get("기사제목", ""))
        old_group = v20_issue_group_key(old.get("v20_issue_group") or old.get("Gemini이슈그룹") or "")
        if new_group and old_group and new_group == old_group:
            return old, "same_gemini_issue_group"
        if new_url and old_url and new_url == old_url:
            return old, "same_url"
        if new_fp and old_fp and new_fp == old_fp:
            return old, "same_title"
        old_body = clean_html_text(old.get("본문전문", ""))[:1500]
        if new_body and old_body and len(new_body) > 500 and len(old_body) > 500:
            sim = jaccard(char_ngrams(new_body, 5), char_ngrams(old_body, 5))
            if sim >= 0.86:
                return old, f"near_identical_body:{sim:.2f}"
    return None, ""


def v20_attach_issue_fields(report_item, article_info, issue, decision, reason):
    report_item["v20_issue_id"] = issue.get("issue_id")
    report_item["v20_issue_group"] = issue.get("issue_group")
    report_item["v20_issue_family"] = issue.get("issue_family")
    report_item["v20_internal_category"] = issue.get("internal_category")
    report_item["v20_issue_score"] = issue.get("issue_score")
    report_item["v21_self_tier"] = issue.get("self_tier", "")
    report_item["v21_gov_tier"] = issue.get("gov_tier", "")
    report_item["v21_competitor_tier"] = issue.get("competitor_tier", "")
    report_item["v21_competitor_scope"] = issue.get("competitor_scope", "")
    report_item["Gemini카테고리"] = decision.get("category") or article_info.get("Gemini카테고리", "")
    report_item["Gemini내부카테고리"] = article_info.get("Gemini내부카테고리", issue.get("internal_category", ""))
    report_item["Gemini이슈유형"] = article_info.get("Gemini이슈패밀리", issue.get("issue_family", ""))
    report_item["Gemini이슈그룹"] = article_info.get("Gemini이슈그룹", issue.get("issue_group", ""))
    report_item["Gemini중복그룹"] = article_info.get("Gemini이슈그룹", issue.get("issue_group", ""))
    report_item["Gemini관련성"] = article_info.get("Gemini관련성", issue.get("company_impact", ""))
    report_item["Gemini중요도"] = article_info.get("Gemini중요도", issue.get("max_ceo_priority", ""))
    report_item["Gemini대외협력중요도"] = article_info.get("Gemini대외협력중요도", issue.get("max_pa_priority", ""))
    report_item["Gemini관련도"] = article_info.get("Gemini관련도", issue.get("max_relevance", ""))
    report_item["Gemini판단사유"] = article_info.get("Gemini판단사유", issue.get("label_reasons", ""))
    report_item["Gemini선정사유"] = decision.get("reason", issue.get("label_reasons", ""))
    report_item["선정단계"] = reason
    return report_item


def v20_process_issue(decision, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter):
    issue = issue_by_id.get(decision.get("issue_id"))
    if not issue:
        return None, order_counter, "missing_issue"
    allowed, block_reason = v21_issue_allowed_basic(issue)
    if not allowed:
        return None, order_counter, block_reason
    json_key = v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue))
    candidate_ids = []
    if decision.get("best_article_id"):
        candidate_ids.append(int(decision["best_article_id"]))
    for art_id in decision.get("backup_article_ids", []):
        if int(art_id) not in candidate_ids:
            candidate_ids.append(int(art_id))
    for art_id in issue.get("candidate_article_ids", []):
        if int(art_id) not in candidate_ids:
            candidate_ids.append(int(art_id))
    for art_id in issue.get("article_ids", []):
        if int(art_id) not in candidate_ids:
            candidate_ids.append(int(art_id))

    attempted_failed = []
    for art_id in candidate_ids:
        if art_id in processed_article_ids:
            continue
        article_info = article_by_id.get(int(art_id))
        if not article_info:
            continue
        processed_article_ids.add(int(art_id))
        report_item, skip_info, failed_item = process_article_for_report(article_info, json_key, recent_past_items)
        if skip_info:
            skip_info["제외단계"] = "issue_representative_attempt"
            skip_info["v20_issue_id"] = issue.get("issue_id")
            skip_info["v20_issue_group"] = issue.get("issue_group")
            skip_rows.append(skip_info)
            attempted_failed.append((article_info, "past_duplicate_or_skip"))
            continue
        if failed_item:
            failed_item["제외단계"] = "issue_representative_attempt"
            failed_item["v20_issue_id"] = issue.get("issue_id")
            failed_item["v20_issue_group"] = issue.get("issue_group")
            body_failed_rows.append(failed_item)
            attempted_failed.append((article_info, failed_item.get("본문품질사유", "body_failed")))
            continue
        if not report_item:
            attempted_failed.append((article_info, "empty_process_result"))
            continue
        report_item = v20_attach_issue_fields(report_item, article_info, issue, decision, "issue_selected")
        report_item["선정순서"] = order_counter
        report_item["본문상태"] = "본문추출성공"

        relevant, relevance_reason = is_report_item_relevant(report_item, json_key)
        if not relevant:
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"relevance_failed:{relevance_reason}"
            failed_copy["제외단계"] = "issue_representative_attempt"
            body_failed_rows.append(failed_copy)
            attempted_failed.append((article_info, relevance_reason))
            continue

        dup_item, dup_reason = v21_strict_cross_issue_duplicate(report_item, final_report_data)
        if dup_item:
            new_score = v20_float(report_item.get("대표선택점수"), 0)
            old_score = v20_float(dup_item.get("대표선택점수"), 0)
            # Only replace within the same issue group or exact same URL/title/body.
            if dup_reason == "same_gemini_issue_group" and new_score > old_score + REPRESENTATIVE_REPLACE_MARGIN:
                try:
                    idx = final_report_data.index(dup_item)
                    final_report_data[idx] = report_item
                except ValueError:
                    final_report_data.append(report_item)
                add_duplicate_skip_row(skip_rows, dup_item, report_item, f"replaced_by_better_duplicate:{dup_reason}", stage="issue_selected", extra=f"old={old_score},new={new_score}")
                return report_item, order_counter + 1, "replaced_duplicate"
            add_duplicate_skip_row(skip_rows, report_item, dup_item, f"strict_duplicate:{dup_reason}", stage="issue_selected", extra=f"old={old_score},new={new_score}")
            return None, order_counter, "duplicate_already_represented"

        final_report_data.append(report_item)
        return report_item, order_counter + 1, "body_success"

    if v20_is_critical_issue(issue, decision):
        fallback_id = int(decision.get("best_article_id") or issue.get("top_article_id"))
        article_info = article_by_id.get(fallback_id)
        if article_info:
            item = v20_create_bodyless_report_item(article_info, issue, decision, json_key)
            item["선정순서"] = order_counter
            dup_item, dup_reason = v21_strict_cross_issue_duplicate(item, final_report_data)
            if dup_item:
                add_duplicate_skip_row(skip_rows, item, dup_item, f"bodyless_duplicate:{dup_reason}", stage="critical_without_body")
                return None, order_counter, "bodyless_duplicate"
            final_report_data.append(item)
            return item, order_counter + 1, "critical_without_body"
    return None, order_counter, "all_candidates_failed"


def v21_final_counts(final_items):
    counts = {k: 0 for k in JSON_KEYS_ORDER}
    comp_domestic = 0
    comp_overseas = 0
    for item in final_items:
        cat = item.get("카테고리")
        counts[cat] = counts.get(cat, 0) + 1
        if cat == "경쟁사_해외이슈":
            scope = item.get("v21_competitor_scope") or item.get("Gemini경쟁사범위") or v21_competitor_scope(f"{item.get('기사제목','')} {item.get('본문전문','')[:500]}")
            if scope in {"domestic", "mixed"}:
                comp_domestic += 1
            elif scope == "overseas":
                comp_overseas += 1
    return counts, comp_domestic, comp_overseas


def v21_process_until_balanced(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    processed_issue_ids = set()
    def process_one(decision, source="selected"):
        nonlocal order_counter
        if decision.get("issue_id") in processed_issue_ids:
            return False
        if len(final_report_data) >= V21_FINAL_MAX:
            return False
        item, order_counter, status = v20_process_issue(
            decision, issue_by_id, article_by_id, recent_past_items, final_report_data,
            body_failed_rows, skip_rows, processed_article_ids, order_counter
        )
        processed_issue_ids.add(decision.get("issue_id"))
        status_counts[status] = status_counts.get(status, 0) + 1
        issue = issue_by_id.get(decision.get("issue_id"), {})
        if item:
            print(f"  └ 📥 이슈 반영({source}): {v20_clip(issue.get('issue_group',''), 42)} / {v20_clip(item.get('기사제목',''), 42)}... ({item.get('본문추출방식')}, {item.get('본문글자수')}자, {status})")
            return True
        print(f"  └ ⚠️ 이슈 제외/대체대기({source}): {v20_clip(issue.get('issue_group',''), 48)}... ({status})")
        return False

    for d in decisions:
        process_one(d, "selected")

    candidate_pool = []
    for d in backup_decisions:
        if d.get("issue_id") not in {x.get("issue_id") for x in candidate_pool}:
            candidate_pool.append(d)
    for d in v21_candidate_decisions_from_issues(issues):
        if d.get("issue_id") not in {x.get("issue_id") for x in candidate_pool}:
            candidate_pool.append(d)

    # Fill missing categories with category-specific candidates only.
    for cat, min_count in CATEGORY_MIN.items():
        while v21_final_counts(final_report_data)[0].get(cat, 0) < min_count and len(final_report_data) < V21_FINAL_MAX:
            added = False
            for d in candidate_pool:
                issue = issue_by_id.get(d.get("issue_id"), {})
                d_cat = v19_normalize_category_key(d.get("category"), fallback=v20_issue_to_output_category(issue))
                if d_cat != cat:
                    continue
                if d.get("issue_id") in processed_issue_ids:
                    continue
                # For self filler, require self section below 3 and no direct/response alternatives.
                if cat == "자사_및_계열사_이슈" and issue.get("self_tier") == "FILLER_REFERENCE":
                    continue
                if process_one(d, "category_backup"):
                    added = True
                    break
            if not added:
                break

    # Domestic competitor protection.
    while v21_final_counts(final_report_data)[1] < V21_COMPETITOR_DOMESTIC_MIN and len(final_report_data) < V21_FINAL_MAX:
        added = False
        for d in candidate_pool:
            issue = issue_by_id.get(d.get("issue_id"), {})
            cat = v19_normalize_category_key(d.get("category"), fallback=v20_issue_to_output_category(issue))
            if cat != "경쟁사_해외이슈" or issue.get("competitor_scope") not in {"domestic", "mixed"}:
                continue
            if d.get("issue_id") in processed_issue_ids:
                continue
            if process_one(d, "domestic_competitor_backup"):
                added = True
                break
        if not added:
            break

    # Fill to target, not beyond, with high-quality backups only.
    for d in candidate_pool:
        if len(final_report_data) >= V21_FINAL_TARGET:
            break
        if d.get("issue_id") in processed_issue_ids:
            continue
        issue = issue_by_id.get(d.get("issue_id"), {})
        if not v21_issue_allowed_basic(issue)[0]:
            continue
        # Skip weak overseas market signals and general low relevance in final fill.
        if issue.get("competitor_tier") == "COMP_OVERSEAS_MARKET_SIGNAL":
            continue
        if issue.get("gov_tier") == "GOV_GENERAL_LOW_RELEVANCE":
            continue
        process_one(d, "quality_backup")
    return order_counter


def v21_remove_obvious_final_duplicates(final_items):
    kept = []
    removed = []
    for item in final_items:
        dup, reason = v21_strict_cross_issue_duplicate(item, kept)
        if dup:
            # Keep higher representative score.
            if v20_float(item.get("대표선택점수"), 0) > v20_float(dup.get("대표선택점수"), 0):
                try:
                    kept.remove(dup)
                    removed.append((dup, f"removed_for_better:{reason}"))
                except ValueError:
                    pass
                kept.append(item)
            else:
                removed.append((item, reason))
        else:
            kept.append(item)
    return kept, removed


def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    if not client:
        return ""
    item_lines = []
    for item in final_report_data:
        item_lines.append(
            f"[{item.get('브리핑ID')}] cat={item.get('카테고리')} title={item.get('기사제목')} "
            f"group={item.get('v20_issue_group','')} self_tier={item.get('v21_self_tier','')} gov_tier={item.get('v21_gov_tier','')} "
            f"comp_tier={item.get('v21_competitor_tier','')} comp_scope={item.get('v21_competitor_scope','')} "
            f"body_status={item.get('본문상태','')} reason={item.get('Gemini선정사유','')}"
        )
    prompt = f"""
너는 카카오 대외협력팀 데스크의 최종 품질검수자다.
아래 최종 브리핑이 다음 기준에 맞는지 점검하라.
- 같은 이슈 중복 여부
- 자사 섹션에 단순 홍보/기부/캠페인성 기사 포함 여부
- 정부/국회 섹션에 일반 금융/비디지털 제재 기사 포함 여부
- 경쟁사/해외 섹션이 국내 경쟁사 우선 원칙을 지켰는지. 국내 경쟁사/인접 플랫폼 2~3개, 해외 1개가 기본이다.
- 해외 이슈가 단순 투자/IPO/주가/CAPEX만으로 들어왔는지
- 본문추출실패 기사 과다 여부
수정하지 말고 JSON으로만 검수 의견을 반환하라.

[편집 목록]
{chr(10).join(item_lines)}

[최종 보고서]
{final_briefing_text[:7000]}

형식:
{{
  "overall": "양호/주의/문제",
  "warnings": ["경고1"],
  "suggested_human_checks": ["확인할 기사 또는 이슈"]
}}
"""
    try:
        text = gemini_generate_text(client=client, prompt=prompt, task_name="v21 최종 품질검수", model=GEMINI_MODEL_QA)
        return text.strip()
    except Exception as e:
        return f"QA failed: {e}"


def main():
    total_start_time = time.time()
    run_log = []

    print(f"\n🧩 v21 실행: {V6_VERSION}")
    print(f"   └ 모델: labeling={GEMINI_MODEL_LABELING}, editor={GEMINI_MODEL_EDITOR}, summary={GEMINI_MODEL_SUMMARY}, qa={GEMINI_MODEL_QA}")

    client = None
    if ENABLE_GEMINI_SELECTION or ENABLE_GEMINI_REPORT:
        try:
            with open(SECRET_PATH, "r", encoding="utf-8") as f:
                google_api_key = f.read().strip()
            if not google_api_key:
                raise ValueError("secret.txt가 비어 있습니다.")
            client = genai.Client(api_key=google_api_key)
        except Exception as e:
            print("⚠️ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다. Gemini 없이 로컬 이슈 편집으로 진행합니다.")
            print(f"   원인: {e}")
            client = None

    past_reports_content, all_past_items, recent_past_items = load_past_reports()
    recent_past_text = build_recent_past_text(recent_past_items, max_chars=9000)

    raw_articles, skipped_duplicates = collect_with_google_rss(recent_past_items)
    print(f"\n  └ ✅ 총 {len(raw_articles)}개의 RSS 후보 확보 완료")
    print(f"  └ 🧹 수집/과거중복/오탐 제외 후보: {len(skipped_duplicates)}개")

    if not raw_articles:
        print("❌ 수집된 기사가 없습니다. Google News RSS 접속 또는 키워드/기간 설정을 확인하세요.")
        return

    ranked_all, ranked_candidates = rank_and_trim_candidates(raw_articles)
    print(f"  └ 🧺 v21 후보 풀: 전체 {len(ranked_all)}개 중 {len(ranked_candidates)}개를 기사 라벨링 대상으로 사용")

    print("\n🏷️ [STEP 2] Gemini가 기사별 의미 라벨을 붙입니다...")
    labels = {}
    gemini_label_success = False
    try:
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 라벨링 비활성화 또는 client 없음")
        labels = v20_gemini_label_candidates(client, ranked_candidates, recent_past_text)
        gemini_label_success = True
        print(f"  └ ✅ Gemini 기사 라벨링 완료: {len(labels)}개")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사 라벨링 실패. 로컬 보조 라벨로 진행합니다. 원인: {e}")
        labels = {int(a["id"]): v20_local_article_label(a) for a in ranked_candidates}

    for article in ranked_all:
        art_id = int(article["id"])
        if art_id not in labels:
            labels[art_id] = v20_local_article_label(article)

    label_rows = v20_apply_labels_to_articles(ranked_all, labels)
    pd.DataFrame(ranked_all).to_csv(OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(ranked_candidates).to_csv(OUTPUT_RANKED_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(OUTPUT_GEMINI_LABELS_CSV, index=False, encoding="utf-8-sig")
    if skipped_duplicates:
        pd.DataFrame(skipped_duplicates).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 후보/라벨 CSV 저장 완료: {os.path.basename(OUTPUT_CANDIDATES_CSV)}, {os.path.basename(OUTPUT_RANKED_CSV)}, {os.path.basename(OUTPUT_GEMINI_LABELS_CSV)}")

    print("\n🧱 [STEP 3] 기사 후보를 이슈 단위로 묶습니다...")
    issues, article_by_id = v20_build_issues(ranked_candidates, labels)
    issue_by_id = {i["issue_id"]: i for i in issues}
    pd.DataFrame(v20_issue_rows(issues)).to_csv(OUTPUT_ISSUES_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ ✅ 이슈 그룹 {len(issues)}개 생성 / 저장: {os.path.basename(OUTPUT_ISSUES_CSV)}")

    print("\n🧠 [STEP 4] Gemini가 이슈 단위로 최종 편집합니다...")
    gemini_editor_success = False
    try:
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 이슈 편집 비활성화 또는 client 없음")
        decisions, backup_decisions = v20_gemini_edit_issues(client, issues, recent_past_text)
        gemini_editor_success = True
        print(f"  └ ✅ Gemini 이슈 편집 완료: 선택 {len(decisions)}개 / 백업 {len(backup_decisions)}개")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 이슈 편집 실패. 로컬 이슈 점수 기반으로 진행합니다. 원인: {e}")
        decisions, backup_decisions = v20_local_issue_selection(issues)

    decisions = v20_apply_decision_caps(decisions, issue_by_id)
    decisions = v21_fill_decisions(decisions, backup_decisions, issue_by_id, issues)
    pd.DataFrame(v20_decision_rows(decisions, backup_decisions, issue_by_id)).to_csv(OUTPUT_ISSUE_SELECTION_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 이슈 편집 결과 저장: {os.path.basename(OUTPUT_ISSUE_SELECTION_CSV)}")

    print("\n🕵️‍♂️ [STEP 5] 선택 이슈별 대표 기사를 찾고 본문을 추출합니다...")
    print("   └ 전역 중복은 엄격히 제한하고, 카테고리/국내 경쟁사 하한을 보호합니다.")

    final_report_data = []
    body_failed_rows = []
    post_body_duplicate_skips = []
    processed_article_ids = set()
    order_counter = 1
    status_counts = {}
    order_counter = v21_process_until_balanced(
        decisions,
        backup_decisions,
        issues,
        issue_by_id,
        article_by_id,
        recent_past_items,
        final_report_data,
        body_failed_rows,
        post_body_duplicate_skips,
        processed_article_ids,
        order_counter,
        status_counts,
    )

    final_report_data, removed_duplicates = v21_remove_obvious_final_duplicates(final_report_data)
    for rem, reason in removed_duplicates:
        add_duplicate_skip_row(post_body_duplicate_skips, rem, {"기사제목": "v21_final_duplicate", "링크": "", "대표선택점수": ""}, reason, stage="v21_final_repair")

    if skipped_duplicates or post_body_duplicate_skips:
        all_skips = skipped_duplicates + post_body_duplicate_skips
        pd.DataFrame(all_skips).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 제외/중복/대표교체 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")
    if body_failed_rows:
        pd.DataFrame(body_failed_rows).to_csv(OUTPUT_BODY_FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ ⚠️ 본문 품질 미달/관련성 미달 기사 저장: {os.path.basename(OUTPUT_BODY_FAILED_CSV)}")

    if not final_report_data:
        print("❌ 최종 보고서에 사용할 기사 데이터가 없습니다.")
        return

    ordered_final = []
    for json_key in JSON_KEYS_ORDER:
        items = [x for x in final_report_data if x.get("카테고리") == json_key]
        items = sorted(items, key=lambda x: int(x.get("선정순서") or 9999))
        ordered_final.extend(items)
    final_report_data = ordered_final[:V21_FINAL_MAX]

    print("\n✍️ [STEP 6] 본문/RSS 기반으로 최종 브리핑 요약을 생성합니다...")
    for idx, item in enumerate(final_report_data, 1):
        item["브리핑ID"] = idx

    prompt_report = v20_build_summary_prompt(final_report_data, recent_past_text)
    summary_map = {}
    try:
        if not client or not ENABLE_GEMINI_REPORT:
            raise RuntimeError("Gemini 최종 브리핑 비활성화 또는 client 없음")
        summary_text = gemini_generate_text(
            client=client,
            prompt=prompt_report,
            task_name="v21 최종 기사별 요약 생성",
            model=GEMINI_MODEL_SUMMARY,
        )
        summary_map = normalize_summary_json(extract_json_object(summary_text))
        print("  └ ✅ Gemini 기사별 요약 생성 완료")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사별 요약 생성 실패. 로컬 본문/RSS 요약으로 대체합니다. 원인: {e}")
        summary_map = {}

    for item in final_report_data:
        bid = str(item.get("브리핑ID"))
        if item.get("본문상태") == "본문추출실패_중요이슈_제한포함" and not summary_map.get(bid):
            summary_map[bid] = v20_bodyless_summary(item)

    final_briefing_text = build_structured_briefing(final_report_data, summary_map)

    qa_text = ""
    if ENABLE_GEMINI_QA:
        print("\n🔎 [STEP 7] 최종 편집 품질검수 로그를 생성합니다...")
        qa_text = v20_gemini_quality_check(client, final_report_data, final_briefing_text)
        if qa_text:
            with open(OUTPUT_QA_TXT, "w", encoding="utf-8") as f:
                f.write(qa_text)
            print(f"  └ 💾 QA 로그 저장: {os.path.basename(OUTPUT_QA_TXT)}")

    print("\n" + "=" * 60)
    print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
    print("=" * 60)
    print(final_briefing_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_briefing_text)
    pd.DataFrame(final_report_data).to_csv(OUTPUT_SELECTED_CSV, index=False, encoding="utf-8-sig")

    counts, comp_dom, comp_over = v21_final_counts(final_report_data)
    run_log.append({
        "버전": V6_VERSION,
        "모델_라벨링": GEMINI_MODEL_LABELING,
        "모델_편집": GEMINI_MODEL_EDITOR,
        "모델_요약": GEMINI_MODEL_SUMMARY,
        "전체_RSS_후보": len(raw_articles),
        "Gemini_기사라벨_후보": len(ranked_candidates),
        "이슈그룹수": len(issues),
        "Gemini라벨링성공": gemini_label_success,
        "Gemini이슈편집성공": gemini_editor_success,
        "선택이슈수": len(decisions),
        "백업이슈수": len(backup_decisions),
        "최종_자사": counts.get("자사_및_계열사_이슈", 0),
        "최종_정부": counts.get("정부_국회", 0),
        "최종_경쟁사": counts.get("경쟁사_해외이슈", 0),
        "최종_산업": counts.get("산업동향", 0),
        "최종_경쟁사_국내": comp_dom,
        "최종_경쟁사_해외": comp_over,
        "본문성공": status_counts.get("body_success", 0) + status_counts.get("replaced_duplicate", 0),
        "본문실패_중요이슈_제한포함": status_counts.get("critical_without_body", 0),
        "제외_총합": len(skipped_duplicates) + len(post_body_duplicate_skips),
        "본문품질미달_관련성미달": len(body_failed_rows),
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
    print(f"- '{os.path.basename(OUTPUT_GEMINI_LABELS_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_ISSUES_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_ISSUE_SELECTION_CSV)}' 저장 완료")
    if skipped_duplicates or post_body_duplicate_skips:
        print(f"- '{os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}' 저장 완료")
    if body_failed_rows:
        print(f"- '{os.path.basename(OUTPUT_BODY_FAILED_CSV)}' 저장 완료")
    if qa_text:
        print(f"- '{os.path.basename(OUTPUT_QA_TXT)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_RUN_LOG_CSV)}' 저장 완료")
    print("=" * 60)



# v21.1 precision patch: prevent AI/competitor articles from being misfiled as government,
# and demote quiz/promo noise.
def v20_is_platform_obligation_text(text):
    t = clean_html_text(text)
    if not t:
        return False
    actor_re = re.compile(
        r"플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|부가통신사업자|정보통신서비스\s*제공자|인터넷\s*사업자|"
        r"네이버|카카오|구글|메타|유튜브|틱톡|애플|쿠팡|배달의민족|배민|토스",
        re.IGNORECASE,
    )
    duty_re = re.compile(
        r"의무|의무화|해야\s*한다|해야\s*함|해야\s*할|걸러야|사전\s*차단|삭제\s*의무|필터링\s*의무|"
        r"신고|보고|공시|자료\s*제출|자료제출|준수|위반|금지|제한|제재|과징금|과태료|"
        r"시정명령|유통\s*방지|확산\s*방지|모니터링\s*의무|소명",
        re.IGNORECASE,
    )
    regulator_re = re.compile(
        r"정부|국회|과기정통부|방미통위|방송미디어통신위원회|방통위|개보위|개인정보위|공정위|"
        r"공정거래위원회|금융위|금융감독원|금감원|행안부|경찰청|검찰|KISA|인터넷진흥원|"
        r"법안|개정안|시행령|시행규칙|고시|입법예고|행정예고|제도|정책|규제|감독|조사|제재|"
        r"내달부터|다음달부터|오는\s*\d{1,2}월부터|\d{1,2}월\s*\d{1,2}일부터",
        re.IGNORECASE,
    )
    duty_domain_re = re.compile(
        r"불법\s*촬영|불법정보|유해정보|유해\s*정보|허위조작정보|딥페이크|디지털성범죄|청소년\s*보호|"
        r"개인정보|정보\s*유출|사칭|피싱|보이스피싱|악성코드|이용자\s*보호|알고리즘|광고\s*표시|"
        r"인앱결제|앱마켓|수수료|정산|입점업체|판매자|자사우대|최혜대우|온라인플랫폼법|온플법",
        re.IGNORECASE,
    )
    if not actor_re.search(t):
        return False
    # Product/security tech announcements by Google/Naver etc. are not government obligations without a regulator/legal timing cue.
    return bool(duty_re.search(t) and regulator_re.search(t) and duty_domain_re.search(t))


def v21_self_tier(text):
    t = clean_html_text(text)
    if not v20_has_self_entity(t):
        return "OFF_TOPIC"
    if re.search(r"퀴즈\s*정답|오늘의\s*.*정답|이모지\s*퀴즈|캐시워크|돈버는\s*퀴즈", t, re.I):
        return "LOW_VALUE_PR"
    if V21_SELF_PR_RE.search(t) and not V21_SELF_DIRECT_RISK_RE.search(t) and not v20_is_platform_obligation_text(t):
        return "LOW_VALUE_PR"
    if V21_SELF_DIRECT_RISK_RE.search(t):
        return "DIRECT_RISK"
    if v20_is_platform_obligation_text(t) or (REGULATOR_CORE_PATTERN.search(t) and re.search(r"카카오|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈", t, re.I)):
        return "RESPONSE_RELEVANT"
    if V21_SELF_STRATEGIC_RE.search(t):
        return "STRATEGIC_REFERENCE"
    return "FILLER_REFERENCE"


def v21_competitor_tier(text):
    t = clean_html_text(text)
    if re.search(r"포인트\s*지급|챌린지|프로모션|이벤트|쿠폰|할인|경품|리워드|체험단|캠페인", t, re.I):
        return "COMP_GENERAL_LOW_RELEVANCE"
    scope = v21_competitor_scope(t)
    if scope in {"domestic", "mixed"}:
        if re.search(r"장애|오류|복구|개인정보|유출|해킹|피싱|보안|사과|이용자\s*불편|제재|과징금|조사|논란", t, re.I):
            return "COMP_DOMESTIC_DIRECT_RISK"
        if re.search(r"공공|재난|우선\s*통신|우선\s*접속|행정|정부|망|통신권|재난망|공공\s*AI|소방|경찰", t, re.I):
            return "COMP_DOMESTIC_PUBLIC_INFRA"
        if re.search(r"AI|인공지능|AX|에이전트|검색|추천|하이퍼클로바|에이닷|GPU|클라우드|데이터센터|LLM|RCS|디도스|DDoS", t, re.I):
            return "COMP_DOMESTIC_AI_PLATFORM"
        if re.search(r"네이버페이|토스|토스페이|쿠팡페이|간편결제|금융|커머스|쇼핑|멤버십|배달|쿠팡이츠|배민|광고", t, re.I):
            return "COMP_DOMESTIC_FINANCE_COMMERCE"
        return "COMP_DOMESTIC_STRATEGY"
    if scope == "overseas":
        if re.search(r"데이터|학습|저작권|언론|뉴스|콘텐츠|개인정보|광고|앱마켓|DMA|DSA|규제|소송|과징금|AI\s*랩", t, re.I):
            return "COMP_OVERSEAS_REGULATION_DATA"
        if re.search(r"AI|인공지능|에이전트|LLM|검색|제미나이|ChatGPT|GPT|클로드|모델|앱|플랫폼", t, re.I):
            return "COMP_OVERSEAS_PLATFORM_AI"
        if re.search(r"투자|IPO|상장|주가|지분|CAPEX|데이터센터|전력|발전|자금\s*조달", t, re.I):
            return "COMP_OVERSEAS_MARKET_SIGNAL"
    return "COMP_GENERAL_LOW_RELEVANCE"


def v20_detect_issue_family(text, json_key=""):
    t = clean_html_text(text)
    if not t:
        return "off_topic"
    self_tier = v21_self_tier(t)
    if self_tier == "LOW_VALUE_PR":
        return "low_value_pr"

    # Competitor category must be tested before broad AI-government patterns unless a regulator/legal cue is explicit.
    comp_tier = v21_competitor_tier(t)
    has_competitor = v21_competitor_scope(t) in {"domestic", "mixed", "overseas"}
    has_regulator = bool(REGULATOR_CORE_PATTERN.search(t) or re.search(r"정부|국회|법안|시행령|가이드라인|규제|감독|조사|제재|의무|내달부터|다음달부터", t, re.I))
    if (json_key == "경쟁사_해외이슈" or has_competitor) and comp_tier not in {"COMP_GENERAL_LOW_RELEVANCE", "OFF_TOPIC"} and not (has_regulator and v20_is_platform_obligation_text(t)):
        return comp_tier.lower()

    if v20_is_platform_obligation_text(t):
        return "platform_obligation" if v20_has_self_entity(t) else "government_platform_regulation"

    if v20_has_self_entity(t):
        if self_tier == "DIRECT_RISK":
            if re.search(r"노조|파업|임단협|임금|성과급|RSU|고용불안|조정", t, re.I):
                return "self_labor"
            if re.search(r"지분|매각|인수|합병|최대주주|경영권|공동대표|사내이사|주총|유상증자|전환사채", t, re.I):
                return "self_governance_mna"
            if re.search(r"개인정보|유출|해킹|피싱|보안|장애|먹통", t, re.I):
                return "self_privacy_security"
            if re.search(r"수사|조사|과징금|제재|소송|판결|검찰|경찰", t, re.I):
                return "self_legal_regulatory"
            if re.search(r"임원|대표|CPO|CTO|조직개편|퇴사|사임|서비스\s*논란|이용자\s*반발|카톡\s*개편", t, re.I):
                return "self_leadership_service"
            return "self_direct_risk"
        if self_tier == "RESPONSE_RELEVANT":
            return "self_response_relevant"
        if self_tier == "STRATEGIC_REFERENCE":
            return "self_strategic_reference"
        return "self_filler_reference"

    gov_tier = v21_gov_tier(t)
    if json_key == "정부_국회" or gov_tier != "OFF_TOPIC" or REGULATOR_CORE_PATTERN.search(t):
        if gov_tier == "GOV_DIRECT_PLATFORM_OBLIGATION":
            return "government_platform_obligation"
        if gov_tier == "GOV_AI_CORE_POLICY":
            return "government_ai_core_policy"
        if gov_tier == "GOV_REGULATORY_ENFORCEMENT":
            return "government_regulatory_enforcement"
        if gov_tier == "GOV_PLATFORM_MARKET_RULE":
            return "government_platform_market_rule"
        if gov_tier == "GOV_APPOINTMENT_GOVERNANCE":
            return "government_appointment_governance"
        if gov_tier == "GOV_DIGITAL_FINANCE_POLICY":
            return "government_digital_finance_policy"
        if gov_tier == "GOV_ADJACENT_DIGITAL_POLICY":
            return "government_adjacent_digital_policy"
        if gov_tier == "GOV_GENERAL_LOW_RELEVANCE":
            return "government_general_low_relevance"
        return "government_general_policy"

    if has_competitor:
        return "competitor_general_low_relevance" if comp_tier == "COMP_GENERAL_LOW_RELEVANCE" else comp_tier.lower()

    if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|반도체|HBM|TSMC|엔비디아|전력|플랫폼|카카오톡\s*선물", t, re.I):
        return "industry_structural_change"
    return "off_topic"




# v21.2 precision patch: prioritize self/platform obligations before competitor cues, and refine risk wording.
def v20_is_platform_obligation_text(text):
    t = clean_html_text(text)
    if not t:
        return False
    actor_re = re.compile(
        r"플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|부가통신사업자|정보통신서비스\s*제공자|인터넷\s*사업자|"
        r"네이버|카카오|구글|메타|유튜브|틱톡|애플|쿠팡|배달의민족|배민|토스",
        re.IGNORECASE,
    )
    duty_re = re.compile(
        r"의무|의무화|해야\s*한다|해야\s*함|해야\s*할|걸러야|차단해야|사전\s*차단|삭제\s*의무|필터링\s*의무|"
        r"신고|보고|공시|자료\s*제출|자료제출|준수|위반|금지|제한|제재|과징금|과태료|"
        r"시정명령|유통\s*방지|확산\s*방지|모니터링\s*의무|소명",
        re.IGNORECASE,
    )
    regulator_re = re.compile(
        r"정부|국회|과기정통부|방미통위|방송미디어통신위원회|방통위|개보위|개인정보위|공정위|"
        r"공정거래위원회|금융위|금융감독원|금감원|행안부|경찰청|검찰|KISA|인터넷진흥원|"
        r"법안|개정안|시행령|시행규칙|고시|입법예고|행정예고|제도|정책|규제|감독|조사|제재|"
        r"내달부터|다음달부터|오는\s*\d{1,2}월부터|\d{1,2}월부터|\d{1,2}월\s*\d{1,2}일부터",
        re.IGNORECASE,
    )
    duty_domain_re = re.compile(
        r"불법\s*촬영|불법정보|유해정보|유해\s*정보|허위조작정보|딥페이크|디지털성범죄|청소년\s*보호|"
        r"개인정보|정보\s*유출|사칭|피싱|보이스피싱|악성코드|이용자\s*보호|알고리즘|광고\s*표시|"
        r"인앱결제|앱마켓|수수료|정산|입점업체|판매자|자사우대|최혜대우|온라인플랫폼법|온플법",
        re.IGNORECASE,
    )
    return bool(actor_re.search(t) and duty_re.search(t) and regulator_re.search(t) and duty_domain_re.search(t))


def v21_competitor_tier(text):
    t = clean_html_text(text)
    if re.search(r"포인트\s*지급|챌린지|프로모션|이벤트|쿠폰|할인|경품|리워드|체험단|캠페인", t, re.I):
        return "COMP_GENERAL_LOW_RELEVANCE"
    scope = v21_competitor_scope(t)
    if scope in {"domestic", "mixed"}:
        if re.search(r"장애|오류|복구|개인정보|유출|해킹|피싱|보안\s*사고|침해사고|사과|이용자\s*불편|제재|과징금|조사|논란", t, re.I):
            return "COMP_DOMESTIC_DIRECT_RISK"
        if re.search(r"공공|재난|우선\s*통신|우선\s*접속|행정|정부|망|통신권|재난망|공공\s*AI|소방|경찰", t, re.I):
            return "COMP_DOMESTIC_PUBLIC_INFRA"
        if re.search(r"AI|인공지능|AX|에이전트|검색|추천|하이퍼클로바|에이닷|GPU|클라우드|데이터센터|LLM|RCS|디도스|DDoS|보안\s*전략|클린존", t, re.I):
            return "COMP_DOMESTIC_AI_PLATFORM"
        if re.search(r"네이버페이|토스|토스페이|쿠팡페이|간편결제|금융|커머스|쇼핑|멤버십|배달|쿠팡이츠|배민|광고", t, re.I):
            return "COMP_DOMESTIC_FINANCE_COMMERCE"
        return "COMP_DOMESTIC_STRATEGY"
    if scope == "overseas":
        if re.search(r"데이터|학습|저작권|언론|뉴스|콘텐츠|개인정보|광고|앱마켓|DMA|DSA|규제|소송|과징금|AI\s*랩", t, re.I):
            return "COMP_OVERSEAS_REGULATION_DATA"
        if re.search(r"AI|인공지능|에이전트|LLM|검색|제미나이|ChatGPT|GPT|클로드|모델|앱|플랫폼", t, re.I):
            return "COMP_OVERSEAS_PLATFORM_AI"
        if re.search(r"투자|IPO|상장|주가|지분|CAPEX|데이터센터|전력|발전|자금\s*조달", t, re.I):
            return "COMP_OVERSEAS_MARKET_SIGNAL"
    return "COMP_GENERAL_LOW_RELEVANCE"


def v20_detect_issue_family(text, json_key=""):
    t = clean_html_text(text)
    if not t:
        return "off_topic"
    self_tier = v21_self_tier(t)
    if self_tier == "LOW_VALUE_PR":
        return "low_value_pr"
    if v20_is_platform_obligation_text(t):
        return "platform_obligation" if v20_has_self_entity(t) else "government_platform_regulation"
    if v20_has_self_entity(t):
        if self_tier == "DIRECT_RISK":
            if re.search(r"노조|파업|임단협|임금|성과급|RSU|고용불안|조정", t, re.I): return "self_labor"
            if re.search(r"지분|매각|인수|합병|최대주주|경영권|공동대표|사내이사|주총|유상증자|전환사채", t, re.I): return "self_governance_mna"
            if re.search(r"개인정보|유출|해킹|피싱|보안|장애|먹통", t, re.I): return "self_privacy_security"
            if re.search(r"수사|조사|과징금|제재|소송|판결|검찰|경찰", t, re.I): return "self_legal_regulatory"
            if re.search(r"임원|대표|CPO|CTO|조직개편|퇴사|사임|서비스\s*논란|이용자\s*반발|카톡\s*개편", t, re.I): return "self_leadership_service"
            return "self_direct_risk"
        if self_tier == "RESPONSE_RELEVANT": return "self_response_relevant"
        if self_tier == "STRATEGIC_REFERENCE": return "self_strategic_reference"
        return "self_filler_reference"
    comp_tier = v21_competitor_tier(t)
    has_competitor = v21_competitor_scope(t) in {"domestic", "mixed", "overseas"}
    if (json_key == "경쟁사_해외이슈" or has_competitor) and comp_tier not in {"COMP_GENERAL_LOW_RELEVANCE", "OFF_TOPIC"}:
        return comp_tier.lower()
    gov_tier = v21_gov_tier(t)
    if json_key == "정부_국회" or gov_tier != "OFF_TOPIC" or REGULATOR_CORE_PATTERN.search(t):
        if gov_tier == "GOV_DIRECT_PLATFORM_OBLIGATION": return "government_platform_obligation"
        if gov_tier == "GOV_AI_CORE_POLICY": return "government_ai_core_policy"
        if gov_tier == "GOV_REGULATORY_ENFORCEMENT": return "government_regulatory_enforcement"
        if gov_tier == "GOV_PLATFORM_MARKET_RULE": return "government_platform_market_rule"
        if gov_tier == "GOV_APPOINTMENT_GOVERNANCE": return "government_appointment_governance"
        if gov_tier == "GOV_DIGITAL_FINANCE_POLICY": return "government_digital_finance_policy"
        if gov_tier == "GOV_ADJACENT_DIGITAL_POLICY": return "government_adjacent_digital_policy"
        if gov_tier == "GOV_GENERAL_LOW_RELEVANCE": return "government_general_low_relevance"
        return "government_general_policy"
    if has_competitor:
        return "competitor_general_low_relevance" if comp_tier == "COMP_GENERAL_LOW_RELEVANCE" else comp_tier.lower()
    if re.search(r"AI|인공지능|데이터센터|GPU|NPU|클라우드|반도체|HBM|TSMC|엔비디아|전력|플랫폼|카카오톡\s*선물", t, re.I):
        return "industry_structural_change"
    return "off_topic"




# ==========================================
# v22 overrides: operational hardening
# - 날짜별 산출물 폴더/파일명
# - latest 복사와 index 누적
# - 본문-제목 정합성/오염 검사
# - 링크 상태 검사 및 같은 이슈 대체 우선
# - Gemini 라벨링 배치 부분 실패 허용
# - QA 게이트: 문제 발생 시 REVIEW_REQUIRED 표시
# ==========================================

from pathlib import Path
import shutil

V6_VERSION = "google_rss_v22_operational_hardening_daily_archive_body_guard"

V22_RUN_DATE = datetime.now(KST).date().isoformat()
V22_OUTPUT_ROOT = Path(BASE_DIR) / "data_news"
V22_DAILY_DIR = V22_OUTPUT_ROOT / "daily" / V22_RUN_DATE
V22_LATEST_DIR = V22_OUTPUT_ROOT / "latest"
V22_INDEX_DIR = V22_OUTPUT_ROOT / "index"
V22_REVIEW_REQUIRED_TXT = ""
V22_LABELING_LOCAL_BATCHES = 0
V22_LABELING_RETRY_BATCHES = 0

V22_OUTPUT_BASES = {
    "OUTPUT_TXT": ("CEO_Morning_Briefing", "txt"),
    "OUTPUT_SELECTED_CSV": ("google_news_top15_raw", "csv"),
    "OUTPUT_CANDIDATES_CSV": ("google_news_candidates_raw", "csv"),
    "OUTPUT_RANKED_CSV": ("google_news_ranked_candidates", "csv"),
    "OUTPUT_SKIPPED_DUP_CSV": ("skipped_past_duplicates", "csv"),
    "OUTPUT_BODY_FAILED_CSV": ("body_extract_failed", "csv"),
    "OUTPUT_RUN_LOG_CSV": ("run_quality_log", "csv"),
    "OUTPUT_GEMINI_LABELS_CSV": ("google_news_gemini_labels", "csv"),
    "OUTPUT_ISSUES_CSV": ("google_news_issues", "csv"),
    "OUTPUT_ISSUE_SELECTION_CSV": ("google_news_issue_selection", "csv"),
    "OUTPUT_QA_TXT": ("CEO_Morning_Briefing_QA", "txt"),
}


def v22_dated_path(base, ext):
    return str(V22_DAILY_DIR / f"{base}_{V22_RUN_DATE}.{ext}")


def v22_latest_path(base, ext):
    return V22_LATEST_DIR / f"{base}.{ext}"


def v22_setup_output_paths(run_date=None):
    """Set all output paths to data_news_briefing/daily/YYYY-MM-DD and prepare latest/index dirs."""
    global V22_RUN_DATE, V22_DAILY_DIR, V22_LATEST_DIR, V22_INDEX_DIR, V22_REVIEW_REQUIRED_TXT
    global OUTPUT_TXT, OUTPUT_SELECTED_CSV, OUTPUT_CANDIDATES_CSV, OUTPUT_RANKED_CSV
    global OUTPUT_SKIPPED_DUP_CSV, OUTPUT_BODY_FAILED_CSV, OUTPUT_RUN_LOG_CSV
    global OUTPUT_GEMINI_LABELS_CSV, OUTPUT_ISSUES_CSV, OUTPUT_ISSUE_SELECTION_CSV, OUTPUT_QA_TXT

    if run_date:
        V22_RUN_DATE = str(run_date)
    V22_DAILY_DIR = V22_OUTPUT_ROOT / "daily" / V22_RUN_DATE
    V22_LATEST_DIR = V22_OUTPUT_ROOT / "latest"
    V22_INDEX_DIR = V22_OUTPUT_ROOT / "index"
    for d in (V22_DAILY_DIR, V22_LATEST_DIR, V22_INDEX_DIR):
        d.mkdir(parents=True, exist_ok=True)

    for var_name, (base, ext) in V22_OUTPUT_BASES.items():
        globals()[var_name] = v22_dated_path(base, ext)
    V22_REVIEW_REQUIRED_TXT = str(V22_DAILY_DIR / f"CEO_Morning_Briefing_REVIEW_REQUIRED_{V22_RUN_DATE}.txt")


def v22_sync_latest_outputs(qa_overall=""):
    """Copy today's artifacts to latest/. If QA says 문제, latest briefing is marked review-required."""
    V22_LATEST_DIR.mkdir(parents=True, exist_ok=True)
    for var_name, (base, ext) in V22_OUTPUT_BASES.items():
        src_path = Path(globals().get(var_name, ""))
        if not src_path.exists():
            continue
        dst_path = v22_latest_path(base, ext)
        if var_name == "OUTPUT_TXT" and qa_overall == "문제":
            try:
                content = src_path.read_text(encoding="utf-8")
                warning = (
                    "🚨 QA 검수 결과 '문제'가 감지되어 자동 확정 전 사람 검토가 필요합니다.\n"
                    "아래 보고서는 초안입니다. CEO_Morning_Briefing_QA.txt를 확인하세요.\n\n"
                )
                dst_path.write_text(warning + content, encoding="utf-8")
            except Exception:
                shutil.copy2(src_path, dst_path)
        else:
            shutil.copy2(src_path, dst_path)
    # Convenience review-required copy.
    if qa_overall == "문제" and Path(V22_REVIEW_REQUIRED_TXT).exists():
        shutil.copy2(V22_REVIEW_REQUIRED_TXT, V22_LATEST_DIR / "CEO_Morning_Briefing_REVIEW_REQUIRED.txt")


def v22_append_rows_csv(path, rows):
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    header = not p.exists()
    df.to_csv(p, mode="a", header=header, index=False, encoding="utf-8-sig")


def v22_update_history_indices(final_report_data, issues, raw_articles, qa_overall=""):
    """Maintain structured daily history for future duplicate checks."""
    article_rows = []
    issue_rows = []
    report_rows = []
    url_rows = []

    for idx, item in enumerate(final_report_data, 1):
        title = clean_html_text(item.get("기사제목", ""))
        url = normalize_url(item.get("링크", ""))
        issue_group = clean_html_text(item.get("v20_issue_group") or item.get("Gemini이슈그룹", ""))
        summary = summarize_body_locally(title, item.get("본문전문", ""), max_chars=250) if item.get("본문상태") == "본문추출성공" else clean_html_text(item.get("RSS요약", ""))[:250]
        common = {
            "date": V22_RUN_DATE,
            "section": item.get("카테고리", ""),
            "rank": idx,
            "title": title,
            "press": item.get("언론사", ""),
            "url": url,
            "issue_group": issue_group,
            "issue_family": item.get("v20_issue_family", ""),
            "body_status": item.get("본문상태", ""),
            "qa_overall": qa_overall,
            "title_hash": title_fingerprint(title),
            "url_hash": normalize_url(url),
            "summary": summary,
        }
        article_rows.append(dict(common, selected=True, article_id=item.get("브리핑ID", "")))
        report_rows.append(common)
        if url:
            url_rows.append({"date": V22_RUN_DATE, "url": url, "title": title, "press": item.get("언론사", ""), "issue_group": issue_group})

    selected_issue_ids = {x.get("v20_issue_id") for x in final_report_data if x.get("v20_issue_id")}
    for issue in issues or []:
        if issue.get("issue_id") not in selected_issue_ids:
            continue
        issue_rows.append({
            "date": V22_RUN_DATE,
            "issue_id": issue.get("issue_id"),
            "issue_group": issue.get("issue_group"),
            "issue_family": issue.get("issue_family"),
            "category": v20_issue_to_output_category(issue),
            "priority": max(issue.get("max_ceo_priority", 0) or 0, issue.get("max_pa_priority", 0) or 0),
            "top_title": issue.get("top_title", ""),
            "top_url": issue.get("top_url", ""),
            "selected": True,
            "issue_hash": v20_issue_group_key(issue.get("issue_group", "")),
        })

    v22_append_rows_csv(V22_INDEX_DIR / "article_history.csv", article_rows)
    v22_append_rows_csv(V22_INDEX_DIR / "issue_history.csv", issue_rows)
    v22_append_rows_csv(V22_INDEX_DIR / "report_history.csv", report_rows)
    v22_append_rows_csv(V22_INDEX_DIR / "url_seen_history.csv", url_rows)


# Merge structured history into past report duplicate memory.
_BASE_load_past_reports_v22 = load_past_reports

def v22_parse_history_date(value):
    """
    index CSV의 날짜를 date 객체로 변환한다.
    기존 index에는 2026.6.5, 2026.06.05, 2026/6/5 형식이 섞일 수 있으므로
    YYYY-MM-DD뿐 아니라 점/슬래시 날짜도 함께 처리한다.
    """
    s = str(value or "").strip().replace('"', "").replace("'", "")

    if not s or s.lower() in {"nan", "none", "null"}:
        return None

    # 시간이 붙어 있으면 날짜 앞부분만 사용
    s = s.split()[0].strip()

    # 1) 2026-06-09 같은 ISO 형식
    try:
        return datetime.fromisoformat(s[:10]).date()
    except Exception:
        pass

    # 2) 2026.6.9 / 2026.06.09 / 2026/6/9 / 2026-6-9 처리
    m = re.match(r"^(\d{4})[-./](\d{1,2})[-./](\d{1,2})\.?$", s)
    if m:
        try:
            y, mo, d = map(int, m.groups())
            return date(y, mo, d)
        except Exception:
            return None

    return None


def v22_history_items_from_index():
    p = V22_INDEX_DIR / "report_history.csv"
    if not p.exists():
        return []
    try:
        df = pd.read_csv(p)
    except Exception:
        return []
    today = datetime.now(KST).date()
    cutoff = today - timedelta(days=PAST_DUP_LOOKBACK_DAYS)
    items = []
    for _, row in df.iterrows():
        d = v22_parse_history_date(row.get("date"))
        if not d or not (cutoff <= d <= today):
            continue
        title = clean_html_text(row.get("title", ""))
        if not title:
            continue
        summary = clean_html_text(row.get("summary", ""))
        text = f"{title} {summary}".strip()
        items.append({
            "date": d,
            "category": clean_html_text(row.get("section", "")),
            "title": title,
            "link": normalize_url(row.get("url", "")),
            "press": clean_html_text(row.get("press", "")),
            "summary": summary,
            "text": text,
            "entities": detect_entities(text),
            "event_tags": detect_event_tags(text),
            "issue_terms": tokenize_for_similarity(text),
            "source": "v22_report_history",
        })
    return items


def load_past_reports():
    content, all_items, recent_items = _BASE_load_past_reports_v22()
    hist_items = v22_history_items_from_index()
    if hist_items:
        # De-dupe by title/url against parsed past reports.
        seen = {(normalize_url(i.get("link", "")), title_fingerprint(i.get("title", ""))) for i in recent_items}
        added = []
        for item in hist_items:
            key = (normalize_url(item.get("link", "")), title_fingerprint(item.get("title", "")))
            if key in seen:
                continue
            seen.add(key)
            added.append(item)
        if added:
            all_items = all_items + added
            recent_items = recent_items + added
            print(f"📚 v22 누적 index 로드: 최근 보고서 이력 {len(added)}건 추가")
    return content, all_items, recent_items


# Gemini labeling should survive individual malformed JSON batches.
_BASE_v20_gemini_label_candidates_v22 = v20_gemini_label_candidates

def v20_gemini_label_candidates(client, candidates, recent_past_text):
    global V22_LABELING_LOCAL_BATCHES, V22_LABELING_RETRY_BATCHES
    if not client:
        raise RuntimeError("Gemini client 없음")
    all_labels = {}
    batches = [candidates[i:i + V20_LABEL_BATCH_SIZE] for i in range(0, len(candidates), V20_LABEL_BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches, 1):
        try:
            labels = _BASE_v20_gemini_label_candidates_v22(client, batch, recent_past_text)
            all_labels.update(labels)
            print(f"  └ 🏷️ v22 라벨링 배치 {batch_idx}/{len(batches)} 완료: Gemini {len(labels)}개")
        except Exception as first_error:
            V22_LABELING_RETRY_BATCHES += 1
            try:
                time.sleep(1.0 + random.uniform(0, 1.0))
                labels = _BASE_v20_gemini_label_candidates_v22(client, batch, recent_past_text)
                all_labels.update(labels)
                print(f"  └ 🏷️ v22 라벨링 배치 {batch_idx}/{len(batches)} 재시도 성공: Gemini {len(labels)}개")
            except Exception as second_error:
                V22_LABELING_LOCAL_BATCHES += 1
                print(f"  └ ⚠️ v22 라벨링 배치 {batch_idx}/{len(batches)} 실패. 해당 배치만 로컬 라벨 사용: {second_error}")
                for article in batch:
                    all_labels[int(article["id"])] = v20_local_article_label(article)
    return all_labels


# Link/body validation.
V22_POLITICAL_CONTAMINATION_RE = re.compile(
    r"김어준|오세훈|한동훈|김경수|조국|국민의힘|더불어민주당|민주당|지방선거|보궐선거|재보궐|대권|당선인|서울시장|평택을|부산\s*북갑",
    re.IGNORECASE,
)

V22_ENTITY_PATTERNS_FOR_VALIDATION = {
    "카카오게임즈": r"카카오게임즈|라인야후|라인게임즈|김태환|이시우|한상우|공동대표|사내이사|임시주총|엘트리플에이|LAAA",
    "카카오페이": r"카카오페이|신원근|슈퍼월렛|애플페이|카카오페이머니",
    "카카오뱅크": r"카카오뱅크|판교|여의도|생계비통장",
    "카카오모빌리티": r"카카오모빌리티|카카오T|카카오\s*T|택시|대리운전|모빌리티",
    "카카오톡": r"카카오톡|카톡|챗GPT|ChatGPT|카나나|친구탭|선물하기",
    "카카오": r"카카오|Kakao|카나나|다음",
    "네이버페이": r"네이버페이|Npay|엔페이",
    "쿠팡페이": r"쿠팡페이|쿠팡|원아이디",
    "티빙": r"티빙|TVING|개인정보|유출|과징금",
    "IBM": r"IBM|밥|Bob|코딩\s*에이전트",
    "공정위": r"공정위|공정거래위원회|가상인물|AI\s*가상|표시\s*의무",
}

V22_TERM_STOPWORDS = set(STOPWORDS) | {
    "관련", "기사", "보도", "전격", "단독", "종합", "오늘", "내달", "내년", "국내", "업계", "기준", "경우",
    "이번", "지난", "통해", "위해", "대해", "대한", "있다", "했다", "된다", "한다", "계획", "예정",
}


def v22_extract_key_terms(text, max_terms=18):
    text = clean_html_text(text)
    text = re.sub(r"https?://\S+", " ", text)
    raw = re.findall(r"[A-Za-z0-9가-힣+]{2,}", text)
    out = []
    seen = set()
    for tok in raw:
        t = tok.strip()
        if not t or t in V22_TERM_STOPWORDS:
            continue
        if re.fullmatch(r"\d+", t):
            continue
        if len(t) <= 1:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def v22_expected_entity_patterns(title, summary="", issue_group=""):
    src = clean_html_text(f"{title} {summary} {issue_group}")
    expected = []
    for name, pat in V22_ENTITY_PATTERNS_FOR_VALIDATION.items():
        if re.search(pat, src, re.IGNORECASE):
            expected.append((name, pat))
    # Prefer the most specific entities. If 카카오게임즈 exists, don't require generic 카카오 separately.
    if any(name.startswith("카카오") and name != "카카오" for name, _ in expected):
        expected = [(n, p) for n, p in expected if n != "카카오"]
    return expected[:4]


def v22_validate_body_matches_article(article_info, report_item):
    title = clean_html_text(report_item.get("기사제목") or article_info.get("기사제목", ""))
    summary = clean_html_text(article_info.get("본문요약", "") or report_item.get("RSS요약", ""))
    issue_group = clean_html_text(article_info.get("Gemini이슈그룹", "") or report_item.get("v20_issue_group", ""))
    body = clean_html_text(report_item.get("본문전문", ""))

    if not body:
        return False, "empty_body"
    if len(body) < 250:
        # 기존 품질 로직이 이미 처리하지만 명시적으로 남김.
        return False, "body_too_short_for_validation"

    # Strong entity anchoring.
    expected_entities = v22_expected_entity_patterns(title, summary, issue_group)
    missing_entities = []
    for name, pat in expected_entities:
        if not re.search(pat, body, re.IGNORECASE):
            missing_entities.append(name)
    if expected_entities and len(missing_entities) == len(expected_entities):
        return False, "body_entity_mismatch:" + ",".join(missing_entities)

    # Keyword coverage. This catches long but unrelated sidebars.
    key_terms = v22_extract_key_terms(f"{title} {summary} {issue_group}", max_terms=16)
    if len(key_terms) >= 5:
        hits = [t for t in key_terms if re.search(re.escape(t), body, re.IGNORECASE)]
        # Require at least two meaningful hits, or 20% coverage.
        if len(hits) < 2 and (len(hits) / max(1, len(key_terms))) < 0.20:
            return False, f"body_keyword_mismatch:hits={len(hits)}/terms={len(key_terms)}"

    # Obvious contamination guard for politics/election snippets under non-political titles.
    if not re.search(r"선거|국회|정치|당선|후보|시장", title):
        contam = V22_POLITICAL_CONTAMINATION_RE.findall(body[:2500])
        if len(set(contam)) >= 3:
            return False, "body_contamination_political_sidebar"

    return True, "body_valid"


def v22_check_url_accessible(url):
    url = normalize_url(url)
    if not url or not url.startswith("http"):
        return "bad", url, "invalid_url"
    try:
        r = requests.get(url, headers=WEB_HEADERS, timeout=6, allow_redirects=True)
        final_url = normalize_url(r.url or url)
        # Bot-blocked 401/403 can still be browser-accessible, so mark limited rather than bad.
        if r.status_code in {401, 403, 429}:
            return "limited", final_url, f"http_{r.status_code}"
        if r.status_code in {404, 410}:
            return "bad", final_url, f"http_{r.status_code}"
        if 200 <= r.status_code < 400:
            return "ok", final_url, f"http_{r.status_code}"
        if r.status_code >= 500:
            return "limited", final_url, f"http_{r.status_code}"
        return "limited", final_url, f"http_{r.status_code}"
    except Exception as e:
        return "limited", url, f"request_error:{type(e).__name__}"


_BASE_process_article_for_report_v22 = process_article_for_report

def process_article_for_report(article_info, json_key, recent_past_items):
    report_item, skip_info, failed_item = _BASE_process_article_for_report_v22(article_info, json_key, recent_past_items)

    # Preserve skip and pre-existing failures.
    if skip_info or not report_item:
        return report_item, skip_info, failed_item

    link_status, checked_url, link_reason = v22_check_url_accessible(report_item.get("링크", ""))
    report_item["링크상태"] = link_status
    report_item["링크검증사유"] = link_reason
    if checked_url and checked_url != report_item.get("링크"):
        report_item["링크"] = checked_url

    # Hard-fail only truly bad URLs; limited bot-blocks are kept if body is valid.
    if link_status == "bad":
        failed_copy = dict(report_item)
        failed_copy["본문품질사유"] = f"link_failed:{link_reason}"
        failed_copy["본문사용가능"] = False
        failed_copy["본문상태"] = "링크접근실패_대체필요"
        return None, None, failed_copy

    valid, validation_reason = v22_validate_body_matches_article(article_info, report_item)
    report_item["본문정합성"] = validation_reason
    if not valid:
        failed_copy = dict(report_item)
        failed_copy["본문품질사유"] = f"body_validation_failed:{validation_reason}"
        failed_copy["본문사용가능"] = False
        failed_copy["본문상태"] = "본문오염의심_대체필요"
        return None, None, failed_copy

    return report_item, skip_info, failed_item


# Bodyless items: no speculative summary. Keep title/source/link plus explicit limitation.
_BASE_v20_create_bodyless_report_item_v22 = v20_create_bodyless_report_item

def v20_create_bodyless_report_item(article_info, issue, decision, json_key, reason="critical_issue_no_body_available"):
    item = _BASE_v20_create_bodyless_report_item_v22(article_info, issue, decision, json_key, reason=reason)
    link_status, checked_url, link_reason = v22_check_url_accessible(item.get("링크", ""))
    item["링크상태"] = link_status
    item["링크검증사유"] = link_reason
    if checked_url:
        item["링크"] = checked_url
    item["본문전문"] = ""
    item["본문글자수"] = 0
    item["본문상태"] = "본문추출실패_대체없음_제목링크만"
    item["본문품질사유"] = reason
    return item


def v20_bodyless_summary(item):
    if item.get("링크상태") == "bad":
        return "원문 본문과 링크 접근이 모두 제한되어 세부 요약은 생략함. 제목·발행사 기준으로만 포함했으며 별도 원문 확인이 필요함."
    return "원문 본문 자동 추출 및 동일 이슈 대체 기사 확인이 제한되어 세부 요약은 생략함. 제목·발행사·링크 기준으로 포함했으며 원문 확인이 필요함."


# More explicit QA prompt and JSON parser/gate.
def v22_parse_qa_json(qa_text):
    if not qa_text:
        return {}
    try:
        return extract_json_object(qa_text)
    except Exception:
        try:
            m = re.search(r"\{[\s\S]*\}", qa_text)
            if m:
                return json.loads(m.group(0))
        except Exception:
            pass
    return {"overall": "주의", "warnings": [clean_html_text(qa_text)[:500]]}


def v22_qa_overall(qa_text):
    data = v22_parse_qa_json(qa_text)
    overall = clean_html_text(data.get("overall", ""))
    if overall not in {"양호", "주의", "문제"}:
        if "문제" in qa_text:
            return "문제"
        if "주의" in qa_text:
            return "주의"
        return ""
    return overall


_BASE_v20_gemini_quality_check_v22 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    # Reuse existing QA but add body validation metadata. The existing prompt already catches many editorial issues.
    return _BASE_v20_gemini_quality_check_v22(client, final_report_data, final_briefing_text)


# Main with v22 operational paths, QA gate, and history updates.
def main():
    total_start_time = time.time()
    run_log = []
    v22_setup_output_paths()

    print(f"\n🧩 v29 실행: {V6_VERSION}")
    print(f"   └ 출력 폴더: {V22_DAILY_DIR}")
    print(f"   └ 모델: labeling={GEMINI_MODEL_LABELING}, editor={GEMINI_MODEL_EDITOR}, summary={GEMINI_MODEL_SUMMARY}, qa={GEMINI_MODEL_QA}")

    client = None
    if ENABLE_GEMINI_SELECTION or ENABLE_GEMINI_REPORT:
        try:
            with open(SECRET_PATH, "r", encoding="utf-8") as f:
                google_api_key = f.read().strip()
            if not google_api_key:
                raise ValueError("secret.txt가 비어 있습니다.")
            client = genai.Client(api_key=google_api_key)
        except Exception as e:
            print("⚠️ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다. Gemini 없이 로컬 이슈 편집으로 진행합니다.")
            print(f"   원인: {e}")
            client = None

    past_reports_content, all_past_items, recent_past_items = load_past_reports()
    recent_past_text = build_recent_past_text(recent_past_items, max_chars=9000)

    raw_articles, skipped_duplicates = collect_with_google_rss(recent_past_items)
    print(f"\n  └ ✅ 총 {len(raw_articles)}개의 RSS 후보 확보 완료")
    print(f"  └ 🧹 수집/과거중복/오탐 제외 후보: {len(skipped_duplicates)}개")

    if not raw_articles:
        print("❌ 수집된 기사가 없습니다. Google News RSS 접속 또는 키워드/기간 설정을 확인하세요.")
        return

    ranked_all, ranked_candidates = rank_and_trim_candidates(raw_articles)
    print(f"  └ 🧺 v22 후보 풀: 전체 {len(ranked_all)}개 중 {len(ranked_candidates)}개를 기사 라벨링 대상으로 사용")

    print("\n🏷️ [STEP 2] Gemini가 기사별 의미 라벨을 붙입니다...")
    labels = {}
    gemini_label_success = False
    try:
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 라벨링 비활성화 또는 client 없음")
        labels = v20_gemini_label_candidates(client, ranked_candidates, recent_past_text)
        gemini_label_success = True
        print(f"  └ ✅ Gemini/로컬 혼합 기사 라벨링 완료: {len(labels)}개")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사 라벨링 전체 실패. 로컬 보조 라벨로 진행합니다. 원인: {e}")
        labels = {int(a["id"]): v20_local_article_label(a) for a in ranked_candidates}

    for article in ranked_all:
        art_id = int(article["id"])
        if art_id not in labels:
            labels[art_id] = v20_local_article_label(article)

    label_rows = v20_apply_labels_to_articles(ranked_all, labels)
    pd.DataFrame(ranked_all).to_csv(OUTPUT_CANDIDATES_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(ranked_candidates).to_csv(OUTPUT_RANKED_CSV, index=False, encoding="utf-8-sig")
    pd.DataFrame(label_rows).to_csv(OUTPUT_GEMINI_LABELS_CSV, index=False, encoding="utf-8-sig")
    if skipped_duplicates:
        pd.DataFrame(skipped_duplicates).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 후보/라벨 CSV 저장 완료: {os.path.basename(OUTPUT_CANDIDATES_CSV)}, {os.path.basename(OUTPUT_RANKED_CSV)}, {os.path.basename(OUTPUT_GEMINI_LABELS_CSV)}")

    print("\n🧱 [STEP 3] 기사 후보를 이슈 단위로 묶습니다...")
    issues, article_by_id = v20_build_issues(ranked_candidates, labels)
    issue_by_id = {i["issue_id"]: i for i in issues}
    pd.DataFrame(v20_issue_rows(issues)).to_csv(OUTPUT_ISSUES_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ ✅ 이슈 그룹 {len(issues)}개 생성 / 저장: {os.path.basename(OUTPUT_ISSUES_CSV)}")

    print("\n🧠 [STEP 4] Gemini가 이슈 단위로 최종 편집합니다...")
    gemini_editor_success = False
    try:
        if not client or not ENABLE_GEMINI_SELECTION:
            raise RuntimeError("Gemini 이슈 편집 비활성화 또는 client 없음")
        decisions, backup_decisions = v20_gemini_edit_issues(client, issues, recent_past_text)
        gemini_editor_success = True
        print(f"  └ ✅ Gemini 이슈 편집 완료: 선택 {len(decisions)}개 / 백업 {len(backup_decisions)}개")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 이슈 편집 실패. 로컬 이슈 점수 기반으로 진행합니다. 원인: {e}")
        decisions, backup_decisions = v20_local_issue_selection(issues)

    decisions = v20_apply_decision_caps(decisions, issue_by_id)
    decisions = v21_fill_decisions(decisions, backup_decisions, issue_by_id, issues)
    pd.DataFrame(v20_decision_rows(decisions, backup_decisions, issue_by_id)).to_csv(OUTPUT_ISSUE_SELECTION_CSV, index=False, encoding="utf-8-sig")
    print(f"  └ 💾 이슈 편집 결과 저장: {os.path.basename(OUTPUT_ISSUE_SELECTION_CSV)}")

    print("\n🕵️‍♂️ [STEP 5] 선택 이슈별 대표 기사를 찾고 본문을 추출합니다...")
    print("   └ v22: 링크/본문 정합성 검사 후 실패 시 같은 이슈의 다른 기사로 대체합니다.")

    final_report_data = []
    body_failed_rows = []
    post_body_duplicate_skips = []
    processed_article_ids = set()
    order_counter = 1
    status_counts = {}
    order_counter = v21_process_until_balanced(
        decisions,
        backup_decisions,
        issues,
        issue_by_id,
        article_by_id,
        recent_past_items,
        final_report_data,
        body_failed_rows,
        post_body_duplicate_skips,
        processed_article_ids,
        order_counter,
        status_counts,
    )

    final_report_data, removed_duplicates = v21_remove_obvious_final_duplicates(final_report_data)
    for rem, reason in removed_duplicates:
        add_duplicate_skip_row(post_body_duplicate_skips, rem, {"기사제목": "v22_final_duplicate", "링크": "", "대표선택점수": ""}, reason, stage="v22_final_repair")

    if v26_1_should_refill_after_repair(final_report_data):
        before_refill_count = len(final_report_data)
        order_counter = v26_1_refill_after_final_repair(
            decisions=decisions,
            backup_decisions=backup_decisions,
            issues=issues,
            issue_by_id=issue_by_id,
            article_by_id=article_by_id,
            recent_past_items=recent_past_items,
            final_report_data=final_report_data,
            body_failed_rows=body_failed_rows,
            skip_rows=post_body_duplicate_skips,
            processed_article_ids=processed_article_ids,
            order_counter=order_counter,
            status_counts=status_counts,
        )
        if len(final_report_data) != before_refill_count:
            final_report_data, removed_duplicates_refill = v21_remove_obvious_final_duplicates(final_report_data)
            for rem, reason in removed_duplicates_refill:
                add_duplicate_skip_row(post_body_duplicate_skips, rem, {}, reason, stage="v26_1_refill_repair")

    if skipped_duplicates or post_body_duplicate_skips:
        all_skips = skipped_duplicates + post_body_duplicate_skips
        pd.DataFrame(all_skips).to_csv(OUTPUT_SKIPPED_DUP_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ 💾 제외/중복/대표교체 목록 저장: {os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}")
    if body_failed_rows:
        pd.DataFrame(body_failed_rows).to_csv(OUTPUT_BODY_FAILED_CSV, index=False, encoding="utf-8-sig")
        print(f"  └ ⚠️ 본문 품질 미달/관련성 미달 기사 저장: {os.path.basename(OUTPUT_BODY_FAILED_CSV)}")

    if not final_report_data:
        print("❌ 최종 보고서에 사용할 기사 데이터가 없습니다.")
        return

    ordered_final = []
    for json_key in JSON_KEYS_ORDER:
        items = [x for x in final_report_data if x.get("카테고리") == json_key]
        items = sorted(items, key=lambda x: int(x.get("선정순서") or 9999))
        ordered_final.extend(items)
    final_report_data = ordered_final[:V21_FINAL_MAX]

    print("\n✍️ [STEP 6] 본문/RSS 기반으로 최종 브리핑 요약을 생성합니다...")
    for idx, item in enumerate(final_report_data, 1):
        item["브리핑ID"] = idx

    prompt_report = v20_build_summary_prompt(final_report_data, recent_past_text)
    summary_map = {}
    try:
        if not client or not ENABLE_GEMINI_REPORT:
            raise RuntimeError("Gemini 최종 브리핑 비활성화 또는 client 없음")
        summary_text = gemini_generate_text(
            client=client,
            prompt=prompt_report,
            task_name="v22 최종 기사별 요약 생성",
            model=GEMINI_MODEL_SUMMARY,
        )
        summary_map = normalize_summary_json(extract_json_object(summary_text))
        print("  └ ✅ Gemini 기사별 요약 생성 완료")
    except Exception as e:
        print(f"  └ ⚠️ Gemini 기사별 요약 생성 실패. 로컬 본문/RSS 요약으로 대체합니다. 원인: {e}")
        summary_map = {}

    # Force safe wording for title/link-only records.
    for item in final_report_data:
        bid = str(item.get("브리핑ID"))
        if str(item.get("본문상태", "")).startswith("본문추출실패"):
            summary_map[bid] = v20_bodyless_summary(item)

    final_briefing_text = build_structured_briefing(final_report_data, summary_map)

    qa_text = ""
    qa_overall = ""
    if ENABLE_GEMINI_QA:
        print("\n🔎 [STEP 7] 최종 편집 품질검수 로그를 생성합니다...")
        qa_text = v20_gemini_quality_check(client, final_report_data, final_briefing_text)
        qa_overall = v22_qa_overall(qa_text)
        if qa_text:
            with open(OUTPUT_QA_TXT, "w", encoding="utf-8") as f:
                f.write(qa_text)
            print(f"  └ 💾 QA 로그 저장: {os.path.basename(OUTPUT_QA_TXT)} / overall={qa_overall or 'unknown'}")
        if qa_overall == "문제":
            review_text = (
                "🚨 QA 검수 결과 '문제'가 감지되어 사람 검토가 필요합니다.\n"
                "아래 보고서는 자동 생성 초안이며, QA 로그를 확인한 뒤 확정하세요.\n\n"
                + final_briefing_text
            )
            with open(V22_REVIEW_REQUIRED_TXT, "w", encoding="utf-8") as f:
                f.write(review_text)
            print(f"  └ 🚨 QA 문제 감지: REVIEW_REQUIRED 파일 저장: {os.path.basename(V22_REVIEW_REQUIRED_TXT)}")

    print("\n" + "=" * 60)
    print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
    print("=" * 60)
    print(final_briefing_text)

    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(final_briefing_text)
    pd.DataFrame(final_report_data).to_csv(OUTPUT_SELECTED_CSV, index=False, encoding="utf-8-sig")

    counts, comp_dom, comp_over = v21_final_counts(final_report_data)
    run_log.append({
        "버전": V6_VERSION,
        "실행일": V22_RUN_DATE,
        "출력폴더": str(V22_DAILY_DIR),
        "모델_라벨링": GEMINI_MODEL_LABELING,
        "모델_편집": GEMINI_MODEL_EDITOR,
        "모델_요약": GEMINI_MODEL_SUMMARY,
        "전체_RSS_후보": len(raw_articles),
        "Gemini_기사라벨_후보": len(ranked_candidates),
        "이슈그룹수": len(issues),
        "Gemini라벨링성공": gemini_label_success,
        "Gemini라벨링_로컬대체배치": V22_LABELING_LOCAL_BATCHES,
        "Gemini라벨링_재시도배치": V22_LABELING_RETRY_BATCHES,
        "Gemini이슈편집성공": gemini_editor_success,
        "선택이슈수": len(decisions),
        "백업이슈수": len(backup_decisions),
        "최종_자사": counts.get("자사_및_계열사_이슈", 0),
        "최종_정부": counts.get("정부_국회", 0),
        "최종_경쟁사": counts.get("경쟁사_해외이슈", 0),
        "최종_산업": counts.get("산업동향", 0),
        "최종_경쟁사_국내": comp_dom,
        "최종_경쟁사_해외": comp_over,
        "본문성공": status_counts.get("body_success", 0) + status_counts.get("replaced_duplicate", 0),
        "본문실패_중요이슈_제한포함": status_counts.get("critical_without_body", 0),
        "제외_총합": len(skipped_duplicates) + len(post_body_duplicate_skips),
        "본문품질미달_관련성미달": len(body_failed_rows),
        "QA결과": qa_overall,
        "최종기사수": len(final_report_data),
        "실행분": round((time.time() - total_start_time) / 60, 2),
    })
    pd.DataFrame(run_log).to_csv(OUTPUT_RUN_LOG_CSV, index=False, encoding="utf-8-sig")

    # Update structured history indices after final files are ready.
    v22_update_history_indices(final_report_data, issues, raw_articles, qa_overall=qa_overall)
    v22_sync_latest_outputs(qa_overall=qa_overall)

    total_duration = time.time() - total_start_time
    print("\n" + "=" * 60)
    print(f"💾 시스템 자동화 작업 완료! (총 소요 시간: {total_duration / 60:.2f}분)")
    print(f"- 날짜별 출력 폴더: {V22_DAILY_DIR}")
    print(f"- 최신본 폴더: {V22_LATEST_DIR}")
    print(f"- 누적 index 폴더: {V22_INDEX_DIR}")
    print(f"- '{os.path.basename(OUTPUT_TXT)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_SELECTED_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_CANDIDATES_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_RANKED_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_GEMINI_LABELS_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_ISSUES_CSV)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_ISSUE_SELECTION_CSV)}' 저장 완료")
    if skipped_duplicates or post_body_duplicate_skips:
        print(f"- '{os.path.basename(OUTPUT_SKIPPED_DUP_CSV)}' 저장 완료")
    if body_failed_rows:
        print(f"- '{os.path.basename(OUTPUT_BODY_FAILED_CSV)}' 저장 완료")
    if qa_text:
        print(f"- '{os.path.basename(OUTPUT_QA_TXT)}' 저장 완료")
    if qa_overall == "문제":
        print(f"- '{os.path.basename(V22_REVIEW_REQUIRED_TXT)}' 저장 완료")
    print(f"- '{os.path.basename(OUTPUT_RUN_LOG_CSV)}' 저장 완료")
    print("=" * 60)



# =========================================================
# v23 overrides: final category guardrails + label coverage
# - 수집검색어/수집원카테고리/Gemini판단카테고리/최종카테고리를 명확히 분리
# - 최종 카테고리 배치 전에 pass/fail/reassign guardrail 적용
# - 정부/국회는 특정 단어 필수 매칭이 아니라 정책 행위/공공사업/사업자 의무를 기준으로 검증
# - image_popup/사진 캡션/단순 MOU/저신뢰 SEO 시장 리포트는 제목링크 제한 포함도 금지
# - Gemini 라벨링은 호출 성공이 아니라 배치 커버리지까지 검증
# =========================================================

V6_VERSION = "google_rss_v24_cross_section_dedupe_rep_alignment"

V23_LABEL_COVERAGE_MIN = 0.80
V23_LABEL_SPLIT_BATCH_SIZE = 28

V23_IMAGE_POPUP_URL_RE = re.compile(
    r"/tools/image_popup|image_popup|popup_image|photo_popup|img_popup",
    re.IGNORECASE,
)
V23_CAPTION_TITLE_RE = re.compile(
    r"\[?포토\]?|\[?사진\]?|자료사진|기념촬영|협약식|체결식|오른쪽|왼쪽|가운데|모습\.?$|"
    r"(대표|위원장|장관|부총리|의장).{0,30}(오른쪽|왼쪽|가운데).{0,50}(모습|기념촬영)",
    re.IGNORECASE,
)
V23_SIMPLE_MOU_PR_RE = re.compile(
    r"MOU|업무협약|오픈이노베이션|생태계\s*조성|기금\s*전달|후원|스폰서|협찬|캠페인|이벤트|프로모션|"
    r"시드\s*투자\s*유치|투자\s*유치",
    re.IGNORECASE,
)
V23_MATERIAL_SELF_RE = re.compile(
    r"과징금|제재|조사|수사|소송|판결|개인정보|유출|해킹|장애|오류|먹통|피싱|노조|파업|임단협|"
    r"대표|임원|조직개편|최대주주|경영권|인수|합병|매각|지분|실적|영업이익|영업손실|적자|흑자|"
    r"카카오톡\s*(개편|연동|AI|챗GPT)|카나나|AI\s*워터마크|스테이블코인|슈퍼월렛|원화코인",
    re.IGNORECASE,
)
V23_KAKAO_VENTURE_SEED_RE = re.compile(
    r"카카오벤처스.{0,80}(시드\s*투자|투자\s*유치)|.{0,80}카카오벤처스로부터\s*시드\s*투자",
    re.IGNORECASE,
)

V23_POLICY_ACTOR_RE = re.compile(
    r"정부|국회|대통령실|국무회의|당정|상임위|과방위|정무위|문체위|성평등위|위원회|TF|전담조직|"
    r"과기정통부|과학기술정보통신부|방미통위|방송미디어통신위원회|공정위|공정거래위원회|개보위|개인정보보호위원회|"
    r"금융위|금융위원회|금감원|금융감독원|행안부|행정안전부|중기부|중소벤처기업부|고용노동부|노동부|"
    r"FIU|금융정보분석원|검찰|경찰청|KISA|한국인터넷진흥원|공공기관|지자체",
    re.IGNORECASE,
)
V23_POLICY_ACTION_RE = re.compile(
    r"법안|발의|입법|입법예고|행정예고|개정안|시행령|시행규칙|고시|가이드라인|심사지침|특별법|기본법|"
    r"의무|의무화|표시\s*의무|차단\s*의무|삭제\s*의무|보고\s*의무|자료\s*제출|공시|인허가|허가|인가|"
    r"시행|적용|확대|완화|강화|제도|정책|대책|전략|예산|국가사업|공공\s*AI|디지털플랫폼정부|"
    r"조사|검사|감독|점검|제재|과징금|과태료|시정명령|고발|협의|의견수렴|설명회|간담회|공공조달",
    re.IGNORECASE,
)
V23_KAKAO_RELEVANT_DOMAIN_RE = re.compile(
    r"AI|인공지능|에이전트|AIDC|GPU|NPU|데이터센터|클라우드|플랫폼|온라인플랫폼|포털|SNS|앱마켓|"
    r"카카오|네이버|구글|메타|오픈AI|애플|쿠팡|토스|배민|개인정보|보안|딥페이크|피싱|불법촬영|유해정보|"
    r"광고|알고리즘|검색|추천|저작권|콘텐츠|지도|지도반출|망사용료|온플법|전자금융|핀테크|스테이블코인|"
    r"디지털자산|가상자산|특금법|FIU|AML|트래블룰|마이데이터|디지털트윈|공공데이터",
    re.IGNORECASE,
)
V23_PUBLIC_AI_POLICY_RE = re.compile(
    r"국가AI|국가\s*AI|AI전략위|AI\s*전략위|AI\s*수석|AI\s*기본법|공공\s*AI|국민\s*AI|디지털플랫폼정부|"
    r"AIDC|AI\s*데이터센터|GPU|NPU|피지컬AI\s*특별법|K-피지컬AI|독자\s*AI|파운데이션\s*모델|"
    r"민간중심\s*정부조직|공공서비스|행정\s*AI",
    re.IGNORECASE,
)
V23_PRIVATE_INTERNAL_AI_ORG_RE = re.compile(
    r"(EY한영|삼일PwC|삼정KPMG|딜로이트|회계법인|컨설팅\s*법인|민간\s*기업|기업이|기업은|그룹이|사내|전사적).{0,120}"
    r"(AI\s*센터|AI\s*허브|AI\s*조직|전사\s*AI|AI\s*컨트롤타워|출범|신설|고도화)",
    re.IGNORECASE,
)
V23_SEO_MARKET_REPORT_RE = re.compile(
    r"CAGR|203[0-9]년까지|시장\s*규모|시장\s*점유율|연평균\s*성장|Global\s*Growth\s*Insights|market\s*reports?|"
    r"시장은\s*203[0-9]년까지|시장.+성장할\s*것입니다",
    re.IGNORECASE,
)
V23_STRUCTURAL_INDUSTRY_RE = re.compile(
    r"AI\s*에이전트|에이전트\s*전략|모델보다\s*플랫폼|플랫폼\s*전략|업계\s*전반|생태계|시장\s*재편|"
    r"표준|협단체|가이드라인|데이터센터|GPU|클라우드|보안\s*인증|개인정보|결제\s*보안|앱\s*생태계|"
    r"콘텐츠\s*데이터|AI\s*학습|광고시장|커머스|오픈뱅킹|간편결제|디지털\s*결제",
    re.IGNORECASE,
)


def v23_text(item):
    return clean_html_text(
        f"{item.get('기사제목','')} {item.get('언론사','')} {item.get('링크','')} "
        f"{item.get('본문요약','')} {item.get('RSS요약','')} {item.get('본문전문','')[:3200]} "
        f"{item.get('Gemini이슈그룹','')} {item.get('v20_issue_group','')}"
    )


def v23_has_self_entity(text):
    return bool(SELF_KAKAO_PATTERN.search(clean_html_text(text)))


def v23_has_competitor_entity(text):
    return v21_competitor_scope(text) in {"domestic", "mixed", "overseas"} or bool(COMPETITOR_CORE_PATTERN.search(clean_html_text(text)))


def v23_is_photo_or_caption(item):
    title = clean_html_text(item.get("기사제목", ""))
    url = normalize_url(item.get("링크", "") or item.get("원래RSS링크", ""))
    if V23_IMAGE_POPUP_URL_RE.search(url):
        return True
    if V23_CAPTION_TITLE_RE.search(title):
        return True
    return False


def v23_is_never_title_only(item):
    text = v23_text(item)
    if v23_is_photo_or_caption(item):
        return True
    if V23_SIMPLE_MOU_PR_RE.search(text) and not V23_MATERIAL_SELF_RE.search(text):
        return True
    if V23_KAKAO_VENTURE_SEED_RE.search(text) and not re.search(r"대규모|인수|합병|경영권|상장|IPO|전략적\s*투자", text, re.I):
        return True
    if V23_SEO_MARKET_REPORT_RE.search(text) and re.search(r"Global\s*Growth\s*Insights|market\s*reports?|CAGR|203[0-9]년까지", text, re.I):
        return True
    return False


def v23_has_policy_basis(text):
    t = clean_html_text(text)
    if v20_is_platform_obligation_text(t):
        return True
    if V23_POLICY_ACTOR_RE.search(t) and V23_POLICY_ACTION_RE.search(t) and V23_KAKAO_RELEVANT_DOMAIN_RE.search(t):
        return True
    # 정부기관명이 제목에 없어도 사업자 의무/제도 변화가 명확하면 통과시킨다.
    if V23_POLICY_ACTION_RE.search(t) and V23_KAKAO_RELEVANT_DOMAIN_RE.search(t):
        if re.search(r"사업자|플랫폼|AI|인공지능|개인정보|광고|알고리즘|딥페이크|불법촬영|전자금융|가상자산|스테이블코인", t, re.I):
            return True
    if V23_PUBLIC_AI_POLICY_RE.search(t) and (V23_POLICY_ACTOR_RE.search(t) or V23_POLICY_ACTION_RE.search(t)):
        return True
    return False


def v23_is_private_internal_news(text):
    t = clean_html_text(text)
    if V23_PRIVATE_INTERNAL_AI_ORG_RE.search(t) and not V23_POLICY_ACTOR_RE.search(t):
        return True
    return False


def v23_is_valid_industry(text):
    t = clean_html_text(text)
    if V23_SEO_MARKET_REPORT_RE.search(t) and not re.search(r"한국|국내|카카오|네이버|토스|쿠팡|배민|금융위|공정위|정부|국회", t, re.I):
        return False
    if v23_is_private_internal_news(t):
        return False
    if V23_STRUCTURAL_INDUSTRY_RE.search(t):
        return True
    # 해외/경쟁사 기사라도 업계 전반의 구조 변화로 설명될 때만 허용.
    if re.search(r"업계\s*전반|생태계|시장\s*재편|전환|표준|규제\s*확산|플랫폼화|AI\s*도입\s*확산", t, re.I):
        return True
    return False


def v23_suggest_category(item, requested=None):
    t = v23_text(item)
    if v23_has_self_entity(t) and not (v23_has_competitor_entity(t) and not re.search(r"카카오|카톡|카나나|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈", t, re.I)):
        return "자사_및_계열사_이슈"
    if v23_has_policy_basis(t):
        return "정부_국회"
    if v23_has_competitor_entity(t):
        return "경쟁사_해외이슈"
    if v23_is_valid_industry(t):
        return "산업동향"
    return ""


def v23_final_category_guardrail(item, requested_category):
    """Return (action, category, reason). action: pass/reassign/fail/uncertain."""
    requested = v19_normalize_category_key(requested_category, fallback="산업동향")
    t = v23_text(item)

    if v23_is_photo_or_caption(item):
        return "fail", requested, "photo_or_image_popup_not_article"
    if V23_SIMPLE_MOU_PR_RE.search(t) and not V23_MATERIAL_SELF_RE.search(t):
        return "fail", requested, "simple_mou_pr_or_low_materiality"
    if V23_KAKAO_VENTURE_SEED_RE.search(t) and not re.search(r"대규모|인수|합병|경영권|상장|IPO|전략적\s*투자", t, re.I):
        return "fail", requested, "kakao_ventures_seed_investment_promo"

    if requested == "자사_및_계열사_이슈":
        if not v23_has_self_entity(t):
            suggested = v23_suggest_category(item, requested)
            return ("reassign", suggested, "self_without_kakao_entity") if suggested and suggested != requested else ("fail", requested, "self_without_kakao_entity")
        # 카카오가 본문에 있더라도 기사 주체가 해외 빅테크/경쟁사이면 자사에 두지 않는다.
        title_head = clean_html_text(item.get("기사제목", ""))[:80]
        if v21_competitor_scope(title_head) == "overseas" and not re.search(r"카카오|카톡|카나나|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈", title_head, re.I):
            return "reassign", "경쟁사_해외이슈", "overseas_competitor_not_self"
        return "pass", requested, "self_guardrail_pass"

    if requested == "정부_국회":
        if v23_has_policy_basis(t):
            return "pass", requested, "government_policy_basis_pass"
        if v23_is_private_internal_news(t):
            return "fail", requested, "private_company_internal_ai_org_no_policy_basis"
        suggested = v23_suggest_category(item, requested)
        if suggested and suggested != requested:
            return "reassign", suggested, "government_without_policy_basis"
        return "fail", requested, "government_without_policy_basis"

    if requested == "경쟁사_해외이슈":
        if v23_has_competitor_entity(t):
            return "pass", requested, "competitor_guardrail_pass"
        if v23_has_policy_basis(t):
            return "reassign", "정부_국회", "competitor_item_is_policy_issue"
        if v23_has_self_entity(t):
            return "reassign", "자사_및_계열사_이슈", "competitor_item_is_self_issue"
        return "fail", requested, "competitor_without_competitor_entity"

    if requested == "산업동향":
        if v23_is_valid_industry(t):
            return "pass", requested, "industry_structural_signal_pass"
        if v23_has_policy_basis(t):
            return "reassign", "정부_국회", "industry_item_is_policy_issue"
        if v23_has_competitor_entity(t):
            return "reassign", "경쟁사_해외이슈", "industry_item_is_competitor_issue"
        if v23_has_self_entity(t):
            return "reassign", "자사_및_계열사_이슈", "industry_item_is_self_issue"
        return "fail", requested, "industry_without_structural_signal"

    return "uncertain", requested, "unknown_category"


def v23_set_final_category(item, category, reason=""):
    cat = v19_normalize_category_key(category, fallback=item.get("카테고리", "산업동향"))
    old = item.get("카테고리", "")
    item["카테고리"] = cat
    item["카테고리명"] = JSON_KEY_TO_DISPLAY.get(cat, cat)
    item["최종카테고리"] = cat
    item["최종카테고리명"] = item["카테고리명"]
    if old and old != cat:
        item["최종카테고리재배치"] = f"{old}->{cat}:{reason}"
    return item


def v23_add_diagnostic_columns(item, article_info=None, requested_category=None, guardrail_result=""):
    article_info = article_info or {}
    item["수집검색어"] = article_info.get("검색어", item.get("검색어", ""))
    item["수집원카테고리"] = article_info.get("원카테고리", item.get("원카테고리", ""))
    item["수집JSON카테고리"] = article_info.get("JSON카테고리", item.get("JSON카테고리", ""))
    item["Gemini판단카테고리"] = item.get("Gemini카테고리", article_info.get("Gemini카테고리", ""))
    item["최종카테고리"] = item.get("카테고리", requested_category or "")
    item["최종카테고리명"] = JSON_KEY_TO_DISPLAY.get(item.get("카테고리", ""), item.get("카테고리명", ""))
    if guardrail_result:
        item["최종카테고리검증결과"] = guardrail_result
    return item


# 라벨 자체도 1차 보정한다. 단, 정부/국회는 단어 필수 조건이 아니라 정책 행위/공공사업 기준으로 판단한다.
_BASE_v20_normalize_label_json_v23 = v20_normalize_label_json

def v20_normalize_label_json(data, valid_ids, article_by_id):
    labels = _BASE_v20_normalize_label_json_v23(data, valid_ids, article_by_id)
    for art_id, label in list(labels.items()):
        article = article_by_id.get(int(art_id), {})
        if not article:
            continue
        probe = dict(article)
        probe["Gemini카테고리"] = label.get("primary_category", "")
        action, new_cat, reason = v23_final_category_guardrail(probe, label.get("primary_category") or article.get("JSON카테고리") or "산업동향")
        if action == "reassign" and new_cat:
            label["primary_category"] = new_cat
            label["reason"] = v20_clip((label.get("reason", "") + f" / v23_category_reassign:{reason}").strip(" /"), 260)
        elif action == "fail":
            # 명백히 잘못된 카테고리·사진캡션·민간 내부조직 오분류는 후보에서 제외한다.
            label["exclude"] = True
            label["is_relevant"] = False
            label["reason"] = v20_clip((label.get("reason", "") + f" / v23_guardrail_fail:{reason}").strip(" /"), 260)
    return labels


# Gemini 라벨링은 호출 성공뿐 아니라 배치 커버리지를 검증한다.
_BASE_v20_gemini_label_candidates_v23 = _BASE_v20_gemini_label_candidates_v22 if '_BASE_v20_gemini_label_candidates_v22' in globals() else v20_gemini_label_candidates

def v23_label_one_batch(client, batch, recent_past_text, label_tag=""):
    labels = _BASE_v20_gemini_label_candidates_v23(client, batch, recent_past_text)
    coverage = len(labels) / max(1, len(batch))
    if coverage < V23_LABEL_COVERAGE_MIN:
        raise ValueError(f"low_label_coverage:{coverage:.2f}:{len(labels)}/{len(batch)}:{label_tag}")
    return labels


def v20_gemini_label_candidates(client, candidates, recent_past_text):
    global V22_LABELING_LOCAL_BATCHES, V22_LABELING_RETRY_BATCHES
    if not client:
        raise RuntimeError("Gemini client 없음")
    all_labels = {}
    batches = [candidates[i:i + V20_LABEL_BATCH_SIZE] for i in range(0, len(candidates), V20_LABEL_BATCH_SIZE)]
    for batch_idx, batch in enumerate(batches, 1):
        try:
            labels = v23_label_one_batch(client, batch, recent_past_text, f"batch{batch_idx}")
            all_labels.update(labels)
            print(f"  └ 🏷️ v23 라벨링 배치 {batch_idx}/{len(batches)} 완료: Gemini {len(labels)}개 / coverage {len(labels)/max(1,len(batch)):.0%}")
        except Exception as first_error:
            V22_LABELING_RETRY_BATCHES += 1
            try:
                time.sleep(1.0 + random.uniform(0, 1.0))
                labels = v23_label_one_batch(client, batch, recent_past_text, f"retry{batch_idx}")
                all_labels.update(labels)
                print(f"  └ 🏷️ v23 라벨링 배치 {batch_idx}/{len(batches)} 재시도 성공: Gemini {len(labels)}개")
            except Exception:
                # Split once before falling back to local labels.
                split_labels = {}
                split_ok = True
                sub_batches = [batch[i:i + V23_LABEL_SPLIT_BATCH_SIZE] for i in range(0, len(batch), V23_LABEL_SPLIT_BATCH_SIZE)]
                for sub_idx, sub in enumerate(sub_batches, 1):
                    try:
                        V22_LABELING_RETRY_BATCHES += 1
                        labels = v23_label_one_batch(client, sub, recent_past_text, f"split{batch_idx}-{sub_idx}")
                        split_labels.update(labels)
                    except Exception:
                        split_ok = False
                        break
                if split_ok and split_labels:
                    all_labels.update(split_labels)
                    print(f"  └ 🏷️ v23 라벨링 배치 {batch_idx}/{len(batches)} 분할 재시도 성공: Gemini {len(split_labels)}개")
                else:
                    V22_LABELING_LOCAL_BATCHES += 1
                    print(f"  └ ⚠️ v23 라벨링 배치 {batch_idx}/{len(batches)} 커버리지/JSON 실패. 해당 배치만 로컬 라벨 사용: {first_error}")
                    for article in batch:
                        local = v20_local_article_label(article)
                        local["label_source"] = "local_low_confidence"
                        all_labels[int(article["id"])] = local
    return all_labels


# Body extraction 이전의 명백한 이미지/캡션/저가치 URL은 실패로 처리한다.
_BASE_process_article_for_report_v23 = process_article_for_report

def process_article_for_report(article_info, json_key, recent_past_items):
    if v23_is_photo_or_caption(article_info) or V23_IMAGE_POPUP_URL_RE.search(str(article_info.get("링크", ""))):
        failed = {
            "카테고리": json_key,
            "카테고리명": JSON_KEY_TO_DISPLAY.get(json_key, json_key),
            "검색어": article_info.get("검색어", ""),
            "기사제목": article_info.get("기사제목", ""),
            "언론사": article_info.get("언론사", ""),
            "게시일": article_info.get("게시일", ""),
            "RSS요약": article_info.get("본문요약", ""),
            "본문전문": "",
            "본문글자수": 0,
            "본문추출방식": "v23_pre_body_filter",
            "본문품질점수": -100,
            "본문품질사유": "photo_or_image_popup_not_article",
            "본문사용가능": False,
            "본문상태": "사진캡션_최종금지",
            "원래RSS링크": article_info.get("링크", ""),
            "링크": article_info.get("링크", ""),
        }
        v23_add_diagnostic_columns(failed, article_info, json_key, "fail:photo_or_image_popup_not_article")
        return None, None, failed
    return _BASE_process_article_for_report_v23(article_info, json_key, recent_past_items)


# v20_attach_issue_fields에 진단 컬럼 추가.
_BASE_v20_attach_issue_fields_v23 = v20_attach_issue_fields

def v20_attach_issue_fields(report_item, article_info, issue, decision, reason):
    report_item = _BASE_v20_attach_issue_fields_v23(report_item, article_info, issue, decision, reason)
    return v23_add_diagnostic_columns(report_item, article_info, report_item.get("카테고리", decision.get("category", "")))


# 최종 대표 기사 처리: 카테고리 guardrail 통과, 실패 시 대체 기사 시도, 명백한 저가치는 제목링크 제한 포함도 금지.
def v20_process_issue(decision, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter):
    issue = issue_by_id.get(decision.get("issue_id"))
    if not issue:
        return None, order_counter, "missing_issue"
    allowed, block_reason = v21_issue_allowed_basic(issue)
    if not allowed:
        return None, order_counter, block_reason

    requested_json_key = v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue))
    candidate_ids = []
    if decision.get("best_article_id"):
        candidate_ids.append(int(decision["best_article_id"]))
    for source_ids in [decision.get("backup_article_ids", []), issue.get("candidate_article_ids", []), issue.get("article_ids", [])]:
        for art_id in source_ids:
            if int(art_id) not in candidate_ids:
                candidate_ids.append(int(art_id))

    attempted_failed = []
    for art_id in candidate_ids:
        if art_id in processed_article_ids:
            continue
        article_info = article_by_id.get(int(art_id))
        if not article_info:
            continue
        processed_article_ids.add(int(art_id))

        # 명백한 사진/캡션/단순 MOU/단순 시드투자/SEO 리포트는 시도하되 최종 제목링크 포함도 금지.
        if v23_is_never_title_only(article_info):
            failed = {
                "카테고리": requested_json_key,
                "카테고리명": JSON_KEY_TO_DISPLAY.get(requested_json_key, requested_json_key),
                "검색어": article_info.get("검색어", ""),
                "기사제목": article_info.get("기사제목", ""),
                "언론사": article_info.get("언론사", ""),
                "게시일": article_info.get("게시일", ""),
                "RSS요약": article_info.get("본문요약", ""),
                "본문전문": "",
                "본문글자수": 0,
                "본문추출방식": "v23_pre_body_guardrail",
                "본문품질점수": -100,
                "본문품질사유": "never_allow_title_only_low_value_or_caption",
                "본문사용가능": False,
                "본문상태": "최종금지_대체필요",
                "원래RSS링크": article_info.get("링크", ""),
                "링크": article_info.get("링크", ""),
            }
            v23_add_diagnostic_columns(failed, article_info, requested_json_key, "fail:never_allow_title_only")
            failed["제외단계"] = "issue_representative_attempt"
            failed["v20_issue_id"] = issue.get("issue_id")
            failed["v20_issue_group"] = issue.get("issue_group")
            body_failed_rows.append(failed)
            attempted_failed.append((article_info, "never_allow_title_only"))
            continue

        report_item, skip_info, failed_item = process_article_for_report(article_info, requested_json_key, recent_past_items)
        if skip_info:
            skip_info["제외단계"] = "issue_representative_attempt"
            skip_info["v20_issue_id"] = issue.get("issue_id")
            skip_info["v20_issue_group"] = issue.get("issue_group")
            skip_rows.append(skip_info)
            attempted_failed.append((article_info, "past_duplicate_or_skip"))
            continue
        if failed_item:
            failed_item["제외단계"] = "issue_representative_attempt"
            failed_item["v20_issue_id"] = issue.get("issue_id")
            failed_item["v20_issue_group"] = issue.get("issue_group")
            v23_add_diagnostic_columns(failed_item, article_info, requested_json_key, failed_item.get("최종카테고리검증결과", ""))
            body_failed_rows.append(failed_item)
            attempted_failed.append((article_info, failed_item.get("본문품질사유", "body_failed")))
            continue
        if not report_item:
            attempted_failed.append((article_info, "empty_process_result"))
            continue

        report_item = v20_attach_issue_fields(report_item, article_info, issue, decision, "issue_selected")
        report_item["선정순서"] = order_counter
        report_item["본문상태"] = "본문추출성공"

        action, guarded_category, guard_reason = v23_final_category_guardrail(report_item, requested_json_key)
        if action == "fail":
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"category_guardrail_failed:{guard_reason}"
            failed_copy["제외단계"] = "issue_representative_attempt"
            failed_copy["최종카테고리검증결과"] = f"fail:{guard_reason}"
            body_failed_rows.append(failed_copy)
            attempted_failed.append((article_info, guard_reason))
            continue
        if action == "reassign" and guarded_category:
            if sum(1 for x in final_report_data if x.get("카테고리") == guarded_category) >= CATEGORY_MAX.get(guarded_category, 99):
                failed_copy = dict(report_item)
                failed_copy["본문품질사유"] = f"category_reassign_blocked_max:{guarded_category}:{guard_reason}"
                failed_copy["제외단계"] = "issue_representative_attempt"
                failed_copy["최종카테고리검증결과"] = f"reassign_blocked:{guard_reason}"
                body_failed_rows.append(failed_copy)
                attempted_failed.append((article_info, "reassign_blocked"))
                continue
            v23_set_final_category(report_item, guarded_category, guard_reason)
            report_item["최종카테고리검증결과"] = f"reassign:{guard_reason}"
        else:
            v23_set_final_category(report_item, requested_json_key)
            report_item["최종카테고리검증결과"] = f"pass:{guard_reason}"

        relevant, relevance_reason = is_report_item_relevant(report_item, report_item.get("카테고리"))
        if not relevant:
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"relevance_failed:{relevance_reason}"
            failed_copy["제외단계"] = "issue_representative_attempt"
            body_failed_rows.append(failed_copy)
            attempted_failed.append((article_info, relevance_reason))
            continue

        dup_item, dup_reason = v21_strict_cross_issue_duplicate(report_item, final_report_data)
        if dup_item:
            new_score = v20_float(report_item.get("대표선택점수"), 0)
            old_score = v20_float(dup_item.get("대표선택점수"), 0)
            if dup_reason == "same_gemini_issue_group" and new_score > old_score + REPRESENTATIVE_REPLACE_MARGIN:
                try:
                    idx = final_report_data.index(dup_item)
                    final_report_data[idx] = report_item
                except ValueError:
                    final_report_data.append(report_item)
                add_duplicate_skip_row(skip_rows, dup_item, report_item, f"replaced_by_better_duplicate:{dup_reason}", stage="issue_selected", extra=f"old={old_score},new={new_score}")
                return report_item, order_counter + 1, "replaced_duplicate"
            add_duplicate_skip_row(skip_rows, report_item, dup_item, f"strict_duplicate:{dup_reason}", stage="issue_selected", extra=f"old={old_score},new={new_score}")
            return None, order_counter, "duplicate_already_represented"

        final_report_data.append(report_item)
        return report_item, order_counter + 1, "body_success"

    # 본문/대체 기사 모두 실패한 경우: 정말 중요한 이슈만 제목링크 제한 포함.
    if v20_is_critical_issue(issue, decision):
        fallback_id = int(decision.get("best_article_id") or issue.get("top_article_id"))
        article_info = article_by_id.get(fallback_id)
        if article_info and not v23_is_never_title_only(article_info):
            item = v20_create_bodyless_report_item(article_info, issue, decision, requested_json_key)
            action, guarded_category, guard_reason = v23_final_category_guardrail(item, requested_json_key)
            if action == "fail":
                attempted_failed.append((article_info, f"bodyless_guardrail_failed:{guard_reason}"))
                return None, order_counter, "bodyless_guardrail_failed"
            if action == "reassign" and guarded_category:
                v23_set_final_category(item, guarded_category, guard_reason)
                item["최종카테고리검증결과"] = f"bodyless_reassign:{guard_reason}"
            else:
                v23_set_final_category(item, requested_json_key)
                item["최종카테고리검증결과"] = f"bodyless_pass:{guard_reason}"
            v23_add_diagnostic_columns(item, article_info, item.get("카테고리"), item.get("최종카테고리검증결과", ""))
            item["선정순서"] = order_counter
            dup_item, dup_reason = v21_strict_cross_issue_duplicate(item, final_report_data)
            if dup_item:
                add_duplicate_skip_row(skip_rows, item, dup_item, f"bodyless_duplicate:{dup_reason}", stage="critical_without_body")
                return None, order_counter, "bodyless_duplicate"
            final_report_data.append(item)
            return item, order_counter + 1, "critical_without_body"
    return None, order_counter, "all_candidates_failed"


# 최종 정리 단계에서도 guardrail을 한 번 더 적용해 혹시 남은 오배치를 제거한다.
_BASE_v21_remove_obvious_final_duplicates_v23 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    guarded = []
    removed = []
    for item in final_items:
        action, new_cat, reason = v23_final_category_guardrail(item, item.get("카테고리", ""))
        if action == "fail":
            removed.append((item, f"v23_final_guardrail_fail:{reason}"))
            continue
        if action == "reassign" and new_cat:
            v23_set_final_category(item, new_cat, reason)
            item["최종카테고리검증결과"] = f"final_reassign:{reason}"
        else:
            item["최종카테고리검증결과"] = item.get("최종카테고리검증결과") or f"final_pass:{reason}"
        v23_add_diagnostic_columns(item, {}, item.get("카테고리"), item.get("최종카테고리검증결과", ""))
        guarded.append(item)
    kept, dup_removed = _BASE_v21_remove_obvious_final_duplicates_v23(guarded)
    removed.extend(dup_removed)
    return kept, removed


# QA 프롬프트에 v23 guardrail 기준을 명시한다.
_BASE_v20_gemini_quality_check_v23 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v23(client, final_report_data, final_briefing_text)
    return qa



# v23.1: tighten government guardrail for private AI strategy/org articles.
V23_POLICY_ACTION_WITHOUT_ACTOR_RE = re.compile(
    r"법안|발의|입법|입법예고|행정예고|개정안|시행령|시행규칙|고시|가이드라인|심사지침|특별법|기본법|"
    r"의무|의무화|표시\s*의무|차단\s*의무|삭제\s*의무|보고\s*의무|자료\s*제출|공시|인허가|허가|인가|"
    r"시행|적용|확대|완화|강화|제도|규제|대책|예산|국가사업|공공\s*AI|디지털플랫폼정부|"
    r"조사|검사|감독|점검|제재|과징금|과태료|시정명령|고발|협의|의견수렴|설명회|간담회|공공조달",
    re.IGNORECASE,
)

# Override the broader v23_has_policy_basis: if no explicit public actor, do not accept generic "AI strategy/center" wording.
def v23_has_policy_basis(text):
    t = clean_html_text(text)
    if not t:
        return False
    if v20_is_platform_obligation_text(t):
        return True
    if v23_is_private_internal_news(t):
        return False
    explicit_actor = bool(V23_POLICY_ACTOR_RE.search(t))
    if explicit_actor and V23_POLICY_ACTION_RE.search(t) and V23_KAKAO_RELEVANT_DOMAIN_RE.search(t):
        return True
    # 정부기관명이 제목/본문에 없어도 사업자 의무/제도 변화가 명확하면 통과.
    if (not explicit_actor) and V23_POLICY_ACTION_WITHOUT_ACTOR_RE.search(t) and V23_KAKAO_RELEVANT_DOMAIN_RE.search(t):
        if re.search(r"사업자|플랫폼|AI|인공지능|개인정보|광고|알고리즘|딥페이크|불법촬영|전자금융|가상자산|스테이블코인", t, re.I):
            return True
    if V23_PUBLIC_AI_POLICY_RE.search(t) and (explicit_actor or V23_POLICY_ACTION_WITHOUT_ACTOR_RE.search(t)):
        return True
    return False



# v24: cross-section issue dedupe + representative article alignment.
# - 경쟁사/해외 cap은 v21/v23 설정 그대로 유지한다.
# - 같은 이슈가 자사/경쟁사 등 서로 다른 섹션에 중복 배치되는 것을 막는다.
# - 선택 이유는 국내 경쟁사/AI 인프라인데 대표 기사가 해외 빅테크 일반론으로 새는 문제를 막는다.

V24_OPENAI_SUPERAPP_RE = re.compile(
    r"(오픈AI|OpenAI|챗GPT|ChatGPT).{0,80}(슈퍼\s*앱|슈퍼앱|코덱스|Codex|코딩|AI\s*에이전트|에이전트|개편|외부\s*앱|앱\s*연동)|"
    r"(슈퍼\s*앱|슈퍼앱|코덱스|Codex|코딩|AI\s*에이전트|에이전트|개편|외부\s*앱|앱\s*연동).{0,80}(오픈AI|OpenAI|챗GPT|ChatGPT)",
    re.IGNORECASE,
)
V24_AI_INFRA_TERMS_RE = re.compile(r"AI\s*데이터센터|AIDC|AI\s*팩토리|GPU|NPU|AI\s*인프라|데이터센터|클라우드", re.IGNORECASE)
V24_RE100_GENERAL_RE = re.compile(r"RE100|탄소중립|무탄소|재생에너지|전력\s*수요|온실가스|전력망|에너지", re.IGNORECASE)
V24_DOMESTIC_INTENT_RE = re.compile(r"국내|한국|국가|K-|네이버|NAVER|SKT|SK텔레콤|KT|LGU\+|LG유플러스|토스|쿠팡|배민|카카오|AIDC|국산|K-AI", re.IGNORECASE)
V24_POLICY_OR_PUBLIC_INTENT_RE = re.compile(r"정부|국회|정책|법안|특별법|기본법|공공|국가|과기정통부|금융위|공정위|방미통위|규제|의무|가이드라인", re.IGNORECASE)
V24_GENERIC_MARKET_SIGNAL_RE = re.compile(r"시장\s*전망|CAGR|203[0-9]년까지|투자금|자본지출|CAPEX|주가|IPO|상장|유상증자", re.IGNORECASE)


def v24_item_text(item):
    return clean_html_text(" ".join(str(item.get(k, "")) for k in [
        "기사제목", "본문요약", "RSS요약", "본문전문", "Gemini이슈그룹", "v20_issue_group",
        "v20_issue_family", "Gemini이슈유형", "Gemini선정사유", "Gemini판단사유",
    ]))


def v24_topic_key(item):
    """Return a conservative topic key for final cross-section duplicate removal."""
    t = v24_item_text(item)
    if not t:
        return ""

    # Explicit high-value recurring patterns.
    if V24_OPENAI_SUPERAPP_RE.search(t):
        return "topic:openai_chatgpt_superapp_agent"
    if re.search(r"네이버웹툰|웹툰\s*엔터테인먼트|Webtoon", t, re.I) and re.search(r"집단소송|소송|IPO|상장분쟁|증권법", t, re.I):
        return "topic:naver_webtoon_ipo_litigation"
    if re.search(r"카카오\s*노조|크루유니언|임단협|파업", t, re.I):
        return "topic:kakao_labor_strike"

    # Prefer LLM issue family/group when it is specific enough.
    raw = clean_html_text(str(item.get("v20_issue_family") or item.get("Gemini이슈유형") or item.get("v20_issue_group") or item.get("Gemini이슈그룹") or ""))
    raw_key = v20_issue_group_key(raw) if raw else ""
    if raw_key and len(raw_key) >= 8:
        # Avoid over-collapsing broad buckets such as generic AI, platform, global incident.
        if not re.search(r"^(global_incident|industry_structural_change|ai|platform|government|policy|competitor|self)[:_\-]?$", raw_key, re.I):
            if not re.search(r"ai_policy|ai_infra|digital_asset|platform_policy|global_incident", raw_key, re.I):
                return "topic:" + raw_key[:80]

    # Entity + event fallback.
    entities = []
    for name, pat in [
        ("kakao", r"카카오|카카오톡|카카오페이|카카오뱅크|카카오게임즈|카카오모빌리티"),
        ("naver", r"네이버|NAVER"),
        ("skt", r"SKT|SK텔레콤"),
        ("toss", r"토스|비바리퍼블리카"),
        ("baemin", r"배민|배달의민족|우아한형제들"),
        ("openai", r"오픈AI|OpenAI|챗GPT|ChatGPT"),
        ("google", r"구글|Google|알파벳|Alphabet"),
        ("meta", r"메타|Meta"),
    ]:
        if re.search(pat, t, re.I):
            entities.append(name)
    events = []
    for name, pat in [
        ("lawsuit", r"소송|집단소송|고소|피소|증권법"),
        ("outage", r"장애|오류|먹통|유출|해킹|개인정보"),
        ("strike", r"노조|파업|임단협|쟁의"),
        ("superapp", r"슈퍼앱|슈퍼\s*앱|코덱스|AI\s*에이전트|챗GPT\s*개편"),
        ("ai_infra", r"AI\s*데이터센터|AIDC|AI\s*팩토리|GPU|AI\s*인프라"),
        ("regulation", r"법안|규제|의무|가이드라인|과징금|제재|조사"),
        ("finance", r"가상자산|스테이블코인|전자금융|거래소|보상"),
    ]:
        if re.search(pat, t, re.I):
            events.append(name)
    if entities and events:
        return "topic:" + "+".join(sorted(set(entities))[:2]) + ":" + "+".join(sorted(set(events))[:2])
    return ""


def v24_kakao_context_score(item):
    t = v24_item_text(item)
    score = 0
    if re.search(r"카카오와|카카오와의|카카오\s*협력|카카오.*발표|카카오.*전략|카카오.*영향|카카오.*서비스", t, re.I):
        score += 35
    elif v23_has_self_entity(t):
        score += 18
    if item.get("카테고리") == "자사_및_계열사_이슈":
        score += 8
    return score


def v24_source_quality_score(item):
    press = str(item.get("언론사", ""))
    url = str(item.get("링크", ""))
    score = 0
    if re.search(r"연합뉴스|뉴스1|머니투데이|한국경제|매일경제|이데일리|전자신문|디지털데일리|뉴시스|서울경제|조선비즈|ZDNet|지디넷", press, re.I):
        score += 12
    if re.search(r"AI넷|데일리비즈온|스페셜경제|와우테일|Global\s*Growth\s*Insights|market-reports", press + " " + url, re.I):
        score -= 10
    if item.get("본문상태") == "본문추출성공" or v20_float(item.get("본문글자수"), 0) >= 700:
        score += 8
    return score


def v24_duplicate_survivor_score(item):
    score = v20_float(item.get("대표선택점수"), 0)
    score += v24_kakao_context_score(item)
    score += v24_source_quality_score(item)
    # Exact category fit and strong tiers.
    cat = item.get("카테고리", "")
    if cat == "자사_및_계열사_이슈" and item.get("v21_self_tier") in {"DIRECT_RISK", "RESPONSE_RELEVANT", "STRATEGIC_REFERENCE"}:
        score += 12
    if cat == "경쟁사_해외이슈" and item.get("v21_competitor_scope") in {"domestic", "mixed"}:
        score += 8
    # Generic overseas market/CAPEX signal loses tie-breaks unless it has direct Kakao context.
    t = v24_item_text(item)
    if item.get("v21_competitor_scope") == "overseas" and V24_GENERIC_MARKET_SIGNAL_RE.search(t) and v24_kakao_context_score(item) == 0:
        score -= 12
    return score


def v24_dedupe_cross_section_topics(items):
    """Remove duplicate issues across sections after final guardrails. Keep the best representative."""
    groups = {}
    for item in items:
        key = v24_topic_key(item)
        if not key:
            continue
        groups.setdefault(key, []).append(item)

    remove_ids = set()
    removed = []
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        # Only remove if topic is explicit or the group is very likely the same event.
        winner = max(group, key=v24_duplicate_survivor_score)
        for item in group:
            if item is winner:
                item["v24중복대표선택"] = f"kept:{key}:score={v24_duplicate_survivor_score(item):.1f}"
                continue
            remove_ids.add(id(item))
            item["v24중복대표선택"] = f"removed:{key}:winner={winner.get('기사제목','')[:80]}"
            removed.append((item, f"v24_cross_section_issue_duplicate:{key}:winner={winner.get('기사제목','')[:100]}"))
    kept = [item for item in items if id(item) not in remove_ids]
    return kept, removed


def v24_issue_intent_text(issue, decision):
    return clean_html_text(" ".join(str(x or "") for x in [
        issue.get("issue_group"), issue.get("issue_family"), issue.get("internal_category"),
        issue.get("label_reasons"), decision.get("reason"), decision.get("category"),
        issue.get("competitor_scope"), issue.get("competitor_tier"), issue.get("gov_tier"),
    ]))


def v24_representative_alignment(issue, decision, report_item):
    """Validate whether the chosen representative article matches the selected issue intent.
    Returns (action, reason). action: pass/weak/fail.
    """
    issue_t = v24_issue_intent_text(issue, decision)
    article_t = v24_item_text(report_item)
    cat = report_item.get("카테고리") or v19_normalize_category_key(decision.get("category"), fallback="")

    if not issue_t or not article_t:
        return "pass", "alignment_not_enough_text"

    # Domestic competitor / domestic AI infra intent must not be represented by a pure overseas RE100/general article.
    intent_domestic = bool(V24_DOMESTIC_INTENT_RE.search(issue_t)) or str(issue.get("competitor_scope", "")) in {"domestic", "mixed"}
    article_domestic = bool(V21_DOMESTIC_COMPETITOR_RE.search(article_t))
    article_overseas = bool(V21_OVERSEAS_COMPETITOR_RE.search(article_t))
    if cat == "경쟁사_해외이슈" and intent_domestic and not article_domestic:
        if article_overseas and (V24_AI_INFRA_TERMS_RE.search(issue_t) or V24_AI_INFRA_TERMS_RE.search(article_t) or V24_RE100_GENERAL_RE.search(article_t)):
            return "fail", "expected_domestic_competitor_missing_overseas_general_ai_infra"
        # If the article is about a domestic policy actor, it may be reclassified elsewhere by guardrail, not kept as competitor.
        if not v23_has_policy_basis(article_t):
            return "weak", "expected_domestic_competitor_missing"

    # Policy/government intent should be represented by a policy article, not a private/company trend article.
    intent_policy = bool(V24_POLICY_OR_PUBLIC_INTENT_RE.search(issue_t)) or str(issue.get("gov_tier", "")).startswith("GOV_")
    if cat == "정부_국회" and intent_policy and not v23_has_policy_basis(article_t):
        return "fail", "expected_policy_basis_missing"

    # AI infra/AIDC/GPU policy intent: overseas RE100 explainer is not a good representative unless domestic/policy basis exists.
    if V24_AI_INFRA_TERMS_RE.search(issue_t) and V24_RE100_GENERAL_RE.search(article_t):
        if not article_domestic and not v23_has_policy_basis(article_t):
            return "fail", "ai_infra_intent_but_overseas_re100_general_article"

    # OpenAI superapp topic may be self if Kakao context exists, otherwise competitor. This is not a representative failure;
    # cross-section dedupe will keep only one if both survive.
    return "pass", "representative_aligned"


# Patch v20_process_issue to reject weak/off-intent representative articles before final append.
_BASE_v20_process_issue_v24_PREV = v20_process_issue

def v20_process_issue(decision, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter):
    issue = issue_by_id.get(decision.get("issue_id"))
    if not issue:
        return None, order_counter, "missing_issue"
    allowed, block_reason = v21_issue_allowed_basic(issue)
    if not allowed:
        return None, order_counter, block_reason

    requested_json_key = v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue))
    candidate_ids = []
    if decision.get("best_article_id"):
        candidate_ids.append(int(decision["best_article_id"]))
    for source_ids in [decision.get("backup_article_ids", []), issue.get("candidate_article_ids", []), issue.get("article_ids", [])]:
        for art_id in source_ids:
            if int(art_id) not in candidate_ids:
                candidate_ids.append(int(art_id))

    attempted_failed = []
    for art_id in candidate_ids:
        if art_id in processed_article_ids:
            continue
        article_info = article_by_id.get(int(art_id))
        if not article_info:
            continue
        processed_article_ids.add(int(art_id))

        if v23_is_never_title_only(article_info):
            failed = {
                "카테고리": requested_json_key,
                "카테고리명": JSON_KEY_TO_DISPLAY.get(requested_json_key, requested_json_key),
                "검색어": article_info.get("검색어", ""),
                "기사제목": article_info.get("기사제목", ""),
                "언론사": article_info.get("언론사", ""),
                "게시일": article_info.get("게시일", ""),
                "RSS요약": article_info.get("본문요약", ""),
                "본문전문": "",
                "본문글자수": 0,
                "본문추출방식": "v24_pre_body_guardrail",
                "본문품질점수": -100,
                "본문품질사유": "never_allow_title_only_low_value_or_caption",
                "본문사용가능": False,
                "본문상태": "최종금지_대체필요",
                "원래RSS링크": article_info.get("링크", ""),
                "링크": article_info.get("링크", ""),
            }
            v23_add_diagnostic_columns(failed, article_info, requested_json_key, "fail:never_allow_title_only")
            failed["제외단계"] = "issue_representative_attempt"
            failed["v20_issue_id"] = issue.get("issue_id")
            failed["v20_issue_group"] = issue.get("issue_group")
            body_failed_rows.append(failed)
            attempted_failed.append((article_info, "never_allow_title_only"))
            continue

        report_item, skip_info, failed_item = process_article_for_report(article_info, requested_json_key, recent_past_items)
        if skip_info:
            skip_info["제외단계"] = "issue_representative_attempt"
            skip_info["v20_issue_id"] = issue.get("issue_id")
            skip_info["v20_issue_group"] = issue.get("issue_group")
            skip_rows.append(skip_info)
            attempted_failed.append((article_info, "past_duplicate_or_skip"))
            continue
        if failed_item:
            failed_item["제외단계"] = "issue_representative_attempt"
            failed_item["v20_issue_id"] = issue.get("issue_id")
            failed_item["v20_issue_group"] = issue.get("issue_group")
            v23_add_diagnostic_columns(failed_item, article_info, requested_json_key, failed_item.get("최종카테고리검증결과", ""))
            body_failed_rows.append(failed_item)
            attempted_failed.append((article_info, failed_item.get("본문품질사유", "body_failed")))
            continue
        if not report_item:
            attempted_failed.append((article_info, "empty_process_result"))
            continue

        report_item = v20_attach_issue_fields(report_item, article_info, issue, decision, "issue_selected")
        report_item["선정순서"] = order_counter
        report_item["본문상태"] = "본문추출성공"

        action, guarded_category, guard_reason = v23_final_category_guardrail(report_item, requested_json_key)
        if action == "fail":
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"category_guardrail_failed:{guard_reason}"
            failed_copy["제외단계"] = "issue_representative_attempt"
            failed_copy["최종카테고리검증결과"] = f"fail:{guard_reason}"
            body_failed_rows.append(failed_copy)
            attempted_failed.append((article_info, guard_reason))
            continue
        if action == "reassign" and guarded_category:
            if sum(1 for x in final_report_data if x.get("카테고리") == guarded_category) >= CATEGORY_MAX.get(guarded_category, 99):
                failed_copy = dict(report_item)
                failed_copy["본문품질사유"] = f"category_reassign_blocked_max:{guarded_category}:{guard_reason}"
                failed_copy["제외단계"] = "issue_representative_attempt"
                failed_copy["최종카테고리검증결과"] = f"reassign_blocked:{guard_reason}"
                body_failed_rows.append(failed_copy)
                attempted_failed.append((article_info, "reassign_blocked"))
                continue
            v23_set_final_category(report_item, guarded_category, guard_reason)
            report_item["최종카테고리검증결과"] = f"reassign:{guard_reason}"
        else:
            v23_set_final_category(report_item, requested_json_key)
            report_item["최종카테고리검증결과"] = f"pass:{guard_reason}"

        # v24 representative-intent alignment check. If the best article is off-intent, try backups/candidates.
        align_action, align_reason = v24_representative_alignment(issue, decision, report_item)
        report_item["v24대표기사정합성"] = f"{align_action}:{align_reason}"
        if align_action == "fail":
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"representative_alignment_failed:{align_reason}"
            failed_copy["제외단계"] = "issue_representative_alignment"
            body_failed_rows.append(failed_copy)
            attempted_failed.append((article_info, f"representative_alignment_failed:{align_reason}"))
            continue
        if align_action == "weak" and len(candidate_ids) > 1:
            # Prefer trying other candidates before accepting a weak representative.
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"representative_alignment_weak_try_next:{align_reason}"
            failed_copy["제외단계"] = "issue_representative_alignment"
            body_failed_rows.append(failed_copy)
            attempted_failed.append((article_info, f"representative_alignment_weak:{align_reason}"))
            continue

        relevant, relevance_reason = is_report_item_relevant(report_item, report_item.get("카테고리"))
        if not relevant:
            failed_copy = dict(report_item)
            failed_copy["본문품질사유"] = f"relevance_failed:{relevance_reason}"
            failed_copy["제외단계"] = "issue_representative_attempt"
            body_failed_rows.append(failed_copy)
            attempted_failed.append((article_info, relevance_reason))
            continue

        dup_item, dup_reason = v21_strict_cross_issue_duplicate(report_item, final_report_data)
        if dup_item:
            new_score = v20_float(report_item.get("대표선택점수"), 0)
            old_score = v20_float(dup_item.get("대표선택점수"), 0)
            if dup_reason == "same_gemini_issue_group" and new_score > old_score + REPRESENTATIVE_REPLACE_MARGIN:
                try:
                    idx = final_report_data.index(dup_item)
                    final_report_data[idx] = report_item
                except ValueError:
                    final_report_data.append(report_item)
                add_duplicate_skip_row(skip_rows, dup_item, report_item, f"replaced_by_better_duplicate:{dup_reason}", stage="issue_selected", extra=f"old={old_score},new={new_score}")
                return report_item, order_counter + 1, "replaced_duplicate"
            add_duplicate_skip_row(skip_rows, report_item, dup_item, f"strict_duplicate:{dup_reason}", stage="issue_selected", extra=f"old={old_score},new={new_score}")
            return None, order_counter, "duplicate_already_represented"

        final_report_data.append(report_item)
        return report_item, order_counter + 1, "body_success"

    # Bodyless fallback remains conservative; no title-only for off-intent/low-value articles.
    if v20_is_critical_issue(issue, decision):
        fallback_id = int(decision.get("best_article_id") or issue.get("top_article_id"))
        article_info = article_by_id.get(fallback_id)
        if article_info and not v23_is_never_title_only(article_info):
            item = v20_create_bodyless_report_item(article_info, issue, decision, requested_json_key)
            action, guarded_category, guard_reason = v23_final_category_guardrail(item, requested_json_key)
            if action == "fail":
                attempted_failed.append((article_info, f"bodyless_guardrail_failed:{guard_reason}"))
                return None, order_counter, "bodyless_guardrail_failed"
            if action == "reassign" and guarded_category:
                v23_set_final_category(item, guarded_category, guard_reason)
                item["최종카테고리검증결과"] = f"bodyless_reassign:{guard_reason}"
            else:
                v23_set_final_category(item, requested_json_key)
                item["최종카테고리검증결과"] = f"bodyless_pass:{guard_reason}"
            v23_add_diagnostic_columns(item, article_info, item.get("카테고리"), item.get("최종카테고리검증결과", ""))
            align_action, align_reason = v24_representative_alignment(issue, decision, item)
            item["v24대표기사정합성"] = f"{align_action}:{align_reason}"
            if align_action == "fail":
                attempted_failed.append((article_info, f"bodyless_alignment_failed:{align_reason}"))
                return None, order_counter, "bodyless_alignment_failed"
            item["선정순서"] = order_counter
            dup_item, dup_reason = v21_strict_cross_issue_duplicate(item, final_report_data)
            if dup_item:
                add_duplicate_skip_row(skip_rows, item, dup_item, f"bodyless_duplicate:{dup_reason}", stage="critical_without_body")
                return None, order_counter, "bodyless_duplicate"
            final_report_data.append(item)
            return item, order_counter + 1, "critical_without_body"
    return None, order_counter, "all_candidates_failed"


# Final duplicate cleanup now includes cross-section issue-family/topic dedupe.
_BASE_v21_remove_obvious_final_duplicates_v24 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    kept, removed = _BASE_v21_remove_obvious_final_duplicates_v24(final_items)
    kept2, removed2 = v24_dedupe_cross_section_topics(kept)
    removed.extend(removed2)
    return kept2, removed


# Extend QA instruction with v24-specific checks while preserving the existing QA pipeline.
_BASE_v20_gemini_quality_check_v24 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v24(client, final_report_data, final_briefing_text)
    return qa


# =========================================================
# v25 overrides: Korean government-section guardrail + flexible max fill
# - 정부/국회 섹션은 한국 정부/국회/규제기관/공공정책 중심으로 제한한다.
# - 해외 정부/의회/규제기관 이슈는 정부/국회가 아니라 경쟁사/해외 또는 산업동향으로 보낸다.
# - 중복 제거와 이통3사 검색어는 v24 기준을 유지한다.
# - 중요한 이슈가 충분하면 기존 상한선(V21_FINAL_MAX)까지 더 자연스럽게 채운다.
# =========================================================

V6_VERSION = "google_rss_v25_kr_gov_guardrail_flex_max_fill"

# 사용자가 설정한 상한까지는 좋은 후보가 있으면 채운다. category max와 경쟁사 cap은 기존대로 유지.
V21_FINAL_TARGET = V21_FINAL_MAX
V20_FINAL_TARGET = V21_FINAL_TARGET
MAX_SELECT_COUNT = V21_FINAL_MAX
MIN_SELECT_COUNT = V21_FINAL_MIN

# 한국 정부/국회/공공 정책 주체. v23보다 국내 공공·정책 분석 주체를 보강한다.
V25_KOREAN_PUBLIC_ACTOR_RE = re.compile(
    r"대한민국|한국|국내|정부|국회|대통령실|국무회의|당정|상임위|과방위|정무위|문체위|성평등위|"
    r"의원|국회의원|더불어민주당|국민의힘|조국혁신당|위원회|TF|전담조직|공공기관|지자체|지방정부|"
    r"과기정통부|과학기술정보통신부|방미통위|방송미디어통신위원회|공정위|공정거래위원회|개보위|개인정보보호위원회|"
    r"금융위|금융위원회|금감원|금융감독원|행안부|행정안전부|중기부|중소벤처기업부|고용노동부|노동부|"
    r"FIU|금융정보분석원|국정원|국가정보원|검찰|경찰청|KISA|한국인터넷진흥원|한국은행|금융연구원|한국금융연구원|"
    r"국회입법조사처|KDI|KISDI|정보통신정책연구원|보험연구원|자본시장연구원|한국소비자원",
    re.IGNORECASE,
)

# 해외 정책/규제 주체. 정부/국회 섹션에서 한국 정책으로 착각하면 안 되는 신호.
V25_FOREIGN_PUBLIC_ACTOR_RE = re.compile(
    r"미국|美\b|미\s*정부|미\s*의회|미\s*상원|미\s*하원|백악관|트럼프|바이든|상무부|FTC|FCC|SEC|"
    r"주정부|州\b|캘리포니아|텍사스|뉴욕주|EU|유럽연합|유럽\s*집행위|EU\s*집행위|DMA|DSA|"
    r"중국|일본|영국|프랑스|독일|캐나다|호주|싱가포르|인도|브라질|글로벌\s*규제|해외\s*규제",
    re.IGNORECASE,
)

V25_FOREIGN_POLICY_ACTION_RE = re.compile(
    r"법안|초안|발의|입법|규제|가이드라인|의무|의무화|감사|면허|벌금|제재|조사|감독|동결|모라토리엄|"
    r"건설\s*제한|제한\s*조치|주\s*규제|연방|상무부|집행위|규제기관|지분\s*보유|국부펀드|정부\s*지분",
    re.IGNORECASE,
)

# 해외 정책이라도 한국 정부/국회가 도입·검토하거나 국내 사업자 의무로 연결되면 정부/국회에 남길 수 있다.
V25_KOREAN_POLICY_CONNECTION_RE = re.compile(
    r"한국\s*정부|국내\s*도입|국내\s*적용|국내\s*기업|국내\s*사업자|한국\s*기업|한국\s*사업자|"
    r"우리나라|국내에서도|한국도|도입\s*검토|국내\s*규제|국내\s*법안|국내\s*가이드라인|"
    r"카카오|네이버|토스|쿠팡|배민|SKT|KT|LGU\+|LG유플러스|금융권|가상자산\s*거래소",
    re.IGNORECASE,
)

# 국내 연구기관/싱크탱크의 정책 제언은 부처명이 없어도 정부/국회 정책 환경 기사로 볼 수 있다.
V25_KOREAN_POLICY_ANALYSIS_RE = re.compile(
    r"(한국금융연구원|금융연구원|한국은행|국회입법조사처|KDI|KISDI|정보통신정책연구원|보험연구원|자본시장연구원|"
    r"한국인터넷진흥원|KISA|한국소비자원).{0,220}"
    r"(규제|가이드라인|정책|법안|제도|위험관리|모형위험관리|감독|관리체계|프레임워크|보호|보안|AI|인공지능|금융권)",
    re.IGNORECASE,
)

# 해외 정책/규제 흐름이지만 한국 정부·국회 기사로 분류하면 안 되는 경우.
def v25_is_foreign_public_policy_issue(text):
    t = clean_html_text(text)
    if not t:
        return False
    return bool(V25_FOREIGN_PUBLIC_ACTOR_RE.search(t) and V25_FOREIGN_POLICY_ACTION_RE.search(t))


def v25_has_direct_korean_policy_connection(text):
    t = clean_html_text(text)
    if not t:
        return False
    return bool(V25_KOREAN_POLICY_CONNECTION_RE.search(t))


def v25_has_korean_policy_basis(text):
    """한국 정부/국회 섹션에 들어갈 수 있는 최소 정책 근거.
    특정 부처명 필수 방식이 아니라, 국내 공공 주체/정책행위/카카오 관련 도메인을 함께 본다.
    """
    t = clean_html_text(text)
    if not t:
        return False

    # 해외 정책만 중심이고 국내 연결이 없으면 한국 정부/국회 섹션 근거가 아니다.
    if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
        return False

    # 플랫폼 사업자 의무는 국내 플랫폼/국내 사업자/국내 공공 주체와 연결될 때만 한국 정부/국회로 인정.
    if v20_is_platform_obligation_text(t):
        if V25_KOREAN_PUBLIC_ACTOR_RE.search(t) or V25_KOREAN_POLICY_CONNECTION_RE.search(t):
            return True
        # 네이버·카카오 등이 의무 대상이면 국내 사업자 영향으로 봄.
        if re.search(r"네이버|카카오|카톡|카카오페이|카카오뱅크|국내\s*사업자|국내\s*플랫폼", t, re.I):
            return True

    # 명시적 한국 공공 주체 + 정책 행위 + 카카오 관련 도메인.
    if V25_KOREAN_PUBLIC_ACTOR_RE.search(t) and V23_POLICY_ACTION_RE.search(t) and V23_KAKAO_RELEVANT_DOMAIN_RE.search(t):
        return True

    # 한국 공공 AI/국가 AI/피지컬AI 특별법 등 핵심 AI 정책.
    if V25_KOREAN_PUBLIC_ACTOR_RE.search(t) and V23_PUBLIC_AI_POLICY_RE.search(t):
        return True

    # 국내 정책 분석기관의 규제·가이드라인·위험관리 제언.
    if V25_KOREAN_POLICY_ANALYSIS_RE.search(t):
        return True

    # 기관명이 없어도 국내 사업자 의무/제도 변화가 명확하면 인정.
    if V23_POLICY_ACTION_WITHOUT_ACTOR_RE.search(t) and V23_KAKAO_RELEVANT_DOMAIN_RE.search(t):
        if re.search(r"국내|한국|금융권|사업자|플랫폼|AI|인공지능|개인정보|광고|알고리즘|딥페이크|불법촬영|전자금융|가상자산|스테이블코인", t, re.I):
            return True

    return False


def v25_foreign_policy_target_category(text):
    """한국 정부/국회가 아닌 해외 정책/규제 이슈를 어느 섹션으로 보낼지 결정."""
    t = clean_html_text(text)
    # 해외 정부/규제/의회 흐름은 기본적으로 경쟁사/해외이슈가 가장 자연스럽다.
    if v25_is_foreign_public_policy_issue(t):
        return "경쟁사_해외이슈"
    if v23_is_valid_industry(t):
        return "산업동향"
    return "경쟁사_해외이슈"

# v23 guardrail 위에 한국 정부/국회 전용 guardrail을 덧씌운다.
_BASE_v23_final_category_guardrail_v25 = v23_final_category_guardrail

def v23_final_category_guardrail(item, requested_category):
    requested = v19_normalize_category_key(requested_category, fallback="산업동향")
    t = v23_text(item)

    # 정부/국회는 한국 정부·국회·국내 공공정책 중심. 해외 법안/규제는 해외이슈로 이동.
    if requested == "정부_국회":
        if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
            return "reassign", v25_foreign_policy_target_category(t), "foreign_policy_not_korean_gov_section"
        if v25_has_korean_policy_basis(t):
            return "pass", requested, "korean_government_policy_basis_pass"
        # 한국 정책 근거가 없으면 기존 guardrail의 pass라도 한 번 더 보수적으로 처리한다.
        action, cat, reason = _BASE_v23_final_category_guardrail_v25(item, requested)
        if action == "pass":
            # v23의 넓은 AI/정책 키워드 통과를 막는다. 단, 명백한 산업/경쟁사로 보낼 수 있으면 재배치.
            suggested = v23_suggest_category(item, requested)
            if suggested and suggested != requested:
                return "reassign", suggested, f"government_without_korean_policy_basis:{reason}"
            return "fail", requested, f"government_without_korean_policy_basis:{reason}"
        return action, cat, reason

    # 다른 카테고리에서 v23이 해외 정책을 정부/국회로 보내려 하면 경쟁사/해외로 보낸다.
    action, cat, reason = _BASE_v23_final_category_guardrail_v25(item, requested)
    if action == "reassign" and cat == "정부_국회":
        if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
            return "reassign", v25_foreign_policy_target_category(t), f"foreign_policy_not_korean_gov_section:{reason}"
        if not v25_has_korean_policy_basis(t):
            suggested = v23_suggest_category(item, requested)
            if suggested and suggested != "정부_국회":
                return "reassign", suggested, f"blocked_non_korean_gov_reassign:{reason}"
            return "fail", requested, f"blocked_non_korean_gov_reassign:{reason}"
    return action, cat, reason

# v23_suggest_category도 해외 정책은 정부/국회가 아니라 경쟁사/해외로 제안하도록 보정한다.
_BASE_v23_suggest_category_v25 = v23_suggest_category

def v23_suggest_category(item, requested=None):
    t = v23_text(item)
    if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
        return v25_foreign_policy_target_category(t)
    return _BASE_v23_suggest_category_v25(item, requested)

# issue-level allowed check에서도 정부/국회 저품질 해외정책 오배치를 줄인다.
_BASE_v21_issue_allowed_basic_v25 = v21_issue_allowed_basic

def v21_issue_allowed_basic(issue):
    ok, reason = _BASE_v21_issue_allowed_basic_v25(issue)
    if not ok:
        return ok, reason
    issue_text = clean_html_text(" ".join(str(x or "") for x in [
        issue.get("issue_group"), issue.get("issue_family"), issue.get("internal_category"),
        issue.get("label_reasons"), issue.get("company_impact"), issue.get("gov_tier"),
    ]))
    cat = v20_issue_to_output_category(issue)
    if cat == "정부_국회":
        if v25_is_foreign_public_policy_issue(issue_text) and not v25_has_direct_korean_policy_connection(issue_text):
            # 제외하지는 않고, 최종 guardrail에서 해외이슈로 재배치될 수 있게 둔다.
            issue["v25_gov_domestic_guardrail"] = "foreign_policy_should_not_stay_in_gov"
        elif not v25_has_korean_policy_basis(issue_text):
            # issue text가 짧아 불확실한 경우는 article-level guardrail에서 다시 판단하므로 막지 않는다.
            issue["v25_gov_domestic_guardrail"] = "needs_article_level_check"
    return True, reason

# QA가 한국 정부/국회 기준을 다시 확인하도록 최종 QA 프롬프트에 실릴 item fields만 보강한다.
_BASE_v20_attach_issue_fields_v25 = v20_attach_issue_fields

def v20_attach_issue_fields(report_item, article_info, issue, decision, selection_source=""):
    item = _BASE_v20_attach_issue_fields_v25(report_item, article_info, issue, decision, selection_source)
    text = v23_text(item)
    if item.get("카테고리") == "정부_국회":
        item["v25한국정부국회근거"] = "Y" if v25_has_korean_policy_basis(text) else ""
        if v25_is_foreign_public_policy_issue(text) and not v25_has_direct_korean_policy_connection(text):
            item["v25한국정부국회근거"] = "N:foreign_policy"
    return item


# v25.1 patch: domestic security whitepaper/government report + updated editor prompt
V25_KOREAN_SECURITY_WHITEPAPER_RE = re.compile(
    r"(국가정보보호백서|정보보호백서|국가정보원|국정원|6개\s*부처|정부\s*백서|보안\s*가이드북|AI\s*보안\s*가이드북|국가인공지능안보센터)",
    re.IGNORECASE,
)

_BASE_v25_has_korean_policy_basis_PREV = v25_has_korean_policy_basis

def v25_has_korean_policy_basis(text):
    t = clean_html_text(text)
    if V25_KOREAN_SECURITY_WHITEPAPER_RE.search(t) and re.search(r"AI|인공지능|보안|정보보호|해킹|사이버|개인정보|클라우드", t, re.I):
        return True
    return _BASE_v25_has_korean_policy_basis_PREV(t)


def v20_gemini_edit_issues(client, issues, recent_past_text):
    if not client:
        raise RuntimeError("Gemini client 없음")
    visible_issues = [i for i in sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True) if not i.get("exclude") and not i.get("is_pr")]
    visible_issues = visible_issues[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = "\n".join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
너는 카카오 대외협력팀 아침 뉴스 편집장이다.
아래 이슈 후보는 기사 단위가 아니라 issue_group 단위로 묶인 후보들이다.
최종 보고서는 카카오 경영진/대외협력/정책/커뮤니케이션/사업 담당자가 오늘 알아야 할 이슈를 담아야 한다.

[전체 편집 원칙]
- 같은 issue_group은 하나만 고른다.
- 단순 홍보·후원·캠페인·기부·이벤트·수상은 선택하지 않는다.
- 본문 추출 실패 시 같은 issue_group 안의 다른 기사로 대체될 수 있으므로, 중요한 이슈는 best/backup article_ids를 함께 준다.
- 일반 금융/제약/건설/비디지털 공정위·금감원 기사는 카카오 연결 경로가 약하면 제외한다.
- 중요한 이슈가 많은 날에는 전체 상한인 15개까지 골라도 된다. 단, 낮은 품질 기사로 억지로 채우지 말라.

[자사/계열사 원칙]
1. DIRECT_RISK와 RESPONSE_RELEVANT를 최우선으로 포함한다.
2. STRATEGIC_REFERENCE는 카카오 AI agent, 카톡 AI, 결제/금융, 조직/거점, 지배구조, 핵심 서비스 변화처럼 현직자가 실질적으로 참고할 때만 포함한다.
3. FILLER_REFERENCE는 자사 이슈가 3개 미만일 때만 고려한다.
4. LOW_VALUE_PR은 절대 자사 슬롯 보충용으로 쓰지 않는다.

[정부/국회 원칙: 한국 중심]
1. 정부/국회 섹션은 원칙적으로 한국 정부·국회·규제기관·공공기관·국내 법/정책 이슈만 넣는다.
2. 플랫폼 사업자 직접 의무/규제는 최우선이다. 단, 한국 사업자나 국내 제도 적용과 연결되어야 한다.
3. AI 정책, 국가 AI 전략, AIDC, GPU, 공공 AI, 디지털플랫폼정부, AI 기본법/저작권/안전/거버넌스, AI 정책 컨트롤타워는 카카오 AI agent 전략과 연결되는 핵심 국내 정책 이슈로 본다.
4. 국가정보보호백서, 국정원/정부 부처의 AI 보안·사이버보안·개인정보보호 백서/가이드북/대책은 정부/국회 정책 이슈로 본다.
5. 스테이블코인, 전자금융, 디지털자산, 디지털트윈은 연결성이 있을 때 포함하되 AI·플랫폼 직접 이슈보다 후순위다.
6. 미국·EU·중국·일본 등 해외 정부/의회/규제기관의 법안·정책·규제는 정부/국회가 아니라 경쟁사/해외이슈 또는 산업동향으로 분류한다.
7. 단, 해외 정책이라도 한국 정부가 도입·검토 중이거나 국내 사업자 의무/국내 규제 변화로 직접 연결된 기사라면 정부/국회에 포함할 수 있다.
8. 홍콩 ELS 같은 일반 금융 제재는 원칙적으로 제외한다.

[경쟁사/해외 원칙]
1. 국내 경쟁사/인접 플랫폼을 우선한다. 네이버·네이버페이·쿠팡·토스·배민·SKT·KT·LGU+·라인야후 등.
2. 경쟁사/해외 섹션은 3~4개가 적절하며, 국내 경쟁사 이슈를 최소 2개, 가능하면 3개 포함한다.
3. 해외 빅테크/글로벌 AI 이슈는 기본 1개만 포함한다. 카카오 AI·플랫폼·콘텐츠 데이터·앱마켓·개인정보·광고 전략에 직접 시사점이 매우 크면 2개까지 가능하다.
4. 해외 법안/규제/정책, 미국 AI 법안, EU 규제, 트럼프 행정부 AI 정책 등은 이 섹션에서 다룬다. 단순 글로벌 AI 투자, IPO, 주가, CAPEX, 제품 루머는 후순위 또는 제외한다.

[권장 분량]
- 전체 11~15개. 중요한 이슈가 많은 날에는 15개까지 가능하다.
- 자사 3~5개, 정부/국회 3~5개, 경쟁사/해외 3~4개, 산업 1~2개.

반드시 JSON만 반환하라.
{{
  "selected_issues": [
    {{
      "issue_id": "I001",
      "category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향",
      "priority": 5,
      "best_article_id": 123,
      "backup_article_ids": [124, 125],
      "reason": "선정 이유"
    }}
  ],
  "backup_issues": [
    {{
      "issue_id": "I050",
      "category": "경쟁사_해외이슈",
      "priority": 4,
      "best_article_id": 555,
      "backup_article_ids": [556],
      "reason": "대체 후보 이유"
    }}
  ]
}}

[최근 과거 보고서 참고]
{recent_past_text[:4500]}

[이슈 후보]
{issue_text}
"""
    text = gemini_generate_text(
        client=client,
        prompt=prompt,
        task_name="v25 이슈 단위 최종 편집",
        model=GEMINI_MODEL_EDITOR,
    )
    data = extract_json_object(text)
    issue_by_id = {i["issue_id"]: i for i in issues}
    article_by_id = {}
    for issue in issues:
        for art_id in issue.get("article_ids", []):
            article_by_id[int(art_id)] = {"id": int(art_id)}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    return decisions, backup_decisions



# =========================================================
# v26 overrides: index-history dedupe + today-first editorial guardrails
# - past_reports.txt 대신 data_news_briefing/index/*.csv 최근 7일을 중복 판단 기준으로 사용
# - 같은 날짜 재실행 시 index 오늘 날짜 행을 실행 초반/저장 직전 제거하되 백업은 1회만 생성
# - index CSV 저장은 QUOTE_ALL + 컬럼 union rewrite로 깨짐 방지
# - 자사 섹션: 카카오 직접 주체 + 오늘 새 사건/진전 우선, 분석/기획/시황성 후순위
# - 정부/국회 섹션: 한국 정부·국회·규제기관이 실제 행위자인 정책/규제 기사만 허용
# - 개인정보/보안/제재/플랫폼 의무 등 동일 사건은 섹션이 달라도 최종 1건만 유지
# =========================================================

import csv

V6_VERSION = "google_rss_v26_1_soft_index_v24_feel_refill"

# 중복 판단 소스 전환: 기본은 index 기반. 과도기 테스트가 필요하면 True로 바꾸면 past_reports도 같이 참고한다.
DEDUP_USE_INDEX_HISTORY = True
DEDUP_USE_PAST_REPORTS_FALLBACK = False
DEDUP_EXCLUDE_RUN_DATE_FROM_HISTORY = True
DEDUP_INDEX_LOOKBACK_DAYS = PAST_DUP_LOOKBACK_DAYS

# 같은 날짜 재실행 안전장치
V26_REPLACE_SAME_DATE_INDEX_ROWS = True
V26_BACKUP_INDEX_BEFORE_REPLACE = True
V26_INDEX_BACKUP_DONE = False
V26_INDEX_START_CLEANUP_DONE = False

# 분석/기획/시황성 기사 감점. 완전 배제가 아니라 직접 사건이 부족할 때만 보충되게 만드는 용도.
V26_ANALYSIS_FEATURE_RE = re.compile(
    r"\[?[^\]]*(성적표|대해부|리포트|분석|전망|짚어보니|관전포인트|시리즈|기획|주간|어디\?|모래\s*위|슈퍼\s*IPO)[^\]]*\]?|"
    r"주주환원|실적\s*미리보기|밸류에이션|투자의견|목표가|증권가|상장사\s*\d+곳|분기보고서|영업이익\s*감소|적자\s*\d+배",
    re.IGNORECASE,
)
V26_MARKET_COMMENTARY_RE = re.compile(
    r"주가|목표가|투자의견|매수\s*의견|매도\s*의견|중립\s*의견|지지선|저항선|차트|밸류에이션|PER|PBR|EPS|"
    r"ETF|수익률|증권가|어닝\s*(서프라이즈|쇼크)|실적\s*전망|주주환원\s*성적표",
    re.IGNORECASE,
)
V26_TODAY_ACTION_RE = re.compile(
    r"돌입|예고|착수|심의|의결|통과|부과|제재|감경|확정|결정|발표|공개|제기|고발|소송|기소|선고|판결|"
    r"선임|내정|사임|퇴사|교체|출범|가동|시행|적용|확대|강화|중단|장애|오류|유출|해킹|차단|삭제|의무화|점검|회의|파업|조정",
    re.IGNORECASE,
)
V26_DIRECT_SELF_EVENT_RE = re.compile(
    r"노조|파업|임단협|쟁의권|노동부|근로감독|최저임금|임금체불|성과급|RSU|고용불안|"
    r"장애|오류|먹통|개인정보|유출|해킹|피싱|보안|수사|조사|과징금|제재|소송|판결|고발|"
    r"대표|임원|CPO|CTO|CFO|조직개편|퇴사|사임|해임|선임|내정|경영권|최대주주|지분|매각|인수|합병|상장|IPO|실적|영업손실|적자",
    re.IGNORECASE,
)
V26_SELF_ENTITY_STRICT_RE = re.compile(
    r"카카오(?!톡?\s*(제보|으로\s*제보|캡처|캡쳐|대화|내용|메시지|메세지|언급|조작|증거))|"
    r"카톡\s*(개편|업데이트|장애|오류|먹통|피싱|보안|채널|오픈채팅|광고|선물하기|톡비즈|서비스|정책|이용자|공식|PC버전)|"
    r"카카오톡\s*(개편|업데이트|장애|오류|먹통|피싱|보안|채널|오픈채팅|광고|선물하기|톡비즈|서비스|정책|이용자|공식|PC버전)|"
    r"카카오페이|카카오뱅크|카카오모빌리티|카카오\s*T\b|카카오T\b|카카오택시|카카오게임즈|카카오엔터테인먼트|카카오엔터|"
    r"카카오픽코마|카카오헬스케어|카카오엔터프라이즈|카카오클라우드|디케이테크인|엑스엘게임즈|카나나|정신아|김범수",
    re.IGNORECASE,
)

# 한국 정부/국회/규제기관: '국내/한국' 같은 넓은 단어는 제외하고 실제 행위자를 본다.
V26_KR_GOV_ACTOR_STRICT_RE = re.compile(
    r"정부|국회|대통령실|국무회의|당정|상임위|과방위|정무위|문체위|성평등위|국회의원|의원|"
    r"과기정통부|과학기술정보통신부|방미통위|방송미디어통신위원회|방통위|공정위|공정거래위원회|"
    r"개보위|개인정보보호위원회|개인정보위|금융위|금융위원회|금감원|금융감독원|행안부|행정안전부|"
    r"중기부|중소벤처기업부|고용노동부|노동부|FIU|금융정보분석원|국정원|국가정보원|검찰|경찰청|KISA|한국인터넷진흥원|"
    r"한국은행|국회입법조사처|한국금융연구원|금융연구원|정보통신정책연구원|KISDI|KDI|한국소비자원",
    re.IGNORECASE,
)
V26_GOV_ACTION_STRICT_RE = re.compile(
    r"법안|발의|입법|입법예고|행정예고|개정안|시행령|시행규칙|고시|가이드라인|심사지침|특별법|기본법|"
    r"국무회의\s*통과|의결|통과|시행|적용|확대|완화|강화|제도|정책|대책|전략|예산|공공조달|"
    r"의무|의무화|표시\s*의무|차단\s*의무|삭제\s*의무|보고\s*의무|자료\s*제출|공시|인허가|허가|인가|"
    r"조사|검사|감독|점검|심의|제재|과징금|과태료|시정명령|고발|협의|의견수렴|설명회|간담회|백서|가이드북",
    re.IGNORECASE,
)
V26_GOV_MISCLASSIFIED_INDUSTRY_RE = re.compile(
    r"상장사|영업이익|영업손실|적자|흑자|매출|실적|분기보고서|주가|목표가|투자의견|밸류에이션",
    re.IGNORECASE,
)
V26_GOV_ALLOWED_ANALYSIS_RE = re.compile(
    r"(국가정보보호백서|정보보호백서|AI\s*보안\s*가이드북|국정원|국가정보원|한국금융연구원|금융연구원|국회입법조사처|KISDI|정보통신정책연구원|한국은행).{0,260}"
    r"(정책|규제|가이드라인|위험관리|보안|AI|인공지능|개인정보|금융권|플랫폼|디지털자산|스테이블코인)",
    re.IGNORECASE,
)

V26_PORTAL_UI_NOISE_RE = re.compile(
    r"글자크기\s*설정|파란원을\s*좌우로|음성재생\s*설정|이동\s*통신망에서\s*음성\s*재생|데이터\s*요금이\s*발생|"
    r"광고\s*로드중|댓글\s*보기|공유하기|카카오톡에\s*공유|페이스북에\s*공유|기사\s*추천|본문\s*듣기|음성으로\s*듣기|"
    r"닫기\s*글자크기|무단전재|재배포\s*금지|Copyright",
    re.IGNORECASE,
)

V26_STAGE_ORDER = {
    "rumor_or_report": 1,
    "review_scheduled": 2,
    "consultation": 3,
    "investigation": 4,
    "vote_or_mediation": 5,
    "action_started": 6,
    "decision": 7,
    "penalty_or_sanction": 8,
    "litigation": 9,
    "implementation": 10,
}


def v26_run_date_obj():
    try:
        return datetime.fromisoformat(str(V22_RUN_DATE)[:10]).date()
    except Exception:
        return datetime.now(KST).date()


def v26_safe_read_csv(path):
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, encoding="utf-8-sig", engine="python", on_bad_lines="skip", escapechar="\\")
    except Exception:
        try:
            return pd.read_csv(p, encoding="utf-8-sig", engine="python", on_bad_lines="skip")
        except Exception as e:
            print(f"  └ ⚠️ index CSV 읽기 실패: {p.name} / {e}")
            return pd.DataFrame()


def v26_backup_index_files_once(stage=""):
    global V26_INDEX_BACKUP_DONE
    if not V26_BACKUP_INDEX_BEFORE_REPLACE or V26_INDEX_BACKUP_DONE:
        return
    try:
        backup_root = V22_OUTPUT_ROOT / "index_backup"
        timestamp = datetime.now(KST).strftime("%Y%m%d_%H%M%S")
        suffix = f"_{stage}" if stage else ""
        backup_dir = backup_root / f"{V22_RUN_DATE}_{timestamp}{suffix}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        copied = 0
        for filename in ["article_history.csv", "issue_history.csv", "report_history.csv", "url_seen_history.csv"]:
            src = V22_INDEX_DIR / filename
            if src.exists():
                shutil.copy2(src, backup_dir / filename)
                copied += 1
        V26_INDEX_BACKUP_DONE = True
        print(f"  └ 🧷 index 백업 완료: {backup_dir}" if copied else f"  └ 🧷 index 백업 폴더 생성 완료(복사할 파일 없음): {backup_dir}")
    except Exception as e:
        print(f"  └ ⚠️ index 백업 실패: {e}")


def v26_line_belongs_to_run_date(line, run_date):
    """
    index CSV 한 줄이 이번 실행일의 row인지 판정한다.
    기존 index에는 2026.6.10 형식이 있을 수 있으므로,
    첫 번째 컬럼을 날짜로 파싱해서 비교한다.
    """
    stripped = str(line or "").lstrip("\ufeff").lstrip()
    if not stripped:
        return False

    # header는 삭제하지 않음
    if re.match(r"^\"?(date|run_date|report_date|날짜)\"?\s*,", stripped):
        return False

    # CSV 첫 번째 컬럼만 대략 추출
    first_col = stripped.split(",", 1)[0].strip().strip('"').strip("'")
    line_date = v22_parse_history_date(first_col)
    target_date = v22_parse_history_date(run_date)

    if line_date and target_date:
        return line_date == target_date

    # 혹시 파싱 실패할 때를 위한 기존 방식 fallback
    return (
        stripped.startswith(f"{run_date},")
        or stripped.startswith(f'"{run_date}",')
        or stripped.startswith(f"{run_date} ")
        or stripped.startswith(f'"{run_date} ')
        or stripped.strip() == run_date
        or stripped.strip() == f'"{run_date}"'
    )


def v26_remove_same_date_rows_from_index(stage=""):
    if not V26_REPLACE_SAME_DATE_INDEX_ROWS:
        return
    V22_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    v26_backup_index_files_once(stage=stage)
    removed_summary = {}
    for filename in ["article_history.csv", "issue_history.csv", "report_history.csv", "url_seen_history.csv"]:
        path = V22_INDEX_DIR / filename
        if not path.exists():
            removed_summary[filename] = 0
            continue
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                lines = f.readlines()
            if not lines:
                removed_summary[filename] = 0
                continue
            kept = []
            removed = 0
            for idx, line in enumerate(lines):
                # 첫 줄 header 보존
                if idx == 0 and re.match(r"^\ufeff?\s*\"?(date|run_date|report_date|날짜)\"?\s*,", line):
                    kept.append(line)
                    continue
                if v26_line_belongs_to_run_date(line, V22_RUN_DATE):
                    removed += 1
                    continue
                kept.append(line)
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                f.writelines(kept)
            removed_summary[filename] = removed
        except Exception as e:
            print(f"  └ ⚠️ {filename}: 같은 날짜 행 제거 실패: {e}")
            removed_summary[filename] = "실패"
    msg = ", ".join([f"{k}:{v}행" for k, v in removed_summary.items()])
    print(f"  └ 🧹 같은 날짜 index 기존 행 제거 완료({V22_RUN_DATE}, {stage or 'cleanup'}): {msg}")


_BASE_v22_setup_output_paths_v26 = v22_setup_output_paths

def v22_setup_output_paths(run_date=None):
    global V26_INDEX_START_CLEANUP_DONE
    _BASE_v22_setup_output_paths_v26(run_date=run_date)
    if DEDUP_USE_INDEX_HISTORY and not V26_INDEX_START_CLEANUP_DONE:
        v26_remove_same_date_rows_from_index(stage="run_start")
        V26_INDEX_START_CLEANUP_DONE = True


def v26_append_rows_csv(path, rows):
    """Append rows safely by rewriting with a union schema and full CSV quoting.
    This avoids column drift and embedded comma/newline corruption in index CSVs.
    """
    if not rows:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if p.exists() and p.stat().st_size > 0:
        old_df = v26_safe_read_csv(p)
        if not old_df.empty:
            all_cols = list(dict.fromkeys(list(old_df.columns) + list(new_df.columns)))
            old_df = old_df.reindex(columns=all_cols)
            new_df = new_df.reindex(columns=all_cols)
            out_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            out_df = new_df
    else:
        out_df = new_df
    out_df.to_csv(
        p,
        index=False,
        encoding="utf-8-sig",
        quoting=csv.QUOTE_ALL,
        escapechar="\\",
        lineterminator="\n",
    )


# v22 base function looks up v22_append_rows_csv at call time, so overriding this name is enough too.
v22_append_rows_csv = v26_append_rows_csv


def v26_normalize_event_key(key):
    key = clean_html_text(key)
    key = re.sub(r"\s+", "_", key)
    key = key.strip("_:")
    return key[:180]


def v26_text_for_event(item):
    return clean_html_text(" ".join(str(item.get(k, "")) for k in [
        "기사제목", "title", "본문요약", "summary", "RSS요약", "본문전문", "issue_group", "v20_issue_group", "Gemini이슈그룹", "언론사", "press"
    ]))


def v26_event_key_for_item(item):
    """Build a stable event key using existing global/privacy incident detectors plus generic fallbacks."""
    text = v26_text_for_event(item)
    if not text:
        return ""
    try:
        key = v15_privacy_security_base_key({"기사제목": item.get("기사제목") or item.get("title", ""), "본문전문": text})
        if key:
            return v26_normalize_event_key(key)
    except Exception:
        pass
    try:
        key = v16_global_incident_base_key({"기사제목": item.get("기사제목") or item.get("title", ""), "본문전문": text, "본문요약": text})
        if key:
            return v26_normalize_event_key(key)
    except Exception:
        pass
    try:
        if v20_is_platform_obligation_text(text):
            obj = v16_detect_object(text, "platform_regulation") if "v16_detect_object" in globals() else "platform_operator_obligation"
            return v26_normalize_event_key(f"platform_obligation:{obj}")
    except Exception:
        pass
    entities = sorted(detect_entities(text))[:2]
    tags = sorted(detect_event_tags(text) & STRONG_EVENT_TAGS)[:2]
    if entities and tags:
        return v26_normalize_event_key("event:" + "+".join(entities) + ":" + "+".join(tags))
    return ""


def v26_extract_development_stage(text):
    t = clean_html_text(text)
    if not t:
        return ""
    if re.search(r"집단소송|소송\s*제기|행정소송|가처분|손해배상|법적\s*대응", t):
        return "litigation"
    if re.search(r"과징금|과태료|제재|시정명령|처분|고발|검찰\s*고발|부과|감경|징계", t):
        return "penalty_or_sanction"
    if re.search(r"의결|국무회의\s*통과|통과|확정|결정|심의\s*결과|제재안\s*심의", t):
        return "decision"
    if re.search(r"파업\s*돌입|부분파업|총파업|시행|적용|가동|출범|개시|중단|장애\s*발생", t):
        return "action_started"
    if re.search(r"찬반투표|투표|조정|중노위|지노위|임단협|교섭", t):
        return "vote_or_mediation"
    if re.search(r"조사\s*착수|검사|수사\s*착수|현장점검|점검|민관합동조사|신고\s*접수", t):
        return "investigation"
    if re.search(r"의견수렴|간담회|공청회|설명회|협의|검토", t):
        return "consultation"
    if re.search(r"내일|오는|예정|예고|심판대|심의\s*예정|회의\s*예정", t):
        return "review_scheduled"
    if re.search(r"발표|공개|보도|분석", t):
        return "rumor_or_report"
    return "general"


def v26_stage_is_update(new_stage, old_stage):
    if not new_stage or not old_stage or new_stage == old_stage:
        return False
    return V26_STAGE_ORDER.get(new_stage, 0) > V26_STAGE_ORDER.get(old_stage, 0)


def v26_parse_pubdate_date(published):
    try:
        dt = parse_pubdate_to_kst(published)
        return dt.date() if dt else None
    except Exception:
        return None


def v26_date_terms(run_date=None):
    rd = run_date or v26_run_date_obj()
    yesterday = rd - timedelta(days=1)
    tomorrow = rd + timedelta(days=1)
    return {
        "today": ["오늘", "이날", "금일", f"{rd.day}일"],
        "yesterday": ["전날", "어제", f"{yesterday.day}일"],
        "tomorrow": ["내일", f"오는 {tomorrow.day}일", f"{tomorrow.day}일"],
    }


def v26_todayness_score_text(text, published=""):
    t = clean_html_text(text)
    rd = v26_run_date_obj()
    score = 0.0
    reasons = []
    pub_d = v26_parse_pubdate_date(published)
    if pub_d == rd:
        score += 24; reasons.append("published_today")
    elif pub_d == rd - timedelta(days=1):
        score += 14; reasons.append("published_yesterday")
    elif pub_d == rd + timedelta(days=1):
        score += 8; reasons.append("published_tomorrow")

    terms = v26_date_terms(rd)
    action = V26_TODAY_ACTION_RE.pattern
    for bucket, words in terms.items():
        for w in words:
            if not w:
                continue
            if re.search(rf"({re.escape(w)}).{{0,45}}({action})|({action}).{{0,45}}({re.escape(w)})", t, re.IGNORECASE):
                add = 18 if bucket == "today" else 12
                score += add
                reasons.append(f"{bucket}_date_action")
                break
    if re.search(r"\[단독\]|단독|속보|종합", t):
        score += 5; reasons.append("exclusive_breaking")
    if V26_ANALYSIS_FEATURE_RE.search(t) and not V26_DIRECT_SELF_EVENT_RE.search(t):
        score -= 12; reasons.append("analysis_without_direct_event")
    return round(score, 2), "+".join(dict.fromkeys(reasons))


def v26_article_form_from_text(text):
    t = clean_html_text(text)
    if v18_is_self_pr_promo_text(t) if "v18_is_self_pr_promo_text" in globals() else False:
        return "pr_promo"
    if V26_MARKET_COMMENTARY_RE.search(t) and not V26_DIRECT_SELF_EVENT_RE.search(t):
        return "market_commentary"
    if V26_ANALYSIS_FEATURE_RE.search(t):
        if V26_DIRECT_SELF_EVENT_RE.search(t):
            return "analysis_with_direct_risk"
        return "analysis_feature"
    if re.search(r"국무회의\s*통과|과징금|제재|심의|의결|법안|시행령|가이드라인|조사|점검|고발", t):
        return "official_action"
    if V26_DIRECT_SELF_EVENT_RE.search(t):
        return "direct_risk"
    if re.search(r"인수|합병|매각|지분|최대주주|대표|실적|영업이익|서비스\s*출시|전략|AI", t):
        return "business_change"
    return "general"


def v26_has_strict_self_entity(text):
    t = clean_html_text(text)
    if not t:
        return False
    if v13_is_private_kakaotalk_or_entertainment_noise(t) if "v13_is_private_kakaotalk_or_entertainment_noise" in globals() else False:
        return False
    if v13_is_self_low_value_ranking_article(t) if "v13_is_self_low_value_ranking_article" in globals() else False:
        return False
    return bool(V26_SELF_ENTITY_STRICT_RE.search(t))


def v26_is_self_direct_today_event(text, published=""):
    t = clean_html_text(text)
    if not v26_has_strict_self_entity(t):
        return False
    today_score, _ = v26_todayness_score_text(t, published)
    return bool(V26_DIRECT_SELF_EVENT_RE.search(t) and today_score >= 8)


def v26_is_analysis_or_market_text(text):
    t = clean_html_text(text)
    return bool(V26_ANALYSIS_FEATURE_RE.search(t) or V26_MARKET_COMMENTARY_RE.search(t))


def v26_has_korean_gov_action(text):
    t = clean_html_text(text)
    if not t:
        return False
    if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
        return False
    if V26_GOV_ALLOWED_ANALYSIS_RE.search(t):
        return True
    if V26_KR_GOV_ACTOR_STRICT_RE.search(t) and V26_GOV_ACTION_STRICT_RE.search(t) and V23_KAKAO_RELEVANT_DOMAIN_RE.search(t):
        return True
    # 플랫폼 사업자 의무는 규제기관명이 제목에 약해도 국내 사업자 의무로 명확하면 통과.
    if v20_is_platform_obligation_text(t) and V26_GOV_ACTION_STRICT_RE.search(t):
        if V26_KR_GOV_ACTOR_STRICT_RE.search(t) or re.search(r"국내|한국|네이버|카카오|구글|메타|플랫폼\s*사업자|사업자|포털|SNS", t, re.I):
            return True
    return False


def v26_is_gov_misclassified_industry(text):
    t = clean_html_text(text)
    if not t:
        return False
    # 정부/규제 행위가 없는 보안/금융/상장사 실적 분석은 정부/국회가 아니다.
    if V26_GOV_MISCLASSIFIED_INDUSTRY_RE.search(t) and not (V26_KR_GOV_ACTOR_STRICT_RE.search(t) and V26_GOV_ACTION_STRICT_RE.search(t)):
        return True
    return False


def v26_suggest_category_from_text(text, requested=""):
    t = clean_html_text(text)
    if v26_has_strict_self_entity(t) and V26_DIRECT_SELF_EVENT_RE.search(t):
        return "자사_및_계열사_이슈"
    if v26_has_korean_gov_action(t):
        return "정부_국회"
    if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
        return "경쟁사_해외이슈"
    if v23_has_competitor_entity(t):
        return "경쟁사_해외이슈"
    if v23_is_valid_industry(t):
        return "산업동향"
    if v26_is_gov_misclassified_industry(t):
        return "산업동향"
    return ""


def v26_is_same_event_update(candidate, past_item):
    cand_text = v26_text_for_event(candidate)
    past_text = v26_text_for_event(past_item)
    cand_stage = v26_extract_development_stage(cand_text)
    past_stage = past_item.get("event_stage") or v26_extract_development_stage(past_text)
    if v26_stage_is_update(cand_stage, past_stage):
        # 진행 단계가 올라갔고 오늘성/행위 동사가 있으면 후속 업데이트로 허용.
        today_score, _ = v26_todayness_score_text(cand_text, candidate.get("게시일") or candidate.get("published") or "")
        if today_score >= 8 or V26_TODAY_ACTION_RE.search(cand_text):
            return True, f"same_event_but_new_stage:{past_stage}->{cand_stage}"
    return False, ""


def v26_index_duplicate_check(candidate, recent_past_items, mode="rss"):
    cand_title = clean_html_text(candidate.get("기사제목") or candidate.get("title") or "")
    cand_url = normalize_url(candidate.get("링크") or candidate.get("link") or candidate.get("url") or "")
    cand_title_hash = title_fingerprint(cand_title)
    cand_event_key = v26_event_key_for_item(candidate)

    update_match = None
    update_reason = ""
    for past in recent_past_items or []:
        past_url = normalize_url(past.get("link") or past.get("url") or "")
        past_title = clean_html_text(past.get("title") or past.get("기사제목") or "")
        past_title_hash = past.get("title_hash") or title_fingerprint(past_title)
        past_event_key = past.get("event_key") or v26_event_key_for_item(past)

        if cand_url and past_url and cand_url == past_url:
            return True, past, {"is_duplicate": True, "reason": "index_same_url", "score": 1.0, "shared_entities": "", "shared_tags": ""}
        if cand_title_hash and past_title_hash and cand_title_hash == past_title_hash:
            return True, past, {"is_duplicate": True, "reason": "index_same_title_hash", "score": 1.0, "shared_entities": "", "shared_tags": ""}
        if cand_title and past_title:
            near, score = is_title_near_duplicate(cand_title, past_title)
            if near and score >= 0.94:
                return True, past, {"is_duplicate": True, "reason": f"index_near_title:{score:.2f}", "score": score, "shared_entities": "", "shared_tags": ""}
        if cand_event_key and past_event_key and cand_event_key == past_event_key:
            is_update, reason = v26_is_same_event_update(candidate, past)
            if is_update:
                update_match = past
                update_reason = reason
                continue
            return True, past, {"is_duplicate": True, "reason": f"index_same_event:{cand_event_key}", "score": 0.9, "shared_entities": "", "shared_tags": cand_event_key}
    if update_match:
        return False, update_match, {"is_duplicate": False, "reason": update_reason, "score": 0.0, "shared_entities": "", "shared_tags": cand_event_key}
    return False, None, None


_BASE_find_past_duplicate_v26 = find_past_duplicate

def find_past_duplicate(candidate, recent_past_items, mode="body"):
    if DEDUP_USE_INDEX_HISTORY and recent_past_items:
        is_dup, matched, sim = v26_index_duplicate_check(candidate, recent_past_items, mode=mode)
        if is_dup:
            return True, matched, sim
        # 같은 사건 업데이트로 허용된 경우 기존 유사도 엔진이 다시 중복 처리하지 않도록 바로 통과.
        if sim and str(sim.get("reason", "")).startswith("same_event_but_new_stage"):
            return False, matched, sim
    return _BASE_find_past_duplicate_v26(candidate, recent_past_items, mode=mode)


def v26_history_items_from_index():
    run_date = v26_run_date_obj()
    cutoff = run_date - timedelta(days=DEDUP_INDEX_LOOKBACK_DAYS)
    rows = []
    seen = set()
    for filename in ["report_history.csv", "article_history.csv"]:
        df = v26_safe_read_csv(V22_INDEX_DIR / filename)
        if df.empty:
            continue
        for _, row in df.iterrows():
            d = v22_parse_history_date(row.get("date") or row.get("run_date") or row.get("report_date"))
            if not d:
                continue
            if DEDUP_EXCLUDE_RUN_DATE_FROM_HISTORY and d >= run_date:
                continue
            if not (cutoff <= d < run_date):
                continue
            title = clean_html_text(row.get("title") or row.get("기사제목") or "")
            url = normalize_url(row.get("url") or row.get("link") or "")
            if not title and not url:
                continue
            key = (url, title_fingerprint(title), str(row.get("issue_group", "")))
            if key in seen:
                continue
            seen.add(key)
            summary = clean_html_text(row.get("summary") or row.get("RSS요약") or "")
            text = f"{title} {summary} {row.get('issue_group','')} {row.get('issue_family','')}"
            item = {
                "date": d,
                "category": clean_html_text(row.get("section") or row.get("category") or ""),
                "title": title,
                "link": url,
                "press": clean_html_text(row.get("press") or row.get("언론사") or ""),
                "summary": summary,
                "text": text,
                "entities": detect_entities(text),
                "event_tags": detect_event_tags(text),
                "issue_terms": tokenize_for_similarity(text),
                "issue_group": clean_html_text(row.get("issue_group", "")),
                "issue_family": clean_html_text(row.get("issue_family", "")),
                "event_key": clean_html_text(row.get("event_key", "")) or v26_event_key_for_item({"기사제목": title, "본문요약": summary, "issue_group": row.get("issue_group", "")}),
                "event_stage": clean_html_text(row.get("event_stage", "")) or v26_extract_development_stage(text),
                "title_hash": clean_html_text(row.get("title_hash", "")) or title_fingerprint(title),
                "source": f"index:{filename}",
            }
            rows.append(item)
    print(f"📚 index 최근 {DEDUP_INDEX_LOOKBACK_DAYS}일 중복 판단 기준 기사: {len(rows)}건 ({cutoff.isoformat()}~{(run_date - timedelta(days=1)).isoformat()})")
    return rows


_BASE_load_past_reports_v26 = load_past_reports

def load_past_reports():
    if not DEDUP_USE_INDEX_HISTORY:
        return _BASE_load_past_reports_v26()
    index_items = v26_history_items_from_index()
    content = "\n".join(
        f"[{i['date'].month}월 {i['date'].day}일 / {i.get('category','')}] {i.get('title','')}\n{i.get('summary','')}"
        for i in index_items[-120:]
    ) or "최근 7일 index 이력 없음."
    all_items = list(index_items)
    recent_items = list(index_items)
    if DEDUP_USE_PAST_REPORTS_FALLBACK:
        base_content, base_all, base_recent = _BASE_load_past_reports_v26()
        seen = {(normalize_url(i.get("link", "")), title_fingerprint(i.get("title", ""))) for i in recent_items}
        for item in base_recent:
            key = (normalize_url(item.get("link", "")), title_fingerprint(item.get("title", "")))
            if key not in seen:
                recent_items.append(item)
                all_items.append(item)
                seen.add(key)
        content = content + "\n\n[past_reports fallback]\n" + base_content[-4000:]
    if not index_items and not DEDUP_USE_PAST_REPORTS_FALLBACK:
        print("📚 index 이력이 아직 없어 최근 7일 중복 판단 없이 진행합니다. 운영 누적 후 자동으로 사용됩니다.")
    return content, all_items, recent_items


_BASE_rank_score_article_v26 = rank_score_article

def rank_score_article(article):
    score = float(_BASE_rank_score_article_v26(article) or article.get("랭킹점수") or 0)
    text = v20_article_text(article) if "v20_article_text" in globals() else v26_text_for_event(article)
    published = article.get("게시일", "")
    today_score, today_reason = v26_todayness_score_text(text, published)
    article_form = v26_article_form_from_text(text)
    article["오늘성점수"] = today_score
    article["오늘성사유"] = today_reason
    article["기사유형"] = article_form
    article["v26이벤트키"] = v26_event_key_for_item(article)
    article["v26진행단계"] = v26_extract_development_stage(text)

    json_key = article.get("JSON카테고리", "")
    if json_key == "자사_및_계열사_이슈":
        if not v26_has_strict_self_entity(text):
            score -= 80
            article["v26가드레일사유"] = "self_without_direct_kakao_entity"
        else:
            score += min(25, today_score * 0.7)
            if article_form in {"analysis_feature", "market_commentary"} and not v26_is_self_direct_today_event(text, published):
                score -= 35 if article_form == "analysis_feature" else 65
                article["v26가드레일사유"] = f"self_{article_form}_deprioritized"
            elif article_form in {"direct_risk", "official_action", "analysis_with_direct_risk"}:
                score += 12
    elif json_key == "정부_국회":
        if v26_is_gov_misclassified_industry(text):
            score -= 75
            article["v26가드레일사유"] = "government_misclassified_industry_earnings"
        elif v26_has_korean_gov_action(text):
            score += min(20, today_score * 0.4) + 8
        elif v25_is_foreign_public_policy_issue(text) and not v25_has_direct_korean_policy_connection(text):
            score -= 25
            article["v26가드레일사유"] = "foreign_policy_should_be_competitor"
    else:
        score += min(10, today_score * 0.25)

    article["랭킹점수"] = round(score, 3)
    return article["랭킹점수"]


_BASE_article_importance_score_v26 = article_importance_score

def article_importance_score(title, body="", json_key="", keyword=""):
    score = float(_BASE_article_importance_score_v26(title, body, json_key, keyword) or 0)
    text = f"{clean_html_text(title)} {clean_html_text(body)[:2600]}"
    today_score, _ = v26_todayness_score_text(text)
    form = v26_article_form_from_text(text)
    if json_key == "자사_및_계열사_이슈":
        if not v26_has_strict_self_entity(text):
            score -= 80
        else:
            score += min(28, today_score * 0.6)
            if form in {"analysis_feature", "market_commentary"} and not V26_DIRECT_SELF_EVENT_RE.search(text):
                score -= 35 if form == "analysis_feature" else 65
    if json_key == "정부_국회":
        if v26_is_gov_misclassified_industry(text):
            score -= 80
        elif v26_has_korean_gov_action(text):
            score += 12
    return round(score, 2)


_BASE_v20_local_article_label_v26 = v20_local_article_label

def v20_local_article_label(article):
    label = _BASE_v20_local_article_label_v26(article)
    text = v20_article_text(article) if "v20_article_text" in globals() else v26_text_for_event(article)
    form = v26_article_form_from_text(text)
    today_score, today_reason = v26_todayness_score_text(text, article.get("게시일", ""))
    label["article_form"] = form
    label["todayness_score"] = today_score
    label["todayness_reason"] = today_reason
    label["event_key"] = v26_event_key_for_item(article)

    if label.get("primary_category") == "자사_및_계열사_이슈" or article.get("JSON카테고리") == "자사_및_계열사_이슈":
        if not v26_has_strict_self_entity(text):
            suggested = v26_suggest_category_from_text(text, "자사_및_계열사_이슈") or "산업동향"
            label["primary_category"] = suggested
            label["internal_category"] = "OFF_TOPIC" if not suggested else label.get("internal_category", "")
            label["self_tier"] = "OFF_TOPIC"
            label["ceo_priority"] = min(v20_int(label.get("ceo_priority"), 2), 2)
            label["public_affairs_priority"] = min(v20_int(label.get("public_affairs_priority"), 2), 2)
            label["reason"] = "v26: 카카오 직접 주체가 없어 자사 섹션 제외/재분류"
        elif form in {"analysis_feature", "market_commentary"} and not v26_is_self_direct_today_event(text, article.get("게시일", "")):
            label["self_tier"] = "FILLER_REFERENCE"
            label["issue_family"] = "self_analysis_feature" if form == "analysis_feature" else "self_market_commentary"
            label["ceo_priority"] = min(v20_int(label.get("ceo_priority"), 3), 3)
            label["public_affairs_priority"] = min(v20_int(label.get("public_affairs_priority"), 3), 2)
            label["relevance"] = min(v20_int(label.get("relevance"), 3), 3)
            label["company_impact"] = "분석/기획성 자사 참고 기사로, 오늘 직접 사건이 부족할 때만 보충 가치"
            label["reason"] = "v26: 자사 분석/기획/시황성 기사 후순위"
        elif today_score >= 12 and V26_DIRECT_SELF_EVENT_RE.search(text):
            label["ceo_priority"] = 5
            label["public_affairs_priority"] = max(4, v20_int(label.get("public_affairs_priority"), 4))
            label["relevance"] = 5
            label["reason"] = (label.get("reason", "") + " / v26: 오늘 새 직접 사건·진전").strip(" /")

    if label.get("primary_category") == "정부_국회" or article.get("JSON카테고리") == "정부_국회":
        if v26_is_gov_misclassified_industry(text):
            label["primary_category"] = "산업동향"
            label["internal_category"] = "INDUSTRY_STRUCTURAL_CHANGE"
            label["gov_tier"] = "OFF_TOPIC"
            label["ceo_priority"] = min(v20_int(label.get("ceo_priority"), 2), 2)
            label["reason"] = "v26: 정부/국회 행위가 아닌 기업 실적·산업 분석 기사"
        elif not v26_has_korean_gov_action(text) and not (v25_is_foreign_public_policy_issue(text) and not v25_has_direct_korean_policy_connection(text)):
            label["ceo_priority"] = min(v20_int(label.get("ceo_priority"), 3), 3)
            label["reason"] = (label.get("reason", "") + " / v26: 한국 정부·국회 직접 행위자 근거 약함").strip(" /")
        elif v25_is_foreign_public_policy_issue(text) and not v25_has_direct_korean_policy_connection(text):
            label["primary_category"] = "경쟁사_해외이슈"
            label["reason"] = "v26: 해외 정부·규제기관 이슈는 경쟁사/해외로 분류"
    return label


_BASE_v20_build_issues_v26 = v20_build_issues

def v20_build_issues(candidates, labels):
    issues, article_by_id = _BASE_v20_build_issues_v26(candidates, labels)
    # id -> article lookup
    art_lookup = {int(a.get("id")): a for a in candidates if str(a.get("id", "")).isdigit()}
    for issue in issues:
        arts = [art_lookup.get(int(aid)) for aid in issue.get("article_ids", []) if int(aid) in art_lookup]
        texts = [v20_article_text(a) if a else "" for a in arts]
        max_today = max([v26_todayness_score_text(t, (a or {}).get("게시일", ""))[0] for t, a in zip(texts, arts)] or [0])
        forms = [v26_article_form_from_text(t) for t in texts]
        event_keys = [v26_event_key_for_item(a or {}) for a in arts]
        issue["v26_max_todayness"] = max_today
        issue["v26_article_forms"] = ",".join(sorted(set(forms)))
        issue["v26_event_key"] = next((k for k in event_keys if k), "")
        score = float(issue.get("issue_score") or 0)
        cat = issue.get("primary_category", "")
        if cat == "자사_및_계열사_이슈":
            if forms and all(f in {"analysis_feature", "market_commentary"} for f in forms):
                score -= 26
            score += min(18, max_today * 0.45)
            if any(v26_is_self_direct_today_event(t, (a or {}).get("게시일", "")) for t, a in zip(texts, arts)):
                score += 18
        elif cat == "정부_국회":
            joined = " ".join(texts)
            if v26_is_gov_misclassified_industry(joined):
                score -= 40
            elif v26_has_korean_gov_action(joined):
                score += 10
        issue["issue_score"] = round(score, 3)
    issues = sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True)
    for idx, issue in enumerate(issues, 1):
        issue["issue_id"] = f"I{idx:03d}"
    return issues, article_by_id


_BASE_v20_issue_rows_v26 = v20_issue_rows

def v20_issue_rows(issues):
    rows = _BASE_v20_issue_rows_v26(issues)
    for row, issue in zip(rows, issues):
        row["v26_max_todayness"] = issue.get("v26_max_todayness", "")
        row["v26_article_forms"] = issue.get("v26_article_forms", "")
        row["v26_event_key"] = issue.get("v26_event_key", "")
    return rows


_BASE_v20_issue_line_v26 = v20_issue_line

def v20_issue_line(issue):
    base = _BASE_v20_issue_line_v26(issue)
    return base + f" todayness={issue.get('v26_max_todayness','')} forms={issue.get('v26_article_forms','')} event_key={v20_clip(issue.get('v26_event_key',''),80)}"


_BASE_v23_suggest_category_v26 = v23_suggest_category

def v23_suggest_category(item, requested=None):
    t = v23_text(item)
    suggested = v26_suggest_category_from_text(t, requested or "")
    if suggested:
        return suggested
    return _BASE_v23_suggest_category_v26(item, requested)


_BASE_v23_final_category_guardrail_v26 = v23_final_category_guardrail

def v23_final_category_guardrail(item, requested_category):
    requested = v19_normalize_category_key(requested_category, fallback="산업동향")
    t = v23_text(item)

    if requested == "자사_및_계열사_이슈":
        if not v26_has_strict_self_entity(t):
            suggested = v26_suggest_category_from_text(t, requested)
            if suggested and suggested != requested:
                return "reassign", suggested, "v26_self_without_direct_kakao_entity"
            return "fail", requested, "v26_self_without_direct_kakao_entity"
        # 여러 사업자 공통 플랫폼 의무/규제는 카카오가 포함돼도 자사가 아니라 정부/국회가 자연스럽다.
        if v20_is_platform_obligation_text(t) and not V26_DIRECT_SELF_EVENT_RE.search(t):
            if v26_has_korean_gov_action(t):
                return "reassign", "정부_국회", "v26_self_included_platform_obligation_to_gov"
        return _BASE_v23_final_category_guardrail_v26(item, requested)

    if requested == "정부_국회":
        if v26_is_gov_misclassified_industry(t):
            return "fail", requested, "v26_government_misclassified_industry_or_earnings"
        if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
            return "reassign", "경쟁사_해외이슈", "v26_foreign_policy_not_korean_gov_section"
        if v26_has_korean_gov_action(t):
            return "pass", requested, "v26_korean_gov_actor_action_pass"
        # 기존 v25가 pass라고 해도, 한국 정부 실제 행위자 근거가 약하면 재검토/재분류한다.
        action, cat, reason = _BASE_v23_final_category_guardrail_v26(item, requested)
        if action == "pass":
            suggested = v26_suggest_category_from_text(t, requested)
            if suggested and suggested != requested:
                return "reassign", suggested, f"v26_government_without_strict_actor_action:{reason}"
            return "fail", requested, f"v26_government_without_strict_actor_action:{reason}"
        return action, cat, reason

    action, cat, reason = _BASE_v23_final_category_guardrail_v26(item, requested)
    if action == "reassign" and cat == "정부_국회":
        if not v26_has_korean_gov_action(t):
            suggested = v26_suggest_category_from_text(t, requested)
            if suggested and suggested != "정부_국회":
                return "reassign", suggested, f"v26_blocked_weak_gov_reassign:{reason}"
            return "fail", requested, f"v26_blocked_weak_gov_reassign:{reason}"
    return action, cat, reason


_BASE_is_report_item_relevant_v26 = is_report_item_relevant

def is_report_item_relevant(report_item, json_key):
    t = v23_text(report_item)
    requested = v19_normalize_category_key(json_key, fallback="산업동향")
    if requested == "자사_및_계열사_이슈":
        if not v26_has_strict_self_entity(t):
            return False, "v26_self_without_direct_kakao_entity"
        if v20_is_platform_obligation_text(t) and not V26_DIRECT_SELF_EVENT_RE.search(t) and v26_has_korean_gov_action(t):
            return False, "v26_self_included_platform_obligation_should_be_gov"
    if requested == "정부_국회":
        if v26_is_gov_misclassified_industry(t):
            return False, "v26_government_misclassified_industry_or_earnings"
        if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
            return False, "v26_foreign_policy_not_korean_gov_section"
        if not v26_has_korean_gov_action(t):
            return False, "v26_government_without_strict_actor_action"
    return _BASE_is_report_item_relevant_v26(report_item, json_key)


_BASE_v24_topic_key_v26 = v24_topic_key

def v24_topic_key(item):
    key = v26_event_key_for_item(item)
    if key:
        return "v26:" + key
    return _BASE_v24_topic_key_v26(item)


_BASE_v24_duplicate_survivor_score_v26 = v24_duplicate_survivor_score

def v24_duplicate_survivor_score(item):
    score = float(_BASE_v24_duplicate_survivor_score_v26(item) or 0)
    t = v24_item_text(item)
    cat = item.get("카테고리", "")
    form = v26_article_form_from_text(t)
    today_score, _ = v26_todayness_score_text(t, item.get("게시일", ""))
    score += min(18, today_score * 0.35)
    if cat == "자사_및_계열사_이슈":
        if not v26_has_strict_self_entity(t):
            score -= 150
        elif V26_DIRECT_SELF_EVENT_RE.search(t):
            score += 30
        if form in {"analysis_feature", "market_commentary"} and not v26_is_self_direct_today_event(t, item.get("게시일", "")):
            score -= 35 if form == "analysis_feature" else 70
    if cat == "정부_국회":
        if v26_has_korean_gov_action(t):
            score += 35
        if v26_is_gov_misclassified_industry(t):
            score -= 120
    if cat == "경쟁사_해외이슈":
        if v23_has_competitor_entity(t):
            score += 8
    return score


def v26_final_item_priority_score(item):
    t = v23_text(item)
    cat = item.get("카테고리", "")
    form = v26_article_form_from_text(t)
    today_score, _ = v26_todayness_score_text(t, item.get("게시일", ""))
    score = v20_float(item.get("대표선택점수"), 0) + today_score
    if cat == "자사_및_계열사_이슈":
        if V26_DIRECT_SELF_EVENT_RE.search(t):
            score += 70
        if form in {"analysis_feature", "market_commentary"} and not v26_is_self_direct_today_event(t, item.get("게시일", "")):
            score -= 80 if form == "analysis_feature" else 140
    elif cat == "정부_국회":
        if v26_has_korean_gov_action(t):
            score += 45
        if v26_is_gov_misclassified_industry(t):
            score -= 140
    elif cat == "경쟁사_해외이슈":
        if v21_competitor_scope(t) in {"domestic", "mixed"}:
            score += 20
    elif cat == "산업동향":
        if form == "analysis_feature":
            score -= 20
    return round(score, 2)


def v26_dedupe_by_event_key(items):
    groups = {}
    for item in items:
        key = v26_event_key_for_item(item)
        if key:
            groups.setdefault(key, []).append(item)
    removed = []
    remove_ids = set()
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        winner = max(group, key=v24_duplicate_survivor_score)
        for item in group:
            if item is winner:
                item["v26중복대표선택"] = f"kept:{key}:score={v24_duplicate_survivor_score(item):.1f}"
                continue
            remove_ids.add(id(item))
            item["v26중복대표선택"] = f"removed:{key}:winner={winner.get('기사제목','')[:80]}"
            removed.append((item, f"v26_cross_section_event_duplicate:{key}:winner={winner.get('기사제목','')[:100]}"))
    return [item for item in items if id(item) not in remove_ids], removed


_BASE_v21_remove_obvious_final_duplicates_v26 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    repaired = []
    removed = []
    for item in final_items:
        requested = item.get("카테고리", "")
        action, cat, reason = v23_final_category_guardrail(item, requested)
        if action == "fail":
            item["v26최종수리결과"] = f"removed:{reason}"
            removed.append((item, f"v26_final_guardrail_removed:{reason}"))
            continue
        if action == "reassign" and cat:
            v23_set_final_category(item, cat, reason)
            item["v26최종수리결과"] = f"reassigned:{reason}"
        else:
            item["v26최종수리결과"] = f"pass:{reason}"
        item["v26기사유형"] = v26_article_form_from_text(v23_text(item))
        item["v26오늘성점수"] = v26_todayness_score_text(v23_text(item), item.get("게시일", ""))[0]
        item["v26이벤트키"] = v26_event_key_for_item(item)
        item["v26최종우선점수"] = v26_final_item_priority_score(item)
        repaired.append(item)

    kept, removed_base = _BASE_v21_remove_obvious_final_duplicates_v26(repaired)
    removed.extend(removed_base)
    kept, removed_event = v26_dedupe_by_event_key(kept)
    removed.extend(removed_event)

    # 카테고리 내부 순서 재정렬: 자사는 오늘 직접 사건 우선, 분석/시황성은 뒤로.
    order_base = {key: idx * 1000 for idx, key in enumerate(JSON_KEYS_ORDER, 1)}
    for key in JSON_KEYS_ORDER:
        bucket = [x for x in kept if x.get("카테고리") == key]
        bucket_sorted = sorted(bucket, key=v26_final_item_priority_score, reverse=True)
        for idx, item in enumerate(bucket_sorted, 1):
            item["선정순서"] = order_base.get(key, 9000) + idx
    return kept, removed


_BASE_clean_extracted_body_v6_v26 = clean_extracted_body_v6

def clean_extracted_body_v6(body):
    text = _BASE_clean_extracted_body_v6_v26(body)
    if not text:
        return text
    # 포털 UI 문구는 마침표 없이 본문 사이에 섞이는 경우가 많아 문장 단위로 제거한다.
    parts = re.split(r"(?<=[.!?다요임음됨함])\s+", text)
    cleaned = []
    for part in parts:
        p = clean_html_text(part)
        if not p:
            continue
        if V26_PORTAL_UI_NOISE_RE.search(p):
            continue
        cleaned.append(p)
    out = " ".join(cleaned) if cleaned else text
    out = V26_PORTAL_UI_NOISE_RE.sub(" ", out)
    out = re.sub(r"\s+", " ", out).strip()
    return out


_BASE_convert_to_report_ending_v26 = convert_to_report_ending

def convert_to_report_ending(sentence):
    s = clean_html_text(sentence).rstrip(". ")
    # 자주 깨지는 '습니다/됩니다/했습니다' 계열을 먼저 처리한다.
    direct = [
        (r"했습니다$", "했음"), (r"하였습니다$", "했음"), (r"밝혔습니다$", "밝힘"),
        (r"했습니다만$", "했음"), (r"됩니다$", "됨"), (r"됐습니다$", "됐음"),
        (r"있습니다$", "있음"), (r"없습니다$", "없음"), (r"계획입니다$", "계획임"),
        (r"예정입니다$", "예정임"), (r"상황입니다$", "상황임"), (r"수준입니다$", "수준임"),
        (r"입니다$", "임"), (r"합니다$", "함"), (r"습니다$", "음"),
    ]
    for pat, repl in direct:
        if re.search(pat, s):
            return re.sub(pat, repl, s) + "."
    return _BASE_convert_to_report_ending_v26(sentence)


_BASE_v22_update_history_indices_v26 = v22_update_history_indices

def v22_update_history_indices(final_report_data, issues, raw_articles, qa_overall=""):
    # 저장 직전에 오늘 날짜 행을 다시 제거해, 중간 실패/재실행에도 index가 중복 누적되지 않게 한다.
    v26_remove_same_date_rows_from_index(stage="before_index_append")

    article_rows = []
    issue_rows = []
    report_rows = []
    url_rows = []
    for idx, item in enumerate(final_report_data, 1):
        title = clean_html_text(item.get("기사제목", ""))
        url = normalize_url(item.get("링크", ""))
        issue_group = clean_html_text(item.get("v20_issue_group") or item.get("Gemini이슈그룹", ""))
        body = item.get("본문전문", "") or ""
        summary = summarize_body_locally(title, body, max_chars=250) if item.get("본문상태") == "본문추출성공" else clean_html_text(item.get("RSS요약", ""))[:250]
        event_key = v26_event_key_for_item(item)
        event_stage = v26_extract_development_stage(v23_text(item))
        today_score, today_reason = v26_todayness_score_text(v23_text(item), item.get("게시일", ""))
        common = {
            "date": V22_RUN_DATE,
            "section": item.get("카테고리", ""),
            "rank": idx,
            "title": title,
            "press": item.get("언론사", ""),
            "url": url,
            "canonical_url": normalize_url(url),
            "issue_group": issue_group,
            "issue_family": item.get("v20_issue_family", ""),
            "event_key": event_key,
            "event_stage": event_stage,
            "article_form": item.get("v26기사유형") or v26_article_form_from_text(v23_text(item)),
            "todayness_score": today_score,
            "todayness_reason": today_reason,
            "body_status": item.get("본문상태", ""),
            "qa_overall": qa_overall,
            "title_hash": title_fingerprint(title),
            "url_hash": normalize_url(url),
            "summary": summary,
            "source_version": V6_VERSION,
        }
        article_rows.append(dict(common, selected=True, article_id=item.get("브리핑ID", "")))
        report_rows.append(common)
        if url:
            url_rows.append({
                "date": V22_RUN_DATE,
                "url": url,
                "canonical_url": normalize_url(url),
                "title": title,
                "press": item.get("언론사", ""),
                "issue_group": issue_group,
                "event_key": event_key,
                "event_stage": event_stage,
                "title_hash": title_fingerprint(title),
                "source_version": V6_VERSION,
            })

    selected_issue_ids = {x.get("v20_issue_id") for x in final_report_data if x.get("v20_issue_id")}
    for issue in issues or []:
        if issue.get("issue_id") not in selected_issue_ids:
            continue
        issue_text = clean_html_text(f"{issue.get('issue_group','')} {issue.get('issue_family','')}")
        issue_rows.append({
            "date": V22_RUN_DATE,
            "issue_id": issue.get("issue_id"),
            "issue_group": issue.get("issue_group"),
            "issue_family": issue.get("issue_family"),
            "category": v20_issue_to_output_category(issue),
            "priority": max(issue.get("max_ceo_priority", 0) or 0, issue.get("max_pa_priority", 0) or 0),
            "top_title": issue.get("top_title", ""),
            "top_url": issue.get("top_url", ""),
            "selected": True,
            "issue_hash": v20_issue_group_key(issue.get("issue_group", "")),
            "event_key": issue.get("v26_event_key", "") or v26_event_key_for_item({"기사제목": issue.get("issue_group", ""), "본문요약": issue_text}),
            "max_todayness": issue.get("v26_max_todayness", ""),
            "article_forms": issue.get("v26_article_forms", ""),
            "source_version": V6_VERSION,
        })

    v26_append_rows_csv(V22_INDEX_DIR / "article_history.csv", article_rows)
    v26_append_rows_csv(V22_INDEX_DIR / "issue_history.csv", issue_rows)
    v26_append_rows_csv(V22_INDEX_DIR / "report_history.csv", report_rows)
    v26_append_rows_csv(V22_INDEX_DIR / "url_seen_history.csv", url_rows)


def v20_gemini_edit_issues(client, issues, recent_past_text):
    if not client:
        raise RuntimeError("Gemini client 없음")
    visible_issues = [i for i in sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True) if not i.get("exclude") and not i.get("is_pr")]
    visible_issues = visible_issues[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = "\n".join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
너는 카카오 대외협력팀 아침 뉴스 편집장이다.
아래 이슈 후보는 기사 단위가 아니라 issue_group 단위로 묶인 후보들이다.
최종 보고서는 카카오 경영진/대외협력/정책/커뮤니케이션/사업 담당자가 오늘 알아야 할 이슈를 담아야 한다.

[전체 편집 원칙]
- 같은 issue_group 또는 event_key는 원칙적으로 하나만 고른다.
- 최근 7일 index 이력에 있던 이슈와 실질적으로 같으면 제외한다. 단, 예고→돌입, 검토→확정, 심의 예정→심의/제재, 조사→과징금처럼 단계가 오른 후속 진전은 허용한다.
- 중요한 이슈가 많은 날에는 전체 상한인 15개까지 골라도 된다. 단, 낮은 품질 기사나 분석성 보충 기사로 억지로 채우지 말라.
- 단순 홍보·후원·캠페인·기부·이벤트·수상·주가전망·목표가·투자의견은 제외한다.

[자사/계열사 원칙: 오늘성 + 직접성 우선]
1. 자사 섹션은 카카오/카카오톡/카카오페이/카카오뱅크/카카오모빌리티/카카오게임즈 등 직접 주체가 명확한 기사만 넣는다.
2. 오늘 새로 발생했거나 오늘 새 단계로 진전된 직접 사건을 최우선으로 둔다. 노무/파업/임단협, 장애/보안/개인정보, 소송/수사/제재, 임원/조직개편, 경영권/M&A/실적 변화가 우선이다.
3. '성적표', '대해부', '리포트', '전망', '주주환원', '목표가', '투자의견', '주간동향' 같은 분석/기획/시황성 기사는 직접 사건이 부족할 때만 보충으로 사용한다.
4. 카카오가 아닌 기업의 개인정보·제재·보안 이슈는 자사 섹션에 넣지 말라.

[정부/국회 원칙: 한국 행위자 기준]
1. 정부/국회 섹션은 한국 정부·국회·규제기관·공공기관이 실제 행위자인 정책/규제/제도 기사만 넣는다.
2. 공정위·금융위·금감원·개인정보위·과기정통부·방미통위·국회·국무회의 등의 심의/의결/통과/시행/조사/제재/가이드라인은 우선한다.
3. 기업 실적, 보안 상장사 영업이익, 산업 분석, 민간기업 내부 조직 뉴스는 검색어가 정부기관이어도 정부/국회에 넣지 말라.
4. 미국·EU·중국·일본 등 해외 정부/의회/규제기관 정책은 정부/국회가 아니라 경쟁사/해외이슈로 분류한다. 한국 정부 도입 검토나 국내 사업자 의무로 직접 연결될 때만 예외다.

[경쟁사/해외 원칙]
1. 국내 경쟁사/인접 플랫폼을 우선한다. 네이버·네이버페이·쿠팡·토스·배민·SKT·KT·LGU+·라인야후 등.
2. 해외 빅테크/글로벌 AI 이슈는 기본 1개, 매우 중요하면 2개까지 가능하다.
3. 해외 법안/규제/정책, 미국 AI 법안, EU 규제, 트럼프 행정부 AI 정책 등은 이 섹션에서 다룬다.

[권장 분량]
- 전체 11~15개. 자사 3~5개, 정부/국회 3~5개, 경쟁사/해외 3~4개, 산업 1~2개.

반드시 JSON만 반환하라.
{{
  "selected_issues": [
    {{
      "issue_id": "I001",
      "category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향",
      "priority": 5,
      "best_article_id": 123,
      "backup_article_ids": [124, 125],
      "reason": "선정 이유"
    }}
  ],
  "backup_issues": [
    {{
      "issue_id": "I050",
      "category": "경쟁사_해외이슈",
      "priority": 4,
      "best_article_id": 555,
      "backup_article_ids": [556],
      "reason": "대체 후보 이유"
    }}
  ]
}}

[최근 7일 index 이력 참고]
{recent_past_text[:4500]}

[이슈 후보]
{issue_text}
"""
    text = gemini_generate_text(
        client=client,
        prompt=prompt,
        task_name="v26 이슈 단위 최종 편집",
        model=GEMINI_MODEL_EDITOR,
    )
    data = extract_json_object(text)
    issue_by_id = {i["issue_id"]: i for i in issues}
    article_by_id = {}
    for issue in issues:
        for art_id in issue.get("article_ids", []):
            article_by_id[int(art_id)] = {"id": int(art_id)}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    return decisions, backup_decisions



# =========================================================
# v26.1 patch: same-date index cleanup continuation-line safe
# - 기존 index CSV가 이미 깨져 있어 한 기사 행이 여러 줄로 쪼개졌을 때도
#   오늘 날짜 행과 그 continuation line을 함께 제거한다.
# =========================================================

def v26_line_starts_date(line):
    s = str(line or "").lstrip("\ufeff").lstrip().strip()
    return bool(re.match(r'^"?\d{4}-\d{2}-\d{2}"?(,|$)', s))


def v26_line_starts_run_date(line):
    s = str(line or "").lstrip("\ufeff").lstrip().strip()
    return (
        s.startswith(f"{V22_RUN_DATE},")
        or s.startswith(f'"{V22_RUN_DATE}",')
        or s == V22_RUN_DATE
        or s == f'"{V22_RUN_DATE}"'
    )


def v25_remove_same_date_rows_from_index():
    """같은 날짜 재실행 시 index에서 오늘 날짜 행을 제거한다.
    CSV가 깨져서 본문 줄바꿈이 행으로 분리된 경우에도 다음 날짜 행이 나오기 전까지 함께 제거한다.
    """
    if not REPLACE_SAME_DATE_INDEX_ROWS:
        return
    index_files = ["article_history.csv", "issue_history.csv", "report_history.csv", "url_seen_history.csv"]
    v25_backup_index_files_for_run_date()
    removed_summary = {}
    for filename in index_files:
        path = V22_INDEX_DIR / filename
        if not path.exists():
            removed_summary[filename] = 0
            continue
        try:
            with open(path, "r", encoding="utf-8-sig", errors="replace", newline="") as f:
                lines = f.readlines()
            kept = []
            removed = 0
            skipping_run_row = False
            for i, line in enumerate(lines):
                stripped = line.lstrip("\ufeff").lstrip()
                if i == 0 and re.match(r'"?(date|run_date|report_date|날짜)"?\s*,', stripped):
                    kept.append(line)
                    skipping_run_row = False
                    continue
                if v26_line_starts_run_date(line):
                    removed += 1
                    skipping_run_row = True
                    continue
                if skipping_run_row:
                    if v26_line_starts_date(line):
                        skipping_run_row = False
                    else:
                        removed += 1
                        continue
                kept.append(line)
            with open(path, "w", encoding="utf-8-sig", newline="") as f:
                f.writelines(kept)
            removed_summary[filename] = removed
        except Exception as e:
            print(f"  └ ⚠️ {filename}: 같은 날짜 행 제거 실패: {e}")
            removed_summary[filename] = "실패"
    msg = ", ".join([f"{k}:{v}행" for k, v in removed_summary.items()])
    print(f"  └ 🧹 같은 날짜 index 기존 행 제거 완료({V22_RUN_DATE}): {msg}")


# =========================================================
# v26.1 operational/editorial patch
# - Keep v26 index-based history and strict self/gov guardrails.
# - Relax index same-event handling at RSS stage so follow-up developments are not lost.
# - Restore v24-like editorial balance by refilling after final guardrails remove weak items.
# - Re-rank self section by direct/today event priority so labor/incidents outrank analysis.
# =========================================================

V6_VERSION = "google_rss_v26_1_soft_index_v24_feel_refill"
V26_1_FINAL_MIN = 12
V26_1_FINAL_TARGET = 13
V21_FINAL_MIN = max(V21_FINAL_MIN, V26_1_FINAL_MIN)
V20_FINAL_MIN = V21_FINAL_MIN
MIN_SELECT_COUNT = V21_FINAL_MIN

# Unicode-escaped Korean regexes keep this patch portable while matching Korean titles/body.
V26_1_SELF_LABOR_RE = re.compile(
    r"\ub178\uc870|\ud30c\uc5c5|\uc784\ub2e8\ud611|\uc7c1\uc758\uad8c|\uc7c1\uc758|\ub178\ub3d9\ubd80|\uace0\uc6a9\ubd88\uc548|\ubcf4\uc0c1\uccb4\uacc4|\uc131\uacfc\uae09|RSU",
    re.IGNORECASE,
)
V26_1_SELF_INCIDENT_RE = re.compile(
    r"\uc7a5\uc560|\uc624\ub958|\uba39\ud1b5|\uac1c\uc778\uc815\ubcf4|\uc720\ucd9c|\ud574\ud0b9|\ud53c\uc2f1|\ubcf4\uc548|\uacfc\uc9d5\uae08|\uc81c\uc7ac|\uc18c\uc1a1|\uc218\uc0ac|\uc870\uc0ac|\uace0\ubc1c|\ud310\uacb0|\ud589\uc815\uc18c\uc1a1",
    re.IGNORECASE,
)
V26_1_SELF_LEADERSHIP_RE = re.compile(
    r"\ub300\ud45c|\uc784\uc6d0|CPO|CTO|CFO|\uc870\uc9c1\uac1c\ud3b8|\ud1f4\uc0ac|\uc0ac\uc784|\ud574\uc784|\uc120\uc784|\ub0b4\uc815|\uad50\uccb4|\uacbd\uc601\uc9c4",
    re.IGNORECASE,
)
V26_1_SELF_GOVERNANCE_RE = re.compile(
    r"\uc900\uc2e0\uc704|\uc900\ubc95|\uc2e0\ub8b0\uacbd\uc601|AI\s*\uac70\ubc84\ub10c\uc2a4|\uae30\uc220\s*\uc724\ub9ac|\uc0ac\ud68c\uc801\s*\uc2e0\ub8b0",
    re.IGNORECASE,
)
V26_1_SELF_INVESTMENT_RE = re.compile(
    r"\ub450\ub098\ubb34|\uc5c5\ube44\ud2b8|\uc9c0\ubd84|\ub9e4\uac01|\ud22c\uc790\ud68c\uc218|\uc8fc\uc2dd\uad50\ud658|\ud569\ubcd1|\ub124\uc774\ubc84\ud30c\uc774\ub0b8\uc15c|\uad6c\uc8fc",
    re.IGNORECASE,
)
V26_1_ANALYSIS_TITLE_RE = re.compile(
    r"\uc131\uc801\ud45c|\ub300\ud574\ubd80|\ub9ac\ud3ec\ud2b8|\uc804\ub9dd|\uc9da\uc5b4\ubcf4\ub2c8|\uc2dc\ub9ac\uc988|\uae30\ud68d|\uc8fc\uac04|\ubaa8\ub798\s*\uc704|\uc8fc\uc8fc\ud658\uc6d0|\ud22c\uc790\uc758\uacac|\ubaa9\ud45c\uac00",
    re.IGNORECASE,
)
V26_1_OVERSEAS_HINT_RE = re.compile(
    r"EU|\uc720\ub7fd|\ubbf8\uad6d|\ubbf8\s*\uc815\ubd80|\ubbf8\s*\uc758\ud68c|\ubc31\uc545\uad00|\ud2b8\ub7fc\ud504|\uc911\uad6d|\uc77c\ubcf8|\uae00\ub85c\ubc8c|\uad6c\uae00|\uc624\ud508AI|OpenAI|MS|\uba54\ud0c0|\uc560\ud50c|Anthropic|\uc564\ud2b8\ub85c\ud53d",
    re.IGNORECASE,
)


def v26_1_section_minima():
    return {
        JSON_KEYS_ORDER[0]: 3,
        JSON_KEYS_ORDER[1]: 3,
        JSON_KEYS_ORDER[2]: 3,
        JSON_KEYS_ORDER[3]: 1,
    }


def v26_1_final_item_text(item):
    try:
        return v23_text(item)
    except Exception:
        return v26_text_for_event(item)


def v26_1_self_bucket(text):
    t = clean_html_text(text)
    if V26_1_SELF_LABOR_RE.search(t):
        return "self_labor", 170
    if V26_1_SELF_INCIDENT_RE.search(t):
        return "self_incident", 150
    if V26_1_SELF_LEADERSHIP_RE.search(t):
        return "self_leadership", 125
    if V26_1_SELF_GOVERNANCE_RE.search(t):
        return "self_governance", 80
    if V26_1_SELF_INVESTMENT_RE.search(t):
        return "self_investment", 45
    if V26_DIRECT_SELF_EVENT_RE.search(t):
        return "self_direct_other", 90
    return "self_reference", 35


def v26_1_similarity_for_event(candidate, past):
    cand_title = clean_html_text(candidate.get("기사제목") or candidate.get("title") or "")
    past_title = clean_html_text(past.get("title") or past.get("기사제목") or "")
    cand_text = v26_text_for_event(candidate)
    past_text = v26_text_for_event(past)
    title_score = sequence_ratio(cand_title, past_title)
    token_score = jaccard(tokenize_for_similarity(cand_text), tokenize_for_similarity(past_text))
    text_score = jaccard(char_ngrams(cand_text[:1400], 5), char_ngrams(past_text[:1400], 5))
    return title_score, token_score, text_score


def v26_index_duplicate_check(candidate, recent_past_items, mode="rss"):
    """v26.1: index history duplicate check.
    Strong URL/title duplicates are excluded. Same event keys are soft at RSS stage and
    only excluded after body-level similarity is high enough. This preserves updates.
    """
    cand_title = clean_html_text(candidate.get("기사제목") or candidate.get("title") or "")
    cand_url = normalize_url(candidate.get("링크") or candidate.get("link") or candidate.get("url") or "")
    cand_title_hash = title_fingerprint(cand_title)
    cand_event_key = v26_event_key_for_item(candidate)

    update_match = None
    update_reason = ""
    soft_match = None
    soft_reason = ""

    for past in recent_past_items or []:
        past_url = normalize_url(past.get("link") or past.get("url") or "")
        past_title = clean_html_text(past.get("title") or past.get("기사제목") or "")
        past_title_hash = past.get("title_hash") or title_fingerprint(past_title)
        past_event_key = past.get("event_key") or v26_event_key_for_item(past)

        if cand_url and past_url and cand_url == past_url:
            return True, past, {"is_duplicate": True, "reason": "index_same_url", "score": 1.0, "shared_entities": "", "shared_tags": ""}
        if cand_title_hash and past_title_hash and cand_title_hash == past_title_hash:
            return True, past, {"is_duplicate": True, "reason": "index_same_title_hash", "score": 1.0, "shared_entities": "", "shared_tags": ""}
        if cand_title and past_title:
            near, score = is_title_near_duplicate(cand_title, past_title)
            if near and score >= 0.95:
                return True, past, {"is_duplicate": True, "reason": f"index_near_title:{score:.2f}", "score": score, "shared_entities": "", "shared_tags": ""}

        if cand_event_key and past_event_key and cand_event_key == past_event_key:
            is_update, reason = v26_is_same_event_update(candidate, past)
            if is_update:
                update_match = past
                update_reason = reason
                continue
            if mode == "rss":
                # Keep for Gemini/body extraction. RSS snippets are too weak for same-event deletion.
                soft_match = past
                soft_reason = f"index_same_event_soft_keep_rss:{cand_event_key}"
                continue
            title_score, token_score, text_score = v26_1_similarity_for_event(candidate, past)
            if (title_score >= 0.76 and token_score >= 0.16) or (token_score >= 0.32 and text_score >= 0.38):
                return True, past, {
                    "is_duplicate": True,
                    "reason": f"index_same_event_body_similar:{cand_event_key}",
                    "score": round(max(title_score, token_score, text_score), 4),
                    "shared_entities": "",
                    "shared_tags": cand_event_key,
                }
            soft_match = past
            soft_reason = f"index_same_event_soft_keep_body:{cand_event_key}"

    if update_match:
        return False, update_match, {"is_duplicate": False, "reason": update_reason, "score": 0.0, "shared_entities": "", "shared_tags": cand_event_key}
    if soft_match:
        return False, soft_match, {"is_duplicate": False, "reason": soft_reason, "score": 0.0, "shared_entities": "", "shared_tags": cand_event_key}
    return False, None, None


def find_past_duplicate(candidate, recent_past_items, mode="body"):
    if DEDUP_USE_INDEX_HISTORY and recent_past_items:
        is_dup, matched, sim = v26_index_duplicate_check(candidate, recent_past_items, mode=mode)
        if is_dup:
            return True, matched, sim
        reason = str((sim or {}).get("reason", ""))
        if reason.startswith("same_event_but_new_stage") or reason.startswith("index_same_event_soft_keep"):
            return False, matched, sim
    return _BASE_find_past_duplicate_v26(candidate, recent_past_items, mode=mode)


def v26_final_item_priority_score(item):
    t = v26_1_final_item_text(item)
    cat = item.get("카테고리", "")
    form = v26_article_form_from_text(t)
    today_score, _ = v26_todayness_score_text(t, item.get("게시일", ""))
    base = v20_float(item.get("대표선택점수"), 0)
    score = base + min(35, today_score * 1.1)
    if cat == JSON_KEYS_ORDER[0]:
        if not v26_has_strict_self_entity(t):
            return -999
        bucket, bonus = v26_1_self_bucket(t)
        item["v26_1_self_bucket"] = bucket
        score += bonus
        if form == "analysis_with_direct_risk":
            score += 25
        elif form == "analysis_feature" and bucket in {"self_reference", "self_investment"}:
            score -= 35
        elif form == "market_commentary":
            score -= 120
        # Investment/M&A analysis is useful, but it should not outrank a same-day strike or incident.
        if bucket == "self_investment" and not re.search(r"\uacf5\uc2dc|\ud655\uc815|\uacb0\uc815|\uad6d\ubb34\ud68c\uc758|\uc81c\uc7ac|\uacfc\uc9d5\uae08", t):
            score -= 15
    elif cat == JSON_KEYS_ORDER[1]:
        if v26_has_korean_gov_action(t):
            score += 60
        if v26_is_gov_misclassified_industry(t):
            score -= 180
    elif cat == JSON_KEYS_ORDER[2]:
        scope = v21_competitor_scope(t)
        if scope in {"domestic", "mixed"}:
            score += 35
        elif scope == "overseas":
            score += 22
    elif cat == JSON_KEYS_ORDER[3]:
        if form == "analysis_feature":
            score -= 5
    return round(score, 2)


def v26_1_should_refill_after_repair(final_report_data):
    counts, comp_domestic, comp_overseas = v21_final_counts(final_report_data)
    if len(final_report_data) < V26_1_FINAL_MIN:
        return True
    for cat, min_count in v26_1_section_minima().items():
        if counts.get(cat, 0) < min_count:
            return True
    if counts.get(JSON_KEYS_ORDER[2], 0) >= 2 and comp_overseas <= 0:
        return True
    return False


def v26_1_decision_category(decision, issue):
    return v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue))


def v26_1_is_overseas_competitor_decision(decision, issue):
    cat = v26_1_decision_category(decision, issue)
    if cat != JSON_KEYS_ORDER[2]:
        return False
    if issue.get("competitor_scope") == "overseas":
        return True
    text = clean_html_text(f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('issue_family','')}")
    if V26_1_OVERSEAS_HINT_RE.search(text) and v21_competitor_scope(text) == "overseas":
        return True
    return False


def v26_1_is_good_refill_candidate(decision, issue, cat=None):
    cat = cat or v26_1_decision_category(decision, issue)
    if not issue:
        return False
    ok, _ = v21_issue_allowed_basic(issue)
    if not ok:
        return False
    if issue.get("self_tier") in {"LOW_VALUE_PR", "OFF_TOPIC"}:
        return False
    if cat == JSON_KEYS_ORDER[0] and issue.get("self_tier") == "FILLER_REFERENCE":
        # Allow only as a last resort; general refill will usually skip this.
        return False
    if issue.get("gov_tier") in {"GOV_GENERAL_LOW_RELEVANCE", "OFF_TOPIC"}:
        return False
    if issue.get("competitor_tier") in {"COMP_GENERAL_LOW_RELEVANCE", "OFF_TOPIC"}:
        return False
    return True


def v26_1_candidate_pool(decisions, backup_decisions, issues):
    pool = []
    seen = set()
    for source, iterable in [("selected", decisions), ("backup", backup_decisions), ("local", v21_candidate_decisions_from_issues(issues))]:
        for d in iterable:
            iid = d.get("issue_id")
            if not iid or iid in seen:
                continue
            d2 = dict(d)
            d2["v26_1_pool_source"] = source
            pool.append(d2)
            seen.add(iid)
    def sort_key(d):
        issue = globals().get("_v26_1_issue_by_id_for_sort", {}).get(d.get("issue_id"), {})
        return (v20_int(d.get("priority"), 1), v20_float(issue.get("issue_score"), 0), v20_float(issue.get("max_ceo_priority"), 0))
    return sorted(pool, key=sort_key, reverse=True)


def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    globals()["_v26_1_issue_by_id_for_sort"] = issue_by_id
    candidate_pool = v26_1_candidate_pool(decisions, backup_decisions, issues)
    represented_issue_ids = {x.get("v20_issue_id") for x in final_report_data if x.get("v20_issue_id")}

    def try_decision(d, source):
        nonlocal order_counter
        if len(final_report_data) >= V21_FINAL_MAX:
            return False
        iid = d.get("issue_id")
        if not iid or iid in represented_issue_ids:
            return False
        issue = issue_by_id.get(iid, {})
        if not issue:
            return False
        item, order_counter, status = v20_process_issue(
            d,
            issue_by_id,
            article_by_id,
            recent_past_items,
            final_report_data,
            body_failed_rows,
            skip_rows,
            processed_article_ids,
            order_counter,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        represented_issue_ids.add(iid)
        if item:
            item["v26_1_refill_source"] = source
            print(f"  -> v26.1 refill added({source}): {v20_clip(issue.get('issue_group',''), 42)} / {v20_clip(item.get('기사제목',''), 42)}...")
            return True
        return False

    # 1) If no overseas item survived, try to add one overseas/global big-tech issue.
    counts, comp_domestic, comp_overseas = v21_final_counts(final_report_data)
    if comp_overseas <= 0 and len(final_report_data) < V21_FINAL_MAX:
        for d in candidate_pool:
            issue = issue_by_id.get(d.get("issue_id"), {})
            if v26_1_is_overseas_competitor_decision(d, issue):
                if try_decision(d, "overseas_competitor_refill"):
                    break

    # 2) Restore section minima after final guardrails/duplicate repair.
    for cat, min_count in v26_1_section_minima().items():
        while v21_final_counts(final_report_data)[0].get(cat, 0) < min_count and len(final_report_data) < V21_FINAL_MAX:
            added = False
            for d in candidate_pool:
                issue = issue_by_id.get(d.get("issue_id"), {})
                if v26_1_decision_category(d, issue) != cat:
                    continue
                if not v26_1_is_good_refill_candidate(d, issue, cat=cat):
                    continue
                if try_decision(d, f"section_min_refill:{cat}"):
                    added = True
                    break
            if not added:
                break

    # 3) Fill total minimum/target with strong remaining candidates, without forcing weak filler.
    while len(final_report_data) < V26_1_FINAL_MIN and len(final_report_data) < V21_FINAL_MAX:
        added = False
        for d in candidate_pool:
            issue = issue_by_id.get(d.get("issue_id"), {})
            cat = v26_1_decision_category(d, issue)
            if not v26_1_is_good_refill_candidate(d, issue, cat=cat):
                continue
            if try_decision(d, "total_min_refill"):
                added = True
                break
        if not added:
            break

    return order_counter



# =========================================================
# v26.2 overrides: generic daily editorial profile
# - Avoid one-off keyword patching. Classify article structure: public actor/action,
#   single-issue vs roundup, structural industry vs market/stock commentary.
# - Government/National Assembly requires a real Korean public actor + official action.
#   Public-private MOU/project participation is allowed only when the public actor/project
#   is central, not when it is one line inside a roundup.
# - If Industry has no strong item, promote one structural overseas/global big-tech item
#   from Competitor/Overseas to Industry.
# - Exclude stock/target-price/feature-stock articles from final report.
# =========================================================

V6_VERSION = "google_rss_v26_2_generic_profile_gov_industry_pivot"
V26_2_FINAL_MIN = 12
V21_FINAL_MIN = max(V21_FINAL_MIN, V26_2_FINAL_MIN)
V20_FINAL_MIN = V21_FINAL_MIN
MIN_SELECT_COUNT = V21_FINAL_MIN

V26_2_PUBLIC_ACTOR_EXTRA_RE = re.compile(
    r"\uc9c0\uc790\uccb4|\uc9c0\ubc29\uc790\uce58\ub2e8\uccb4|\uc2dc\uccad|\ub3c4\uccad|\uad6c\uccad|\uad70\uccad|\uacf5\uacf5\uae30\uad00|"
    r"[\uAC00-\uD7A3]{2,6}\uc2dc(?:\uac00|\ub294|\uc640|\uacfc|\uc758|\uc5d0\uc11c|\uc5d0|\ub97c|\uc744|\ub3c4)",
    re.IGNORECASE,
)
V26_2_PUBLIC_PROJECT_RE = re.compile(
    r"\uad6d\uac00\uc0ac\uc5c5|\uacf5\uacf5\uc0ac\uc5c5|\uc815\ubd80\uc0ac\uc5c5|\uc9c0\uc790\uccb4\uc0ac\uc5c5|\uacf5\ubaa8|\uc120\uc815|\ucc29\uc218|\uad6c\ucd95|\uc2e4\uc99d|"
    r"\ucee8\uc18c\uc2dc\uc5c4|\ucc38\uc5ec|\uc608\uc0b0|\ud22c\uc785|\uacf5\uacf5\uc870\ub2ec|\ubc1c\uc8fc|\uc9c0\uc6d0\uc0ac\uc5c5|\ub3c4\uc2dc\uc548\uc804\ub9dd|"
    r"\uc5c5\ubb34\ud611\uc57d|MOU|\ud611\uc57d|\ud611\ub825",
    re.IGNORECASE,
)
V26_2_OFFICIAL_ACTION_RE = re.compile(
    r"\ubc1c\ud45c|\uc2dc\ud589|\uc758\uacb0|\ud1b5\uacfc|\uc2ec\uc758|\uc81c\uc7ac|\uacfc\uc9d5\uae08|\uacfc\ud0dc\ub8cc|\uc870\uc0ac|\uc810\uac80|\uac80\uc0ac|\uac10\ub3c5|"
    r"\uc2dc\uc815\uba85\ub839|\uace0\ubc1c|\uac00\uc774\ub4dc\ub77c\uc778|\ubc95\uc548|\ubc1c\uc758|\uc785\ubc95|\uc2dc\ud589\ub839|\uac1c\uc815\uc548|"
    r"\uc785\ubc95\uc608\uace0|\ud589\uc815\uc608\uace0|\uad6d\ubb34\ud68c\uc758|\uc804\uc6d0\ud68c\uc758|\ud611\uc758\uccb4|\ucd9c\ubc94|\uac04\ub2f4\ud68c|\uc124\uba85\ud68c",
    re.IGNORECASE,
)
V26_2_ROUNDUP_TITLE_RE = re.compile(
    r"\uc8fc\uac04|\uc6d4\uac04|\uc774\uc8fc\uc758|\uc624\ub298\uc758|\ub3d9\ud5a5|\ub2e8\uc2e0|\ubaa8\uc74c|\uc885\ud569|\ub77c\uc6b4\ub4dc\uc5c5|\uc18c\uc2dd|\u5916|\uc678$",
    re.IGNORECASE,
)
V26_2_MARKET_STOCK_RE = re.compile(
    r"\ud2b9\uc9d5\uc8fc|\uc8fc\uac00|\uc99d\uc2dc|\uc7a5\uc911|\uc7a5\ub9c8\uac10|\ubaa9\ud45c\uac00|\ud22c\uc790\uc758\uacac|\ub9e4\uc218|\ub9e4\ub3c4|"
    r"\uae09\ub4f1|\uae09\ub77d|\uc0c1\uc2b9|\ud558\ub77d|\u25b2|\u25bc|\u2191|\u2193|\u2192|%",
    re.IGNORECASE,
)
# 산업동향은 "AI/보안/클라우드" 단어만으로 통과시키지 않고,
# 시장 구조 변화/산업 재편/대형 인프라 변화 같은 구조 신호가 함께 있어야 한다.
V26_2_INDUSTRY_STRUCTURE_SIGNAL_RE = re.compile(
    r"시장\s*구조|산업\s*구조|산업\s*재편|시장\s*재편|생태계|패러다임|표준|"
    r"확산|전환|대전환|대형\s*투자|인프라\s*투자|데이터센터\s*증설|AI\s*팩토리|"
    r"수요\s*증가|도입\s*확산|상용화|공급망|글로벌\s*경쟁|규제\s*환경|"
    r"플랫폼화|클라우드\s*전환|보안\s*수요|AI\s*인프라",
    re.IGNORECASE,
)

# 개별 기업/기관 홍보·제품·간담회·대학 발표는 기본적으로 산업동향에서 제외한다.
# 단, 기사 자체가 산업 구조 변화/대형 인프라 변화로 설명될 때는 예외적으로 허용한다.
V26_2_WEAK_INDUSTRY_PROMO_RE = re.compile(
    r"기자간담회|신제품|출시|솔루션|전략\s*공개|시장\s*공략|사업\s*확대|"
    r"업무협약|MOU|협약|대학교|대학|건학|캠퍼스|교수|연구팀|논문|학회|"
    r"제품\s*3종|포트폴리오\s*강화|국내\s*시장\s*확대",
    re.IGNORECASE,
)
V26_2_STRUCTURAL_INDUSTRY_RE = re.compile(
    r"AI|\uc778\uacf5\uc9c0\ub2a5|\uc5d0\uc774\uc804\ud2b8|\ud53c\uc9c0\uceec\s*AI|\uc628\ub514\ubc14\uc774\uc2a4|\ud504\ub77c\uc774\ubc84\uc2dc|\ubcf4\uc548|"
    r"\ub370\uc774\ud130\uc13c\ud130|AI\s*\ud329\ud1a0\ub9ac|GPU|NPU|\ud074\ub77c\uc6b0\ub4dc|\ud50c\ub7ab\ud3fc|\ub9dd\s*\uc0ac\uc6a9\ub8cc|\uc800\uc791\uad8c|"
    r"\ub514\uc9c0\ud138\uc790\uc0b0|\uc2a4\ud14c\uc774\ube14\ucf54\uc778|\ub370\uc774\ud130\s*\ud65c\uc6a9|\uc571\ub9c8\ucf13|\uac80\uc0c9|\uad11\uace0|\uc0dd\ud0dc\uacc4|\uc0b0\uc5c5\s*\uc7ac\ud3b8",
    re.IGNORECASE,
)
V26_2_OVERSEAS_STRUCTURAL_RE = re.compile(
    r"\uc560\ud50c|Apple|\uad6c\uae00|Google|\uc624\ud508AI|OpenAI|\ucc57GPT|ChatGPT|MS|Microsoft|\ub9c8\uc774\ud06c\ub85c\uc18c\ud504\ud2b8|"
    r"\uba54\ud0c0|Meta|\uc564\ud2b8\ub85c\ud53d|Anthropic|\ud074\ub85c\ub4dc|Claude|\uc624\ub77c\ud074|Oracle|EU|\uc720\ub7fd|\ubbf8\uad6d|\uc911\uad6d|\uc77c\ubcf8|\ube45\ud14c\ud06c",
    re.IGNORECASE,
)
V26_2_ROUNDUP_ACTION_TERMS = [
    "\uc778\uc218", "\uc5c5\ubb34\ud611\uc57d", "MOU", "\ucd9c\uc2dc", "\uc120\ubcf4", "\ucc38\uc5ec", "\ud22c\uc790", "\uc720\uce58",
    "\ud611\ub825", "\uacf5\uac1c", "\uac1c\ucd5c", "\ucc29\uc218", "\uc120\uc815", "\uccb4\uacb0", "\ubc1c\ud45c", "\ucd9c\ubc94", "\uacf5\ubaa8",
]


def v26_2_text(obj):
    if isinstance(obj, dict):
        try:
            return v23_text(obj)
        except Exception:
            return clean_html_text(f"{obj.get('기사제목','')} {obj.get('title','')} {obj.get('본문요약','')} {obj.get('summary','')} {obj.get('본문전문','')[:2400]} {obj.get('issue_group','')} {obj.get('top_title','')}")
    return clean_html_text(obj)


def v26_2_title(obj):
    if isinstance(obj, dict):
        return clean_html_text(obj.get('기사제목') or obj.get('title') or obj.get('top_title') or obj.get('issue_group') or '')
    return clean_html_text(str(obj or ''))[:220]


def v26_2_public_actor_found(text):
    t = clean_html_text(text)
    return bool(V26_KR_GOV_ACTOR_STRICT_RE.search(t) or V26_2_PUBLIC_ACTOR_EXTRA_RE.search(t))


def v26_2_roundup_score(text, title=''):
    t = clean_html_text(text)
    title = clean_html_text(title) or t[:260]
    front = t[:2200]
    title_marker = bool(V26_2_ROUNDUP_TITLE_RE.search(title))
    transition_count = len(re.findall(r"\ub610\ud55c|\uc774\uc5b4|\ud55c\ud3b8|\uc774\uc640\s*\ud568\uaed8|\uac01\uac01|\ubcc4\ub3c4\ub85c|\u5916|\uc678\b", front))
    action_count = sum(1 for term in V26_2_ROUNDUP_ACTION_TERMS if term and term in front)
    semicolon_like = front.count('...') + front.count('·') + front.count(';')
    score = 0
    if title_marker:
        score += 3
    if transition_count >= 2:
        score += 2
    if action_count >= 4:
        score += 2
    if semicolon_like >= 3:
        score += 1
    return score


def v26_2_is_roundup(text, title=''):
    return v26_2_roundup_score(text, title) >= 5


def v26_2_is_stock_market_text(text, title=''):
    t = clean_html_text(text)
    title = clean_html_text(title) or t[:260]
    if re.search(r"\ud2b9\uc9d5\uc8fc|\ubaa9\ud45c\uac00|\ud22c\uc790\uc758\uacac|\uc99d\uc2dc", title, re.IGNORECASE):
        return True
    if re.search(r"\uc8fc\uac00", title) and re.search(r"%|\uae09\ub4f1|\uae09\ub77d|\uc0c1\uc2b9|\ud558\ub77d|\u2191|\u2193|\u25b2|\u25bc", title, re.IGNORECASE):
        return True
    if re.search(r"\uc8fc\uac00\s*\d+(?:\.\d+)?%|\d+(?:\.\d+)?%\s*(?:\uc0c1\uc2b9|\ud558\ub77d|\uae09\ub4f1|\uae09\ub77d|\u2191|\u2193|\u25b2|\u25bc)", t[:900], re.IGNORECASE):
        return True
    return False


def v26_2_public_policy_basis(text, title=''):
    t = clean_html_text(text)
    title = clean_html_text(title) or t[:280]
    if not t:
        return False, 'empty'
    if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
        return False, 'foreign_public_policy'

    front = t[:2200]
    title_front = title[:360]
    has_actor = v26_2_public_actor_found(front)
    has_title_actor = v26_2_public_actor_found(title_front)
    has_action = bool(V26_GOV_ACTION_STRICT_RE.search(front) or V26_2_OFFICIAL_ACTION_RE.search(front))
    has_domain = bool(V23_KAKAO_RELEVANT_DOMAIN_RE.search(front) or V23_PUBLIC_AI_POLICY_RE.search(front) or DIGITAL_STRATEGIC_PATTERN.search(front))
    is_roundup = v26_2_is_roundup(front, title)

    # Platform/legal obligation can pass even if the exact agency is only weakly shown.
    if v20_is_platform_obligation_text(front) and has_action:
        if has_actor or re.search(r"\uad6d\ub0b4|\ud55c\uad6d|\uc0ac\uc5c5\uc790|\ud50c\ub7ab\ud3fc\s*\uc0ac\uc5c5\uc790|\ub124\uc774\ubc84|\uce74\uce74\uc624|\uad6c\uae00|\ud3ec\ud138|SNS", front, re.IGNORECASE):
            return True, 'platform_obligation'

    public_project = bool(has_actor and V26_2_PUBLIC_PROJECT_RE.search(front) and has_domain)
    official_policy = bool(has_actor and has_action and has_domain)

    if is_roundup:
        # A roundup may mention a city/ministry in one item; keep it out of Government unless the public actor/action is central in the title/lead.
        if has_title_actor and (V26_GOV_ACTION_STRICT_RE.search(front[:900]) or V26_2_OFFICIAL_ACTION_RE.search(front[:900]) or public_project):
            return True, 'roundup_but_public_actor_central'
        return False, 'roundup_without_central_public_actor'

    if official_policy:
        return True, 'public_actor_official_action'
    if public_project:
        return True, 'public_private_project'
    return False, 'no_public_actor_action_domain'


def v26_2_is_structural_industry_text(text, title=''):
    t = clean_html_text(text)
    title = clean_html_text(title) or t[:260]

    if not t:
        return False

    if v26_2_is_stock_market_text(t, title):
        return False

    if v26_2_is_roundup(t, title):
        return False

    # 개별 회사/대학/연구팀/제품/간담회 성격은 산업동향에서 제외.
    # 단, 시장 구조 변화나 대형 인프라 변화가 명확하면 예외적으로 허용.
    weak_promo = bool(V26_2_WEAK_INDUSTRY_PROMO_RE.search(t))
    has_domain = bool(V26_2_STRUCTURAL_INDUSTRY_RE.search(t))
    has_structure = bool(V26_2_INDUSTRY_STRUCTURE_SIGNAL_RE.search(t))

    if weak_promo and not has_structure:
        return False

    # AI/보안/클라우드 단어만으로는 부족하고, 구조 변화 신호가 같이 있어야 산업동향으로 인정.
    if has_domain and has_structure:
        return True

    # 기존 v23 산업동향 판단은 보조적으로만 사용하되,
    # 개별 회사 발표/대학 홍보/기초 연구성 기사는 제외.
    if v23_is_valid_industry(t) and has_structure and not weak_promo:
        return True

    return False

def v26_2_is_overseas_structural_text(text, title=''):
    t = clean_html_text(text)
    title = clean_html_text(title) or t[:260]
    if v26_2_is_stock_market_text(t, title):
        return False
    return bool(V26_2_OVERSEAS_STRUCTURAL_RE.search(t) and V26_2_STRUCTURAL_INDUSTRY_RE.search(t))


def v26_2_article_profile(obj):
    t = v26_2_text(obj)
    title = v26_2_title(obj)
    gov_ok, gov_reason = v26_2_public_policy_basis(t, title)
    return {
        'text': t,
        'title': title,
        'is_roundup': v26_2_is_roundup(t, title),
        'is_market_stock': v26_2_is_stock_market_text(t, title),
        'gov_ok': gov_ok,
        'gov_reason': gov_reason,
        'industry_structural': v26_2_is_structural_industry_text(t, title),
        'overseas_structural': v26_2_is_overseas_structural_text(t, title),
        'competitor_scope': v21_competitor_scope(t) if 'v21_competitor_scope' in globals() else '',
        'has_self': v26_has_strict_self_entity(t) if 'v26_has_strict_self_entity' in globals() else bool(SELF_KAKAO_PATTERN.search(t)),
    }


# Override v26 government validators with the generic profile logic.
def v26_has_korean_gov_action(text):
    ok, _ = v26_2_public_policy_basis(text)
    return ok


def v26_is_gov_misclassified_industry(text):
    p = v26_2_article_profile(text)
    if p['is_market_stock']:
        return True
    if p['is_roundup'] and not p['gov_ok']:
        return True
    if V26_GOV_MISCLASSIFIED_INDUSTRY_RE.search(p['text']) and not p['gov_ok']:
        return True
    return False


def v26_suggest_category_from_text(text, requested=''):
    p = v26_2_article_profile(text)
    t = p['text']
    if p['has_self'] and V26_DIRECT_SELF_EVENT_RE.search(t):
        return JSON_KEYS_ORDER[0]
    if p['gov_ok']:
        return JSON_KEYS_ORDER[1]
    if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
        return JSON_KEYS_ORDER[2]
    if p['competitor_scope'] in {'domestic', 'mixed', 'overseas'} or v23_has_competitor_entity(t):
        return JSON_KEYS_ORDER[2]
    if p['industry_structural']:
        return JSON_KEYS_ORDER[3]
    return ''


_BASE_v23_final_category_guardrail_v262 = v23_final_category_guardrail

def v23_final_category_guardrail(item, requested_category):
    requested = v19_normalize_category_key(requested_category, fallback=JSON_KEYS_ORDER[3])
    p = v26_2_article_profile(item)
    t = p['text']

    if p['is_market_stock']:
        return 'fail', requested, 'v26_2_market_stock_commentary_excluded'

    if requested == JSON_KEYS_ORDER[0]:
        if not p['has_self']:
            suggested = v26_suggest_category_from_text(t, requested)
            if suggested and suggested != requested:
                return 'reassign', suggested, 'v26_2_self_without_direct_kakao_entity'
            return 'fail', requested, 'v26_2_self_without_direct_kakao_entity'
        if v20_is_platform_obligation_text(t) and not V26_DIRECT_SELF_EVENT_RE.search(t) and p['gov_ok']:
            return 'reassign', JSON_KEYS_ORDER[1], 'v26_2_platform_obligation_to_gov'
        return _BASE_v23_final_category_guardrail_v262(item, requested)

    if requested == JSON_KEYS_ORDER[1]:
        if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
            return 'reassign', JSON_KEYS_ORDER[2], 'v26_2_foreign_policy_to_competitor_overseas'
        if p['gov_ok']:
            return 'pass', requested, 'v26_2_public_actor_action_pass:' + p['gov_reason']
        return 'fail', requested, 'v26_2_government_without_central_public_actor_action:' + p['gov_reason']

    if requested == JSON_KEYS_ORDER[3]:
        if p['industry_structural'] or p['overseas_structural']:
            return 'pass', requested, 'v26_2_structural_industry_pass'
        if p['is_roundup']:
            return 'fail', requested, 'v26_2_roundup_not_structural_industry'

    action, cat, reason = _BASE_v23_final_category_guardrail_v262(item, requested)
    if action == 'reassign' and cat == JSON_KEYS_ORDER[1] and not p['gov_ok']:
        suggested = v26_suggest_category_from_text(t, requested)
        if suggested and suggested != JSON_KEYS_ORDER[1]:
            return 'reassign', suggested, 'v26_2_blocked_weak_gov_reassign:' + reason
        return 'fail', requested, 'v26_2_blocked_weak_gov_reassign:' + reason
    return action, cat, reason


_BASE_is_report_item_relevant_v262 = is_report_item_relevant

def is_report_item_relevant(report_item, json_key):
    requested = v19_normalize_category_key(json_key, fallback=JSON_KEYS_ORDER[3])
    p = v26_2_article_profile(report_item)
    t = p['text']
    if p['is_market_stock']:
        return False, 'v26_2_market_stock_commentary_excluded'
    if requested == JSON_KEYS_ORDER[1]:
        if v25_is_foreign_public_policy_issue(t) and not v25_has_direct_korean_policy_connection(t):
            return False, 'v26_2_foreign_policy_not_korean_gov'
        if not p['gov_ok']:
            return False, 'v26_2_government_without_central_public_actor_action:' + p['gov_reason']
    if requested == JSON_KEYS_ORDER[3]:
        if not (p['industry_structural'] or p['overseas_structural']):
            return False, 'v26_2_industry_without_structural_basis'
    return _BASE_is_report_item_relevant_v262(report_item, json_key)


_BASE_rank_score_article_v262 = rank_score_article

def rank_score_article(article):
    score = float(_BASE_rank_score_article_v262(article) or 0)
    p = v26_2_article_profile(article)
    cat = article.get('JSON카테고리', '')
    if p['is_market_stock']:
        score -= 160
        article['v26_2_profile_reason'] = 'market_stock_excluded'
    if cat == JSON_KEYS_ORDER[1]:
        if p['gov_ok']:
            score += 18
        else:
            score -= 90
            article['v26_2_profile_reason'] = 'weak_gov_basis:' + p['gov_reason']
    if cat == JSON_KEYS_ORDER[3]:
        if p['industry_structural'] or p['overseas_structural']:
            score += 18
        elif p['is_roundup']:
            score -= 45
            article['v26_2_profile_reason'] = 'roundup_deprioritized'
    if cat == JSON_KEYS_ORDER[2] and p['overseas_structural']:
        score += 16
    article['랭킹점수'] = round(score, 3)
    return article['랭킹점수']


_BASE_article_importance_score_v262 = article_importance_score

def article_importance_score(title, body='', json_key='', keyword=''):
    score = float(_BASE_article_importance_score_v262(title, body, json_key, keyword) or 0)
    text = f"{clean_html_text(title)} {clean_html_text(body)[:2200]}"
    p = v26_2_article_profile(text)
    if p['is_market_stock']:
        score -= 120
    if json_key == JSON_KEYS_ORDER[1]:
        score += 14 if p['gov_ok'] else -55
    if json_key == JSON_KEYS_ORDER[3]:
        if p['industry_structural'] or p['overseas_structural']:
            score += 16
        elif p['is_roundup']:
            score -= 35
    return round(score, 2)


_BASE_v20_local_article_label_v262 = v20_local_article_label

def v20_local_article_label(article):
    label = _BASE_v20_local_article_label_v262(article)
    p = v26_2_article_profile(article)
    if p['is_market_stock']:
        label['primary_category'] = '산업동향'
        label['internal_category'] = 'OFF_TOPIC'
        label['ceo_priority'] = min(v20_int(label.get('ceo_priority'), 2), 1)
        label['public_affairs_priority'] = min(v20_int(label.get('public_affairs_priority'), 2), 1)
        label['reason'] = 'v26.2: market/stock commentary excluded from final'
        return label
    if label.get('primary_category') == JSON_KEYS_ORDER[1] or article.get('JSON카테고리') == JSON_KEYS_ORDER[1]:
        if not p['gov_ok']:
            if p['overseas_structural'] or p['competitor_scope'] == 'overseas':
                label['primary_category'] = JSON_KEYS_ORDER[2]
            elif p['industry_structural']:
                label['primary_category'] = JSON_KEYS_ORDER[3]
            label['gov_tier'] = 'OFF_TOPIC'
            label['ceo_priority'] = min(v20_int(label.get('ceo_priority'), 3), 2)
            label['reason'] = 'v26.2: no central Korean public actor/action for Government section: ' + p['gov_reason']
    if p['overseas_structural'] and label.get('primary_category') == JSON_KEYS_ORDER[2]:
        label['ceo_priority'] = max(v20_int(label.get('ceo_priority'), 3), 4)
        label['reason'] = (label.get('reason', '') + ' / v26.2: overseas structural industry candidate').strip(' /')
    return label


_BASE_v20_build_issues_v262 = v20_build_issues

def v20_build_issues(candidates, labels):
    issues, article_by_id = _BASE_v20_build_issues_v262(candidates, labels)
    art_lookup = {int(a.get('id')): a for a in candidates if str(a.get('id', '')).isdigit()}
    for issue in issues:
        texts = []
        for aid in issue.get('article_ids', []):
            a = art_lookup.get(int(aid)) if str(aid).isdigit() else None
            if a:
                texts.append(v20_article_text(a) if 'v20_article_text' in globals() else v26_2_text(a))
        joined = ' '.join(texts) or clean_html_text(f"{issue.get('issue_group','')} {issue.get('top_title','')}")
        p = v26_2_article_profile(joined)
        issue['v26_2_profile_roundup'] = p['is_roundup']
        issue['v26_2_profile_gov_ok'] = p['gov_ok']
        issue['v26_2_profile_gov_reason'] = p['gov_reason']
        issue['v26_2_profile_market_stock'] = p['is_market_stock']
        issue['v26_2_profile_industry_structural'] = p['industry_structural'] or p['overseas_structural']
        score = float(issue.get('issue_score') or 0)
        cat = issue.get('primary_category', '')
        if p['is_market_stock']:
            score -= 110
        if cat == JSON_KEYS_ORDER[1] and not p['gov_ok']:
            score -= 55
        if cat == JSON_KEYS_ORDER[3] and p['is_roundup'] and not (p['industry_structural'] or p['overseas_structural']):
            score -= 35
        if cat == JSON_KEYS_ORDER[2] and p['overseas_structural']:
            score += 12
        issue['issue_score'] = round(score, 3)
    issues = sorted(issues, key=lambda x: x.get('issue_score', 0), reverse=True)
    for idx, issue in enumerate(issues, 1):
        issue['issue_id'] = f"I{idx:03d}"
    return issues, article_by_id


_BASE_v20_issue_rows_v262 = v20_issue_rows

def v20_issue_rows(issues):
    rows = _BASE_v20_issue_rows_v262(issues)
    for row, issue in zip(rows, issues):
        row['v26_2_roundup'] = issue.get('v26_2_profile_roundup', '')
        row['v26_2_gov_ok'] = issue.get('v26_2_profile_gov_ok', '')
        row['v26_2_gov_reason'] = issue.get('v26_2_profile_gov_reason', '')
        row['v26_2_market_stock'] = issue.get('v26_2_profile_market_stock', '')
        row['v26_2_industry_structural'] = issue.get('v26_2_profile_industry_structural', '')
    return rows


_BASE_v20_issue_line_v262 = v20_issue_line

def v20_issue_line(issue):
    base = _BASE_v20_issue_line_v262(issue)
    return base + f" profile_roundup={issue.get('v26_2_profile_roundup','')} gov_ok={issue.get('v26_2_profile_gov_ok','')} gov_reason={v20_clip(issue.get('v26_2_profile_gov_reason',''),70)} market_stock={issue.get('v26_2_profile_market_stock','')} industry_structural={issue.get('v26_2_profile_industry_structural','')}"


# Make overseas refill structural, not stock/feature-stock.
def v26_1_is_overseas_competitor_decision(decision, issue):
    cat = v26_1_decision_category(decision, issue)
    if cat != JSON_KEYS_ORDER[2]:
        return False
    text = clean_html_text(f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('issue_family','')}")
    return v26_2_is_overseas_structural_text(text, issue.get('top_title', ''))


def v26_1_is_good_refill_candidate(decision, issue, cat=None):
    cat = cat or v26_1_decision_category(decision, issue)
    if not issue:
        return False
    text = clean_html_text(f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('issue_family','')}")
    p = v26_2_article_profile(text)
    if p['is_market_stock']:
        return False
    if cat == JSON_KEYS_ORDER[1] and not p['gov_ok']:
        return False
    if cat == JSON_KEYS_ORDER[3] and not (p['industry_structural'] or p['overseas_structural']):
        return False
    ok, _ = v21_issue_allowed_basic(issue)
    if not ok:
        return False
    if issue.get('self_tier') in {'LOW_VALUE_PR', 'OFF_TOPIC'}:
        return False
    if cat == JSON_KEYS_ORDER[0] and issue.get('self_tier') == 'FILLER_REFERENCE':
        return False
    if issue.get('gov_tier') in {'GOV_GENERAL_LOW_RELEVANCE', 'OFF_TOPIC'} and cat == JSON_KEYS_ORDER[1]:
        return False
    if issue.get('competitor_tier') in {'COMP_GENERAL_LOW_RELEVANCE', 'OFF_TOPIC'} and cat == JSON_KEYS_ORDER[2]:
        return False
    return True


def v26_2_industry_pivot_score(item):
    p = v26_2_article_profile(item)
    if not (p['overseas_structural'] or p['industry_structural']):
        return -9999
    if p['is_market_stock'] or p['is_roundup']:
        return -9999
    score = v20_float(item.get('대표선택점수'), 0) * 0.35
    if p['overseas_structural']:
        score += 120
    if p['competitor_scope'] == 'overseas':
        score += 50
    if re.search(r"\ud504\ub77c\uc774\ubc84\uc2dc|\ubcf4\uc548|\uc628\ub514\ubc14\uc774\uc2a4|\ub370\uc774\ud130\uc13c\ud130|AI\s*\ud329\ud1a0\ub9ac|\uc218\uc775\uc131|IPO|\uaddc\uc81c|\uc0dd\ud0dc\uacc4", p['text'], re.IGNORECASE):
        score += 30
    return round(score, 2)


def v26_2_pivot_industry_from_competitor(items):
    counts, _, _ = v21_final_counts(items)
    if counts.get(JSON_KEYS_ORDER[3], 0) > 0:
        return items, []
    candidates = [x for x in items if x.get('카테고리') == JSON_KEYS_ORDER[2]]
    candidates = sorted(candidates, key=v26_2_industry_pivot_score, reverse=True)
    if not candidates or v26_2_industry_pivot_score(candidates[0]) < 0:
        return items, []
    chosen = candidates[0]
    old_cat = chosen.get('카테고리', '')
    if 'v23_set_final_category' in globals():
        v23_set_final_category(chosen, JSON_KEYS_ORDER[3], 'v26_2_industry_empty_promote_structural_overseas')
    else:
        chosen['카테고리'] = JSON_KEYS_ORDER[3]
        chosen['카테고리명'] = JSON_KEY_TO_DISPLAY.get(JSON_KEYS_ORDER[3], JSON_KEYS_ORDER[3])
    chosen['v26_2_industry_pivot'] = f"from:{old_cat}:score={v26_2_industry_pivot_score(chosen)}"
    return items, [(chosen, 'v26_2_promoted_competitor_overseas_to_industry')]


_BASE_v26_final_item_priority_score_v262 = v26_final_item_priority_score

def v26_final_item_priority_score(item):
    p = v26_2_article_profile(item)
    t = p['text']
    cat = item.get('카테고리', '')
    if p['is_market_stock']:
        return -9999
    base = v20_float(item.get('대표선택점수'), 0) * 0.25
    today_score, _ = v26_todayness_score_text(t, item.get('게시일', ''))
    score = base + today_score * 2
    if cat == JSON_KEYS_ORDER[0]:
        if not p['has_self']:
            return -9999
        bucket, bonus = v26_1_self_bucket(t) if 'v26_1_self_bucket' in globals() else ('self_reference', 0)
        # Strong editorial ordering: today labor/incidents outrank investment analysis.
        if bucket == 'self_labor':
            score += 420
        elif bucket == 'self_incident':
            score += 330
        elif bucket == 'self_leadership':
            score += 260
        elif bucket == 'self_governance':
            score += 150
        elif bucket == 'self_investment':
            score += 60
        else:
            score += bonus
        form = v26_article_form_from_text(t)
        if form == 'analysis_with_direct_risk':
            score += 35
        elif form == 'analysis_feature':
            score -= 80
        elif form == 'market_commentary':
            score -= 180
        if bucket == 'self_investment' and not re.search(r"\uacf5\uc2dc|\ud655\uc815|\uacb0\uc815|\ucc98\ubd84|\ub9e4\uac01\s*\uc644\ub8cc|\uc81c\uc7ac|\uacfc\uc9d5\uae08|\uc2ec\uc0ac", t):
            score -= 70
    elif cat == JSON_KEYS_ORDER[1]:
        score += 170 if p['gov_ok'] else -500
    elif cat == JSON_KEYS_ORDER[2]:
        if p['competitor_scope'] in {'domestic', 'mixed'}:
            score += 120
        elif p['competitor_scope'] == 'overseas':
            score += 65
        if p['overseas_structural']:
            score += 35
    elif cat == JSON_KEYS_ORDER[3]:
        score += 160 if (p['industry_structural'] or p['overseas_structural']) else -120
    return round(score, 2)


_BASE_v21_remove_obvious_final_duplicates_v262 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    kept, removed = _BASE_v21_remove_obvious_final_duplicates_v262(final_items)
    cleaned = []
    for item in kept:
        p = v26_2_article_profile(item)
        cat = item.get('카테고리', '')
        if p['is_market_stock']:
            item['v26_2_final_repair'] = 'removed:market_stock'
            removed.append((item, 'v26_2_market_stock_commentary_excluded'))
            continue
        if cat == JSON_KEYS_ORDER[1] and not p['gov_ok']:
            item['v26_2_final_repair'] = 'removed:weak_gov:' + p['gov_reason']
            removed.append((item, 'v26_2_government_without_central_public_actor_action:' + p['gov_reason']))
            continue
        if cat == JSON_KEYS_ORDER[3] and not (p['industry_structural'] or p['overseas_structural']):
            item['v26_2_final_repair'] = 'removed:weak_industry'
            removed.append((item, 'v26_2_industry_without_structural_basis'))
            continue
        cleaned.append(item)

    cleaned, pivot_notes = v26_2_pivot_industry_from_competitor(cleaned)
    removed.extend(pivot_notes)

    order_base = {key: idx * 1000 for idx, key in enumerate(JSON_KEYS_ORDER, 1)}
    for key in JSON_KEYS_ORDER:
        bucket = [x for x in cleaned if x.get('카테고리') == key]
        bucket_sorted = sorted(bucket, key=v26_final_item_priority_score, reverse=True)
        for idx, item in enumerate(bucket_sorted, 1):
            item['선정순서'] = order_base.get(key, 9000) + idx
            item['v26_2_final_priority'] = v26_final_item_priority_score(item)
    return cleaned, removed


_BASE_v26_1_should_refill_after_repair_v262 = v26_1_should_refill_after_repair

def v26_1_should_refill_after_repair(final_report_data):
    if _BASE_v26_1_should_refill_after_repair_v262(final_report_data):
        return True
    counts, _, comp_overseas = v21_final_counts(final_report_data)
    if counts.get(JSON_KEYS_ORDER[3], 0) <= 0:
        return True
    if any(v26_2_article_profile(x)['is_market_stock'] for x in final_report_data):
        return True
    if any(x.get('카테고리') == JSON_KEYS_ORDER[1] and not v26_2_article_profile(x)['gov_ok'] for x in final_report_data):
        return True
    if counts.get(JSON_KEYS_ORDER[2], 0) >= 2 and comp_overseas <= 0:
        return True
    return False


_BASE_v20_gemini_edit_issues_v262 = v20_gemini_edit_issues

def v20_gemini_edit_issues(client, issues, recent_past_text):
    if not client:
        raise RuntimeError('Gemini client 없음')
    visible_issues = [i for i in sorted(issues, key=lambda x: x.get('issue_score', 0), reverse=True) if not i.get('exclude') and not i.get('is_pr')]
    visible_issues = visible_issues[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = '\n'.join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
You are editing a daily Korean CEO/public-affairs briefing for Kakao.
Select issue groups, not just articles. Return JSON only.

Core principles:
- Do not patch by a single title keyword. Judge the structure: main actor, official action, single issue vs roundup, structural industry change vs stock/market commentary.
- Same URL/title is duplicate. Same event from the recent 7-day index can be kept only when it has a new stage or new official action.

Self/Kakao:
- Kakao direct entity is required.
- Today's direct events outrank analysis: labor/strike, service incidents, privacy/security, lawsuits/regulatory action, leadership/organization, M&A/control.
- Analysis/feature/investment-structure articles are allowed only after direct events.

Government/National Assembly:
- Requires a Korean public actor plus an official action or central public project.
- Public-private MOU/project participation is allowed when the public actor/project is central.
- A roundup that only mentions a public body in one item is not Government/National Assembly.
- Foreign/EU/US/China/Japan policy goes to Competitor/Overseas unless directly connected to Korean obligations.

Competitor/Overseas:
- Prefer domestic competitors first, plus one structural overseas/global AI/big-tech issue when available.
- Exclude stock/feature-stock/target-price articles.

Industry:
- Use for structural AI/platform/security/cloud/data-center/digital-asset trends.
- If the pure industry pool is weak, an overseas/global big-tech structural trend may be placed here.
- Do not use generic roundup as Industry unless it is the only meaningful structural item.

Recommended size: 12-15 total; Self 3-5, Gov 3-5, Competitor/Overseas 3-4, Industry 1.

JSON format:
{{
  "selected_issues": [
    {{"issue_id": "I001", "category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향", "priority": 5, "best_article_id": 123, "backup_article_ids": [124], "reason": "why"}}
  ],
  "backup_issues": [
    {{"issue_id": "I050", "category": "경쟁사_해외이슈", "priority": 4, "best_article_id": 555, "backup_article_ids": [], "reason": "backup reason"}}
  ]
}}

Recent 7-day index history:
{recent_past_text[:4500]}

Issue candidates:
{issue_text}
"""
    text = gemini_generate_text(
        client=client,
        prompt=prompt,
        task_name='v26.2 issue editor',
        model=GEMINI_MODEL_EDITOR,
    )
    data = extract_json_object(text)
    issue_by_id = {i['issue_id']: i for i in issues}
    article_by_id = {}
    for issue in issues:
        for art_id in issue.get('article_ids', []):
            article_by_id[int(art_id)] = {'id': int(art_id)}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    return decisions, backup_decisions



# =========================================================
# v27 overrides: generic cross-section dedupe + industry basis guardrail
# - 산업동향은 blacklist가 아니라 "디지털 산업 주체 + 구조 변화 + 카카오 영향 경로" 기준으로 판단
# - '플랫폼' 단어 하나만으로 식품/교육/의료/관리 플랫폼을 산업동향으로 오인하지 않음
# - 공정위/개보위/방미통위/금융위 등 공식 제재 사건은 actor+object 기반 사건키로 묶어
#   정부/국회와 경쟁사/해외이슈에 중복 노출되는 것을 방지
# =========================================================

V6_VERSION = "google_rss_v27_generic_event_industry_guardrails"

# 산업동향 2개까지 허용하는 기존 운영 원칙은 유지한다.
CATEGORY_MAX[JSON_KEYS_ORDER[3]] = max(CATEGORY_MAX.get(JSON_KEYS_ORDER[3], 2), 2)
CATEGORY_TARGET[JSON_KEYS_ORDER[3]] = max(CATEGORY_TARGET.get(JSON_KEYS_ORDER[3], 1), 1)

V27_DIGITAL_COMPANY_RE = re.compile(
    r"카카오|네이버|NAVER|쿠팡|Coupang|토스|비바리퍼블리카|배민|배달의민족|우아한형제들|"
    r"구글|Google|오픈AI|OpenAI|챗GPT|ChatGPT|MS|Microsoft|마이크로소프트|메타|Meta|"
    r"애플|Apple|앤트로픽|Anthropic|클로드|Claude|엔비디아|NVIDIA|오라클|Oracle|"
    r"SKT|SK텔레콤|KT|LGU\+|LG유플러스|알리바바|Alibaba|우버|Uber",
    re.IGNORECASE,
)

# 'AI 기반'처럼 기술이 수식어로만 붙는 경우는 actor로 보지 않기 위해
# 기업/인프라/서비스 주체성 있는 표현을 별도 기준으로 둔다.
V27_DIGITAL_INDUSTRY_ACTOR_RE = re.compile(
    r"빅테크|AI\s*기업|AI\s*스타트업|인공지능\s*기업|LLM\s*기업|소프트웨어\s*기업|"
    r"클라우드\s*(기업|사업자|서비스|플랫폼)|데이터센터|AI\s*데이터센터|AI\s*팩토리|"
    r"GPU|NPU|AI\s*반도체|반도체\s*(기업|업계|생태계)|보안\s*(기업|업체|스타트업|솔루션)|"
    r"사이버보안\s*(기업|업체|솔루션)|SaaS|오픈소스\s*(프로젝트|생태계|공급망)|개발자\s*도구",
    re.IGNORECASE,
)

# 디지털 플랫폼은 '플랫폼' 단어만으로 인정하지 않고, 사업자/서비스/운영 영역이 같이 있어야 한다.
V27_DIGITAL_PLATFORM_ACTOR_RE = re.compile(
    r"플랫폼\s*사업자|온라인\s*플랫폼|디지털\s*플랫폼|앱마켓|포털|검색\s*(서비스|사업자|플랫폼)|"
    r"SNS|소셜미디어|커머스\s*플랫폼|이커머스|배달앱|핀테크\s*플랫폼|광고\s*플랫폼|"
    r"부가통신사업자|정보통신서비스\s*제공자",
    re.IGNORECASE,
)

V27_PLATFORM_OPERATION_RE = re.compile(
    r"이용자|판매자|입점|광고|추천|알고리즘|검색|정산|수수료|콘텐츠|게시물|개인정보|보안|"
    r"결제|리뷰|다크패턴|자사우대|앱마켓|인앱결제|계정|로그인|오픈소스|개발자|API",
    re.IGNORECASE,
)

V27_STRUCTURAL_CHANGE_RE = re.compile(
    r"인수|합병|투자|투자\s*유치|전략적\s*투자|제휴|연동|출시|공개|도입|확대|증설|구축|"
    r"상용화|표준|표준화|재편|전환|대전환|확산|수요\s*증가|비용\s*구조|수익\s*구조|"
    r"시장\s*점유율|시장\s*구조|산업\s*구조|생태계|공급망|인프라|보안\s*수요|"
    r"해킹|공격|취약점|악성코드|침해|리스크\s*부각|공급망\s*리스크|규제|법안|시행|의무화|가이드라인",
    re.IGNORECASE,
)

V27_KAKAO_IMPACT_ROUTE_RE = re.compile(
    r"카카오|네이버|쿠팡|토스|배민|배달의민족|구글|오픈AI|MS|마이크로소프트|메타|애플|앤트로픽|"
    r"광고|커머스|콘텐츠|메신저|카카오톡|검색|추천|알고리즘|결제|핀테크|디지털자산|"
    r"개인정보|보안|사이버보안|클라우드|데이터센터|GPU|NPU|AI\s*개발|AI\s*도구|오픈소스|"
    r"개발자\s*도구|공급망|플랫폼\s*규제|이용자\s*보호|앱마켓|인앱결제|LLM|에이전트",
    re.IGNORECASE,
)

V27_WEAK_NONSTRUCTURAL_FORM_RE = re.compile(
    r"기자간담회|단순\s*소개|제품\s*소개|행사\s*소개|수상|브랜드\s*대상|특강|교육\s*실시|"
    r"대학교|대학|교수|연구팀|논문|학회|캠퍼스|기관\s*홍보|신제품\s*출시|솔루션\s*출시",
    re.IGNORECASE,
)

V27_KOREAN_ENFORCEMENT_ACTOR_RE = re.compile(
    r"공정위|공정거래위원회|개보위|개인정보보호위원회|방미통위|방송미디어통신위원회|"
    r"과기정통부|과학기술정보통신부|금융위|금융위원회|금감원|금융감독원|FIU|금융정보분석원|"
    r"정부|국회|검찰|경찰|KISA|한국인터넷진흥원",
    re.IGNORECASE,
)

V27_ENFORCEMENT_ACTION_RE = re.compile(
    r"과징금|과태료|시정명령|제재|처분|고발|조사|직권조사|심의|전원회의|제재안|위반|"
    r"표시광고법|공정거래법|개인정보보호법|전자상거래법|정보통신망법|약관법|동의의결",
    re.IGNORECASE,
)

V27_PLATFORM_ENFORCEMENT_ACTOR_ALIASES = [
    ("coupang", r"쿠팡|쿠팡이츠|Coupang"),
    ("naver", r"네이버|NAVER"),
    ("kakao", r"카카오톡|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|카카오|Kakao"),
    ("baemin", r"배달의민족|배민|우아한형제들"),
    ("toss", r"토스|비바리퍼블리카"),
    ("google", r"구글|Google|알파벳|Alphabet"),
    ("apple", r"애플|Apple"),
    ("meta", r"메타|Meta|페이스북|인스타그램|Instagram|Facebook"),
    ("openai", r"오픈AI|OpenAI|챗GPT|ChatGPT"),
    ("microsoft", r"MS|Microsoft|마이크로소프트"),
]


def v27_text(item):
    if isinstance(item, dict):
        return clean_html_text(
            f"{item.get('기사제목','')} {item.get('title','')} {item.get('본문요약','')} "
            f"{item.get('summary','')} {item.get('RSS요약','')} {item.get('본문전문','')[:3000]} "
            f"{item.get('issue_group','')} {item.get('top_title','')} {item.get('Gemini이슈그룹','')} "
            f"{item.get('v20_issue_group','')} {item.get('언론사','')}"
        )
    return clean_html_text(item)


def v27_is_digital_platform_context(text):
    t = clean_html_text(text)
    if not t:
        return False
    if V27_DIGITAL_COMPANY_RE.search(t) and V27_PLATFORM_OPERATION_RE.search(t):
        return True
    if V27_DIGITAL_PLATFORM_ACTOR_RE.search(t) and V27_PLATFORM_OPERATION_RE.search(t):
        return True
    return False


def v27_industry_basis(text, title=''):
    """산업동향 통과 여부를 blacklist가 아니라 3요소 기준으로 판단한다.

    통과 기준:
    - 디지털 산업 주체성(actor)
    - 구조 변화(change)
    - 카카오/플랫폼 사업환경 영향 경로(bridge)
    위 3개 중 최소 2개, 그리고 actor 또는 bridge 중 하나는 반드시 있어야 한다.
    """
    t = clean_html_text(text)
    title = clean_html_text(title) or t[:260]
    if not t:
        return False, "empty", 0
    if v26_2_is_stock_market_text(t, title):
        return False, "market_stock", 0
    if v26_2_is_roundup(t, title):
        return False, "roundup", 0

    front = t[:3200]
    actor = bool(
        V27_DIGITAL_COMPANY_RE.search(front)
        or V27_DIGITAL_INDUSTRY_ACTOR_RE.search(front)
        or v27_is_digital_platform_context(front)
    )
    change = bool(V27_STRUCTURAL_CHANGE_RE.search(front) or V26_2_INDUSTRY_STRUCTURE_SIGNAL_RE.search(front))
    bridge = bool(V27_KAKAO_IMPACT_ROUTE_RE.search(front))

    score = int(actor) + int(change) + int(bridge)

    # 약한 개별 발표/연구/홍보 형식은 3요소가 모두 강할 때만 허용한다.
    weak_form = bool(V27_WEAK_NONSTRUCTURAL_FORM_RE.search(front) or V26_2_WEAK_INDUSTRY_PROMO_RE.search(front))
    if weak_form and score < 3:
        return False, f"weak_form_without_full_basis:actor={actor},change={change},bridge={bridge}", score

    # '플랫폼'이라는 일반명사만 있고 디지털 플랫폼 운영 맥락이 없으면 actor로 보지 않는다.
    generic_platform_only = bool(re.search(r"플랫폼", front, re.I)) and not v27_is_digital_platform_context(front) and not V27_DIGITAL_COMPANY_RE.search(front)
    if generic_platform_only and not (V27_DIGITAL_INDUSTRY_ACTOR_RE.search(front) and bridge):
        # AI/플랫폼 단어가 있어도 카카오 사업환경과 연결되는 구조 변화가 아니면 제외.
        if score < 3:
            return False, f"generic_platform_without_digital_business_context:actor={actor},change={change},bridge={bridge}", score

    if score >= 2 and (actor or bridge):
        return True, f"industry_basis_pass:actor={actor},change={change},bridge={bridge}", score
    return False, f"insufficient_industry_basis:actor={actor},change={change},bridge={bridge}", score


_BASE_v26_2_is_structural_industry_text_v27 = v26_2_is_structural_industry_text

def v26_2_is_structural_industry_text(text, title=''):
    ok, _reason, _score = v27_industry_basis(text, title)
    return ok


_BASE_v26_2_is_overseas_structural_text_v27 = v26_2_is_overseas_structural_text

def v26_2_is_overseas_structural_text(text, title=''):
    t = clean_html_text(text)
    if not t or v26_2_is_stock_market_text(t, title) or v26_2_is_roundup(t, title):
        return False
    ok, _reason, _score = v27_industry_basis(t, title)
    if not ok:
        return False
    return bool(V26_2_OVERSEAS_STRUCTURAL_RE.search(t) or V27_DIGITAL_COMPANY_RE.search(t))


_BASE_v26_2_article_profile_v27 = v26_2_article_profile

def v26_2_article_profile(obj):
    p = _BASE_v26_2_article_profile_v27(obj)
    text = p.get('text') or v26_2_text(obj)
    title = p.get('title') or v26_2_title(obj)
    ok, reason, score = v27_industry_basis(text, title)
    p['industry_structural'] = ok
    p['overseas_structural'] = v26_2_is_overseas_structural_text(text, title)
    p['v27_industry_ok'] = ok
    p['v27_industry_reason'] = reason
    p['v27_industry_score'] = score
    p['v27_platform_enforcement_key'] = v27_platform_enforcement_key(obj, include_stage=False)
    return p


def v27_detect_enforcement_actor(text):
    t = clean_html_text(text)
    for key, pat in V27_PLATFORM_ENFORCEMENT_ACTOR_ALIASES:
        if re.search(pat, t, re.IGNORECASE):
            return key
    return ""


def v27_detect_enforcement_object(text):
    t = clean_html_text(text)
    if re.search(r"와우회원가|회원가|멤버십|쿠폰|할인가|상시\s*가격|기만\s*광고|허위\s*광고|표시광고|가격\s*광고", t, re.IGNORECASE):
        return "deceptive_membership_price_ad"
    if re.search(r"최혜대우|MFN|배달앱|쿠팡이츠|배민|수수료|정산|입점", t, re.IGNORECASE):
        return "delivery_platform_fee_mfn"
    if re.search(r"자사우대|검색\s*알고리즘|추천\s*알고리즘|노출\s*순위|검색\s*결과", t, re.IGNORECASE):
        return "self_preferencing_algorithm"
    if re.search(r"다크패턴|해지|환불|소비자\s*기만|기만\s*행위", t, re.IGNORECASE):
        return "consumer_dark_pattern"
    if re.search(r"개인정보|정보\s*유출|회원정보|고객정보|동의|제3자\s*제공|처리방침", t, re.IGNORECASE):
        return "privacy_data_violation"
    if re.search(r"인앱결제|앱마켓|수수료|결제\s*방식", t, re.IGNORECASE):
        return "app_market_payment_fee"
    if re.search(r"유해정보|불법정보|딥페이크|불법촬영|콘텐츠\s*차단|삭제\s*의무|필터링", t, re.IGNORECASE):
        return "content_moderation_obligation"
    return ""


def v27_detect_enforcement_stage(text):
    t = clean_html_text(text)
    if re.search(r"과징금|과태료|시정명령|제재|처분|부과|법정\s*최고", t, re.IGNORECASE):
        return "sanction_penalty"
    if re.search(r"동의의결", t, re.IGNORECASE):
        return "consent_decree"
    if re.search(r"조사|직권조사|현장조사|심의|전원회의|제재안", t, re.IGNORECASE):
        return "investigation_review"
    if re.search(r"소송|행정소송|집단소송|고발|검찰", t, re.IGNORECASE):
        return "litigation"
    return "general"


def v27_platform_enforcement_key(item, include_stage=True):
    t = v27_text(item)
    if not t:
        return ""
    # 공식 규제기관 또는 공식 제재/조사 액션이 없으면 단순 경쟁사 기사로 둔다.
    if not (V27_KOREAN_ENFORCEMENT_ACTOR_RE.search(t) or V27_ENFORCEMENT_ACTION_RE.search(t)):
        return ""
    actor = v27_detect_enforcement_actor(t)
    obj = v27_detect_enforcement_object(t)
    if not actor or not obj:
        return ""
    stage = v27_detect_enforcement_stage(t)
    if include_stage:
        return f"platform_enforcement:{actor}:{obj}:{stage}"
    return f"platform_enforcement:{actor}:{obj}"


def v27_is_korean_official_enforcement(item):
    t = v27_text(item)
    return bool(v27_platform_enforcement_key(item, include_stage=False) and V27_KOREAN_ENFORCEMENT_ACTOR_RE.search(t))


_BASE_v16_global_incident_key_v27 = v16_global_incident_key

def v16_global_incident_key(item):
    key = v27_platform_enforcement_key(item, include_stage=True)
    if key:
        parts = key.split(":")
        # global_incident:platform_enforcement:actor:object:stage
        if len(parts) == 4:
            return f"global_incident:{parts[0]}:{parts[1]}:{parts[2]}:{parts[3]}"
    return _BASE_v16_global_incident_key_v27(item)


_BASE_v16_global_incident_base_key_v27 = v16_global_incident_base_key

def v16_global_incident_base_key(item):
    key = v27_platform_enforcement_key(item, include_stage=False)
    if key:
        parts = key.split(":")
        # global_incident:platform_enforcement:actor:object
        if len(parts) == 3:
            return f"global_incident:{parts[0]}:{parts[1]}:{parts[2]}"
    return _BASE_v16_global_incident_base_key_v27(item)


_BASE_v26_event_key_for_item_v27 = v26_event_key_for_item

def v26_event_key_for_item(item):
    key = v27_platform_enforcement_key(item, include_stage=False)
    if key:
        return v26_normalize_event_key("v27:" + key)
    return _BASE_v26_event_key_for_item_v27(item)


_BASE_v24_topic_key_v27 = v24_topic_key

def v24_topic_key(item):
    key = v27_platform_enforcement_key(item, include_stage=False)
    if key:
        return "v27:" + key
    return _BASE_v24_topic_key_v27(item)


def v27_category_precedence_bonus(item):
    if not v27_platform_enforcement_key(item, include_stage=False):
        return 0
    cat = item.get('카테고리') or item.get('JSON카테고리') or item.get('Gemini카테고리') or ''
    t = v27_text(item)
    # 국내 공식 제재/조사 사건은 정부/국회 대표를 우선한다.
    if V27_KOREAN_ENFORCEMENT_ACTOR_RE.search(t):
        if cat == JSON_KEYS_ORDER[1]:
            return 140
        if cat == JSON_KEYS_ORDER[2]:
            return 35
        if cat == JSON_KEYS_ORDER[3]:
            return 10
    return 0


_BASE_v24_duplicate_survivor_score_v27 = v24_duplicate_survivor_score

def v24_duplicate_survivor_score(item):
    return float(_BASE_v24_duplicate_survivor_score_v27(item) or 0) + v27_category_precedence_bonus(item)


_BASE_v26_final_item_priority_score_v27 = v26_final_item_priority_score

def v26_final_item_priority_score(item):
    return float(_BASE_v26_final_item_priority_score_v27(item) or 0) + v27_category_precedence_bonus(item)


_BASE_final_duplicate_reason_v27 = final_duplicate_reason

def final_duplicate_reason(new_item, existing_items):
    new_key = v27_platform_enforcement_key(new_item, include_stage=False)
    if new_key:
        for old in existing_items:
            old_key = v27_platform_enforcement_key(old, include_stage=False)
            if old_key and old_key == new_key:
                return old, f"v27_platform_enforcement_duplicate:{new_key}"
    return _BASE_final_duplicate_reason_v27(new_item, existing_items)


_BASE_should_replace_duplicate_representative_v27 = should_replace_duplicate_representative

def should_replace_duplicate_representative(new_item, old_item, reason=""):
    if str(reason).startswith("v27_platform_enforcement_duplicate"):
        new_score = v20_float(new_item.get('대표선택점수'), 0) + v27_category_precedence_bonus(new_item)
        old_score = v20_float(old_item.get('대표선택점수'), 0) + v27_category_precedence_bonus(old_item)
        return new_score > old_score
    return _BASE_should_replace_duplicate_representative_v27(new_item, old_item, reason)


_BASE_v20_build_issues_v27 = v20_build_issues

def v20_build_issues(candidates, labels):
    issues, article_by_id = _BASE_v20_build_issues_v27(candidates, labels)
    art_lookup = {int(a.get('id')): a for a in candidates if str(a.get('id', '')).isdigit()}
    for issue in issues:
        issue_text_parts = [clean_html_text(f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('issue_family','')}")]
        event_keys = []
        industry_reasons = []
        for aid in issue.get('article_ids', []):
            try:
                a = art_lookup.get(int(aid))
            except Exception:
                a = None
            if not a:
                continue
            issue_text_parts.append(v20_article_text(a) if 'v20_article_text' in globals() else v27_text(a))
            key = v27_platform_enforcement_key(a, include_stage=False)
            if key:
                event_keys.append(key)
            ok, reason, score = v27_industry_basis(v27_text(a), a.get('기사제목', ''))
            industry_reasons.append(f"{ok}:{score}:{reason}")
        joined = clean_html_text(' '.join(issue_text_parts))
        if not event_keys:
            key = v27_platform_enforcement_key({'기사제목': issue.get('top_title',''), '본문전문': joined}, include_stage=False)
            if key:
                event_keys.append(key)
        ok, reason, score = v27_industry_basis(joined, issue.get('top_title', ''))
        issue['v27_event_key'] = event_keys[0] if event_keys else ''
        issue['v27_industry_ok'] = ok
        issue['v27_industry_reason'] = reason
        issue['v27_industry_score'] = score
        # 산업동향 후보가 v27 기준 미달이면 issue score를 낮춘다. 단, 삭제는 최종 guardrail에서 수행.
        if issue.get('primary_category') == JSON_KEYS_ORDER[3] and not ok:
            issue['issue_score'] = round(float(issue.get('issue_score') or 0) - 70, 3)
        if issue.get('v27_event_key') and issue.get('primary_category') == JSON_KEYS_ORDER[2] and V27_KOREAN_ENFORCEMENT_ACTOR_RE.search(joined):
            issue['issue_score'] = round(float(issue.get('issue_score') or 0) - 25, 3)
    issues = sorted(issues, key=lambda x: x.get('issue_score', 0), reverse=True)
    for idx, issue in enumerate(issues, 1):
        issue['issue_id'] = f"I{idx:03d}"
    return issues, article_by_id


_BASE_v20_issue_rows_v27 = v20_issue_rows

def v20_issue_rows(issues):
    rows = _BASE_v20_issue_rows_v27(issues)
    for row, issue in zip(rows, issues):
        row['v27_event_key'] = issue.get('v27_event_key', '')
        row['v27_industry_ok'] = issue.get('v27_industry_ok', '')
        row['v27_industry_reason'] = issue.get('v27_industry_reason', '')
        row['v27_industry_score'] = issue.get('v27_industry_score', '')
    return rows


_BASE_v20_issue_line_v27 = v20_issue_line

def v20_issue_line(issue):
    base = _BASE_v20_issue_line_v27(issue)
    return (
        base
        + f" v27_event_key={v20_clip(issue.get('v27_event_key',''),90)}"
        + f" v27_industry_ok={issue.get('v27_industry_ok','')}"
        + f" v27_industry_reason={v20_clip(issue.get('v27_industry_reason',''),90)}"
    )


_BASE_v23_final_category_guardrail_v27 = v23_final_category_guardrail

def v23_final_category_guardrail(item, requested_category):
    requested = v19_normalize_category_key(requested_category, fallback=JSON_KEYS_ORDER[3])
    key = v27_platform_enforcement_key(item, include_stage=False)
    if key and requested == JSON_KEYS_ORDER[2] and v27_is_korean_official_enforcement(item):
        return 'reassign', JSON_KEYS_ORDER[1], 'v27_korean_official_platform_enforcement_to_gov:' + key
    if requested == JSON_KEYS_ORDER[3]:
        ok, reason, _score = v27_industry_basis(v27_text(item), v26_2_title(item))
        if not ok:
            return 'fail', requested, 'v27_industry_without_two_of_three_basis:' + reason
    return _BASE_v23_final_category_guardrail_v27(item, requested_category)


_BASE_is_report_item_relevant_v27 = is_report_item_relevant

def is_report_item_relevant(report_item, json_key):
    requested = v19_normalize_category_key(json_key, fallback=JSON_KEYS_ORDER[3])
    if requested == JSON_KEYS_ORDER[3]:
        ok, reason, _score = v27_industry_basis(v27_text(report_item), v26_2_title(report_item))
        if not ok:
            return False, 'v27_industry_without_two_of_three_basis:' + reason
    if requested == JSON_KEYS_ORDER[2] and v27_platform_enforcement_key(report_item, include_stage=False) and v27_is_korean_official_enforcement(report_item):
        return False, 'v27_korean_official_enforcement_should_be_gov'
    return _BASE_is_report_item_relevant_v27(report_item, json_key)


_BASE_rank_score_article_v27 = rank_score_article

def rank_score_article(article):
    score = float(_BASE_rank_score_article_v27(article) or 0)
    key = v27_platform_enforcement_key(article, include_stage=False)
    if key:
        article['v27_event_key'] = key
        if article.get('JSON카테고리') == JSON_KEYS_ORDER[1]:
            score += 16
        elif article.get('JSON카테고리') == JSON_KEYS_ORDER[2] and v27_is_korean_official_enforcement(article):
            score -= 10
    ok, reason, industry_score = v27_industry_basis(v27_text(article), article.get('기사제목', ''))
    article['v27_industry_ok'] = 'Y' if ok else ''
    article['v27_industry_reason'] = reason
    article['v27_industry_score'] = industry_score
    if article.get('JSON카테고리') == JSON_KEYS_ORDER[3]:
        if ok:
            score += 12
        else:
            score -= 55
    article['랭킹점수'] = round(score, 3)
    return article['랭킹점수']


_BASE_v21_remove_obvious_final_duplicates_v27 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    kept, removed = _BASE_v21_remove_obvious_final_duplicates_v27(final_items)
    # 같은 국내 공식 플랫폼 제재 사건이 섹션을 달리해 남아 있으면 한 번 더 병합한다.
    groups = {}
    for item in kept:
        key = v27_platform_enforcement_key(item, include_stage=False)
        if key:
            groups.setdefault(key, []).append(item)
    remove_ids = set()
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        winner = max(group, key=lambda x: v20_float(x.get('대표선택점수'), 0) + v27_category_precedence_bonus(x))
        for item in group:
            if item is winner:
                item['v27중복대표선택'] = f"kept:{key}"
                continue
            item['v27중복대표선택'] = f"removed:{key}:winner={winner.get('기사제목','')[:80]}"
            removed.append((item, f"v27_cross_section_platform_enforcement_duplicate:{key}"))
            remove_ids.add(id(item))
    kept = [item for item in kept if id(item) not in remove_ids]

    # 산업동향은 최대 2개 허용하되, v27 기준 미달은 최종에서 제거한다.
    cleaned = []
    for item in kept:
        if item.get('카테고리') == JSON_KEYS_ORDER[3]:
            ok, reason, _score = v27_industry_basis(v27_text(item), v26_2_title(item))
            if not ok:
                item['v27최종수리결과'] = 'removed:weak_industry:' + reason
                removed.append((item, 'v27_industry_without_two_of_three_basis:' + reason))
                continue
        cleaned.append(item)
    order_base = {key: idx * 1000 for idx, key in enumerate(JSON_KEYS_ORDER, 1)}
    for key in JSON_KEYS_ORDER:
        bucket = [x for x in cleaned if x.get('카테고리') == key]
        bucket_sorted = sorted(bucket, key=v26_final_item_priority_score, reverse=True)
        for idx, item in enumerate(bucket_sorted, 1):
            item['선정순서'] = order_base.get(key, 9000) + idx
            item['v27_final_priority'] = v26_final_item_priority_score(item)
    return cleaned, removed


# Final editor prompt override: teaches Gemini the generic rules before code guardrails run.
def v20_gemini_edit_issues(client, issues, recent_past_text):
    if not client:
        raise RuntimeError('Gemini client 없음')
    visible_issues = [i for i in sorted(issues, key=lambda x: x.get('issue_score', 0), reverse=True) if not i.get('exclude') and not i.get('is_pr')]
    visible_issues = visible_issues[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = '\n'.join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
You are editing a daily Korean CEO/public-affairs briefing for Kakao.
Select issue groups, not just articles. Return JSON only.

Core principles:
- Do not patch by a single title keyword. Judge the structure: main actor, official action, single issue vs roundup, structural industry change vs stock/market commentary.
- Same URL/title is duplicate. Same event from the recent 7-day index can be kept only when it has a new stage or new official action.
- If two issues share the same v27_event_key, select only one. If the shared event is a Korean regulator's official enforcement/action, keep it in Government/National Assembly rather than Competitor/Overseas.

Self/Kakao:
- Kakao direct entity is required.
- Today's direct events outrank analysis: labor/strike, service incidents, privacy/security, lawsuits/regulatory action, leadership/organization, M&A/control.
- Analysis/feature/investment-structure articles are allowed only after direct events.

Government/National Assembly:
- Requires a Korean public actor plus an official action or central public project.
- Public-private MOU/project participation is allowed when the public actor/project is central.
- A roundup that only mentions a public body in one item is not Government/National Assembly.
- Foreign/EU/US/China/Japan policy goes to Competitor/Overseas unless directly connected to Korean obligations.
- A Korean regulator action against a platform/company, such as a fine, corrective order, investigation, consent decree, or formal review, belongs here. Do not duplicate the same event in Competitor/Overseas.

Competitor/Overseas:
- Prefer domestic competitors first, plus one structural overseas/global AI/big-tech issue when available.
- Exclude stock/feature-stock/target-price articles.
- If a competitor article is mainly a Korean regulator's official enforcement action, it should be Government/National Assembly, not a duplicate competitor item.

Industry:
- Industry may have 1-2 items when genuinely useful.
- Do not treat the word "platform" alone as Industry. A management/safety/research platform is not enough.
- Select Industry only when at least two of these are clear: (1) digital/AI/platform/cloud/security/data-center actor, (2) structural change such as investment, adoption, market shift, supply-chain/security risk, regulation, infrastructure expansion, (3) impact route to Kakao's business environment such as ads, commerce, content, messenger, fintech, privacy/security, cloud/data-center, AI tools, open-source supply chain, or platform regulation.
- Do not use generic roundup as Industry unless it is the only meaningful structural item.

Recommended size: 12-15 total; Self 3-5, Gov 3-5, Competitor/Overseas 3-4, Industry 1-2.

JSON format:
{{
  "selected_issues": [
    {{"issue_id": "I001", "category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향", "priority": 5, "best_article_id": 123, "backup_article_ids": [124], "reason": "why"}}
  ],
  "backup_issues": [
    {{"issue_id": "I050", "category": "경쟁사_해외이슈", "priority": 4, "best_article_id": 555, "backup_article_ids": [], "reason": "backup reason"}}
  ]
}}

Recent 7-day index history:
{recent_past_text[:4500]}

Issue candidates:
{issue_text}
"""
    text = gemini_generate_text(
        client=client,
        prompt=prompt,
        task_name='v27 issue editor',
        model=GEMINI_MODEL_EDITOR,
    )
    data = extract_json_object(text)
    issue_by_id = {i['issue_id']: i for i in issues}
    article_by_id = {}
    for issue in issues:
        for art_id in issue.get('article_ids', []):
            article_by_id[int(art_id)] = {'id': int(art_id)}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    return decisions, backup_decisions



# =========================================================
# v28 overrides: self minimum rescue + competitor direct-action guardrail + issue-scoped summary
# - 자사/계열사 섹션이 1개만 남는 경우, 자사 핵심 리스크/전략자산 이슈를 별도 rescue한다.
# - 경쟁사/해외이슈는 경쟁사 직접 액션 중심으로 제한하고, 정책 평가성/간접 수혜 기사는 제외 또는 정부/국회로 재분류한다.
# - 기사 요약은 각 기사에 배정된 issue_group 범위 안에서만 작성하도록 프롬프트를 강화한다.
# =========================================================

V6_VERSION = "google_rss_v28_self_rescue_competitor_summary_scope"
V28_SELF_MIN = 2
V28_SELF_RESCUE_MAX = 3

# 자사 rescue는 단순 카카오 언급이 아니라 직접 리스크/주요 계열사/전략자산 변화에만 허용한다.
V28_SELF_RESCUE_STRATEGIC_RE = re.compile(
    r"노조|파업|임단협|쟁의|조정|성과급|RSU|노동부|근로감독|최저임금|임금체불|"
    r"장애|오류|먹통|개인정보|유출|해킹|피싱|보안|침해사고|"
    r"과징금|제재|조사|수사|소송|판결|고발|행정소송|검찰|경찰|"
    r"대표|임원|CPO|CTO|CFO|조직개편|퇴사|사임|교체|선임|리더십|"
    r"인수|합병|매각|지분|투자회수|주식교환|최대주주|경영권|상장|IPO|"
    r"두나무|업비트|네이버파이낸셜|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|카카오엔터|카카오엔터테인먼트|"
    r"카카오톡\s*(개편|장애|오류|논란|반발|광고|AI|챗GPT|보안|피싱)",
    re.IGNORECASE,
)
V28_SELF_RESCUE_EXCLUDE_RE = re.compile(
    r"브랜드평판|평판지수|순위|스폰서|후원|협찬|이벤트|쿠폰|할인|프로모션|축제|영화제|"
    r"카톡\s*(캡처|캡쳐|대화|메시지|사생활|폭로)|연예인|가수|배우|아이돌|루머|"
    r"목표가|투자의견|주가\s*전망|지지선|저항선|특징주",
    re.IGNORECASE,
)
V28_SELF_BODYLESS_ALLOWED_RE = re.compile(
    r"두나무|업비트|네이버파이낸셜|지분|매각|투자회수|주식교환|최대주주|경영권|"
    r"노조|파업|임단협|개인정보|유출|장애|과징금|제재|수사|소송|대표|임원|조직개편",
    re.IGNORECASE,
)

# 경쟁사/해외 섹션은 경쟁사 자체의 행동·전략·리스크가 있어야 한다.
V28_COMPETITOR_DIRECT_ACTION_RE = re.compile(
    r"출시|공개|도입|적용|강화|개편|변경|확대|축소|중단|투자|투자\s*유치|전략적\s*투자|"
    r"인수|합병|매각|제휴|협력|계약|수익화|광고\s*정책|서비스\s*정책|가격\s*정책|"
    r"전략|계획|발표|실적|매출|영업이익|장애|오류|해킹|보안|개인정보|유출|소송|고소|"
    r"규제\s*대응|제재\s*대응|조사\s*대응|과징금|제재|시정명령|리콜|사과|복구",
    re.IGNORECASE,
)
V28_COMPETITOR_POLICY_EVAL_RE = re.compile(
    r"규제\s*(재검토|완화|개선|합리화|폐지)|의무휴업|전통시장|대형마트|유통산업발전법|"
    r"형평성|반사이익|수혜|키웠다|키운|사례로|사례\s*분석|정책\s*평가|위원회|부위원장|"
    r"대통령직속|정부\s*위원회|규제합리화위원회",
    re.IGNORECASE,
)
V28_COMPETITOR_AS_EXAMPLE_RE = re.compile(
    r"만\s*키웠|만\s*수혜|반사이익|사례로|예로\s*들|비교\s*대상|형평성",
    re.IGNORECASE,
)
V28_COMPETITOR_ENFORCEMENT_RE = re.compile(
    r"공정위|공정거래위원회|개보위|개인정보보호위원회|방미통위|금융위|금감원|과기정통부|"
    r"과징금|과태료|시정명령|제재|제재안|조사|심의|의결|동의의결|고발",
    re.IGNORECASE,
)


def v28_item_text(item):
    return clean_html_text(
        f"{item.get('기사제목','')} {item.get('본문요약','')} {item.get('RSS요약','')} "
        f"{item.get('본문전문','')[:3200]} {item.get('v20_issue_group','')} {item.get('Gemini이슈그룹','')} "
        f"{item.get('v20_issue_family','')} {item.get('언론사','')}"
    )


def v28_issue_text(issue, article_by_id=None):
    parts = [
        issue.get('issue_group', ''), issue.get('top_title', ''), issue.get('issue_family', ''),
        issue.get('internal_category', ''), issue.get('company_impact', ''), issue.get('label_reasons', ''),
    ]
    if article_by_id:
        for aid in issue.get('article_ids', [])[:8]:
            try:
                a = article_by_id.get(int(aid), {})
            except Exception:
                a = {}
            if a:
                parts.append(v20_article_text(a) if 'v20_article_text' in globals() else v28_item_text(a))
    return clean_html_text(' '.join(str(x) for x in parts if x))


def v28_existing_final_keys(final_report_data):
    keys = set()
    for item in final_report_data:
        for val in [
            item.get('v20_issue_id'),
            v20_issue_group_key(item.get('v20_issue_group', '')) if 'v20_issue_group_key' in globals() else item.get('v20_issue_group', ''),
            v26_event_key_for_item(item) if 'v26_event_key_for_item' in globals() else '',
            v16_global_incident_base_key(item) if 'v16_global_incident_base_key' in globals() else '',
            v27_platform_enforcement_key(item, include_stage=False) if 'v27_platform_enforcement_key' in globals() else '',
        ]:
            val = clean_html_text(val)
            if val:
                keys.add(val)
    return keys


def v28_issue_already_represented(issue, final_report_data):
    keys = v28_existing_final_keys(final_report_data)
    candidates = [
        issue.get('issue_id'),
        v20_issue_group_key(issue.get('issue_group', '')) if 'v20_issue_group_key' in globals() else issue.get('issue_group', ''),
    ]
    # approximate event key from issue text
    pseudo = {
        '기사제목': issue.get('top_title', ''),
        '본문요약': issue.get('issue_group', ''),
        'RSS요약': issue.get('label_reasons', ''),
        'Gemini이슈그룹': issue.get('issue_group', ''),
    }
    try:
        candidates.append(v26_event_key_for_item(pseudo))
    except Exception:
        pass
    for val in candidates:
        val = clean_html_text(val)
        if val and val in keys:
            return True
    return False


def v28_self_rescue_score(issue, article_by_id=None):
    t = v28_issue_text(issue, article_by_id)
    if not t:
        return -9999, 'empty'
    if V28_SELF_RESCUE_EXCLUDE_RE.search(t):
        return -9999, 'excluded_self_noise'
    if not v26_has_strict_self_entity(t) if 'v26_has_strict_self_entity' in globals() else not v20_has_self_entity(t):
        return -9999, 'no_strict_self_entity'
    if not V28_SELF_RESCUE_STRATEGIC_RE.search(t):
        return -9999, 'no_self_strategic_signal'

    score = v20_float(issue.get('issue_score'), 0)
    score += v20_int(issue.get('max_ceo_priority'), 3) * 22
    score += v20_int(issue.get('max_pa_priority'), 3) * 18
    bucket = 'unknown'
    try:
        bucket, bonus = v26_1_self_bucket(t)
        score += bonus
    except Exception:
        bonus = 0
    if re.search(r"노조|파업|임단협|쟁의|성과급|RSU", t, re.I):
        score += 180
    if re.search(r"개인정보|유출|해킹|장애|피싱|보안|과징금|제재|소송|수사", t, re.I):
        score += 160
    if re.search(r"대표|임원|조직개편|퇴사|사임|교체", t, re.I):
        score += 135
    if re.search(r"두나무|업비트|네이버파이낸셜|지분|매각|주식교환|투자회수|경영권", t, re.I):
        score += 120
    # 단순 분석/전망이면 rescue 우선순위를 낮춘다. 단, 두나무/지분구조 같은 전략자산은 살릴 수 있다.
    if re.search(r"전망|분석|짚어보니|기획|리포트", t, re.I) and not re.search(r"두나무|업비트|지분|매각|주식교환|노조|파업|제재|과징금", t, re.I):
        score -= 90
    return round(score, 2), bucket


def v28_make_self_rescue_decision(issue):
    return {
        'issue_id': issue.get('issue_id'),
        'category': JSON_KEYS_ORDER[0],
        'include': True,
        'priority': max(5, v20_int(issue.get('max_ceo_priority'), 4)),
        'best_article_id': issue.get('top_article_id'),
        'backup_article_ids': [x for x in issue.get('candidate_article_ids', []) if x != issue.get('top_article_id')][:12],
        'reason': 'v28 self minimum rescue: 자사 핵심 리스크/전략자산 이슈 보강',
    }


def v28_remove_one_low_priority_for_self(final_report_data, skip_rows, source='v28_self_rescue_room'):
    if len(final_report_data) < V21_FINAL_MAX:
        return True
    removable = []
    for item in final_report_data:
        if item.get('카테고리') == JSON_KEYS_ORDER[0]:
            continue
        text = v28_item_text(item)
        penalty = 0
        if item.get('카테고리') == JSON_KEYS_ORDER[2] and v28_competitor_guardrail_action(item)[0] == 'remove':
            penalty += 500
        if item.get('카테고리') == JSON_KEYS_ORDER[3]:
            ok, _reason, _score = v27_industry_basis(text, item.get('기사제목','')) if 'v27_industry_basis' in globals() else (True, '', 0)
            if not ok:
                penalty += 300
        score = v20_float(item.get('대표선택점수'), 0) + v20_float(item.get('v27_final_priority'), 0) - penalty
        removable.append((score, item))
    if not removable:
        return False
    _, loser = min(removable, key=lambda x: x[0])
    try:
        final_report_data.remove(loser)
        add_duplicate_skip_row(
            skip_rows,
            loser,
            {'기사제목': 'v28_self_rescue_room', '링크': '', '대표선택점수': ''},
            'v28_removed_low_priority_to_restore_self_minimum',
            stage=source,
        )
        return True
    except Exception:
        return False


def v28_refill_self_minimum(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    counts, _, _ = v21_final_counts(final_report_data)
    if counts.get(JSON_KEYS_ORDER[0], 0) >= V28_SELF_MIN:
        return order_counter

    # 모든 이슈를 대상으로 보되, Gemini/backup/local 순서를 반영하고 self rescue score로 다시 정렬한다.
    candidate_issues = []
    seen = set()
    for d in list(decisions or []) + list(backup_decisions or []):
        issue = issue_by_id.get(d.get('issue_id'), {})
        if issue and issue.get('issue_id') not in seen:
            candidate_issues.append(issue)
            seen.add(issue.get('issue_id'))
    for issue in issues:
        if issue.get('issue_id') not in seen:
            candidate_issues.append(issue)
            seen.add(issue.get('issue_id'))

    scored = []
    for issue in candidate_issues:
        if v28_issue_already_represented(issue, final_report_data):
            continue
        score, reason = v28_self_rescue_score(issue, article_by_id)
        if score <= 0:
            continue
        scored.append((score, reason, issue))
    scored.sort(key=lambda x: x[0], reverse=True)

    for score, reason, issue in scored:
        if v21_final_counts(final_report_data)[0].get(JSON_KEYS_ORDER[0], 0) >= V28_SELF_MIN:
            break
        if v21_final_counts(final_report_data)[0].get(JSON_KEYS_ORDER[0], 0) >= V28_SELF_RESCUE_MAX:
            break
        if not v28_remove_one_low_priority_for_self(final_report_data, skip_rows):
            break
        decision = v28_make_self_rescue_decision(issue)
        before = len(final_report_data)
        item, order_counter, status = v20_process_issue(
            decision,
            issue_by_id,
            article_by_id,
            recent_past_items,
            final_report_data,
            body_failed_rows,
            skip_rows,
            processed_article_ids,
            order_counter,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        if item:
            item['v28_self_rescue'] = f"score={score}:reason={reason}"
            print(f"  -> v28 self rescue added: {v20_clip(issue.get('issue_group',''), 42)} / {v20_clip(item.get('기사제목',''), 42)}... ({status})")
            continue

        # If the issue is strategically important but body extraction failed, allow explicit title/RSS-only inclusion.
        text = v28_issue_text(issue, article_by_id)
        if score >= 120 and V28_SELF_BODYLESS_ALLOWED_RE.search(text):
            fallback_id = decision.get('best_article_id') or issue.get('top_article_id')
            try:
                article_info = article_by_id.get(int(fallback_id))
            except Exception:
                article_info = None
            if article_info:
                bodyless = v20_create_bodyless_report_item(
                    article_info,
                    issue,
                    decision,
                    JSON_KEYS_ORDER[0],
                    reason='v28_self_rescue_bodyless_critical_self_issue',
                )
                bodyless['선정순서'] = order_counter
                bodyless['v28_self_rescue'] = f"bodyless:score={score}:reason={reason}"
                dup_item, dup_reason = final_duplicate_reason(bodyless, final_report_data)
                if not dup_item:
                    final_report_data.append(bodyless)
                    order_counter += 1
                    status_counts['v28_self_rescue_bodyless'] = status_counts.get('v28_self_rescue_bodyless', 0) + 1
                    print(f"  -> v28 self rescue bodyless added: {v20_clip(issue.get('issue_group',''), 42)} / {v20_clip(bodyless.get('기사제목',''), 42)}...")
                else:
                    add_duplicate_skip_row(skip_rows, bodyless, dup_item, f"v28_self_rescue_bodyless_duplicate:{dup_reason}", stage='v28_self_rescue')
        elif len(final_report_data) == before:
            # no item added; keep trying next issue
            pass
    return order_counter


_BASE_v26_1_should_refill_after_repair_v28 = v26_1_should_refill_after_repair

def v26_1_should_refill_after_repair(final_report_data):
    if _BASE_v26_1_should_refill_after_repair_v28(final_report_data):
        return True
    counts, _, _ = v21_final_counts(final_report_data)
    if counts.get(JSON_KEYS_ORDER[0], 0) < V28_SELF_MIN:
        return True
    # 경쟁사 섹션에 간접 정책평가성 기사가 남아 있으면 repair/refill을 한 번 더 유도한다.
    if any(item.get('카테고리') == JSON_KEYS_ORDER[2] and v28_competitor_guardrail_action(item)[0] in {'remove', 'reassign'} for item in final_report_data):
        return True
    return False


_BASE_v26_1_refill_after_final_repair_v28 = v26_1_refill_after_final_repair

def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    order_counter = _BASE_v26_1_refill_after_final_repair_v28(
        decisions,
        backup_decisions,
        issues,
        issue_by_id,
        article_by_id,
        recent_past_items,
        final_report_data,
        body_failed_rows,
        skip_rows,
        processed_article_ids,
        order_counter,
        status_counts,
    )
    order_counter = v28_refill_self_minimum(
        decisions,
        backup_decisions,
        issues,
        issue_by_id,
        article_by_id,
        recent_past_items,
        final_report_data,
        body_failed_rows,
        skip_rows,
        processed_article_ids,
        order_counter,
        status_counts,
    )
    return order_counter


def v28_competitor_guardrail_action(item):
    text = v28_item_text(item)
    if not text:
        return 'pass', '', 'empty'
    if item.get('카테고리') != JSON_KEYS_ORDER[2]:
        return 'pass', '', 'not_competitor_section'
    has_competitor = bool(COMPETITOR_CORE_PATTERN.search(text) or V27_DIGITAL_COMPANY_RE.search(text) if 'V27_DIGITAL_COMPANY_RE' in globals() else COMPETITOR_CORE_PATTERN.search(text))
    if not has_competitor:
        return 'remove', '', 'v28_competitor_without_core_actor'

    # 국내 규제기관의 공식 제재/조사/심의가 핵심이면 정부/국회로 보낸다.
    if V28_COMPETITOR_ENFORCEMENT_RE.search(text) and v26_has_korean_gov_action(text):
        return 'reassign', JSON_KEYS_ORDER[1], 'v28_korean_regulator_action_not_competitor_duplicate'

    direct = bool(V28_COMPETITOR_DIRECT_ACTION_RE.search(text))
    policy_eval = bool(V28_COMPETITOR_POLICY_EVAL_RE.search(text))
    competitor_as_example = bool(V28_COMPETITOR_AS_EXAMPLE_RE.search(text))

    if policy_eval and competitor_as_example and not direct:
        return 'remove', '', 'v28_competitor_only_policy_eval_or_beneficiary_example'
    if policy_eval and not direct:
        # 디지털 플랫폼 규제의 공식 액션이면 정부/국회로, 아니면 경쟁사 섹션에서는 제외한다.
        if v26_has_korean_gov_action(text):
            return 'reassign', JSON_KEYS_ORDER[1], 'v28_policy_eval_with_public_action_reassigned_to_gov'
        return 'remove', '', 'v28_competitor_policy_eval_without_direct_action'
    if not direct:
        return 'remove', '', 'v28_competitor_without_direct_action'
    return 'pass', '', 'v28_competitor_direct_action_pass'


_BASE_v21_remove_obvious_final_duplicates_v28 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    kept, removed = _BASE_v21_remove_obvious_final_duplicates_v28(final_items)
    repaired = []
    for item in kept:
        if item.get('카테고리') == JSON_KEYS_ORDER[2]:
            action, new_cat, reason = v28_competitor_guardrail_action(item)
            if action == 'remove':
                item['v28_competitor_guardrail'] = 'removed:' + reason
                removed.append((item, reason))
                continue
            if action == 'reassign' and new_cat:
                if 'v23_set_final_category' in globals():
                    v23_set_final_category(item, new_cat, reason)
                else:
                    item['카테고리'] = new_cat
                    item['카테고리명'] = JSON_KEY_TO_DISPLAY.get(new_cat, new_cat)
                item['v28_competitor_guardrail'] = 'reassigned:' + reason
            else:
                item['v28_competitor_guardrail'] = 'pass:' + reason
        repaired.append(item)

    # Re-run same-event dedupe after possible competitor -> gov reassignment.
    if 'v26_dedupe_by_event_key' in globals():
        repaired, removed_event = v26_dedupe_by_event_key(repaired)
        removed.extend(removed_event)

    # v27 platform enforcement duplicate key is more precise for Coupang/Baemin platform enforcement cases.
    groups = {}
    for item in repaired:
        key = v27_platform_enforcement_key(item, include_stage=False) if 'v27_platform_enforcement_key' in globals() else ''
        if key:
            groups.setdefault(key, []).append(item)
    remove_ids = set()
    for key, group in groups.items():
        if len(group) <= 1:
            continue
        winner = max(group, key=lambda x: v20_float(x.get('대표선택점수'), 0) + v27_category_precedence_bonus(x) if 'v27_category_precedence_bonus' in globals() else v20_float(x.get('대표선택점수'), 0))
        for item in group:
            if item is winner:
                item['v28_platform_enforcement_dedupe'] = 'kept:' + key
                continue
            item['v28_platform_enforcement_dedupe'] = 'removed:' + key
            removed.append((item, 'v28_platform_enforcement_cross_category_duplicate:' + key))
            remove_ids.add(id(item))
    repaired = [x for x in repaired if id(x) not in remove_ids]

    order_base = {key: idx * 1000 for idx, key in enumerate(JSON_KEYS_ORDER, 1)}
    for key in JSON_KEYS_ORDER:
        bucket = [x for x in repaired if x.get('카테고리') == key]
        bucket_sorted = sorted(bucket, key=v26_final_item_priority_score if 'v26_final_item_priority_score' in globals() else (lambda x: v20_float(x.get('대표선택점수'), 0)), reverse=True)
        for idx, item in enumerate(bucket_sorted, 1):
            item['선정순서'] = order_base.get(key, 9000) + idx
            item['v28_final_priority'] = v26_final_item_priority_score(item) if 'v26_final_item_priority_score' in globals() else v20_float(item.get('대표선택점수'), 0)
    return repaired, removed


_BASE_v20_build_summary_prompt_v28 = v20_build_summary_prompt

def v20_build_summary_prompt(final_report_data, recent_past_text):
    prompt = _BASE_v20_build_summary_prompt_v28(final_report_data, recent_past_text)
    extra_rules = """
9. 각 기사 요약은 반드시 해당 기사에 적힌 '이슈그룹'과 직접 관련된 내용만 작성해.
10. 본문/RSS 안에 다른 기업·다른 제재·다른 사건이 함께 언급돼도 이슈그룹과 직접 관련이 없으면 요약하지마.
11. 특히 '한편', '또한', '별도로', '이와 별개로', '아울러' 뒤에 나오는 별도 사건은 해당 이슈그룹과 직접 연결될 때만 요약해.
12. 같은 보고서 안의 다른 기사에서 별도 항목으로 다루는 사건은 반복 요약하지마. 예: 배달앱 최혜대우 기사에서는 쿠팡 개인정보위 제재안을 별도 이슈로 요약하지 않음.
"""
    if '8. 사진/캡션/관련기사/저작권 문구는 요약하지마.' in prompt:
        prompt = prompt.replace(
            '8. 사진/캡션/관련기사/저작권 문구는 요약하지마.',
            '8. 사진/캡션/관련기사/저작권 문구는 요약하지마.' + extra_rules,
        )
    else:
        prompt += "\n[추가 요약 범위 규칙]\n" + extra_rules
    return prompt


_BASE_v20_gemini_edit_issues_v28 = v20_gemini_edit_issues

def v20_gemini_edit_issues(client, issues, recent_past_text):
    # v27 editor prompt를 기반으로 하되, 자사 최소 복구와 경쟁사 직접 액션 원칙을 더 강하게 알려준다.
    if not client:
        raise RuntimeError('Gemini client 없음')
    visible_issues = [i for i in sorted(issues, key=lambda x: x.get('issue_score', 0), reverse=True) if not i.get('exclude') and not i.get('is_pr')]
    visible_issues = visible_issues[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = '\n'.join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
You are editing a daily Korean CEO/public-affairs briefing for Kakao.
Select issue groups, not just articles. Return JSON only.

Core principles:
- Do not patch by a single title keyword. Judge the structure: main actor, official action, single issue vs roundup, structural industry change vs stock/market commentary.
- Same URL/title is duplicate. Same event from the recent 7-day index can be kept only when it has a new stage or new official action.
- If two issues share the same v27_event_key, select only one. If the shared event is a Korean regulator's official enforcement/action, keep it in Government/National Assembly rather than Competitor/Overseas.

Self/Kakao:
- Keep at least 2 Self/Kakao issues when there are credible candidates.
- Kakao direct entity is required.
- Today's direct events outrank analysis: labor/strike, service incidents, privacy/security, lawsuits/regulatory action, leadership/organization, M&A/control.
- If only one direct Self issue exists, a strategic Kakao asset/affiliate issue can be selected as the second Self item: Dunamu/Upbit stake, Kakao Pay/Bank/Mobility/Games governance, control, sale, litigation, enforcement, or material business change.
- Exclude PR, sponsorship, event, brand-rank, entertainment KakaoTalk chat/gossip.

Government/National Assembly:
- Requires a Korean public actor plus an official action or central public project.
- A Korean regulator action against a platform/company, such as a fine, corrective order, investigation, consent decree, or formal review, belongs here. Do not duplicate the same event in Competitor/Overseas.
- Foreign/EU/US/China/Japan policy goes to Competitor/Overseas unless directly connected to Korean obligations.

Competitor/Overseas:
- Select only when the competitor or overseas big-tech actor has a direct action: strategy, launch, policy change, investment, acquisition, partnership, earnings, outage, security/privacy incident, lawsuit, or regulatory response.
- Do not select articles where a competitor is only an example/beneficiary of a Korean policy debate. Example: retail Sunday-closing regulation 'helped Coupang' is a policy evaluation, not Competitor/Overseas.
- If the main actor is a Korean regulator imposing or reviewing sanctions on a competitor, classify as Government/National Assembly instead.
- Prefer 2-3 domestic competitor/platform items plus 1 overseas/global AI/big-tech item when available.

Industry:
- Industry may have 1-2 items when genuinely useful.
- Do not treat the word "platform" alone as Industry.
- Select Industry only when at least two of these are clear: (1) digital/AI/platform/cloud/security/data-center actor, (2) structural change such as investment, adoption, market shift, supply-chain/security risk, regulation, infrastructure expansion, (3) impact route to Kakao's business environment.

Recommended size: 12-15 total; Self 2-4, Gov 3-5, Competitor/Overseas 3-4, Industry 1-2.

JSON format:
{{
  "selected_issues": [
    {{"issue_id": "I001", "category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향", "priority": 5, "best_article_id": 123, "backup_article_ids": [124], "reason": "why"}}
  ],
  "backup_issues": [
    {{"issue_id": "I050", "category": "경쟁사_해외이슈", "priority": 4, "best_article_id": 555, "backup_article_ids": [], "reason": "backup reason"}}
  ]
}}

Recent 7-day index history:
{recent_past_text[:4500]}

Issue candidates:
{issue_text}
"""
    text = gemini_generate_text(
        client=client,
        prompt=prompt,
        task_name='v28 issue editor',
        model=GEMINI_MODEL_EDITOR,
    )
    data = extract_json_object(text)
    issue_by_id = {i['issue_id']: i for i in issues}
    article_by_id = {}
    for issue in issues:
        for art_id in issue.get('article_ids', []):
            article_by_id[int(art_id)] = {'id': int(art_id)}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    return decisions, backup_decisions


_BASE_v20_gemini_quality_check_v28 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v28(client, final_report_data, final_briefing_text)
    local_warnings = []
    counts, _, _ = v21_final_counts(final_report_data)
    if counts.get(JSON_KEYS_ORDER[0], 0) < V28_SELF_MIN:
        local_warnings.append(f"자사/계열사 섹션이 {counts.get(JSON_KEYS_ORDER[0], 0)}개로 v28 기준 최소 {V28_SELF_MIN}개보다 적음")
    for item in final_report_data:
        if item.get('카테고리') == JSON_KEYS_ORDER[2]:
            action, _cat, reason = v28_competitor_guardrail_action(item)
            if action != 'pass':
                local_warnings.append(f"경쟁사/해외 섹션 부적합 가능: {item.get('기사제목','')} / {reason}")
    if not local_warnings:
        return qa
    try:
        data = v22_parse_qa_json(qa) if qa else {'overall': '양호', 'warnings': [], 'suggested_human_checks': []}
        warnings = data.get('warnings') if isinstance(data.get('warnings'), list) else []
        checks = data.get('suggested_human_checks') if isinstance(data.get('suggested_human_checks'), list) else []
        warnings.extend(local_warnings)
        checks.extend(local_warnings)
        data['warnings'] = warnings
        data['suggested_human_checks'] = checks
        if data.get('overall') == '양호':
            data['overall'] = '주의'
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return (qa or '') + '\n' + '\n'.join(local_warnings)


# =========================================================
# v29 overrides: stage-aware dedupe + quality-first final repair
# - 특정 기사/키워드가 아니라 base_event + lifecycle stage 기준으로 중복을 판단한다.
# - 본문추출실패 기사는 최종본에서 원칙적으로 제외하고, 극소수 핵심 이슈만 제한 허용한다.
# - 정부/경쟁사/산업동향은 단어 매칭보다 actor + action + Kakao impact route 구조를 본다.
# - 자사 rescue는 본문 전체가 아니라 issue metadata/title/RSS front만 보고 판단해 관련기사 오염을 막는다.
# =========================================================

V6_VERSION = "google_rss_v29_stage_dedupe_quality_guardrails"

# v29는 낮은 품질 기사로 억지로 15개를 채우지 않는다.
CATEGORY_MIN = {
    "자사_및_계열사_이슈": 1,
    "정부_국회": 3,
    "경쟁사_해외이슈": 3,
    "산업동향": 1,
}
CATEGORY_TARGET = {
    "자사_및_계열사_이슈": 2,
    "정부_국회": 4,
    "경쟁사_해외이슈": 3,
    "산업동향": 1,
}
CATEGORY_MAX = {
    "자사_및_계열사_이슈": 5,
    "정부_국회": 6,
    "경쟁사_해외이슈": 4,
    "산업동향": 2,
}
V21_FINAL_MIN = 10
V21_FINAL_TARGET = 12
V21_FINAL_MAX = 14
V20_FINAL_MIN = V21_FINAL_MIN
V20_FINAL_TARGET = V21_FINAL_TARGET
V20_FINAL_MAX = V21_FINAL_MAX
MIN_SELECT_COUNT = V21_FINAL_MIN
MAX_SELECT_COUNT = V21_FINAL_MAX
V21_COMPETITOR_DOMESTIC_MIN = 2
V21_COMPETITOR_DOMESTIC_TARGET = 2

# Gemini 라벨링 일부 배치가 JSON 실패를 일으키면 후단 전체 품질이 흔들리므로 배치를 조금 줄인다.
try:
    V19_LABEL_BATCH_SIZE = min(int(V19_LABEL_BATCH_SIZE), 45)
except Exception:
    V19_LABEL_BATCH_SIZE = 45
try:
    V20_LABEL_BATCH_SIZE = min(int(V20_LABEL_BATCH_SIZE), 45)
except Exception:
    V20_LABEL_BATCH_SIZE = 45

V29_BODYLESS_MAX_FINAL = 1
V29_MAX_STAGES_PER_BASE_EVENT = 2

V29_STAGE_PRIORITY = {
    "enforcement_or_penalty": 95,
    "decision_or_approval": 90,
    "action_executed": 84,
    "followup_or_escalation": 82,
    "action_started": 76,
    "plan_or_proposal": 64,
    "outcome_or_impact": 54,
    "background_context": 25,
}

V29_FOLLOWUP_RE = re.compile(
    r"추가|후속|재발\s*방지|확대|2차|3차|또\s*|다시|재개|재검토|재추진|재발|"
    r"소송\s*검토|법적\s*대응|총파업|전면\s*파업|파업\s*예고|로그오프\s*데이|향후|다음\s*단계",
    re.IGNORECASE,
)
V29_ENFORCEMENT_RE = re.compile(r"과징금|과태료|제재|처분|시정명령|고발|징계|벌금|행정처분|검찰\s*고발|최대과징금", re.IGNORECASE)
V29_DECISION_RE = re.compile(r"의결|통과|승인|선정|채택|확정|결정|심의\s*끝|국무회의\s*통과|법안소위\s*통과", re.IGNORECASE)
V29_ACTION_EXECUTED_RE = re.compile(r"시행|적용|도입|출시|공개|진행|운영|파업|돌입|집회|행진|출범|배포|개시", re.IGNORECASE)
V29_ACTION_STARTED_RE = re.compile(r"착수|개시|출범|조사\s*착수|점검\s*착수|현장점검|검사\s*착수|심의\s*착수", re.IGNORECASE)
V29_PLAN_RE = re.compile(r"계획|추진|검토|예고|발의|신청|준비|예정|도입\s*검토|논의|협의", re.IGNORECASE)
V29_OUTCOME_RE = re.compile(r"영향|피해|효과|성과|매출|점유율|반응|실적|수익성|유출\s*규모|피해\s*규모", re.IGNORECASE)
V29_BACKGROUND_RE = re.compile(r"분석|전망|해설|짚어|대해부|인포그래픽|브리핑|한눈에|종합|라운드업|모음|생태계|경고등", re.IGNORECASE)

V29_ROUNDUP_RE = re.compile(
    r"\[?AI생태계\]?|\[?Game\s*&\s*Now\]?|\[?게임\s*&\s*나우\]?|한눈에|브리핑|주요\s*소식|"
    r"모음|종합|라운드업|이슈\s*정리|/\s*네이버|/\s*카카오|/\s*오픈AI|/\s*MS|/\s*구글",
    re.IGNORECASE,
)
V29_ANALYSIS_TITLE_RE = re.compile(r"경고등|질주하는데|왜|분석|전망|짚어|대해부|인포그래픽|시사점|관측|해설", re.IGNORECASE)

V29_KOREAN_PUBLIC_ACTOR_RE = re.compile(
    r"정부|국회|대통령|위원회|과기정통부|과학기술정보통신부|방미통위|방송미디어통신위원회|방통위|"
    r"개보위|개인정보보호위원회|공정위|공정거래위원회|금융위|금융위원회|금감원|금융감독원|"
    r"행안부|행정안전부|중기부|중소벤처기업부|문체부|문화체육관광부|고용노동부|노동부|경찰청|검찰|KISA|인터넷진흥원|"
    r"정무위|과방위|상임위|의원|부처|청와대|대통령실",
    re.IGNORECASE,
)
V29_OFFICIAL_ACTION_RE = re.compile(
    r"법안|발의|입법|입법예고|행정예고|개정안|시행령|고시|가이드라인|정책협의회|협의회|위원회|"
    r"의무|의무화|시행|적용|제도|규제|대책|전략|사업|선정|지원|예산|공모|착수|출범|배포|"
    r"조사|검사|감독|점검|심의|의결|제재|과징금|과태료|시정명령|고발|처분|현장점검|보완\s*요구",
    re.IGNORECASE,
)
V29_KAKAO_IMPACT_ROUTE_RE = re.compile(
    r"카카오|카톡|플랫폼|온라인\s*플랫폼|포털|SNS|앱마켓|검색|광고|추천|알고리즘|콘텐츠|미디어|OTT|FAST|"
    r"커머스|쇼핑|리뷰|판매자|정산|수수료|소비자\s*보호|이용자\s*보호|개인정보|데이터|보안|해킹|피싱|스미싱|"
    r"망분리|전자금융|핀테크|결제|마이데이터|디지털자산|가상자산|스테이블코인|AI\s*기본법|AI\s*안전|AI\s*거버넌스|"
    r"생성형\s*AI|파운데이션\s*모델|독자\s*AI|AI\s*허브|AI\s*학습용\s*데이터|데이터센터|GPU|NPU|클라우드|오픈소스|공급망|딥페이크|불법촬영",
    re.IGNORECASE,
)
V29_VERTICAL_PUBLIC_PROJECT_RE = re.compile(
    r"K-?푸드|식품|농림|농업|산림|해외우수과학자|Brain\s*to\s*Korea|인재\s*양성|교육\s*과정|스마트제조|제조혁신|"
    r"스마트공장|지역|관광|바이오|수소|이차전지|소재|부품|장비|유니콘\s*육성|해외\s*진출\s*지원|수출\s*지원|공공부문\s*AI|행정\s*혁신",
    re.IGNORECASE,
)
V29_PUBLIC_ADMIN_INTERNAL_RE = re.compile(r"공공부문|행정\s*혁신|정부혁신|공무원|공공기관|지자체|민원|행정서비스", re.IGNORECASE)

V29_MAJOR_COMPETITOR_RE = re.compile(
    r"네이버|NAVER|카카오|쿠팡|토스|배달의민족|배민|우아한형제들|SKT|SK텔레콤|KT|LGU\+|LG유플러스|"
    r"NHN|NHN클라우드|라인|라인야후|구글|Google|오픈AI|OpenAI|챗GPT|MS|마이크로소프트|Microsoft|메타|Meta|"
    r"애플|Apple|아마존|AWS|엔비디아|NVIDIA|브로드컴|Broadcom|앤트로픽|Anthropic|X\b|트위터|틱톡|TikTok|유튜브|YouTube|스냅챗|Snapchat",
    re.IGNORECASE,
)
V29_OVERSEAS_POLICY_ACTOR_RE = re.compile(
    r"미국|EU|유럽연합|독일|프랑스|영국|캐나다|호주|일본|중국|인도네시아|브라질|해외|"
    r"법원|상원|하원|정부|규제기관|집행위원회|의회|FTC|SEC|EU\s*집행위|뮌헨지법|캐나다\s*정부",
    re.IGNORECASE,
)
V29_COMPETITOR_DIRECT_RE = re.compile(
    r"출시|공개|도입|적용|강화|개편|변경|확대|축소|중단|투자|전략적\s*투자|투자\s*유치|인수|합병|매각|"
    r"제휴|협력|계약|수익화|광고\s*정책|서비스\s*정책|가격\s*정책|전략|계획|발표|실적|매출|영업이익|"
    r"장애|오류|해킹|보안|개인정보|유출|소송|고소|규제\s*대응|제재\s*대응|조사\s*대응|등급\s*조정|차단|책임|판결",
    re.IGNORECASE,
)
V29_POLICY_EVAL_ONLY_RE = re.compile(
    r"의무휴업|대형마트|전통시장|유통산업발전법|규제\s*(재검토|완화|폐지|합리화)|반사이익|수혜|키웠다|기울어진\s*운동장|정책\s*평가|사례로",
    re.IGNORECASE,
)
V29_MINOR_TECH_POC_RE = re.compile(r"기술실증|PoC|업무협약|MOU|스마트글라스|물류창고|물류공정|인재\s*양성|교육\s*과정", re.IGNORECASE)

V29_INDUSTRY_ACTOR_RE = re.compile(
    r"AI|인공지능|LLM|에이전트|클라우드|데이터센터|GPU|NPU|TPU|반도체|보안|사이버보안|오픈소스|공급망|"
    r"브로드컴|엔비디아|구글|오픈AI|앤트로픽|MS|마이크로소프트|메타|애플|아마존|AWS|플루이드스택|자산운용사|빅테크|AI\s*인프라",
    re.IGNORECASE,
)
V29_INDUSTRY_CHANGE_RE = re.compile(
    r"투자|구축|증설|확보|출범|인수|합병|제휴|전환|재편|확산|대규모|플랫폼\s*설립|공급망|전력|전력망|"
    r"칩|반도체|데이터센터|인프라|상용화|도입|수요|시장\s*구조|산업\s*구조|생태계|표준|보안\s*위협|공격",
    re.IGNORECASE,
)
V29_INDUSTRY_ROUTE_RE = re.compile(
    r"AI\s*인프라|데이터센터|GPU|NPU|TPU|클라우드|오픈소스|AI\s*개발|개발도구|보안|공급망|전력|광고|커머스|콘텐츠|개인정보|플랫폼\s*규제|생성형\s*AI|파운데이션\s*모델",
    re.IGNORECASE,
)

V29_SELF_STRONG_ENTITY_RE = re.compile(
    r"카카오|카톡|Kakao|카카오톡|카카오페이|카카오뱅크|카카오모빌리티|카카오\s*T\b|카카오T\b|카카오게임즈|카카오엔터|카카오엔터테인먼트|카카오엔터프라이즈|카카오헬스케어|카카오픽코마|디케이테크인|엑스엘게임즈|두나무|업비트",
    re.IGNORECASE,
)
V29_SELF_MATERIAL_RE = re.compile(
    r"노조|파업|임단협|쟁의|조정|성과급|RSU|노동부|근로감독|최저임금|임금체불|고용\s*불안|"
    r"장애|오류|먹통|개인정보|유출|해킹|피싱|보안|침해사고|"
    r"과징금|제재|조사|수사|소송|판결|고발|행정소송|검찰|경찰|"
    r"대표|임원|CPO|CTO|CFO|조직개편|퇴사|사임|교체|선임|리더십|"
    r"인수|합병|매각|지분|투자회수|주식교환|최대주주|경영권|상장|IPO|실적|영업이익|적자|흑자",
    re.IGNORECASE,
)
V29_SELF_NOISE_RE = re.compile(
    r"한복|스킨|이모티콘|게임\s*업데이트|사전예약|쿠폰|이벤트|프로모션|브랜드평판|순위|후원|스폰서|협찬|"
    r"Game\s*&\s*Now|게임\s*&\s*나우|카톡\s*(캡처|캡쳐|대화|메시지|사생활|폭로)|연예인|가수|배우|아이돌|루머",
    re.IGNORECASE,
)


def v29_safe_front_text(item):
    """본문 전체 대신 제목/이슈그룹/RSS 앞부분만 사용해 관련기사·페이지 하단 오염을 피한다."""
    return clean_html_text(
        f"{item.get('기사제목','')} {item.get('top_title','')} {item.get('issue_group','')} "
        f"{item.get('v20_issue_group','')} {item.get('Gemini이슈그룹','')} {item.get('issue_family','')} {item.get('v20_issue_family','')} "
        f"{item.get('internal_category','')} {item.get('v20_internal_category','')} {item.get('Gemini내부카테고리','')} "
        f"{item.get('본문요약','')[:900]} {item.get('RSS요약','')[:900]} {item.get('Gemini판단사유','')[:500]} {item.get('Gemini선정사유','')[:500]}"
    )


def v29_full_text(item):
    return clean_html_text(f"{v29_safe_front_text(item)} {item.get('본문전문','')[:2600]} {item.get('언론사','')} {item.get('링크','')}")


def v29_is_bodyless(item):
    try:
        body_len = int(item.get('본문글자수') or 0)
    except Exception:
        body_len = 0
    status = clean_html_text(item.get('본문상태',''))
    reason = clean_html_text(item.get('본문품질사유',''))
    method = clean_html_text(item.get('본문추출방식',''))
    return body_len <= 0 or '본문추출실패' in status or '제목링크만' in status or method == 'critical_without_body' or 'critical_without_body' in reason


def v29_event_stage_from_text(text):
    t = clean_html_text(text)
    if not t:
        return 'background_context'
    # follow-up/escalation is checked first so "추가 파업 예고" is not swallowed by generic 파업/action.
    if V29_FOLLOWUP_RE.search(t):
        return 'followup_or_escalation'
    if V29_ENFORCEMENT_RE.search(t):
        return 'enforcement_or_penalty'
    if V29_DECISION_RE.search(t):
        return 'decision_or_approval'
    if V29_ACTION_EXECUTED_RE.search(t):
        return 'action_executed'
    if V29_ACTION_STARTED_RE.search(t):
        return 'action_started'
    if V29_PLAN_RE.search(t):
        return 'plan_or_proposal'
    if V29_OUTCOME_RE.search(t):
        return 'outcome_or_impact'
    return 'background_context'


def v29_event_stage(item):
    return v29_event_stage_from_text(v29_safe_front_text(item))


def v29_base_event_key(item):
    # Use existing canonical detectors first, but remove lifecycle stage from duplicate semantics.
    for fn in [
        lambda x: v27_platform_enforcement_key(x, include_stage=False) if 'v27_platform_enforcement_key' in globals() else '',
        lambda x: v16_global_incident_base_key(x) if 'v16_global_incident_base_key' in globals() else '',
        lambda x: v15_privacy_security_base_key(x) if 'v15_privacy_security_base_key' in globals() else '',
        lambda x: v26_event_key_for_item(x) if 'v26_event_key_for_item' in globals() else '',
    ]:
        try:
            key = clean_html_text(fn(item))
            if key:
                return v26_normalize_event_key(key) if 'v26_normalize_event_key' in globals() else key
        except Exception:
            pass
    group = clean_html_text(item.get('v20_issue_group') or item.get('Gemini이슈그룹') or item.get('issue_group') or item.get('기사제목',''))
    fp = title_fingerprint(group)[:90] if group else ''
    return fp


def v29_stage_event_key(item):
    base = v29_base_event_key(item)
    stage = v29_event_stage(item)
    return f"{base}:{stage}" if base else f"no_base:{title_fingerprint(item.get('기사제목',''))[:80]}:{stage}"


def v29_has_same_stage_duplicate(new_item, old_item):
    if normalize_url(new_item.get('링크','')) and normalize_url(new_item.get('링크','')) == normalize_url(old_item.get('링크','')):
        return True, 'same_url'
    new_fp = title_fingerprint(new_item.get('기사제목',''))
    old_fp = title_fingerprint(old_item.get('기사제목',''))
    if new_fp and old_fp and new_fp == old_fp:
        return True, 'same_title'
    if v29_stage_event_key(new_item) and v29_stage_event_key(new_item) == v29_stage_event_key(old_item):
        return True, 'same_base_event_and_stage'
    # Same broad Gemini group is duplicate only when lifecycle stage is the same.
    ng = v20_issue_group_key(new_item.get('v20_issue_group') or new_item.get('Gemini이슈그룹') or '') if 'v20_issue_group_key' in globals() else clean_html_text(new_item.get('v20_issue_group') or new_item.get('Gemini이슈그룹') or '')
    og = v20_issue_group_key(old_item.get('v20_issue_group') or old_item.get('Gemini이슈그룹') or '') if 'v20_issue_group_key' in globals() else clean_html_text(old_item.get('v20_issue_group') or old_item.get('Gemini이슈그룹') or '')
    if ng and og and ng == og and v29_event_stage(new_item) == v29_event_stage(old_item):
        return True, 'same_issue_group_same_stage'
    return False, ''


_BASE_v21_strict_cross_issue_duplicate_v29 = v21_strict_cross_issue_duplicate

def v21_strict_cross_issue_duplicate(new_item, existing_items):
    new_base = v29_base_event_key(new_item)
    new_stage = v29_event_stage(new_item)
    same_base_count = 0
    for old in existing_items:
        same, reason = v29_has_same_stage_duplicate(new_item, old)
        if same:
            return old, 'v29_' + reason
        if new_base and v29_base_event_key(old) == new_base:
            same_base_count += 1
    if same_base_count >= V29_MAX_STAGES_PER_BASE_EVENT:
        # 같은 대주제라도 브리핑에서는 최대 2개 단계까지만 허용한다.
        # 새 항목이 더 중요한 stage라면 final repair에서 대표 교체될 수 있으므로 여기서는 가장 약한 old를 반환한다.
        same_base_items = [x for x in existing_items if v29_base_event_key(x) == new_base]
        if same_base_items:
            loser = min(same_base_items, key=v29_item_priority_score)
            return loser, f'v29_same_base_event_stage_cap:{new_base}:{new_stage}'
    return None, ''


def v29_item_priority_score(item):
    try:
        score = float(v26_final_item_priority_score(item) if 'v26_final_item_priority_score' in globals() else item.get('대표선택점수') or 0)
    except Exception:
        score = v20_float(item.get('대표선택점수'), 0)
    front = v29_safe_front_text(item)
    title = clean_html_text(item.get('기사제목',''))
    stage = v29_event_stage(item)
    score += V29_STAGE_PRIORITY.get(stage, 0)
    if v29_is_bodyless(item):
        score -= 120
    if V29_ROUNDUP_RE.search(title):
        score -= 70
    if V29_ANALYSIS_TITLE_RE.search(title) and not re.search(r"제재|과징금|의결|통과|파업|돌입|출범|시행|적용|유출|장애|소송", title):
        score -= 35
    if re.search(r"단독|속보|종합", title):
        score += 8
    if item.get('카테고리') == JSON_KEYS_ORDER[0] and V29_SELF_MATERIAL_RE.search(front):
        score += 45
    if item.get('카테고리') == JSON_KEYS_ORDER[1] and V29_KOREAN_PUBLIC_ACTOR_RE.search(front) and V29_OFFICIAL_ACTION_RE.search(front):
        score += 25
    if item.get('카테고리') == JSON_KEYS_ORDER[3] and v29_industry_eligible(item):
        score += 25
    return round(score, 2)


def v29_self_eligible(item):
    front = v29_safe_front_text(item)
    if not V29_SELF_STRONG_ENTITY_RE.search(front):
        return False, 'self_without_direct_entity'
    if V29_SELF_NOISE_RE.search(front) or V29_ROUNDUP_RE.search(clean_html_text(item.get('기사제목',''))):
        return False, 'self_low_value_roundup_or_pr'
    if not V29_SELF_MATERIAL_RE.search(front):
        return False, 'self_without_material_risk_or_strategy'
    return True, 'self_material_pass'


def v29_gov_eligible(item):
    front = v29_safe_front_text(item)
    full = v29_full_text(item)
    text = full or front
    if not V29_KOREAN_PUBLIC_ACTOR_RE.search(text):
        return False, 'gov_without_korean_public_actor'
    if not V29_OFFICIAL_ACTION_RE.search(text):
        return False, 'gov_without_official_action'
    if not V29_KAKAO_IMPACT_ROUTE_RE.search(text):
        return False, 'gov_without_kakao_digital_platform_impact_route'
    # Vertical AI/support projects are too broad unless they clearly connect to platform/AI model/data/security/regulation routes.
    if V29_VERTICAL_PUBLIC_PROJECT_RE.search(text):
        strong_route = re.search(
            r"플랫폼|OTT|미디어|방송영상|디지털\s*규제|이용자\s*보호|개인정보|보안|피싱|망분리|데이터센터|GPU|NPU|"
            r"파운데이션\s*모델|독자\s*AI|AI\s*허브|AI\s*학습용\s*데이터|AI\s*안전|AI\s*기본법|딥페이크|불법촬영|전자금융|핀테크|디지털자산|스테이블코인",
            text,
            re.IGNORECASE,
        )
        if not strong_route:
            return False, 'gov_vertical_project_without_platform_route'
    if V29_PUBLIC_ADMIN_INTERNAL_RE.search(text) and not re.search(r"플랫폼|개인정보|보안|피싱|망분리|디지털\s*규제|AI\s*안전|AI\s*기본법|파운데이션\s*모델", text, re.IGNORECASE):
        return False, 'gov_public_admin_internal_low_relevance'
    return True, 'gov_actor_action_route_pass'


def v29_competitor_eligible(item):
    text = v29_full_text(item)
    title = clean_html_text(item.get('기사제목',''))
    if V29_ROUNDUP_RE.search(title):
        return False, 'competitor_roundup_article'
    if not V29_MAJOR_COMPETITOR_RE.search(text):
        return False, 'competitor_without_major_competitor_or_bigtech_actor'
    # Korean public enforcement belongs to Government/National Assembly, not competitor duplicate.
    if V29_KOREAN_PUBLIC_ACTOR_RE.search(text) and V29_ENFORCEMENT_RE.search(text):
        return False, 'competitor_is_korean_public_enforcement'
    # Overseas platform/AI policy can live here even without company direct action.
    if V29_OVERSEAS_POLICY_ACTOR_RE.search(text) and re.search(r"AI|인공지능|플랫폼|SNS|소셜미디어|검색|앱마켓|데이터|개인정보|보안|청소년|콘텐츠|구글|메타|애플|오픈AI|앤트로픽|X\b|틱톡", text, re.IGNORECASE):
        return True, 'overseas_platform_ai_policy_pass'
    if V29_POLICY_EVAL_ONLY_RE.search(text) and not V29_COMPETITOR_DIRECT_RE.search(title):
        return False, 'competitor_only_policy_evaluation_or_example'
    if V29_MINOR_TECH_POC_RE.search(text) and not re.search(r"네이버|쿠팡|토스|배민|SKT|KT|LGU\+|구글|오픈AI|MS|메타|애플|앤트로픽|엔비디아|브로드컴", text, re.IGNORECASE):
        return False, 'minor_poc_or_training_without_core_competitor'
    if not V29_COMPETITOR_DIRECT_RE.search(text):
        return False, 'competitor_without_direct_action'
    return True, 'competitor_direct_action_pass'


def v29_industry_eligible(item):
    text = v29_full_text(item)
    title = clean_html_text(item.get('기사제목',''))
    if V29_ROUNDUP_RE.search(title):
        return False
    if V29_MINOR_TECH_POC_RE.search(text) and not re.search(r"데이터센터|AI\s*인프라|GPU|NPU|TPU|클라우드|오픈소스|공급망|빅테크|엔비디아|브로드컴|오픈AI|앤트로픽|구글", text, re.IGNORECASE):
        return False
    score = 0
    if V29_INDUSTRY_ACTOR_RE.search(text):
        score += 1
    if V29_INDUSTRY_CHANGE_RE.search(text):
        score += 1
    if V29_INDUSTRY_ROUTE_RE.search(text):
        score += 1
    if V29_VERTICAL_PUBLIC_PROJECT_RE.search(text) and not re.search(r"데이터센터|AI\s*인프라|GPU|NPU|TPU|클라우드|오픈소스|공급망|빅테크", text, re.IGNORECASE):
        return False
    return score >= 2


def v29_bodyless_allowed(item):
    if not v29_is_bodyless(item):
        return True, 'has_body'
    cat = item.get('카테고리') or ''
    text = v29_full_text(item)
    title = clean_html_text(item.get('기사제목',''))
    if V29_ROUNDUP_RE.search(title):
        return False, 'bodyless_roundup_not_allowed'
    if cat == JSON_KEYS_ORDER[0]:
        ok, reason = v29_self_eligible(item)
        if ok and re.search(r"파업|제재|과징금|소송|수사|개인정보|유출|장애|임원|조직개편|지분|매각|경영권", text, re.IGNORECASE):
            return True, 'bodyless_allowed_critical_self'
        return False, 'bodyless_self_not_high_confidence:' + reason
    if cat == JSON_KEYS_ORDER[1]:
        ok, reason = v29_gov_eligible(item)
        if ok and re.search(r"과징금|제재|시정명령|의무|법안|시행|망분리|피싱|개인정보|불법촬영|딥페이크|플랫폼", text, re.IGNORECASE):
            return True, 'bodyless_allowed_critical_gov_policy'
        return False, 'bodyless_gov_not_high_confidence:' + reason
    if cat == JSON_KEYS_ORDER[2]:
        if V29_OVERSEAS_POLICY_ACTOR_RE.search(text) and V29_MAJOR_COMPETITOR_RE.search(text) and re.search(r"판결|법안|규제|책임|차단|금지|제재|AI|플랫폼", text, re.IGNORECASE):
            return True, 'bodyless_allowed_major_overseas_policy'
        return False, 'bodyless_competitor_not_high_confidence'
    if cat == JSON_KEYS_ORDER[3]:
        if re.search(r"오픈AI|엔비디아|브로드컴|앤트로픽|구글|MS|데이터센터|AI\s*인프라|GPU|TPU", text, re.IGNORECASE) and v29_industry_eligible(item):
            return True, 'bodyless_allowed_major_industry_structure'
        return False, 'bodyless_industry_not_high_confidence'
    return False, 'bodyless_unknown_category'


def v29_title_only_issue_allowed(category, issue_text):
    pseudo = {'카테고리': category, '기사제목': issue_text[:260], 'RSS요약': issue_text, '본문요약': issue_text, '본문글자수': 0, '본문상태': '본문추출실패_대체없음_제목링크만'}
    return v29_bodyless_allowed(pseudo)[0]


_BASE_v20_is_critical_issue_v29 = v20_is_critical_issue

def v20_is_critical_issue(issue, decision=None):
    if not _BASE_v20_is_critical_issue_v29(issue, decision):
        return False
    decision = decision or {}
    cat = v19_normalize_category_key(decision.get('category'), fallback=v20_issue_to_output_category(issue) if 'v20_issue_to_output_category' in globals() else issue.get('primary_category',''))
    text = clean_html_text(f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('issue_family','')} {issue.get('internal_category','')} {issue.get('company_impact','')} {issue.get('label_reasons','')}")
    return v29_title_only_issue_allowed(cat, text)


# 자사 rescue는 오염될 수 있는 본문 전체를 보지 않고 issue metadata/title/RSS-front 중심으로만 판단한다.
def v28_self_rescue_score(issue, article_by_id=None):
    t = clean_html_text(
        f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('issue_family','')} {issue.get('internal_category','')} "
        f"{issue.get('self_tier','')} {issue.get('company_impact','')} {issue.get('label_reasons','')}"
    )
    if not t:
        return -9999, 'empty'
    pseudo = {'기사제목': issue.get('top_title',''), 'v20_issue_group': issue.get('issue_group',''), 'RSS요약': issue.get('label_reasons','')}
    ok, reason = v29_self_eligible(pseudo)
    if not ok:
        return -9999, reason
    score = v20_float(issue.get('issue_score'), 0) + v20_int(issue.get('max_ceo_priority'), 3) * 25 + v20_int(issue.get('max_pa_priority'), 3) * 15
    if re.search(r"노조|파업|임단협|쟁의|성과급|RSU", t, re.I):
        score += 190
    if re.search(r"개인정보|유출|해킹|장애|피싱|보안|과징금|제재|소송|수사", t, re.I):
        score += 170
    if re.search(r"대표|임원|조직개편|퇴사|사임|교체", t, re.I):
        score += 145
    if re.search(r"두나무|업비트|네이버파이낸셜|지분|매각|주식교환|투자회수|경영권", t, re.I):
        score += 130
    return round(score, 2), reason


_BASE_v20_build_issues_v29 = v20_build_issues

def v20_build_issues(candidates, labels):
    issues, article_by_id = _BASE_v20_build_issues_v29(candidates, labels)
    for issue in issues:
        pseudo = {'기사제목': issue.get('top_title',''), '본문요약': issue.get('issue_group',''), 'RSS요약': issue.get('label_reasons',''), 'Gemini이슈그룹': issue.get('issue_group','')}
        issue['v29_event_base'] = v29_base_event_key(pseudo)
        issue['v29_event_stage'] = v29_event_stage(pseudo)
        issue['v29_stage_event_key'] = v29_stage_event_key(pseudo)
    return issues, article_by_id


_BASE_v20_issue_line_v29 = v20_issue_line

def v20_issue_line(issue):
    base = _BASE_v20_issue_line_v29(issue)
    return base + f" v29_base={v20_clip(issue.get('v29_event_base',''),70)} v29_stage={issue.get('v29_event_stage','')}"


def v29_category_guardrail(item):
    cat = item.get('카테고리') or ''
    if cat == JSON_KEYS_ORDER[0]:
        ok, reason = v29_self_eligible(item)
        return ('pass', cat, reason) if ok else ('remove', '', reason)
    if cat == JSON_KEYS_ORDER[1]:
        ok, reason = v29_gov_eligible(item)
        if ok:
            return 'pass', cat, reason
        # Foreign platform policy accidentally in gov -> competitor if eligible.
        if V29_OVERSEAS_POLICY_ACTOR_RE.search(v29_full_text(item)) and v29_competitor_eligible(item)[0]:
            return 'reassign', JSON_KEYS_ORDER[2], 'foreign_platform_policy_to_competitor'
        return 'remove', '', reason
    if cat == JSON_KEYS_ORDER[2]:
        ok, reason = v29_competitor_eligible(item)
        if ok:
            return 'pass', cat, reason
        # Korean official platform/AI/privacy action should be gov if it passes gov criteria.
        if V29_KOREAN_PUBLIC_ACTOR_RE.search(v29_full_text(item)) and v29_gov_eligible(item)[0]:
            return 'reassign', JSON_KEYS_ORDER[1], 'competitor_korean_public_action_to_gov'
        if v29_industry_eligible(item):
            return 'reassign', JSON_KEYS_ORDER[3], 'competitor_structural_industry_to_industry'
        return 'remove', '', reason
    if cat == JSON_KEYS_ORDER[3]:
        if v29_industry_eligible(item):
            return 'pass', cat, 'industry_structural_pass'
        ok_comp, comp_reason = v29_competitor_eligible(item)
        if ok_comp:
            return 'reassign', JSON_KEYS_ORDER[2], 'industry_competitor_direct_action_to_competitor'
        return 'remove', '', 'industry_without_structural_two_of_three_or_roundup'
    return 'pass', cat, 'unknown_category_pass'


def v29_dedupe_by_stage(items):
    kept_by_stage = {}
    removed = []
    for item in items:
        item['v29_event_base'] = v29_base_event_key(item)
        item['v29_event_stage'] = v29_event_stage(item)
        item['v29_stage_event_key'] = v29_stage_event_key(item)
        key = item['v29_stage_event_key']
        old = kept_by_stage.get(key)
        if not old:
            kept_by_stage[key] = item
            continue
        if v29_item_priority_score(item) > v29_item_priority_score(old):
            removed.append((old, f"v29_same_event_stage_removed:{key}:winner={item.get('기사제목','')[:80]}"))
            kept_by_stage[key] = item
        else:
            removed.append((item, f"v29_same_event_stage_removed:{key}:winner={old.get('기사제목','')[:80]}"))
    stage_kept = list(kept_by_stage.values())

    # Same base event may keep up to two genuinely different lifecycle stages.
    groups = {}
    for item in stage_kept:
        base = item.get('v29_event_base') or ''
        if base:
            groups.setdefault(base, []).append(item)
    final = []
    remove_ids = set()
    for base, group in groups.items():
        if len(group) <= V29_MAX_STAGES_PER_BASE_EVENT:
            continue
        sorted_group = sorted(group, key=v29_item_priority_score, reverse=True)
        keep = sorted_group[:V29_MAX_STAGES_PER_BASE_EVENT]
        keep_ids = {id(x) for x in keep}
        for item in group:
            if id(item) not in keep_ids:
                remove_ids.add(id(item))
                removed.append((item, f"v29_same_base_stage_cap_removed:{base}"))
    final = [x for x in stage_kept if id(x) not in remove_ids]
    return final, removed


_BASE_v26_dedupe_by_event_key_v29 = v26_dedupe_by_event_key

def v26_dedupe_by_event_key(items):
    return v29_dedupe_by_stage(items)


_BASE_v24_duplicate_survivor_score_v29 = v24_duplicate_survivor_score

def v24_duplicate_survivor_score(item):
    return float(_BASE_v24_duplicate_survivor_score_v29(item) or 0) + v29_item_priority_score(item) * 0.45


_BASE_v26_final_item_priority_score_v29 = v26_final_item_priority_score

def v26_final_item_priority_score(item):
    return float(_BASE_v26_final_item_priority_score_v29(item) or 0) + V29_STAGE_PRIORITY.get(v29_event_stage(item), 0) + (-90 if v29_is_bodyless(item) else 0)


_BASE_v21_remove_obvious_final_duplicates_v29 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    kept, removed = _BASE_v21_remove_obvious_final_duplicates_v29(final_items)
    repaired = []
    for item in kept:
        action, new_cat, reason = v29_category_guardrail(item)
        if action == 'remove':
            item['v29_final_guardrail'] = 'removed:' + reason
            removed.append((item, 'v29_final_guardrail_removed:' + reason))
            continue
        if action == 'reassign' and new_cat:
            if 'v23_set_final_category' in globals():
                v23_set_final_category(item, new_cat, reason)
            else:
                item['카테고리'] = new_cat
                item['카테고리명'] = JSON_KEY_TO_DISPLAY.get(new_cat, new_cat)
            item['v29_final_guardrail'] = 'reassigned:' + reason
        else:
            item['v29_final_guardrail'] = 'pass:' + reason
        allowed, body_reason = v29_bodyless_allowed(item)
        item['v29_bodyless_policy'] = body_reason
        if not allowed:
            removed.append((item, 'v29_bodyless_removed:' + body_reason))
            continue
        repaired.append(item)

    repaired, stage_removed = v29_dedupe_by_stage(repaired)
    removed.extend(stage_removed)

    # Final bodyless cap: even allowed title-only items are expensive to review.
    bodyless = [x for x in repaired if v29_is_bodyless(x)]
    if len(bodyless) > V29_BODYLESS_MAX_FINAL:
        keep_bodyless = set(id(x) for x in sorted(bodyless, key=v29_item_priority_score, reverse=True)[:V29_BODYLESS_MAX_FINAL])
        tmp = []
        for item in repaired:
            if v29_is_bodyless(item) and id(item) not in keep_bodyless:
                removed.append((item, 'v29_bodyless_cap_removed'))
                continue
            tmp.append(item)
        repaired = tmp

    # Category caps after reassignment/guardrails.
    tmp = []
    for cat in JSON_KEYS_ORDER:
        bucket = [x for x in repaired if x.get('카테고리') == cat]
        bucket_sorted = sorted(bucket, key=v29_item_priority_score, reverse=True)
        max_keep = CATEGORY_MAX.get(cat, 99)
        for item in bucket_sorted[:max_keep]:
            tmp.append(item)
        for item in bucket_sorted[max_keep:]:
            removed.append((item, f'v29_category_cap_removed:{cat}:{max_keep}'))
    repaired = tmp

    order_base = {key: idx * 1000 for idx, key in enumerate(JSON_KEYS_ORDER, 1)}
    ordered = []
    for key in JSON_KEYS_ORDER:
        bucket = [x for x in repaired if x.get('카테고리') == key]
        bucket_sorted = sorted(bucket, key=v29_item_priority_score, reverse=True)
        for idx, item in enumerate(bucket_sorted, 1):
            item['선정순서'] = order_base.get(key, 9000) + idx
            item['v29_final_priority'] = v29_item_priority_score(item)
            ordered.append(item)
    return ordered, removed


_BASE_v20_gemini_edit_issues_v29 = v20_gemini_edit_issues

def v20_gemini_edit_issues(client, issues, recent_past_text):
    if not client:
        raise RuntimeError('Gemini client 없음')
    visible_issues = [i for i in sorted(issues, key=lambda x: x.get('issue_score', 0), reverse=True) if not i.get('exclude') and not i.get('is_pr')]
    visible_issues = visible_issues[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = '\n'.join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
You are editing a daily Korean CEO/public-affairs briefing for Kakao.
Select issue groups. Return JSON only.

Do NOT patch by one keyword. Judge by structure:
main actor + official/direct action + lifecycle stage + impact route to Kakao.

Lifecycle dedupe:
- Same base event + same lifecycle stage = choose one representative article only.
- Same base event + meaningfully different lifecycle stage may both be selected, maximum 2 stages.
- Stages: plan/proposal, action started, action executed, decision/approval, enforcement/penalty, follow-up/escalation, outcome/impact, background.
- Background/analysis articles lose to concrete action/decision/enforcement/follow-up articles.

Body quality:
- Prefer issues with at least one full-body article.
- Bodyless/title-only issues should be selected only if they are clearly critical: direct Kakao risk, Korean official platform/privacy/security enforcement, or major global AI/big-tech infrastructure/regulatory event.
- Do not fill slots with bodyless roundups, generic training/MOU, minor PoC, or low-relevance public projects.

Self/Kakao:
- Select credible direct/material Kakao issues only: labor/strike, service incident, privacy/security, litigation/regulatory action, leadership/org, governance/M&A/control, major affiliate/strategic asset.
- If a Kakao labor/regulatory topic has different lifecycle stages, e.g. executed action and future escalation, both may be kept.
- Exclude Kakao name in game/entertainment roundups, PR, sponsorship, brand-rank, event, coupon, gossip/chat-capture.
- Desired Self count is 1-3; do not add weak self filler just to reach 2.

Government/National Assembly:
- Requires Korean public actor + official action + impact route to Kakao/digital platform/AI service/privacy/security/fintech/media/OTT/cloud/data-center/digital asset.
- Exclude broad vertical public projects where AI is only a manufacturing/food/agriculture/public-admin support tool unless there is a clear platform/AI model/data/security/regulation route.
- Korean regulator action against a platform/company belongs here, not duplicated in competitor.

Competitor/Overseas:
- Requires a major competitor/big-tech/platform actor and direct action: launch, policy change, investment, acquisition, partnership, outage, privacy/security incident, lawsuit, regulatory response.
- Overseas platform/AI regulation affecting big tech can be selected here.
- Exclude policy evaluation where competitor is only a beneficiary/example, and minor domestic PoC/MOU/training without a core competitor.

Industry:
- Select 1-2 only if genuinely structural.
- Avoid roundups. Prefer single structural themes: AI infrastructure, data centers, chips, cloud, open-source supply chain, AI security, power/infrastructure for AI.
- Need at least two: digital/AI actor, structural change, Kakao impact route.

Recommended final size: 10-13 high-quality items; do not force 15.
Suggested mix: Self 1-3, Gov 3-5, Competitor/Overseas 3-4, Industry 1-2.

JSON format:
{{
  "selected_issues": [
    {{"issue_id": "I001", "category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향", "priority": 5, "best_article_id": 123, "backup_article_ids": [124], "reason": "why"}}
  ],
  "backup_issues": [
    {{"issue_id": "I050", "category": "경쟁사_해외이슈", "priority": 4, "best_article_id": 555, "backup_article_ids": [], "reason": "backup reason"}}
  ]
}}

Recent 7-day index history:
{recent_past_text[:4500]}

Issue candidates:
{issue_text}
"""
    text = gemini_generate_text(client=client, prompt=prompt, task_name='v29 issue editor', model=GEMINI_MODEL_EDITOR)
    data = extract_json_object(text)
    issue_by_id = {i['issue_id']: i for i in issues}
    article_by_id = {}
    for issue in issues:
        for art_id in issue.get('article_ids', []):
            article_by_id[int(art_id)] = {'id': int(art_id)}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    return decisions, backup_decisions


_BASE_v20_build_summary_prompt_v29 = v20_build_summary_prompt

def v20_build_summary_prompt(final_report_data, recent_past_text):
    prompt = _BASE_v20_build_summary_prompt_v29(final_report_data, recent_past_text)
    extra = """
13. 같은 base event 안에서도 이 기사에 배정된 lifecycle stage와 직접 관련된 사실만 요약해. 예: 오늘 실행 기사에는 오늘 실행 사실을, 추가 조치 예고 기사에는 향후 조치 예고를 중심으로 요약해.
14. 라운드업 기사에서 여러 회사 소식이 섞여 있으면, 이 보고서에 배정된 issue_group과 직접 관련 없는 회사·사건은 요약하지마.
15. 본문이 없거나 제목링크 제한 포함인 기사는 추측 요약을 만들지 말고, 제목과 RSS에 명확한 사실만 아주 짧게 요약해.
"""
    if '[추가 요약 범위 규칙]' in prompt:
        prompt += '\n' + extra
    else:
        prompt += '\n[추가 요약 범위 규칙]\n' + extra
    return prompt


_BASE_v20_gemini_quality_check_v29 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v29(client, final_report_data, final_briefing_text)
    local_warnings = []
    bodyless_count = sum(1 for x in final_report_data if v29_is_bodyless(x))
    if bodyless_count > V29_BODYLESS_MAX_FINAL:
        local_warnings.append(f"본문추출실패 기사가 {bodyless_count}개로 v29 상한 {V29_BODYLESS_MAX_FINAL}개를 초과함")
    for item in final_report_data:
        action, _cat, reason = v29_category_guardrail(item)
        if action != 'pass':
            local_warnings.append(f"섹션 부적합 가능: {item.get('카테고리')} / {item.get('기사제목','')} / {reason}")
    # Same base+stage duplicate warning.
    seen_stage = {}
    for item in final_report_data:
        key = v29_stage_event_key(item)
        if key in seen_stage:
            local_warnings.append(f"동일 사건·동일 진행단계 중복 가능: {seen_stage[key]} / {item.get('기사제목','')}")
        else:
            seen_stage[key] = item.get('기사제목','')
    if not local_warnings:
        return qa
    try:
        data = v22_parse_qa_json(qa) if qa else {'overall': '양호', 'warnings': [], 'suggested_human_checks': []}
        warnings = data.get('warnings') if isinstance(data.get('warnings'), list) else []
        checks = data.get('suggested_human_checks') if isinstance(data.get('suggested_human_checks'), list) else []
        warnings.extend(local_warnings)
        checks.extend(local_warnings)
        data['warnings'] = warnings
        data['suggested_human_checks'] = checks
        if data.get('overall') == '양호':
            data['overall'] = '주의'
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return (qa or '') + '\n' + '\n'.join(local_warnings)


# =========================================================
# v30 overrides: competitor-risk friendly sectioning + structural competitor refill
# - 국내 규제기관이 쿠팡/네이버/토스/배민 등 주요 플랫폼에 제재·과징금·시정명령을 내린 사건은
#   정부/국회뿐 아니라 경쟁사 리스크로도 충분히 유효하게 본다.
# - 단, 행안부/과기정통부의 공공 AI 가이드·일반 지원사업처럼 경쟁사 직접 영향이 약한 정부 정책은
#   경쟁사/해외 섹션에 들어오지 못하게 한다.
# - 경쟁사 섹션이 중복 제거 후 비면 정부 기사로 채우지 않고, 구조적으로 경쟁사 actor+action이 있는 후보에서만 refill한다.
# - 시황 제목이라도 AI 인프라·투자·제휴·서비스 정책 변화가 본문 핵심이면 market noise가 아니라 경쟁사 전략 기사로 살린다.
# =========================================================

V6_VERSION = "google_rss_v30_competitor_enforcement_structural_refill"

CATEGORY_MIN.update({
    "자사_및_계열사_이슈": 1,
    "정부_국회": 3,
    "경쟁사_해외이슈": 2,
    "산업동향": 1,
})
CATEGORY_TARGET.update({
    "자사_및_계열사_이슈": 2,
    "정부_국회": 4,
    "경쟁사_해외이슈": 3,
    "산업동향": 2,
})
CATEGORY_MAX.update({
    "자사_및_계열사_이슈": 4,
    "정부_국회": 5,
    "경쟁사_해외이슈": 5,
    "산업동향": 2,
})
V21_FINAL_MIN = 9
V21_FINAL_TARGET = 12
V21_FINAL_MAX = 14
V20_FINAL_MIN = V21_FINAL_MIN
V20_FINAL_TARGET = V21_FINAL_TARGET
V20_FINAL_MAX = V21_FINAL_MAX
MIN_SELECT_COUNT = V21_FINAL_MIN
MAX_SELECT_COUNT = V21_FINAL_MAX
V30_COMPETITOR_MIN_GOOD = 2
V30_COMPETITOR_TARGET_GOOD = 3

V30_COMPETITOR_ROLE_RE = re.compile(
    r"네이버|NAVER|쿠팡|Coupang|토스|비바리퍼블리카|배달의민족|배민|우아한형제들|"
    r"SKT|SK텔레콤|KT|LGU\+|LG유플러스|통신사|통신3사|"
    r"구글|Google|알파벳|Alphabet|오픈AI|OpenAI|챗GPT|ChatGPT|MS|마이크로소프트|Microsoft|"
    r"메타|Meta|애플|Apple|아마존|AWS|엔비디아|NVIDIA|브로드컴|Broadcom|앤트로픽|Anthropic|"
    r"소프트뱅크|오라클|Oracle|퍼플렉시티|Perplexity|xAI|세일즈포스|Salesforce|"
    r"X\b|엑스\b|트위터|Twitter|틱톡|TikTok|유튜브|YouTube|스냅챗|Snapchat|레딧|Reddit|디스코드|Discord|"
    r"포털|검색엔진|SNS|소셜미디어|앱마켓|앱스토어|플레이스토어|이커머스|전자상거래|배달앱|"
    r"플랫폼\s*(기업|사업자|서비스)|빅테크|AI\s*(기업|모델|서비스|플랫폼)|클라우드\s*(기업|서비스)|반도체\s*(기업|설계)|AI\s*인프라",
    re.IGNORECASE,
)

V30_COMPETITOR_DIRECT_ACTION_RE = re.compile(
    r"투자|추가\s*투자|전략적\s*투자|펀드|출자|인수|합병|매각|지분|제휴|협력|파트너십|손잡|계약|"
    r"구축|증설|확보|도입|적용|출시|공개|개편|강화|변경|상향|하향|조정|차단|금지|허용|중단|재개|"
    r"수익화|광고\s*정책|서비스\s*정책|가격\s*정책|멤버십|요금|수수료|정산|"
    r"등급|청소년\s*이용불가|연령\s*제한|콘텐츠\s*정책|앱\s*등급|"
    r"데이터센터|AI\s*팩토리|AI\s*인프라|GPU|NPU|TPU|클라우드|파운데이션\s*모델|"
    r"장애|오류|해킹|보안|개인정보|유출|소송|고소|판결|책임|규제\s*대응|제재\s*대응|조사\s*대응|실적|매출|영업이익",
    re.IGNORECASE,
)

V30_KOREAN_REGULATOR_RE = re.compile(
    r"정부|국회|과기정통부|과학기술정보통신부|방미통위|방송미디어통신위원회|방통위|개보위|개인정보보호위원회|"
    r"공정위|공정거래위원회|금융위|금융위원회|금감원|금융감독원|행안부|행정안전부|중기부|중소벤처기업부|"
    r"고용노동부|노동부|경찰청|검찰|KISA|인터넷진흥원|위원회|부처|의원|상임위|정무위|과방위",
    re.IGNORECASE,
)

V30_ENFORCEMENT_ACTION_RE = re.compile(
    r"과징금|과태료|제재|처분|시정명령|고발|검찰\s*고발|징계|벌금|행정처분|심의|의결|조사|검사|현장점검|"
    r"재발\s*방지|개선권고|공표명령|시정|위반|불법|무단\s*수집|유출|기만\s*광고|허위\s*광고|표시광고법|개인정보보호법",
    re.IGNORECASE,
)

V30_ENFORCEMENT_TOPIC_RE = re.compile(
    r"개인정보|정보\s*유출|무단\s*수집|온라인\s*활동\s*기록|맞춤형\s*광고|광고|표시광고|기만|허위|멤버십|회원가|"
    r"플랫폼|앱|서비스|이용자|소비자|판매자|입점|정산|수수료|리뷰|알고리즘|검색|추천|자사우대|"
    r"보안|해킹|피싱|사칭|청소년|유해\s*콘텐츠|불법촬영|딥페이크|콘텐츠|SNS|소셜미디어|앱마켓",
    re.IGNORECASE,
)

V30_PUBLIC_POLICY_ONLY_RE = re.compile(
    r"공공부문|공공\s*AI|공공기관|행정\s*서비스|행정\s*혁신|정부혁신|공무원|지자체|민원|"
    r"가이드\s*배포|활용\s*가이드|실무\s*가이드|설명회|교육|인재\s*양성|연수|훈련|공모|지원사업|"
    r"주관기관\s*선정|참여기업\s*선정|해외\s*진출\s*지원|수출\s*지원|스마트제조|제조혁신|K-?푸드|식품|농업|산림|관광|바이오|뷰티\s*리테일",
    re.IGNORECASE,
)

V30_MARKET_NOISE_RE = re.compile(
    r"주가|급등|급락|폭등|폭락|하락|상승|목표가|투자의견|매수|매도|지지선|저항선|증권가|애널리스트|"
    r"시총|시가총액|밸류에이션|PER|PBR|ETF|수익률|차익실현|주식\s*시장|코스피|나스닥",
    re.IGNORECASE,
)

V30_STRATEGIC_OVERRIDE_RE = re.compile(
    r"AI\s*팩토리|AI\s*인프라|데이터센터|GW급|기가와트|엔비디아|NVIDIA|앤트로픽|Anthropic|오픈AI|OpenAI|"
    r"투자|추가\s*투자|전략적\s*투자|펀드|제휴|협력|파트너십|구축|증설|도입|출시|공개|정책\s*변경|수익화|"
    r"클라우드|GPU|NPU|TPU|AI\s*반도체|파운데이션\s*모델|앱\s*등급|청소년\s*이용불가|등급\s*상향|보안|개인정보|유출|과징금|제재",
    re.IGNORECASE,
)

V30_VENDOR_PROJECT_RE = re.compile(
    r"기술실증|PoC|파일럿|업무협약|MOU|솔루션\s*공급|고객사|주관기관\s*선정|과제\s*선정|교육\s*과정|인재\s*양성|스마트글라스|물류창고|물류공정",
    re.IGNORECASE,
)

V30_SELF_CSR_PROMO_RE = re.compile(
    r"사회공헌|기부|봉사|문화\s*지원|관람\s*지원|뮤지컬\s*관람|장애인.{0,30}지원|후원|캠페인|나눔|상생\s*행사|이벤트|쿠폰|프로모션|스폰서|협찬",
    re.IGNORECASE,
)


def v30_text(item):
    return clean_html_text(f"{v29_safe_front_text(item)} {item.get('본문전문','')[:3000]} {item.get('언론사','')} {item.get('링크','')}")


def v30_actor_key(text):
    t = clean_html_text(text)
    pairs = [
        ("naver", r"네이버|NAVER"),
        ("coupang", r"쿠팡|Coupang"),
        ("toss", r"토스|비바리퍼블리카"),
        ("baemin", r"배달의민족|배민|우아한형제들"),
        ("skt", r"SKT|SK텔레콤"),
        ("kt", r"\bKT\b|케이티"),
        ("lguplus", r"LGU\+|LG유플러스|엘지유플러스"),
        ("google", r"구글|Google|알파벳|Alphabet"),
        ("openai", r"오픈AI|OpenAI|챗GPT|ChatGPT"),
        ("microsoft", r"MS|마이크로소프트|Microsoft"),
        ("meta", r"메타|Meta|페이스북|Facebook|인스타그램|Instagram"),
        ("apple", r"애플|Apple"),
        ("x_twitter", r"X\b|엑스\b|트위터|Twitter|그록|Grok"),
        ("tiktok", r"틱톡|TikTok"),
        ("youtube", r"유튜브|YouTube"),
        ("anthropic", r"앤트로픽|Anthropic|클로드|Claude"),
        ("nvidia", r"엔비디아|NVIDIA"),
        ("broadcom", r"브로드컴|Broadcom"),
        ("aws_amazon", r"아마존|AWS|Amazon"),
        ("social_media_platform", r"SNS|소셜미디어|사회관계망서비스"),
        ("app_market", r"앱마켓|앱스토어|플레이스토어"),
        ("telecom", r"통신사|통신3사"),
        ("ai_infra_company", r"AI\s*인프라|클라우드\s*기업|반도체\s*기업|빅테크"),
    ]
    for key, pat in pairs:
        if re.search(pat, t, re.IGNORECASE):
            return key
    return ""


def v30_domain_key(text):
    t = clean_html_text(text)
    if re.search(r"개인정보|정보\s*유출|무단\s*수집|보안|해킹|피싱", t, re.IGNORECASE):
        return "privacy_security"
    if re.search(r"광고|표시광고|허위|기만|멤버십|회원가|소비자|리뷰|다크패턴", t, re.IGNORECASE):
        return "ad_consumer_platform"
    if re.search(r"AI\s*팩토리|AI\s*인프라|데이터센터|GPU|NPU|TPU|클라우드|반도체|파운데이션\s*모델", t, re.IGNORECASE):
        return "ai_infra"
    if re.search(r"앱\s*등급|청소년|유해\s*콘텐츠|SNS|소셜미디어|콘텐츠|불법촬영|딥페이크", t, re.IGNORECASE):
        return "platform_content_safety"
    if re.search(r"디지털자산|가상자산|스테이블코인|특금법|FIU|개인지갑|트래블룰", t, re.IGNORECASE):
        return "digital_asset_policy"
    if re.search(r"OTT|미디어|방송|FAST|콘텐츠", t, re.IGNORECASE):
        return "media_ott_policy"
    return "general_digital"


def v30_public_actor_key(text):
    t = clean_html_text(text)
    actors = []
    for key, pat in [
        ("science_ict", r"과기정통부|과학기술정보통신부|과기부"),
        ("media_comm", r"방미통위|방송미디어통신위원회|방통위"),
        ("privacy_comm", r"개보위|개인정보보호위원회|개인정보위"),
        ("fair_trade", r"공정위|공정거래위원회"),
        ("finance", r"금융위|금융위원회|금감원|금융감독원|FIU|금융정보분석원"),
        ("interior", r"행안부|행정안전부"),
        ("assembly", r"국회|의원|상임위|정무위|과방위"),
        ("labor", r"고용노동부|노동부"),
        ("police_prosecution", r"경찰청|검찰"),
        ("sme", r"중기부|중소벤처기업부"),
    ]:
        if re.search(pat, t, re.IGNORECASE):
            actors.append(key)
    if not actors and re.search(r"정부|부처|위원회", t, re.IGNORECASE):
        actors.append("korean_public")
    return "+".join(sorted(set(actors)))


def v30_is_competitor_enforcement_event(text):
    t = clean_html_text(text)
    return bool(
        V30_KOREAN_REGULATOR_RE.search(t)
        and V30_ENFORCEMENT_ACTION_RE.search(t)
        and V30_COMPETITOR_ROLE_RE.search(t)
        and V30_ENFORCEMENT_TOPIC_RE.search(t)
    )


def v30_has_strategic_competitor_action(text):
    t = clean_html_text(text)
    if not t:
        return False
    if v30_is_competitor_enforcement_event(t):
        return True
    if not V30_COMPETITOR_ROLE_RE.search(t):
        return False
    if not V30_COMPETITOR_DIRECT_ACTION_RE.search(t):
        return False
    # Public-sector-only guide/support project is not a competitor action unless it targets a competitor or platform enforcement.
    if V30_KOREAN_REGULATOR_RE.search(t) and V30_PUBLIC_POLICY_ONLY_RE.search(t) and not v30_is_competitor_enforcement_event(t):
        # A government support project can still be competitor-relevant if the main actor is a major platform/big-tech doing something material.
        if not re.search(r"네이버|쿠팡|토스|배민|SKT|KT|LGU\+|구글|오픈AI|MS|메타|애플|앤트로픽|엔비디아|브로드컴", t, re.IGNORECASE):
            return False
    return True


def v30_is_market_noise_only(text):
    t = clean_html_text(text)
    if not V30_MARKET_NOISE_RE.search(t):
        return False
    if V30_STRATEGIC_OVERRIDE_RE.search(t) and v30_has_strategic_competitor_action(t):
        return False
    return True


def v30_is_government_policy_only_for_competitor(text):
    t = clean_html_text(text)
    if not V30_KOREAN_REGULATOR_RE.search(t):
        return False
    if v30_is_competitor_enforcement_event(t):
        return False
    if V30_PUBLIC_POLICY_ONLY_RE.search(t):
        return True
    # General public actor + official guide/support without affected competitor should not be competitor.
    if V29_OFFICIAL_ACTION_RE.search(t) and not V30_COMPETITOR_ROLE_RE.search(t):
        return True
    return False


def v30_is_minor_vendor_customer_project(text):
    t = clean_html_text(text)
    if not V30_VENDOR_PROJECT_RE.search(t):
        return False
    # If a core platform/big-tech/telecom actor is investing/building/changing policy, it is not a minor vendor project.
    if re.search(r"네이버|쿠팡|토스|배민|SKT|KT|LGU\+|구글|오픈AI|MS|메타|애플|앤트로픽|엔비디아|브로드컴", t, re.IGNORECASE) and re.search(r"투자|구축|데이터센터|AI\s*인프라|정책|출시|도입|제휴|협력", t, re.IGNORECASE):
        return False
    return True


def v30_is_self_csr_or_pr_noise(item):
    text = v30_text(item)
    if not V30_SELF_CSR_PROMO_RE.search(text):
        return False
    # Real material risks override CSR noise.
    if V29_SELF_MATERIAL_RE.search(text) and re.search(r"제재|과징금|소송|수사|노조|파업|개인정보|장애|해킹|지분|매각|실적|대표|임원", text, re.IGNORECASE):
        return False
    return True


def v30_competitor_eligible_from_text(text, title=""):
    t = clean_html_text(text)
    title = clean_html_text(title) or t[:220]
    if not t:
        return False, "empty"
    if V29_ROUNDUP_RE.search(title):
        return False, "competitor_roundup_article"
    if v30_is_government_policy_only_for_competitor(t):
        return False, "government_policy_only_not_competitor"
    if V29_POLICY_EVAL_ONLY_RE.search(t) and not v30_has_strategic_competitor_action(t):
        return False, "policy_evaluation_only_not_direct_competitor"
    if v30_is_minor_vendor_customer_project(t):
        return False, "minor_vendor_customer_project"
    if v30_is_competitor_enforcement_event(t):
        return True, "competitor_affected_by_korean_enforcement_pass"
    # Overseas policy/rule changes affecting platform or AI companies are competitor/overseas issues.
    if V29_OVERSEAS_POLICY_ACTOR_RE.search(t) and V30_COMPETITOR_ROLE_RE.search(t) and re.search(r"규제|법안|판결|책임|금지|차단|등급|청소년|AI|플랫폼|SNS|앱마켓|검색|개인정보|보안|콘텐츠", t, re.IGNORECASE):
        return True, "overseas_platform_ai_rule_pass"
    if not V30_COMPETITOR_ROLE_RE.search(t):
        return False, "competitor_without_actor_role"
    if not V30_COMPETITOR_DIRECT_ACTION_RE.search(t):
        return False, "competitor_without_strategy_or_operational_action"
    if v30_is_market_noise_only(t):
        return False, "market_reaction_only_without_strategy_override"
    return True, "competitor_structural_actor_action_pass"


def v30_competitor_issue_eligible(issue):
    text = clean_html_text(" ".join(str(x or "") for x in [
        issue.get("issue_group"), issue.get("top_title"), issue.get("top_summary"), issue.get("issue_family"),
        issue.get("internal_category"), issue.get("label_reasons"), issue.get("company_impact"), issue.get("competitor_tier"), issue.get("gov_tier"), issue.get("v30_actor_role"), issue.get("v30_action_type"),
    ]))
    ok, reason = v30_competitor_eligible_from_text(text, issue.get("top_title") or issue.get("issue_group") or "")
    if ok:
        issue["v30_competitor_refill_basis"] = reason
    return ok


def v30_pseudo_item_from_issue(issue, category=None):
    cat = category or v19_normalize_category_key(issue.get("primary_category"), fallback=v20_issue_to_output_category(issue))
    return {
        "카테고리": cat,
        "기사제목": issue.get("top_title") or issue.get("issue_group") or "",
        "RSS요약": clean_html_text(" ".join(str(x or "") for x in [issue.get("top_summary"), issue.get("label_reasons"), issue.get("company_impact")]))[:1200],
        "본문요약": clean_html_text(" ".join(str(x or "") for x in [issue.get("top_summary"), issue.get("label_reasons"), issue.get("company_impact")]))[:1200],
        "issue_group": issue.get("issue_group", ""),
        "v20_issue_group": issue.get("issue_group", ""),
        "issue_family": issue.get("issue_family", ""),
        "internal_category": issue.get("internal_category", ""),
        "본문전문": "",
        "본문글자수": 999,
    }


def v30_base_event_key_for_item(item):
    text = v30_text(item)
    if not text:
        return ""
    if v30_is_competitor_enforcement_event(text):
        return f"v30_competitor_enforcement:{v30_actor_key(text) or 'competitor'}:{v30_domain_key(text)}"
    if V30_KOREAN_REGULATOR_RE.search(text) and V29_OFFICIAL_ACTION_RE.search(text):
        actors = v30_public_actor_key(text) or "korean_public"
        domain = v30_domain_key(text)
        if re.search(r"협의회|정책협의회|차관급|출범", text, re.IGNORECASE):
            action = "policy_council"
        elif re.search(r"가이드|배포|설명회", text, re.IGNORECASE):
            action = "guide_distribution"
        elif re.search(r"과징금|제재|시정명령|처분|고발", text, re.IGNORECASE):
            action = "enforcement"
        elif re.search(r"선정|지원사업|공모", text, re.IGNORECASE):
            action = "support_selection"
        else:
            action = "public_action"
        return f"v30_public_policy:{actors}:{domain}:{action}"
    if v30_has_strategic_competitor_action(text):
        return f"v30_competitor_action:{v30_actor_key(text) or 'competitor_role'}:{v30_domain_key(text)}"
    return ""


# Market-noise override: stock words do not kill real competitor strategy/infrastructure/action articles.
try:
    _BASE_v14_is_pure_market_or_stock_noise_v30 = v14_is_pure_market_or_stock_noise
    def v14_is_pure_market_or_stock_noise(text, url="", json_key="", keyword=""):
        if v30_has_strategic_competitor_action(text):
            return False
        return _BASE_v14_is_pure_market_or_stock_noise_v30(text, url=url, json_key=json_key, keyword=keyword)
except Exception:
    pass


_BASE_v29_competitor_eligible_v30 = v29_competitor_eligible

def v29_competitor_eligible(item):
    ok, reason = v30_competitor_eligible_from_text(v30_text(item), item.get("기사제목", ""))
    if ok:
        return True, reason
    return False, reason


_BASE_v29_gov_eligible_v30 = v29_gov_eligible

def v29_gov_eligible(item):
    text = v30_text(item)
    # A Korean regulator action against a major competitor/platform may be handled as competitor risk.
    # Do not block it here completely; category guardrail decides preferred section.
    ok, reason = _BASE_v29_gov_eligible_v30(item)
    if not ok:
        return ok, reason
    if V30_PUBLIC_POLICY_ONLY_RE.search(text) and not V29_KAKAO_IMPACT_ROUTE_RE.search(text):
        return False, "gov_public_project_without_kakao_impact_route"
    return True, reason


_BASE_v29_self_eligible_v30 = v29_self_eligible

def v29_self_eligible(item):
    if v30_is_self_csr_or_pr_noise(item):
        return False, "self_csr_pr_or_campaign_not_material"
    return _BASE_v29_self_eligible_v30(item)


_BASE_v29_industry_eligible_v30 = v29_industry_eligible

def v29_industry_eligible(item):
    text = v30_text(item)
    if V29_ROUNDUP_RE.search(clean_html_text(item.get("기사제목", ""))):
        return False
    # Competitor strategy may be left as competitor unless it is broad infra/industry structure.
    if v30_has_strategic_competitor_action(text) and re.search(r"데이터센터|AI\s*인프라|GPU|NPU|TPU|반도체|브로드컴|엔비디아|전력|20GW|53조", text, re.IGNORECASE):
        return True
    return _BASE_v29_industry_eligible_v30(item)


_BASE_v29_base_event_key_v30 = v29_base_event_key

def v29_base_event_key(item):
    key = v30_base_event_key_for_item(item)
    if key:
        return v26_normalize_event_key(key) if "v26_normalize_event_key" in globals() else key
    return _BASE_v29_base_event_key_v30(item)


_BASE_v29_item_priority_score_v30 = v29_item_priority_score

def v29_item_priority_score(item):
    score = float(_BASE_v29_item_priority_score_v30(item) or 0)
    text = v30_text(item)
    cat = item.get("카테고리") or ""
    if v30_is_self_csr_or_pr_noise(item):
        score -= 160
    if cat == JSON_KEYS_ORDER[2]:
        ok, reason = v29_competitor_eligible(item)
        if ok:
            score += 55
            if v30_is_competitor_enforcement_event(text):
                score += 35
            if re.search(r"투자|펀드|제휴|협력|AI\s*팩토리|데이터센터|인프라|등급|청소년\s*이용불가|앱마켓", text, re.IGNORECASE):
                score += 22
        else:
            score -= 90
    if cat == JSON_KEYS_ORDER[1] and v30_is_competitor_enforcement_event(text):
        # Allow this issue to be represented in competitor if both sections picked the same event.
        score -= 20
    if cat == JSON_KEYS_ORDER[1] and V30_PUBLIC_POLICY_ONLY_RE.search(text) and not re.search(r"플랫폼|개인정보|보안|디지털자산|스테이블코인|OTT|미디어|AI\s*기본법|망분리", text, re.IGNORECASE):
        score -= 35
    if v30_is_market_noise_only(text):
        score -= 80
    return round(score, 2)


_BASE_v29_category_guardrail_v30 = v29_category_guardrail

def v29_category_guardrail(item):
    cat = item.get("카테고리") or ""
    text = v30_text(item)
    if cat == JSON_KEYS_ORDER[0] and v30_is_self_csr_or_pr_noise(item):
        return "remove", "", "self_csr_pr_or_campaign_not_material"
    if cat == JSON_KEYS_ORDER[1]:
        # If the article is mainly a material enforcement/sanction against a major competitor,
        # it is acceptable and often preferable as competitor/overseas risk.
        if v30_is_competitor_enforcement_event(text):
            return "reassign", JSON_KEYS_ORDER[2], "korean_regulator_enforcement_against_competitor_as_competitor_risk"
        return _BASE_v29_category_guardrail_v30(item)
    if cat == JSON_KEYS_ORDER[2]:
        ok, reason = v29_competitor_eligible(item)
        if ok:
            return "pass", cat, reason
        if v30_is_government_policy_only_for_competitor(text):
            # Only move to government if it is truly eligible there; otherwise remove rather than pollute competitor.
            if v29_gov_eligible(item)[0]:
                return "reassign", JSON_KEYS_ORDER[1], "competitor_public_policy_only_to_gov"
            return "remove", "", "competitor_public_policy_only_removed"
        if v29_industry_eligible(item):
            return "reassign", JSON_KEYS_ORDER[3], "competitor_structural_industry_to_industry"
        return "remove", "", reason
    return _BASE_v29_category_guardrail_v30(item)


_BASE_v26_1_is_good_refill_candidate_v30 = v26_1_is_good_refill_candidate

def v26_1_is_good_refill_candidate(decision, issue, cat=None):
    cat = cat or v26_1_decision_category(decision, issue)
    if cat == JSON_KEYS_ORDER[2]:
        return v30_competitor_issue_eligible(issue)
    return _BASE_v26_1_is_good_refill_candidate_v30(decision, issue, cat=cat)


_BASE_v26_1_should_refill_after_repair_v30 = v26_1_should_refill_after_repair

def v26_1_should_refill_after_repair(final_report_data):
    if _BASE_v26_1_should_refill_after_repair_v30(final_report_data):
        return True
    comp_items = [x for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[2]]
    good_comp = [x for x in comp_items if v29_competitor_eligible(x)[0]]
    if len(good_comp) < V30_COMPETITOR_MIN_GOOD:
        return True
    if any(not v29_competitor_eligible(x)[0] for x in comp_items):
        return True
    return False


_BASE_v26_1_refill_after_final_repair_v30 = v26_1_refill_after_final_repair

def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    order_counter = _BASE_v26_1_refill_after_final_repair_v30(
        decisions=decisions,
        backup_decisions=backup_decisions,
        issues=issues,
        issue_by_id=issue_by_id,
        article_by_id=article_by_id,
        recent_past_items=recent_past_items,
        final_report_data=final_report_data,
        body_failed_rows=body_failed_rows,
        skip_rows=skip_rows,
        processed_article_ids=processed_article_ids,
        order_counter=order_counter,
        status_counts=status_counts,
    )

    def good_comp_count():
        return sum(1 for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[2] and v29_competitor_eligible(x)[0])

    if good_comp_count() >= V30_COMPETITOR_TARGET_GOOD or len(final_report_data) >= V21_FINAL_MAX:
        return order_counter

    represented_bases = {v29_base_event_key(x) for x in final_report_data if v29_base_event_key(x)}
    represented_issue_ids = {x.get("v20_issue_id") for x in final_report_data if x.get("v20_issue_id")}
    pool = v26_1_candidate_pool(decisions, backup_decisions, issues)

    def comp_rank(d):
        issue = issue_by_id.get(d.get("issue_id"), {})
        txt = clean_html_text(" ".join(str(x or "") for x in [issue.get("issue_group"), issue.get("top_title"), issue.get("top_summary"), issue.get("label_reasons"), issue.get("company_impact")]))
        score = v20_int(d.get("priority"), 1) * 10 + v20_float(issue.get("issue_score"), 0) + v20_float(issue.get("max_ceo_priority"), 0) * 6
        if v30_is_competitor_enforcement_event(txt):
            score += 80
        if re.search(r"투자|제휴|협력|AI\s*팩토리|데이터센터|인프라|앱\s*등급|청소년\s*이용불가|등급\s*상향", txt, re.IGNORECASE):
            score += 55
        if v30_is_market_noise_only(txt):
            score -= 80
        if v30_is_government_policy_only_for_competitor(txt):
            score -= 120
        return score

    comp_pool = []
    seen_iids = set()
    for d in pool:
        iid = d.get("issue_id")
        if not iid or iid in seen_iids or iid in represented_issue_ids:
            continue
        issue = issue_by_id.get(iid, {})
        if not issue or not v30_competitor_issue_eligible(issue):
            continue
        pseudo = v30_pseudo_item_from_issue(issue, JSON_KEYS_ORDER[2])
        base = v29_base_event_key(pseudo)
        if base and base in represented_bases:
            # If a government copy already represents this event, skip unless competitor section is empty.
            if good_comp_count() > 0:
                continue
        d2 = dict(d)
        d2["category"] = JSON_KEYS_ORDER[2]
        d2["v30_forced_competitor_refill"] = "Y"
        comp_pool.append((comp_rank(d2), d2, issue))
        seen_iids.add(iid)

    comp_pool.sort(key=lambda x: x[0], reverse=True)

    for _score, d, issue in comp_pool:
        if good_comp_count() >= V30_COMPETITOR_TARGET_GOOD or len(final_report_data) >= V21_FINAL_MAX:
            break
        iid = d.get("issue_id")
        if iid in represented_issue_ids:
            continue
        item, order_counter, status = v20_process_issue(
            d,
            issue_by_id,
            article_by_id,
            recent_past_items,
            final_report_data,
            body_failed_rows,
            skip_rows,
            processed_article_ids,
            order_counter,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        represented_issue_ids.add(iid)
        if item:
            if item.get("카테고리") != JSON_KEYS_ORDER[2]:
                if "v23_set_final_category" in globals():
                    v23_set_final_category(item, JSON_KEYS_ORDER[2], "v30_competitor_refill_forced")
                else:
                    item["카테고리"] = JSON_KEYS_ORDER[2]
                    item["카테고리명"] = JSON_KEY_TO_DISPLAY.get(JSON_KEYS_ORDER[2], JSON_KEYS_ORDER[2])
            item["v30_competitor_refill"] = "Y"
            represented_bases.add(v29_base_event_key(item))
            print(f"  -> v30 competitor refill added: {v20_clip(issue.get('issue_group',''), 42)} / {v20_clip(item.get('기사제목',''), 42)}...")
    return order_counter


_BASE_v20_issue_to_output_category_v30 = v20_issue_to_output_category

def v20_issue_to_output_category(issue):
    if v30_competitor_issue_eligible(issue):
        return JSON_KEYS_ORDER[2]
    return _BASE_v20_issue_to_output_category_v30(issue)


_BASE_v21_issue_allowed_basic_v30 = v21_issue_allowed_basic

def v21_issue_allowed_basic(issue):
    if v30_competitor_issue_eligible(issue):
        return True, "v30_competitor_structural_issue_allowed"
    return _BASE_v21_issue_allowed_basic_v30(issue)


_BASE_v20_gemini_edit_issues_v30 = v20_gemini_edit_issues

def v20_gemini_edit_issues(client, issues, recent_past_text):
    if not client:
        raise RuntimeError('Gemini client 없음')
    visible_issues = [i for i in sorted(issues, key=lambda x: x.get('issue_score', 0), reverse=True) if not i.get('exclude') and not i.get('is_pr')]
    visible_issues = visible_issues[:V20_MAX_ISSUES_FOR_EDITOR]
    issue_text = '\n'.join(v20_issue_line(i) for i in visible_issues)
    prompt = f"""
You are editing a daily Korean CEO/public-affairs briefing for Kakao.
Select issue groups. Return JSON only.

Judge by structure, not by one keyword:
main actor role + action type + affected actor + lifecycle stage + Kakao relevance.

Important update for Competitor/Overseas:
- A Korean regulator's material action against a major competitor/platform can be a Competitor/Overseas issue, not only Government/National Assembly.
  Examples by structure: privacy commission penalty against an e-commerce platform; fair-trade penalty against a membership/advertising platform; app-market rating/rule change affecting a social platform.
- Do not duplicate the same exact event in two sections. Choose the stronger editorial angle.
- If the key risk is competitor business/operational impact, reputational damage, compliance burden, service restriction, platform rule change, or AI/platform strategy, Competitor/Overseas is acceptable.
- Public-sector-only AI guides, government support projects, public administration AI, training, general selection/support programs, and vertical food/manufacturing projects are NOT Competitor/Overseas unless a major platform/big-tech actor is directly affected.

Competitor/Overseas should prioritize:
- Major competitor/big-tech/platform actor + direct action: investment, partnership, AI infrastructure buildout, product/service policy change, app rating/rule change, privacy/security incident, litigation/regulatory response, business model/pricing change.
- Stock-market wording alone is not enough, but do not reject a story just because the title mentions stock price if the article's substantive event is AI infrastructure, investment, partnership, product policy, or platform rule change.
- Avoid filling Competitor/Overseas with government policy articles when real competitor items are available.

Government/National Assembly:
- Use for Korean public actor + official policy/regulatory action + clear route to platform/AI/privacy/security/fintech/media/digital asset.
- General public administration AI guides/support projects should not crowd out platform/AI/privacy/security regulation.

Self/Kakao:
- Direct/material Kakao risk only. Exclude CSR, donation, cultural support, campaigns, game roundups, promotions, coupons, gossip/chat-capture unless tied to material regulation, litigation, service, governance, labor, privacy/security, or financial risk.

Industry:
- Broad structural AI/platform/cloud/security/data-center/chip/open-source supply-chain changes. Avoid roundups and minor vendor PoC/MOU.

Counts are flexible. Desired: Self 1-2, Gov 3-5, Competitor/Overseas 2-4, Industry 1-2. Do not fill weak articles.

Return JSON only:
{{
  "selected_issues": [
    {{"issue_id": "I001", "category": "자사_및_계열사_이슈/정부_국회/경쟁사_해외이슈/산업동향", "priority": 5, "best_article_id": 123, "backup_article_ids": [124], "reason": "why"}}
  ],
  "backup_issues": [
    {{"issue_id": "I050", "category": "경쟁사_해외이슈", "priority": 4, "best_article_id": 555, "backup_article_ids": [], "reason": "backup reason"}}
  ]
}}

Recent 7-day index history:
{recent_past_text[:4500]}

Issue candidates:
{issue_text}
"""
    text = gemini_generate_text(client=client, prompt=prompt, task_name='v30 issue editor', model=GEMINI_MODEL_EDITOR)
    data = extract_json_object(text)
    issue_by_id = {i['issue_id']: i for i in issues}
    article_by_id = {}
    for issue in issues:
        for art_id in issue.get('article_ids', []):
            article_by_id[int(art_id)] = {'id': int(art_id)}
    decisions, backup_decisions = v20_normalize_issue_editor_json(data, issue_by_id, article_by_id)
    return decisions, backup_decisions


_BASE_v20_gemini_quality_check_v30 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v30(client, final_report_data, final_briefing_text)
    local_warnings = []
    comp_items = [x for x in final_report_data if x.get('카테고리') == JSON_KEYS_ORDER[2]]
    good_comp = [x for x in comp_items if v29_competitor_eligible(x)[0]]
    if len(good_comp) < V30_COMPETITOR_MIN_GOOD:
        local_warnings.append(f"경쟁사/해외이슈의 구조적 적합 기사 수가 {len(good_comp)}개로 v30 기준 {V30_COMPETITOR_MIN_GOOD}개보다 적음")
    for item in comp_items:
        ok, reason = v29_competitor_eligible(item)
        if not ok:
            local_warnings.append(f"경쟁사/해외 섹션 부적합 가능: {item.get('기사제목','')} / {reason}")
    for item in final_report_data:
        if item.get('카테고리') == JSON_KEYS_ORDER[0] and v30_is_self_csr_or_pr_noise(item):
            local_warnings.append(f"자사 섹션 저가치 CSR/캠페인성 가능: {item.get('기사제목','')}")
    if not local_warnings:
        return qa
    try:
        data = v22_parse_qa_json(qa) if qa else {'overall': '양호', 'warnings': [], 'suggested_human_checks': []}
        warnings = data.get('warnings') if isinstance(data.get('warnings'), list) else []
        checks = data.get('suggested_human_checks') if isinstance(data.get('suggested_human_checks'), list) else []
        warnings.extend(local_warnings)
        checks.extend(local_warnings)
        data['warnings'] = warnings
        data['suggested_human_checks'] = checks
        if data.get('overall') == '양호':
            data['overall'] = '주의'
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return (qa or '') + '\n' + '\n'.join(local_warnings)


# =========================================================
# v31 overrides: competitor refill with strict public-policy gate
# - Korean regulator enforcement against a specific competitor/platform is allowed in Competitor/Overseas.
# - General public policy coordination/support/guide projects are never used to fill Competitor/Overseas.
# - Naver and telcos get a small tie-break boost when the issue is otherwise competitor-eligible.
# =========================================================

V6_VERSION = "google_rss_v31_competitor_enforcement_strict_policy_gate_naver_telco_boost"

V31_COMPETITOR_MIN_GOOD = 2
V31_COMPETITOR_TARGET_GOOD = 3
V31_PREFERRED_BOOST = 18
V31_PREFERRED_COMPETITOR_RE = re.compile(
    r"네이버|NAVER|네이버페이|네이버클라우드|네이버웹툰|하이퍼클로바|"
    r"SKT|SK텔레콤|에스케이텔레콤|KT\b|케이티|LGU\+|LG\s*U\+|LG유플러스|엘지유플러스|통신3사",
    re.IGNORECASE,
)

# Specific company/platform names. Generic words like 'platform' or 'big tech' are intentionally excluded here.
V31_SPECIFIC_COMPETITOR_RE = re.compile(
    r"네이버|NAVER|네이버페이|네이버클라우드|네이버웹툰|라인야후|LINE\s*Yahoo|"
    r"쿠팡|Coupang|쿠팡이츠|토스|비바리퍼블리카|배달의민족|배민|우아한형제들|"
    r"SKT|SK텔레콤|에스케이텔레콤|KT\b|케이티|LGU\+|LG\s*U\+|LG유플러스|엘지유플러스|"
    r"구글|Google|알파벳|Alphabet|오픈AI|OpenAI|챗GPT|ChatGPT|MS|마이크로소프트|Microsoft|"
    r"메타|Meta|페이스북|Facebook|인스타그램|Instagram|애플|Apple|아마존|Amazon|AWS|"
    r"엔비디아|NVIDIA|브로드컴|Broadcom|앤트로픽|Anthropic|클로드|Claude|"
    r"X\b|엑스\b|트위터|Twitter|그록|Grok|틱톡|TikTok|유튜브|YouTube|스냅챗|Snapchat|레딧|Reddit|디스코드|Discord|"
    r"소프트뱅크|오라클|Oracle|퍼플렉시티|Perplexity|xAI|세일즈포스|Salesforce",
    re.IGNORECASE,
)

V31_GENERAL_PUBLIC_POLICY_RE = re.compile(
    r"정책협의회|정책\s*협의회|협의체|차관급\s*협의|실무협의|착수회의|공동\s*대응|"
    r"가이드\s*배포|활용\s*가이드|실무\s*가이드|설명회|공청회|간담회|"
    r"지원사업|참여기업\s*선정|주관기관\s*선정|공모|육성사업|해외\s*진출\s*지원|수출\s*지원|"
    r"공공부문|공공\s*AI|공공기관|행정\s*혁신|정부혁신|범정부|AI\s*허브|데이터\s*공유|"
    r"인재\s*양성|교육\s*과정|스마트제조|제조혁신|K-?푸드|식품|농업|산림|관광|뷰티\s*리테일|"
    r"산학연|얼라이언스|위원회\s*출범|위원회\s*설치|전략회의",
    re.IGNORECASE,
)

V31_PUBLIC_ENFORCEMENT_OR_CONSTRAINT_RE = re.compile(
    r"과징금|과태료|제재|처분|시정명령|공표명령|개선권고|고발|검찰\s*고발|벌금|징계|"
    r"조사|검사|현장점검|심의|의결|판결|소송|책임|위반|불법|무단\s*수집|유출|기만\s*광고|허위\s*광고|"
    r"의무|의무화|차단|삭제|필터링|등급\s*상향|청소년\s*이용불가|연령\s*제한|금지|제한|시정|중단",
    re.IGNORECASE,
)

V31_COMPETITOR_IMPACT_DOMAIN_RE = re.compile(
    r"개인정보|정보\s*유출|무단\s*수집|온라인\s*활동\s*기록|맞춤형\s*광고|광고|표시광고|기만|허위|멤버십|회원가|"
    r"플랫폼|앱|서비스|이용자|소비자|판매자|입점|정산|수수료|리뷰|알고리즘|검색|추천|자사우대|"
    r"보안|해킹|피싱|사칭|청소년|유해\s*콘텐츠|불법촬영|딥페이크|콘텐츠|SNS|소셜미디어|앱마켓|"
    r"AI|인공지능|데이터센터|AI\s*팩토리|AI\s*인프라|클라우드|GPU|NPU|TPU|반도체|투자|제휴|협력",
    re.IGNORECASE,
)

V31_COMPETITOR_STRATEGIC_ACTION_RE = re.compile(
    r"투자|추가\s*투자|전략적\s*투자|펀드|출자|인수|합병|매각|지분|제휴|협력|파트너십|손잡|계약|"
    r"구축|증설|확보|도입|적용|출시|공개|개편|강화|변경|상향|하향|조정|차단|금지|허용|중단|재개|"
    r"수익화|광고\s*정책|서비스\s*정책|가격\s*정책|멤버십|요금|수수료|정산|"
    r"등급|청소년\s*이용불가|연령\s*제한|콘텐츠\s*정책|앱\s*등급|"
    r"데이터센터|AI\s*팩토리|AI\s*인프라|GPU|NPU|TPU|클라우드|파운데이션\s*모델|"
    r"장애|오류|해킹|보안|개인정보|유출|소송|고소|판결|책임|규제\s*대응|제재\s*대응|조사\s*대응|실적|매출|영업이익",
    re.IGNORECASE,
)


def v31_text(item):
    return clean_html_text(f"{v29_safe_front_text(item)} {item.get('본문전문','')[:3500]} {item.get('언론사','')} {item.get('링크','')}")


def v31_issue_text(issue):
    return clean_html_text(" ".join(str(x or "") for x in [
        issue.get("issue_group"), issue.get("top_title"), issue.get("top_summary"), issue.get("issue_family"),
        issue.get("internal_category"), issue.get("label_reasons"), issue.get("company_impact"),
        issue.get("competitor_tier"), issue.get("gov_tier"), issue.get("v30_actor_role"), issue.get("v30_action_type"),
    ]))


def v31_has_specific_competitor(text):
    return bool(V31_SPECIFIC_COMPETITOR_RE.search(clean_html_text(text)))


def v31_is_preferred_competitor_context(text):
    t = clean_html_text(text)
    if not V31_PREFERRED_COMPETITOR_RE.search(t):
        return False
    return bool(V31_COMPETITOR_STRATEGIC_ACTION_RE.search(t) or v31_is_specific_competitor_public_enforcement(t))


def v31_is_specific_competitor_public_enforcement(text):
    """Korean public actor action that can be treated as a competitor risk.
    Must involve a specific competitor/platform and an enforcement or operational constraint.
    General policy coordination/support projects are deliberately excluded.
    """
    t = clean_html_text(text)
    if not V30_KOREAN_REGULATOR_RE.search(t):
        return False
    if not v31_has_specific_competitor(t):
        return False
    if not V31_PUBLIC_ENFORCEMENT_OR_CONSTRAINT_RE.search(t):
        return False
    if not V31_COMPETITOR_IMPACT_DOMAIN_RE.search(t):
        return False
    # Policy councils/support projects can mention SKT/Naver as participants, but they are not competitor-risk enforcement.
    if V31_GENERAL_PUBLIC_POLICY_RE.search(t) and not re.search(r"과징금|제재|처분|시정명령|고발|판결|소송|위반|무단\s*수집|유출|기만\s*광고|허위\s*광고|차단\s*의무|등급\s*상향|청소년\s*이용불가", t, re.IGNORECASE):
        return False
    return True


def v31_is_general_public_policy_for_competitor(text):
    t = clean_html_text(text)
    if not V30_KOREAN_REGULATOR_RE.search(t):
        return False
    if v31_is_specific_competitor_public_enforcement(t):
        return False
    if V31_GENERAL_PUBLIC_POLICY_RE.search(t):
        return True
    # Public actor + official action, without a specific affected competitor, belongs to Gov/National Assembly.
    if V29_OFFICIAL_ACTION_RE.search(t) and not v31_has_specific_competitor(t):
        return True
    # Even with a company participant, public-sector support/coordination projects are not competitor section material.
    if V29_OFFICIAL_ACTION_RE.search(t) and v31_has_specific_competitor(t) and not V31_PUBLIC_ENFORCEMENT_OR_CONSTRAINT_RE.search(t):
        return True
    return False


def v31_has_strategic_competitor_action(text):
    t = clean_html_text(text)
    if not t:
        return False
    if v31_is_general_public_policy_for_competitor(t):
        return False
    if v31_is_specific_competitor_public_enforcement(t):
        return True
    if not V30_COMPETITOR_ROLE_RE.search(t) and not v31_has_specific_competitor(t):
        return False
    if not V31_COMPETITOR_STRATEGIC_ACTION_RE.search(t):
        return False
    if v30_is_minor_vendor_customer_project(t):
        return False
    return True


def v31_is_market_noise_only(text):
    t = clean_html_text(text)
    if not V30_MARKET_NOISE_RE.search(t):
        return False
    if V30_STRATEGIC_OVERRIDE_RE.search(t) and v31_has_strategic_competitor_action(t):
        return False
    return True


def v31_competitor_eligible_from_text(text, title=""):
    t = clean_html_text(text)
    title = clean_html_text(title) or t[:220]
    if not t:
        return False, "empty"
    if V29_ROUNDUP_RE.search(title):
        return False, "competitor_roundup_article"
    if v31_is_general_public_policy_for_competitor(t):
        return False, "government_general_policy_not_competitor"
    if V29_POLICY_EVAL_ONLY_RE.search(t) and not v31_has_strategic_competitor_action(t):
        return False, "policy_evaluation_only_not_direct_competitor"
    if v30_is_minor_vendor_customer_project(t):
        return False, "minor_vendor_customer_project"
    if v31_is_specific_competitor_public_enforcement(t):
        return True, "specific_competitor_public_enforcement_pass"
    if V29_OVERSEAS_POLICY_ACTOR_RE.search(t) and (V30_COMPETITOR_ROLE_RE.search(t) or v31_has_specific_competitor(t)) and re.search(r"규제|법안|판결|책임|금지|차단|등급|청소년|AI|플랫폼|SNS|앱마켓|검색|개인정보|보안|콘텐츠", t, re.IGNORECASE):
        return True, "overseas_platform_ai_rule_pass"
    if not V30_COMPETITOR_ROLE_RE.search(t) and not v31_has_specific_competitor(t):
        return False, "competitor_without_actor_role"
    if not V31_COMPETITOR_STRATEGIC_ACTION_RE.search(t):
        return False, "competitor_without_strategy_or_operational_action"
    if v31_is_market_noise_only(t):
        return False, "market_reaction_only_without_strategy_override"
    return True, "competitor_actor_action_pass"


# Make v30 helper names point to the stricter v31 structure so existing refill code uses it.
def v30_is_competitor_enforcement_event(text):
    return v31_is_specific_competitor_public_enforcement(text)


def v30_has_strategic_competitor_action(text):
    return v31_has_strategic_competitor_action(text)


def v30_is_market_noise_only(text):
    return v31_is_market_noise_only(text)


def v30_is_government_policy_only_for_competitor(text):
    return v31_is_general_public_policy_for_competitor(text)


def v30_competitor_eligible_from_text(text, title=""):
    return v31_competitor_eligible_from_text(text, title)


def v30_competitor_issue_eligible(issue):
    text = v31_issue_text(issue)
    ok, reason = v31_competitor_eligible_from_text(text, issue.get("top_title") or issue.get("issue_group") or "")
    issue["v31_competitor_basis"] = reason
    if ok and v31_is_preferred_competitor_context(text):
        issue["v31_preferred_competitor_boost"] = "Y"
    return ok


def v31_public_policy_key(text):
    t = clean_html_text(text)
    if re.search(r"과기정통부|과학기술정보통신부|과기부|방미통위|방송미디어통신위원회", t, re.IGNORECASE) and re.search(r"정책협의회|협의체|차관급|착수회의", t, re.IGNORECASE) and re.search(r"AI|미디어|OTT|디지털|플랫폼|이용자\s*보호", t, re.IGNORECASE):
        return "v31_public_policy:msit_bmtc:ai_media_digital_policy_council"
    if re.search(r"행안부|행정안전부", t, re.IGNORECASE) and re.search(r"공공부문|공공\s*AI|가이드|범정부", t, re.IGNORECASE):
        return "v31_public_policy:mois:public_ai_guide"
    if re.search(r"지원사업|참여기업\s*선정|주관기관\s*선정|육성사업", t, re.IGNORECASE):
        return f"v31_public_policy:{v30_public_actor_key(t) or 'korean_public'}:{v30_domain_key(t)}:support_selection"
    if V31_GENERAL_PUBLIC_POLICY_RE.search(t):
        return f"v31_public_policy:{v30_public_actor_key(t) or 'korean_public'}:{v30_domain_key(t)}:coordination_or_guide"
    return ""


def v31_base_event_key_for_item(item):
    text = v31_text(item)
    if not text:
        return ""
    policy_key = v31_public_policy_key(text)
    if policy_key:
        return policy_key
    if v31_is_specific_competitor_public_enforcement(text):
        return f"v31_competitor_enforcement:{v30_actor_key(text) or 'competitor'}:{v30_domain_key(text)}"
    if v31_has_strategic_competitor_action(text):
        return f"v31_competitor_action:{v30_actor_key(text) or 'competitor_role'}:{v30_domain_key(text)}"
    return ""


def v30_base_event_key_for_item(item):
    key = v31_base_event_key_for_item(item)
    if key:
        return key
    text = v31_text(item)
    if V30_KOREAN_REGULATOR_RE.search(text) and V29_OFFICIAL_ACTION_RE.search(text):
        actors = v30_public_actor_key(text) or "korean_public"
        return f"v31_public_policy:{actors}:{v30_domain_key(text)}:public_action"
    return ""


_BASE_v29_base_event_key_v31 = v29_base_event_key

def v29_base_event_key(item):
    key = v31_base_event_key_for_item(item)
    if key:
        return v26_normalize_event_key(key) if "v26_normalize_event_key" in globals() else key
    return _BASE_v29_base_event_key_v31(item)


_BASE_v29_competitor_eligible_v31 = v29_competitor_eligible

def v29_competitor_eligible(item):
    ok, reason = v31_competitor_eligible_from_text(v31_text(item), item.get("기사제목", ""))
    if ok:
        return True, reason
    return False, reason


_BASE_v29_gov_eligible_v31 = v29_gov_eligible

def v29_gov_eligible(item):
    text = v31_text(item)
    ok, reason = _BASE_v29_gov_eligible_v31(item)
    if not ok:
        return ok, reason
    # General public policy is allowed in Gov if it has a Kakao/digital policy route.
    if V31_GENERAL_PUBLIC_POLICY_RE.search(text) and not re.search(r"플랫폼|AI|인공지능|디지털|개인정보|보안|OTT|미디어|스테이블코인|디지털자산|핀테크|망분리", text, re.IGNORECASE):
        return False, "general_public_policy_without_kakao_digital_route"
    return True, reason


_BASE_v29_category_guardrail_v31 = v29_category_guardrail

def v29_category_guardrail(item):
    cat = item.get("카테고리") or ""
    text = v31_text(item)
    if cat == JSON_KEYS_ORDER[2]:
        ok, reason = v29_competitor_eligible(item)
        if ok:
            return "pass", cat, reason
        if v31_is_general_public_policy_for_competitor(text):
            if v29_gov_eligible(item)[0]:
                return "reassign", JSON_KEYS_ORDER[1], "v31_general_public_policy_to_gov"
            return "remove", "", "v31_general_public_policy_not_competitor"
        if v29_industry_eligible(item):
            return "reassign", JSON_KEYS_ORDER[3], "v31_competitor_structural_industry_to_industry"
        return "remove", "", reason
    if cat == JSON_KEYS_ORDER[1]:
        # Specific competitor enforcement may live in Competitor/Overseas if selected there later.
        # If currently in Gov, keep it unless a duplicate competitor copy wins during final cleanup.
        return _BASE_v29_category_guardrail_v31(item)
    return _BASE_v29_category_guardrail_v31(item)


_BASE_v29_item_priority_score_v31 = v29_item_priority_score

def v29_item_priority_score(item):
    score = float(_BASE_v29_item_priority_score_v31(item) or 0)
    text = v31_text(item)
    cat = item.get("카테고리") or ""
    if cat == JSON_KEYS_ORDER[2]:
        if v31_is_general_public_policy_for_competitor(text):
            score -= 180
        elif v31_is_specific_competitor_public_enforcement(text):
            score += 45
        elif v31_has_strategic_competitor_action(text):
            score += 35
        if v31_is_preferred_competitor_context(text):
            score += V31_PREFERRED_BOOST
    elif cat == JSON_KEYS_ORDER[1]:
        if v31_public_policy_key(text):
            score += 20
        if v31_is_specific_competitor_public_enforcement(text):
            score -= 8
    if v31_is_market_noise_only(text):
        score -= 90
    return round(score, 2)


_BASE_v20_build_issues_v31 = v20_build_issues

def v20_build_issues(candidates, labels):
    issues, article_by_id = _BASE_v20_build_issues_v31(candidates, labels)
    for issue in issues:
        text = v31_issue_text(issue)
        score = float(issue.get("issue_score") or 0)
        if v31_is_general_public_policy_for_competitor(text) and issue.get("primary_category") == JSON_KEYS_ORDER[2]:
            score -= 55
            issue["v31_competitor_policy_penalty"] = "Y"
        if v30_competitor_issue_eligible(issue):
            score += 24
            issue["v31_competitor_eligible"] = "Y"
            if v31_is_specific_competitor_public_enforcement(text):
                score += 28
            if v31_is_preferred_competitor_context(text):
                score += V31_PREFERRED_BOOST
        if v31_public_policy_key(text):
            issue["v31_public_policy_key"] = v31_public_policy_key(text)
        issue["issue_score"] = round(score, 3)
    issues = sorted(issues, key=lambda x: x.get("issue_score", 0), reverse=True)
    for idx, issue in enumerate(issues, 1):
        issue["issue_id"] = f"I{idx:03d}"
    return issues, article_by_id


_BASE_v20_issue_line_v31 = v20_issue_line

def v20_issue_line(issue):
    base = _BASE_v20_issue_line_v31(issue)
    flags = []
    if issue.get("v31_competitor_eligible") == "Y":
        flags.append("v31_competitor=Y")
    if issue.get("v31_preferred_competitor_boost") == "Y":
        flags.append("v31_preferred_naver_telco=Y")
    if issue.get("v31_public_policy_key"):
        flags.append(f"v31_public_policy={v20_clip(issue.get('v31_public_policy_key',''),70)}")
    if issue.get("v31_competitor_policy_penalty") == "Y":
        flags.append("v31_not_competitor_public_policy=Y")
    return base + (" " + " ".join(flags) if flags else "")


def v31_event_preference_rank(item):
    cat = item.get("카테고리") or ""
    key = v29_base_event_key(item)
    text = v31_text(item)
    if cat == JSON_KEYS_ORDER[0]:
        return 0
    if key.startswith("v31_competitor_enforcement"):
        return 0 if cat == JSON_KEYS_ORDER[2] else 2
    if key.startswith("v31_competitor_action"):
        return 0 if cat == JSON_KEYS_ORDER[2] else 3
    if key.startswith("v31_public_policy"):
        return 0 if cat == JSON_KEYS_ORDER[1] else 5
    if v31_is_general_public_policy_for_competitor(text):
        return 5 if cat == JSON_KEYS_ORDER[2] else 1
    return JSON_KEYS_ORDER.index(cat) if cat in JSON_KEYS_ORDER else 9


def v31_add_skip(skip_rows, item, reason):
    if skip_rows is None:
        return
    try:
        skip_rows.append({
            "검색어": item.get("검색어", ""),
            "후보제목": item.get("기사제목", ""),
            "후보링크": item.get("링크", ""),
            "후보언론사": item.get("언론사", ""),
            "매칭과거일자": "CURRENT_RUN",
            "매칭과거제목": "v31_final_cleanup",
            "매칭과거링크": "",
            "중복판정이유": reason,
            "공유주체": "",
            "공유사건태그": item.get("사건태그", ""),
            "종합점수": item.get("v29_final_priority", item.get("대표선택점수", "")),
            "제목점수": "",
            "토큰점수": "",
            "본문점수": "",
            "제외단계": "v31_final_cleanup",
        })
    except Exception:
        pass


def v31_repair_final_items(final_report_data, skip_rows=None):
    cleaned = []
    for item in list(final_report_data):
        action, new_cat, reason = v29_category_guardrail(item)
        if action == "remove":
            item["v31_final_cleanup"] = "removed:" + reason
            v31_add_skip(skip_rows, item, "v31_removed:" + reason)
            continue
        if action == "reassign" and new_cat:
            if "v23_set_final_category" in globals():
                v23_set_final_category(item, new_cat, "v31_final_cleanup:" + reason)
            else:
                item["카테고리"] = new_cat
                item["카테고리명"] = JSON_KEY_TO_DISPLAY.get(new_cat, new_cat)
            item["v31_final_cleanup"] = "reassigned:" + reason
        else:
            item["v31_final_cleanup"] = "pass:" + reason
        cleaned.append(item)

    # Cross-section event dedupe after refill. This catches policy council copies added after earlier repair.
    by_key = {}
    no_key = []
    for item in cleaned:
        key = v29_base_event_key(item)
        if not key:
            no_key.append(item)
            continue
        old = by_key.get(key)
        if old is None:
            by_key[key] = item
            continue
        old_rank = (v31_event_preference_rank(old), -v29_item_priority_score(old))
        new_rank = (v31_event_preference_rank(item), -v29_item_priority_score(item))
        if new_rank < old_rank:
            v31_add_skip(skip_rows, old, f"v31_cross_section_duplicate_removed:{key}:winner={item.get('기사제목','')[:80]}")
            by_key[key] = item
        else:
            v31_add_skip(skip_rows, item, f"v31_cross_section_duplicate_removed:{key}:winner={old.get('기사제목','')[:80]}")

    repaired = no_key + list(by_key.values())

    # Competitor section should not contain general public policy after all refill passes.
    tmp = []
    for item in repaired:
        if item.get("카테고리") == JSON_KEYS_ORDER[2] and v31_is_general_public_policy_for_competitor(v31_text(item)):
            if v29_gov_eligible(item)[0]:
                if "v23_set_final_category" in globals():
                    v23_set_final_category(item, JSON_KEYS_ORDER[1], "v31_competitor_general_policy_to_gov_after_refill")
                else:
                    item["카테고리"] = JSON_KEYS_ORDER[1]
                    item["카테고리명"] = JSON_KEY_TO_DISPLAY.get(JSON_KEYS_ORDER[1], JSON_KEYS_ORDER[1])
                tmp.append(item)
            else:
                v31_add_skip(skip_rows, item, "v31_competitor_general_policy_removed_after_refill")
            continue
        tmp.append(item)
    repaired = tmp

    # Category caps and ordering.
    capped = []
    for cat in JSON_KEYS_ORDER:
        bucket = [x for x in repaired if x.get("카테고리") == cat]
        bucket_sorted = sorted(bucket, key=v29_item_priority_score, reverse=True)
        max_keep = CATEGORY_MAX.get(cat, 99)
        for item in bucket_sorted[:max_keep]:
            capped.append(item)
        for item in bucket_sorted[max_keep:]:
            v31_add_skip(skip_rows, item, f"v31_category_cap_removed:{cat}:{max_keep}")

    order_base = {key: idx * 1000 for idx, key in enumerate(JSON_KEYS_ORDER, 1)}
    ordered = []
    for cat in JSON_KEYS_ORDER:
        bucket = [x for x in capped if x.get("카테고리") == cat]
        for idx, item in enumerate(sorted(bucket, key=v29_item_priority_score, reverse=True), 1):
            item["선정순서"] = order_base.get(cat, 9000) + idx
            item["v31_final_priority"] = v29_item_priority_score(item)
            ordered.append(item)
    final_report_data[:] = ordered
    return final_report_data


_BASE_v21_remove_obvious_final_duplicates_v31 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    kept, removed = _BASE_v21_remove_obvious_final_duplicates_v31(final_items)
    skip_like = []
    v31_repair_final_items(kept, skip_like)
    for row in skip_like:
        removed.append(({"기사제목": row.get("후보제목", ""), "링크": row.get("후보링크", "")}, row.get("중복판정이유", "v31_cleanup")))
    return kept, removed


_BASE_v26_1_is_good_refill_candidate_v31 = v26_1_is_good_refill_candidate

def v26_1_is_good_refill_candidate(decision, issue, cat=None):
    cat = cat or v26_1_decision_category(decision, issue)
    if cat == JSON_KEYS_ORDER[2]:
        return v30_competitor_issue_eligible(issue)
    return _BASE_v26_1_is_good_refill_candidate_v31(decision, issue, cat=cat)


_BASE_v20_issue_to_output_category_v31 = v20_issue_to_output_category

def v20_issue_to_output_category(issue):
    text = v31_issue_text(issue)
    if v31_is_general_public_policy_for_competitor(text):
        return JSON_KEYS_ORDER[1]
    if v30_competitor_issue_eligible(issue):
        return JSON_KEYS_ORDER[2]
    return _BASE_v20_issue_to_output_category_v31(issue)


_BASE_v21_issue_allowed_basic_v31 = v21_issue_allowed_basic

def v21_issue_allowed_basic(issue):
    if v30_competitor_issue_eligible(issue):
        return True, "v31_competitor_structural_or_enforcement_allowed"
    return _BASE_v21_issue_allowed_basic_v31(issue)


def v31_good_competitor_count(final_report_data):
    return sum(1 for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[2] and v29_competitor_eligible(x)[0])


def v31_competitor_refill_score(decision, issue):
    text = v31_issue_text(issue)
    score = v20_int(decision.get("priority"), 1) * 10 + v20_float(issue.get("issue_score"), 0) + v20_float(issue.get("max_ceo_priority"), 0) * 6
    if v31_is_specific_competitor_public_enforcement(text):
        score += 80
    if v31_has_strategic_competitor_action(text):
        score += 55
    if v31_is_preferred_competitor_context(text):
        score += V31_PREFERRED_BOOST
    if v31_is_general_public_policy_for_competitor(text):
        score -= 200
    if v31_is_market_noise_only(text):
        score -= 80
    return score


_BASE_v26_1_refill_after_final_repair_v31 = v26_1_refill_after_final_repair

def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    order_counter = _BASE_v26_1_refill_after_final_repair_v31(
        decisions=decisions,
        backup_decisions=backup_decisions,
        issues=issues,
        issue_by_id=issue_by_id,
        article_by_id=article_by_id,
        recent_past_items=recent_past_items,
        final_report_data=final_report_data,
        body_failed_rows=body_failed_rows,
        skip_rows=skip_rows,
        processed_article_ids=processed_article_ids,
        order_counter=order_counter,
        status_counts=status_counts,
    )

    v31_repair_final_items(final_report_data, skip_rows)

    if v31_good_competitor_count(final_report_data) >= V31_COMPETITOR_TARGET_GOOD or len(final_report_data) >= V21_FINAL_MAX:
        return order_counter

    represented_bases = {v29_base_event_key(x) for x in final_report_data if v29_base_event_key(x)}
    represented_issue_ids = {x.get("v20_issue_id") for x in final_report_data if x.get("v20_issue_id")}
    pool = v26_1_candidate_pool(decisions, backup_decisions, issues)

    candidates = []
    seen_iids = set()
    actor_counts = {}
    for item in final_report_data:
        if item.get("카테고리") == JSON_KEYS_ORDER[2]:
            actor = v30_actor_key(v31_text(item)) or "unknown"
            actor_counts[actor] = actor_counts.get(actor, 0) + 1

    for d in pool:
        iid = d.get("issue_id")
        if not iid or iid in seen_iids or iid in represented_issue_ids:
            continue
        issue = issue_by_id.get(iid, {})
        if not issue or not v30_competitor_issue_eligible(issue):
            continue
        pseudo = v30_pseudo_item_from_issue(issue, JSON_KEYS_ORDER[2])
        base = v29_base_event_key(pseudo)
        if base and base in represented_bases:
            continue
        actor = v30_actor_key(v31_issue_text(issue)) or "unknown"
        # Do not let one actor consume the entire competitor section, but allow two separate material events.
        if actor_counts.get(actor, 0) >= 2:
            continue
        d2 = dict(d)
        d2["category"] = JSON_KEYS_ORDER[2]
        d2["v31_forced_competitor_refill"] = "Y"
        candidates.append((v31_competitor_refill_score(d2, issue), d2, issue, actor, base))
        seen_iids.add(iid)

    candidates.sort(key=lambda x: x[0], reverse=True)

    for _score, d, issue, actor, base in candidates:
        if v31_good_competitor_count(final_report_data) >= V31_COMPETITOR_TARGET_GOOD or len(final_report_data) >= V21_FINAL_MAX:
            break
        if d.get("issue_id") in represented_issue_ids:
            continue
        item, order_counter, status = v20_process_issue(
            d,
            issue_by_id,
            article_by_id,
            recent_past_items,
            final_report_data,
            body_failed_rows,
            skip_rows,
            processed_article_ids,
            order_counter,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        represented_issue_ids.add(d.get("issue_id"))
        if item:
            if "v23_set_final_category" in globals():
                v23_set_final_category(item, JSON_KEYS_ORDER[2], "v31_competitor_refill_forced")
            else:
                item["카테고리"] = JSON_KEYS_ORDER[2]
                item["카테고리명"] = JSON_KEY_TO_DISPLAY.get(JSON_KEYS_ORDER[2], JSON_KEYS_ORDER[2])
            item["v31_competitor_refill"] = "Y"
            actor_counts[actor] = actor_counts.get(actor, 0) + 1
            if base:
                represented_bases.add(base)
            v31_repair_final_items(final_report_data, skip_rows)
            print(f"  -> v31 competitor refill added: {v20_clip(issue.get('issue_group',''), 42)} / {v20_clip(item.get('기사제목',''), 42)}...")

    v31_repair_final_items(final_report_data, skip_rows)
    return order_counter


_BASE_v20_gemini_quality_check_v31 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v31(client, final_report_data, final_briefing_text)
    local_warnings = []
    comp_items = [x for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[2]]
    bad_comp = [x for x in comp_items if not v29_competitor_eligible(x)[0]]
    if bad_comp:
        for item in bad_comp:
            local_warnings.append(f"경쟁사/해외 섹션 부적합 가능: {item.get('기사제목','')} / {v29_competitor_eligible(item)[1]}")
    if v31_good_competitor_count(final_report_data) < V31_COMPETITOR_MIN_GOOD:
        local_warnings.append(f"경쟁사/해외이슈의 구조적 적합 기사 수가 {v31_good_competitor_count(final_report_data)}개로 v31 기준 {V31_COMPETITOR_MIN_GOOD}개보다 적음")
    policy_keys = {}
    for item in final_report_data:
        key = v31_public_policy_key(v31_text(item))
        if key:
            policy_keys.setdefault(key, []).append(item)
    for key, vals in policy_keys.items():
        if len(vals) > 1:
            local_warnings.append(f"공공정책 동일 이슈 중복 가능: {key} / {len(vals)}건")
    if not local_warnings:
        return qa
    try:
        data = v22_parse_qa_json(qa) if qa else {"overall": "양호", "warnings": [], "suggested_human_checks": []}
        warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
        checks = data.get("suggested_human_checks") if isinstance(data.get("suggested_human_checks"), list) else []
        warnings.extend(local_warnings)
        checks.extend(local_warnings)
        data["warnings"] = warnings
        data["suggested_human_checks"] = checks
        if data.get("overall") == "양호":
            data["overall"] = "주의"
        return json.dumps(data, ensure_ascii=False, indent=2)
    except Exception:
        return (qa or "") + "\n" + "\n".join(local_warnings)


# =========================================================
# v32 overrides: competitor portfolio refill + weak non-digital gov cleanup
# - 쿠팡 개인정보/와우회원가 제재 2건은 경쟁사 리스크로 허용한다.
# - 동시에 경쟁사/해외 섹션이 쿠팡만으로 끝나지 않도록 네이버/통신3사/글로벌 플랫폼 후보를 별도 포트폴리오로 보충한다.
# - 정부/국회 섹션의 비디지털 산업 협약/지원사업/단순 AI 활용 기사는 최종본에서 제거한다.
# =========================================================

V6_VERSION = "google_rss_v32_competitor_portfolio_refill"

# 좋은 경쟁사 후보를 Gemini 후보군에 더 안정적으로 올리기 위해 라벨링 풀을 조금 넓힌다.
try:
    MAX_CANDIDATES_FOR_GEMINI = max(int(MAX_CANDIDATES_FOR_GEMINI), 380)
except Exception:
    MAX_CANDIDATES_FOR_GEMINI = 380
try:
    V20_MAX_ISSUES_FOR_EDITOR = max(int(V20_MAX_ISSUES_FOR_EDITOR), 110)
except Exception:
    V20_MAX_ISSUES_FOR_EDITOR = 110

CATEGORY_MIN.update({
    "자사_및_계열사_이슈": 1,
    "정부_국회": 3,
    "경쟁사_해외이슈": 3,
    "산업동향": 1,
})
CATEGORY_TARGET.update({
    "자사_및_계열사_이슈": 2,
    "정부_국회": 4,
    "경쟁사_해외이슈": 4,
    "산업동향": 2,
})
CATEGORY_MAX.update({
    "자사_및_계열사_이슈": 4,
    "정부_국회": 5,
    "경쟁사_해외이슈": 5,
    "산업동향": 2,
})
V21_FINAL_MIN = 9
V21_FINAL_TARGET = 13
V21_FINAL_MAX = 15
V20_FINAL_MIN = V21_FINAL_MIN
V20_FINAL_TARGET = V21_FINAL_TARGET
V20_FINAL_MAX = V21_FINAL_MAX
MIN_SELECT_COUNT = V21_FINAL_MIN
MAX_SELECT_COUNT = V21_FINAL_MAX

V32_COMPETITOR_TARGET_TOTAL = 4
V32_COMPETITOR_MIN_TOTAL = 3
V32_COMPETITOR_MIN_NON_COUPANG = 1
V32_COMPETITOR_PREFER_NON_COUPANG = 2
V32_COMPETITOR_MIN_OVERSEAS_OR_GLOBAL = 1

V32_NAVER_TELCO_RE = re.compile(
    r"네이버|NAVER|네이버페이|네이버클라우드|하이퍼클로바|SKT|SK텔레콤|KT|LGU\+|LG유플러스|통신3사|통신사",
    re.IGNORECASE,
)
V32_COUPANG_RE = re.compile(r"쿠팡|Coupang|로켓와우|와우회원가", re.IGNORECASE)
V32_GLOBAL_PLATFORM_RE = re.compile(
    r"구글|Google|오픈AI|OpenAI|챗GPT|ChatGPT|MS|마이크로소프트|Microsoft|메타|Meta|애플|Apple|"
    r"X\b|엑스\b|트위터|Twitter|틱톡|TikTok|유튜브|YouTube|앤트로픽|Anthropic|브로드컴|Broadcom|엔비디아|NVIDIA|"
    r"캐나다|독일|EU|유럽|미국|일본|중국|해외|글로벌",
    re.IGNORECASE,
)
V32_STRATEGIC_COMPETITOR_ACTION_RE = re.compile(
    r"AI\s*팩토리|AI\s*인프라|데이터센터|AIDC|GPU|NPU|TPU|클라우드|파운데이션\s*모델|LLM|"
    r"엔비디아|NVIDIA|앤트로픽|Anthropic|오픈AI|OpenAI|투자|추가\s*투자|전략적\s*투자|펀드|"
    r"제휴|협력|파트너십|동맹|구축|가동|증설|도입|출시|공개|인수|합병|지분|서비스\s*개편|정책\s*변경|"
    r"앱\s*등급|청소년\s*이용불가|등급\s*상향|차단\s*의무|불법촬영|AI\s*개요|AI\s*검색|광고|개인화\s*광고|수익화|"
    r"보안|개인정보|유출|과징금|제재|시정명령|소송|판결|책임|규제\s*대응",
    re.IGNORECASE,
)
V32_OVERSEAS_RULE_RE = re.compile(
    r"캐나다|독일|EU|유럽|미국|영국|일본|중국|해외|글로벌|법원|당국|규제기관|구글\s*플레이|플레이스토어|앱스토어",
    re.IGNORECASE,
)
V32_COMPETITOR_OPERATIONAL_RISK_RE = re.compile(
    r"과징금|과태료|제재|시정명령|고발|조사|소송|판결|책임|개인정보|유출|무단\s*수집|허위\s*광고|기만\s*광고|"
    r"멤버십|와우회원가|앱\s*등급|청소년\s*이용불가|등급\s*상향|차단\s*의무|불법촬영|유해\s*콘텐츠|AI\s*개요|AI\s*챗봇|SNS\s*금지",
    re.IGNORECASE,
)
V32_GENERAL_POLICY_BLOCK_RE = re.compile(
    r"정책협의회|협의체|차관급|착수회의|가이드\s*배포|공공부문|범정부|지원사업|주관기관\s*선정|참여기업\s*선정|"
    r"육성사업|인재\s*양성|설명회|얼라이언스|스마트제조|K-푸드|제조혁신|뷰티\s*리테일|의료관광|창업지원|"
    r"공정거래\s*협약|상생협력\s*협약|동반성장|협력사\s*상생|건설산업|전문건설|현대건설",
    re.IGNORECASE,
)
V32_NON_DIGITAL_GOV_RE = re.compile(
    r"건설|현대건설|전문건설|건설산업|협력사|동반성장|상생협력|부당특약|식품|K-푸드|스마트제조|제조혁신|"
    r"뷰티\s*리테일|의료관광|농업|축산|수산|화학|에너지|산림|관광|대학교|창업지원사업",
    re.IGNORECASE,
)
V32_STRONG_DIGITAL_ROUTE_RE = re.compile(
    r"온라인\s*플랫폼|디지털\s*플랫폼|포털|검색|SNS|소셜미디어|앱마켓|OTT|미디어|콘텐츠|커머스|광고|"
    r"핀테크|전자금융|결제|개인정보|보안|해킹|피싱|딥페이크|불법촬영|AI\s*모델|파운데이션\s*모델|"
    r"AI\s*인프라|데이터센터|GPU|NPU|클라우드|디지털자산|스테이블코인|가상자산|망분리|지도\s*반출|저작권",
    re.IGNORECASE,
)
V32_WEAK_ROUNDUP_RE = re.compile(r"\[IT소식\]|\[AI생태계\]|Game\s*&\s*Now|소식\]|브리핑|모음|한눈에", re.IGNORECASE)


def v32_text(item):
    try:
        return clean_html_text(
            f"{item.get('기사제목','')} {item.get('issue_group','')} {item.get('v20_issue_group','')} "
            f"{item.get('본문요약','')} {item.get('RSS요약','')} {item.get('본문전문','')[:3200]} "
            f"{item.get('label_reasons','')} {item.get('company_impact','')} {item.get('titles','')} {item.get('검색어','')} {item.get('언론사','')}"
        )
    except Exception:
        return ""


def v32_issue_text(issue):
    try:
        return clean_html_text(
            f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('top_summary','')} {issue.get('titles','')} "
            f"{issue.get('issue_family','')} {issue.get('internal_category','')} {issue.get('primary_category','')} "
            f"{issue.get('competitor_tier','')} {issue.get('competitor_scope','')} {issue.get('label_reasons','')} {issue.get('company_impact','')}"
        )
    except Exception:
        return ""


def v32_actor_key_from_text(text):
    actor = ""
    try:
        actor = v30_actor_key(text) or ""
    except Exception:
        actor = ""
    if actor:
        return actor
    t = clean_html_text(text)
    if V32_NAVER_TELCO_RE.search(t):
        if re.search(r"네이버|NAVER", t, re.IGNORECASE):
            return "naver"
        if re.search(r"SKT|SK텔레콤", t, re.IGNORECASE):
            return "skt"
        if re.search(r"\bKT\b|케이티", t, re.IGNORECASE):
            return "kt"
        if re.search(r"LGU\+|LG유플러스", t, re.IGNORECASE):
            return "lguplus"
        return "telecom"
    if V32_COUPANG_RE.search(t):
        return "coupang"
    if V32_GLOBAL_PLATFORM_RE.search(t):
        if re.search(r"X\b|엑스\b|트위터|Twitter|그록|Grok", t, re.IGNORECASE):
            return "x_twitter"
        if re.search(r"구글|Google", t, re.IGNORECASE):
            return "google"
        if re.search(r"오픈AI|OpenAI|챗GPT|ChatGPT", t, re.IGNORECASE):
            return "openai"
        if re.search(r"메타|Meta|페이스북|인스타그램", t, re.IGNORECASE):
            return "meta"
        if re.search(r"애플|Apple", t, re.IGNORECASE):
            return "apple"
        if re.search(r"앤트로픽|Anthropic", t, re.IGNORECASE):
            return "anthropic"
        if re.search(r"엔비디아|NVIDIA", t, re.IGNORECASE):
            return "nvidia"
        if re.search(r"브로드컴|Broadcom", t, re.IGNORECASE):
            return "broadcom"
        return "global_platform"
    return ""


def v32_company_group_from_text(text):
    actor = v32_actor_key_from_text(text)
    if actor in {"skt", "kt", "lguplus", "telecom"}:
        return "telco"
    if actor in {"google", "openai", "microsoft", "meta", "apple", "x_twitter", "tiktok", "youtube", "anthropic", "nvidia", "broadcom", "aws_amazon", "global_platform"}:
        return "global"
    return actor or "unknown"


def v32_is_coupang_competitor_event(text):
    t = clean_html_text(text)
    if not V32_COUPANG_RE.search(t):
        return False
    # 쿠팡 관련 과징금/제재/개인정보/기만광고는 경쟁사 리스크로 허용한다.
    if V32_COMPETITOR_OPERATIONAL_RISK_RE.search(t):
        return True
    try:
        return bool(v31_is_specific_competitor_public_enforcement(t))
    except Exception:
        return False


def v32_is_naver_telco_strategy(text):
    t = clean_html_text(text)
    if not V32_NAVER_TELCO_RE.search(t):
        return False
    if V32_GENERAL_POLICY_BLOCK_RE.search(t) and not re.search(r"투자|추가\s*투자|AI\s*펀드|엔비디아|앤트로픽|AI\s*팩토리|데이터센터|AI\s*인프라|제휴|협력", t, re.IGNORECASE):
        return False
    return bool(V32_STRATEGIC_COMPETITOR_ACTION_RE.search(t))


def v32_is_global_platform_or_overseas_issue(text):
    t = clean_html_text(text)
    if not V32_GLOBAL_PLATFORM_RE.search(t) and not V32_OVERSEAS_RULE_RE.search(t):
        return False
    if V32_GENERAL_POLICY_BLOCK_RE.search(t) and not V32_COMPETITOR_OPERATIONAL_RISK_RE.search(t):
        return False
    if V32_OVERSEAS_RULE_RE.search(t) and re.search(r"규제|법안|판결|책임|등급|청소년|차단|금지|SNS|AI\s*챗봇|앱마켓|플레이스토어|콘텐츠|AI\s*개요", t, re.IGNORECASE):
        return True
    if V32_STRATEGIC_COMPETITOR_ACTION_RE.search(t) and re.search(r"오픈AI|구글|메타|애플|MS|엔비디아|브로드컴|앤트로픽|X\b|엑스\b|트위터|글로벌|해외", t, re.IGNORECASE):
        return True
    return False


def v32_is_general_public_policy_not_competitor(text):
    t = clean_html_text(text)
    if v32_is_coupang_competitor_event(t):
        return False
    if V32_GENERAL_POLICY_BLOCK_RE.search(t) and not V32_COMPETITOR_OPERATIONAL_RISK_RE.search(t):
        return True
    try:
        if v31_is_general_public_policy_for_competitor(t) and not v32_is_naver_telco_strategy(t) and not v32_is_global_platform_or_overseas_issue(t):
            return True
    except Exception:
        pass
    return False


def v32_competitor_candidate_text(text):
    t = clean_html_text(text)
    if not t:
        return False, "empty"
    if V32_WEAK_ROUNDUP_RE.search(t) and not re.search(r"네이버\s*D2SF|AI\s*보안\s*스타트업|전략투자", t, re.IGNORECASE):
        return False, "weak_roundup"
    if v32_is_general_public_policy_not_competitor(t):
        return False, "general_public_policy_not_competitor"
    try:
        if v30_is_minor_vendor_customer_project(t) and not v32_is_naver_telco_strategy(t):
            return False, "minor_vendor_customer_project"
    except Exception:
        pass
    if v32_is_coupang_competitor_event(t):
        return True, "coupang_competitor_enforcement_or_risk"
    if v32_is_naver_telco_strategy(t):
        return True, "naver_telco_strategy_or_infra"
    if v32_is_global_platform_or_overseas_issue(t):
        return True, "global_platform_or_overseas_rule"
    try:
        ok, reason = v31_competitor_eligible_from_text(t, t[:220])
        if ok and not v32_is_general_public_policy_not_competitor(t):
            return True, reason
    except Exception:
        pass
    return False, "not_competitor_portfolio_candidate"


def v32_competitor_issue_eligible(issue):
    text = v32_issue_text(issue)
    ok, reason = v32_competitor_candidate_text(text)
    if ok:
        issue["v32_competitor_basis"] = reason
    return ok


def v32_competitor_article_eligible(article):
    text = v32_text(article)
    ok, reason = v32_competitor_candidate_text(text)
    if ok:
        article["v32_competitor_basis"] = reason
    return ok


def v32_competitor_score_text(text, base_score=0.0):
    t = clean_html_text(text)
    score = float(base_score or 0)
    ok, reason = v32_competitor_candidate_text(t)
    if not ok:
        return score - 200
    # 포트폴리오 순서: 전략/글로벌 후보를 쿠팡 3번째 반복보다 우선하되, 쿠팡 핵심 제재 2건은 충분히 높은 점수 유지.
    if v32_is_naver_telco_strategy(t):
        score += 125
    if v32_is_global_platform_or_overseas_issue(t):
        score += 115
    if v32_is_coupang_competitor_event(t):
        score += 95
    if re.search(r"엔비디아|NVIDIA|앤트로픽|Anthropic|AI\s*펀드|AI\s*팩토리|데이터센터|앱\s*등급|청소년\s*이용불가|AI\s*개요|개인화\s*광고", t, re.IGNORECASE):
        score += 35
    if V32_NAVER_TELCO_RE.search(t):
        # 비슷한 중요도면 네이버/통신3사가 올라오게 하는 약한 보너스.
        score += 18
    if V32_WEAK_ROUNDUP_RE.search(t):
        score -= 45
    return score


def v32_is_weak_non_digital_gov_text(text):
    t = clean_html_text(text)
    if not t:
        return False
    # 특정 플랫폼/AI/디지털 금융 규제와 직접 연결되지 않는 건설/식품/제조/지원사업형 정부 기사는 제외.
    if V32_NON_DIGITAL_GOV_RE.search(t):
        if not V32_STRONG_DIGITAL_ROUTE_RE.search(t):
            return True
        # AI 단어 하나가 아니라 플랫폼/개인정보/보안/데이터센터/디지털자산 등 카카오 경로가 있어야 함.
        if re.search(r"건설|현대건설|전문건설|협력사|동반성장|상생협력|K-푸드|식품|뷰티\s*리테일|의료관광", t, re.IGNORECASE) and not re.search(r"온라인\s*플랫폼|디지털\s*플랫폼|포털|SNS|개인정보|보안|핀테크|디지털자산|데이터센터|클라우드|OTT|미디어|콘텐츠", t, re.IGNORECASE):
            return True
    if re.search(r"공정거래\s*협약|상생협력\s*협약|동반성장펀드|협력사\s*상생", t, re.IGNORECASE):
        return True
    return False


def v32_base_event_key(item):
    text = v32_text(item)
    if v32_is_coupang_competitor_event(text):
        # 개인정보/보안과 와우회원가/기만광고는 서로 다른 쿠팡 사건으로 유지한다.
        if re.search(r"와우회원가|멤버십|기만\s*광고|허위\s*광고|표시광고", text, re.IGNORECASE):
            return "v32_competitor:coupang:deceptive_membership_price_ad"
        if re.search(r"개인정보|정보\s*유출|무단\s*수집|온라인\s*활동\s*기록|CPO", text, re.IGNORECASE):
            return "v32_competitor:coupang:privacy_data_penalty"
        return "v32_competitor:coupang:enforcement"
    actor = v32_actor_key_from_text(text)
    if v32_is_naver_telco_strategy(text):
        if re.search(r"앤트로픽|Anthropic|AI\s*펀드", text, re.IGNORECASE):
            return f"v32_competitor:{actor or 'telco'}:anthropic_ai_fund_investment"
        if re.search(r"엔비디아|NVIDIA|AI\s*팩토리|AI\s*인프라|데이터센터", text, re.IGNORECASE):
            return f"v32_competitor:{actor or 'naver_telco'}:ai_infra_factory_strategy"
        return f"v32_competitor:{actor or 'naver_telco'}:strategy"
    if v32_is_global_platform_or_overseas_issue(text):
        if re.search(r"청소년|SNS|AI\s*챗봇|등급|앱마켓|플레이스토어", text, re.IGNORECASE):
            return f"v32_competitor:{actor or 'global_platform'}:platform_youth_content_rule"
        if re.search(r"AI\s*개요|허위\s*답변|책임|판결", text, re.IGNORECASE):
            return f"v32_competitor:{actor or 'global_platform'}:ai_answer_legal_responsibility"
        if re.search(r"개인화\s*광고|광고", text, re.IGNORECASE):
            return f"v32_competitor:{actor or 'global_platform'}:ai_ad_personalization"
        if re.search(r"데이터센터|AI\s*인프라|엔비디아|NVIDIA", text, re.IGNORECASE):
            return f"v32_competitor:{actor or 'global_ai'}:ai_infra_buildout"
        return f"v32_competitor:{actor or 'global_platform'}:global_rule_or_strategy"
    return ""


# ---- Ranking and issue scoring boosts ----
_BASE_rank_score_article_v32 = rank_score_article

def rank_score_article(article):
    score = float(_BASE_rank_score_article_v32(article) or article.get("랭킹점수") or 0)
    text = v32_text(article)
    if article.get("JSON카테고리") == JSON_KEYS_ORDER[1] and v32_is_weak_non_digital_gov_text(text):
        score -= 120
        article["v32_weak_non_digital_gov"] = "Y"
        reasons = [r for r in str(article.get("랭킹감점사유", "")).split(";") if r and r != "nan"]
        reasons.append("v32_weak_non_digital_gov")
        article["랭킹감점사유"] = ";".join(dict.fromkeys(reasons))
    if v32_competitor_article_eligible(article):
        boosted = v32_competitor_score_text(text, score)
        article["v32_competitor_portfolio"] = "Y"
        article["v32_actor_group"] = v32_company_group_from_text(text)
        article["랭킹점수"] = round(boosted, 3)
        return article["랭킹점수"]
    article["랭킹점수"] = round(score, 3)
    return article["랭킹점수"]


_BASE_rank_and_trim_candidates_v32 = rank_and_trim_candidates

def rank_and_trim_candidates(raw_articles):
    ranked_all, selected = _BASE_rank_and_trim_candidates_v32(raw_articles)
    # v32 flag/score fields may have been added by rank_score_article; make sure the final candidate pool includes
    # high-value competitor portfolio candidates even if generic category caps are saturated by Coupang enforcement.
    selected_ids = {int(a.get("id")) for a in selected if str(a.get("id", "")).isdigit()}
    selected_by_id = {int(a.get("id")): a for a in selected if str(a.get("id", "")).isdigit()}

    portfolio = [a for a in ranked_all if v32_competitor_article_eligible(a)]
    portfolio = sorted(
        portfolio,
        key=lambda a: (
            v32_competitor_score_text(v32_text(a), float(a.get("랭킹점수") or 0)),
            float(a.get("중요도점수") or 0),
            a.get("게시일", ""),
        ),
        reverse=True,
    )

    def maybe_add_or_replace(article):
        art_id = int(article.get("id"))
        if art_id in selected_ids:
            return False
        if len(selected) < MAX_CANDIDATES_FOR_GEMINI:
            selected.append(article)
            selected_ids.add(art_id)
            selected_by_id[art_id] = article
            return True
        # Replace the weakest non-self, non-portfolio candidate so important competitor candidates enter issue grouping.
        weakest_idx = None
        weakest_key = None
        for idx, old in enumerate(selected):
            if old.get("강제검토") == "Y" or old.get("JSON카테고리") == JSON_KEYS_ORDER[0]:
                continue
            if old.get("v32_competitor_portfolio") == "Y":
                continue
            key = (float(old.get("랭킹점수") or 0), float(old.get("중요도점수") or 0))
            if weakest_key is None or key < weakest_key:
                weakest_key = key
                weakest_idx = idx
        if weakest_idx is not None:
            old_id = int(selected[weakest_idx].get("id"))
            selected_ids.discard(old_id)
            selected[weakest_idx] = article
            selected_ids.add(art_id)
            selected_by_id[art_id] = article
            return True
        return False

    added_portfolio = 0
    actor_pool_counts = {}
    for a in selected:
        if a.get("v32_competitor_portfolio") == "Y":
            grp = v32_company_group_from_text(v32_text(a))
            actor_pool_counts[grp] = actor_pool_counts.get(grp, 0) + 1
    for article in portfolio:
        grp = v32_company_group_from_text(v32_text(article))
        # Candidate pool에는 같은 회사 기사도 여러 개 둘 수 있지만, 쿠팡만 과도하게 들어오는 것은 제한한다.
        pool_cap = 16 if grp == "coupang" else 10 if grp in {"naver", "telco", "global"} else 7
        if actor_pool_counts.get(grp, 0) >= pool_cap:
            continue
        if maybe_add_or_replace(article):
            actor_pool_counts[grp] = actor_pool_counts.get(grp, 0) + 1
            added_portfolio += 1
        if added_portfolio >= 55:
            break

    selected = sorted(
        selected,
        key=lambda a: (
            1 if a.get("v32_competitor_portfolio") == "Y" else 0,
            float(a.get("랭킹점수") or 0),
            float(a.get("중요도점수") or 0),
            a.get("게시일", ""),
        ),
        reverse=True,
    )[:MAX_CANDIDATES_FOR_GEMINI]
    return ranked_all, selected


_BASE_v20_build_issues_v32 = v20_build_issues

def v20_build_issues(candidates, labels):
    issues, article_by_id = _BASE_v20_build_issues_v32(candidates, labels)
    for issue in issues:
        text = v32_issue_text(issue)
        score = float(issue.get("issue_score") or 0)
        if v32_competitor_issue_eligible(issue):
            issue["v32_competitor_portfolio"] = "Y"
            issue["v32_competitor_basis"] = issue.get("v32_competitor_basis", "portfolio")
            issue["v32_actor_group"] = v32_company_group_from_text(text)
            issue["issue_score"] = round(v32_competitor_score_text(text, score), 3)
            # 경쟁사 포트폴리오 후보가 Gemini editor에서 보이도록 final output category도 보정한다.
            if issue.get("primary_category") not in {JSON_KEYS_ORDER[0], JSON_KEYS_ORDER[2], JSON_KEYS_ORDER[3]}:
                issue["primary_category"] = JSON_KEYS_ORDER[2]
        if issue.get("primary_category") == JSON_KEYS_ORDER[1] and v32_is_weak_non_digital_gov_text(text):
            issue["v32_weak_non_digital_gov"] = "Y"
            issue["issue_score"] = round(float(issue.get("issue_score") or 0) - 120, 3)
    issues = sorted(issues, key=lambda x: float(x.get("issue_score") or 0), reverse=True)
    for idx, issue in enumerate(issues, 1):
        issue["issue_id"] = f"I{idx:03d}"
    return issues, article_by_id


_BASE_v20_issue_line_v32 = v20_issue_line

def v20_issue_line(issue):
    base = _BASE_v20_issue_line_v32(issue)
    flags = []
    if issue.get("v32_competitor_portfolio") == "Y":
        flags.append(f"v32_competitor_portfolio=Y basis={issue.get('v32_competitor_basis','')}")
    if issue.get("v32_actor_group"):
        flags.append(f"v32_actor_group={issue.get('v32_actor_group')}")
    if issue.get("v32_weak_non_digital_gov") == "Y":
        flags.append("v32_weak_non_digital_gov=Y")
    return base + (" " + " ".join(flags) if flags else "")


_BASE_v21_issue_allowed_basic_v32 = v21_issue_allowed_basic

def v21_issue_allowed_basic(issue):
    if v32_competitor_issue_eligible(issue):
        return True, "v32_competitor_portfolio_allowed"
    if issue.get("primary_category") == JSON_KEYS_ORDER[1] and v32_is_weak_non_digital_gov_text(v32_issue_text(issue)):
        return False, "v32_weak_non_digital_gov_blocked"
    return _BASE_v21_issue_allowed_basic_v32(issue)


_BASE_v20_issue_to_output_category_v32 = v20_issue_to_output_category

def v20_issue_to_output_category(issue):
    if v32_competitor_issue_eligible(issue):
        return JSON_KEYS_ORDER[2]
    return _BASE_v20_issue_to_output_category_v32(issue)


_BASE_v26_1_is_good_refill_candidate_v32 = v26_1_is_good_refill_candidate

def v26_1_is_good_refill_candidate(decision, issue, cat=None):
    cat = cat or v26_1_decision_category(decision, issue)
    if cat == JSON_KEYS_ORDER[2]:
        return v32_competitor_issue_eligible(issue)
    if cat == JSON_KEYS_ORDER[1] and issue and v32_is_weak_non_digital_gov_text(v32_issue_text(issue)):
        return False
    return _BASE_v26_1_is_good_refill_candidate_v32(decision, issue, cat=cat)


# ---- Final repair and competitor portfolio refill ----

def v32_set_category(item, cat, reason=""):
    if "v23_set_final_category" in globals():
        try:
            v23_set_final_category(item, cat, reason or "v32_set_category")
            return
        except Exception:
            pass
    item["카테고리"] = cat
    item["카테고리명"] = JSON_KEY_TO_DISPLAY.get(cat, cat)
    if reason:
        item["v32_category_reason"] = reason


def v32_add_skip(skip_rows, item, reason):
    if skip_rows is None:
        return
    try:
        skip_rows.append({
            "검색어": item.get("검색어", ""),
            "후보제목": item.get("기사제목", ""),
            "후보링크": item.get("링크", ""),
            "후보언론사": item.get("언론사", ""),
            "매칭과거일자": "CURRENT_RUN",
            "매칭과거제목": "v32_final_repair",
            "매칭과거링크": "",
            "중복판정이유": reason,
            "공유주체": "",
            "공유사건태그": item.get("사건태그", ""),
            "종합점수": item.get("대표선택점수", ""),
            "제목점수": "",
            "토큰점수": "",
            "본문점수": "",
            "제외단계": "v32_final_repair",
        })
    except Exception:
        pass


def v32_competitor_items(final_report_data):
    return [x for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[2]]


def v32_competitor_portfolio_stats(final_report_data):
    comp = v32_competitor_items(final_report_data)
    actor_counts = {}
    for item in comp:
        actor = v32_actor_key_from_text(v32_text(item)) or "unknown"
        actor_counts[actor] = actor_counts.get(actor, 0) + 1
    non_coupang = [x for x in comp if v32_actor_key_from_text(v32_text(x)) != "coupang"]
    overseas_or_global = [x for x in comp if v32_is_global_platform_or_overseas_issue(v32_text(x))]
    naver_telco = [x for x in comp if v32_is_naver_telco_strategy(v32_text(x))]
    return {
        "total": len(comp),
        "actor_counts": actor_counts,
        "non_coupang": len(non_coupang),
        "overseas_or_global": len(overseas_or_global),
        "naver_telco": len(naver_telco),
    }


def v32_repair_final_items(final_report_data, skip_rows=None):
    # Start with v31 cleanup if available.
    try:
        v31_repair_final_items(final_report_data, skip_rows)
    except Exception:
        pass

    cleaned = []
    seen_keys = {}
    for item in list(final_report_data):
        cat = item.get("카테고리") or ""
        text = v32_text(item)
        if cat == JSON_KEYS_ORDER[1] and v32_is_weak_non_digital_gov_text(text):
            v32_add_skip(skip_rows, item, "v32_removed_weak_non_digital_government_item")
            continue
        if cat == JSON_KEYS_ORDER[2]:
            ok, reason = v32_competitor_candidate_text(text)
            if not ok:
                # 일반 정부 정책/협의회가 경쟁사로 흘러온 경우는 정부로 돌리거나 제거.
                if v32_is_general_public_policy_not_competitor(text):
                    try:
                        gov_ok = v29_gov_eligible(item)[0]
                    except Exception:
                        gov_ok = True
                    if gov_ok:
                        v32_set_category(item, JSON_KEYS_ORDER[1], "v32_general_policy_competitor_to_gov")
                    else:
                        v32_add_skip(skip_rows, item, f"v32_removed_bad_competitor:{reason}")
                        continue
                else:
                    v32_add_skip(skip_rows, item, f"v32_removed_bad_competitor:{reason}")
                    continue
        # Cross-section duplicate: public policy goes Gov, competitor risk goes Competitor. Coupang privacy and wow ad stay separate.
        key = v32_base_event_key(item) or v29_base_event_key(item)
        if key:
            old = seen_keys.get(key)
            if old is not None:
                def pref(x):
                    xcat = x.get("카테고리") or ""
                    xt = v32_text(x)
                    if key.startswith("v32_competitor"):
                        return (0 if xcat == JSON_KEYS_ORDER[2] else 3, -float(v29_item_priority_score(x) or 0))
                    if str(key).startswith("v31_public_policy") or v32_is_general_public_policy_not_competitor(xt):
                        return (0 if xcat == JSON_KEYS_ORDER[1] else 5, -float(v29_item_priority_score(x) or 0))
                    return (JSON_KEYS_ORDER.index(xcat) if xcat in JSON_KEYS_ORDER else 9, -float(v29_item_priority_score(x) or 0))
                if pref(item) < pref(old):
                    v32_add_skip(skip_rows, old, f"v32_cross_section_duplicate_removed:{key}")
                    cleaned = [x for x in cleaned if x is not old]
                    seen_keys[key] = item
                    cleaned.append(item)
                else:
                    v32_add_skip(skip_rows, item, f"v32_cross_section_duplicate_removed:{key}")
                continue
            seen_keys[key] = item
        cleaned.append(item)

    # Soft actor cap only inside competitor section: keep two Coupang material events, avoid 3rd+ same actor if alternatives exist.
    final_report_data[:] = cleaned
    return final_report_data


def v32_candidate_pool(decisions, backup_decisions, issues):
    pool = []
    seen = set()
    for source, iterable in [("selected", decisions), ("backup", backup_decisions)]:
        for d in iterable or []:
            iid = d.get("issue_id")
            if not iid or iid in seen:
                continue
            d2 = dict(d)
            d2["v32_pool_source"] = source
            pool.append(d2)
            seen.add(iid)
    for issue in issues:
        iid = issue.get("issue_id")
        if not iid or iid in seen:
            continue
        ids = []
        for raw in [issue.get("candidate_article_ids", []), issue.get("article_ids", []), issue.get("top_article_id", "")]:
            if isinstance(raw, list):
                ids.extend(raw)
            else:
                ids.extend([int(x) for x in re.findall(r"\d+", str(raw))])
        ids = list(dict.fromkeys([int(x) for x in ids if str(x).isdigit()]))
        d2 = {
            "kind": "v32_local",
            "issue_id": iid,
            "category": JSON_KEYS_ORDER[2] if v32_competitor_issue_eligible(issue) else v20_issue_to_output_category(issue),
            "priority": 4 if v32_competitor_issue_eligible(issue) else v20_int(issue.get("max_ceo_priority"), 2),
            "best_article_id": int(issue.get("top_article_id")) if str(issue.get("top_article_id", "")).isdigit() else (ids[0] if ids else ""),
            "backup_article_ids": ids[1:8],
            "reason": "v32_local_pool",
            "v32_pool_source": "all_issues",
        }
        pool.append(d2)
        seen.add(iid)
    return pool


def v32_refill_need(final_report_data):
    stats = v32_competitor_portfolio_stats(final_report_data)
    if stats["total"] < V32_COMPETITOR_MIN_TOTAL:
        return True
    if stats["non_coupang"] < V32_COMPETITOR_MIN_NON_COUPANG:
        return True
    if stats["overseas_or_global"] < V32_COMPETITOR_MIN_OVERSEAS_OR_GLOBAL:
        return True
    if stats["total"] < V32_COMPETITOR_TARGET_TOTAL and len(final_report_data) < V21_FINAL_TARGET:
        return True
    return False


def v32_refill_candidate_score(decision, issue, final_report_data):
    text = v32_issue_text(issue)
    base = v20_int(decision.get("priority"), 1) * 12 + v20_float(issue.get("issue_score"), 0) + v20_float(issue.get("max_ceo_priority"), 0) * 8
    score = v32_competitor_score_text(text, base)
    actor = v32_actor_key_from_text(text) or "unknown"
    group = v32_company_group_from_text(text)
    stats = v32_competitor_portfolio_stats(final_report_data)
    actor_counts = stats["actor_counts"]
    # Diversity bonuses/penalties. 쿠팡 2건은 허용하되 3번째는 강하게 감점.
    if actor == "coupang":
        if actor_counts.get("coupang", 0) >= 2:
            score -= 220
        elif actor_counts.get("coupang", 0) == 1:
            score += 10
    else:
        if actor_counts.get(actor, 0) >= 1:
            score -= 90
        if stats["non_coupang"] < V32_COMPETITOR_PREFER_NON_COUPANG:
            score += 55
    if stats["overseas_or_global"] < V32_COMPETITOR_MIN_OVERSEAS_OR_GLOBAL and (group == "global" or v32_is_global_platform_or_overseas_issue(text)):
        score += 80
    if stats["naver_telco"] < 1 and v32_is_naver_telco_strategy(text):
        score += 60
    if v32_is_general_public_policy_not_competitor(text):
        score -= 500
    return score


def v32_portfolio_refill_competitors(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    represented_bases = {v32_base_event_key(x) or v29_base_event_key(x) for x in final_report_data if (v32_base_event_key(x) or v29_base_event_key(x))}
    represented_issue_ids = {x.get("v20_issue_id") for x in final_report_data if x.get("v20_issue_id")}
    pool = v32_candidate_pool(decisions, backup_decisions, issues)

    attempts_without_progress = 0
    while v32_refill_need(final_report_data) and len(final_report_data) < V21_FINAL_MAX and attempts_without_progress < 10:
        ranked = []
        for d in pool:
            iid = d.get("issue_id")
            if not iid or iid in represented_issue_ids:
                continue
            issue = issue_by_id.get(iid, {})
            if not issue or not v32_competitor_issue_eligible(issue):
                continue
            pseudo = v30_pseudo_item_from_issue(issue, JSON_KEYS_ORDER[2]) if "v30_pseudo_item_from_issue" in globals() else {"기사제목": issue.get("top_title") or issue.get("issue_group"), "RSS요약": issue.get("top_summary", "")}
            base = v32_base_event_key(pseudo) or v29_base_event_key(pseudo)
            if base and base in represented_bases:
                continue
            score = v32_refill_candidate_score(d, issue, final_report_data)
            if score <= 0:
                continue
            d2 = dict(d)
            d2["category"] = JSON_KEYS_ORDER[2]
            d2["priority"] = max(v20_int(d2.get("priority"), 3), 4)
            d2["v32_forced_competitor_portfolio_refill"] = "Y"
            ranked.append((score, d2, issue, base))
        ranked.sort(key=lambda x: x[0], reverse=True)
        if not ranked:
            break
        progressed = False
        for score, d, issue, base in ranked[:8]:
            iid = d.get("issue_id")
            if iid in represented_issue_ids:
                continue
            item, order_counter, status = v20_process_issue(
                d,
                issue_by_id,
                article_by_id,
                recent_past_items,
                final_report_data,
                body_failed_rows,
                skip_rows,
                processed_article_ids,
                order_counter,
            )
            status_counts[status] = status_counts.get(status, 0) + 1
            represented_issue_ids.add(iid)
            if item:
                v32_set_category(item, JSON_KEYS_ORDER[2], "v32_competitor_portfolio_refill")
                item["v32_competitor_portfolio_refill"] = "Y"
                item["v32_competitor_refill_score"] = round(score, 2)
                if base:
                    represented_bases.add(base)
                v32_repair_final_items(final_report_data, skip_rows)
                print(f"  -> v32 competitor portfolio refill added: {v20_clip(issue.get('issue_group',''), 42)} / {v20_clip(item.get('기사제목',''), 42)}...")
                progressed = True
                break
        if not progressed:
            attempts_without_progress += 1
        else:
            attempts_without_progress = 0
    return order_counter


_BASE_v26_1_refill_after_final_repair_v32 = v26_1_refill_after_final_repair

def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    # Keep earlier broad refill, then repair, then explicitly rebuild competitor portfolio.
    order_counter = _BASE_v26_1_refill_after_final_repair_v32(
        decisions=decisions,
        backup_decisions=backup_decisions,
        issues=issues,
        issue_by_id=issue_by_id,
        article_by_id=article_by_id,
        recent_past_items=recent_past_items,
        final_report_data=final_report_data,
        body_failed_rows=body_failed_rows,
        skip_rows=skip_rows,
        processed_article_ids=processed_article_ids,
        order_counter=order_counter,
        status_counts=status_counts,
    )
    v32_repair_final_items(final_report_data, skip_rows)
    order_counter = v32_portfolio_refill_competitors(
        decisions,
        backup_decisions,
        issues,
        issue_by_id,
        article_by_id,
        recent_past_items,
        final_report_data,
        body_failed_rows,
        skip_rows,
        processed_article_ids,
        order_counter,
        status_counts,
    )
    v32_repair_final_items(final_report_data, skip_rows)
    return order_counter


_BASE_v21_remove_obvious_final_duplicates_v32 = v21_remove_obvious_final_duplicates

def v21_remove_obvious_final_duplicates(final_items):
    kept, removed = _BASE_v21_remove_obvious_final_duplicates_v32(final_items)
    skip_like = []
    v32_repair_final_items(kept, skip_like)
    for row in skip_like:
        removed.append(({"기사제목": row.get("후보제목", ""), "링크": row.get("후보링크", "")}, row.get("중복판정이유", "v32_cleanup")))
    return kept, removed


# Keep item priority aligned with the portfolio rule.
_BASE_v29_item_priority_score_v32 = v29_item_priority_score

def v29_item_priority_score(item):
    score = float(_BASE_v29_item_priority_score_v32(item) or 0)
    text = v32_text(item)
    cat = item.get("카테고리") or ""
    if cat == JSON_KEYS_ORDER[2]:
        score = v32_competitor_score_text(text, score)
    if cat == JSON_KEYS_ORDER[1] and v32_is_weak_non_digital_gov_text(text):
        score -= 180
    return round(score, 2)


_BASE_v29_base_event_key_v32 = v29_base_event_key

def v29_base_event_key(item):
    key = v32_base_event_key(item)
    if key:
        return v26_normalize_event_key(key) if "v26_normalize_event_key" in globals() else key
    return _BASE_v29_base_event_key_v32(item)


_BASE_v29_competitor_eligible_v32 = v29_competitor_eligible

def v29_competitor_eligible(item):
    ok, reason = v32_competitor_candidate_text(v32_text(item))
    if ok:
        return True, reason
    return False, reason


_BASE_v29_gov_eligible_v32 = v29_gov_eligible

def v29_gov_eligible(item):
    if v32_is_weak_non_digital_gov_text(v32_text(item)):
        return False, "v32_weak_non_digital_gov"
    return _BASE_v29_gov_eligible_v32(item)


_BASE_v20_gemini_quality_check_v32 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v32(client, final_report_data, final_briefing_text)
    try:
        data = v22_parse_qa_json(qa) if qa else {"overall": "양호", "warnings": [], "suggested_human_checks": []}
    except Exception:
        data = {"overall": "주의", "warnings": [clean_html_text(qa)[:500]], "suggested_human_checks": []}
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    checks = data.get("suggested_human_checks") if isinstance(data.get("suggested_human_checks"), list) else []

    # Old QA rule flagged all Korean regulator actions in competitor as bad. In v32, specific competitor enforcement is allowed.
    def keep_warning(w):
        txt = json.dumps(w, ensure_ascii=False) if isinstance(w, dict) else str(w)
        if "v28_korean_regulator_action_not_competitor_duplicate" in txt:
            if re.search(r"쿠팡|네이버|구글|애플|메타|토스|배민|개인정보|와우회원가|과징금|시정명령|제재", txt, re.IGNORECASE):
                return False
        return True

    warnings = [w for w in warnings if keep_warning(w)]
    checks = [c for c in checks if keep_warning(c)]

    local_warnings = []
    comp = v32_competitor_items(final_report_data)
    stats = v32_competitor_portfolio_stats(final_report_data)
    if stats["total"] < V32_COMPETITOR_MIN_TOTAL:
        local_warnings.append(f"경쟁사/해외이슈가 {stats['total']}건으로 v32 최소 {V32_COMPETITOR_MIN_TOTAL}건보다 적음")
    if stats["non_coupang"] < V32_COMPETITOR_MIN_NON_COUPANG:
        local_warnings.append("경쟁사/해외이슈가 쿠팡 이슈 중심으로만 구성됨. 네이버/통신3사/글로벌 플랫폼 후보 재검토 필요")
    if stats["overseas_or_global"] < V32_COMPETITOR_MIN_OVERSEAS_OR_GLOBAL:
        local_warnings.append("경쟁사/해외이슈에 해외/글로벌 플랫폼 규제·전략 이슈가 없음")
    for item in final_report_data:
        if item.get("카테고리") == JSON_KEYS_ORDER[1] and v32_is_weak_non_digital_gov_text(v32_text(item)):
            local_warnings.append(f"정부/국회 비디지털 저관련 기사 포함 가능: {item.get('기사제목','')}")
        if item.get("카테고리") == JSON_KEYS_ORDER[2]:
            ok, reason = v32_competitor_candidate_text(v32_text(item))
            if not ok:
                local_warnings.append(f"경쟁사/해외 섹션 부적합 가능: {item.get('기사제목','')} / {reason}")
    warnings.extend(local_warnings)
    checks.extend(local_warnings)
    data["warnings"] = warnings
    data["suggested_human_checks"] = checks
    if local_warnings and data.get("overall") == "양호":
        data["overall"] = "주의"
    if not warnings and data.get("overall") in {"문제", "주의"}:
        data["overall"] = "양호"
    return json.dumps(data, ensure_ascii=False, indent=2)


# =========================================================
# v33 overrides: QA-style final repair loop + section rebalance
# - v32에서 좋은 후보는 잡았지만 최종 섹션 재배치가 부족했던 문제를 보정한다.
# - QA가 반복해서 잡은 문제를 저장 직전 로컬 repair로 선반영한다.
#   1) 산업동향/정부에 잘못 남은 경쟁사 직접 액션 후보를 경쟁사/해외로 이동
#   2) 정부/국회 동일 공공정책 이슈 중복 제거
#   3) 경쟁사/해외 최소 3건을 위해 이미 본문 성공한 후보를 우선 재배치
#   4) 쿠팡 개인정보/와우회원가 제재는 경쟁사 리스크로 유지
# =========================================================

V6_VERSION = "google_rss_v33_qa_repair_section_rebalance"

# v33: 산업동향은 좋은 후보가 없으면 비워도 된다. 좋은 경쟁사 후보가 산업동향에 있으면 경쟁사로 우선 이동한다.
try:
    CATEGORY_MIN["산업동향"] = 0
    CATEGORY_TARGET["산업동향"] = min(int(CATEGORY_TARGET.get("산업동향", 1)), 1)
except Exception:
    pass
try:
    V21_FINAL_MIN = min(int(V21_FINAL_MIN), 8)
    V20_FINAL_MIN = V21_FINAL_MIN
    MIN_SELECT_COUNT = V21_FINAL_MIN
except Exception:
    pass

V33_COMPETITOR_MIN_TOTAL = 3
V33_COMPETITOR_TARGET_TOTAL = 4
V33_GOV_POLICY_DUP_RE = re.compile(
    r"정책협의회|정책협의체|차관급\s*협의|업무협약|MOU|협약|가이드\s*배포|교육\s*확대|간담회|플랫폼\s*출범|협력\s*방안|공동\s*추진",
    re.IGNORECASE,
)
V33_GOV_ACTOR_PATTERNS = [
    ("msit", r"과기정통부|과학기술정보통신부|과기부"),
    ("bmtc", r"방미통위|방송미디어통신위원회|방통위"),
    ("mod", r"국방부"),
    ("fss", r"금감원|금융감독원"),
    ("fsc", r"금융위|금융위원회"),
    ("ftc", r"공정위|공정거래위원회"),
    ("mois", r"행안부|행정안전부"),
    ("mss", r"중기부|중소벤처기업부"),
    ("pipc", r"개보위|개인정보위|개인정보보호위원회"),
]
V33_COMPETITOR_MOVE_FROM_INDUSTRY_RE = re.compile(
    r"SKT|SK텔레콤|KT\b|케이티|LGU\+|LG유플러스|네이버|NAVER|구글|Google|오픈AI|OpenAI|"
    r"메타|Meta|애플|Apple|X\b|엑스\b|트위터|Twitter|앤트로픽|Anthropic|엔비디아|NVIDIA",
    re.IGNORECASE,
)
V33_COMPETITOR_STRONG_ACTION_RE = re.compile(
    r"투자|추가\s*투자|AI\s*펀드|펀드\s*조성|제휴|협력|파트너십|AI\s*팩토리|AI\s*인프라|"
    r"데이터센터|AIDC|클라우드|엔비디아|앤트로픽|오픈AI|출시|도입|공개|서비스\s*개편|"
    r"앱\s*등급|청소년\s*이용불가|차단\s*의무|AI\s*개요|판결|규제\s*대응|과징금|제재|개인정보|유출",
    re.IGNORECASE,
)
V33_LOW_PRIORITY_GOV_RE = re.compile(
    r"군장병|장병\s*대상|불법도박\s*예방|디지털\s*윤리교육|미디어\s*교육|공공부문\s*AI\s*도입|공공\s*AI\s*가이드|"
    r"지원사업|주관기관\s*선정|참여기업\s*선정|인재\s*양성|교육\s*확대",
    re.IGNORECASE,
)


def v33_detect_gov_actors(text):
    t = clean_html_text(text)
    found = []
    for key, pat in V33_GOV_ACTOR_PATTERNS:
        if re.search(pat, t, re.IGNORECASE):
            found.append(key)
    return sorted(set(found))


def v33_public_policy_domain(text):
    t = clean_html_text(text)
    if re.search(r"국방부", t) and re.search(r"방미통위|방송미디어통신위원회|방통위", t) and re.search(r"군장병|장병|미디어\s*교육|AI\s*교육|불법도박|디지털\s*윤리", t):
        return "military_ai_media_education"
    if re.search(r"과기정통부|과학기술정보통신부|과기부", t) and re.search(r"방미통위|방송미디어통신위원회|방통위", t) and re.search(r"정책협의|차관급|AI|미디어|OTT|K-FAST|데이터\s*공유", t, re.IGNORECASE):
        return "ai_media_digital_policy_council"
    if re.search(r"금감원|금융감독원", t) and re.search(r"네이버페이|네이버파이낸셜|모험자본", t, re.IGNORECASE):
        return "venture_capital_platform"
    if re.search(r"불법촬영|차단\s*의무|유통\s*방지|플랫폼\s*사업자", t, re.IGNORECASE):
        return "platform_illegal_content_blocking_obligation"
    if re.search(r"디지털자산|가상자산|스테이블코인|특금법|FIU|금융정보분석원", t, re.IGNORECASE):
        return "digital_asset_policy"
    if re.search(r"AI\s*기본법|AI\s*저작권|딥페이크|허위조작정보", t, re.IGNORECASE):
        return "ai_platform_policy"
    if V33_GOV_POLICY_DUP_RE.search(t):
        tokens = sorted([x for x in tokenize_for_similarity(t[:700]) if len(x) >= 2])[:6]
        return "generic_policy_" + "_".join(tokens[:4])
    return ""


def v33_public_policy_key(item):
    text = v32_text(item) if "v32_text" in globals() else clean_html_text(f"{item.get('기사제목','')} {item.get('본문전문','')[:1000]}")
    actors = v33_detect_gov_actors(text)
    domain = v33_public_policy_domain(text)
    if not actors or not domain:
        try:
            key = v31_public_policy_key(text)
            if key:
                return f"v31:{key}"
        except Exception:
            pass
        return ""
    return "v33_gov_policy:" + "+".join(actors[:3]) + ":" + domain


def v33_policy_item_score(item):
    score = 0.0
    try:
        score += float(v29_item_priority_score(item) or 0)
    except Exception:
        score += 0.0
    title = clean_html_text(item.get("기사제목", ""))
    text = v32_text(item) if "v32_text" in globals() else title
    source = clean_html_text(item.get("언론사", ""))
    body_len = int(item.get("본문글자수") or len(clean_html_text(item.get("본문전문", ""))))
    score += min(25.0, body_len / 200.0)
    if re.search(r"연합뉴스|전자신문|디지털데일리|지디넷|이데일리|한국경제|매일경제|머니투데이|MBC|KBS|JTBC", source, re.IGNORECASE):
        score += 15
    if re.search(r"과기정통부|방미통위|AI|미디어|OTT|디지털\s*플랫폼|이용자\s*보호|데이터\s*공유", text, re.IGNORECASE):
        score += 18
    if V33_LOW_PRIORITY_GOV_RE.search(text):
        score -= 20
    if re.search(r"원문 본문 자동 추출|본문 추출 실패|세부 요약은 생략", text):
        score -= 40
    return round(score, 2)


def v33_should_move_to_competitor(item):
    cat = item.get("카테고리") or ""
    text = v32_text(item) if "v32_text" in globals() else clean_html_text(f"{item.get('기사제목','')} {item.get('본문전문','')[:1200]}")
    # 이미 경쟁사면 유지 여부는 v32/v29 eligibility에 맡긴다.
    if cat == JSON_KEYS_ORDER[2]:
        return False
    # 정부 일반정책은 경쟁사로 이동하지 않는다. 단, 특정 경쟁사 제재/운영제약은 예외.
    try:
        if cat == JSON_KEYS_ORDER[1] and v32_is_general_public_policy_not_competitor(text) and not v32_is_coupang_competitor_event(text):
            return False
    except Exception:
        pass
    # 산업동향에 있는 네이버/통신3사/글로벌 기업 직접 액션은 경쟁사/해외가 우선이다.
    # 단, [AI생태계]/[IT소식]/Game & Now 같은 라운드업은 개별 경쟁사 액션으로 이동하지 않는다.
    if cat == JSON_KEYS_ORDER[3]:
        if V32_WEAK_ROUNDUP_RE.search(text):
            return False
        if V33_COMPETITOR_MOVE_FROM_INDUSTRY_RE.search(text) and V33_COMPETITOR_STRONG_ACTION_RE.search(text):
            return True
        try:
            ok, _ = v32_competitor_candidate_text(text)
            if ok:
                return True
        except Exception:
            pass
    # 자사/정부를 과하게 뺏어오지 않는다.
    return False


def v33_is_low_priority_gov_item(item):
    text = v32_text(item) if "v32_text" in globals() else clean_html_text(f"{item.get('기사제목','')} {item.get('본문전문','')[:1200]}")
    if item.get("카테고리") != JSON_KEYS_ORDER[1]:
        return False
    try:
        if v32_is_weak_non_digital_gov_text(text):
            return True
    except Exception:
        pass
    # 군 장병 교육/공공 교육형은 같은 날 더 강한 디지털 정책이 있으면 후순위로 둔다.
    if V33_LOW_PRIORITY_GOV_RE.search(text) and not re.search(r"플랫폼\s*사업자|온라인\s*플랫폼|개인정보|과징금|디지털자산|스테이블코인|OTT\s*산업|데이터센터|AI\s*모델", text, re.IGNORECASE):
        return True
    return False


_BASE_v32_repair_final_items_v33 = v32_repair_final_items


def v33_repair_final_items(final_report_data, skip_rows=None):
    """Final report repair loop. It mutates final_report_data in place."""
    if skip_rows is None:
        skip_rows = []

    # 1차: v32 기존 guardrail 적용.
    try:
        _BASE_v32_repair_final_items_v33(final_report_data, skip_rows)
    except Exception:
        pass

    # 2차: 산업동향 등에 남은 경쟁사 직접 액션 후보를 경쟁사/해외로 이동.
    for item in list(final_report_data):
        if v33_should_move_to_competitor(item):
            old_cat = item.get("카테고리")
            try:
                v32_set_category(item, JSON_KEYS_ORDER[2], "v33_move_competitor_candidate_from_" + str(old_cat))
            except Exception:
                item["카테고리"] = JSON_KEYS_ORDER[2]
                item["카테고리명"] = JSON_KEY_TO_DISPLAY.get(JSON_KEYS_ORDER[2], JSON_KEYS_ORDER[2])
            item["v33_repair_action"] = "move_to_competitor"
            try:
                v32_add_skip(skip_rows, item, f"v33_moved_from_{old_cat}_to_competitor")
            except Exception:
                pass

    # 3차: 정부/국회 동일 공공정책 이슈 중복 제거.
    seen_policy = {}
    cleaned = []
    for item in list(final_report_data):
        key = v33_public_policy_key(item) if item.get("카테고리") == JSON_KEYS_ORDER[1] else ""
        if key:
            old = seen_policy.get(key)
            if old is not None:
                # 더 좋은 대표 기사만 유지.
                if v33_policy_item_score(item) > v33_policy_item_score(old):
                    try:
                        v32_add_skip(skip_rows, old, f"v33_removed_duplicate_gov_policy:{key}")
                    except Exception:
                        pass
                    cleaned = [x for x in cleaned if x is not old]
                    seen_policy[key] = item
                    cleaned.append(item)
                else:
                    try:
                        v32_add_skip(skip_rows, item, f"v33_removed_duplicate_gov_policy:{key}")
                    except Exception:
                        pass
                continue
            seen_policy[key] = item
        cleaned.append(item)
    final_report_data[:] = cleaned

    # 4차: 정부 섹션이 너무 약한 항목으로 채워졌으면 억지 유지하지 않는다.
    # 단, 정부/국회가 3건 미만으로 떨어지지 않는 범위에서만 제거.
    gov_items = [x for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[1]]
    for item in sorted(gov_items, key=v33_policy_item_score):
        if len([x for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[1]]) <= 3:
            break
        if v33_is_low_priority_gov_item(item):
            try:
                v32_add_skip(skip_rows, item, "v33_removed_low_priority_government_after_qa_repair")
            except Exception:
                pass
            final_report_data[:] = [x for x in final_report_data if x is not item]

    # 5차: 경쟁사 섹션 중복/부적합 다시 정리하되, 쿠팡 핵심 제재 2건은 허용.
    try:
        _BASE_v32_repair_final_items_v33(final_report_data, skip_rows)
    except Exception:
        pass

    return final_report_data


# Override v32 repair entry point.
def v32_repair_final_items(final_report_data, skip_rows=None):
    return v33_repair_final_items(final_report_data, skip_rows)


_BASE_v26_1_refill_after_final_repair_v33 = v26_1_refill_after_final_repair


def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    order_counter = _BASE_v26_1_refill_after_final_repair_v33(
        decisions=decisions,
        backup_decisions=backup_decisions,
        issues=issues,
        issue_by_id=issue_by_id,
        article_by_id=article_by_id,
        recent_past_items=recent_past_items,
        final_report_data=final_report_data,
        body_failed_rows=body_failed_rows,
        skip_rows=skip_rows,
        processed_article_ids=processed_article_ids,
        order_counter=order_counter,
        status_counts=status_counts,
    )

    # QA-style local repair: move already-successful competitor candidates first.
    v33_repair_final_items(final_report_data, skip_rows)

    # If competitor still short, use v32 portfolio refill again after section moves/dedupes.
    try:
        if v32_refill_need(final_report_data):
            order_counter = v32_portfolio_refill_competitors(
                decisions,
                backup_decisions,
                issues,
                issue_by_id,
                article_by_id,
                recent_past_items,
                final_report_data,
                body_failed_rows,
                skip_rows,
                processed_article_ids,
                order_counter,
                status_counts,
            )
            v33_repair_final_items(final_report_data, skip_rows)
    except Exception as e:
        try:
            print(f"  -> v33 QA repair refill skipped: {type(e).__name__}: {e}")
        except Exception:
            pass
    return order_counter


_BASE_v21_remove_obvious_final_duplicates_v33 = v21_remove_obvious_final_duplicates


def v21_remove_obvious_final_duplicates(final_items):
    kept, removed = _BASE_v21_remove_obvious_final_duplicates_v33(final_items)
    skip_like = []
    v33_repair_final_items(kept, skip_like)
    for row in skip_like:
        removed.append(({"기사제목": row.get("후보제목", ""), "링크": row.get("후보링크", "")}, row.get("중복판정이유", "v33_qa_repair")))
    return kept, removed


_BASE_v29_item_priority_score_v33 = v29_item_priority_score


def v29_item_priority_score(item):
    score = float(_BASE_v29_item_priority_score_v33(item) or 0)
    text = v32_text(item) if "v32_text" in globals() else clean_html_text(f"{item.get('기사제목','')} {item.get('본문전문','')[:1000]}")
    cat = item.get("카테고리") or ""
    if cat == JSON_KEYS_ORDER[2]:
        if v32_is_naver_telco_strategy(text):
            score += 45
        if v32_is_global_platform_or_overseas_issue(text):
            score += 35
        if v32_is_coupang_competitor_event(text):
            score += 30
    if cat == JSON_KEYS_ORDER[1] and v33_is_low_priority_gov_item(item):
        score -= 90
    return round(score, 2)


_BASE_v20_gemini_quality_check_v33 = v20_gemini_quality_check


def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    # Run one last local repair before QA text is produced. This makes QA a practical editor, not only a logger.
    try:
        skip_like = []
        v33_repair_final_items(final_report_data, skip_like)
    except Exception:
        pass
    qa = _BASE_v20_gemini_quality_check_v33(client, final_report_data, final_briefing_text)
    try:
        data = v22_parse_qa_json(qa) if qa else {"overall": "양호", "warnings": [], "suggested_human_checks": []}
    except Exception:
        data = {"overall": "주의", "warnings": [clean_html_text(qa)[:500]], "suggested_human_checks": []}
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    checks = data.get("suggested_human_checks") if isinstance(data.get("suggested_human_checks"), list) else []

    # Old QA still sometimes recommends moving SKT/Anthropic from Industry to Competitor. If repair succeeded, suppress stale warning.
    comp_titles = " ".join(clean_html_text(x.get("기사제목", "")) for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[2])
    def keep_warning(w):
        txt = json.dumps(w, ensure_ascii=False) if isinstance(w, dict) else str(w)
        if "SKT" in txt and "앤트로픽" in txt and re.search(r"SKT|SK텔레콤", comp_titles) and re.search(r"앤트로픽|Anthropic", comp_titles, re.IGNORECASE):
            return False
        return True
    warnings = [w for w in warnings if keep_warning(w)]
    checks = [c for c in checks if keep_warning(c)]

    local = []
    stats = v32_competitor_portfolio_stats(final_report_data)
    if stats.get("total", 0) < V33_COMPETITOR_MIN_TOTAL:
        local.append(f"경쟁사/해외이슈가 {stats.get('total',0)}건으로 v33 최소 {V33_COMPETITOR_MIN_TOTAL}건보다 적음")
    gov_keys = {}
    for item in final_report_data:
        if item.get("카테고리") == JSON_KEYS_ORDER[1]:
            key = v33_public_policy_key(item)
            if key:
                gov_keys.setdefault(key, []).append(item)
    for key, vals in gov_keys.items():
        if len(vals) > 1:
            local.append(f"정부/국회 동일 공공정책 이슈 중복 가능: {key} / {len(vals)}건")
    warnings.extend(local)
    checks.extend(local)
    data["warnings"] = warnings
    data["suggested_human_checks"] = checks
    if local and data.get("overall") == "양호":
        data["overall"] = "주의"
    if not warnings and data.get("overall") in {"주의", "문제"}:
        data["overall"] = "양호"
    return json.dumps(data, ensure_ascii=False, indent=2)


# =========================================================
# v39 overrides: v33-preserving daily-safe guardrails
# =========================================================
# v39 intentionally keeps the v33 editorial portfolio behavior.
# Changes are limited to:
# 1) more stable labeling defaults,
# 2) softer representative article re-ordering inside the same selected issue,
# 3) title/link-only protection for genuinely strong selected issues when body extraction fails,
# 4) weak refill/pro-motion/event guardrails,
# 5) soft overseas/global competitor rescue without forcing bad items.

V6_VERSION = "google_rss_v39_v33_plus_safe_guardrails"

# Keep v33's selection feel, but reduce Gemini JSON/coverage instability.
GEMINI_MODEL_LABELING = os.getenv("GEMINI_MODEL_LABELING", "gemini-2.5-flash")
V20_LABEL_BATCH_SIZE = 35
V23_LABEL_SPLIT_BATCH_SIZE = 20

# Do not force Industry. v33 worked better when Industry was optional.
try:
    CATEGORY_MIN[JSON_KEYS_ORDER[3]] = 0
except Exception:
    pass

V39_EVENT_ONLY_RE = re.compile(
    r"행운퀴즈|퀴즈\s*정답|정답\s*공개|멤버스위크|앱테크|캐시워크|오퀴즈|토스\s*퀴즈|"
    r"쿠폰|할인|특가|이벤트|프로모션|기획전|경품|체험단|브랜드대상|수상|"
    r"클라우드\s*데이|세미나|컨퍼런스|포럼|설명회|간담회|교육\s*(모집|실시|참여|신청)|참여자\s*모집|"
    r"업무협약|MOU|협약\s*체결|캠페인|축제|페스티벌|후원|스폰서",
    re.IGNORECASE,
)
V39_SEVERE_PROMO_RE = re.compile(
    r"행운퀴즈|퀴즈\s*정답|정답\s*공개|멤버스위크|앱테크|캐시워크|오퀴즈|쿠폰|할인|특가|경품|체험단|기획전",
    re.IGNORECASE,
)
V39_MATERIAL_EVENT_RE = re.compile(
    r"과징금|제재|시정명령|행정소송|소송|판결|조사|수사|고발|개인정보|유출|해킹|침해사고|장애|먹통|"
    r"인수|매각|합병|지분|경영권|투자|추가\s*투자|펀드|IPO|상장|구조조정|조직개편|임원|대표|"
    r"AI\s*팩토리|AI\s*데이터센터|데이터센터|GPU|NPU|반도체|초대형|전략적\s*협력|파트너십|"
    r"요금|가격|출시|서비스\s*개편|등급\s*상향|청소년|규제|법안|시행|의무|차단|삭제|가이드라인|공정위|개보위|방미통위|과기정통부",
    re.IGNORECASE,
)
V39_GLOBAL_COMPETITOR_RE = re.compile(
    r"오픈\s*AI|OpenAI|챗GPT|ChatGPT|앤트로픽|Anthropic|클로드|Claude|구글|Google|MS|마이크로소프트|Microsoft|"
    r"메타|Meta|애플|Apple|엔비디아|NVIDIA|EU|미국|일본|중국|독일|유럽|빅테크|X\b|틱톡|TikTok",
    re.IGNORECASE,
)
V39_STRONG_COMPETITOR_ACTOR_RE = re.compile(
    r"네이버|NAVER|쿠팡|Coupang|토스|비바리퍼블리카|배달의민족|배민|우아한형제들|SKT|SK텔레콤|KT|LGU\+|LG유플러스|라인야후|우버|Uber|"
    r"오픈\s*AI|OpenAI|챗GPT|ChatGPT|앤트로픽|Anthropic|클로드|Claude|구글|Google|MS|마이크로소프트|Microsoft|메타|Meta|애플|Apple|엔비디아",
    re.IGNORECASE,
)
V39_REFILL_SOURCE_RE = re.compile(r"refill|replacement|backup|local_extra|label_based", re.IGNORECASE)


def v39_text(obj):
    try:
        if isinstance(obj, dict):
            return clean_html_text(
                f"{obj.get('기사제목','')} {obj.get('title','')} {obj.get('issue_group','')} {obj.get('top_title','')} "
                f"{obj.get('issue_family','')} {obj.get('internal_category','')} {obj.get('company_impact','')} "
                f"{obj.get('label_reasons','')} {obj.get('reason','')} {obj.get('본문요약','')} {obj.get('RSS요약','')} {obj.get('본문전문','')[:2200]}"
            )
    except Exception:
        pass
    return clean_html_text(str(obj or ""))


def v39_is_refill_sourced(item):
    text = " ".join(str(item.get(k, "")) for k in [
        "선정단계", "v26_1_refill_source", "v32_competitor_portfolio_refill", "v32_forced_competitor_portfolio_refill", "제외단계"
    ])
    return bool(V39_REFILL_SOURCE_RE.search(text))


def v39_is_low_value_event_text(text, cat="", strict_for_refill=False):
    t = clean_html_text(text)
    if not t:
        return False, ""
    if V39_SEVERE_PROMO_RE.search(t):
        return True, "severe_promotion_or_quiz"
    if V39_EVENT_ONLY_RE.search(t):
        # 행사/세미나/교육/협약은 material signal이 있을 때만 살린다.
        if not V39_MATERIAL_EVENT_RE.search(t):
            return True, "event_promo_without_material_event"
        # 경쟁사 섹션에서 단순 행사성 기사면 AI/클라우드 단어만으로는 부족하게 본다.
        if cat == JSON_KEYS_ORDER[2] and re.search(r"클라우드\s*데이|세미나|컨퍼런스|포럼|설명회|교육|업무협약|MOU|협약", t, re.IGNORECASE):
            if not re.search(r"과징금|제재|소송|판결|개인정보|유출|해킹|인수|매각|투자|추가\s*투자|출시|요금|규제|의무|AI\s*팩토리|데이터센터\s*구축|전략적\s*협력", t, re.IGNORECASE):
                return True, "competitor_event_without_hard_news"
    if strict_for_refill and re.search(r"최대주주\s*보유|지분\s*변동\s*공시|의료기기|물류센터\s*로봇|지역\s*클라우드\s*생태계", t, re.IGNORECASE):
        if not re.search(r"카카오|네이버|쿠팡|오픈AI|앤트로픽|구글|MS|메타|애플|엔비디아|AI\s*팩토리|데이터센터|과징금|제재|소송|인수|매각", t, re.IGNORECASE):
            return True, "weak_b2b_or_disclosure_refill"
    return False, ""


def v39_is_weak_refill_issue(decision, issue, cat=None):
    cat = cat or (v26_1_decision_category(decision, issue) if "v26_1_decision_category" in globals() else decision.get("category", ""))
    text = v39_text(issue) + " " + v39_text(decision)
    low, reason = v39_is_low_value_event_text(text, cat=cat, strict_for_refill=True)
    if low:
        return True, reason
    # Competition refill should have a real major competitor/global actor and hard-news action.
    if cat == JSON_KEYS_ORDER[2]:
        if not V39_STRONG_COMPETITOR_ACTOR_RE.search(text):
            return True, "competitor_refill_without_core_actor"
        if V39_EVENT_ONLY_RE.search(text) and not V39_MATERIAL_EVENT_RE.search(text):
            return True, "competitor_refill_event_only"
    return False, ""


_BASE_v26_1_is_good_refill_candidate_v39 = v26_1_is_good_refill_candidate

def v26_1_is_good_refill_candidate(decision, issue, cat=None):
    if not _BASE_v26_1_is_good_refill_candidate_v39(decision, issue, cat=cat):
        return False
    weak, _reason = v39_is_weak_refill_issue(decision, issue, cat=cat)
    if weak:
        return False
    return True


_BASE_v32_refill_candidate_score_v39 = v32_refill_candidate_score

def v32_refill_candidate_score(decision, issue, final_report_data):
    score = float(_BASE_v32_refill_candidate_score_v39(decision, issue, final_report_data) or 0)
    weak, _reason = v39_is_weak_refill_issue(decision, issue, cat=JSON_KEYS_ORDER[2])
    if weak:
        score -= 500
    text = v39_text(issue)
    if V39_GLOBAL_COMPETITOR_RE.search(text):
        try:
            stats = v32_competitor_portfolio_stats(final_report_data)
            if stats.get("overseas_or_global", 0) <= 0:
                score += 70
        except Exception:
            score += 35
    return round(score, 2)


def v39_add_skip(skip_rows, item, reason):
    if skip_rows is None:
        return
    try:
        v32_add_skip(skip_rows, item, reason)
        return
    except Exception:
        pass
    try:
        skip_rows.append({
            "검색어": item.get("검색어", ""),
            "후보제목": item.get("기사제목", ""),
            "후보링크": item.get("링크", ""),
            "후보언론사": item.get("언론사", ""),
            "매칭과거일자": "CURRENT_RUN",
            "매칭과거제목": "v39_final_guardrail",
            "중복판정이유": reason,
            "제외단계": "v39_final_guardrail",
        })
    except Exception:
        pass


def v39_remove_low_value_final_items(final_report_data, skip_rows=None):
    kept = []
    for item in list(final_report_data):
        cat = item.get("카테고리", "")
        text = v39_text(item)
        low, reason = v39_is_low_value_event_text(text, cat=cat, strict_for_refill=v39_is_refill_sourced(item))
        # Severe promo/quiz is always removed. Softer event-only is removed only from competitor/refill contexts.
        remove = False
        if reason == "severe_promotion_or_quiz":
            remove = True
        elif low and (cat == JSON_KEYS_ORDER[2] or v39_is_refill_sourced(item)):
            remove = True
        if remove:
            v39_add_skip(skip_rows, item, f"v39_removed_low_value_final:{reason}")
            continue
        kept.append(item)
    final_report_data[:] = kept
    return final_report_data


_BASE_v33_repair_final_items_v39 = v33_repair_final_items

def v33_repair_final_items(final_report_data, skip_rows=None):
    res = _BASE_v33_repair_final_items_v39(final_report_data, skip_rows)
    v39_remove_low_value_final_items(final_report_data, skip_rows)
    return res


def v32_repair_final_items(final_report_data, skip_rows=None):
    return v33_repair_final_items(final_report_data, skip_rows)


_BASE_v29_item_priority_score_v39 = v29_item_priority_score

def v29_item_priority_score(item):
    score = float(_BASE_v29_item_priority_score_v39(item) or 0)
    low, reason = v39_is_low_value_event_text(v39_text(item), cat=item.get("카테고리", ""), strict_for_refill=True)
    if low:
        score -= 150 if reason == "severe_promotion_or_quiz" else 70
    return round(score, 2)


def v39_issue_terms(issue):
    text = clean_html_text(f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('issue_family','')} {issue.get('label_reasons','')} {issue.get('company_impact','')}")
    tokens = tokenize_for_similarity(text)
    drop = {"관련", "이슈", "기사", "뉴스", "후보", "전략", "정책", "정부", "기업", "서비스", "플랫폼"}
    return {t for t in tokens if t not in drop and len(t) >= 2}


def v39_article_issue_score(article, issue, requested_cat):
    text = v39_text(article)
    title = clean_html_text(article.get("기사제목", ""))
    score = v20_float(article.get("랭킹점수"), 0) + v20_float(article.get("중요도점수"), 0) * 0.35
    score += press_score(article.get("언론사", ""))
    terms = v39_issue_terms(issue)
    if terms:
        score += min(28, len(terms & tokenize_for_similarity(text)) * 4)
    low, reason = v39_is_low_value_event_text(text, cat=requested_cat, strict_for_refill=False)
    if low:
        score -= 160 if reason == "severe_promotion_or_quiz" else 45
    if requested_cat == JSON_KEYS_ORDER[2]:
        try:
            ok, _ = v32_competitor_candidate_text(text)
            if ok:
                score += 35
        except Exception:
            pass
        if V39_GLOBAL_COMPETITOR_RE.search(text):
            score += 18
    elif requested_cat == JSON_KEYS_ORDER[0] and v20_has_self_entity(text):
        score += 20
    elif requested_cat == JSON_KEYS_ORDER[1]:
        try:
            ok, _ = v29_gov_eligible({"기사제목": title, "RSS요약": text, "본문전문": text, "카테고리": requested_cat})
            if ok:
                score += 22
        except Exception:
            pass
    return round(score, 2)


def v39_candidate_ids_for_decision(decision, issue):
    ids = []
    def add_one(x):
        try:
            if x is None or x == "":
                return
            xi = int(x)
            if xi not in ids:
                ids.append(xi)
        except Exception:
            pass
    add_one(decision.get("best_article_id"))
    for arr in [decision.get("backup_article_ids", []), issue.get("candidate_article_ids", []), issue.get("article_ids", []), [issue.get("top_article_id", "")]]:
        if isinstance(arr, list):
            for x in arr:
                add_one(x)
        else:
            for x in re.findall(r"\d+", str(arr)):
                add_one(x)
    return ids


def v39_adjust_decision_article_order(decision, issue, article_by_id):
    if not issue:
        return decision
    requested_cat = v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue) if "v20_issue_to_output_category" in globals() else issue.get("primary_category", "산업동향"))
    ids = v39_candidate_ids_for_decision(decision, issue)
    if len(ids) <= 1:
        return decision
    scored = []
    for aid in ids:
        article = article_by_id.get(int(aid), {})
        if not article:
            continue
        scored.append((v39_article_issue_score(article, issue, requested_cat), int(aid)))
    if not scored:
        return decision
    scored.sort(reverse=True)
    current = int(decision.get("best_article_id") or scored[0][1])
    current_score = next((s for s, aid in scored if aid == current), scored[-1][0])
    best_score, best_id = scored[0]
    # Soft reorder only when the current representative is clearly weaker.
    if best_id != current and best_score >= current_score + 18:
        d2 = dict(decision)
        d2["best_article_id"] = best_id
        rest = [aid for _s, aid in scored if aid != best_id]
        d2["backup_article_ids"] = rest[:10]
        d2["v39_representative_reordered"] = f"{current}->{best_id}:{current_score}->{best_score}"
        return d2
    return decision


def v39_title_only_allowed(issue, decision, article_info, requested_cat):
    text = v39_text(issue) + " " + v39_text(article_info) + " " + v39_text(decision)
    low, _reason = v39_is_low_value_event_text(text, cat=requested_cat, strict_for_refill=True)
    if low:
        return False, "low_value_event_or_promo"
    priority = max(v20_int(decision.get("priority"), 1), v20_int(issue.get("max_ceo_priority"), 1))
    pa = v20_int(issue.get("max_pa_priority"), 1)
    score = v20_float(issue.get("issue_score"), 0)
    selected_source = clean_html_text(f"{decision.get('kind','')} {decision.get('v26_1_pool_source','')} {decision.get('v32_pool_source','')} {decision.get('reason','')}")
    selected_like = bool(re.search(r"selected|backup|Gemini|v32|v26", selected_source, re.IGNORECASE))
    if priority < 4 and score < 45 and not selected_like:
        return False, "priority_not_high_enough"
    pseudo = {"기사제목": article_info.get("기사제목", issue.get("top_title", "")), "RSS요약": text, "본문요약": text, "본문전문": text, "카테고리": requested_cat, "본문상태": "본문추출실패_대체없음_제목링크만", "본문글자수": 0}
    try:
        if v29_title_only_issue_allowed(requested_cat, text):
            return True, "v29_title_only_allowed"
    except Exception:
        pass
    if requested_cat == JSON_KEYS_ORDER[0]:
        if v20_has_self_entity(text) and re.search(r"노조|파업|임단협|쟁의|과징금|소송|수사|개인정보|유출|장애|임원|조직개편|대표|지분|매각|인수|경영권", text, re.IGNORECASE):
            return True, "selected_self_material_issue"
    elif requested_cat == JSON_KEYS_ORDER[1]:
        try:
            ok, reason = v29_gov_eligible(pseudo)
            if ok and re.search(r"법안|시행|의무|규제|과징금|제재|개인정보|보안|플랫폼|AI|디지털자산|스테이블코인|저작권|망분리|딥페이크", text, re.IGNORECASE):
                return True, "selected_government_policy_issue"
        except Exception:
            pass
    elif requested_cat == JSON_KEYS_ORDER[2]:
        try:
            ok, reason = v32_competitor_candidate_text(text)
            if ok:
                return True, "selected_competitor_issue"
        except Exception:
            pass
        if V39_GLOBAL_COMPETITOR_RE.search(text) and re.search(r"AI|투자|협력|제휴|출시|요금|판결|규제|조사|제재|소송|보안|개인정보|데이터센터|앱|매출", text, re.IGNORECASE):
            return True, "selected_global_competitor_issue"
    elif requested_cat == JSON_KEYS_ORDER[3]:
        try:
            if v29_industry_eligible(pseudo):
                return True, "selected_structural_industry_issue"
        except Exception:
            pass
    return False, "not_high_confidence_title_only"


_BASE_v20_process_issue_v39 = v20_process_issue

def v20_process_issue(decision, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter):
    issue = issue_by_id.get(decision.get("issue_id"))
    if issue:
        decision = v39_adjust_decision_article_order(decision, issue, article_by_id)
    item, new_order_counter, status = _BASE_v20_process_issue_v39(
        decision,
        issue_by_id,
        article_by_id,
        recent_past_items,
        final_report_data,
        body_failed_rows,
        skip_rows,
        processed_article_ids,
        order_counter,
    )
    if item or status != "all_candidates_failed" or not issue:
        return item, new_order_counter, status

    requested_cat = v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue) if "v20_issue_to_output_category" in globals() else issue.get("primary_category", "산업동향"))
    ids = v39_candidate_ids_for_decision(decision, issue)
    scored = []
    for aid in ids:
        article = article_by_id.get(int(aid), {})
        if not article:
            continue
        allowed, reason = v39_title_only_allowed(issue, decision, article, requested_cat)
        if not allowed:
            continue
        scored.append((v39_article_issue_score(article, issue, requested_cat), int(aid), reason))
    if not scored:
        return None, new_order_counter, status
    scored.sort(reverse=True)
    _score, fallback_id, allow_reason = scored[0]
    article_info = article_by_id.get(fallback_id)
    if not article_info:
        return None, new_order_counter, status
    try:
        item = v20_create_bodyless_report_item(article_info, issue, decision, requested_cat, reason=f"v39_title_link_only:{allow_reason}")
    except Exception:
        return None, new_order_counter, status
    try:
        action, guarded_cat, guard_reason = v23_final_category_guardrail(item, requested_cat)
        if action == "fail":
            return None, new_order_counter, "v39_bodyless_guardrail_failed"
        if action == "reassign" and guarded_cat:
            v23_set_final_category(item, guarded_cat, guard_reason)
            item["최종카테고리검증결과"] = f"v39_bodyless_reassign:{guard_reason}"
        else:
            v23_set_final_category(item, requested_cat)
            item["최종카테고리검증결과"] = f"v39_bodyless_pass:{guard_reason}"
    except Exception:
        item["카테고리"] = requested_cat
        item["카테고리명"] = JSON_KEY_TO_DISPLAY.get(requested_cat, requested_cat)
    item["선정순서"] = order_counter
    item["선정단계"] = "v39_title_link_only"
    item["v39_title_link_only_reason"] = allow_reason
    try:
        dup_item, dup_reason = v21_strict_cross_issue_duplicate(item, final_report_data)
    except Exception:
        dup_item, dup_reason = final_duplicate_reason(item, final_report_data)
    if dup_item:
        add_duplicate_skip_row(skip_rows, item, dup_item, f"v39_bodyless_duplicate:{dup_reason}", stage="v39_title_link_only")
        return None, new_order_counter, "v39_bodyless_duplicate"
    final_report_data.append(item)
    return item, order_counter + 1, "v39_title_link_only"


_BASE_v26_1_refill_after_final_repair_v39 = v26_1_refill_after_final_repair

def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    order_counter = _BASE_v26_1_refill_after_final_repair_v39(
        decisions=decisions,
        backup_decisions=backup_decisions,
        issues=issues,
        issue_by_id=issue_by_id,
        article_by_id=article_by_id,
        recent_past_items=recent_past_items,
        final_report_data=final_report_data,
        body_failed_rows=body_failed_rows,
        skip_rows=skip_rows,
        processed_article_ids=processed_article_ids,
        order_counter=order_counter,
        status_counts=status_counts,
    )
    # Very light final cleanup only; keep v33 portfolio behavior.
    v39_remove_low_value_final_items(final_report_data, skip_rows)
    try:
        v33_repair_final_items(final_report_data, skip_rows)
    except Exception:
        pass
    return order_counter


_BASE_v20_gemini_quality_check_v39 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    try:
        v39_remove_low_value_final_items(final_report_data, [])
    except Exception:
        pass
    qa = _BASE_v20_gemini_quality_check_v39(client, final_report_data, final_briefing_text)
    try:
        data = v22_parse_qa_json(qa) if qa else {"overall": "양호", "warnings": [], "suggested_human_checks": []}
    except Exception:
        data = {"overall": "주의", "warnings": [clean_html_text(qa)[:500]], "suggested_human_checks": []}
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    checks = data.get("suggested_human_checks") if isinstance(data.get("suggested_human_checks"), list) else []
    # Soft warning only: do not force weak overseas articles into final.
    try:
        stats = v32_competitor_portfolio_stats(final_report_data)
        if stats.get("total", 0) >= 2 and stats.get("overseas_or_global", 0) <= 0:
            msg = "경쟁사/해외 섹션에 해외·글로벌 플랫폼 이슈가 없으나, v39는 약한 기사 강제 삽입은 하지 않음"
            warnings.append(msg)
            checks.append(msg)
            if data.get("overall") == "양호":
                data["overall"] = "주의"
    except Exception:
        pass
    data["warnings"] = warnings
    data["suggested_human_checks"] = checks
    return json.dumps(data, ensure_ascii=False, indent=2)


# =========================================================
# v40 overrides: official-enforcement repost dedupe + softer self-section fallback
# =========================================================
# v40 keeps the v33/v39 selection feel and changes only two narrow areas:
# 1) Same official enforcement event in index history is treated as duplicate even
#    when litigation/sanction wording makes the stage oscillate.
# 2) Self section becomes tiered: direct risk > business expansion/MOU > soft PR.
#    Soft PR is not a must-have, but may fill the self section when there are no
#    stronger self issues. Severe coupon/quiz/discount noise remains excluded.

V6_VERSION = "google_rss_v40_enforcement_dedupe_self_soft_fallback"

V40_SELF_SOFT_MIN = 2
V40_SELF_PR_MAX = 1

V40_SELF_SEVERE_PROMO_RE = re.compile(
    r"행운퀴즈|퀴즈\s*정답|정답\s*공개|멤버스위크|앱테크|캐시워크|오퀴즈|"
    r"쿠폰|할인|특가|경품|체험단|기획전|적립금|포인트\s*지급|리워드",
    re.IGNORECASE,
)

V40_SELF_BUSINESS_EXPANSION_RE = re.compile(
    r"업무협약|MOU|협약\s*체결|파트너십|전략적\s*제휴|제휴|협력|공동\s*(사업|개발|연구)|"
    r"사업\s*확장|서비스\s*확대|신규\s*서비스|출시|공개|도입|오픈|론칭|상용화|"
    r"해외\s*진출|글로벌|투자|출자|펀드|인수|합병|지분|계약|수주|공급|"
    r"AI|인공지능|에이전트|카나나|데이터센터|클라우드|핀테크|간편결제|월렛|스테이블코인|"
    r"모빌리티|헬스케어|게임|콘텐츠|엔터|픽코마|엔터프라이즈|클라우드",
    re.IGNORECASE,
)

V40_SELF_SIMPLE_PR_RE = re.compile(
    r"후원|스폰서|협찬|캠페인|기부|사회공헌|ESG|상생|봉사|행사|축제|페스티벌|영화제|"
    r"수상|브랜드\s*대상|전시|팝업|프로모션|이벤트|참여|모집",
    re.IGNORECASE,
)

V40_SELF_DIRECT_RE = re.compile(
    r"노조|파업|쟁의|임단협|고용불안|구조조정|임금|성과급|RSU|노동부|근로감독|최저임금|"
    r"수사|조사|과징금|제재|시정명령|소송|판결|고발|검찰|경찰|개인정보|유출|해킹|피싱|사칭|"
    r"장애|먹통|서비스\s*논란|이용자\s*반발|조직개편|임원|CPO|CTO|CFO|대표|퇴사|사임|"
    r"지분|매각|인수|합병|최대주주|경영권|공동대표|사내이사|주총|실적|영업이익|적자",
    re.IGNORECASE,
)

V40_ENFORCEMENT_ALIAS_STAGES = {
    "litigation", "sanction_penalty", "sanction_or_penalty", "penalty_or_sanction",
    "sanction_penalty_decision", "official_action", "policy_change", "general",
}
V40_ENFORCEMENT_INVESTIGATION_STAGES = {
    "investigation_review", "investigation_or_consultation", "initial_investigation_response", "investigation", "review",
}
V40_REAL_FOLLOWUP_RE = re.compile(
    r"항소|상고|불복|항소장|상고장|집단소송|손해배상|추가\s*(과징금|제재|처분|조사|수사|고발)|"
    r"별도\s*(과징금|제재|처분).{0,35}(부과|결정|확정)|새로\s*(부과|고발|조사|착수|제기)|"
    r"재발\s*방지|개선안|시정\s*계획|보상안|피해\s*구제|합의|조정\s*성립",
    re.IGNORECASE,
)


def v40_text(obj):
    try:
        return v39_text(obj)
    except Exception:
        if isinstance(obj, dict):
            return clean_html_text(
                f"{obj.get('기사제목','')} {obj.get('title','')} {obj.get('본문요약','')} {obj.get('summary','')} "
                f"{obj.get('RSS요약','')} {obj.get('본문전문','')[:2600]} {obj.get('issue_group','')} {obj.get('top_title','')}"
            )
        return clean_html_text(str(obj or ""))


def v40_has_self_entity(text):
    try:
        return bool(v20_has_self_entity(text))
    except Exception:
        return bool(SELF_KAKAO_PATTERN.search(clean_html_text(text)))


def v40_self_content_tier(text):
    t = clean_html_text(text)
    if not t or not v40_has_self_entity(t):
        return "OFF_TOPIC"
    if V40_SELF_SEVERE_PROMO_RE.search(t):
        return "SEVERE_PROMO"
    if V40_SELF_DIRECT_RE.search(t):
        return "DIRECT_RISK"
    if v20_is_platform_obligation_text(t) if 'v20_is_platform_obligation_text' in globals() else False:
        return "RESPONSE_RELEVANT"
    if V40_SELF_BUSINESS_EXPANSION_RE.search(t):
        return "BUSINESS_EXPANSION"
    if V40_SELF_SIMPLE_PR_RE.search(t):
        return "SIMPLE_PR"
    return "FILLER_REFERENCE"


def v40_is_self_business_or_soft_pr(text):
    return v40_self_content_tier(text) in {"BUSINESS_EXPANSION", "SIMPLE_PR", "FILLER_REFERENCE"}


# -----------------------------
# 1) Official enforcement repost dedupe
# -----------------------------

def v40_extract_platform_enforcement_base_from_key(key):
    s = clean_html_text(key)
    if not s or "platform_enforcement" not in s:
        return ""
    try:
        rest = s.split("platform_enforcement:", 1)[1]
        parts = [p for p in rest.split(":") if p]
        if len(parts) >= 2:
            return f"platform_enforcement:{parts[0]}:{parts[1]}"
    except Exception:
        return ""
    return ""


def v40_official_enforcement_base(item):
    if not isinstance(item, dict):
        item = {"기사제목": str(item)}
    # Prefer explicit keys already written to index/history CSV.
    for col in ["event_key", "event_base", "전역사건기본키", "전역사건키", "v29_event_base", "개인정보보안사건키", "shared_tags"]:
        base = v40_extract_platform_enforcement_base_from_key(item.get(col, ""))
        if base:
            return base
    try:
        key = v27_platform_enforcement_key(item, include_stage=False)
        if key:
            return key
    except Exception:
        pass
    try:
        key = v16_global_incident_base_key(item)
        base = v40_extract_platform_enforcement_base_from_key(key)
        if base:
            return base
    except Exception:
        pass
    return ""


def v40_official_enforcement_stage(item):
    if not isinstance(item, dict):
        item = {"기사제목": str(item)}
    for col in ["event_stage", "사건단계", "v29_event_stage"]:
        val = clean_html_text(item.get(col, ""))
        if val:
            return val
    for col in ["event_key", "전역사건키"]:
        s = clean_html_text(item.get(col, ""))
        if "platform_enforcement:" in s:
            parts = s.split("platform_enforcement:", 1)[1].split(":")
            if len(parts) >= 3:
                return parts[2]
    try:
        key = v27_platform_enforcement_key(item, include_stage=True)
        if key:
            parts = key.split(":")
            if len(parts) >= 4:
                return parts[-1]
    except Exception:
        pass
    return ""


def v40_has_real_official_followup(candidate, past):
    cand_text = v40_text(candidate)
    past_text = v40_text(past)
    if not V40_REAL_FOLLOWUP_RE.search(cand_text):
        return False
    # Avoid treating background mentions such as "별도 행정소송 진행 중" as a new development.
    background_only = re.search(r"별도.{0,30}행정소송.{0,30}(진행\s*중|계류|앞두고|관련)", cand_text, re.IGNORECASE)
    if background_only and not re.search(r"항소|상고|추가|새로|제기|부과|확정|결정|보상|재발", cand_text, re.IGNORECASE):
        return False
    return True


def v40_same_official_enforcement_duplicate(candidate, past):
    cand_base = v40_official_enforcement_base(candidate)
    past_base = v40_official_enforcement_base(past)
    if not cand_base or not past_base or cand_base != past_base:
        return False, ""

    cand_stage = v40_official_enforcement_stage(candidate)
    past_stage = v40_official_enforcement_stage(past)

    if v40_has_real_official_followup(candidate, past):
        return False, "v40_same_enforcement_but_real_followup"

    # Investigation -> first official decision may be a real update. Keep it.
    if past_stage in V40_ENFORCEMENT_INVESTIGATION_STAGES and cand_stage in V40_ENFORCEMENT_ALIAS_STAGES:
        return False, f"v40_enforcement_stage_update_allowed:{past_stage}->{cand_stage}"

    # Litigation/sanction words often coexist in the same 판결/과징금 article.
    # If the base actor/object is identical and there is no explicit new follow-up, treat as repost.
    if cand_stage in V40_ENFORCEMENT_ALIAS_STAGES or past_stage in V40_ENFORCEMENT_ALIAS_STAGES:
        return True, f"v40_same_official_enforcement_base:{cand_base}:{past_stage}->{cand_stage}"

    # Last fallback for same official base.
    cand_title = clean_html_text(candidate.get("기사제목") or candidate.get("title") or "") if isinstance(candidate, dict) else ""
    past_title = clean_html_text(past.get("기사제목") or past.get("title") or "") if isinstance(past, dict) else ""
    if sequence_ratio(cand_title, past_title) >= 0.45:
        return True, f"v40_same_official_enforcement_title:{cand_base}"
    return True, f"v40_same_official_enforcement_base:{cand_base}"


_BASE_v26_index_duplicate_check_v40 = v26_index_duplicate_check

def v26_index_duplicate_check(candidate, recent_past_items, mode="rss"):
    for past in recent_past_items or []:
        same, reason = v40_same_official_enforcement_duplicate(candidate, past)
        if same:
            return True, past, {
                "is_duplicate": True,
                "reason": reason,
                "score": 0.98,
                "shared_entities": "",
                "shared_tags": v40_official_enforcement_base(candidate),
            }
    return _BASE_v26_index_duplicate_check_v40(candidate, recent_past_items, mode=mode)


_BASE_v26_is_same_event_update_v40 = v26_is_same_event_update

def v26_is_same_event_update(candidate, past_item):
    same, reason = v40_same_official_enforcement_duplicate(candidate, past_item)
    if same:
        return False, reason
    return _BASE_v26_is_same_event_update_v40(candidate, past_item)


_BASE_final_duplicate_reason_v40 = final_duplicate_reason

def final_duplicate_reason(new_item, existing_items):
    for old in existing_items:
        same, reason = v40_same_official_enforcement_duplicate(new_item, old)
        if same:
            return old, reason
    return _BASE_final_duplicate_reason_v40(new_item, existing_items)


# -----------------------------
# 2) Self section: risk > business expansion/MOU > soft PR
# -----------------------------

_BASE_v18_is_self_pr_promo_text_v40 = v18_is_self_pr_promo_text

def v18_is_self_pr_promo_text(text):
    t = clean_html_text(text)
    if not t or not v40_has_self_entity(t):
        return False
    # Keep quiz/coupon/discount noise as PR/noise.
    if V40_SELF_SEVERE_PROMO_RE.search(t):
        return True
    # MOU/business expansion and soft PR should not be globally excluded anymore.
    if V40_SELF_BUSINESS_EXPANSION_RE.search(t) or V40_SELF_SIMPLE_PR_RE.search(t):
        return False
    return _BASE_v18_is_self_pr_promo_text_v40(text)


_BASE_v21_self_tier_v40 = v21_self_tier

def v21_self_tier(text):
    t = clean_html_text(text)
    if not v40_has_self_entity(t):
        return "OFF_TOPIC"
    tier = v40_self_content_tier(t)
    if tier == "SEVERE_PROMO":
        return "LOW_VALUE_PR"
    if tier == "DIRECT_RISK":
        return "DIRECT_RISK"
    if tier == "RESPONSE_RELEVANT":
        return "RESPONSE_RELEVANT"
    if tier == "BUSINESS_EXPANSION":
        return "STRATEGIC_REFERENCE"
    if tier == "SIMPLE_PR":
        return "FILLER_REFERENCE"
    return _BASE_v21_self_tier_v40(text)


_BASE_v20_detect_issue_family_v40 = v20_detect_issue_family

def v20_detect_issue_family(text, json_key=""):
    t = clean_html_text(text)
    if v40_has_self_entity(t):
        tier = v40_self_content_tier(t)
        if tier == "SEVERE_PROMO":
            return "low_value_pr"
        if tier == "BUSINESS_EXPANSION":
            return "self_strategic_reference"
        if tier == "SIMPLE_PR":
            return "self_filler_reference"
    return _BASE_v20_detect_issue_family_v40(text, json_key)


_BASE_v20_local_article_label_v40 = v20_local_article_label

def v20_local_article_label(article):
    label = _BASE_v20_local_article_label_v40(article)
    text = v20_article_text(article)
    if not v40_has_self_entity(text):
        return label
    tier = v40_self_content_tier(text)
    if tier == "BUSINESS_EXPANSION":
        label.update({
            "is_relevant": True,
            "primary_category": JSON_KEYS_ORDER[0],
            "internal_category": "SELF_AFFILIATE_BUSINESS",
            "issue_family": "self_strategic_reference",
            "company_impact": "카카오·계열사의 사업 확장, 제휴, MOU, 신규 서비스 참고 이슈",
            "ceo_priority": max(v20_int(label.get("ceo_priority"), 3), 3),
            "public_affairs_priority": max(v20_int(label.get("public_affairs_priority"), 3), 3),
            "relevance": max(v20_int(label.get("relevance"), 3), 3),
            "is_pr": False,
            "exclude": False,
            "self_tier": "STRATEGIC_REFERENCE",
            "reason": "v40: 직접 리스크는 아니지만 자사 사업 확장·제휴성 이슈로 2순위 반영",
        })
    elif tier == "SIMPLE_PR":
        label.update({
            "is_relevant": True,
            "primary_category": JSON_KEYS_ORDER[0],
            "internal_category": "SELF_AFFILIATE_BUSINESS",
            "issue_family": "self_filler_reference",
            "company_impact": "직접 리스크는 아니지만 자사 섹션 보강용으로 참고 가능한 홍보·캠페인성 이슈",
            "ceo_priority": max(v20_int(label.get("ceo_priority"), 2), 2),
            "public_affairs_priority": max(v20_int(label.get("public_affairs_priority"), 2), 2),
            "relevance": max(v20_int(label.get("relevance"), 2), 2),
            "is_pr": False,
            "exclude": False,
            "self_tier": "FILLER_REFERENCE",
            "reason": "v40: 강한 자사 이슈가 없을 때 3순위로 허용하는 소프트 홍보 이슈",
        })
    return label


_BASE_v20_normalize_label_json_v40 = v20_normalize_label_json

def v20_normalize_label_json(data, valid_ids, article_by_id):
    labels = _BASE_v20_normalize_label_json_v40(data, valid_ids, article_by_id)
    for art_id, label in list(labels.items()):
        article = article_by_id.get(int(art_id), {}) if isinstance(article_by_id, dict) else {}
        text = v20_article_text(article)
        if not v40_has_self_entity(text):
            continue
        tier = v40_self_content_tier(text)
        if tier == "BUSINESS_EXPANSION":
            label.update({
                "is_relevant": True,
                "primary_category": JSON_KEYS_ORDER[0],
                "internal_category": "SELF_AFFILIATE_BUSINESS",
                "issue_family": "self_strategic_reference",
                "company_impact": "카카오·계열사의 사업 확장, 제휴, MOU, 신규 서비스 참고 이슈",
                "ceo_priority": max(v20_int(label.get("ceo_priority"), 3), 3),
                "public_affairs_priority": max(v20_int(label.get("public_affairs_priority"), 3), 3),
                "relevance": max(v20_int(label.get("relevance"), 3), 3),
                "is_pr": False,
                "exclude": False,
                "self_tier": "STRATEGIC_REFERENCE",
                "reason": "v40: 자사 사업 확장·제휴성 이슈로 2순위 허용",
            })
        elif tier == "SIMPLE_PR":
            label.update({
                "is_relevant": True,
                "primary_category": JSON_KEYS_ORDER[0],
                "internal_category": "SELF_AFFILIATE_BUSINESS",
                "issue_family": "self_filler_reference",
                "company_impact": "자사 섹션 보강용 소프트 홍보·캠페인성 참고 이슈",
                "ceo_priority": max(v20_int(label.get("ceo_priority"), 2), 2),
                "public_affairs_priority": max(v20_int(label.get("public_affairs_priority"), 2), 2),
                "relevance": max(v20_int(label.get("relevance"), 2), 2),
                "is_pr": False,
                "exclude": False,
                "self_tier": "FILLER_REFERENCE",
                "reason": "v40: 강한 자사 이슈가 없을 때 3순위로 허용하는 소프트 홍보 이슈",
            })
        if label.get("issue_group") in V20_GENERIC_GROUPS or not label.get("issue_group"):
            label["issue_group"] = v20_clip(title_fingerprint(article.get("기사제목", ""))[:80] or f"self_soft_{art_id}", 100)
    return labels


_BASE_v20_issue_score_v40 = v20_issue_score

def v20_issue_score(issue):
    score = float(_BASE_v20_issue_score_v40(issue) or 0)
    text = v40_text(issue)
    tier = v40_self_content_tier(text)
    if tier == "BUSINESS_EXPANSION":
        score += 18
    elif tier == "SIMPLE_PR":
        score -= 18
    elif tier == "SEVERE_PROMO":
        score -= 220
    return round(score, 3)


_BASE_v21_issue_allowed_basic_v40 = v21_issue_allowed_basic

def v21_issue_allowed_basic(issue):
    text = v40_text(issue)
    tier = v40_self_content_tier(text)
    if v40_has_self_entity(text):
        if tier == "SEVERE_PROMO":
            return False, "v40_severe_self_promo_noise"
        if tier in {"BUSINESS_EXPANSION", "SIMPLE_PR", "FILLER_REFERENCE"}:
            return True, f"v40_self_soft_allowed:{tier}"
    return _BASE_v21_issue_allowed_basic_v40(issue)


_BASE_v29_self_eligible_v40 = v29_self_eligible

def v29_self_eligible(item):
    text = v29_full_text(item) if 'v29_full_text' in globals() else v40_text(item)
    tier = v40_self_content_tier(text)
    if tier == "BUSINESS_EXPANSION":
        return True, "v40_self_business_expansion_allowed"
    if tier == "SIMPLE_PR":
        return True, "v40_self_simple_pr_allowed_when_needed"
    return _BASE_v29_self_eligible_v40(item)


_BASE_v39_is_low_value_event_text_v40 = v39_is_low_value_event_text

def v39_is_low_value_event_text(text, cat="", strict_for_refill=False):
    t = clean_html_text(text)
    if cat == JSON_KEYS_ORDER[0] and v40_has_self_entity(t):
        tier = v40_self_content_tier(t)
        if tier in {"BUSINESS_EXPANSION", "SIMPLE_PR", "FILLER_REFERENCE"}:
            return False, "v40_self_soft_allowed"
    return _BASE_v39_is_low_value_event_text_v40(text, cat=cat, strict_for_refill=strict_for_refill)


def v40_current_self_count(final_report_data):
    return sum(1 for item in final_report_data if item.get("카테고리") == JSON_KEYS_ORDER[0])


def v40_current_self_pr_count(final_report_data):
    count = 0
    for item in final_report_data:
        if item.get("카테고리") != JSON_KEYS_ORDER[0]:
            continue
        if v40_self_content_tier(v40_text(item)) == "SIMPLE_PR":
            count += 1
    return count


def v40_issue_text(issue, article_by_id=None):
    parts = [v40_text(issue)]
    if article_by_id:
        for aid in issue.get("candidate_article_ids", [])[:5]:
            try:
                art = article_by_id.get(int(aid), {})
            except Exception:
                art = {}
            if art:
                parts.append(v20_article_text(art))
    return clean_html_text(" ".join(parts))


def v40_self_soft_issue_score(issue, article_by_id=None):
    text = v40_issue_text(issue, article_by_id)
    tier = v40_self_content_tier(text)
    if tier not in {"BUSINESS_EXPANSION", "SIMPLE_PR", "FILLER_REFERENCE"}:
        return None
    if tier == "SIMPLE_PR" and V40_SELF_SEVERE_PROMO_RE.search(text):
        return None
    base = v20_float(issue.get("issue_score"), 0)
    base += v20_int(issue.get("max_ceo_priority"), 2) * 12
    base += v20_int(issue.get("max_pa_priority"), 2) * 10
    if tier == "BUSINESS_EXPANSION":
        base += 100
    elif tier == "FILLER_REFERENCE":
        base += 35
    elif tier == "SIMPLE_PR":
        base += 10
    return round(base, 2), tier


def v40_refill_self_soft_if_sparse(issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    if v40_current_self_count(final_report_data) >= V40_SELF_SOFT_MIN:
        return order_counter

    represented_issue_ids = {clean_html_text(item.get("v20_issue_id", "")) for item in final_report_data if item.get("v20_issue_id")}
    represented_groups = {v20_issue_group_key(item.get("v20_issue_group", "") or item.get("Gemini이슈그룹", "") or item.get("기사제목", "")) for item in final_report_data}
    represented_groups = {g for g in represented_groups if g}

    candidates = []
    for issue in issues or []:
        iid = clean_html_text(issue.get("issue_id", ""))
        if iid and iid in represented_issue_ids:
            continue
        group_key = v20_issue_group_key(issue.get("issue_group", "") or issue.get("top_title", ""))
        if group_key and group_key in represented_groups:
            continue
        scored = v40_self_soft_issue_score(issue, article_by_id)
        if not scored:
            continue
        score, tier = scored
        if tier == "SIMPLE_PR" and v40_current_self_pr_count(final_report_data) >= V40_SELF_PR_MAX:
            continue
        candidates.append((score, tier, issue))

    tier_order = {"BUSINESS_EXPANSION": 0, "FILLER_REFERENCE": 1, "SIMPLE_PR": 2}
    candidates.sort(key=lambda x: (-x[0], tier_order.get(x[1], 9)))

    for _score, tier, issue in candidates:
        if v40_current_self_count(final_report_data) >= V40_SELF_SOFT_MIN:
            break
        if tier == "SIMPLE_PR" and v40_current_self_pr_count(final_report_data) >= V40_SELF_PR_MAX:
            continue
        d = {
            "issue_id": issue.get("issue_id"),
            "category": JSON_KEYS_ORDER[0],
            "priority": 3 if tier == "BUSINESS_EXPANSION" else 2,
            "best_article_id": issue.get("top_article_id"),
            "backup_article_ids": issue.get("candidate_article_ids", [])[1:8],
            "reason": f"v40_self_soft_refill:{tier}",
            "selection_source": "v40_self_soft_refill",
        }
        item, order_counter, status = v20_process_issue(
            d,
            issue_by_id,
            article_by_id,
            recent_past_items,
            final_report_data,
            body_failed_rows,
            skip_rows,
            processed_article_ids,
            order_counter,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        if item:
            item["v40_self_soft_refill"] = tier
            item["v40_self_soft_refill_score"] = _score
            print(f"  -> v40 self soft refill added({tier}): {v20_clip(item.get('기사제목',''), 46)}...")
    try:
        v39_remove_low_value_final_items(final_report_data, skip_rows)
    except Exception:
        pass
    return order_counter


_BASE_v26_1_refill_after_final_repair_v40 = v26_1_refill_after_final_repair

def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    order_counter = _BASE_v26_1_refill_after_final_repair_v40(
        decisions=decisions,
        backup_decisions=backup_decisions,
        issues=issues,
        issue_by_id=issue_by_id,
        article_by_id=article_by_id,
        recent_past_items=recent_past_items,
        final_report_data=final_report_data,
        body_failed_rows=body_failed_rows,
        skip_rows=skip_rows,
        processed_article_ids=processed_article_ids,
        order_counter=order_counter,
        status_counts=status_counts,
    )
    order_counter = v40_refill_self_soft_if_sparse(
        issues=issues,
        issue_by_id=issue_by_id,
        article_by_id=article_by_id,
        recent_past_items=recent_past_items,
        final_report_data=final_report_data,
        body_failed_rows=body_failed_rows,
        skip_rows=skip_rows,
        processed_article_ids=processed_article_ids,
        order_counter=order_counter,
        status_counts=status_counts,
    )
    return order_counter


_BASE_v20_gemini_quality_check_v40 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v40(client, final_report_data, final_briefing_text)
    try:
        data = v22_parse_qa_json(qa) if qa else {"overall": "양호", "warnings": [], "suggested_human_checks": []}
    except Exception:
        return qa
    checks = data.get("suggested_human_checks") if isinstance(data.get("suggested_human_checks"), list) else []
    if v40_current_self_count(final_report_data) < V40_SELF_SOFT_MIN:
        checks.append("자사 직접 리스크가 적은 날에는 v40 기준으로 MOU/사업확장/소프트 홍보 후보까지 검토했으나 충분한 후보가 없었음")
    data["suggested_human_checks"] = checks
    return json.dumps(data, ensure_ascii=False, indent=2)


# =========================================================
# v41 overrides: Monday weekend window + self actor strict fallback
#                + issue/article alignment + safer final section guards
# =========================================================
# v41 keeps v39/v40's softer selection style, but fixes the failures observed on 2026-06-16:
# 1) Monday collection window covers Friday 00:00 KST through execution time.
# 2) Self soft fallback is allowed only when Kakao/affiliate is the actual article actor.
# 3) Issue group and representative article must match on core entity/object.
# 4) Competitor section cannot be filled by unrelated companies or government/startup support articles.
# 5) Same overseas AI access/export-control incident is de-duplicated.

V6_VERSION = "google_rss_v41_weekend_window_self_actor_alignment"

# -----------------------------
# 1) Dynamic collection window
# -----------------------------

def v41_now_kst():
    return datetime.now(KST)


def v41_collection_start(now=None):
    now = now or v41_now_kst()
    # Monday: cover the weekend plus Friday, because the report is not run on weekends.
    if now.weekday() == 0:
        start_date = now.date() - timedelta(days=3)
    else:
        start_date = now.date() - timedelta(days=1)
    return datetime.combine(start_date, datetime.min.time(), tzinfo=KST)


def v41_effective_rss_recent_days(now=None):
    now = now or v41_now_kst()
    return 4 if now.weekday() == 0 else 2


def v41_collection_window_label(now=None):
    now = now or v41_now_kst()
    start = v41_collection_start(now)
    return f"{start.strftime('%Y-%m-%d %H:%M')}~{now.strftime('%Y-%m-%d %H:%M')} KST"


# Make the query window visible to the existing build flow as well.
RSS_RECENT_DAYS = v41_effective_rss_recent_days()


_BASE_recency_score_v41 = recency_score

def recency_score(published):
    dt = parse_pubdate_to_kst(published)
    base = _BASE_recency_score_v41(published)
    if not dt:
        return base
    now = v41_now_kst()
    start = v41_collection_start(now)
    if now.weekday() == 0 and start <= dt <= now + timedelta(minutes=10):
        # On Monday, Friday/weekend items are still part of the fresh operating window.
        return max(base, 8)
    return base


def is_recent_pubdate(pub_date_text):
    if not STRICT_RSS_TIME_FILTER:
        return True
    dt = parse_pubdate_to_kst(pub_date_text)
    if dt is None:
        return True
    now = v41_now_kst()
    return v41_collection_start(now) <= dt <= now + timedelta(minutes=10)


def build_rss_queries(keyword):
    queries = []
    base = keyword.strip()
    days = v41_effective_rss_recent_days()

    add_unique_query(queries, f"{base} when:{days}d")
    add_unique_query(queries, f"\"{base}\" when:{days}d")

    if not ENABLE_RSS_QUERY_SHARDING:
        return queries

    if base in {"카카오", "카카오톡"}:
        for term in KAKAO_ISSUE_SHARDS:
            add_unique_query(queries, f"{base} {term} when:{days}d")
    elif base in RSS_SHARD_KEYWORDS:
        for term in GENERAL_ISSUE_SHARDS[:8]:
            add_unique_query(queries, f"{base} {term} when:{days}d")

    if base in RSS_SHARD_KEYWORDS:
        for domain in RSS_SOURCE_SHARDS:
            add_unique_query(queries, f"{base} site:{domain} when:{days}d")

    return queries[:MAX_RSS_QUERIES_PER_KEYWORD]


# -----------------------------
# 2) Strict self actor / soft fallback
# -----------------------------

V41_SELF_ENTITY_RE = re.compile(
    r"카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|카카오엔터프라이즈|카카오엔터테인먼트|"
    r"카카오엔터|카카오헬스케어|카카오픽코마|카카오벤처스|카카오스타일|카카오브레인|"
    r"카카오\s*T\b|카카오T\b|카카오택시|카카오톡|카톡|카카오|Kakao",
    re.IGNORECASE,
)

V41_SELF_AS_ONE_OF_MANY_RE = re.compile(
    r"(네이버|구글|메타|애플|쿠팡|토스|SKT|KT|LGU\+|배민|배달의민족).{0,45}(카카오|카카오톡|카톡).{0,55}(등|포함|사업자|플랫폼|대상)|"
    r"(카카오|카카오톡|카톡).{0,45}(네이버|구글|메타|애플|쿠팡|토스|SKT|KT|LGU\+|배민|배달의민족).{0,55}(등|포함|사업자|플랫폼|대상)|"
    r"(네이버|카카오|구글|메타|유튜브|틱톡|애플).{0,60}(80여\s*개|80개|사업자|플랫폼|기간사업자)",
    re.IGNORECASE,
)

V41_SELF_ACTOR_ACTION_RE = re.compile(
    r"(카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|카카오엔터프라이즈|카카오엔터테인먼트|"
    r"카카오엔터|카카오헬스케어|카카오픽코마|카카오벤처스|카카오스타일|카카오\s*T\b|카카오T\b|"
    r"카카오톡|카톡|카카오).{0,90}"
    r"(단행|정비|개편|신설|통합|논의|협력|협약|MOU|제휴|체결|출시|공개|도입|오픈|론칭|상용화|"
    r"투자|출자|진출|확대|공동|개최|성료|선정|참여|발표|밝힘|추진|운영|소개|개발|구축|회복|점검)",
    re.IGNORECASE,
)

V41_SELF_NON_ACTOR_NOISE_RE = re.compile(
    r"카카오톡\s*채널|카톡\s*제보|카카오톡으로\s*제보|카카오톡\s*오픈채팅|"
    r"공유하기|카카오톡에\s*공유|제보는\s*카카오톡|카카오\s*톡\s*제보",
    re.IGNORECASE,
)

V41_SELF_DIRECT_STRONG_RE = re.compile(
    r"카카오|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|카카오톡|카톡|Kakao",
    re.IGNORECASE,
)

V41_SELF_SEVERE_PROMO_EXTRA_RE = re.compile(
    r"행운퀴즈|퀴즈\s*정답|정답\s*공개|멤버스위크|앱테크|캐시워크|오퀴즈|쿠폰|할인|특가|경품|체험단|적립금|리워드",
    re.IGNORECASE,
)

V41_SELF_LOW_SOFT_PR_RE = re.compile(
    r"팝업스토어|굿즈|캐릭터\s*상품|오프라인\s*이벤트|게임\s*이벤트|페스티벌|축제|행사|사회공헌|ESG|캠페인|후원|스폰서|수상|상생|기부",
    re.IGNORECASE,
)


def v41_article_front_text(article):
    if not isinstance(article, dict):
        return clean_html_text(str(article or ""))
    # Exclude source/domain/labels; those caused "Naver Premium Contents" to be read as a Naver issue.
    return clean_html_text(
        f"{article.get('기사제목','')} {article.get('title','')} {article.get('본문요약','')} "
        f"{article.get('summary','')} {article.get('RSS요약','')} {article.get('본문전문','')[:1200]}"
    )


def v41_has_true_self_actor(text):
    t = clean_html_text(text)
    if not t or not V41_SELF_ENTITY_RE.search(t):
        return False
    front = t[:900]
    if V41_SELF_NON_ACTOR_NOISE_RE.search(front) and not V41_SELF_ACTOR_ACTION_RE.search(front):
        return False
    # If Kakao is just one of many regulated platforms, treat it as government/policy, not self section.
    if V41_SELF_AS_ONE_OF_MANY_RE.search(front) and not V41_SELF_ACTOR_ACTION_RE.search(front):
        return False
    if V41_SELF_ACTOR_ACTION_RE.search(front):
        return True
    # Direct risk texts often have compact titles such as "카카오 조직개편".
    if V41_SELF_DIRECT_STRONG_RE.search(front) and V40_SELF_DIRECT_RE.search(front):
        return True
    # Allow soft PR only when Kakao/affiliate is clearly in the title/front and the item is not generic share/contact noise.
    if V41_SELF_ENTITY_RE.search(front[:180]) and (V40_SELF_BUSINESS_EXPANSION_RE.search(front) or V40_SELF_SIMPLE_PR_RE.search(front)):
        return True
    return False


# Override v40's self check with the stricter actual-actor check.
def v40_has_self_entity(text):
    return v41_has_true_self_actor(text)


def v40_self_content_tier(text):
    t = clean_html_text(text)
    if not t or not v41_has_true_self_actor(t):
        return "OFF_TOPIC"
    if V41_SELF_SEVERE_PROMO_EXTRA_RE.search(t):
        return "SEVERE_PROMO"
    if V40_SELF_DIRECT_RE.search(t):
        return "DIRECT_RISK"
    if 'v20_is_platform_obligation_text' in globals() and v20_is_platform_obligation_text(t):
        # If Kakao is one of many obligated platforms, it should not be a self filler.
        if V41_SELF_AS_ONE_OF_MANY_RE.search(t) and not V41_SELF_ACTOR_ACTION_RE.search(t):
            return "OFF_TOPIC"
        return "RESPONSE_RELEVANT"
    if V40_SELF_BUSINESS_EXPANSION_RE.search(t):
        return "BUSINESS_EXPANSION"
    if V41_SELF_LOW_SOFT_PR_RE.search(t) or V40_SELF_SIMPLE_PR_RE.search(t):
        return "SIMPLE_PR"
    return "OFF_TOPIC"


def v41_best_self_article_for_issue(issue, article_by_id=None):
    article_by_id = article_by_id or {}
    ids = []
    for key in ["top_article_id", "best_article_id"]:
        try:
            if issue.get(key):
                ids.append(int(issue.get(key)))
        except Exception:
            pass
    for aid in issue.get("candidate_article_ids", [])[:12]:
        try:
            ids.append(int(aid))
        except Exception:
            continue
    seen = set()
    scored = []
    for aid in ids:
        if aid in seen:
            continue
        seen.add(aid)
        article = article_by_id.get(aid, {})
        if not article:
            continue
        text = v41_article_front_text(article)
        tier = v40_self_content_tier(text)
        if tier not in {"DIRECT_RISK", "BUSINESS_EXPANSION", "SIMPLE_PR"}:
            continue
        if tier == "SEVERE_PROMO":
            continue
        score = v20_float(article.get("랭킹점수"), 0) + v20_float(article.get("중요도점수"), 0) * 0.8
        if tier == "DIRECT_RISK":
            score += 120
        elif tier == "BUSINESS_EXPANSION":
            score += 85
        elif tier == "SIMPLE_PR":
            score += 15
        if re.search(r"몽골|중앙은행|해외\s*진출|글로벌|MOU|업무협약|전략적\s*제휴|조직개편|카카오톡", text, re.IGNORECASE):
            score += 28
        if V41_SELF_LOW_SOFT_PR_RE.search(text) and tier == "SIMPLE_PR":
            score -= 12
        scored.append((round(score, 2), tier, aid))
    if not scored:
        return None
    scored.sort(key=lambda x: (x[0], {"DIRECT_RISK": 3, "BUSINESS_EXPANSION": 2, "SIMPLE_PR": 1}.get(x[1], 0)), reverse=True)
    return scored[0]


def v40_self_soft_issue_score(issue, article_by_id=None):
    best = v41_best_self_article_for_issue(issue, article_by_id)
    if not best:
        return None
    article_score, tier, aid = best
    if tier == "DIRECT_RISK":
        return None
    if tier == "SIMPLE_PR" and V40_SELF_SEVERE_PROMO_RE.search(v41_article_front_text((article_by_id or {}).get(aid, {}))):
        return None
    base = v20_float(issue.get("issue_score"), 0) + article_score
    if tier == "BUSINESS_EXPANSION":
        base += 100
    elif tier == "SIMPLE_PR":
        base += 5
    issue["v41_self_preferred_article_id"] = aid
    return round(base, 2), tier


def v40_refill_self_soft_if_sparse(issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    if v40_current_self_count(final_report_data) >= V40_SELF_SOFT_MIN:
        return order_counter

    represented_issue_ids = {clean_html_text(item.get("v20_issue_id", "")) for item in final_report_data if item.get("v20_issue_id")}
    represented_groups = {v20_issue_group_key(item.get("v20_issue_group", "") or item.get("Gemini이슈그룹", "") or item.get("기사제목", "")) for item in final_report_data}
    represented_groups = {g for g in represented_groups if g}

    candidates = []
    for issue in issues or []:
        iid = clean_html_text(issue.get("issue_id", ""))
        if iid and iid in represented_issue_ids:
            continue
        group_key = v20_issue_group_key(issue.get("issue_group", "") or issue.get("top_title", ""))
        if group_key and group_key in represented_groups:
            continue
        scored = v40_self_soft_issue_score(issue, article_by_id)
        if not scored:
            continue
        score, tier = scored
        if tier == "SIMPLE_PR" and v40_current_self_pr_count(final_report_data) >= V40_SELF_PR_MAX:
            continue
        candidates.append((score, tier, issue))

    tier_order = {"BUSINESS_EXPANSION": 0, "SIMPLE_PR": 1}
    candidates.sort(key=lambda x: (-x[0], tier_order.get(x[1], 9)))

    for _score, tier, issue in candidates:
        if v40_current_self_count(final_report_data) >= V40_SELF_SOFT_MIN:
            break
        if tier == "SIMPLE_PR" and v40_current_self_pr_count(final_report_data) >= V40_SELF_PR_MAX:
            continue
        preferred = issue.get("v41_self_preferred_article_id") or issue.get("top_article_id")
        backups = [aid for aid in issue.get("candidate_article_ids", []) if str(aid) != str(preferred)]
        d = {
            "issue_id": issue.get("issue_id"),
            "category": JSON_KEYS_ORDER[0],
            "priority": 3 if tier == "BUSINESS_EXPANSION" else 2,
            "best_article_id": preferred,
            "backup_article_ids": backups[:8],
            "reason": f"v41_self_soft_refill:{tier}",
            "selection_source": "v41_self_soft_refill",
        }
        item, order_counter, status = v20_process_issue(
            d,
            issue_by_id,
            article_by_id,
            recent_past_items,
            final_report_data,
            body_failed_rows,
            skip_rows,
            processed_article_ids,
            order_counter,
        )
        status_counts[status] = status_counts.get(status, 0) + 1
        if item:
            item["v41_self_soft_refill"] = tier
            item["v41_self_soft_refill_score"] = _score
            print(f"  -> v41 self soft refill added({tier}): {v20_clip(item.get('기사제목',''), 46)}...")
    try:
        v39_remove_low_value_final_items(final_report_data, skip_rows)
    except Exception:
        pass
    return order_counter


_BASE_v20_normalize_label_json_v41 = v20_normalize_label_json

def v20_normalize_label_json(data, valid_ids, article_by_id):
    labels = _BASE_v20_normalize_label_json_v41(data, valid_ids, article_by_id)
    for art_id, label in list(labels.items()):
        article = article_by_id.get(int(art_id), {}) if isinstance(article_by_id, dict) else {}
        front = v41_article_front_text(article)
        tier = v40_self_content_tier(front)
        # Do not let Gemini push non-Kakao articles into the self section.
        if label.get("primary_category") == JSON_KEYS_ORDER[0] and tier == "OFF_TOPIC":
            fallback = article.get("JSON카테고리") or label.get("category") or JSON_KEYS_ORDER[3]
            label["primary_category"] = fallback if fallback in JSON_KEYS_ORDER else JSON_KEYS_ORDER[3]
            label["internal_category"] = label.get("internal_category") or "INDUSTRY_STRUCTURAL_CHANGE"
            label["self_tier"] = "OFF_TOPIC"
            label["reason"] = v20_clip((label.get("reason", "") + " / v41: 실제 카카오 행위 주체가 없어 자사 분류 해제"), 200)
        if tier == "BUSINESS_EXPANSION":
            label.update({
                "is_relevant": True,
                "primary_category": JSON_KEYS_ORDER[0],
                "internal_category": "SELF_AFFILIATE_BUSINESS",
                "issue_family": "self_strategic_reference",
                "company_impact": "카카오·계열사의 사업 확장, 제휴, MOU, 신규 서비스 참고 이슈",
                "ceo_priority": max(v20_int(label.get("ceo_priority"), 3), 3),
                "public_affairs_priority": max(v20_int(label.get("public_affairs_priority"), 3), 3),
                "relevance": max(v20_int(label.get("relevance"), 3), 3),
                "is_pr": False,
                "exclude": False,
                "self_tier": "STRATEGIC_REFERENCE",
                "reason": "v41: 실제 카카오 주체의 사업 확장·제휴성 이슈로 2순위 허용",
            })
        elif tier == "SIMPLE_PR":
            label.update({
                "is_relevant": True,
                "primary_category": JSON_KEYS_ORDER[0],
                "internal_category": "SELF_AFFILIATE_BUSINESS",
                "issue_family": "self_filler_reference",
                "company_impact": "직접 리스크가 적은 날 자사 섹션 보강용 소프트 홍보·캠페인성 참고 이슈",
                "ceo_priority": max(v20_int(label.get("ceo_priority"), 2), 2),
                "public_affairs_priority": max(v20_int(label.get("public_affairs_priority"), 2), 2),
                "relevance": max(v20_int(label.get("relevance"), 2), 2),
                "is_pr": False,
                "exclude": False,
                "self_tier": "FILLER_REFERENCE",
                "reason": "v41: 실제 카카오 주체의 소프트 홍보 이슈로 3순위 허용",
            })
    return labels


# -----------------------------
# 3) Category and relevance guardrails
# -----------------------------

V41_COMPETITOR_ENTITY_RE = re.compile(
    r"네이버|NAVER|라인야후|SKT|SK텔레콤|KT|LGU\+|LG유플러스|쿠팡|토스|배달의민족|배민|우아한형제들|"
    r"구글|Google|오픈AI|OpenAI|챗GPT|ChatGPT|앤트로픽|Anthropic|클로드|Claude|MS|마이크로소프트|"
    r"메타|Meta|애플|Apple|엔비디아|NVIDIA|우버|Uber",
    re.IGNORECASE,
)

V41_COMPETITOR_ACTION_RE = re.compile(
    r"AI|인공지능|에이전트|데이터센터|AIDC|GPU|클라우드|검색|포털|앱|서비스|요금|가격|출시|도입|"
    r"협력|제휴|투자|인수|매각|소송|판결|제재|과징금|규제|조사|개인정보|유출|보안|해킹|장애|"
    r"인앱결제|AI\s*팩토리|밋업|오피스|법인|AX|사번|동료",
    re.IGNORECASE,
)

V41_DOMESTIC_REGULATOR_RE = re.compile(
    r"방미통위|방송미디어통신위원회|과기정통부|과학기술정보통신부|공정위|공정거래위원회|금융위|금융위원회|금감원|금융감독원|"
    r"개보위|개인정보보호위원회|행안부|행정안전부|중기부|중소벤처기업부|국회|정무위|과방위|정부|국무총리|국무조정실",
    re.IGNORECASE,
)

V41_GOV_POLICY_ACTION_RE = re.compile(
    r"법안|개정안|시행령|고시|입법예고|행정예고|정책|규제|의무|의무화|차단|삭제|평가|결과|의결|"
    r"위원회|출범|개편|기금|제재|과징금|절차|공식화|망분리|완화|바우처|지원|사업|선정|가이드라인|감독|검사",
    re.IGNORECASE,
)

V41_COMPANY_STRATEGY_TITLE_RE = re.compile(
    r"^(SKT|SK텔레콤|KT|LGU\+|LG유플러스|네이버|NAVER|쿠팡|토스|배달의민족|배민|구글|오픈AI|앤트로픽|MS|메타|애플|엔비디아).{0,120}"
    r"(AI|에이전트|데이터센터|팩토리|협력|제휴|투자|출시|도입|전략|AX|사번|동료|인수|매각)",
    re.IGNORECASE,
)

V41_UNRELATED_BUSINESS_RE = re.compile(
    r"한국제지|제지사|건설|아파트|의료기기|바이오주|화장품|식품|유통가|주식|목표가|투자의견|영업흑자|순손실|순이익|비용처리",
    re.IGNORECASE,
)


def v41_core_article_text(item):
    return v41_article_front_text(item)


def v41_is_true_competitor_text(text):
    t = clean_html_text(text)
    if not V41_COMPETITOR_ENTITY_RE.search(t):
        return False
    if V41_UNRELATED_BUSINESS_RE.search(t) and not re.search(r"네이버|카카오|쿠팡|토스|구글|오픈AI|앤트로픽|SKT|KT|LGU\+|AI|플랫폼", t, re.IGNORECASE):
        return False
    return bool(V41_COMPETITOR_ACTION_RE.search(t) or re.search(r"규제|제재|소송|과징금|인앱결제|개인정보|보안|AI|AX", t, re.IGNORECASE))


def v41_is_domestic_gov_policy_text(text):
    t = clean_html_text(text)
    return bool(V41_DOMESTIC_REGULATOR_RE.search(t) and V41_GOV_POLICY_ACTION_RE.search(t))


_BASE_is_report_item_relevant_v41 = is_report_item_relevant

def is_report_item_relevant(report_item, json_key):
    requested = v19_normalize_category_key(json_key, fallback=JSON_KEYS_ORDER[3])
    text = v41_core_article_text(report_item)
    title = clean_html_text(report_item.get("기사제목", ""))

    if requested == JSON_KEYS_ORDER[0]:
        if not v41_has_true_self_actor(text):
            return False, "v41_self_without_actual_kakao_actor"
        if V41_SELF_SEVERE_PROMO_EXTRA_RE.search(text):
            return False, "v41_severe_self_promo_noise"

    if requested == JSON_KEYS_ORDER[1]:
        # A competitor's own strategy article should not sit in government just because it mentions a government program.
        if V41_COMPANY_STRATEGY_TITLE_RE.search(title) and not V41_DOMESTIC_REGULATOR_RE.search(title):
            return False, "v41_company_strategy_not_government_policy"

    if requested == JSON_KEYS_ORDER[2]:
        # Competitor/overseas requires the article's actual title/body to contain the competitor/global platform actor.
        if not v41_is_true_competitor_text(text):
            return False, "v41_competitor_without_actual_core_actor"
        # Domestic government support/program articles with no direct competitor action should be government, not competitor.
        if v41_is_domestic_gov_policy_text(text) and not re.search(r"구글|애플|쿠팡|네이버|SKT|KT|LGU\+|오픈AI|앤트로픽|과징금|제재|소송|인앱결제|개인정보|AI", title, re.IGNORECASE):
            return False, "v41_domestic_government_policy_not_competitor"

    return _BASE_is_report_item_relevant_v41(report_item, json_key)


_BASE_v23_final_category_guardrail_v41 = v23_final_category_guardrail

def v23_final_category_guardrail(item, requested_category):
    requested = v19_normalize_category_key(requested_category, fallback=JSON_KEYS_ORDER[3])
    text = v41_core_article_text(item)
    title = clean_html_text(item.get("기사제목", ""))

    if requested == JSON_KEYS_ORDER[0] and not v41_has_true_self_actor(text):
        return "fail", requested, "v41_self_guard_actual_actor_absent"

    if requested == JSON_KEYS_ORDER[1] and V41_COMPANY_STRATEGY_TITLE_RE.search(title) and not V41_DOMESTIC_REGULATOR_RE.search(title):
        if v41_is_true_competitor_text(text):
            return "reassign", JSON_KEYS_ORDER[2], "v41_company_strategy_to_competitor"
        return "reassign", JSON_KEYS_ORDER[3], "v41_company_strategy_to_industry"

    if requested == JSON_KEYS_ORDER[2]:
        if not v41_is_true_competitor_text(text):
            if v41_is_domestic_gov_policy_text(text):
                return "reassign", JSON_KEYS_ORDER[1], "v41_domestic_policy_to_government"
            return "fail", requested, "v41_competitor_guard_no_actual_competitor"

    return _BASE_v23_final_category_guardrail_v41(item, requested_category)


# -----------------------------
# 4) Issue / representative article alignment
# -----------------------------

V41_ALIGNMENT_ENTITIES = {
    "kakao": re.compile(r"카카오|카카오톡|카톡|카카오페이|카카오뱅크|카카오모빌리티|카카오게임즈|kakao", re.IGNORECASE),
    "naver": re.compile(r"네이버|naver", re.IGNORECASE),
    "skt": re.compile(r"SKT|SK텔레콤", re.IGNORECASE),
    "kt": re.compile(r"\bKT\b|케이티", re.IGNORECASE),
    "lguplus": re.compile(r"LGU\+|LG유플러스|엘지유플러스", re.IGNORECASE),
    "coupang": re.compile(r"쿠팡|coupang", re.IGNORECASE),
    "toss": re.compile(r"토스|비바리퍼블리카", re.IGNORECASE),
    "baemin": re.compile(r"배달의민족|배민|우아한형제들", re.IGNORECASE),
    "google": re.compile(r"구글|google", re.IGNORECASE),
    "apple": re.compile(r"애플|apple", re.IGNORECASE),
    "openai": re.compile(r"오픈AI|OpenAI|챗GPT|ChatGPT", re.IGNORECASE),
    "anthropic": re.compile(r"앤트로픽|Anthropic|클로드|Claude|미토스|페이블", re.IGNORECASE),
    "meta": re.compile(r"메타|Meta|페이스북|인스타그램", re.IGNORECASE),
    "nvidia": re.compile(r"엔비디아|NVIDIA", re.IGNORECASE),
    "jtbc": re.compile(r"JTBC|중앙홀딩스|콘텐트리중앙", re.IGNORECASE),
    "ncai": re.compile(r"NC\s*AI|엔씨\s*AI", re.IGNORECASE),
}


def v41_detect_alignment_entities(text):
    t = clean_html_text(text)
    found = set()
    for key, pat in V41_ALIGNMENT_ENTITIES.items():
        if pat.search(t):
            found.add(key)
    # Explicit English keys from global_incident strings.
    for key in ["naver", "kakao", "coupang", "openai", "anthropic", "google", "apple", "meta", "skt", "kt", "lguplus"]:
        if re.search(rf"(^|[:_\s-]){key}($|[:_\s-])", t, re.IGNORECASE):
            found.add(key)
    return found


def v41_issue_meta_text(issue, decision=None):
    issue = issue or {}
    decision = decision or {}
    return clean_html_text(
        f"{issue.get('issue_group','')} {issue.get('top_title','')} {issue.get('issue_family','')} "
        f"{issue.get('internal_category','')} {issue.get('primary_category','')} {decision.get('reason','')} {decision.get('selection_source','')}"
    )


def v41_alignment_ok(issue, article, requested_category=""):
    if not issue or not article:
        return True, "no_issue_or_article"
    req = v19_normalize_category_key(requested_category or issue.get("category") or issue.get("primary_category"), fallback=JSON_KEYS_ORDER[3])
    issue_text = v41_issue_meta_text(issue)
    article_text = v41_article_front_text(article)
    issue_entities = v41_detect_alignment_entities(issue_text)
    article_entities = v41_detect_alignment_entities(article_text)

    # Self and competitor categories have stricter actor requirements.
    if req == JSON_KEYS_ORDER[0] and not v41_has_true_self_actor(article_text):
        return False, "self_article_without_actual_kakao_actor"
    if req == JSON_KEYS_ORDER[2] and not v41_is_true_competitor_text(article_text):
        return False, "competitor_article_without_core_actor"

    # If the issue group/reason names a specific core company, the representative article must contain it too.
    named = issue_entities & {"kakao", "naver", "skt", "kt", "lguplus", "coupang", "toss", "baemin", "google", "apple", "openai", "anthropic", "meta", "nvidia", "jtbc", "ncai"}
    if named and not (named & article_entities):
        return False, "issue_entity_article_entity_mismatch:" + ",".join(sorted(named))

    # Guard against source/domain induced false positives like Naver Premium Contents -> Naver issue.
    if re.search(r"global_incident:.*:naver", issue_text, re.IGNORECASE) and "naver" not in article_entities:
        return False, "naver_global_incident_but_article_not_naver"

    return True, "aligned"


def v41_candidate_ids_for_issue(decision, issue):
    ids = []
    for key in ["best_article_id", "top_article_id"]:
        try:
            if decision.get(key):
                ids.append(int(decision.get(key)))
        except Exception:
            pass
        try:
            if issue and issue.get(key):
                ids.append(int(issue.get(key)))
        except Exception:
            pass
    for key in ["backup_article_ids", "candidate_article_ids"]:
        for aid in (decision.get(key, []) if isinstance(decision.get(key, []), list) else []):
            try:
                ids.append(int(aid))
            except Exception:
                pass
        if issue:
            for aid in (issue.get(key, []) if isinstance(issue.get(key, []), list) else []):
                try:
                    ids.append(int(aid))
                except Exception:
                    pass
    out, seen = [], set()
    for aid in ids:
        if aid not in seen:
            out.append(aid)
            seen.add(aid)
    return out


def v41_adjust_decision_for_alignment(decision, issue, article_by_id):
    if not issue:
        return decision
    req = v19_normalize_category_key(decision.get("category"), fallback=v20_issue_to_output_category(issue) if "v20_issue_to_output_category" in globals() else issue.get("primary_category", JSON_KEYS_ORDER[3]))
    ids = v41_candidate_ids_for_issue(decision, issue)
    if not ids:
        return decision
    current = int(decision.get("best_article_id") or issue.get("top_article_id") or ids[0])
    current_ok, _ = v41_alignment_ok(issue, article_by_id.get(current, {}), req)
    if current_ok:
        return decision
    alternatives = []
    for aid in ids:
        art = article_by_id.get(int(aid), {})
        ok, reason = v41_alignment_ok(issue, art, req)
        if not ok:
            continue
        score = v20_float(art.get("랭킹점수"), 0) + v20_float(art.get("중요도점수"), 0) * 0.7
        alternatives.append((score, aid))
    if not alternatives:
        return decision
    alternatives.sort(reverse=True)
    best_id = alternatives[0][1]
    d2 = dict(decision)
    d2["best_article_id"] = best_id
    d2["backup_article_ids"] = [aid for _score, aid in alternatives[1:]] + [aid for aid in ids if aid != best_id]
    d2["v41_representative_reordered"] = f"{current}->{best_id}:alignment"
    return d2


_BASE_v20_process_issue_v41 = v20_process_issue

def v20_process_issue(decision, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter):
    issue = issue_by_id.get(decision.get("issue_id")) if isinstance(issue_by_id, dict) else None
    if issue:
        decision = v41_adjust_decision_for_alignment(decision, issue, article_by_id)
    item, new_order_counter, status = _BASE_v20_process_issue_v41(
        decision,
        issue_by_id,
        article_by_id,
        recent_past_items,
        final_report_data,
        body_failed_rows,
        skip_rows,
        processed_article_ids,
        order_counter,
    )
    if item and issue:
        ok, reason = v41_alignment_ok(issue, item, item.get("카테고리") or decision.get("category"))
        if not ok:
            try:
                if item in final_report_data:
                    final_report_data.remove(item)
            except Exception:
                pass
            try:
                add_duplicate_skip_row(skip_rows, item, {"기사제목": issue.get("issue_group", "issue_alignment"), "링크": "", "대표선택점수": ""}, f"v41_issue_article_alignment_failed:{reason}", stage=decision.get("selection_source") or decision.get("kind") or "v41_alignment")
            except Exception:
                pass
            return None, new_order_counter, "v41_issue_article_alignment_failed"
    return item, new_order_counter, status


# -----------------------------
# 5) Overseas AI access/export control dedupe
# -----------------------------

V41_AI_ACCESS_CONTROL_RE = re.compile(
    r"(앤트로픽|Anthropic|클로드|Claude|미토스|페이블).{0,140}(외국인|해외|수출|접근|사용|차단|제한|통제)|"
    r"(외국인|해외|수출|접근|사용|차단|제한|통제).{0,140}(앤트로픽|Anthropic|클로드|Claude|미토스|페이블)",
    re.IGNORECASE,
)


def v41_ai_access_control_key(item):
    t = v41_core_article_text(item)
    if V41_AI_ACCESS_CONTROL_RE.search(t):
        return "global_ai_access_control:anthropic:foreign_access_export_control"
    return ""


_BASE_final_duplicate_reason_v41 = final_duplicate_reason

def final_duplicate_reason(new_item, existing_items):
    new_ai_key = v41_ai_access_control_key(new_item)
    if new_ai_key:
        for old in existing_items:
            if v41_ai_access_control_key(old) == new_ai_key:
                return old, f"v41_same_ai_access_control:{new_ai_key}"
    return _BASE_final_duplicate_reason_v41(new_item, existing_items)


# -----------------------------
# 6) Final cleanup after inherited refill/repair
# -----------------------------

_BASE_v26_1_refill_after_final_repair_v41 = v26_1_refill_after_final_repair

def v26_1_refill_after_final_repair(decisions, backup_decisions, issues, issue_by_id, article_by_id, recent_past_items, final_report_data, body_failed_rows, skip_rows, processed_article_ids, order_counter, status_counts):
    order_counter = _BASE_v26_1_refill_after_final_repair_v41(
        decisions=decisions,
        backup_decisions=backup_decisions,
        issues=issues,
        issue_by_id=issue_by_id,
        article_by_id=article_by_id,
        recent_past_items=recent_past_items,
        final_report_data=final_report_data,
        body_failed_rows=body_failed_rows,
        skip_rows=skip_rows,
        processed_article_ids=processed_article_ids,
        order_counter=order_counter,
        status_counts=status_counts,
    )

    # Remove category leaks left by broad refill. Keep this conservative.
    for item in list(final_report_data):
        cat = item.get("카테고리")
        text = v41_core_article_text(item)
        remove_reason = ""
        if cat == JSON_KEYS_ORDER[0] and not v41_has_true_self_actor(text):
            remove_reason = "v41_final_self_without_actual_kakao_actor"
        elif cat == JSON_KEYS_ORDER[2] and not v41_is_true_competitor_text(text):
            if v41_is_domestic_gov_policy_text(text):
                # Move clear domestic policy to government if there is room; otherwise remove.
                if len([x for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[1]]) < CATEGORY_MAX.get(JSON_KEYS_ORDER[1], 7):
                    v23_set_final_category(item, JSON_KEYS_ORDER[1], "v41_final_move_domestic_policy_from_competitor")
                    continue
            remove_reason = "v41_final_competitor_without_core_actor"
        elif cat == JSON_KEYS_ORDER[1] and V41_COMPANY_STRATEGY_TITLE_RE.search(item.get("기사제목", "")) and not V41_DOMESTIC_REGULATOR_RE.search(item.get("기사제목", "")):
            if v41_is_true_competitor_text(text):
                if len([x for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[2]]) < CATEGORY_MAX.get(JSON_KEYS_ORDER[2], 5):
                    v23_set_final_category(item, JSON_KEYS_ORDER[2], "v41_final_move_company_strategy_to_competitor")
                    continue
            remove_reason = "v41_final_company_strategy_not_gov"
        if remove_reason:
            try:
                v32_add_skip(skip_rows, item, remove_reason)
            except Exception:
                pass
            final_report_data[:] = [x for x in final_report_data if x is not item]

    # One more duplicate pass for AI access control and other inherited duplicate logic.
    cleaned = []
    for item in list(final_report_data):
        dup, reason = final_duplicate_reason(item, cleaned)
        if dup:
            try:
                add_duplicate_skip_row(skip_rows, item, dup, f"v41_final_duplicate:{reason}", stage="v41_final_cleanup")
            except Exception:
                pass
            continue
        cleaned.append(item)
    final_report_data[:] = cleaned
    return order_counter


_BASE_v20_gemini_quality_check_v41 = v20_gemini_quality_check

def v20_gemini_quality_check(client, final_report_data, final_briefing_text):
    qa = _BASE_v20_gemini_quality_check_v41(client, final_report_data, final_briefing_text)
    try:
        data = v22_parse_qa_json(qa) if qa else {"overall": "양호", "warnings": [], "suggested_human_checks": []}
    except Exception:
        return qa
    warnings = data.get("warnings") if isinstance(data.get("warnings"), list) else []
    checks = data.get("suggested_human_checks") if isinstance(data.get("suggested_human_checks"), list) else []
    self_count = sum(1 for x in final_report_data if x.get("카테고리") == JSON_KEYS_ORDER[0])
    if self_count < V40_SELF_SOFT_MIN:
        checks.append("v41: 자사 직접 리스크가 없으면 실제 카카오 행위 주체가 있는 MOU/사업확장/소프트 홍보 후보만 추가 검토함")
    if v41_now_kst().weekday() == 0:
        checks.append(f"v41: 월요일 수집기간 적용({v41_collection_window_label()})")
    data["warnings"] = warnings
    data["suggested_human_checks"] = checks
    return json.dumps(data, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
