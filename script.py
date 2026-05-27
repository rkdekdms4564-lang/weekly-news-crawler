import requests
import urllib.parse
import time
import random
import re
import json
import os
import html
import pandas as pd
import feedparser

from google import genai
from datetime import datetime
from newspaper import Article, Config
from bs4 import BeautifulSoup


# ==========================================
# 0. 기본 설정
# ==========================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MAX_ATTEMPTS = 5

RECENT_DAYS = 1
MAX_NEWS_PER_KEYWORD = 20
MAX_COMPRESSED_PER_KEYWORD = 8

USE_NAVER_IF_AVAILABLE = True
USE_GOOGLE_RSS = True

OUTPUT_TXT = os.path.join(BASE_DIR, "CEO_Morning_Briefing.txt")
OUTPUT_SELECTED_CSV = os.path.join(BASE_DIR, "google_news_top15_raw.csv")
OUTPUT_CANDIDATES_CSV = os.path.join(BASE_DIR, "news_candidates_raw.csv")

WEB_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

try:
    from googlenewsdecoder import gnewsdecoder
except Exception:
    gnewsdecoder = None


# ==========================================
# 1. Gemini AI 초기 세팅
# ==========================================

try:
    with open(os.path.join(BASE_DIR, "secret.txt"), "r", encoding="utf-8") as f:
        GOOGLE_API_KEY = f.read().strip()

    if not GOOGLE_API_KEY:
        raise ValueError("secret.txt가 비어 있습니다.")

    client = genai.Client(api_key=GOOGLE_API_KEY)

except Exception as e:
    print("❌ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다.")
    print(f"   원인: {e}")
    exit()


# ==========================================
# 2. 과거 보고서 로드
# ==========================================

past_reports_content = ""

past_reports_path = os.path.join(BASE_DIR, "past_reports.txt")

if os.path.exists(past_reports_path):
    with open(past_reports_path, "r", encoding="utf-8") as f:
        past_reports_content = f.read().strip()

    print("📚 'past_reports.txt' 로드 완료! AI가 과거 스타일과 중복 기사를 학습합니다.")
else:
    print("📝 'past_reports.txt' 파일이 없습니다. 새로운 데이터로만 진행합니다.")


# ==========================================
# 3. 유저 확정 고정 키워드 리스트
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

QUOTAS = {
    "자사_및_계열사_이슈": 4,
    "정부_국회": 5,
    "경쟁사_해외이슈": 4,
    "산업동향": 2,
}

KEY_ALIASES = {
    "자사_및_계열사_이슈": [
        "자사_및_계열사_이슈",
        "자사 및 계열사 이슈",
        "자사및계열사이슈",
        "카카오",
        "자사",
    ],
    "정부_국회": [
        "정부_국회",
        "정부/국회",
        "정부·국회",
        "정부 국회",
        "정부",
        "국회",
    ],
    "경쟁사_해외이슈": [
        "경쟁사_해외이슈",
        "경쟁사/해외이슈",
        "경쟁사·해외이슈",
        "경쟁사 해외이슈",
        "해외이슈",
        "경쟁사",
    ],
    "산업동향": [
        "산업동향",
        "산업 동향",
    ],
}

all_keywords = []
keyword_to_category = {}

for category_name, keyword_list in keyword_categories.items():
    for keyword in keyword_list:
        all_keywords.append(keyword)
        keyword_to_category[keyword] = category_name


# ==========================================
# 4. 공통 유틸 함수
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
        netloc = netloc.replace("www.", "")
        return netloc
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


def decode_google_news_url(google_news_url):
    """
    Google News RSS URL을 원문 언론사 URL로 변환.
    googlenewsdecoder가 실패하면 redirect URL을 시도하고,
    그것도 실패하면 원래 Google News URL을 반환.
    """
    if not google_news_url:
        return ""

    if gnewsdecoder is not None and "news.google." in google_news_url:
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


def add_article(
    raw_articles,
    seen_urls,
    article_id,
    keyword,
    title,
    link,
    source="",
    published="",
    summary="",
    collector="",
):
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


def gemini_generate_text(client, prompt, task_name, model=GEMINI_MODEL, max_attempts=GEMINI_MAX_ATTEMPTS):
    """
    Gemini 호출 재시도 함수.
    고정 5초가 아니라 지수 백오프 + 랜덤 지터를 사용.
    """
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
    """
    Gemini 응답에서 JSON 객체만 추출.
    ```json ... ``` 형태도 처리.
    """
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

    # 흔한 trailing comma 제거
    json_text = re.sub(r",\s*([\]}])", r"\1", json_text)

    return json.loads(json_text)


def normalize_selection_json(data, final_candidates):
    """
    Gemini가 반환한 JSON을 기대 포맷으로 정리.
    ID 중복 제거, 존재하지 않는 ID 제거.
    """
    valid_ids = {int(a["id"]) for a in final_candidates}
    used_ids = set()

    result = {key: [] for key in JSON_KEYS_ORDER}

    for expected_key in JSON_KEYS_ORDER:
        values = None

        for alias in KEY_ALIASES.get(expected_key, [expected_key]):
            if alias in data:
                values = data.get(alias)
                break

        if values is None:
            values = []

        if not isinstance(values, list):
            values = []

        cleaned_ids = []

        for value in values:
            try:
                art_id = int(value)
            except Exception:
                continue

            if art_id not in valid_ids:
                continue

            if art_id in used_ids:
                continue

            cleaned_ids.append(art_id)
            used_ids.add(art_id)

        result[expected_key] = cleaned_ids

    return result


def deterministic_selection(final_candidates):
    """
    Gemini 선별 실패 시에도 보고서가 멈추지 않도록 기본 선별 수행.
    카테고리별 앞쪽 기사에서 quota만큼 선택.
    """
    result = {key: [] for key in JSON_KEYS_ORDER}
    used_ids = set()

    for json_key in JSON_KEYS_ORDER:
        quota = QUOTAS.get(json_key, 3)

        category_candidates = [
            a for a in final_candidates
            if a.get("JSON카테고리") == json_key and int(a["id"]) not in used_ids
        ]

        for article in category_candidates[:quota]:
            art_id = int(article["id"])
            result[json_key].append(art_id)
            used_ids.add(art_id)

    # 전체가 13개 미만이면 남은 기사에서 추가
    total_selected = sum(len(v) for v in result.values())

    if total_selected < 13:
        for article in final_candidates:
            art_id = int(article["id"])

            if art_id in used_ids:
                continue

            json_key = article.get("JSON카테고리") or "산업동향"

            if json_key not in result:
                json_key = "산업동향"

            result[json_key].append(art_id)
            used_ids.add(art_id)

            total_selected = sum(len(v) for v in result.values())

            if total_selected >= 13:
                break

    return result


def extract_article_body(url):
    """
    1차: newspaper3k
    2차: requests + BeautifulSoup p 태그 추출
    """
    if not url:
        return "", "none"

    # 1차: newspaper3k
    try:
        config = Config()
        config.browser_user_agent = WEB_HEADERS["User-Agent"]
        config.request_timeout = 12

        article = Article(url=url, language="ko", config=config)
        article.download()
        article.parse()

        text = clean_html_text(article.text)

        if len(text) >= 100:
            return text, "newspaper3k"

    except Exception:
        pass

    # 2차: BeautifulSoup fallback
    try:
        r = requests.get(
            url,
            headers=WEB_HEADERS,
            timeout=12,
            allow_redirects=True,
        )

        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")

        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()

        article_body = soup.find(attrs={"itemprop": "articleBody"})

        paragraphs = []

        if article_body:
            paragraphs = article_body.find_all("p")

        if not paragraphs:
            paragraphs = soup.find_all("p")

        chunks = []

        for p in paragraphs:
            text = clean_html_text(p.get_text(" "))

            if len(text) < 30:
                continue

            # 너무 광고성인 문장 일부 제거
            if "무단전재" in text or "재배포 금지" in text:
                continue

            chunks.append(text)

        joined = "\n".join(chunks)

        if len(joined) >= 100:
            return joined, "bs4"

    except Exception:
        pass

    return "", "failed"


def fallback_summary_from_text(text):
    text = clean_html_text(text)

    if not text:
        return "본문 추출에 실패해 원문 링크 확인이 필요함."

    text = text[:450].strip()

    if text.endswith(("함.", "임.", "됨.", "함", "임", "됨")):
        return text

    text = text.rstrip(".")
    return f"{text} 등으로 보도됨."


def build_fallback_briefing(final_report_data):
    """
    Gemini 최종 브리핑 생성 실패 시 최소 보고서 생성.
    """
    lines = []

    number_emojis = [
        "1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣",
        "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"
    ]

    for json_key in JSON_KEYS_ORDER:
        category_name = JSON_KEY_TO_DISPLAY.get(json_key, json_key)
        category_items = [
            item for item in final_report_data
            if item.get("카테고리") == json_key
        ]

        if not category_items:
            continue

        lines.append(f"☑️ {category_name}")
        lines.append("")

        for idx, item in enumerate(category_items, 1):
            if idx <= len(number_emojis):
                num = number_emojis[idx - 1]
            else:
                num = f"{idx}."

            title = item.get("기사제목", "")
            link = item.get("링크", "")
            press = item.get("언론사", "") or guess_press_name_from_url(link)
            body = item.get("본문전문", "") or item.get("본문요약", "")

            lines.append(f"{num} {title}")
            lines.append(link)
            lines.append(f"({press})")
            lines.append(fallback_summary_from_text(body))
            lines.append("")

    if not lines:
        return "최종 브리핑 생성에 실패했으며, 선별된 기사 데이터도 비어 있음."

    return "\n".join(lines).strip()


# ==========================================
# 5. 뉴스 수집 함수 - 네이버 API
# ==========================================

def load_naver_keys():
    path = os.path.join(BASE_DIR, "secret_naver.txt")

    if not os.path.exists(path):
        return None, None

    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    if len(lines) < 2:
        return None, None

    return lines[0], lines[1]


def collect_with_naver_api(raw_articles, seen_urls, article_id):
    client_id, client_secret = load_naver_keys()

    if not client_id or not client_secret:
        print("📝 'secret_naver.txt'가 없어 네이버 뉴스 API 수집은 건너뜁니다.")
        return article_id

    print("\n🚀 [STEP 1-A] 네이버 뉴스 검색 API로 뉴스 후보 수집 시작...")

    headers = {
        "X-Naver-Client-Id": client_id,
        "X-Naver-Client-Secret": client_secret,
    }

    for idx, keyword in enumerate(all_keywords, 1):
        collected = 0

        params = {
            "query": keyword,
            "display": MAX_NEWS_PER_KEYWORD,
            "start": 1,
            "sort": "date",
        }

        try:
            req = requests.get(
                "https://openapi.naver.com/v1/search/news.json",
                headers=headers,
                params=params,
                timeout=10,
            )

            req.raise_for_status()
            data = req.json()

            for item in data.get("items", []):
                title = clean_html_text(item.get("title", ""))
                link = item.get("originallink") or item.get("link") or ""
                summary = clean_html_text(item.get("description", ""))
                published = clean_html_text(item.get("pubDate", ""))
                source = guess_press_name_from_url(link)

                article_id, added = add_article(
                    raw_articles=raw_articles,
                    seen_urls=seen_urls,
                    article_id=article_id,
                    keyword=keyword,
                    title=title,
                    link=link,
                    source=source,
                    published=published,
                    summary=summary,
                    collector="naver_api",
                )

                if added:
                    collected += 1

            print(
                f"▶ [NAVER {idx}/{len(all_keywords)}] "
                f"'{keyword}' 수집 ➔ {collected}개 확보        ",
                end="\r"
            )

        except Exception as e:
            print(f"\n⚠️ [NAVER {idx}/{len(all_keywords)}] '{keyword}' 수집 실패: {e}")

        time.sleep(random.uniform(0.2, 0.5))

    print("")
    return article_id


# ==========================================
# 6. 뉴스 수집 함수 - Google News RSS
# ==========================================

def collect_with_google_rss(raw_articles, seen_urls, article_id):
    print("\n🚀 [STEP 1-B] Google News RSS로 뉴스 후보 수집 시작...")

    for idx, keyword in enumerate(all_keywords, 1):
        collected = 0

        query = urllib.parse.quote_plus(f"{keyword} when:{RECENT_DAYS}d")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"

        try:
            req = requests.get(
                rss_url,
                headers=WEB_HEADERS,
                timeout=10,
            )

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

                real_url = decode_google_news_url(google_link)

                article_id, added = add_article(
                    raw_articles=raw_articles,
                    seen_urls=seen_urls,
                    article_id=article_id,
                    keyword=keyword,
                    title=title,
                    link=real_url,
                    source=source,
                    published=published,
                    summary=summary,
                    collector="google_rss",
                )

                if added:
                    collected += 1

            print(
                f"▶ [GOOGLE RSS {idx}/{len(all_keywords)}] "
                f"'{keyword}' 수집 ➔ {collected}개 확보        ",
                end="\r"
            )

        except Exception as e:
            print(f"\n⚠️ [GOOGLE RSS {idx}/{len(all_keywords)}] '{keyword}' 수집 실패: {e}")

        time.sleep(random.uniform(0.3, 0.8))

    print("")
    return article_id


# ==========================================
# 7. 메인 실행
# ==========================================

total_start_time = time.time()

raw_articles = []
seen_urls = set()
article_id = 1

print("\n🚀 [STEP 1] 뉴스 후보 수집 시작...")

if USE_NAVER_IF_AVAILABLE:
    article_id = collect_with_naver_api(raw_articles, seen_urls, article_id)

if USE_GOOGLE_RSS:
    article_id = collect_with_google_rss(raw_articles, seen_urls, article_id)

print(f"\n  └ ✅ 총 {len(raw_articles)}개의 뉴스 후보군 풀(Pool) 확보 완료!")

if raw_articles:
    pd.DataFrame(raw_articles).to_csv(
        OUTPUT_CANDIDATES_CSV,
        index=False,
        encoding="utf-8-sig",
    )
    print(f"  └ 💾 전체 후보 기사 백업 저장 완료: {os.path.basename(OUTPUT_CANDIDATES_CSV)}")

if not raw_articles:
    print("❌ 수집된 기사가 없습니다.")
    print("   확인 포인트:")
    print("   1) 네트워크 연결")
    print("   2) feedparser 설치 여부")
    print("   3) Google News RSS 접속 가능 여부")
    print("   4) 네이버 API 사용 시 secret_naver.txt 형식")
    exit()


# ==========================================
# 8. Gemini AI 선별
# ==========================================

print("\n🧠 [STEP 2] Gemini AI가 과거 보고서를 학습하여 중복 기사를 거르고 핵심 기사를 선별합니다...")

compressed_articles = {}

for article in raw_articles:
    keyword = article["검색어"]

    if keyword not in compressed_articles:
        compressed_articles[keyword] = []

    if len(compressed_articles[keyword]) < MAX_COMPRESSED_PER_KEYWORD:
        compressed_articles[keyword].append(article)

final_candidates = [
    item
    for sublist in compressed_articles.values()
    for item in sublist
]

candidate_text = ""

for article in final_candidates:
    candidate_text += (
        f"[{article['id']}] "
        f"카테고리: {article.get('원카테고리', '')} / "
        f"검색어: {article.get('검색어', '')} / "
        f"제목: {article.get('기사제목', '')} / "
        f"언론사: {article.get('언론사', '')} / "
        f"게시일: {article.get('게시일', '')}\n"
    )

prompt_selection = f"""
너는 IT 대기업의 유능한 최고 비서실장이야.
아래 제공하는 [과거 보고서 데이터]를 정독하고, 유저가 어떤 무게감의 기사를 선별했는지 그 기준을 학습해.
그 후, 오늘 수집된 [오늘 뉴스 후보 리스트] 중에서 딱 13~15개의 최정예 기사만 골라내줘.

[기사 선별 가이드라인]
1. 오피니언, 사설, 칼럼, 전문가 기고는 무조건 제외.
2. 서비스 출시 등 단순 이벤트 홍보 기사는 철저히 배제.
3. 카카오 및 계열사는 과징금, 경영진 이슈, 서비스 장애, 지배구조, 실적, 규제, 수사, 소송 등 무겁고 중요한 내용 위주로 선별해.
4. 타사 이름만 들어간 무관한 기사는 걸러내.
5. [과거 보고서 데이터]를 확인하여, 이미 일주일 이내에 다루었던 사건이나 완전히 동일한 내용의 이슈는 제외해.
6. 동일 사건에 대해 여러 기사가 있다면 딱 1개만 대표로 선택해.
7. 카테고리별 할당량은 아래 기준을 최대한 맞춰.
   - 자사 및 계열사 이슈: 3~4개
   - 정부/국회: 4~5개
   - 경쟁사/해외이슈: 3~4개
   - 산업동향: 1~2개

반드시 다른 설명 없이 아래 JSON 형식으로만 응답해.

{{
  "자사_및_계열사_이슈": [ID숫자들],
  "정부_국회": [ID숫자들],
  "경쟁사_해외이슈": [ID숫자들],
  "산업동향": [ID숫자들]
}}

[과거 보고서 데이터]
{past_reports_content[:4000]}

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

    print(f"  └ ✅ AI 선별 완료! 총 {selected_count}개의 핵심 기사 확보.")

except Exception as e:
    print(f"  └ ⚠️ AI 선별 실패. 기본 선별 로직으로 진행합니다. 원인: {e}")

    json_data = deterministic_selection(final_candidates)
    selected_count = sum(len(ids) for ids in json_data.values())

    print(f"  └ ✅ 기본 선별 완료! 총 {selected_count}개의 핵심 기사 확보.")


# ==========================================
# 9. 본문 추출
# ==========================================

print("\n🕵️‍♂️ [STEP 3] 선별된 기사들의 본문을 추출합니다...")

final_report_data = []

article_by_id = {
    int(article["id"]): article
    for article in raw_articles
}

for json_key in JSON_KEYS_ORDER:
    ids = json_data.get(json_key, [])

    for art_id in ids:
        try:
            art_id = int(art_id)
        except Exception:
            continue

        article_info = article_by_id.get(art_id)

        if not article_info:
            continue

        real_url = article_info.get("링크", "")
        body_text, extract_method = extract_article_body(real_url)

        if not body_text or len(body_text.strip()) < 30:
            body_text = article_info.get("본문요약", "")

        if not body_text or len(body_text.strip()) < 30:
            body_text = "본문 추출 실패. 원문 링크를 확인하세요."

        final_report_data.append({
            "카테고리": json_key,
            "카테고리명": JSON_KEY_TO_DISPLAY.get(json_key, json_key),
            "검색어": article_info.get("검색어", ""),
            "기사제목": article_info.get("기사제목", ""),
            "언론사": article_info.get("언론사", "") or guess_press_name_from_url(real_url),
            "게시일": article_info.get("게시일", ""),
            "본문요약": article_info.get("본문요약", ""),
            "본문전문": body_text.strip(),
            "본문추출방식": extract_method,
            "링크": real_url,
        })

        print(
            f"  └ 📥 본문 추출 완료: "
            f"{article_info.get('기사제목', '')[:28]}... "
            f"({extract_method})"
        )

        time.sleep(random.uniform(0.2, 0.5))

if not final_report_data:
    print("❌ 최종 보고서에 사용할 기사 데이터가 없습니다.")
    exit()


# ==========================================
# 10. 팩트 기반 문체 모방 최종 요약 생성
# ==========================================

print("\n✍️ [STEP 4] Gemini AI가 과거 양식을 학습하여 '찐 팩트 요약 브리핑'을 생성합니다...")

final_input_text = ""

for item in final_report_data:
    final_input_text += (
        f"[{item['카테고리명']}]\n"
        f"제목: {item['기사제목']}\n"
        f"언론사: {item['언론사']}\n"
        f"게시일: {item['게시일']}\n"
        f"링크: {item['링크']}\n"
        f"본문:\n{item['본문전문'][:1800]}\n\n"
    )

prompt_report = f"""
너는 최고 경영진에게 매일 아침 뉴스 브리핑을 제공하는 수석 전략가야.
아래 제공된 [과거 보고서 데이터]를 읽고, 출력 양식과 문장 스타일을 최대한 모방해서 [오늘 기사 데이터]에 대한 요약 보고서를 작성해줘.

[작성 및 요약 규칙 - 절대 엄수]
1. 네 생각, 인사이트, 미래 파장, 대응 포인트 같은 분석은 넣지마.
2. 오직 기사 본문에 입각한 객관적 팩트만 요약해.
3. 각 기사의 요약문은 다른 기호 없이 딱 1개의 문단으로 작성해.
4. 모든 문장의 끝은 반드시 문어체 종결어미인 '~함', '~임', '~됨', '~함.' 계열로 끝내.
5. 카테고리 순서와 이모지 양식은 아래 [출력 양식 예시]와 똑같이 맞춰.
6. 언론사 이름은 [오늘 기사 데이터]에 제공된 값을 우선 사용해.
7. 본문 추출 실패라고 적힌 기사는 링크와 제목, 제공된 요약 범위 안에서만 작성해.
8. 기사에 없는 내용, 추정, 전망, 평가, 의미 부여는 절대 쓰지마.

[출력 양식 예시]

☑️ 자사 및 계열사 이슈

1️⃣ 기사 제목
기사 링크 주소
(언론사)
팩트 중심의 깔끔한 요약 내용 1문단.

☑️ 정부/국회

1️⃣ 기사 제목
기사 링크 주소
(언론사)
팩트 중심의 깔끔한 요약 내용 1문단.

☑️ 경쟁사/해외이슈

1️⃣ 기사 제목
기사 링크 주소
(언론사)
팩트 중심의 깔끔한 요약 내용 1문단.

☑️ 산업동향

1️⃣ 기사 제목
기사 링크 주소
(언론사)
팩트 중심의 깔끔한 요약 내용 1문단.

[과거 보고서 데이터]
{past_reports_content[:3000]}

[오늘 기사 데이터]
{final_input_text}
"""

try:
    final_briefing_text = gemini_generate_text(
        client=client,
        prompt=prompt_report,
        task_name="최종 브리핑 생성",
    )

    print("  └ ✅ 최종 브리핑 생성 완료!")

except Exception as e:
    print(f"  └ ⚠️ Gemini 최종 브리핑 생성 실패. 기본 보고서로 대체합니다. 원인: {e}")
    final_briefing_text = build_fallback_briefing(final_report_data)


# ==========================================
# 11. 결과물 저장 및 최종 출력
# ==========================================

print("\n" + "=" * 60)
print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
print("=" * 60)
print(final_briefing_text)

with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
    f.write(final_briefing_text)

df_backup = pd.DataFrame(final_report_data)
df_backup.to_csv(
    OUTPUT_SELECTED_CSV,
    index=False,
    encoding="utf-8-sig",
)

total_duration = time.time() - total_start_time

print("\n" + "=" * 60)
print(f"💾 시스템 자동화 작업 완료! (총 소요 시간: {total_duration / 60:.2f}분)")
print(f"- '{os.path.basename(OUTPUT_TXT)}' 저장 완료")
print(f"- '{os.path.basename(OUTPUT_SELECTED_CSV)}' 저장 완료")
print(f"- '{os.path.basename(OUTPUT_CANDIDATES_CSV)}' 저장 완료")
print("=" * 60)