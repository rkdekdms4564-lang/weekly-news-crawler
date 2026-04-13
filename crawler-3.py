import requests
from bs4 import BeautifulSoup
import json
import os
import re
from datetime import datetime
import urllib3  # 💡 [추가됨] 행안부 사이트 접속을 위한 라이브러리

# 💡 [추가됨] 정부 사이트 SSL 인증서 경고 숨기기
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. 기본 설정 및 변수
# ==========================================
TARGET_AGENCIES = {
    "경제부처": {
        "과학기술정보통신부": "과학기술정보통신부",
        "방송미디어통신위원회": "방송미디어통신위원회",
    },
    "한국은행": {
        "금융위원회": "금융위원회",
        "금융감독원": "금융감독원",
    },
}

BASE_URL = "https://news.einfomax.co.kr"

# 💡 여기에 복사하신 쿠키를 붙여넣으세요! 💡
MY_COOKIE = "__utma=124704158.256945133.1773035052.1775713264.1775715334.17; __utmz=124704158.1773035052.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); _ga=GA1.1.256945133.1773035052; _ga_DV8PW0Y6Y0=GS2.1.s1775715333$o20$g1$t1775715334$j59$l0$h0; _gid=GA1.3.825370968.1775633534; __utmc=124704158"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.3 Safari/605.1.15",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://news.einfomax.co.kr/",
    "Cookie": MY_COOKIE  # 내 진짜 브라우저 신분증 장착!
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# 검색 결과 URL 고정
ECO_SEARCH_URL = "https://news.einfomax.co.kr/news/articleList.html?sc_area=A&view_type=sm&sc_word=%5B%EC%9D%B4%EB%B2%88%EC%A3%BC+%EC%9E%AC%EC%A0%95%EA%B2%BD%EC%A0%9C%EB%B6%80+%EB%93%B1+%EA%B2%BD%EC%A0%9C%EB%B6%80%EC%B2%98+%EC%9D%BC%EC%A0%95%5D"
BOK_SEARCH_URL = "https://news.einfomax.co.kr/news/articleList.html?sc_area=A&view_type=sm&sc_word=%5B%EC%9D%B4%EB%B2%88%EC%A3%BC+%ED%95%9C%EA%B5%AD%EC%9D%80%ED%96%89+%EB%B0%8F+%EA%B8%88%EC%9C%B5%EC%9C%84%C2%B7%EA%B8%88%EA%B0%90%EC%9B%90+%EC%9D%BC%EC%A0%95%5D"

# ==========================================
# 2. 핵심 함수 모음
# ==========================================
def get_week_key(date: datetime) -> str:
    """월과 주차를 계산하여 파일명 생성 (예: 2026-04-W02)"""
    month = f"{date.month:02d}" # 월 (예: 04)
    week_of_month = (date.day - 1) // 7 + 1 # 해당 월의 몇 번째 주인지 계산
    return f"{date.year}-{month}-W{week_of_month:02d}"

def get_latest_article_from_search(search_url: str) -> str:
    res = requests.get(search_url, headers=HEADERS, timeout=10)
    res.raise_for_status()
    res.encoding = 'utf-8'
    
    soup = BeautifulSoup(res.text, "html.parser")
    
    for a in soup.find_all("a", href=True):
        title = a.get_text(strip=True)
        href = a["href"]
        
        # 기사 링크(articleView.html)이면서 텍스트에 '이번주'가 포함된 가장 첫 링크만 획득
        if "articleView.html?idxno=" in href and "이번주" in title:
            if href.startswith("/"):
                return f"{BASE_URL}{href}"
            elif href.startswith("http"):
                return href
            else:
                return f"{BASE_URL}/news/{href}"
    return None

def fetch_article_content(url: str) -> list:
    res = requests.get(url, headers=HEADERS, timeout=10)
    res.raise_for_status()
    res.encoding = "utf-8"
    soup = BeautifulSoup(res.text, "html.parser")
    article = soup.find("article", id="article-view-content-div")
    if not article:
        return []
    lines = []
    for p in article.find_all("p"):
        text = p.get_text(strip=True)
        if text:
            lines.append(text)
    return lines

def parse_agencies(lines: list, agency_names: list) -> dict:
    result = {}
    current_agency = None
    current_date = None

    for line in lines:
        for name in agency_names:
            if f"[{name}]" in line:
                current_agency = name
                current_date = None
                if name not in result:
                    result[name] = {}
                break

        if current_agency is None:
            continue

        date_match = re.match(r"^\*(.+일\([월화수목금토일]\))", line)
        if date_match:
            current_date = date_match.group(1).strip()
            if current_date not in result[current_agency]:
                result[current_agency][current_date] = []
            continue

        if current_date and (line.startswith("▲") or line.startswith("※")):
            result[current_agency][current_date].append(line)

    return result

# 💡 [추가됨] 행정안전부 전용 크롤러 함수
def fetch_mois_schedule(now: datetime) -> dict:
    mois_url = f"https://www.mois.go.kr/mns/a03/selectGpScheduleCalendar.do?cat=90010001&year={now.year}&month={now.month}&day={now.day}"
    print(f"\n📰 행정안전부 홈페이지 수집 중... ({mois_url})")
    
    try:
        res = requests.get(mois_url, headers=HEADERS, timeout=10, verify=False)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, "html.parser")
        
        result = {"행정안전부": {}}
        
        for li in soup.find_all("li"):
            text = li.get_text(separator=" ", strip=True)
            match = re.match(r"^(\d+)\.(\d+)\.\s+(\d{2}:\d{2})\s+(.+)", text)
            
            if match:
                month, day, time, event = match.groups()
                date_key = f"{int(month)}월 {int(day)}일"
                
                if date_key not in result["행정안전부"]:
                    result["행정안전부"][date_key] = []
                
                item = f"▲{time} {event}"
                if item not in result["행정안전부"][date_key]:
                    result["행정안전부"][date_key].append(item)

        return result
    except Exception as e:
        print(f"❌ 행정안전부 수집 실패: {e}")
        return {"행정안전부": {}}

def save_data(week_key: str, date_str: str, all_agencies: dict):
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{week_key}.json")
    payload = {
        "week": week_key,
        "date": date_str,
        "agencies": all_agencies,
    }
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n✅ 저장 완료: {filepath}")

# ==========================================
# 3. 메인 실행 블록
# ==========================================
def main():
    now = datetime.now()
    week_key = get_week_key(now)
    date_str = now.strftime("%Y-%m-%d")

    print(f"📅 수집 시작: {week_key} ({date_str})\n")
    print("🔍 내 쿠키(Cookie)를 장착하고 검색 URL 찌르는 중...")
    
    경제부처_URL = get_latest_article_from_search(ECO_SEARCH_URL)
    한국은행_URL = get_latest_article_from_search(BOK_SEARCH_URL)

    all_agencies = {}

    # 1. 인포맥스: 경제부처 수집
    if 경제부처_URL:
        print(f"  ✓ 경제부처 최신 기사 획득: {경제부처_URL}")
        lines1 = fetch_article_content(경제부처_URL)
        parsed1 = parse_agencies(lines1, list(TARGET_AGENCIES["경제부처"].keys()))
        all_agencies.update(parsed1)
        for name, dates in parsed1.items():
            print(f"  ✓ {name}: {len(dates)}일치 일정")
    else:
        print("❌ 경제부처 기사를 찾을 수 없습니다.")

    # 2. 인포맥스: 한국은행 수집
    if 한국은행_URL:
        print(f"  ✓ 한국은행 최신 기사 획득: {한국은행_URL}")
        lines2 = fetch_article_content(한국은행_URL)
        parsed2 = parse_agencies(lines2, list(TARGET_AGENCIES["한국은행"].keys()))
        all_agencies.update(parsed2)
        for name, dates in parsed2.items():
            print(f"  ✓ {name}: {len(dates)}일치 일정")
    else:
        print("❌ 한국은행 기사를 찾을 수 없습니다.")

    # 💡 3. [추가됨] 행정안전부 수집
    mois_data = fetch_mois_schedule(now)
    if "행정안전부" in mois_data and mois_data["행정안전부"]:
        all_agencies.update(mois_data)
        print(f"  ✓ 행정안전부: {len(mois_data['행정안전부'])}일치 일정")

    # 4. 최종 저장
    if not all_agencies:
        print("\n❌ 파싱된 기관이 하나도 없습니다. 수집을 종료합니다.")
        return

    save_data(week_key, date_str, all_agencies)
    print(f"   총 {len(all_agencies)}개 기관 저장 완료\n")

if __name__ == "__main__":
    main()
