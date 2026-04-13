import requests
from bs4 import BeautifulSoup
import json
import os
import re
import subprocess
from datetime import datetime, timedelta  # 💡 [수정됨] timedelta 추가
import urllib3

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

# 💡 인포맥스 쿠키 (만료 시 이 부분만 교체하세요)
MY_COOKIE = "__utma=124704158.256945133.1773035052.1775713264.1775715334.17; __utmz=124704158.1773035052.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); _ga=GA1.1.256945133.1773035052; _ga_DV8PW0Y6Y0=GS2.1.s1775715333$o20$g1$t1775715334$j59$l0$h0; _gid=GA1.3.825370968.1775633534; __utmc=124704158"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/26.3 Safari/605.1.15",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9",
    "Connection": "keep-alive",
    "Referer": "https://news.einfomax.co.kr/",
    "Cookie": MY_COOKIE
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

ECO_SEARCH_URL = "https://news.einfomax.co.kr/news/articleList.html?sc_area=A&view_type=sm&sc_word=%5B%EC%9D%B4%EB%B2%88%EC%A3%BC+%EC%9E%AC%EC%A0%95%EA%B2%BD%EC%A0%9C%EB%B6%80+%EB%93%B1+%EA%B2%BD%EC%A0%9C%EB%B6%80%EC%B2%98+%EC%9D%BC%EC%A0%95%5D"
BOK_SEARCH_URL = "https://news.einfomax.co.kr/news/articleList.html?sc_area=A&view_type=sm&sc_word=%5B%EC%9D%B4%EB%B2%88%EC%A3%BC+%ED%95%9C%EA%B5%AD%EC%9D%80%ED%96%89+%EB%B0%8F+%EA%B8%88%EC%9C%B5%EC%9C%84%C2%B7%EA%B8%88%EA%B0%90%EC%9B%90+%EC%9D%BC%EC%A0%95%5D"

# ==========================================
# 2. 핵심 함수 모음
# ==========================================
def get_week_key(date: datetime) -> str:
    month = f"{date.month:02d}"
    first_day = date.replace(day=1)
    adjusted_day = date.day + first_day.weekday()
    week_of_month = (adjusted_day - 1) // 7 + 1
    return f"{date.year}-{month}-W{week_of_month:02d}"

def get_latest_article_from_search(search_url: str) -> str:
    try:
        res = requests.get(search_url, headers=HEADERS, timeout=10)
        res.raise_for_status()
        res.encoding = 'utf-8'
        soup = BeautifulSoup(res.text, "html.parser")
        for a in soup.find_all("a", href=True):
            title = a.get_text(strip=True)
            href = a["href"]
            if "articleView.html?idxno=" in href and "이번주" in title:
                if href.startswith("/"): return f"{BASE_URL}{href}"
                elif href.startswith("http"): return href
                else: return f"{BASE_URL}/news/{href}"
    except: return None
    return None

def fetch_article_content(url: str) -> list:
    res = requests.get(url, headers=HEADERS, timeout=10)
    res.raise_for_status()
    soup = BeautifulSoup(res.text, "html.parser")
    article = soup.find("article", id="article-view-content-div")
    return [p.get_text(strip=True) for p in article.find_all("p") if p.get_text(strip=True)] if article else []

def parse_agencies(lines: list, agency_names: list) -> dict:
    result = {}
    current_agency, current_date = None, None
    for line in lines:
        for name in agency_names:
            if f"[{name}]" in line:
                current_agency, current_date = name, None
                if name not in result: result[name] = {}
                break
        if current_agency is None: continue
        date_match = re.match(r"^\*(.+일\([월화수목금토일]\))", line)
        if date_match:
            current_date = date_match.group(1).strip()
            if current_date not in result[current_agency]: result[current_agency][current_date] = []
            continue
        if current_date and (line.startswith("▲") or line.startswith("※")):
            result[current_agency][current_date].append(line)
    return result

# 💡 [수정됨] 행안부 19일 싹쓸이 함수
def fetch_mois_schedule(now: datetime) -> dict:
    weekdays = ["월", "화", "수", "목", "금", "토", "일"]
    
    start_of_week = now - timedelta(days=now.weekday()) # 이번 주 월요일
    end_of_week = start_of_week + timedelta(days=6)     # 이번 주 일요일
    
    # 💡 일~토 달력 문제를 해결하기 위해 월요일, 일요일 두 번 찌릅니다
    target_dates = [start_of_week, end_of_week]
    result = {"행정안전부": {}}
    
    for target in target_dates:
        mois_url = f"https://www.mois.go.kr/mns/a03/selectGpScheduleCalendar.do?cat=90010001&year={target.year}&month={target.month}&day={target.day}"
        print(f"\n📰 행안부 홈페이지 찌르는 중... ({target.month}월 {target.day}일 기준)")
        
        try:
            res = requests.get(mois_url, headers=HEADERS, timeout=10, verify=False)
            soup = BeautifulSoup(res.text, "html.parser")
            
            for li in soup.find_all("li"):
                text = li.get_text(separator=" ", strip=True)
                
                # 💡 "4.19(일) 10:00" 등 모든 형태를 잡아내는 강력한 탐지기
                match = re.search(r"(\d+)\s*\.\s*(\d+)[^\d]*(.*)", text)
                
                if match:
                    month_str, day_str, content = match.groups()
                    
                    try:
                        event_date = datetime(target.year, int(month_str), int(day_str))
                        
                        # 이번 주 월~일 사이의 일정만 쏙쏙 담기
                        if start_of_week.date() <= event_date.date() <= end_of_week.date():
                            date_key = f"{int(month_str)}월 {int(day_str)}일({weekdays[event_date.weekday()]})"
                            
                            if date_key not in result["행정안전부"]:
                                result["행정안전부"][date_key] = []
                            
                            clean_content = content.strip()
                            if not clean_content.startswith("▲") and not clean_content.startswith("※"):
                                if re.match(r"^\d{2}:\d{2}", clean_content):
                                    clean_content = f"▲{clean_content}"
                                else:
                                    clean_content = f"▲ {clean_content}"
                                    
                            if clean_content not in result["행정안전부"][date_key]:
                                result["행정안전부"][date_key].append(clean_content)
                    except ValueError:
                        continue
        except Exception as e:
            print(f"❌ 행정안전부 수집 실패 ({target.day}일 기준): {e}")
            
    return result

# 💡 [수정됨] 에러 안 뱉는 똑똑한 배달 함수
def push_to_github(week_key):
    try:
        print("\n🚀 깃허브로 배달을 시작합니다...")
        
        # 💡 변경된 파일이 있는지 싹 검사
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
        if not status.stdout.strip():
            print("✨ 새로 추가되거나 변경된 일정이 없습니다. (이전과 100% 동일함)")
            return

        subprocess.run(["git", "add", "."], check=True)
        commit_msg = f"Update schedule: {week_key} ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ 배달 완료! 웹사이트에서 새 일정을 확인하세요.")
    except Exception as e:
        print(f"❌ 배달 실패: {e}")

# ==========================================
# 3. 메인 실행 블록
# ==========================================
def main():
    now = datetime.now()
    week_key = get_week_key(now)
    date_str = now.strftime("%Y-%m-%d")

    print(f"📅 수집 시작: {week_key}")

    all_agencies = {}
    
    # 인포맥스 수집
    for category, url in [("경제부처", ECO_SEARCH_URL), ("한국은행", BOK_SEARCH_URL)]:
        article_url = get_latest_article_from_search(url)
        if article_url:
            lines = fetch_article_content(article_url)
            parsed = parse_agencies(lines, list(TARGET_AGENCIES[category].keys()))
            all_agencies.update(parsed)

    # 행안부 수집
    all_agencies.update(fetch_mois_schedule(now))

    if all_agencies:
        os.makedirs(DATA_DIR, exist_ok=True)
        filepath = os.path.join(DATA_DIR, f"{week_key}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({"week": week_key, "date": date_str, "agencies": all_agencies}, f, ensure_ascii=False, indent=2)
        print(f"✅ 데이터 저장 완료 ({len(all_agencies)}개 기관)")
        
        # 수집 종료 후 바로 깃허브 자동 업로드
        push_to_github(week_key)
    else:
        print("❌ 수집된 데이터가 없습니다.")

if __name__ == "__main__":
    main()