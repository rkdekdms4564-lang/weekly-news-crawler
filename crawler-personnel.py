import requests
from bs4 import BeautifulSoup
import json
import os
import time
import subprocess
from datetime import datetime, timedelta
import urllib3
import google.generativeai as genai

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 1. API 키 금고에서 꺼내오기 (절대 코드에 직접 적지 마세요!)
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 🔑 구글 Gemini 키 불러오기
try:
    with open(os.path.join(BASE_DIR, "secret.txt"), "r") as f:
        GOOGLE_API_KEY = f.read().strip()
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash') # 속도/안정성을 위해 1.5-flash 강력 추천!
except Exception as e:
    print("❌ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다.")
    exit()

# 🔑 네이버 API 키 불러오기
try:
    with open(os.path.join(BASE_DIR, "secret_naver.txt"), "r") as f:
        naver_keys = f.read().strip().split('\n')
        NAVER_CLIENT_ID = naver_keys[0].strip()
        NAVER_CLIENT_SECRET = naver_keys[1].strip()
except Exception as e:
    print("❌ secret_naver.txt 파일이 없거나 네이버 API 키를 읽을 수 없습니다.")
    exit()

# ==========================================
# 2. 기본 설정
# ==========================================
AGENCIES = [
    "과학기술정보통신부", "방송미디어통신위원회", "개인정보보호위원회", 
    "공정거래위원회", "행정안전부", "산업통상부", "문화체육관광부", 
    "국무조정실", "국무총리비서실", "금융위원회", "금융감독원"
]

DATA_DIR = os.path.join(BASE_DIR, "data_personnel")

# ==========================================
# 3. 날짜 계산 (월요일은 3일치, 평일은 1일치)
# ==========================================
def get_search_dates(now: datetime):
    # 시간을 00:00:00부터 23:59:59로 깔끔하게 맞춥니다.
    end_date = now.replace(hour=23, minute=59, second=59)
    if now.weekday() == 0: # 월요일이면 지난주 금요일 00시부터
        start_date = (now - timedelta(days=3)).replace(hour=0, minute=0, second=0)
    else: # 평일이면 어제 00시부터
        start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        
    period_label = f"{start_date.strftime('%m.%d')} ~ {now.strftime('%m.%d')}"
    return start_date, end_date, period_label

# ==========================================
# 4. 네이버 뉴스 API 검색 및 AI 요약
# ==========================================
def fetch_naver_news_and_summarize(agency, keyword, start_date, end_date):
    search_query = f"{agency} {keyword}"
    
    # 네이버 뉴스 API 주소 (정확도순이 아닌 최신순(date)으로 최대 50개 호출)
    url = f"https://openapi.naver.com/v1/search/news.json?query={search_query}&display=50&sort=date"
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    print(f"🔍 [{agency}] {keyword} 네이버 기사 찾는 중...")
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        snippets = []
        for item in data.get('items', []):
            # 네이버의 날짜 형식(예: Tue, 14 Apr 2026 15:32:00 +0900)을 파이썬 시간으로 변환
            pub_date_str = item['pubDate']
            try:
                # 맨 뒤의 +0900을 자르고 변환
                pub_date_obj = datetime.strptime(pub_date_str[:-6], "%a, %d %b %Y %H:%M:%S")
            except:
                continue
                
            # 우리가 원하는 수집 기간(1일~3일) 안에 들어오는 기사만 쏙쏙 뽑기
            if start_date <= pub_date_obj <= end_date:
                # <b> 태그 등 HTML 찌꺼기 제거
                title = BeautifulSoup(item['title'], "html.parser").get_text()
                desc = BeautifulSoup(item['description'], "html.parser").get_text()
                snippets.append(f"[제목: {title}] 내용: {desc}")
                
        if not snippets:
            return "해당 없음"
            
        combined_text = "\n".join(snippets)
        
        # 제미나이에게 지시
        prompt = f"""
        다음은 네이버 뉴스에서 '{agency}'의 '{keyword}'와 관련된 최근 기사 검색 결과(제목과 내용)야.
        이 내용 중에서 실제 인사 이동이나 부고 내역을 추출해서 아래 형식으로만 대답해줘. 
        해당 내용이 없거나 전혀 관련 없는 동명이인 기사면 오직 '해당 없음'이라고만 대답해. (설명 추가 절대 금지)

        [출력 형식 예시 - 인사]
        ◇ 국장급 승진
        - 정책기획관 홍길동
        ◇ 과장급 전보
        - 홍보담당관 김철수

        [출력 형식 예시 - 부고]
        김철수(행정안전부 주무관) 부친상 = 14일, 서울병원, 발인 16일

        기사 내용:
        {combined_text}
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        time.sleep(4) # 제미나이 API 제한 보호 (4초 대기)
        
        if not result or "해당 없음" in result:
            return "해당 없음"
            
        return result

    except Exception as e:
        print(f"❌ 검색 에러 ({agency}): {e}")
        return "해당 없음"

# ==========================================
# 5. 깃허브 배달
# ==========================================
def push_to_github(file_name):
    try:
        print("\n🚀 깃허브로 인사/부고 데이터를 배달합니다...")
        subprocess.run(["git", "add", "."], check=True)
        commit_msg = f"Update personnel: {file_name}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
        subprocess.run(["git", "push", "origin", "main"], check=True)
        print("✅ 배달 완료!")
    except Exception as e:
        print("✨ 변경된 내역이 없거나 배달을 건너뜁니다.")

def main():
    now = datetime.now()
    start_date, end_date, period_label = get_search_dates(now)
    date_key = now.strftime("%Y-%m-%d")
    
    print(f"📅 네이버 인사/부고 수집 시작 (기간: {period_label})")
    
    final_data = {
        "date": date_key,
        "period": period_label,
        "인사": {},
        "부고": {}
    }
    
    for agency in AGENCIES:
        final_data["인사"][agency] = fetch_naver_news_and_summarize(agency, "인사", start_date, end_date)
        final_data["부고"][agency] = fetch_naver_news_and_summarize(agency, "부고", start_date, end_date)
            
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{date_key}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ {date_key} 데이터 저장 완료!")
    push_to_github(date_key)

if __name__ == "__main__":
    main()