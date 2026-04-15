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
# 1. API 키 금고에서 꺼내오기
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "secret.txt"), "r") as f:
        GOOGLE_API_KEY = f.read().strip()
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
except Exception as e:
    print("❌ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다.")
    exit()

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
# 3. 날짜 계산 (오늘 검색용 & 어제 파일 찾기용)
# ==========================================
def get_search_dates(now: datetime):
    end_date = now.replace(hour=23, minute=59, second=59)
    if now.weekday() == 0: # 월요일
        start_date = (now - timedelta(days=3)).replace(hour=0, minute=0, second=0)
        prev_file_date = (now - timedelta(days=3)).strftime("%Y-%m-%d") # 지난주 금요일 파일
    else: # 평일
        start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        prev_file_date = (now - timedelta(days=1)).strftime("%Y-%m-%d") # 어제 파일
        
    period_label = f"{start_date.strftime('%m.%d')} ~ {now.strftime('%m.%d')}"
    return start_date, end_date, period_label, prev_file_date

# ==========================================
# 4. 네이버 뉴스 API 검색 및 스마트 필터링 + AI 요약
# ==========================================
# 💡 prev_info (어제 수집된 정보) 파라미터가 추가되었습니다!
def fetch_naver_news_and_summarize(agency, keyword, start_date, end_date, prev_info):
    search_query = f'"[{keyword}] {agency}"'
    url = f"https://openapi.naver.com/v1/search/news.json?query={search_query}&display=5&sort=date"
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    print(f"🔍 [{agency}] {keyword} 검색 중...", end="")
    try:
        res = requests.get(url, headers=headers, timeout=10)
        res.raise_for_status()
        data = res.json()
        
        snippets = []
        for item in data.get('items', []):
            pub_date_str = item['pubDate']
            try:
                pub_date_obj = datetime.strptime(pub_date_str[:-6], "%a, %d %b %Y %H:%M:%S")
            except:
                continue
                
            if start_date <= pub_date_obj <= end_date:
                title = BeautifulSoup(item['title'], "html.parser").get_text()
                
                if f"[{keyword}]" in title or f"<{keyword}>" in title or f"◆ {keyword}" in title or f"■ {keyword}" in title:
                    desc = BeautifulSoup(item['description'], "html.parser").get_text()
                    snippets.append(f"[제목: {title}] 내용: {desc}")
                
        if not snippets:
            print(f" ➔ 관련 기사 없음 (패스 ⚡)")
            return "해당 없음"
            
        print(f" ➔ 찐 기사 발견! (AI 분석 중 🧠)")
        combined_text = "\n".join(snippets)
        
        # 💡 [핵심 최적화] 제미나이에게 "어제 본 사람은 빼고 알려줘!"라고 강력하게 지시합니다.
        prompt = f"""
        다음은 네이버 뉴스에서 '{agency}'의 '{keyword}'와 관련된 찐 기사 모음이야.
        이 내용 중에서 실제 인사 이동이나 부고 내역을 추출해서 아래 형식으로만 대답해줘. 

        [🔥중요: 중복 제거 지시사항]
        아래는 어제 이미 보고된 내역이야.
        <어제 내역 시작>
        {prev_info}
        <어제 내역 끝>
        
        기사 내용에 위 '어제 내역'과 겹치는 사람이나 직책이 있다면 **오늘 결과에서 무조건 제외**해. 오직 "새롭게 추가된 사람"만 추출해야 해.
        중복된 사람을 제외하고 났을 때 남은 사람이 한 명도 없거나, 애초에 관련 없는 기사라면 오직 '해당 없음'이라고만 대답해. (설명 추가 절대 금지)

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
        
        time.sleep(15) 
        
        if not result or "해당 없음" in result:
            return "해당 없음"
            
        return result

    except Exception as e:
        print(f"❌ 검색 에러: {e}")
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
    start_date, end_date, period_label, prev_file_date = get_search_dates(now)
    date_key = now.strftime("%Y-%m-%d")
    
    print(f"📅 네이버 인사/부고 초고속 수집 시작 (기간: {period_label})\n")
    
    # 💡 어제 만들어진 JSON 파일(기존 데이터)을 몰래 열어서 읽어옵니다.
    prev_data = None
    prev_filepath = os.path.join(DATA_DIR, f"{prev_file_date}.json")
    if os.path.exists(prev_filepath):
        print(f"📂 어제 날짜({prev_file_date})의 파일을 찾아 중복 필터를 가동합니다!\n")
        with open(prev_filepath, "r", encoding="utf-8") as f:
            prev_data = json.load(f)
    else:
        print(f"📂 어제 날짜({prev_file_date})의 파일이 없어 필터 없이 수집합니다.\n")
    
    final_data = {
        "date": date_key,
        "period": period_label,
        "인사": {},
        "부고": {}
    }
    
    for agency in AGENCIES:
        # 어제 데이터에서 해당 부처의 기록을 빼옵니다. (없으면 '해당 없음' 처리)
        prev_insa = prev_data["인사"].get(agency, "해당 없음") if prev_data else "해당 없음"
        prev_bugo = prev_data["부고"].get(agency, "해당 없음") if prev_data else "해당 없음"
        
        # 기사 검색할 때 어제 기록(prev_insa, prev_bugo)을 같이 넘겨줍니다!
        final_data["인사"][agency] = fetch_naver_news_and_summarize(agency, "인사", start_date, end_date, prev_insa)
        final_data["부고"][agency] = fetch_naver_news_and_summarize(agency, "부고", start_date, end_date, prev_bugo)
            
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{date_key}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ {date_key} 데이터 저장 완료!")
    push_to_github(date_key)

if __name__ == "__main__":
    main()