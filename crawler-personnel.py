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
# 1. 기본 설정 및 API 키
# ==========================================
genai.configure(api_key="AIzaSyCbKPNaFCLUkS26x0_JUZBC2_JqgthD9wU")
# 💡 요청하신 gemini-2.5-flash 모델 적용 완료
model = genai.GenerativeModel('gemini-2.5-flash')

AGENCIES = [
    "과학기술정보통신부", "방송미디어통신위원회", "개인정보보호위원회", 
    "공정거래위원회", "행정안전부", "산업통상부", "문화체육관광부", 
    "국무조정실", "국무총리비서실", "금융위원회", "금융감독원"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_personnel")

# ==========================================
# 2. 날짜 계산 (월요일은 3일치, 평일은 1일치)
# ==========================================
def get_search_dates(now: datetime):
    if now.weekday() == 0: # 월요일이면
        start_date = now - timedelta(days=3)
    else: # 평일이면
        start_date = now - timedelta(days=1)
        
    sd_str = start_date.strftime("%Y%m%d") + "000000"
    ed_str = now.strftime("%Y%m%d") + "235959"
    period_label = f"{start_date.strftime('%m.%d')} ~ {now.strftime('%m.%d')}"
    return sd_str, ed_str, period_label

# ==========================================
# 3. 다음 뉴스 검색 및 AI 요약
# ==========================================
def fetch_daum_news_and_summarize(agency, keyword, sd_str, ed_str):
    search_query = f"{agency} {keyword}"
    url = f"https://search.daum.net/search?w=news&sort=rec&q={search_query}&period=u&sd={sd_str}&ed={ed_str}"
    
    print(f"🔍 [{agency}] {keyword} 기사 찾는 중...")
    try:
        res = requests.get(url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(res.text, "html.parser")
        
        snippets = [p.get_text(strip=True) for p in soup.select("p.conts_desc")]
            
        if not snippets:
            return "해당 없음"
            
        combined_text = "\n".join(snippets)
        
        # 💡 해당 내용이 없으면 '해당 없음'으로 답하도록 프롬프트 수정
        prompt = f"""
        다음은 '{agency}'의 '{keyword}'와 관련된 최근 뉴스 기사 요약본들이야.
        이 내용 중에서 실제 인사 이동이나 부고 내역을 추출해서 아래 형식으로만 대답해줘. 
        해당 내용이 없거나 전혀 관련 없는 동명이인 등의 기사면 오직 '해당 없음'이라고만 대답해. (설명 추가 절대 금지)

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
        
        time.sleep(3) # API 무료 한도(분당 15회) 보호용 3초 대기
        
        if not result or "해당 없음" in result:
            return "해당 없음"
            
        return result

    except Exception as e:
        print(f"❌ 검색 에러 ({agency}): {e}")
        return "해당 없음"

# ==========================================
# 4. 깃허브 배달
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
    sd_str, ed_str, period_label = get_search_dates(now)
    date_key = now.strftime("%Y-%m-%d")
    
    print(f"📅 인사/부고 수집 시작 (기간: {period_label})")
    
    final_data = {
        "date": date_key,
        "period": period_label,
        "인사": {},
        "부고": {}
    }
    
    for agency in AGENCIES:
        final_data["인사"][agency] = fetch_daum_news_and_summarize(agency, "인사", sd_str, ed_str)
        final_data["부고"][agency] = fetch_daum_news_and_summarize(agency, "부고", sd_str, ed_str)
            
    os.makedirs(DATA_DIR, exist_ok=True)
    filepath = os.path.join(DATA_DIR, f"{date_key}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)
        
    print(f"\n✅ {date_key} 데이터 저장 완료!")
    push_to_github(date_key)

if __name__ == "__main__":
    main()