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
    "국무조정실·국무총리비서실", "금융위원회", "금융감독원"
]

DATA_DIR = os.path.join(BASE_DIR, "data_personnel")

# ==========================================
# 3. 날짜 계산 
# ==========================================
def get_search_dates(now: datetime):
    end_date = now.replace(hour=23, minute=59, second=59)
    if now.weekday() == 0: 
        start_date = (now - timedelta(days=3)).replace(hour=0, minute=0, second=0)
    else: 
        start_date = (now - timedelta(days=1)).replace(hour=0, minute=0, second=0)
        
    period_label = f"{start_date.strftime('%m.%d')} ~ {now.strftime('%m.%d')}"
    return start_date, end_date, period_label

# ==========================================
# 4. 네이버 뉴스 API 검색 및 기사 본문 싹쓸이 + AI 요약
# ==========================================
def fetch_naver_news_and_summarize(agency, keyword, start_date, end_date, prev_info):
    # 💡 [수정됨] 가운데 점(·)이 있는 부처는 쪼개서 'OR(|)' 조건으로 검색하도록 개선!
    if '·' in agency:
        parts = agency.split('·') # ['국무조정실', '국무총리비서실'] 로 쪼갬
        if keyword == "인사":
            search_query = f'"[{keyword}] {parts[0]}" | "[{keyword}] {parts[1]}"'
        else:
            search_query = f"{keyword} {parts[0]} | {keyword} {parts[1]}"
    else:
        if keyword == "인사":
            search_query = f'"[{keyword}] {agency}"' 
        else:
            search_query = f"{keyword} {agency}"
        
    url = f"https://openapi.naver.com/v1/search/news.json?query={search_query}&display=5&sort=date"
    
    headers = {
        "X-Naver-Client-Id": NAVER_CLIENT_ID,
        "X-Naver-Client-Secret": NAVER_CLIENT_SECRET
    }
    
    # 💡 네이버가 로봇인 줄 알고 막는 것을 방지하기 위해 "나 진짜 사람이야(크롬 브라우저야)!" 하고 속이는 신분증
    web_headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
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
                    naver_link = item.get('link', '')
                    article_body = ""
                    
                    # 💡 핵심 로직: 네이버 뉴스 링크(n.news.naver.com)면 직접 들어가서 본문을 통째로 긁어옵니다!
                    if "n.news.naver.com" in naver_link:
                        try:
                            art_res = requests.get(naver_link, headers=web_headers, timeout=5)
                            art_soup = BeautifulSoup(art_res.text, "html.parser")
                            # 네이버 뉴스 본문이 담긴 그릇(dic_area) 찾기
                            body_tag = art_soup.find("article", id="dic_area")
                            if body_tag:
                                article_body = body_tag.get_text(separator="\n", strip=True)
                        except:
                            pass
                    
                    # 본문 긁어오기에 실패했거나 네이버 뉴스가 아니면 아쉬운 대로 미리보기(desc) 사용
                    if not article_body:
                        article_body = BeautifulSoup(item['description'], "html.parser").get_text()
                    
                    # 제미나이가 읽기 좋게 제목과 본문(최대 3000자)을 묶어서 전달
                    snippets.append(f"[제목: {title}]\n내용: {article_body[:3000]}")
                
        if not snippets:
            print(f" ➔ 관련 기사 없음 (패스 ⚡)")
            return "해당 없음"
            
        print(f" ➔ 찐 기사 발견! (본문 확보 완료, AI 분석 중 🧠)")
        combined_text = "\n\n---\n\n".join(snippets)
        
        prompt = f"""
        다음은 네이버 뉴스에서 '{agency}'의 '{keyword}'와 관련된 기사 본문 모음이야.
        이 내용 중에서 실제 인사 이동이나 부고 내역을 추출해서 아래 형식으로만 대답해줘. 

        [🔥중요: 중복 제거 지시사항]
        아래는 최근에 이미 보고된 내역이야.
        <기존 내역 시작>
        {prev_info}
        <기존 내역 끝>
        
        기사 내용에 위 '기존 내역'과 겹치는 사람이나 직책이 있다면 **오늘 결과에서 무조건 제외**해. 오직 "새롭게 추가된 사람"만 추출해야 해.
        중복된 사람을 제외하고 났을 때 남은 사람이 한 명도 없거나, 애초에 관련 없는 기사라면 오직 '해당 없음'이라고만 대답해. (설명 추가 절대 금지)

        [출력 형식 예시 - 인사]
        ◇ 국장급 승진
        - 정책기획관 홍길동
        ◇ 과장급 전보
        - 홍보담당관 김철수

        [출력 형식 예시 - 부고]
        ※ 현직뿐만 아니라 '전직(전 직책)' 및 그 '가족(부인상, 부친상 등)'의 부고도 절대 누락하지 말고 모두 포함해. 기사에 고인의 이름이 있다면 줄을 바꿔서 반드시 적어줘.
        김철수(전 행정안전부 주무관) 부친상
        홍길동 씨 별세 = 14일 서울병원, 발인 16일

        기사 내용:
        {combined_text}
        """
        
        response = model.generate_content(prompt)
        result = response.text.strip()
        
        # AI가 예시 제목을 따라 쓰면 강제로 잘라버리는 방어막
        result = result.replace("[출력 형식 예시 - 인사]", "").replace("[출력 형식 예시 - 부고]", "").strip()
        # 💡 [여기 수정!] 안내문이 길어졌으니 지우는 텍스트도 똑같이 맞춰줍니다.
        result = result.replace("※ 현직뿐만 아니라 '전직(전 직책)' 및 그 '가족(부인상, 부친상 등)'의 부고도 절대 누락하지 말고 모두 포함해. 기사에 고인의 이름이 있다면 줄을 바꿔서 반드시 적어줘.", "").strip()
        
        time.sleep(4)
        
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
    start_date, end_date, period_label = get_search_dates(now)
    date_key = now.strftime("%Y-%m-%d")
    
    print(f"📅 네이버 인사/부고 초고속 수집 시작 (기간: {period_label})\n")
    
    # 💡 [새로운 로직] 최근 최대 4일 치의 기존 데이터를 긁어모읍니다!
    print("📂 최근 4일 치 데이터를 찾아 중복 필터를 가동합니다...")
    prev_data_list = []
    
    for i in range(1, 6): # 최근 1일 전부터 5일 전까지 파일이 있는지 탐색
        check_date_str = (now - timedelta(days=i)).strftime("%Y-%m-%d")
        filepath = os.path.join(DATA_DIR, f"{check_date_str}.json")
        if os.path.exists(filepath):
            with open(filepath, "r", encoding="utf-8") as f:
                prev_data_list.append(json.load(f))
            print(f"   ✔️ {check_date_str} 데이터 확보 완료")
        
        # 💡 여기 숫자를 2에서 4로 바꿔주세요! (최대 나흘 치 기억력 장착)
        if len(prev_data_list) == 4: 
            break

    if not prev_data_list:
        print("   (참고) 이전 데이터가 없어 필터 없이 수집합니다.\n")
    else:
        print()
    
    final_data = {
        "date": date_key,
        "period": period_label,
        "인사": {},
        "부고": {}
    }
    
    for agency in AGENCIES:
        # 이틀 치 데이터를 하나의 거대한 텍스트로 합칩니다.
        prev_insa = ""
        prev_bugo = ""
        
        for p_data in prev_data_list:
            insa_text = p_data["인사"].get(agency, "해당 없음")
            if insa_text != "해당 없음":
                prev_insa += insa_text + "\n"
                
            bugo_text = p_data["부고"].get(agency, "해당 없음")
            if bugo_text != "해당 없음":
                prev_bugo += bugo_text + "\n"
                
        if not prev_insa.strip(): prev_insa = "해당 없음"
        if not prev_bugo.strip(): prev_bugo = "해당 없음"
        
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