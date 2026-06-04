import requests
import urllib.parse
import time
import random
import re
import json
import os
import pandas as pd
import feedparser
from google import genai
from datetime import datetime
from newspaper import Article
from bs4 import BeautifulSoup

# ==========================================
# 1. Gemini AI 초기 세팅
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    with open(os.path.join(BASE_DIR, "secret.txt"), "r") as f:
        GOOGLE_API_KEY = f.read().strip()
    client = genai.Client(api_key=GOOGLE_API_KEY)
except Exception as e:
    print("❌ secret.txt 파일이 없거나 구글 API 키를 읽을 수 없습니다.")
    exit()

# [Few-Shot] 과거 보고서 학습 데이터 로드
past_reports_content = ""
if os.path.exists(os.path.join(BASE_DIR, "past_reports.txt")):
    with open(os.path.join(BASE_DIR, "past_reports.txt"), "r", encoding="utf-8") as f:
        past_reports_content = f.read().strip()
    print("📚 'past_reports.txt' 로드 완료! AI가 유저님의 과거 스타일과 중복 기사를 학습합니다.")
else:
    print("📝 'past_reports.txt' 파일이 없습니다. 새로운 데이터로만 진행합니다.")

web_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9"
}

# ==========================================
# 2. 유저 확정 고정 키워드 리스트
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

all_keywords = [k for k_list in keyword_categories.values() for k in k_list]
raw_articles = []
article_id = 1
total_start_time = time.time()

print("\n🚀 [STEP 1] 구글 RSS 무적 엔진으로 초고속 싹쓸이 시작 (최대 100개 스캔)...")

# ==========================================
# [STEP 1] 구글 RSS 피드 파싱 (안정성 100%)
# ==========================================
for idx, keyword in enumerate(all_keywords, 1):
    encoded_keyword = urllib.parse.quote(keyword)
    rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}%20when:1d&hl=ko&gl=KR&ceid=KR:ko"
    
    try:
        res = requests.get(rss_url, headers=web_headers, timeout=10)
        feed = feedparser.parse(res.text)
    except:
        feed = feedparser.parse("")
        
    collected = 0
    for entry in feed.entries[:100]:
        publisher = entry.source.get('title', '알 수 없음') if 'source' in entry else '알 수 없음'
        if not any(a['링크'] == entry.link for a in raw_articles):
            raw_articles.append({
                "id": article_id,
                "검색어": keyword,
                "언론사": publisher,
                "기사제목": entry.title,
                "요약본": entry.summary,
                "링크": entry.link
            })
            article_id += 1
            collected += 1
            
    print(f"▶ [{idx}/{len(all_keywords)}] '{keyword}' 수집 ➔ {collected}개 후보 확보        ", end="\r")
    time.sleep(random.uniform(0.05, 0.1))

print(f"\n\n  └ ✅ 총 {len(raw_articles)}개의 1차 뉴스 후보군 풀(Pool) 확보 완료!")
if not raw_articles:
    print("❌ 수집된 기사가 없습니다. 네트워크 상태를 확인하세요.")
    exit()

# ==========================================
# [STEP 2] 무료 API 보호용 후보군 1차 압축 & AI 선별
# ==========================================
print("\n🧠 [STEP 2] Gemini AI가 과거 보고서를 학습하여 중복을 거르고 최정예를 선별합니다...")

compressed_articles = {}
for a in raw_articles:
    kw = a['검색어']
    if kw not in compressed_articles:
        compressed_articles[kw] = []
    if len(compressed_articles[kw]) < 10: 
        compressed_articles[kw].append(a)

final_candidates = [item for sublist in compressed_articles.values() for item in sublist]

candidate_text = ""
for a in final_candidates:
    candidate_text += f"[{a['id']}] {a['검색어']} / {a['언론사']} / {a['기사제목']}\n"

prompt_selection = f"""
너는 IT 대기업의 유능한 최고 비서실장이야. 
아래 제공하는 [과거 보고서 데이터]를 정독하고, 유저가 어떤 무게감의 기사를 선별했는지 그 기준을 완벽히 학습해.
그 후, 오늘 수집된 [오늘 뉴스 후보 리스트] 중에서 딱 13~15개의 최정예 기사만 골라내줘.

[기사 선별 가이드라인]
1. 오피니언, 사설, 칼럼, 전문가 기고는 무조건 제외해. 객관적 사실만을 보도한 기사만 선택.
2. 서비스 출시 등 단순 이벤트 홍보 기사(찌라시)는 철저히 배제해.
3. 카카오 및 계열사는 과징금, 경영진 이슈, 서비스 장애 등 무겁고 중요한 내용 위주로 선별해.
4. 타사 이름만 들어간 무관한 기사는 걸러내.
5. (일주일 치 중복 제거 규칙): [과거 보고서 데이터]를 확인하여, 이미 일주일 이내에 다루었던 사건이나 완전히 동일한 내용의 이슈는 무조건 제외해! 
6. 동일 사건에 대해 여러 기사가 있다면 딱 1개만 대표로 선택해. 메이저 매체를 우대해.
7. 카테고리별 할당량: 자사 및 계열사(3~4개) / 정부·국회(4~5개) / 경쟁사·해외이슈(3~4개) / 산업동향(1~2개)

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

for attempt in range(3):
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_selection
        )
        json_data = json.loads(re.search(r'\{.*\}', response.text, re.DOTALL).group())
        break
    except Exception as e:
        print(f"  └ ⚠️ AI 서버 혼잡 (재시도 {attempt+1}/3)... 5초 대기 후 다시 요청합니다.")
        time.sleep(5)
        if attempt == 2:
            json_data = {"자사_및_계열사_이슈": [a['id'] for a in final_candidates[:4]]}

selected_count = sum(len(ids) for ids in json_data.values())
print(f"  └ ✅ AI 선별 완료! 가이드라인을 통과한 총 {selected_count}개의 핵심 기사 확보.")

# ==========================================
# [STEP 3] 💡 구글 링크 해독기 + newspaper3k + BeautifulSoup 3중 콤보 본문 추출
# ==========================================
print("\n🕵️‍♂️ [STEP 3] 선별된 기사들의 '진짜 본문 전문'을 추출합니다...")
final_report_data = []

# 구글 동의 방어막 우회용 쿠키
google_cookies = {'CONSENT': 'YES+cb.20230501-14-p0.ko+FX+478'}

for cat, ids in json_data.items():
    for art_id in ids:
        article_info = next((a for a in raw_articles if a['id'] == art_id), None)
        if not article_info:
            continue
            
        google_url = article_info['링크']
        real_url = google_url
        news_article = ""
        
        # 💡 1. [암호 해독] 구글 우회 링크를 타고 들어가 진짜 언론사 주소 알아내기
        try:
            track_res = requests.get(google_url, headers=web_headers, cookies=google_cookies, allow_redirects=True, timeout=5)
            # 자바스크립트로 숨겨둔 진짜 링크 찾기
            soup_track = BeautifulSoup(track_res.text, "html.parser")
            a_tag = soup_track.select_one('noscript a')
            if a_tag and a_tag.has_attr('href'):
                real_url = a_tag['href']
            else:
                real_url = track_res.url
        except:
            pass

        # 💡 2. [본문 추출] 해독된 진짜 주소로 newspaper3k 가동
        try:
            article = Article(url=real_url, language='ko')
            article.download()
            article.parse()    
            news_article = article.text
        except:
            pass
            
        # 💡 3. [보조 추출] newspaper3k가 막혔다면 BeautifulSoup으로 <p> 태그 싹쓸이
        if not news_article or len(news_article.strip()) < 40:
            try:
                art_res = requests.get(real_url, headers=web_headers, timeout=5)
                art_soup = BeautifulSoup(art_res.text, "html.parser")
                paragraphs = art_soup.find_all('p')
                news_article = "\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
            except:
                pass

        # 💡 4. [최후의 보루] 다 실패하면 구글 요약본 태그 청소해서 삽입
        if not news_article or len(news_article.strip()) < 40:
            news_article = BeautifulSoup(article_info['요약본'], "html.parser").get_text(separator=" ", strip=True)

        final_report_data.append({
            "카테고리": cat,
            "언론사": article_info['언론사'],
            "검색어": article_info['검색어'],
            "기사제목": article_info['기사제목'],
            "본문전문": news_article.strip(),
            "링크": real_url # 보고서에도 이제 진짜 주소가 들어갑니다!
        })
        print(f"  └ 📥 본문 추출 완료: [{article_info['언론사']}] {article_info['기사제목'][:20]}...")
        time.sleep(0.2)

# ==========================================
# [STEP 4] 팩트 기반 문체 모방 최종 요약 생성
# ==========================================
print("\n✍️ [STEP 4] Gemini AI가 과거 양식을 학습하여 '찐 팩트 요약 브리핑'을 생성합니다...")

final_input_text = ""
for f in final_report_data:
    final_input_text += f"[{f['카테고리']}] 언론사: {f['언론사']}\n제목: {f['기사제목']}\n본문:\n{f['본문전문'][:1500]}\n링크: {f['링크']}\n\n"

prompt_report = f"""
너는 최고 경영진에게 매일 아침 뉴스 브리핑을 제공하는 수석 전략가야.
아래 제공된 [과거 보고서 데이터]를 읽고, 출력 양식과 문장 스타일을 완벽하게 모방해서 [오늘 기사 데이터]에 대한 요약 보고서를 작성해줘.

[작성 및 요약 규칙 - 절대 엄수]
1. 네 생각, 인사이트, 미래 파장, 대응 포인트 같은 '분석'은 1%도 넣지마. 오직 본문에 입각한 객관적 팩트만 요약해.
2. 각 기사의 요약문은 다른 기호 없이 딱 '1개의 문단'으로만 깔끔하게 작성해.
3. 모든 문장의 끝은 반드시 문어체 종결어미인 '~함', '~임', '~됨', '~함.'으로만 끝나야 해.
4. 카테고리 순서와 이모지 양식은 반드시 아래 [출력 양식 예시]와 똑같이 맞춰서 출력해줘.

[출력 양식 예시]
☑️ 자사 및 계열사 이슈

1️⃣ 기사 제목
기사 링크 주소
(언론사)
~함/임으로 끝나는 팩트 중심의 깔끔한 요약 내용 1문단.

[과거 보고서 데이터 (양식 및 문체 모방용 교과서)]
{past_reports_content[:3000]}

[오늘 기사 데이터]
{final_input_text}
"""

for attempt in range(3):
    try:
        final_briefing = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt_report
        )
        break
    except Exception as e:
        print(f"  └ ⚠️ AI 서버 혼잡 (재시도 {attempt+1}/3)... 5초 대기 후 다시 요청합니다.")
        time.sleep(5)
        if attempt == 2:
            final_briefing = type("obj", (object,), {"text": "서버 과부하로 최종 브리핑 생성 실패."})()

# ==========================================
# 5. 결과물 저장 및 최종 출력
# ==========================================
print("\n" + "="*60)
print("✨ [오늘 아침 최고경영자(CEO) 뉴스 브리핑 최종 보고서] ✨")
print("="*60)
print(final_briefing.text)

with open("CEO_Morning_Briefing.txt", "w", encoding="utf-8") as f:
    f.write(final_briefing.text)

df_backup = pd.DataFrame(final_report_data)
df_backup.to_csv("google_news_top15_raw.csv", index=False, encoding="utf-8-sig")

total_duration = time.time() - total_start_time
print("\n" + "="*60)
print(f"💾 시스템 자동화 작업 완료! (총 소요 시간: {total_duration/60:.2f}분)")
print("- 'CEO_Morning_Briefing.txt' 저장 완료")
print("- 'google_news_top15_raw.csv' 저장 완료")
print("="*60)