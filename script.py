import feedparser
import requests
import pandas as pd
import urllib.parse
import os
import time
import random
from bs4 import BeautifulSoup
from datetime import datetime
from email.utils import parsedate_to_datetime

# 맥북 브라우저 신분증 장착
web_headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9"
}

# 우리가 기획한 모든 키워드 리스트
keywords = [
    "카카오", "카카오톡", "정신아", "김범수", "카카오모빌리티", "카카오페이", "카카오뱅크", "카카오엔터테인먼트",
    "카카오게임즈", "카카오픽코마", "카카오헬스케어", "카카오엔터프라이즈",
    "과학기술정보통신부", "방송미디어통신위원회", "공정거래위원회", "금융위원회", "금융감독원", "행정안전부",
    "과학기술정보통신위원회", "정무위원회", "플랫폼 규제", "스테이블코인", "국가AI컴퓨팅센터", "국가AI전략위원회",
    "지도반출", "딥페이크", "AI기본법", "AI 저작권", "중소벤처기업부", "성평등가족위원회", "문화체육관광위원회",
    "네이버", "쿠팡", "토스", "배달의민족", "이통3사", "오픈AI", "구글 AI", "MS AI", "메타 AI", "애플 AI",
    "EU 플랫폼 규제", "트럼프 행정부 빅테크", "인공지능 산업 동향", "플랫폼 시장점유율"
]

all_raw_data = []
total_start_time = time.time()

print("🚀 구글 RSS 기반 필터링 없는 '기사 전문 무조건 전체 수집'을 시작합니다...\n")

# --------- [STEP 1] 내부 루프 수정 코드 ---------
for idx, keyword in enumerate(keywords, 1):
    print(f"▶ [{idx}/{len(keywords)}] 검색어: [{keyword}]")
    
    encoded_keyword = urllib.parse.quote(keyword)
    rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}+when:1d&hl=ko&gl=KR&ceid=KR:ko"
    
    collected_count = 0
    
    try:
        # 💡 핵심: requests에 맥북 헤더를 장착하여 구글 RSS 데이터를 사람인 척 먼저 다운로드!
        rss_res = requests.get(rss_url, headers=web_headers, timeout=10)
        
        if rss_res.status_code == 200:
            # 다운로드한 XML 데이터를 feedparser로 분석
            feed = feedparser.parse(rss_res.text)
        else:
            print(f"  └ ⚠️ 구글 RSS 접근 실패 (상태코드: {rss_res.status_code})")
            feed = feedparser.parse("") # 빈 객체 생성
    except Exception as e:
        print(f"  └ ⚠️ 네트워크 에러: {e}")
        feed = feedparser.parse("")

    # 이 아래부터는 기존 entry 도는 코드와 완전히 똑같습니다!
    for entry in feed.entries:
        pub_date = parsedate_to_datetime(entry.published)
        url = entry.link
        
        publisher = entry.source.get('title', '알 수 없음') if 'source' in entry else '알 수 없음'
        full_body = ""
        
        try:
            art_res = requests.get(url, headers=web_headers, timeout=5)
            if art_res.status_code == 200:
                art_soup = BeautifulSoup(art_res.text, "html.parser")
                
                if "news.naver.com" in art_res.url or "n.news.naver.com" in art_res.url:
                    body_tag = art_soup.find("article", id="dic_area")
                    if body_tag:
                        full_body = body_tag.get_text(separator="\n", strip=True)
                
                if not full_body:
                    paragraphs = art_soup.find_all('p')
                    full_body = "\n".join([p.get_text().strip() for p in paragraphs if len(p.get_text().strip()) > 20])
        except:
            pass
            
        if not full_body:
            full_body = entry.summary
            
        all_raw_data.append({
            "검색어": keyword,
            "기사제목": entry.title,
            "발행기관": publisher,
            "기사 전체 내용": full_body,
            "발행일시": pub_date.strftime("%Y-%m-%d %H:%M"),
            "링크": url
        })
        collected_count += 1
        print(f"  └ 📥 기사 전문 낚는 중... [{collected_count}개 완료]", end="\r")
        
        time.sleep(random.uniform(0.2, 0.4))
        
    print(f"  └ ✅ [{keyword}] 수집 완료! (총 {collected_count}개 전문 확보)      ")
    
    # 💡 키워드 1개가 끝날 때마다 구글이 숨 쉴 틈을 주기 위해 1~2초씩 강제 휴식!
    time.sleep(random.uniform(1.0, 2.0))

# ==========================================
# 2. 결과 저장 및 최종 소요 시간 출력
# ==========================================
print("\n💾 데이터 저장 중...")
df = pd.DataFrame(all_raw_data)
df.to_csv("google_news_no_filter_raw.csv", index=False, encoding="utf-8-sig")

total_end_time = time.time()
total_duration = total_end_time - total_start_time

print("\n" + "="*50)
print(f"🎉 테스트 수집이 완료되었습니다! 파일명: 'google_news_no_filter_raw.csv'")
print(f"⏱️ 총 수집된 기사 수: {len(all_raw_data)}개")
print(f"⏱️ 총 소요 시간: {total_duration/60:.2f}분 ({total_duration:.2f}초)")
print("="*50)