import os
import re
import sys
import json
import subprocess
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

NEWS_SCRIPT = os.getenv("NEWS_SCRIPT", "ceo_news_briefing_google_rss_v44.py")
DATA_DIR = os.path.join(BASE_DIR, "data_news")

def get_report_txt_path():
    date_key = datetime.now().strftime("%Y-%m-%d")
    return os.path.join(
        DATA_DIR,
        "daily",
        date_key,
        f"CEO_Morning_Briefing_{date_key}.txt"
    )
CATEGORY_ORDER = [
    "자사 및 계열사 이슈",
    "정부/국회",
    "경쟁사/해외이슈",
    "산업동향",
]

CATEGORY_RE = re.compile(r"^\s*☑️\s*(.+?)\s*$")
ARTICLE_RE = re.compile(r"^\s*(?:\d+\ufe0f?\u20e3|🔟|\d+[.)])\s+(.+?)\s*$")
URL_RE = re.compile(r"https?://\S+")
PRESS_RE = re.compile(r"^\s*\(([^)]+)\)\s*$")


def clean_text(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def run_news_crawler():
    script_path = os.path.join(BASE_DIR, NEWS_SCRIPT)

    if not os.path.exists(script_path):
        print(f"⚠️ {NEWS_SCRIPT} 파일이 없습니다. 기존 CEO_Morning_Briefing.txt만 변환합니다.")
        return

    print(f"📰 뉴스 크롤러 실행: {NEWS_SCRIPT}")
    subprocess.run([sys.executable, script_path], cwd=BASE_DIR, check=True)


def parse_briefing_txt(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} 파일이 없습니다.")

    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.splitlines()

    categories = {name: [] for name in CATEGORY_ORDER}
    current_category = None
    report_header = ""

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if not report_header and line.startswith("[") and "주요 이슈" in line:
            report_header = line

        category_match = CATEGORY_RE.match(line)
        if category_match:
            current_category = clean_text(category_match.group(1))
            if current_category not in categories:
                categories[current_category] = []
            i += 1
            continue

        article_match = ARTICLE_RE.match(line)
        if article_match and current_category:
            title = clean_text(article_match.group(1))
            link = ""
            source = ""
            summary_lines = []

            j = i + 1

            # URL 찾기
            while j < len(lines):
                candidate = lines[j].strip()
                if not candidate:
                    j += 1
                    continue
                url_match = URL_RE.search(candidate)
                if url_match:
                    link = url_match.group(0)
                    j += 1
                    break
                if CATEGORY_RE.match(candidate) or ARTICLE_RE.match(candidate):
                    break
                j += 1

            # 언론사 찾기
            while j < len(lines):
                candidate = lines[j].strip()
                if not candidate:
                    j += 1
                    continue
                press_match = PRESS_RE.match(candidate)
                if press_match:
                    source = clean_text(press_match.group(1))
                    j += 1
                break

            # 요약문 모으기
            while j < len(lines):
                candidate = lines[j].strip()
                if CATEGORY_RE.match(candidate) or ARTICLE_RE.match(candidate):
                    break
                if candidate:
                    summary_lines.append(clean_text(candidate))
                j += 1

            categories.setdefault(current_category, []).append({
                "title": title,
                "link": link,
                "source": source,
                "summary": " ".join(summary_lines).strip()
            })

            i = j
            continue

        i += 1

    return {
        "header": report_header,
        "categories": categories
    }


def save_news_json(parsed):
    os.makedirs(DATA_DIR, exist_ok=True)

    now = datetime.now()
    date_key = now.strftime("%Y-%m-%d")

    data = {
        "date": date_key,
        "generated_at": now.strftime("%Y-%m-%d %H:%M"),
        "header": parsed.get("header", ""),
        "category_order": CATEGORY_ORDER,
        "categories": parsed.get("categories", {}),
    }

    filepath = os.path.join(DATA_DIR, f"{date_key}.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"✅ 뉴스 JSON 저장 완료: {filepath}")
    return date_key, filepath


def push_to_github(date_key):
    try:
        print("\n🚀 깃허브로 일일 뉴스 데이터를 배달합니다...")

        # secret.txt 같은 민감 파일이 딸려 올라가지 않도록 data_news만 add
        subprocess.run(["git", "add", "data_news"], cwd=BASE_DIR, check=True)

        status = subprocess.run(
            ["git", "status", "--porcelain", "data_news"],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            check=True
        )

        if not status.stdout.strip():
            print("✨ 새로 추가되거나 변경된 뉴스 데이터가 없습니다.")
            return

        commit_msg = f"Update daily news: {date_key}"
        subprocess.run(["git", "commit", "-m", commit_msg], cwd=BASE_DIR, check=True)
        subprocess.run(["git", "push", "origin", "main"], cwd=BASE_DIR, check=True)

        print("✅ 배달 완료! 웹사이트에서 일일 뉴스 탭을 확인하세요.")
    except Exception as e:
        print(f"❌ 깃허브 배달 실패: {e}")


def main():
    run_news_crawler()

    report_txt = get_report_txt_path()
    parsed = parse_briefing_txt(report_txt)

    date_key, _ = save_news_json(parsed)
    push_to_github(date_key)


if __name__ == "__main__":
    main()