import os
import json
import re
import datetime
import io
import time
import requests
from bs4 import BeautifulSoup
import pypdf
from google import genai
from google.genai import types

# ==========================================
# 設定エリア (GitHub Secrets からの読み込みに対応)
# ==========================================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")
STATE_FILE = "notified_ipos.json"
JPX_URL = "https://www.jpx.co.jp/listing/stocks/new/"

# リトライ設定
MAX_RETRIES = 3
RETRY_DELAY_SEC = 15
# ==========================================

def load_processed_codes():
    """通知済み銘柄コードの読み込み"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        except Exception:
            return set()
    return set()

def save_processed_codes(codes_set):
    """通知済み銘柄コードの保存"""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(codes_set), f, ensure_ascii=False, indent=2)

def extract_pdf_data(pdf_url):
    """JPXの会社概要PDFからテキストを抽出（リトライ付き）"""
    if not pdf_url:
        return "会社概要PDFなし"
        
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.get(pdf_url, headers=headers, timeout=15)
            if res.status_code == 200:
                pdf_file = io.BytesIO(res.content)
                reader = pypdf.PdfReader(pdf_file)
                text = ""
                for page in reader.pages[:2]:
                    text += page.extract_text() or ""
                clean_text = re.sub(r"\s+", " ", text).strip()
                return clean_text[:1500] if clean_text else "概要テキスト抽出不可"
            elif res.status_code in [500, 502, 503, 504, 429]:
                print(f"    [PDF取得] HTTP {res.status_code} 受信。{RETRY_DELAY_SEC}秒待機して再試行 ({attempt}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY_SEC)
            else:
                return f"会社概要取得失敗 (HTTP {res.status_code})"
        except Exception as e:
            print(f"    [PDF取得] 通信エラー: {e}。{RETRY_DELAY_SEC}秒待機 ({attempt}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY_SEC)
            
    return "会社概要取得失敗 (リトライ上限超過)"

def fetch_jpx_ipos():
    """JPX一覧からスクレイピング"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = requests.get(JPX_URL, headers=headers, timeout=15)
            if res.status_code == 200:
                res.encoding = "utf-8"
                break
            elif res.status_code in [500, 502, 503, 504, 429]:
                print(f"[JPX取得] HTTP {res.status_code} 受信。{RETRY_DELAY_SEC}秒待機 ({attempt}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY_SEC)
        except Exception as e:
            print(f"[JPX取得] 通信エラー: {e}。{RETRY_DELAY_SEC}秒待機 ({attempt}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY_SEC)
    else:
        return []

    soup = BeautifulSoup(res.text, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        return []

    rows = tables[0].find_all("tr")
    ipos = []
    
    i = 2
    while i < len(rows):
        cols_top = rows[i].find_all(["td", "th"])
        cols_bottom = rows[i+1].find_all(["td", "th"]) if i + 1 < len(rows) else []

        if len(cols_top) < 7:
            i += 1
            continue

        text_top = [c.get_text(strip=True) for c in cols_top]
        text_bottom = [c.get_text(strip=True) for c in cols_bottom] if cols_bottom else []

        dates_raw = text_top[0]
        company_name = text_top[1]
        code = text_top[2]
        provisional_price = text_top[5]
        public_shares = text_top[6]

        market = text_bottom[0] if len(text_bottom) > 0 else ""
        selling_shares = text_bottom[4] if len(text_bottom) > 4 else ""

        pdf_link = ""
        if len(cols_top) > 3:
            link_tag = cols_top[3].find("a")
            if link_tag and link_tag.get("href"):
                href = link_tag.get("href")
                pdf_link = "https://www.jpx.co.jp" + href if not href.startswith("http") else href

        date_match = re.findall(r"\d{4}/\d{2}/\d{2}", dates_raw)
        listing_date = date_match[0] if len(date_match) > 0 else dates_raw

        ipos.append({
            "code": code,
            "name": company_name,
            "market": market,
            "listing_date": listing_date,
            "provisional_price": provisional_price,
            "public_shares_k": public_shares,
            "selling_shares_k": selling_shares,
            "pdf_url": pdf_link
        })
        i += 2
        
    return ipos

def analyze_with_gemini(ipo_info, summary_text):
    """Gemini APIによる公募参加判断 ＆ セカンダリ立ち回り分析（503/429リトライ付き）"""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
あなたは冷徹でシビアなIPO専門の株式アナリストです。
以下の新規上場銘柄データと会社概要に基づき、初値高騰期待/公募割れリスクを評価し、【公募申込の是非】および【落選時の初値後セカンダリ戦略】を判定してください。

【銘柄データ】
- 銘柄名: {ipo_info['name']} ({ipo_info['code']})
- 上場市場: {ipo_info['market']}
- 上場予定日: {ipo_info['listing_date']}
- 仮条件: {ipo_info['provisional_price']} 円
- 公募株数: {ipo_info['public_shares_k']} 千株
- 売出株数(OA含む): {ipo_info['selling_shares_k']} 千株

【事業概要・サマリー】
{summary_text}

【出力ルール】
- 挨拶や前置きは一切出力せず、以下のフォーマットのみを出力してください。
- Slack向けに、太字は *テキスト* で囲んでください。

【出力フォーマット】
*【公募判定: S / A / B / C】* （S: 鉄板 / A: 積極参加 / B: 中立 / C: 見送り推奨）
• *BB期間:* （概要テキストから「需要申告期間」や「ブックビルディング期間」の日程を抽出して記載。不明なら上場日から逆算した目安を記載）
• *推定吸収金額:* 約〇〇億円
• *初値見通し:* （例: 公募比 +30%〜+50% / 公募割れ警戒 など）
• *公募アクション:* （例: 全力申込 / ポイント狙いのみ / スルー など）

*■ セカンダリ立ち回り（落選・初値後の狙い目）*
• *初値後の想定値動き:* （例: 寄り天急落警戒 / 初値低調なら押し目買い妙味 など）
• *セカンダリ方針:* （例: 監視対象から除外 / 公募比〇倍以下なら打診買い / 数週間後の需給消化待ち など）

*■ 要点分析*
• *需給・売出:* （具体的に300字以内）
• *事業・成長性:* （具体的に300字以内）
• *主なリスク:* （具体的に300字以内）
"""
    config = types.GenerateContentConfig(
        temperature=0.2
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=config
            )
            if response.text:
                return response.text.strip(), None
        except Exception as e:
            err_msg = str(e)
            print(f"    [Gemini API] エラー発生 ({err_msg})。{RETRY_DELAY_SEC}秒待機して再試行 ({attempt}/{MAX_RETRIES})...")
            time.sleep(RETRY_DELAY_SEC)
            
    return None, f"Gemini API 呼出失敗 (503/過負荷/IP制限等): {MAX_RETRIES}回リトライ上限"

def send_slack_notification(message):
    """Slack Webhookへの通知送信"""
    payload = {
        "text": message,
        "mrkdwn": True
    }
    headers = {"Content-Type": "application/json"}
    try:
        res = requests.post(SLACK_WEBHOOK_URL, data=json.dumps(payload), headers=headers, timeout=10)
        return res.status_code == 200
    except Exception:
        return False

def main():
    today = datetime.date.today()
    print(f"[{today}] IPO監視ボット起動...")
    
    processed_codes = load_processed_codes()
    all_ipos = fetch_jpx_ipos()
    
    target_ipos = []
    for ipo in all_ipos:
        if ipo["code"] in processed_codes:
            continue
            
        try:
            l_date = datetime.datetime.strptime(ipo["listing_date"], "%Y/%m/%d").date()
            if l_date < today:
                continue
        except Exception:
            continue
            
        if ipo["provisional_price"] == "-" or not ipo["provisional_price"]:
            continue
            
        target_ipos.append(ipo)
        
    print(f"通知対象銘柄数: {len(target_ipos)} 件")
    
    for ipo in target_ipos:
        print(f"--> 分析実行中: [{ipo['code']}] {ipo['name']}")
        
        summary_text = extract_pdf_data(ipo["pdf_url"])
        analysis_result, error_msg = analyze_with_gemini(ipo, summary_text)
        
        # 分析失敗時のSlack通知
        if error_msg or not analysis_result:
            fail_msg = f"<!channel> ⚠️ *【IPO分析エラー通知】*\n" \
                       f"*{ipo['name']}* ({ipo['code']} / {ipo['market']}) の分析を試みましたが、" \
                       f"API過負荷（503等）または通信エラーにより失敗しました。\n" \
                       f"• エラー詳細: `{error_msg}`\n" \
                       f"※次回の定期実行（2日後）で再試行されます。"
            send_slack_notification(fail_msg)
            print(f"    分析失敗のためエラー通知送信: {ipo['code']}")
            continue # 処理済みには追加せず次回リトライさせる
        
        # 分析成功時のSlack通知
        slack_msg = f"<!channel> *【IPO公募判定】 {ipo['name']} ({ipo['code']} / {ipo['market']})*\n" \
                    f"🗓 *上場予定日:* `{ipo['listing_date']}`\n" \
                    f"💰 *仮条件:* `{ipo['provisional_price']} 円`\n\n" \
                    f"{analysis_result}\n" \
                    f"────────────────────"
                    
        success = send_slack_notification(slack_msg)
        if success:
            print(f"    Slack通知成功: {ipo['code']}")
            processed_codes.add(ipo["code"])
        else:
            print(f"    Slack通知失敗: {ipo['code']}")
            
    save_processed_codes(processed_codes)
    print("処理完了。")

if __name__ == "__main__":
    main()
