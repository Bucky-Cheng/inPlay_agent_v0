import streamlit as st

import pandas as pd
import numpy as np
from scipy.stats import poisson
import google.generativeai as genai
import json
import requests

# --- 基礎配置 ---
st.set_page_config(page_title="AI 策略終端", layout="wide")
st.title("In-PlayGemini Agent ")

api_key = 'AIzaSyDtA1X_YnrROi9sjh36sC93HIsiUfzro2E'
# 側邊欄：API 與 基礎參數
with st.sidebar:
    st.header("⚙️ 設定")
    #api_key = st.text_input("Gemini API Key", type="password")
    h_base = st.number_input("主隊基礎 Alpha", value=1.2, step=0.1)
    a_base = st.number_input("客隊基礎 Alpha", value=1.0, step=0.1)
    rho_base = st.slider("基礎 Rho (平局修正)", -0.1, 0.1, 0.05)

if api_key:
    genai.configure(api_key=api_key)
    #model = genai.GenerativeModel('gemini-1.5-flash')

# sofa commentry
import requests
import json

import requests

import requests
import json

import json
import cloudscraper

def get_sofascore_analysis_payload(match_id):
    # 1. Initialize the scraper to bypass the '403 Challenge'
    scraper = cloudscraper.create_scraper()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36',
        'Referer': 'https://www.sofascore.com/'
    }

    # 2. Endpoints
    stats_url = f"https://api.sofascore.com/api/v1/event/{match_id}/statistics"
    incidents_url = f"https://api.sofascore.com/api/v1/event/{match_id}/incidents"
    comm_url = f"https://api.sofascore.com/api/v1/event/{match_id}/comments"
    score_url = f"https://api.sofascore.com/api/v1/event/{match_id}"

    # --- PART A: TEAM TOTALS ---
    team_totals = {}
    stats_res = scraper.get(stats_url, headers=headers)
    if stats_res.status_code == 200:
        # data[0] is the 'All' period (full match)
        stats_data = stats_res.json().get('statistics', [])
        if stats_data:
            groups = stats_data[0].get('groups', [])
            target_keys = [
                'Expected goals', 'Ball possession', 'Total shots', 
                'Big chances', 'Big chances missed', 'Corner kicks', 'Goalkeeper saves'
            ]
            for group in groups:
                for item in group.get('statisticsItems', []):
                    if item['name'] in target_keys:
                        # 'h' for home, 'a' for away to save tokens
                        team_totals[item['name']] = {"h": item['home'], "a": item['away']}

    clean_incidents = []
    inc_res = scraper.get(incidents_url, headers=headers)
    if inc_res.status_code == 200:
        for item in inc_res.json().get('incidents', []):
            itype = item.get('incidentType', 'N/A')
            iclass = item.get('incidentClass', 'N/A')
            player = item.get('player', {}).get('name', 'Unknown Player')
            
            # Clean up non-player events per your preference
            if itype in ['period', 'injuryTime', 'substitution']:
                player = 'N/A'
            
            if itype == 'period':
                itype = item.get('text', 'Period End')

            event_info = {
                "m": item.get('time', 0), # Absolute minute for sorting
                "t": f"{item.get('time', 0)}'",
                "type": itype,
                "class": iclass,
                "player": player,
                "team": "h" if item.get('isHome') else "a"
            }

            # Capture conditional details (Assists & Subs)
            if item.get('assistPlayer'):
                event_info["assist"] = item.get('assistPlayer', {}).get('name')
            
            if itype == 'substitution':
                event_info["out"] = item.get('playerOut', {}).get('name')
                event_info["in"] = item.get('playerIn', {}).get('name')

            clean_incidents.append(event_info)
    
    
    # --- PART B: ENHANCED COMMENTARY ---
    enhanced_comm = []
    comm_res = scraper.get(comm_url, headers=headers)
    if comm_res.status_code == 200:
        raw_comments = comm_res.json().get('comments', [])
        for c in raw_comments:
            # Enhanced Time Logic
            raw_time = str(c.get('time', '0'))
            if '+' in raw_time:
                base, extra = map(int, raw_time.split('+'))
                abs_minute = base + extra
            else:
                abs_minute = int(raw_time)

            period = "1st" if abs_minute <= 45 else "2nd"
            if abs_minute > 90: period = "FT/ET"

            enhanced_comm.append({
                "m": abs_minute,        # Sortable absolute minute
                "t": f"{raw_time}'",    # Display time (e.g. 45+2')
                "p": period,
                "txt": c.get('text'),
                "goal": c.get('isGoal', False)
            })

    # --- PART C: CURRENT SCORE ---
    score_res = scraper.get(score_url, headers=headers)
    if score_res.status_code == 200:
        data = score_res.json().get('event', {})
        h_score = data.get('homeScore', {}).get('current', 0)
        a_score = data.get('awayScore', {}).get('current', 0)


    # --- PART D: FINAL PAYLOAD ---
    # Combine everything and sort commentary chronologically
    full_payload = {
        "home_score": h_score,
        "away_score": a_score,
        "stats": team_totals,
        "key_events": sorted(clean_incidents, key=lambda x: x['m']),
        "commentary": sorted(enhanced_comm, key=lambda x: x['m'])
    }

    # Use separators=(',', ':') to strip whitespace for token efficiency
    return h_score, a_score, json.dumps(full_payload, ensure_ascii=False, separators=(',', ':'))

# --- HOW TO USE ---
# match_id = 11352342
# final_prompt_data = get_sofascore_analysis_payload(match_id)
# print(f"Final Prompt String:\n{final_prompt_data}")

# Example Match ID (you'll need a current/recent one from the site)
#print(get_sofascore_analysis_payload(15471610))


# print(get_sofa_enhanced_commentary(15471610))



# --- 數學模型：Dixon-Coles 矩陣 ---
def generate_dc_matrix(l_h, l_a, rho, max_g=5):
    matrix = np.zeros((max_g + 1, max_g + 1))
    for x in range(max_g + 1):
        for y in range(max_g + 1):
            prob = poisson.pmf(x, l_h) * poisson.pmf(y, l_a)
            # Dixon-Coles 修正
            adj = 1.0
            if x == 0 and y == 0: adj = 1 - l_h * l_a * rho
            elif x == 1 and y == 0: adj = 1 + l_h * rho
            elif x == 0 and y == 1: adj = 1 + l_a * rho
            elif x == 1 and y == 1: adj = 1 - rho
            matrix[x, y] = prob * max(0, adj)
    return matrix / matrix.sum()

# --- UI 介面 ---
match_id = st.text_input("🔗 Sofascore 比賽 id", placeholder="id")

if match_id:
    if st.button("🚀 執行全自動分析"):
        with st.spinner("正在同步 Sofascore 數據並啟動 AI..."):
            try:
                
                #events_json = json.dumps(recent_events, ensure_ascii=False, indent=2)
                home_score, away_score, commentry_json = get_sofascore_analysis_payload(match_id)
                # --- 1. 生成可複製的 Prompt ---
                st.subheader("📋 可複製的 AI Agent Prompt")
                full_prompt = f"""你是一位足球博弈專家,分析以下實時足球數據，特別注意進球後雙方的戰術節奏變化：
實時比分：{home_score}-{away_score}
【數據流】: {commentry_json}
【要求】:
1. 分析主客隊進攻強度與換人戰術意圖。
2. 判斷比賽當前處於哪種狀態：
    "locked" = 雙方保守，節奏緩慢，重兵囤積中場。
    "open" = 均衡被打破，大開大合，攻防轉換極快，反擊空間巨大。
2. 預測接下來到完場的進球機率和總進球數。
3. 預測總進球數和進球數區間
4. 輸出修正係數: 
    {{"h_atk": 1.0, // 基準1.0
      "a_atk": 1.0, // 基準1.0
      "rho_adj": 0.0,
      "game_state": "open",  // 或 "locked"
      "momentum_score": 1.4 // 1.0 到 2.0 之間，代表比賽瘋狂程度}}
    """
                st.code(full_prompt, language="text")

                # --- 2. 調用 Gemini 分析 ---
                # if api_key:
                #     st.divider()
                #     st.subheader("🧠 Gemini 實時戰術分析")
                #     response = model.generate_content(f"{full_prompt}\n請先給出直觀分析，最後以 JSON 格式結尾。")
                #     st.markdown(response.text)
                    
                #     # 嘗試從 AI 回覆中提取 JSON (簡化邏輯)
                #     try:
                #         # 這裡預設 AI 會乖乖輸出 JSON，實際可增加更強的解析
                #         res_text = response.text.split('{')[-1].split('}')[0]
                #         ai_data = json.loads("{" + res_text + "}")
                #         h_mod = ai_data.get('h_atk', 1.0)
                #         a_mod = ai_data.get('a_atk', 1.0)
                #         r_adj = ai_data.get('rho_adj', 0.0)
                #     except:
                #         h_mod, a_mod, r_adj = 1.0, 1.0, 0.0
                # else:
                #     h_mod, a_mod, r_adj = 1.0, 1.0, 0.0

                # # --- 3. 輸出矩陣 ---
                # st.divider()
                # st.subheader("📊 AI 修正後之 Dixon-Coles 矩陣")
                # l_h = h_base * h_mod
                # l_a = a_base * a_mod
                # final_rho = rho_base + r_adj
                
                # matrix = generate_dc_matrix(l_h, l_a, final_rho)
                # df_m = pd.DataFrame((matrix*100).round(1), 
                #                     columns=[f"客{i}" for i in range(6)], 
                #                     index=[f"主{i}" for i in range(6)])
                
                # st.dataframe(df_m.style.background_gradient(cmap='Greens'))
                
                # # 中分區間提醒
                # u25 = (matrix[0,0]+matrix[1,0]+matrix[0,1]+matrix[2,0]+matrix[0,2]+matrix[1,1])
                # st.metric("2.5 小球預估機率", f"{u25*100:.1f}%")

            except Exception as e:
                st.error(f"執行出錯: {e}")