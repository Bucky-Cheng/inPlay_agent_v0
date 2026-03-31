import streamlit as st
import ScraperFC
import pandas as pd
import numpy as np
from scipy.stats import poisson
import google.generativeai as genai
import json

# --- 基礎配置 ---
st.set_page_config(page_title="AI 策略終端", layout="wide")
st.title("🛡️ SxGM + Gemini 戰術策略終端")

# 側邊欄：API 與 基礎參數
with st.sidebar:
    st.header("⚙️ 設定")
    api_key = st.text_input("Gemini API Key", type="password")
    h_base = st.number_input("主隊基礎 Alpha", value=1.2, step=0.1)
    a_base = st.number_input("客隊基礎 Alpha", value=1.0, step=0.1)
    rho_base = st.slider("基礎 Rho (平局修正)", -0.1, 0.1, 0.05)

if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

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
url = st.text_input("🔗 Sofascore 比賽 URL", placeholder="https://www.sofascore.com/football/match/...")

if url:
    if st.button("🚀 執行全自動分析"):
        with st.spinner("正在同步 Sofascore 數據並啟動 AI..."):
            try:
                sofa = ScraperFC.Sofascore()
                events = sofa.get_match_events(url)
                recent_events = events[-15:] if events else []
                events_json = json.dumps(recent_events, ensure_ascii=False, indent=2)

                # --- 1. 生成可複製的 Prompt ---
                st.subheader("📋 可複製的 AI Agent Prompt")
                full_prompt = f"""你是一位足球博弈專家。請解析以下實時事件數據：
【數據流】: {events_json}
【要求】:
1. 分析主客隊進攻強度與換人戰術意圖。
2. 預測接下來15分鐘的進球機率。
3. 輸出修正係數: {{"h_atk": 1.0, "a_atk": 1.0, "rho_adj": 0.0}} (基準1.0)"""
                st.code(full_prompt, language="text")

                # --- 2. 調用 Gemini 分析 ---
                if api_key:
                    st.divider()
                    st.subheader("🧠 Gemini 實時戰術分析")
                    response = model.generate_content(f"{full_prompt}\n請先給出直觀分析，最後以 JSON 格式結尾。")
                    st.markdown(response.text)
                    
                    # 嘗試從 AI 回覆中提取 JSON (簡化邏輯)
                    try:
                        # 這裡預設 AI 會乖乖輸出 JSON，實際可增加更強的解析
                        res_text = response.text.split('{')[-1].split('}')[0]
                        ai_data = json.loads("{" + res_text + "}")
                        h_mod = ai_data.get('h_atk', 1.0)
                        a_mod = ai_data.get('a_atk', 1.0)
                        r_adj = ai_data.get('rho_adj', 0.0)
                    except:
                        h_mod, a_mod, r_adj = 1.0, 1.0, 0.0
                else:
                    h_mod, a_mod, r_adj = 1.0, 1.0, 0.0

                # --- 3. 輸出矩陣 ---
                st.divider()
                st.subheader("📊 AI 修正後之 Dixon-Coles 矩陣")
                l_h = h_base * h_mod
                l_a = a_base * a_mod
                final_rho = rho_base + r_adj
                
                matrix = generate_dc_matrix(l_h, l_a, final_rho)
                df_m = pd.DataFrame((matrix*100).round(1), 
                                    columns=[f"客{i}" for i in range(6)], 
                                    index=[f"主{i}" for i in range(6)])
                
                st.dataframe(df_m.style.background_gradient(cmap='Greens'))
                
                # 中分區間提醒
                u25 = (matrix[0,0]+matrix[1,0]+matrix[0,1]+matrix[2,0]+matrix[0,2]+matrix[1,1])
                st.metric("2.5 小球預估機率", f"{u25*100:.1f}%")

            except Exception as e:
                st.error(f"執行出錯: {e}")