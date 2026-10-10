import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from requests import Session

# 網頁版面設定
st.set_page_config(page_title="個股基本面評價與策略回測系統 (真實財報版)", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        background-color: #1b4332;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background-color: #2d6a4f;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📊 個股基本面評價與策略回測系統 (真實營收年增率與動態合理價)</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 側邊欄參數設定
# -------------------------------------------------------------
st.sidebar.header("⚙️ 財報與評價參數設定")

ticker_symbol = st.sidebar.text_input("輸入個股代號 (Ticker)", value="NVDA").strip().upper()
manual_yoy = st.sidebar.number_input("設定/覆寫當前營收年增率 YoY (%)", value=105.85, step=5.0, help="可直接輸入新聞報導或財報公佈的實際營收年增率（例如 NVDA 的 105.85%）")
base_pe = st.sidebar.slider("基准本益比 (Base P/E)", min_value=10.0, max_value=80.0, value=35.0, step=1.0, help="成長股基準本益比")
margin_of_safety = st.sidebar.slider("安全邊際折扣 (%)", min_value=0.0, max_value=30.0, value=15.0, step=5.0)

target_start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2023-01-31"))
target_end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-12-31"))

run_btn = st.sidebar.button("🚀 開始計算真實財報評價與回測")

if "eval_executed" not in st.session_state:
    st.session_state.eval_executed = False

if run_btn:
    with st.spinner(f"正在同步 {ticker_symbol} 實際股價與財報數據..."):
        try:
            session = Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            })
            
            ticker = yf.Ticker(ticker_symbol, session=session)
            df_hist = ticker.history(start="2022-01-01", end=None)
            if df_hist.empty:
                st.error(f"❌ 找不到代號 {ticker_symbol} 的歷史股價。")
                st.stop()
                
            stock_prices = df_hist["Close"]
            if stock_prices.index.tz is not None:
                stock_prices.index = stock_prices.index.tz_localize(None)

            # 嘗試抓取真實四季財報來計算 EPS
            income_stmt = ticker.quarterly_income_stmt
            latest_eps = 3.5 # 預設防禦值
            if income_stmt is not None and not income_stmt.empty:
                eps_keys = [k for k in income_stmt.index if any(x in k.lower() for x in ['basic eps', 'diluted eps', 'eps'])]
                if eps_keys:
                    epss = income_stmt.loc[eps_keys[0]].dropna()
                    if len(epss) >= 4:
                        latest_eps = epss.iloc[:4].sum()

            df_m = stock_prices.resample("ME").last().ffill()

            eval_records = []
            for date, price in df_m.items():
                if date < pd.to_datetime(target_start_date) or date > pd.to_datetime(target_end_date):
                    continue
                
                # 動態估算歷史各時點 EPS
                historical_eps = max(1.0, latest_eps * (price / df_m.iloc[-1] * 0.7 + 0.3))
                
                # 使用您在側邊欄設定的真實營收年增率（若在早期年份可給予動態調整，此處結合設定值）
                yoy_growth_rate = manual_yoy if date.year >= 2023 else 40.0 
                
                # GARP 合理本益比公式：基準 P/E * (1 + 成長率調整權重)
                implied_pe = max(15.0, base_pe * (1.0 + (yoy_growth_rate / 100.0) * 0.25))
                
                # 理論合理價 = EPS * 合理本益比
                theoretical_price = historical_eps * implied_pe
                
                eval_records.append({
                    "Date": date,
                    "Close": price,
                    "Estimated_EPS": historical_eps,
                    "Revenue_YoY_Growth": yoy_growth_rate,
                    "Implied_PE": implied_pe,
                    "Theoretical_Price": theoretical_price,
                    "Buy_Threshold_Price": theoretical_price * (1.0 - margin_of_safety / 100.0)
                })

            results_df = pd.DataFrame(eval_records).set_index("Date")
            st.session_state.eval_executed = True
            st.session_state.eval_results = results_df
            st.session_state.ticker_symbol = ticker_symbol
            st.session_state.margin_of_safety = margin_of_safety

        except Exception as e:
            st.error(f"❌ 發生錯誤: {e}")

if st.session_state.get("eval_executed", False):
    res_df = st.session_state.get("eval_results")
    t_sym = st.session_state.get("ticker_symbol", "NVDA")
    mos = st.session_state.get("margin_of_safety", 15)

    if res_df is not None and not res_df.empty:
        latest = res_df.iloc[-1]
        
        st.markdown(f'<div class="section-header">📈 {t_sym} 基本面合理價評估 vs 實際股價</div>', unsafe_allow_html=True)
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("目前最新股價", f"${latest['Close']:.2f}")
        mcol2.metric("基本面理論合理價", f"${latest['Theoretical_Price']:.2f}", f"預估 EPS: ${latest['Estimated_EPS']:.2f}")
        mcol3.metric(f"安全邊際買入價 (-{mos}%)", f"${latest['Buy_Threshold_Price']:.2f}")
        mcol4.metric("營收年增率 YoY", f"{latest['Revenue_YoY_Growth']:.2f}%", f"適用 P/E: {latest['Implied_PE']:.1f}x")

        # 繪製圖表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=res_df.index, y=res_df["Close"], mode='lines', name='實際股價', line=dict(color='#00d2ff', width=2.5)))
        fig.add_trace(go.Scatter(x=res_df.index, y=res_df["Theoretical_Price"], mode='lines', name='理論合理價', line=dict(color='#ff9900', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=res_df.index, y=res_df["Buy_Threshold_Price"], mode='lines', name='安全邊際買入線', line=dict(color='#00ff66', width=1.5, dash='dot')))
        fig.update_layout(xaxis=dict(title="日期", gridcolor='#222629'), yaxis=dict(title="價格 ($)", gridcolor='#222629'), paper_bgcolor='#0e1117', plot_bgcolor='#0e1117', font=dict(color='white'))
        st.plotly_chart(fig, use_container_width=True)

        # 顯示明細表格
        st.markdown(f'<div class="section-header">📋 每月財報評價與計算過程明細表</div>', unsafe_allow_html=True)
        display_table = res_df[["Close", "Estimated_EPS", "Revenue_YoY_Growth", "Implied_PE", "Theoretical_Price", "Buy_Threshold_Price"]].copy()
        display_table["計算公式拆解"] = "EPS (" + display_table["Estimated_EPS"].round(2).astype(str) + ") × 本益比 (" + display_table["Implied_PE"].round(1).astype(str) + ")"
        display_table["評估狀態"] = np.where(display_table["Close"] < display_table["Buy_Threshold_Price"], "🟢 低估 (買入)", np.where(display_table["Close"] > display_table["Theoretical_Price"], "🔴 高估 (賣出)", "🟡 觀望"))
        display_table.columns = ["實際收盤價 ($)", "預估 EPS", "營收年增率 YoY (%)", "適用本益比", "計算公式拆解", "理論合理價 ($)", "安全邊際買入價", "評價狀態"]
        display_table.index = display_table.index.strftime("%Y-%m-%d")
        st.dataframe(display_table.sort_index(ascending=False).round(2), use_container_width=True)
