import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from requests import Session

# 網頁版面設定
st.set_page_config(page_title="個股基本面評價與價值投資回測系統", layout="wide")

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

st.markdown('<div class="main-title">📊 個股基本面評價與價值投資回測系統 (盈餘成長率、淨利率、營收成長)</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 側邊欄參數設定
# -------------------------------------------------------------
st.sidebar.header("⚙️ 基本面策略參數設定")

ticker_symbol = st.sidebar.text_input("輸入個股代號 (Yahoo Finance)", value="AAPL").strip().upper()
base_pe = st.sidebar.slider("基准本益比 (Base P/E)", min_value=10.0, max_value=35.0, value=20.0, step=1.0, help="用於計算合理本益比的基準")
margin_of_safety = st.sidebar.slider("安全邊際折扣 (%)", min_value=0.0, max_value=30.0, value=15.0, step=5.0, help="當股價低於理論價格多少折數時才執行買入")

target_start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2020-01-31"))
target_end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-12-31"))

run_btn = st.sidebar.button("🚀 開始計算基本面評價與回測")

if "fundamental_executed" not in st.session_state:
    st.session_state.fundamental_executed = False

if run_btn:
    with st.spinner(f"正在取得 {ticker_symbol} 財務報表與市場價格數據..."):
        try:
            session = Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            
            ticker = yf.Ticker(ticker_symbol, session=session)
            
            # 取得歷史股價
            df_hist = ticker.history(start="2018-01-01", end=None)
            if df_hist.empty or "Close" not in df_hist.columns:
                st.error(f"❌ 找不到代號 {ticker_symbol} 的歷史價格數據。")
                st.stop()
                
            stock_prices = df_hist["Close"]
            
            # 【修正】移除時區資訊，避免時區比對錯誤
            if stock_prices.index.tz is not None:
                stock_prices.index = stock_prices.index.tz_localize(None)

            # 嘗試取得財務報表以計算最新基本面特徵
            income_stmt = ticker.quarterly_income_stmt
            
            # 初始化財務指標預設值
            latest_net_margin = 0.15
            latest_rev_growth = 0.08
            latest_eps = 5.0
            
            if income_stmt is not None and not income_stmt.empty:
                rev_rows = [r for r in income_stmt.index if 'Total Revenue' in r or 'Revenue' in r]
                net_rows = [r for r in income_stmt.index if 'Net Income' in r]
                eps_rows = [r for r in income_stmt.index if 'Basic EPS' in r or 'Diluted EPS' in r]
                
                if rev_rows and net_rows:
                    revs = income_stmt.loc[rev_rows[0]].dropna()
                    nets = income_stmt.loc[net_rows[0]].dropna()
                    if len(revs) >= 4 and len(nets) >= 4:
                        ttm_rev = revs.iloc[:4].sum()
                        prev_ttm_rev = revs.iloc[4:8].sum() if len(revs) >= 8 else ttm_rev
                        latest_rev_growth = (ttm_rev - prev_ttm_rev) / prev_ttm_rev if prev_ttm_rev > 0 else 0.08
                        
                        ttm_net = nets.iloc[:4].sum()
                        latest_net_margin = ttm_net / ttm_rev if ttm_rev > 0 else 0.15
                
                if eps_rows:
                    epss = income_stmt.loc[eps_rows[0]].dropna()
                    if len(epss) >= 4:
                        latest_eps = epss.iloc[:4].sum()

            # 建立月頻資料與基本面評價模型
            df_m = stock_prices.resample("ME").last().ffill()
            
            eval_records = []
            for date, price in df_m.items():
                if date < pd.to_datetime(target_start_date):
                    continue
                if date > pd.to_datetime(target_end_date):
                    break
                
                implied_pe = max(10.0, base_pe * (1.0 + latest_rev_growth))
                implied_eps = max(1.0, latest_eps * (price / df_m.iloc[0]))
                theoretical_price = implied_eps * implied_pe
                
                eval_records.append({
                    "Date": date,
                    "Close": price,
                    "Revenue_Growth": latest_rev_growth * 100,
                    "Net_Margin": latest_net_margin * 100,
                    "Theoretical_Price": theoretical_price,
                    "Buy_Threshold_Price": theoretical_price * (1.0 - margin_of_safety / 100.0)
                })

            results_df = pd.DataFrame(eval_records)
            if results_df.empty:
                st.error("在設定的回測期間內沒有足夠資料！")
            else:
                results_df = results_df.set_index("Date")
                st.session_state.fundamental_executed = True
                st.session_state.fundamental_results = results_df
                st.session_state.ticker_symbol = ticker_symbol
                st.session_state.margin_of_safety = margin_of_safety

        except Exception as e:
                st.error(f"❌ 執行過程發生錯誤: {e}")

if st.session_state.get("fundamental_executed", False):
    res_df = st.session_state.get("fundamental_results")
    t_sym = st.session_state.get("ticker_symbol", "AAPL")
    mos = st.session_state.get("margin_of_safety", 15)

    if res_df is not None and not res_df.empty:
        st.markdown(f'<div class="section-header">📈 {t_sym} 基本面合理價評估 vs 實際股價</div>', unsafe_allow_html=True)
        
        latest_row = res_df.iloc[-1]
        curr_p = latest_row["Close"]
        theo_p = latest_row["Theoretical_Price"]
        buy_p = latest_row["Buy_Threshold_Price"]
        rev_g = latest_row["Revenue_Growth"]
        n_mar = latest_row["Net_Margin"]

        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("目前最新股價", f"${curr_p:.2f}")
        mcol2.metric("基本面理論合理價", f"${theo_p:.2f}", f"營收成長: {rev_g:.1f}%")
        mcol3.metric(f"安全邊際買入價 (-{mos}%)", f"${buy_p:.2f}")
        mcol4.metric("淨利潤率 (Profit Margin)", f"{n_mar:.1f}%")

        # -------------------------------------------------------------
        # 📊 互動式圖表
        # -------------------------------------------------------------
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=res_df.index, y=res_df["Close"],
            mode='lines', name='實際股價 (Actual Price)',
            line=dict(color='#00d2ff', width=2.5)
        ))

        fig.add_trace(go.Scatter(
            x=res_df.index, y=res_df["Theoretical_Price"],
            mode='lines', name='基本面理論合理價 (Fair Value)',
            line=dict(color='#ff9900', width=2, dash='dash')
        ))

        fig.add_trace(go.Scatter(
            x=res_df.index, y=res_df["Buy_Threshold_Price"],
            mode='lines', name=f'安全邊際買入線 (-{mos}%)',
            line=dict(color='#00ff66', width=1.5, dash='dot')
        ))

        fig.update_layout(
            xaxis=dict(title="日期", gridcolor='#222629'),
            yaxis=dict(title="價格 ($)", gridcolor='#222629'),
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='white'),
            legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)'),
            margin=dict(l=40, r=40, t=20, b=80)
        )

        st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------
        # 💰 策略回測與淨值曲線
        # -------------------------------------------------------------
        st.markdown(f'<div class="section-header">💰 基本面價值策略模擬淨值曲線 vs 買進持有</div>', unsafe_allow_html=True)
        
        bt_df = res_df.copy()
        bt_df["Signal"] = np.where(bt_df["Close"] < bt_df["Buy_Threshold_Price"], 1, 
                                   np.where(bt_df["Close"] > bt_df["Theoretical_Price"], 0, np.nan))
        bt_df["Signal"] = bt_df["Signal"].ffill().fillna(0)
        
        stock_ret = bt_df["Close"].pct_change().fillna(0)
        bt_df["Strategy_Return"] = bt_df["Signal"].shift(1) * stock_ret
        bt_df["Strategy_Return"] = bt_df["Strategy_Return"].fillna(0)

        bt_df["Benchmark_Nav"] = (1.0 + stock_ret).cumprod()
        bt_df["Strategy_Nav"] = (1.0 + bt_df["Strategy_Return"]).cumprod()

        total_days = (bt_df.index[-1] - bt_df.index[0]).days
        years = max(total_days / 365.25, 0.5)
        
        strat_cagr = (bt_df["Strategy_Nav"].iloc[-1] ** (1 / years)) - 1.0
        bench_cagr = (bt_df["Benchmark_Nav"].iloc[-1] ** (1 / years)) - 1.0

        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.metric("價值策略年化報酬率 (CAGR)", f"{strat_cagr * 100:.2f}%", f"買進持有: {bench_cagr * 100:.2f}%")
        pcol2.metric("價值策略總淨值", f"{bt_df['Strategy_Nav'].iloc[-1]:.2f}x", f"買進持有: {bt_df['Benchmark_Nav'].iloc[-1]:.2f}x")
        pcol3.metric("回測期間", f"{years:.1f} 年")

        nav_chart_df = bt_df[["Strategy_Nav", "Benchmark_Nav"]].copy()
        nav_chart_df.columns = ["基本面價值策略淨值", f"{t_sym} 買進持有"]
        st.line_chart(nav_chart_df)
