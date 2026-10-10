import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from requests import Session
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

from sklearn.linear_model import RidgeCV

# 網頁版面設定
st.set_page_config(page_title="個股機器學習多期評價與策略回測系統 (總經月度高頻版)", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        font-size: 24px;
        font-weight: bold;
        color: #ffffff;
        background-color: #0d1b2a;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 15px;
    }
    .section-header {
        font-size: 18px;
        font-weight: bold;
        color: #ffffff;
        background-color: #1b263b;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 15px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="main-title">📈 個股機器學習多期評價與策略回測系統 (結合每月高頻總經/基本面指標)</div>', unsafe_allow_html=True)

# -------------------------------------------------------------
# 側邊欄參數設定
# -------------------------------------------------------------
st.sidebar.header("⚙️ 回測參數設定")

ticker_symbol = st.sidebar.text_input("輸入個股代號 (Yahoo Finance)", value="AAPL").strip().upper()
forecast_horizon = st.sidebar.selectbox("選擇預測天期 (Horizon)", options=[3, 1], format_func=lambda x: f"預測未來 {x} 個月")

strategy_mode = st.sidebar.selectbox(
    "選擇交易策略模式", 
    options=["多頭策略 (預測漲幅超標則買入，否則現金)", "多空雙向 (預測漲買股，跌放空)"],
    index=0
)

buy_threshold = st.sidebar.slider("買進預測報酬率門檻 (%)", min_value=0.0, max_value=5.0, value=1.0, step=0.5, help="預測報酬率必須大於此門檻才發出買進訊號，否則轉為現金/空手")

momentum_window = st.sidebar.selectbox(
    "技術動能計算週期",
    options=[12, 6, 3, 1],
    index=0, 
    format_func=lambda x: f"{x} 個月動能 (Mom{x}M)"
)

train_window = st.sidebar.slider("滾動訓練月數 (Train Window)", min_value=12, max_value=60, value=36, step=6)
target_start_date = st.sidebar.date_input("回測開始日期", pd.to_datetime("2018-01-31"))
target_end_date = st.sidebar.date_input("回測結束日期", pd.to_datetime("2026-12-31"))

run_btn = st.sidebar.button("🚀 開始執行預測與策略回測")

if "stock_prediction_executed" not in st.session_state:
    st.session_state.stock_prediction_executed = False

if run_btn:
    with st.spinner(f"正在同步公開總經數據與 {ticker_symbol} 股價特徵..."):
        try:
            # 直接透過 FRED 公開 CSV 介面讀取高頻總經數據（免 API Key，永不失效）
            def fetch_fred_series(series_id):
                url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
                df = pd.read_csv(url, index_col=0, parse_dates=True)
                df.columns = [series_id]
                df[series_id] = pd.to_numeric(df[series_id], errors='coerce')
                return df

            napm = fetch_fred_series('NAPM')          # ISM 製造業 PMI
            retail = fetch_fred_series('RSXFS')       # 零售銷售
            indpro = fetch_fred_series('INDPRO')      # 工業生產指數
            cpi = fetch_fred_series('CPIAUCSL')       # 消費者物價指數 (通膨)

            macro_raw = pd.DataFrame({
                'ISM_PMI': napm['NAPM'],
                'Retail_Sales': retail['RSXFS'],
                'Industrial_Production': indpro['INDPRO'],
                'CPI': cpi['CPIAUCSL']
            })
            macro_raw['Retail_MoM'] = macro_raw['Retail_Sales'].pct_change() * 100
            macro_raw['IndPro_MoM'] = macro_raw['Industrial_Production'].pct_change() * 100
            macro_raw['CPI_YoY'] = macro_raw['CPI'].pct_change(12) * 100

        except Exception as e:
            st.error(f"❌ 獲取總經數據失敗: {e}")
            st.stop()

        try:
            session = Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            
            tickers = [ticker_symbol, "^GSPC", "^VIX"]
            df_raw = yf.download(tickers, start="2010-01-01", progress=False, session=session)
            df_prices = df_raw["Adj Close"] if "Adj Close" in df_raw.columns else df_raw["Close"]
            
            if isinstance(df_prices, pd.Series):
                df_prices = df_prices.to_frame(name=ticker_symbol)
                df_prices["^GSPC"] = df_prices[ticker_symbol]
                df_prices["^VIX"] = 20.0
            
            if ticker_symbol not in df_prices.columns:
                st.error(f"❌ 找不到代號 {ticker_symbol} 的數據，請確認代號是否正確。")
                st.stop()
                
            stock_prices = df_prices[ticker_symbol]
            sp500_prices = df_prices["^GSPC"] if "^GSPC" in df_prices.columns else stock_prices
            vix_series = df_prices["^VIX"] if "^VIX" in df_prices.columns else pd.Series(20.0, index=df_prices.index)

            def calculate_rsi(series, period=14):
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss
                return 100 - (100 / (1 + rs))

            def calculate_macd_diff(series, fast=12, slow=26, signal=9):
                exp1 = series.ewm(span=fast, adjust=False).mean()
                exp2 = series.ewm(span=slow, adjust=False).mean()
                macd = exp1 - exp2
                signal_line = macd.ewm(span=signal, adjust=False).mean()
                return macd - signal_line

            daily_df = pd.DataFrame(index=stock_prices.index)
            daily_df["Close"] = stock_prices
            daily_df["RSI_14"] = calculate_rsi(stock_prices, 14)
            daily_df["MACD_Diff"] = calculate_macd_diff(stock_prices)
            daily_df["MA20_Ratio"] = stock_prices / stock_prices.rolling(20).mean() - 1.0
            daily_df["MA60_Ratio"] = stock_prices / stock_prices.rolling(60).mean() - 1.0
            daily_df["Volatility"] = stock_prices.pct_change().rolling(20).std() * np.sqrt(252)

            df_m = daily_df.resample("ME").last().ffill()
            prices_m = stock_prices.resample("ME").last().ffill()
            sp500_m = sp500_prices.resample("ME").last().ffill()
            vix_m = vix_series.resample("ME").last().ffill()

            macro_df = pd.DataFrame(index=df_m.index)
            macro_df["Close"] = df_m["Close"]
            macro_df["RSI_14"] = df_m["RSI_14"]
            macro_df["MACD_Diff"] = df_m["MACD_Diff"]
            macro_df["MA20_Ratio"] = df_m["MA20_Ratio"]
            macro_df["MA60_Ratio"] = df_m["MA60_Ratio"]
            macro_df["Volatility"] = df_m["Volatility"]
            macro_df["VIX"] = vix_m

            # 整合每月高頻總經指標
            macro_df["ISM_PMI"] = macro_raw["ISM_PMI"].reindex(macro_df.index).ffill()
            macro_df["Retail_MoM"] = macro_raw["Retail_MoM"].reindex(macro_df.index).ffill()
            macro_df["IndPro_MoM"] = macro_raw["IndPro_MoM"].reindex(macro_df.index).ffill()
            macro_df["CPI_YoY"] = macro_raw["CPI_YoY"].reindex(macro_df.index).ffill()

            macro_df["Stock_Mom"] = prices_m.pct_change(momentum_window) * 100
            macro_df["Relative_Mom"] = (prices_m.pct_change(momentum_window) - sp500_m.pct_change(momentum_window)) * 100

            macro_df["Future_Close"] = macro_df["Close"].shift(-forecast_horizon)
            macro_df["Target_Return"] = (macro_df["Future_Close"] / macro_df["Close"] - 1.0) * 100

            feature_cols = [
                "ISM_PMI", "Retail_MoM", "IndPro_MoM", "CPI_YoY",
                "Stock_Mom", "Relative_Mom", "RSI_14", "MACD_Diff", "MA20_Ratio", "Volatility", "VIX"
            ]

            dates = macro_df.index.sort_values()
            detailed_records = []

            start_idx = max(train_window, momentum_window, 12)
            if start_idx >= len(dates) - forecast_horizon:
                start_idx = max(12, len(dates) // 2)

            for t in range(start_idx, len(dates)):
                test_date = dates[t]
                test_row = macro_df.iloc[t]
                target_forecast_date = test_date + pd.offsets.MonthEnd(forecast_horizon)

                feat_values = test_row[feature_cols].values
                has_missing = pd.isna(feat_values).any()

                pred_return = np.nan
                impacts = [np.nan] * len(feature_cols)

                if not has_missing:
                    train_subset = macro_df.iloc[:t].dropna(subset=feature_cols + ["Target_Return"])
                    if len(train_subset) >= 12:
                        if len(train_subset) > train_window:
                            train_subset = train_subset.iloc[-train_window:]

                        X_tr = train_subset[feature_cols]
                        y_tr = train_subset["Target_Return"]
                        X_te = test_row[feature_cols].values.reshape(1, -1)

                        scaler = StandardScaler()
                        X_tr_scaled = scaler.fit_transform(X_tr)
                        X_te_scaled = scaler.transform(X_te)

                        if HAS_LGB:
                            model = lgb.LGBMRegressor(n_estimators=50, learning_rate=0.05, max_depth=3, random_state=42, verbose=-1)
                            model.fit(X_tr_scaled, y_tr)
                            pred_return = model.predict(X_te_scaled)[0]
                            if hasattr(model, "feature_importances_"):
                                impacts = X_te_scaled[0] * (model.feature_importances_ / model.feature_importances_.sum())
                        else:
                            model = RidgeCV(alphas=np.logspace(-2, 4, 10)).fit(X_tr_scaled, y_tr)
                            pred_return = model.predict(X_te_scaled)[0]
                            impacts = X_te_scaled[0] * model.coef_

                current_price = test_row["Close"]
                predicted_price = current_price * (1.0 + pred_return / 100.0) if not np.isnan(pred_return) else np.nan

                record = {
                    "Feature_Date": test_date,
                    "Target_Date": target_forecast_date,
                    "Current_Price": current_price,
                    "Actual_Future_Price": test_row["Future_Close"],
                    "Actual_Return": test_row["Target_Return"],
                    "Predicted_Return": pred_return,
                    "Predicted_Price": predicted_price,
                }
                for i, col in enumerate(feature_cols):
                    record[f"{col}_Value"] = test_row[col]
                    record[f"{col}_Impact"] = impacts[i]

                detailed_records.append(record)

            results_df = pd.DataFrame(detailed_records)
            if not results_df.empty:
                results_df["Feature_Date"] = pd.to_datetime(results_df["Feature_Date"])
                results_df = results_df.set_index("Feature_Date")
                results_df = results_df.loc[pd.to_datetime(target_start_date):pd.to_datetime(target_end_date) + pd.DateOffset(months=forecast_horizon)]

            if results_df.empty:
                st.error("在您設定的回測期間內沒有足夠的資料，請將回測開始日期調早！")
            else:
                st.session_state.stock_prediction_executed = True
                st.session_state.results_df = results_df
                st.session_state.feature_cols = feature_cols
                st.session_state.forecast_horizon = forecast_horizon
                st.session_state.strategy_mode = strategy_mode
                st.session_state.buy_threshold = buy_threshold
                st.session_state.momentum_window = momentum_window
                st.session_state.ticker_symbol = ticker_symbol

        except Exception as e:
            st.error(f"❌ 執行過程發生錯誤: {e}")

if st.session_state.get("stock_prediction_executed", False):
    results_df = st.session_state.get("results_df")
    feature_cols = st.session_state.get("feature_cols")
    horizon_val = st.session_state.get("forecast_horizon", 3)
    mode_val = st.session_state.get("strategy_mode")
    thresh_val = st.session_state.get("buy_threshold", 1.0)
    mom_val = st.session_state.get("momentum_window", 12)
    t_sym = st.session_state.get("ticker_symbol", "AAPL")
    
    if results_df is not None and not results_df.empty:
        st.markdown(f'<div class="section-header">📈 {t_sym} 股價評估：實際價格 vs 模型建議價（預測期：+{horizon_val}M）</div>', unsafe_allow_html=True)
        
        valid_pred_rows = results_df.dropna(subset=["Predicted_Price"])
        if not valid_pred_rows.empty:
            latest_row = valid_pred_rows.iloc[-1]
            latest_feature_date = valid_pred_rows.index[-1].strftime('%Y-%m-%d')
            latest_target_date = (valid_pred_rows.index[-1] + pd.DateOffset(months=horizon_val)).strftime('%Y-%m-%d')
            
            curr_p = latest_row["Current_Price"]
            pred_p = latest_row["Predicted_Price"]
            pred_ret = latest_row["Predicted_Return"]

            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric(f"最新基準收盤價 ({t_sym})", f"${curr_p:.2f}", f"資料日: {latest_feature_date}")
            mcol2.metric(f"模型建議目標價 (+{horizon_val}M)", f"${pred_p:.2f}", f"{pred_ret:+.2f}% 預期漲跌")
            mcol3.metric("目標結算日", latest_target_date, f"模型: {'LightGBM' if HAS_LGB else 'RidgeCV'}")

        # -------------------------------------------------------------
        # 📊 互動式圖表
        # -------------------------------------------------------------
        valid_chart_df = results_df.dropna(subset=["Predicted_Price", "Current_Price"]).copy()
        if not valid_chart_df.empty:
            if "現金" in mode_val:
                valid_chart_df["Signal"] = np.where(valid_chart_df["Predicted_Return"] > thresh_val, 1, 0)
            else:
                valid_chart_df["Signal"] = np.where(valid_chart_df["Predicted_Return"] > thresh_val, 1, np.where(valid_chart_df["Predicted_Return"] < -thresh_val, -1, 0))
            
            fig = go.Figure()

            fig.add_trace(go.Scatter(
                x=valid_chart_df["Target_Date"], y=valid_chart_df["Current_Price"],
                mode='lines', name='實際股價 (Actual Price)',
                line=dict(color='#00d2ff', width=2.5)
            ))

            fig.add_trace(go.Scatter(
                x=valid_chart_df["Target_Date"], y=valid_chart_df["Predicted_Price"],
                mode='lines', name=f'模型建議價 (+{horizon_val}M)',
                line=dict(color='#ff9900', width=2, dash='dot')
            ))

            buy_df = valid_chart_df[valid_chart_df["Signal"] == 1]
            if not buy_df.empty:
                fig.add_trace(go.Scatter(
                    x=buy_df["Target_Date"], y=buy_df["Current_Price"],
                    mode='markers', name='🟢 買入/持有',
                    marker=dict(color='#00ff66', size=9, symbol='triangle-up')
                ))

            cash_df = valid_chart_df[valid_chart_df["Signal"] == 0]
            if not cash_df.empty:
                fig.add_trace(go.Scatter(
                    x=cash_df["Target_Date"], y=cash_df["Current_Price"],
                    mode='markers', name='🟡 觀望/現金',
                    marker=dict(color='#ffcc00', size=8, symbol='square')
                ))

            sell_df = valid_chart_df[valid_chart_df["Signal"] == -1]
            if not sell_df.empty:
                fig.add_trace(go.Scatter(
                    x=sell_df["Target_Date"], y=sell_df["Current_Price"],
                    mode='markers', name='🔴 放空/反向',
                    marker=dict(color='#ff3333', size=9, symbol='triangle-down')
                ))

            fig.update_layout(
                xaxis=dict(title="日期", gridcolor='#222629'),
                yaxis=dict(title="股價 ($)", gridcolor='#222629'),
                paper_bgcolor='#0e1117',
                plot_bgcolor='#0e1117',
                font=dict(color='white'),
                legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5, bgcolor='rgba(0,0,0,0)'),
                margin=dict(l=40, r=40, t=20, b=80)
            )

            st.plotly_chart(fig, use_container_width=True)

        # -------------------------------------------------------------
        # 📊 勝率與期望值摘要
        # -------------------------------------------------------------
        valid_eval = results_df.dropna(subset=["Actual_Return", "Predicted_Return"]).copy()
        if not valid_eval.empty:
            valid_eval["Pred_Dir"] = np.where(valid_eval["Predicted_Return"] > thresh_val, 1, -1)
            valid_eval["Actual_Dir"] = np.where(valid_eval["Actual_Return"] > 0, 1, -1)
            valid_eval["Is_Correct"] = valid_eval["Pred_Dir"] == valid_eval["Actual_Dir"]

            win_rate = valid_eval["Is_Correct"].mean()
            total_trades = len(valid_eval)
            correct_trades = valid_eval["Is_Correct"].sum()

            winning_moves = valid_eval.loc[valid_eval["Is_Correct"], "Actual_Return"].abs()
            losing_moves = valid_eval.loc[~valid_eval["Is_Correct"], "Actual_Return"].abs()
            avg_win_size = winning_moves.mean() if not winning_moves.empty else 0.0
            avg_loss_size = losing_moves.mean() if not losing_moves.empty else 0.0
            expectancy = (win_rate * avg_win_size) - ((1.0 - win_rate) * avg_loss_size)
        else:
            win_rate, total_trades, correct_trades, expectancy = 0, 0, 0, 0

        st.markdown(f'<div class="section-header">📊 模型預測勝率與期望值摘要 (門檻 > {thresh_val}%)</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("方向預測勝率 (Win Rate)", f"{win_rate * 100:.1f}%", f"{correct_trades}/{total_trades} 次正確")
        col2.metric("預測期望值 (Expectancy)", f"{expectancy:+.2f}%", "每次預測淨期望報酬")
        col3.metric("模型評估指標", "LightGBM / RidgeCV", f"門檻設定: {thresh_val}%")

        # -------------------------------------------------------------
        # 💰 策略回測與淨值曲線
        # -------------------------------------------------------------
        st.markdown(f'<div class="section-header">💰 策略模擬交易淨值曲線 vs 買進持有 [{mode_val}]</div>', unsafe_allow_html=True)
        
        backtest_df = results_df.dropna(subset=["Predicted_Return", "Current_Price"]).copy()
        if not backtest_df.empty:
            stock_ret = backtest_df["Current_Price"].pct_change()

            if "現金" in mode_val:
                backtest_df["Signal"] = np.where(backtest_df["Predicted_Return"] > thresh_val, 1, 0)
                backtest_df["Strategy_Return"] = backtest_df["Signal"].shift(1) * stock_ret
            else:
                backtest_df["Signal"] = np.where(backtest_df["Predicted_Return"] > thresh_val, 1, np.where(backtest_df["Predicted_Return"] < -thresh_val, -1, 0))
                backtest_df["Strategy_Return"] = backtest_df["Signal"].shift(1) * stock_ret

            backtest_df["Strategy_Return"] = backtest_df["Strategy_Return"].fillna(0)
            backtest_df["Benchmark_Nav"] = (1.0 + stock_ret.fillna(0)).cumprod()
            backtest_df["Strategy_Nav"] = (1.0 + backtest_df["Strategy_Return"]).cumprod()

            total_days = (backtest_df.index[-1] - backtest_df.index[0]).days
            years = max(total_days / 365.25, 0.5)
            
            strat_total_return = backtest_df["Strategy_Nav"].iloc[-1] - 1.0
            strat_cagr = (backtest_df["Strategy_Nav"].iloc[-1] ** (1 / years)) - 1.0
            
            bench_total_return = backtest_df["Benchmark_Nav"].iloc[-1] - 1.0
            bench_cagr = (backtest_df["Benchmark_Nav"].iloc[-1] ** (1 / years)) - 1.0

            strat_rolling_max = backtest_df["Strategy_Nav"].cummax()
            strat_drawdown = (backtest_df["Strategy_Nav"] - strat_rolling_max) / strat_rolling_max
            strat_mdd = strat_drawdown.min()

            bench_rolling_max = backtest_df["Benchmark_Nav"].cummax()
            bench_drawdown = (backtest_df["Benchmark_Nav"] - bench_rolling_max) / bench_rolling_max
            bench_mdd = bench_drawdown.min()

            pcol1, pcol2, pcol3, pcol4 = st.columns(4)
            pcol1.metric("策略年化報酬率 (CAGR)", f"{strat_cagr * 100:.2f}%", f"買進持有: {bench_cagr * 100:.2f}%")
            pcol2.metric("策略總報酬率", f"{strat_total_return * 100:.2f}%", f"買進持有: {bench_total_return * 100:.2f}%")
            pcol3.metric("策略最大回落 (MDD)", f"{strat_mdd * 100:.2f}%", f"買進持有: {bench_mdd * 100:.2f}%")
            pcol4.metric("回測期間", f"{years:.1f} 年", f"{len(backtest_df)} 個交易點")

            nav_chart_df = backtest_df[["Strategy_Nav", "Benchmark_Nav"]].copy()
            nav_chart_df.columns = [f"策略淨值曲線 ({mode_val})", f"{t_sym} 買進持有 (Benchmark)"]
            st.line_chart(nav_chart_df)
        else:
            st.warning("⚠️ 目前回測期間資料不足，無法計算策略績效。")

        # -------------------------------------------------------------
        # 📅 每月預測明細表
        # -------------------------------------------------------------
        st.markdown(f'<div class="section-header">📅 每月預測與模擬交易訊號明細表</div>', unsafe_allow_html=True)
        
        show_table_df = results_df[[
            "Target_Date", "Current_Price", "Predicted_Price", "Predicted_Return", "Actual_Return"
        ]].copy()

        show_table_df["預測看漲超標"] = show_table_df["Predicted_Return"] > thresh_val
        if "現金" in mode_val:
            show_table_df["策略訊號動作"] = np.where(show_table_df["預測看漲超標"], "🟢 買入並持有", "🔴 平倉 / 持有現金")
        else:
            show_table_df["策略訊號動作"] = np.where(
                show_table_df["Predicted_Return"] > thresh_val, "🟢 做多 (Long)",
                np.where(show_table_df["Predicted_Return"] < -thresh_val, "🔴 放空 (Short)", "🟡 觀望 / 現金")
            )

        show_table_df["Pred_Dir"] = show_table_df["Predicted_Return"] > thresh_val
        show_table_df["Actual_Dir"] = show_table_df["Actual_Return"] > 0
        show_table_df["方向勝率判斷"] = np.where(
            show_table_df["Actual_Return"].isna() | show_table_df["Predicted_Return"].isna(),
            "⏳ 最新即時預測 (待揭曉)",
            np.where(show_table_df["Pred_Dir"] == show_table_df["Actual_Dir"], "✅ 正確 (Hit)", "❌ 錯誤 (Miss)")
        )
        
        show_table_df["Target_Date"] = pd.to_datetime(show_table_df["Target_Date"]).dt.strftime("%Y-%m-%d")
        show_table_df.index = show_table_df.index.strftime("%Y-%m-%d")
        
        final_display_df = show_table_df[[
            "Target_Date", "Current_Price", "Predicted_Price", "Predicted_Return", "Actual_Return", "策略訊號動作", "方向預測結果"
        ]].copy()

        final_display_df.columns = [
            f"預測目標月份 (+{horizon_val}M)", "當月基準價 ($)", "模型建議價 ($)", "預測報酬率 (%)", "實際未來報酬 (%)", "策略訊號動作", "方向預測結果"
        ]
        
        final_display_df = final_display_df.sort_index(ascending=False)
        st.dataframe(final_display_df.round(2), use_container_width=True)
