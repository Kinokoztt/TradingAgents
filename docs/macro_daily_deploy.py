import os
import pytz
import pandas as pd
import pandas_gbq
import functions_framework
from fredapi import Fred
from datetime import datetime, timedelta
import pandas_market_calendars as mcal

def process_fred_pipeline(start_date: str, end_date: str) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError("Fatality: FRED_API_KEY is missing.")
        
    fred = Fred(api_key=api_key)
    fetch_start = (pd.to_datetime(start_date) - timedelta(days=15)).strftime('%Y-%m-%d')
    
    # 列正名：DTWEXBGS 是美联储广义名义美元指数(~120)，并非 ICE DXY(~99)；
    # NASDAQ100 是现货指数，而非纳指期货。命名按实际口径对齐。
    series_map = {
        "DGS10": "us10y_yield",
        "T10Y2Y": "us_yield_curve_spread",
        "VIXCLS": "vix_close",
        "NASDAQ100": "nasdaq100_close",
        "DTWEXBGS": "usd_broad_index"
    }

    df_list = []
    for series_id, col_name in series_map.items():
        try:
            s = fred.get_series(series_id, observation_start=fetch_start, observation_end=end_date)
            df_temp = pd.DataFrame(s, columns=[col_name])
            df_list.append(df_temp)
        except Exception as e:
            raise ValueError(f"Fatality: Failed to fetch {series_id}: {e}")

    df_raw = pd.concat(df_list, axis=1)

    return _build_features(df_raw, fetch_start, start_date, end_date)


def _build_features(df_raw: pd.DataFrame, fetch_start: str, start_date: str, end_date: str) -> pd.DataFrame:
    # 以 NYSE 交易日历为对齐基准，而不是自然日 ffill：
    # 自然日 ffill 会把周末/节假日和"数据发布滞后"混为一谈，并在 ffill 后的序列上
    # 算 pct_change，导致真实跳变被周末填平的同值抹成 0%。
    nyse = mcal.get_calendar('NYSE')
    calendar_schedule = nyse.schedule(start_date=fetch_start, end_date=end_date)
    all_trading_days = mcal.date_range(calendar_schedule, frequency='1D').tz_localize(None).normalize()

    df_td = pd.DataFrame(index=all_trading_days)

    # 水平值：在交易日上前向填充。交易日历已剔除周末/节假日，ffill 只用于补齐
    # 个别缺测，并保留最近一次真实观测作为盘前最佳已知值。
    level_cols = ["us10y_yield", "us_yield_curve_spread", "vix_close", "nasdaq100_close", "usd_broad_index"]
    for col in level_cols:
        df_td[col] = df_raw[col].reindex(all_trading_days).ffill()

    # 变化率：直接在真实观测上计算 pct_change（dropna 后），再对齐到交易日。
    # 这样 d/d 变化只反映真实相邻观测之间的跳变，不会被 ffill 填平的行污染。
    df_td['vix_pct_change'] = df_raw['vix_close'].dropna().pct_change().reindex(all_trading_days).ffill()
    df_td['nasdaq100_pct_change'] = df_raw['nasdaq100_close'].dropna().pct_change().reindex(all_trading_days).ffill()

    # 陈旧度：DTWEXBGS(H.10) 发布有滞后，最近一两天往往没有真实观测。
    # 记录每个交易日距上一次真实观测的自然日数，区分"周末/节假日填充"与"数据滞后"，
    # 让下游(L3)知道当前广义美元指数是不是被一路 ffill 出来的陈旧值。
    usd_obs_dates = df_raw['usd_broad_index'].dropna().index.to_series()
    last_obs_per_td = usd_obs_dates.reindex(all_trading_days).ffill()
    df_td['usd_broad_index_stale_days'] = (df_td.index.to_series() - last_obs_per_td).dt.days

    # 核心映射：盘前快照取上一交易日的数据，避免前视偏差。
    df_shifted = df_td.shift(1)

    # 取目标区间内真实的 NYSE 交易日
    target_schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    target_trading_days = mcal.date_range(target_schedule, frequency='1D').tz_localize(None).normalize()

    df_final = df_shifted.reindex(target_trading_days)
    df_final.index.name = 'trade_date'

    if df_final.isnull().values.any():
        raise ValueError("Fatality: Data contains NaNs. Source data might be incomplete.")

    df_final['usd_broad_index_stale_days'] = df_final['usd_broad_index_stale_days'].astype(int)

    return df_final

@functions_framework.http
def handle_http_request(request):
    try:
        est_tz = pytz.timezone('US/Eastern')
        # 如果没有传入参数，默认获取当天的盘前环境
        today_est = datetime.now(est_tz).strftime('%Y-%m-%d')
        
        request_args = request.args if request.args else {}
        START_DATE = request_args.get("START_DATE", today_est)
        END_DATE = request_args.get("END_DATE", today_est)
        
        PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "mystockproject-431701")
        DATASET_ID = os.environ.get("BQ_DATASET_ID", "stock_dataset")
        TABLE_NAME = os.environ.get("BQ_TABLE_NAME", "macro_daily")
        
        print(f"Executing pure FRED pipeline for targets: {START_DATE} to {END_DATE}")
        
        df_final = process_fred_pipeline(START_DATE, END_DATE)
        df_final = df_final.reset_index()
        
        destination_table = f"{DATASET_ID}.{TABLE_NAME}"
        print(f"Pushing {len(df_final)} rows to BigQuery table {destination_table}...")
        
        pandas_gbq.to_gbq(
            df_final,
            destination_table,
            project_id=PROJECT_ID,
            if_exists='append',
            progress_bar=False
        )
        return "OK", 200
        
    except Exception as e:
        print(f"Pipeline Failed: {e}")
        return str(e), 500