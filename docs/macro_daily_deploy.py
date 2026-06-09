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
    
    series_map = {
        "DGS10": "us10y_yield",
        "T10Y2Y": "us_yield_curve_spread",
        "VIXCLS": "vix_close",
        "NASDAQ100": "nq_futures_close",
        "DTWEXBGS": "dxy_close"
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
    
    # 连续日历填充与变化率计算
    all_days = pd.date_range(start=fetch_start, end=end_date, freq='D')
    df_continuous = df_raw.reindex(all_days).ffill()
    
    df_continuous['vix_pct_change'] = df_continuous['vix_close'].pct_change()
    df_continuous['nq_futures_pct_change'] = df_continuous['nq_futures_close'].pct_change()
    
    # 核心映射：向未来推移一天
    df_shifted = df_continuous.shift(1, freq='D')
    
    # 获取目标区间内真实的 NYSE 交易日
    nyse = mcal.get_calendar('NYSE')
    schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    target_trading_days = mcal.date_range(schedule, frequency='1D').tz_localize(None).normalize()
    
    # 生成最终的特征表
    df_final = pd.DataFrame(index=target_trading_days)
    df_final.index.name = 'trade_date'
    df_final = df_final.join(df_shifted)
    
    if df_final.isnull().values.any():
        raise ValueError("Fatality: Data contains NaNs. Source data might be incomplete.")
        
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