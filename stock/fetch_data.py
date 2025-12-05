"""
Stock Data Fetcher
- 获取 QQQ 和 Nasdaq-100 成分股数据
- 存入 SQLite，支持复权价格
- 数据源: Tiingo API
"""

import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

# 默认配置路径
CONFIG_PATH = Path(__file__).parent / "config.json"

# 默认成分股（按市值排序）
DEFAULT_NASDAQ100_TOP = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META",
    "GOOGL", "GOOG", "AVGO", "TSLA", "COST",
    "NFLX", "AMD", "QCOM", "ADBE", "PEP",
]


def load_config() -> dict:
    """加载配置文件，不存在则创建默认配置"""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return json.load(f)
    
    default_config = {
        "nasdaq100_top": DEFAULT_NASDAQ100_TOP,
        "index_etfs": ["QQQ", "SPY"],
        "years": 10,
        "db_path": "stock_data.db",
        "tiingo_api_key": "",
    }
    
    with open(CONFIG_PATH, "w") as f:
        json.dump(default_config, f, indent=2)
    
    print(f"已创建默认配置文件: {CONFIG_PATH}")
    return default_config


def get_tickers_from_config(top_n: int = 15) -> list[str]:
    """从配置文件获取成分股"""
    config = load_config()
    tickers = config.get("nasdaq100_top", DEFAULT_NASDAQ100_TOP)
    return tickers[:top_n]


def fetch_ohlcv_tiingo(
    tickers: list[str],
    api_key: str,
    start_date: str,
    end_date: str = None,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    从 Tiingo 获取股票 OHLCV 数据（已复权）
    
    Args:
        tickers: 股票代码列表
        api_key: Tiingo API key
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期，默认今天
        delay: 请求间隔秒数
    
    Returns:
        包含所有股票数据的 DataFrame
    """
    end_date = end_date or datetime.now().strftime("%Y-%m-%d")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}
    base_url = "https://api.tiingo.com/tiingo/daily/{}/prices"
    
    all_data = []
    failed = []
    
    for i, ticker in enumerate(tickers):
        if i > 0:
            time.sleep(delay)
        
        print(f"下载 {ticker} ({i+1}/{len(tickers)})...")
        
        try:
            url = base_url.format(ticker)
            params = {"startDate": start_date, "endDate": end_date}
            resp = requests.get(url, headers=headers, params=params, timeout=30)
            
            if resp.status_code == 404:
                print(f"  警告: {ticker} 未找到")
                failed.append(ticker)
                continue
            
            resp.raise_for_status()
            data = resp.json()
            
            if not data:
                print(f"  警告: {ticker} 无数据")
                failed.append(ticker)
                continue
            
            df = pd.DataFrame(data)
            # 只保留复权价格列，避免重复
            df = df[["date", "adjOpen", "adjHigh", "adjLow", "adjClose", "adjVolume"]]
            df.columns = ["date", "open", "high", "low", "close", "volume"]
            df["ticker"] = ticker
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            
            all_data.append(df)
            print(f"  成功: {len(df)} 条记录")
            
        except Exception as e:
            print(f"  错误: {ticker} - {e}")
            failed.append(ticker)
    
    if failed:
        print(f"\n失败: {failed}")
    
    if not all_data:
        raise ValueError("没有获取到任何数据")
    
    return pd.concat(all_data, ignore_index=True)


def save_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str = "ohlcv"):
    """保存数据到 SQLite"""
    # 标准化列名
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    
    # 确保日期格式正确
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    
    conn = sqlite3.connect(db_path)
    
    # 写入数据（替换已存在的表）
    df.to_sql(table_name, conn, if_exists="replace", index=False)
    
    # 创建索引加速查询
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_ticker ON {table_name}(ticker)")
    conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table_name}_date ON {table_name}(date)")
    
    conn.commit()
    conn.close()
    
    print(f"数据已保存到 {db_path}，表: {table_name}")


def main(
    tickers: list[str] = None,
    include_qqq: bool = True,
    top_n: int = 16,
    years: int = None,
    db_path: str = None,
    api_key: str = None,
):
    """
    主函数
    
    Args:
        tickers: 手动指定股票列表，None则从配置获取
        include_qqq: 是否包含 QQQ
        top_n: 获取前N只成分股
        years: 获取多少年的数据（None则从配置读取）
        db_path: SQLite 数据库路径（None则从配置读取）
        api_key: Tiingo API key（None则从配置读取）
    """
    config = load_config()
    
    # 使用配置或默认值
    years = years or config.get("years", 10)
    db_path = db_path or config.get("db_path", "stock_data.db")
    api_key = api_key or config.get("tiingo_api_key")
    
    if not api_key:
        raise ValueError("需要 Tiingo API key，请在 config.json 中设置 tiingo_api_key")
    
    # 获取股票列表
    if tickers is None:
        print("从配置文件获取股票列表...")
        tickers = get_tickers_from_config(top_n)
        print(f"成分股: {tickers}")
    
    if include_qqq:
        tickers = ["QQQ"] + [t for t in tickers if t != "QQQ"]
    
    # 计算日期范围
    end_date = datetime.now()
    start_date = end_date - timedelta(days=years * 365)
    
    # 获取数据
    print(f"\n获取 {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')} 的数据...")
    df = fetch_ohlcv_tiingo(
        tickers=tickers,
        api_key=api_key,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )
    
    # 保存到数据库
    save_to_sqlite(df, db_path)
    
    # 打印统计信息
    print(f"\n统计:")
    print(f"  股票数量: {df['ticker'].nunique()}")
    print(f"  总记录数: {len(df)}")
    print(f"  日期范围: {df['date'].min()} ~ {df['date'].max()}")
    
    return df


if __name__ == "__main__":
    main()