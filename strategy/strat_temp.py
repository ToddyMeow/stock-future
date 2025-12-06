"""
双均线策略
- 趋势: MA20 > MA60 = 多头
- 入场: 收盘价突破 MA20
- 止损: 跌破入场价 - 2*ATR
- 止盈: MA20 下穿 MA60
"""

import sqlite3
import pandas as pd


def load_data(db_path: str, ticker: str) -> pd.DataFrame:
    """从数据库加载单只股票数据"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df


def calc_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """计算 ATR"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    return tr.rolling(period).mean()


def run_backtest(
    df: pd.DataFrame,
    ma_fast: int = 5,
    ma_slow: int = 20,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
) -> pd.DataFrame:
    """
    运行回测
    
    Returns:
        包含交易记录的 DataFrame
    """
    # 计算指标
    df = df.copy()
    df["ma_fast"] = df["close"].rolling(ma_fast).mean()
    df["ma_slow"] = df["close"].rolling(ma_slow).mean()
    df["atr"] = calc_atr(df, atr_period)
    
    # 趋势判断
    df["uptrend"] = df["ma_fast"] > df["ma_slow"]
    
    # 回测逻辑
    trades = []
    position = None  # {"entry_date", "entry_price", "stop_loss"}
    
    for i in range(ma_slow, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        
        # 无持仓：检查入场
        if position is None:
            # 多头趋势 + 收盘价突破MA20
            if row["uptrend"] and prev["close"] <= prev["ma_fast"] < row["close"]:
                position = {
                    "entry_date": row["date"],
                    "entry_price": row["close"],
                    "stop_loss": row["close"] - atr_multiplier * row["atr"],
                }
        
        # 有持仓：检查出场
        else:
            exit_reason = None
            
            # 止损：价格跌破止损线
            if row["low"] <= position["stop_loss"]:
                exit_reason = "止损"
                exit_price = position["stop_loss"]
            
            # 止盈：趋势反转（MA20下穿MA60）
            elif prev["uptrend"] and not row["uptrend"]:
                exit_reason = "趋势反转"
                exit_price = row["close"]
            
            if exit_reason:
                pnl = (exit_price - position["entry_price"]) / position["entry_price"]
                trades.append({
                    "entry_date": position["entry_date"],
                    "entry_price": position["entry_price"],
                    "exit_date": row["date"],
                    "exit_price": exit_price,
                    "pnl_pct": pnl * 100,
                    "reason": exit_reason,
                })
                position = None
    
    return pd.DataFrame(trades)


def print_stats(trades: pd.DataFrame, ticker: str):
    """打印统计"""
    if trades.empty:
        print(f"{ticker}: 无交易")
        return
    
    wins = trades[trades["pnl_pct"] > 0]
    
    print(f"\n{'='*40}")
    print(f"{ticker} 回测结果")
    print(f"{'='*40}")
    print(f"交易次数: {len(trades)}")
    print(f"胜率: {len(wins)/len(trades)*100:.1f}%")
    print(f"平均盈亏: {trades['pnl_pct'].mean():.2f}%")
    print(f"最大盈利: {trades['pnl_pct'].max():.2f}%")
    print(f"最大亏损: {trades['pnl_pct'].min():.2f}%")
    print(f"累计收益: {trades['pnl_pct'].sum():.2f}%")


def main(db_path: str = "stock_data.db", tickers: list = None):
    """主函数"""
    if tickers is None:
        # 默认测试 QQQ
        tickers = ["QQQ"]
    
    all_trades = []
    
    for ticker in tickers:
        print(f"\n回测 {ticker}...")
        
        df = load_data(db_path, ticker)
        if df.empty:
            print(f"  {ticker} 无数据")
            continue
        
        trades = run_backtest(df)
        trades["ticker"] = ticker
        all_trades.append(trades)
        
        print_stats(trades, ticker)
    
    if all_trades:
        return pd.concat(all_trades, ignore_index=True)
    return pd.DataFrame()


if __name__ == "__main__":
    trades = main()
    if not trades.empty:
        print(f"\n所有交易明细:")
        print(trades.to_string(index=False))