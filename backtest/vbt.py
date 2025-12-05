"""
双均线策略 - Vectorbt 版本
向量化回测，快速高效
"""

import sqlite3
import pandas as pd
import numpy as np
import vectorbt as vbt


def load_data(db_path: str, ticker: str) -> pd.DataFrame:
    """从数据库加载数据"""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(
        "SELECT * FROM ohlcv WHERE ticker = ? ORDER BY date",
        conn,
        params=(ticker,),
    )
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算 ATR"""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def run_backtest(
    df: pd.DataFrame,
    ma_fast: int = 20,
    ma_slow: int = 60,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    init_cash: float = 100000,
):
    """
    运行 vectorbt 回测
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    
    # 计算指标
    ma_f = close.rolling(ma_fast).mean()
    ma_s = close.rolling(ma_slow).mean()
    atr = calc_atr(high, low, close, atr_period)
    
    # 趋势判断
    uptrend = ma_f > ma_s
    
    # 入场信号：多头趋势 + 收盘价从下往上突破 MA20
    cross_above_ma = (close.shift(1) <= ma_f.shift(1)) & (close > ma_f)
    entries = uptrend & cross_above_ma
    
    # 出场信号：趋势反转（MA20 下穿 MA60）
    trend_reversal = uptrend.shift(1) & ~uptrend
    exits = trend_reversal
    
    # 止损比例（基于 ATR）
    sl_stop = (atr_multiplier * atr / close).fillna(0.02)
    
    # 运行回测
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        sl_stop=sl_stop,
        init_cash=init_cash,
        fees=0.001,  # 0.1% 手续费
        slippage=0.001,  # 0.1% 滑点
        freq="1D",
    )
    
    return pf


def print_stats(pf, ticker: str):
    """打印关键指标"""
    stats = pf.stats()
    trades = pf.trades.records_readable
    
    # 兼容不同版本的 key 名称
    def get_stat(keys):
        """尝试多个可能的 key 名称"""
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            if k in stats.index:
                return stats[k]
        return None
    
    total_return = get_stat(["Total Return [%]", "total_return"])
    annual_return = get_stat(["Annualized Return [%]", "annualized_return", "Ann. Return [%]"])
    sharpe = get_stat(["Sharpe Ratio", "sharpe_ratio"])
    max_dd = get_stat(["Max Drawdown [%]", "max_drawdown", "Max DD [%]"])
    total_trades = get_stat(["Total Trades", "total_trades"])
    win_rate = get_stat(["Win Rate [%]", "win_rate"])
    
    # 计算盈亏比
    profit_factor = 0
    if len(trades) > 0:
        pnl_col = "PnL" if "PnL" in trades.columns else "pnl"
        if pnl_col in trades.columns:
            wins = trades[trades[pnl_col] > 0][pnl_col]
            losses = trades[trades[pnl_col] < 0][pnl_col]
            avg_win = wins.mean() if len(wins) > 0 else 0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
            profit_factor = avg_win / avg_loss if avg_loss > 0 else float("inf")
    
    print(f"\n{'='*50}")
    print(f"{ticker} 回测结果")
    print(f"{'='*50}")
    print(f"总收益率:     {total_return:.2f}%" if total_return else "总收益率:     N/A")
    print(f"年化收益率:   {annual_return:.2f}%" if annual_return else "年化收益率:   N/A")
    print(f"夏普比率:     {sharpe:.2f}" if sharpe else "夏普比率:     N/A")
    print(f"最大回撤:     {max_dd:.2f}%" if max_dd else "最大回撤:     N/A")
    print(f"{'='*50}")
    print(f"交易次数:     {int(total_trades)}" if total_trades else "交易次数:     N/A")
    print(f"胜率:         {win_rate:.1f}%" if win_rate else "胜率:         N/A")
    print(f"盈亏比:       {profit_factor:.2f}")
    
    return stats

def plot_equity_curve(pf, ticker: str, save_path: str = None):
    """绘制资金曲线"""
    import matplotlib.pyplot as plt
    
    equity = pf.value()
    drawdown = pf.drawdown() * 100
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # 资金曲线
    axes[0].plot(equity.index, equity.values, label="Portfolio Value", color="blue")
    axes[0].set_ylabel("Value ($)")
    axes[0].set_title(f"{ticker} - Equity Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 回撤
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.3)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].set_xlabel("Date")
    axes[1].set_title("Drawdown")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"图表已保存: {save_path}")
    
    plt.show()
    return fig


def main(db_path: str = "stock_data.db", tickers: list = None):
    """主函数"""
    if tickers is None:
        tickers = ["QQQ"]
    
    results = {}
    
    for ticker in tickers:
        print(f"\n回测 {ticker}...")
        
        df = load_data(db_path, ticker)
        if df.empty:
            print(f"  {ticker} 无数据")
            continue
        
        pf = run_backtest(df)
        stats = print_stats(pf, ticker)
        results[ticker] = {"portfolio": pf, "stats": stats}

        plot_equity_curve(pf, ticker, f"equity_curve_{ticker}.png")
        
    return results


if __name__ == "__main__":
    results = main(tickers=["QQQ"])
    
    # 可视化（如果在 notebook 环境）
    # results["QQQ"]["portfolio"].plot().show()