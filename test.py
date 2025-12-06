from backtest.vbt import main
trades = main(tickers=["META"],start_date="2025-01-01",end_date="2025-09-30")
print(trades)