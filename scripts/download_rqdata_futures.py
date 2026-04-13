from __future__ import annotations

from pathlib import Path

import pandas as pd
import rqdatac


def main() -> None:
    out_dir = Path("data/cache/raw_rqdata")
    out_dir.mkdir(parents=True, exist_ok=True)

    # 你自己的 RQData 账号密码
    rqdatac.init('+8618049006912', 'Katrina0504.')

    # 这里只是示例；你需要按 RQData 实际可用标识调整
    symbols = {
        "OI99": "SHFE_cu1901.csv"
    }

    start_date = "2018-01-01"
    end_date = "2026-01-01"

    for rq_symbol, filename in symbols.items():
        print(f"downloading {rq_symbol} ...")

        # 这里按 RQData 常见 get_price 风格写
        df = rqdatac.get_price(
            order_book_ids=rq_symbol,
            start_date=start_date,
            end_date=end_date,
            frequency="1d",
            fields=["open", "high", "low", "close", "volume", "open_interest"],
            adjust_type="none",
        )

        # 某些情况下返回 index 为日期
        df.to_csv(out_dir / filename)
        print(f"saved -> {out_dir / filename}")


if __name__ == "__main__":
    main()