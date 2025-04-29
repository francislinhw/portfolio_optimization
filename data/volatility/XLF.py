from datetime import datetime
import pandas as pd


# 根據圖片中 "27-JUL-2025" 到期日建立 market data
xlf_market_data = pd.DataFrame(
    [
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 47.5,
            "market_iv": 25.44 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 48.0,
            "market_iv": 25.34 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 48.5,
            "market_iv": 25.25 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 49.0,
            "market_iv": 25.16 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 49.5,
            "market_iv": 25.06 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 50.0,
            "market_iv": 24.96 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 50.5,
            "market_iv": 24.86 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 51.0,
            "market_iv": 24.75 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 51.5,
            "market_iv": 24.64 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 52.0,
            "market_iv": 24.52 / 100,
        },
    ]
)
