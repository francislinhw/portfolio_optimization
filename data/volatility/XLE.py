from datetime import datetime
import pandas as pd

# Constructed market data for XLE as seen in the image (sample points)
xle_market_data = pd.DataFrame(
    [
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 81.0,
            "market_iv": 30.44 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 82.0,
            "market_iv": 30.15 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 83.0,
            "market_iv": 29.80 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 84.0,
            "market_iv": 29.52 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 85.0,
            "market_iv": 29.24 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 86.0,
            "market_iv": 28.96 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 87.0,
            "market_iv": 28.69 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 88.0,
            "market_iv": 28.42 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 89.0,
            "market_iv": 28.16 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 90.0,
            "market_iv": 27.91 / 100,
        },
    ]
)
