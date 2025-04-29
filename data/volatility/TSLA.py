import pandas as pd
from datetime import datetime

# Constructing the market data for TSLA based on the image (example selection)
tsla_market_data = pd.DataFrame(
    [
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 245.0,
            "market_iv": 70.94 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 247.5,
            "market_iv": 70.25 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 250.0,
            "market_iv": 69.44 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 252.5,
            "market_iv": 68.65 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 255.0,
            "market_iv": 67.86 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 257.5,
            "market_iv": 67.08 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 260.0,
            "market_iv": 66.31 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 262.5,
            "market_iv": 65.55 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 265.0,
            "market_iv": 64.80 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 267.5,
            "market_iv": 64.06 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 270.0,
            "market_iv": 63.33 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 272.5,
            "market_iv": 62.61 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 275.0,
            "market_iv": 61.90 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 277.5,
            "market_iv": 61.20 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 280.0,
            "market_iv": 60.51 / 100,
        },
    ]
)
