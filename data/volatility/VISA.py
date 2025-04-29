from datetime import datetime
import pandas as pd

# Visa (V) market data extracted from the image
visa_market_data = pd.DataFrame(
    [
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 333.82,
            "market_iv": 29.30 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 330.0,
            "market_iv": 28.87 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 335.0,
            "market_iv": 29.44 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 340.0,
            "market_iv": 30.09 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 345.0,
            "market_iv": 30.66 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 350.0,
            "market_iv": 31.20 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 355.0,
            "market_iv": 31.73 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 360.0,
            "market_iv": 32.24 / 100,
        },
        {
            "expiry_date": datetime(2025, 7, 27),
            "strike": 365.0,
            "market_iv": 32.75 / 100,
        },
    ]
)
