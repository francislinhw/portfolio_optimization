import pandas as pd

now_market_data = pd.DataFrame(
    [
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 800.0,
            "market_iv": 41.67 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 805.0,
            "market_iv": 41.57 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 810.0,
            "market_iv": 41.49 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 815.0,
            "market_iv": 41.43 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 820.0,
            "market_iv": 41.37 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 825.0,
            "market_iv": 41.32 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 830.0,
            "market_iv": 41.28 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 835.0,
            "market_iv": 41.25 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 840.0,
            "market_iv": 41.22 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-25"),
            "strike": 845.0,
            "market_iv": 41.19 / 100,
        },
    ]
)
