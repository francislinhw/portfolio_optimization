import pandas as pd

nflx_market_data = pd.DataFrame(
    [
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1030.0,
            "market_iv": 38.67 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1035.0,
            "market_iv": 38.41 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1040.0,
            "market_iv": 38.18 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1045.0,
            "market_iv": 37.97 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1050.0,
            "market_iv": 37.78 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1055.0,
            "market_iv": 37.62 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1060.0,
            "market_iv": 37.48 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1065.0,
            "market_iv": 37.35 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1070.0,
            "market_iv": 37.24 / 100,
        },
        {
            "expiry_date": pd.to_datetime("2025-08-22"),
            "strike": 1075.0,
            "market_iv": 37.15 / 100,
        },
    ]
)
