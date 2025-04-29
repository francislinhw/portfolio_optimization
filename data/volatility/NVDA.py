import pandas as pd

# Constructing NVDA market_data from Bloomberg screen (27 Jul 2025 expiry, strikes from 96 to 110)
# Data approximated from implied volatilities seen in the image
nvda_market_data = pd.DataFrame(
    [
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 96.0,
            "market_iv": 0.5076,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 97.0,
            "market_iv": 0.5040,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 98.0,
            "market_iv": 0.5003,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 99.0,
            "market_iv": 0.4969,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 100.0,
            "market_iv": 0.4932,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 101.0,
            "market_iv": 0.4901,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 102.0,
            "market_iv": 0.4869,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 103.0,
            "market_iv": 0.4837,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 104.0,
            "market_iv": 0.4806,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 105.0,
            "market_iv": 0.4777,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 106.0,
            "market_iv": 0.4751,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 107.0,
            "market_iv": 0.4721,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 108.0,
            "market_iv": 0.4692,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 109.0,
            "market_iv": 0.4662,
        },
        {
            "expiry_date": pd.to_datetime("2025-07-27"),
            "strike": 110.0,
            "market_iv": 0.4634,
        },
    ]
)
