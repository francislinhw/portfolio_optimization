import QuantLib as ql
import numpy as np


def price_option_from_paths(S_paths, K, r, T, option_type="call"):
    if option_type == "call":
        payoff = np.maximum(S_paths[:, -1] - K, 0)
    else:
        payoff = np.maximum(K - S_paths[:, -1], 0)
    return np.exp(-r * T) * np.mean(payoff)


# === Create Market ===
def setup_market(spot_price, maturity_date, r, q, sigma):
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    today = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = today

    day_count = ql.Actual365Fixed()
    maturity = ql.Date(maturity_date.day, maturity_date.month, maturity_date.year)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))
    r_curve = ql.FlatForward(today, r, day_count)
    q_curve = ql.FlatForward(today, q, day_count)
    vol = ql.BlackConstantVol(today, calendar, sigma, day_count)

    return {
        "spot": spot_handle,
        "r": ql.YieldTermStructureHandle(r_curve),
        "q": ql.YieldTermStructureHandle(q_curve),
        "vol": ql.BlackVolTermStructureHandle(vol),
        "maturity": maturity,
    }


def create_option(strike, maturity, option_type):
    payoff = ql.PlainVanillaPayoff(option_type, strike)
    exercise = ql.EuropeanExercise(maturity)
    return ql.VanillaOption(payoff, exercise)
