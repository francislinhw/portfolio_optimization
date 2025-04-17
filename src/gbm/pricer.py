from src.market_setup import setup_market, create_option
import QuantLib as ql


def price_bs(spot, strike, maturity_date, r=0.05, q=0.0, sigma=0.2):
    env = setup_market(spot, maturity_date, r, q, sigma)
    process = ql.BlackScholesMertonProcess(env["spot"], env["q"], env["r"], env["vol"])

    engine = ql.AnalyticEuropeanEngine(process)

    call = create_option(strike, env["maturity"], ql.Option.Call)
    put = create_option(strike, env["maturity"], ql.Option.Put)
    call.setPricingEngine(engine)
    put.setPricingEngine(engine)

    return {"call": call.NPV(), "put": put.NPV()}
