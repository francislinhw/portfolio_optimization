from src.market_setup import setup_market, create_option
import QuantLib as ql


def price_heston(
    spot,
    strike,
    maturity_date,
    r=0.05,
    q=0.0,
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    sigma=0.5,
    rho=-0.7,
):
    env = setup_market(spot, maturity_date, r, q, sigma)

    heston_process = ql.HestonProcess(
        env["r"], env["q"], env["spot"], v0, kappa, theta, sigma, rho
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)

    call = create_option(strike, env["maturity"], ql.Option.Call)
    put = create_option(strike, env["maturity"], ql.Option.Put)
    call.setPricingEngine(engine)
    put.setPricingEngine(engine)

    return {"call": call.NPV(), "put": put.NPV()}
