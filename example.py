from datetime import date
from src.gbm.pricer import price_bs
from src.merton.pricer import price_merton_mc
from src.cev.pricer import price_cev
from src.heston.pricer import price_heston

# Option parameters
spot = 100
strike = 100
maturity = date(2025, 6, 1)

print("BS:", price_bs(spot, strike, maturity))
print("Merton:", price_merton_mc(spot, strike, maturity))
print("CEV:", price_cev(spot, strike, maturity))
print("Heston:", price_heston(spot, strike, maturity))
