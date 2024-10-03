from typing import List

import numpy as np
from naive_order_book.agents.market_makers.market_maker import MarketMaker
from naive_order_book.order import Order


class RandomMarketMaker(MarketMaker):
    def __init__(self, initial_cash: float, amout_to_trade: float):
        agent_id = "random_agent"
        super().__init__(agent_id, initial_cash, amout_to_trade)

    def place_order(
        self,
        mid_price: float,
        spread: float,
        price_ticks: List[float],
        hedge_ticks: List[float],
    ):
        is_market_maker = True
        price_tick = np.random.choice(price_ticks, 2)
        price_ask = mid_price + spread * (1 + price_tick[0])
        price_bid = mid_price - spread * (1 + price_tick[1])

        orders = [
            Order(-1, price_ask, self.amount_to_trade, self.agent_id),
            Order(1, price_bid, self.amount_to_trade, self.agent_id),
        ]

        if self.inventory > 0:
            hedge_tick = np.random.choice(hedge_ticks)
            self.inventory -= self.inventory * hedge_tick

        return orders
