from typing import List

import numpy as np
from naive_order_book.agents.agent import Agent
from naive_order_book.order import Order
from naive_order_book.orderbook import OrderBook


class NoiseAgent(Agent):
    """Class representing a noisy agent which places random
    orders and provides liquidity to the market

    Args:
        Agent (_type_): _description_
    """

    def __init__(self, id: int):
        agent_id = "noise_agent_" + str(id)
        super().__init__(agent_id)

    def place_order(
        self,
        mid_price: float,
        spread: float,
        price_ticks: List[float],
        quantities: List[int],
    ) -> Order:
        quantity = np.random.choice(quantities, 1)[0]
        direction = np.random.choice([1, -1], 1)[0]
        price_tick = np.random.choice(price_ticks, 1)[0]
        price = mid_price - direction * spread * (1 + price_tick)

        return Order(direction, price, quantity)
