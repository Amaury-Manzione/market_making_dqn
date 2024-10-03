from abc import ABC, abstractmethod
from typing import List

from naive_order_book.agents.agent import Agent
from naive_order_book.order import Order


class MarketMaker(Agent, ABC):
    def __init__(self, agent_id: str, initial_cash: float, amount_to_trade: int):
        super().__init__(agent_id)
        self.cash = initial_cash
        self.amount_to_trade = amount_to_trade
        self.inventory = 0

    @abstractmethod
    def place_order(
        self,
        mid_price: float,
        spread: float,
        price_ticks: List[float],
        hedge_ticks: List[float],
    ) -> List[Order]:
        raise NotImplementedError("This method must be implemented by subclasses")
