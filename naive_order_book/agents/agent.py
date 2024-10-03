from abc import ABC, abstractmethod
from typing import List

from naive_order_book.order import Order
from naive_order_book.orderbook import OrderBook


class Agent(ABC):
    """abstract class for Agent interacting in the market

    Args:
        ABC (_type_): _description_
    """

    def __init__(self, agent_id: str, **kwargs):
        """Constructor

        Args:
            agent_id (str): string that uniquely defines the agent
        """
        self.agent_id = agent_id

    @abstractmethod
    def place_order(
        self,
        mid_price: float,
        spread: float,
        price_ticks: List[float],
        **kwargs,
    ):
        raise NotImplementedError("This method must be implemented by subclasses")
