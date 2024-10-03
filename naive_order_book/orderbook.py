from collections import defaultdict, deque
from typing import Dict, Tuple

from naive_order_book.order import Order


class OrderBook:
    """Class représentant un carnet d'ordres."""

    def __init__(self):
        self.bids = defaultdict(deque)  # Bids (achats)
        self.asks = defaultdict(deque)  # Asks (ventes)

    def add_order(self, order: Order):
        """ajoute un ordre

        Args:
            order (Order): ordre
        """
        if order.order_type == 1:
            self.bids[order.price].append(order)
        elif order.order_type == -1:
            self.asks[order.price].append(order)

    def match_orders(self) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        """match les ordres de ventes et d'achats

        Returns:
            Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
            trois dictionnaires avec pour clé l'id des markets makers représentant
            les quantités d'achats / vente échangées et le cash.
        """

        # Initialiser les variables pour suivre les quantités échangées par un market maker
        mm_buy_quantity = {"random_agent": 0, "dqn_agent": 0}
        mm_sell_quantity = {"random_agent": 0, "dqn_agent": 0}
        cash = {"random_agent": 0, "dqn_agent": 0}

        while self.bids and self.asks:
            highest_bid = max(self.bids.keys())
            lowest_ask = min(self.asks.keys())

            # Si les prix ne peuvent pas matcher, on sort de la boucle
            if highest_bid < lowest_ask:
                break

            buy_order = self.bids[highest_bid][0]
            sell_order = self.asks[lowest_ask][0]

            traded_quantity = min(buy_order.quantity, sell_order.quantity)

            if buy_order.agent_id is not None:
                mm_buy_quantity[buy_order.agent_id] += traded_quantity
                cash[buy_order.agent_id] -= traded_quantity * buy_order.price

            if sell_order.agent_id is not None:
                mm_buy_quantity[sell_order.agent_id] += traded_quantity
                cash[sell_order.agent_id] -= traded_quantity * sell_order.price

            # Mise à jour des quantités après exécution
            buy_order.quantity -= traded_quantity
            sell_order.quantity -= traded_quantity

            if buy_order.quantity == 0:
                self.bids[highest_bid].popleft()
                if not self.bids[highest_bid]:
                    del self.bids[highest_bid]

            if sell_order.quantity == 0:
                self.asks[lowest_ask].popleft()
                if not self.asks[lowest_ask]:
                    del self.asks[lowest_ask]

        return mm_buy_quantity, mm_sell_quantity, cash
