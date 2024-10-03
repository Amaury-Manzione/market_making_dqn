import copy
from typing import List

import numpy as np
import torch

from naive_order_book.agents.market_makers.dqn_market_maker import DQNMarketMaker
from naive_order_book.agents.market_makers.random_market_maker import RandomMarketMaker
from naive_order_book.agents.noise_agent import NoiseAgent
from naive_order_book.orderbook import OrderBook


class Market:
    """Class representing the markegt"""

    def __init__(
        self,
        interval: float,
        num_trades: float,
        volatility: float,
        random_mm: RandomMarketMaker,
        dqn_mm: DQNMarketMaker,
        num_noise_agents=100,
    ):
        """_summary_

        Args:
            interval (float): interval of time between transactions
            num_trades (float): number of transactions
            volatility (float): volatility of the market price
            random_mm (RandomMarketMaker): Market Maker that streams randomly
            dqn_mm (DQNMarketMaker): DQN Market Maker
            num_noise_agents (int, optional): noisy agents. Defaults to 100.
        """
        self.interval = interval
        self.num_trades = num_trades
        self.volatility = volatility
        self.num_noise_agent = 100

        # instanciating the agents
        noisy_agents = []
        for i in range(num_noise_agents):
            noisy_agent = NoiseAgent(i)
            noisy_agents.append(noisy_agent)

        self.noisy_agents = noisy_agents

        self.dqn_mm = dqn_mm
        self.random_mm = random_mm
        # persistent_agent = PersistentAgnt(...)

    def simulate(
        self,
        mid_price: float,
        spread: float,
        price_ticks: List[float],
        hedge_ticks: List[float],
        quantities: List[int],
        save_books=False,
    ):
        """simulate one training session

        Args:
            mid_price (float): spot price of the mid price process
            spread (float): spot spread
            price_ticks (List[float]): list of price ticks
            hedge_ticks (List[float]): list of hedge ticks
            quantities (List[int]): possible quantitities of asset to stream
            save_books (bool, optional): save orderbooks for all number of transactions. Defaults to False.

        Returns:
            _type_: _description_
        """
        orderbook = OrderBook()
        Z = np.random.normal(0, self.volatility, self.num_trades + 1)
        timestamps = np.array([self.interval * t for t in range(self.num_trades + 1)])
        mid_prices = mid_price * np.exp(
            -(self.volatility**2) * 0.5 * timestamps
            + self.volatility * np.sqrt(timestamps) * Z
        )
        spreads = Z + spread
        inventory_random_mm = np.zeros(self.num_trades + 1)
        cashs_random_mm = np.zeros(self.num_trades + 1)

        inventory_dqn_mm = np.zeros(self.num_trades + 1)
        cashs_dqn_mm = np.zeros(self.num_trades + 1)

        books = []

        state = [0, 0, 0, 0, 0, spreads[0], 0]
        state = torch.tensor(state, dtype=torch.double)

        for idx in range(1, self.num_trades + 1):
            # streaming rewards and selling a proportion of the inventory for
            # random market maker
            orders_random_mm = self.random_mm.place_order(
                mid_prices[idx], spreads[idx], price_ticks, hedge_ticks
            )
            for order in orders_random_mm:
                orderbook.add_order(order)

            # the same for dqn market maker
            orders_dqn_mm, actions = self.dqn_mm.place_order(
                mid_prices[idx], spreads[idx], price_ticks, hedge_ticks, state
            )
            for order in orders_dqn_mm:
                orderbook.add_order(order)

            # noise agents make random
            for noisy_agent in self.noisy_agents:
                order = noisy_agent.place_order(
                    mid_prices[idx], spreads[idx], price_ticks, quantities
                )
                orderbook.add_order(order)

            if save_books:
                books.append(copy.deepcopy(orderbook))

            highest_bid = max(orderbook.bids.keys())
            lowest_ask = min(orderbook.asks.keys())
            mm_buy_quantity, mm_sell_quantity, cash = orderbook.match_orders()

            # updating random market maker
            self.random_mm.inventory += (
                mm_buy_quantity["random_agent"] - mm_sell_quantity["random_agent"]
            )
            inventory_random_mm[idx] = self.random_mm.inventory
            cashs_random_mm[idx] = cash["random_agent"]

            # updating dqn market maker
            self.random_mm.inventory += (
                mm_buy_quantity["dqn_agent"] - mm_sell_quantity["dqn_agent"]
            )
            inventory_random_mm[idx] = self.random_mm.inventory
            cashs_random_mm[idx] = cash["dqn_agent"]

            # 1) calcuting rewards
            spread_ask = orders_dqn_mm[0].price - lowest_ask
            spread_bid = orders_dqn_mm[1].price - highest_bid
            o_i = (
                mm_buy_quantity["dqn_agent"] * spread_bid
                + mm_sell_quantity["random_agent"] * spread_ask
            )

            pnl_i = self.dqn_mm.inventory * (mid_prices[idx] - mid_prices[idx - 1])

            hci = (inventory_dqn_mm[idx] - inventory_dqn_mm[idx - 1]) * spreads[idx]

            rewards = torch.tensor(o_i + pnl_i - hci, dtype=torch.double)

            # 2) next state
            next_state = [
                mm_buy_quantity["dqn_agent"],
                mm_sell_quantity["dqn_agent"],
                inventory_dqn_mm[idx],
                inventory_dqn_mm[idx - 1],
                mid_prices[idx] - mid_prices[idx - 1],
                spreads[idx],
                spreads[idx - 1],
            ]
            next_state = torch.tensor(next_state, dtype=torch.double)

            # 3) actions to tensor
            actions_index = DQNMarketMaker.get_index_from_action(
                actions[0], actions[1], actions[2]
            )
            actions = torch.tensor(actions, dtype=torch.double)

            done = 1 if idx == self.num_trades else 0

            # 4) updating buffer
            self.dqn_mm.add_to_replay_buffer(
                state, next_state, rewards, actions, actions_index, done
            )

            # 5) updating state
            state = next_state

        return (
            inventory_random_mm,
            cashs_random_mm,
            inventory_dqn_mm,
            cashs_dqn_mm,
            books,
            rewards.item(),
        )
