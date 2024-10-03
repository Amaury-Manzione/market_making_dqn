from typing import List

import numpy as np
import torch
from tensordict import TensorDict
from torch import nn
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer

from naive_order_book.agents.market_makers.market_maker import MarketMaker
from naive_order_book.network import NN_DQN
from naive_order_book.order import Order


class DQNMarketMaker(MarketMaker):
    def __init__(
        self,
        initial_cash: float,
        amount_to_trade: int,
        epsilon: float,
        epsilon_decay: float,
        epsilon_max: float,
        gamma: float,
        replay_memory_int: int,
        replay_memory_capacity: int,
        n_update: int,
        learning_rate: float,
        batch_size: int,
        activation_function,
        list_weights: List[int],
        input_dim: int,
        output_dim: int,
    ):
        """_summary_

        Args:
            initial_cash (float): _description_
            amount_to_trade (int): _description_
            epsilon (float): _description_
            epsilon_decay (float): _description_
            epsilon_max (float): _description_
            gamma (float): _description_
            replay_memory_int (int): _description_
            replay_memory_capacity (int): _description_
            n_update (int): _description_
            learning_rate (float): _description_
            batch_size (int): _description_
            activation_function (_type_): _description_
            list_weights (List[int]): _description_
            input_dim (int): _description_
            output_dim (int): _description_
        """
        agent_id = "dqn_agent"
        super().__init__(agent_id, initial_cash, amount_to_trade)
        self.cash = initial_cash
        self.amount_to_trade = amount_to_trade

        # dqn parameters
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_max = epsilon_max
        self.replay_memory_init = replay_memory_int
        self.replay_memory_capacity = replay_memory_capacity
        self.n_update = n_update
        self.batch_size = batch_size
        self.gamma = gamma
        # replay buffer
        storage = LazyMemmapStorage(replay_memory_capacity)
        replay_buffer = TensorDictReplayBuffer(storage=storage, batch_size=batch_size)
        self.replay_buffer = replay_buffer

        # neural networks
        self.online_network = NN_DQN(
            input_dim, output_dim, activation_function, list_weights
        )
        self.target_network = NN_DQN(
            input_dim, output_dim, activation_function, list_weights
        )
        for p in self.target_network.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate, maximize=False
        )
        self.criterion = nn.MSELoss()

    @staticmethod
    def get_index_from_action(buy_spread_idx, sell_spread_idx, hedge_spread_idx):
        """_Convert an action in a triplet

        Args:
            buy_spread_idx (_type_): _description_
            sell_spread_idx (_type_): _description_
            hedge_spread_idx (_type_): _description_

        Returns:
            _type_: _description_
        """
        index = buy_spread_idx * (11 * 5) + sell_spread_idx * 5 + hedge_spread_idx
        return index

    @staticmethod
    def get_action_from_index(index):
        """
        Convert an index into a triplet (b, s, h).
        Args:
            index (int): The index (0-604).
        Returns:
            tuple: The triplet (buy_spread_idx, sell_spread_idx, hedge_spread_idx).
        """
        hedge_spread_idx = index % 5
        sell_spread_idx = (index // 5) % 11
        buy_spread_idx = index // (11 * 5)
        return (buy_spread_idx, sell_spread_idx, hedge_spread_idx)

    def get_policy(
        self, state: torch.tensor, price_ticks: List[float], hedge_ticks: List[float]
    ):
        self.online_network.eval()
        with torch.no_grad():
            state = state.unsqueeze(0)
            outputs = self.online_network(state)
            index = torch.argmax(torch.squeeze(outputs))
            buy_spread_idx, sell_spread_idx, hedge_spread_idx = (
                DQNMarketMaker.get_action_from_index(index)
            )
            return buy_spread_idx, sell_spread_idx, hedge_spread_idx

    def train_online_network(
        self,
    ) -> tuple[float, float]:
        """
        One epoch of online network training with Bellmann Loss.
        Returns
        -------
        tuple[float, float]
            loss and mean reward per batch
        """

        if len(self.replay_buffer) < self.replay_memory_init:
            return 0

        else:
            self.online_network.train()
            # calculating Q values
            batch = self.replay_buffer.sample()
            outputs = self.online_network(batch["state"])
            outputs = torch.squeeze(outputs)

            list_actions = batch["action_index"].numpy()
            outputs_loss = torch.zeros(self.batch_size)
            for i in range(self.batch_size):
                outputs_loss[i] = outputs[i, int(list_actions[i])]

            # calculating loss function
            target = self.target_network(batch["next_state"])
            target = torch.squeeze(target)
            target = batch["reward"] + self.gamma * (
                torch.max(target, dim=1)[0] * (1 - batch["done"])
            )

            loss = self.criterion(outputs_loss.double(), target.double())

            history_loss = loss.item()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return history_loss

    def update_target_network(self):
        """
        copy online network parameters to the target  network
        """
        self.target_network.load_state_dict(self.online_network.state_dict())

    def add_to_replay_buffer(
        self,
        current_state: torch.tensor,
        new_state: torch.tensor,
        reward: torch.tensor,
        action: torch.tensor,
        action_index: int,
        done: int,
    ):
        """Add experience to replay buffer

        Parameters
        ----------
        current_state : torch.tensor
        new_state : torch.tensor
        reward : torch.tensor
        action : torch.tensor
        done : torch.tensor
        pos : torch.tensor
        """
        self.replay_buffer.add(
            TensorDict(
                {
                    "state": current_state,
                    "next_state": new_state,
                    "action": action,
                    "reward": reward,
                    "action_index": action_index,
                    "done": done,
                },
                batch_size=[],
            )
        )

    def place_order(
        self,
        mid_price: float,
        spread: float,
        price_ticks: List[float],
        hedge_ticks: List[float],
        state: torch.tensor,
    ) -> List[Order]:
        u = np.random.uniform(0, 1, 1)
        # exploration
        if u < 1 - self.epsilon:
            buy_spread = np.random.choice(price_ticks, 1)[0]
            sell_spread = np.random.choice(price_ticks, 1)[0]
            hedge_spread = np.random.choice(hedge_ticks, 1)[0]
        else:
            buy_spread, sell_spread, hedge_spread = self.get_policy(
                state, price_ticks, hedge_ticks
            )

        actions = [buy_spread, sell_spread, hedge_spread]
        price_ask = mid_price + spread * (1 + buy_spread)
        price_bid = mid_price - spread * (1 + sell_spread)
        orders = [
            Order(-1, price_ask, self.amount_to_trade, self.agent_id),
            Order(1, price_bid, self.amount_to_trade, self.agent_id),
        ]

        if self.inventory > 0:
            self.inventory -= self.inventory * hedge_spread

        self.epsilon = min(self.epsilon_max, self.epsilon + self.epsilon_decay)
        return orders, actions
