from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        self.last_buy_day = {}
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to sell, for simlicity we just add it in techical index
                # if self.state[index + 1] > 0: # if we use price<0 to denote a stock is unable to trade in that day, the total asset calculation may be wrong for the price is unreasonable
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # update balance
                    self.state[0] += sell_amount

                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        # update balance
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()

        return sell_num_shares

    def _buy_stock(self, index, action):
        def _do_buy():
            if (
                self.state[index + 2 * self.stock_dim + 1] != True
            ):  # check if the stock is able to buy
                # if self.state[index + 1] >0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )  # when buying stocks, we should consider the cost of trading when calculating available_amount, or we may be have cash<0
                # print('available_amount:{}'.format(available_amount))

                # update balance
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount

                self.state[index + self.stock_dim + 1] += buy_num_shares

                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            df_total_value = pd.DataFrame(self.asset_memory)
            tot_reward = (
                self.state[0]
                + sum(
                    np.array(self.state[1 : (self.stock_dim + 1)])
                    * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                )
                - self.asset_memory[0]
            )
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            df_rewards = pd.DataFrame(self.rewards_memory)
            df_rewards.columns = ["account_rewards"]
            df_rewards["date"] = self.date_memory[:-1]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv")
                df_total_value.to_csv(f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv", index=False)
                df_rewards.to_csv(f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv", index=False)
                plt.plot(self.asset_memory, "r")
                plt.savefig(f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png")
                plt.close()

            return self.state, self.reward, self.terminal, False, {}

        else:
            actions = actions * self.hmax
            max_position = self.hmax * 0.3
            actions = np.clip(actions, -max_position, max_position)

            current_total_asset = self.state[0] + sum(
                np.array(self.state[1:(self.stock_dim + 1)]) * 
                np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )

            risk_level = "LOW"
            sell_adjustment = 1.0
            risk_mode = False
            if len(self.asset_memory) > 1:
                peak_value = max(self.asset_memory)
                drawdown = (peak_value - current_total_asset) / peak_value
                if drawdown > 0.40:
                    risk_level = "CRITICAL"
                    sell_adjustment = 0.1
                    risk_mode = True
                elif drawdown > 0.30:
                    risk_level = "HIGH"
                    sell_adjustment = 0.2
                    risk_mode = True
                elif drawdown > 0.25:
                    risk_level = "MEDIUM"
                    sell_adjustment = 0.4
                    risk_mode = True

                if risk_mode:
                    for i in range(len(actions)):
                        if actions[i] < 0:
                            actions[i] = int(actions[i] * sell_adjustment)

            if not risk_mode:
                holdings = np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
                stocks_with_position = np.sum(holdings > 0)
                if self.day < 30:
                    for i in range(self.stock_dim):
                        if holdings[i] == 0 and self.state[0] > 10000:
                            actions[i] = max(actions[i], 50)
                elif stocks_with_position < 7:
                    no_position_indices = np.where(holdings == 0)[0]
                    if self.state[0] > 20000 and len(no_position_indices) > 0:
                        import random
                        selected_idx = random.choice(no_position_indices)
                        actions[selected_idx] = 80
                    current_active = np.sum(np.abs(actions) > 1)
                    if current_active < 7:
                        inactive_indices = np.where(np.abs(actions) <= 5)[0]
                        needed_stocks = 7 - current_active
                        for i in range(min(needed_stocks, len(inactive_indices))):
                            idx = inactive_indices[i]
                            holding = self.state[idx + self.stock_dim + 1]
                            if holding > 100:
                                actions[idx] = -30
                            elif holding > 0:
                                actions[idx] = 20 if np.random.random() > 0.5 else -20
                            else:
                                actions[idx] = 40
                    if self.state[0] < 20000:
                        holdings_with_idx = [(holdings[i], i) for i in range(len(holdings))]
                        holdings_with_idx.sort(reverse=True)
                        for i in range(min(3, len(holdings_with_idx))):
                            if holdings_with_idx[i][0] > 50:
                                actions[holdings_with_idx[i][1]] = -30

            if risk_mode:
                if risk_level == "CRITICAL": actions = actions * 0.4
                elif risk_level == "HIGH": actions = actions * 0.6
                else: actions = actions * 0.8
            elif self.day % 3 == 0:
                actions = actions * 0.8

            if not risk_mode and self.day % 30 == 0 and self.day > 0:
                holdings = np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
                total_holdings = np.sum(holdings)
                if total_holdings > 0:
                    avg_holding = total_holdings / self.stock_dim
                    for i, holding in enumerate(holdings):
                        if holding > avg_holding * 2.5:
                            actions[i] = -20
                        elif holding == 0:
                            actions[i] = 30

            actions = actions.astype(int)

            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            prev_holdings = np.array(self.state[1 : (self.stock_dim + 1)])
            current_prices = np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            trading_reward = 0
            immediate_sell_penalty = 0

            for index in sell_index:
                prev_shares = self.state[index + self.stock_dim + 1]
                if prev_shares > 0:
                    sell_amount = min(abs(actions[index]), prev_shares)
                    if self.last_buy_day.get(index, -10) >= self.day - 1:
                        immediate_sell_penalty -= 0.1
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])
                self.last_buy_day[index] = self.day

            self.actions_memory.append(actions)

            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]
            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            basic_reward = end_total_asset - begin_total_asset
            risk_management_reward = 0
            if risk_mode and len(self.asset_memory) > 1:
                prev_drawdown = (max(self.asset_memory[:-1]) - self.asset_memory[-1]) / max(self.asset_memory[:-1])
                current_drawdown = (max(self.asset_memory) - current_total_asset) / max(self.asset_memory)
                if current_drawdown < prev_drawdown:
                    risk_management_reward = 0.1

            cash_ratio = self.state[0] / end_total_asset if end_total_asset > 0 else 0
            optimal_cash_ratio = 0.1
            cash_penalty = -abs(cash_ratio - optimal_cash_ratio) * 0.05

            holdings = np.array(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            stocks_with_position = np.sum(holdings > 0)
            if stocks_with_position >= 7:
                diversity_reward = 0.5
            elif stocks_with_position >= 5:
                diversity_reward = 0.1
            else:
                diversity_reward = -0.5

            num_trades = np.sum(np.abs(actions) > 0)
            transaction_cost_penalty = -num_trades * 0.001

            active_actions = np.sum(np.abs(actions) > 10)
            action_diversity_bonus = 0.1 if active_actions >= 6 else -0.05

            sell_actions_count = np.sum(actions < 0)
            buy_actions_count = np.sum(actions > 0)
            sell_bonus = 0.02 if sell_actions_count > 0 else 0
            balanced_trading_bonus = 0.03 if sell_actions_count > 0 and buy_actions_count > 0 else 0
            overholding_penalty = -0.03 if np.sum(holdings) > 3000 else 0

            total_reward = (
                basic_reward +
                immediate_sell_penalty +
                diversity_reward +
                transaction_cost_penalty +
                action_diversity_bonus +
                risk_management_reward +
                cash_penalty +
                sell_bonus +
                balanced_trading_bonus +
                overholding_penalty
            )

            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = total_reward
            self.rewards_memory.append(self.reward)
            self.reward = self.reward * self.reward_scaling
            self.state_memory.append(self.state.copy())

            return self.state, self.reward, self.terminal, False, {}
    
    def reset(
        self,
        *,
        seed=None,
        options=None,
    ):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]

        self.episode += 1

        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )  # append initial stocks_share to initial state, instead of all zero
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        (
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ),
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ),
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # add save_state_memory to preserve state in the trading process
    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        # print(df_states)
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        # print(len(date_list))
        # print(len(asset_list))
        df_account_value = pd.DataFrame(
            {"date": date_list, "account_value": asset_list}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
