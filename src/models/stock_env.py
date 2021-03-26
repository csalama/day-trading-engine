#!/usr/bin/env python

import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

MAX_ACCOUNT_BALANCE = 2147483647
MAX_NUM_SHARES = 2147483647
MAX_SHARE_PRICE = 1000 #### Pending ####
MAX_STEPS = 50000 #### Pending ####

INITIAL_ACCOUNT_BALANCE = 100000 #### Pending ####

class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (0, MAX_ACCOUNT_BALANCE)

        # Actions of the format Buy x%, Sell x%, Hold.  Action space of 2.
        self.action_space = spaces.Box(
            low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)
        # Ranges:
        # (0, 3.0) : Choose 0-1 buy, 1-2 sell, 2-3 hold for the stock
        # (0, 1.0) : Percentage of stock to buy, 0% - 100%

        #This contains all input variables we want our agent to consider scaled 0 to 1
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.df.shape[0], 12), dtype=np.float16)
            #1 by 14 box with (0,1) bounds for each

        MAX_STEPS=self.df.shape[0]

    def _next_observation(self):
        # Get the stock data points next 1 minute and scale to between 0-1

        # Numpy version of next observation
        # frame = np.array([
        #     self.df.loc[self.current_step, 'open'] / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step, 'high'] / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step, 'low'] / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step, 'volume'] / MAX_NUM_SHARES,
        #     self.df.loc[self.current_step, 'MA'] / MAX_SHARE_PRICE,
        #     self.df.loc[self.current_step, 'OBV'] / MAX_NUM_SHARES,
        #     self.df.loc[self.current_step, 'RSI'] / 100
        # ])
        # # Append additional data and scale each value to between 0-1
        # obs = np.append(frame, [
        #     self.balance / MAX_ACCOUNT_BALANCE,
        #     self.max_net_worth / MAX_ACCOUNT_BALANCE,
        #     self.shares_held / MAX_NUM_SHARES,
        #     self.cost_basis / MAX_SHARE_PRICE,
        #     self.total_shares_sold / MAX_NUM_SHARES,
        #     self.total_sales_value / (MAX_NUM_SHARES * MAX_SHARE_PRICE),
        # ], axis=0)

        # List version of next observation
        obs = [
            self.df.loc[self.current_step, 'close'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'volume'] / MAX_NUM_SHARES,
            self.df.loc[self.current_step, 'MA'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'OBV'] / MAX_NUM_SHARES,
            self.df.loc[self.current_step, 'RSI'] / 100,
            self.df.loc[self.current_step, 'TEMA'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'BBAND_u'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'BBAND_m'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'BBAND_l'] / MAX_SHARE_PRICE,
            self.df.loc[self.current_step, 'MOM'] / MAX_SHARE_PRICE,
            self.balance / MAX_ACCOUNT_BALANCE,
            self.shares_held / MAX_NUM_SHARES
        ]

        #print(f'\n\n Observation for step {self.current_step}: {obs}\n')
        return obs

    def _take_action(self, action):
        # Set the current price to a random price within the time step
        # Don't exactly see a reason for this random choice
        current_price = random.uniform(
            self.df.loc[self.current_step, "open"], self.df.loc[self.current_step, "close"])
        #print(f'Action in take_action function: {action}')
        action_type = action[0]
        amount = action[1]

        if action_type < 1:
            # Buy amount % of balance in shares

            #total shares it could possibly buy with the current balance
            total_possible = int(self.balance / current_price)

            #buy amount percentage of the shares it could possibly buy
            shares_bought = int(total_possible * amount)

            #Total amount that was paid for the previous stocks
            prev_cost = self.cost_basis * self.shares_held

            #Cost added now that it's buying more
            additional_cost = shares_bought * current_price

            #subtract the additions from the additional_cost
            self.balance -= additional_cost

            #Set cost basis as total value of the stocks purchased divided by total number of shares
            self.cost_basis = (prev_cost + additional_cost) / (self.shares_held + shares_bought)

            #Add the new shares to the shares_held
            self.shares_held += shares_bought

        elif action_type < 2:
            # Sell amount % of shares held
            shares_sold = int(self.shares_held * amount)
            self.balance += shares_sold * current_price
            self.shares_held -= shares_sold
            self.total_shares_sold += shares_sold
            self.total_sales_value += shares_sold * current_price

        #Net worth is balance plus current value of the stocks
        self.net_worth = self.balance + self.shares_held * current_price

        #If our net worth is greater than the max, set to the max (???)
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

        #If we pass through while 'holding', just set cost_basis to 0
        if self.shares_held == 0:
            self.cost_basis = 0

    def step(self, action):
        # Execute one time step within the current environment
        #print('Action given: {}'.format(action))
        self._take_action(action)
        self.current_step += 1

        #This actually doesn't seem like a good idea unless it's necessary
        if self.current_step >= len(self.df.loc[:, 'open'].values) - 1:
            #done = True
            self.current_step = 0
            #might not accurately show the data if we go back in time randomly

        done = self.net_worth <= 0  #If net worth falls to 0, done
        #Set the reward as the balance * delay multiplier to encourage later rewards

        #delay_modifier = (self.current_step / MAX_STEPS)
        reward = (self.net_worth) #* delay_modifier
        #print(reward)
        obs = self._next_observation()

        if done:
            self.render()

        return obs, reward, done, {}

    def reset(self):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE #Why is our max equal to our initial balance
        self.shares_held = 0
        self.cost_basis = 0
        self.total_shares_sold = 0
        self.total_sales_value = 0

        # Set the current step to a random point within the data frame in order
        # to encourage exploration
        #Always start at 0 instead
        self.current_step = 0 #random.randint(0, len(self.df.loc[:, 'open'].values) - 1)

        #print('Within reset: {0}'.format(temp))
        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE

        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(
            f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')
        print(
            f'Avg cost for held shares: {self.cost_basis} (Total sales value: {self.total_sales_value})')
        print(
            f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
