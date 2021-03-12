#!/usr/bin/env python

import gym

class CustomTradingEnv(gym.Env):

    #Required variables
    action_space = None
    observation_space = None

    #Optional variables
    #metadata = {'render.modes': []}
    #reward_range = (-float('inf'), float('inf'))
    #spec = None

    def __init__(self,stock_df):
        super(CustomTradingEnv,self).__init__()
        self.stock_df = stock_df



    def step(self,action):
        """
        Input: Accept an action (buy, sell, hold)
        Return a tuple:
            - observation (object): agents observation of the current environment after we took this action (stock price of MSFT, technical data for the next minute)
            - reward (float): reward returned after taking the given step.  tbd on this. stock_env uses balance * current_step/max_steps because the value of the portfolio is the value.
            - done (bool): whether the 'episode' has ended. stock_env defines this as net_worth <=0.
            - info (dict): auxiliary diagnostic info, can just pass {}
        """

        pass
