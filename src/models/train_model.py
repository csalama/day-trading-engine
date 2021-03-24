#!/usr/bin/env python
import os
import sys
from dotenv import load_dotenv
load_dotenv()
PROJECT_PATH = os.environ.get('PROJECT_PATH')
sys.path.insert(0,PROJECT_PATH)

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit,train_test_split
from tensorflow.keras.layers import Dense, Flatten

from build_model import train_agent, build_ddpg
from stock_env import StockTradingEnv


FIN_PATH = os.path.join(PROJECT_PATH,'data/processed/MSFT_1year_feat.csv')

def train_model():
    df = import_dataset(FIN_PATH)

    #Train test split the time series
    #tscv = TimeSeriesSplit(n_splits = 2)
    # for train_index, test_index in tscv.split(df):
    #     print("TRAIN RANGE:", train_index[0], "to", train_index[-1])
    #     print("TEST RANGE:",test_index[0], "to", test_index[-1])
    #     df_tr, df_te = df[train_index], df[test_index]
    df_tr,df_te = train_test_split(df,test_size=.33,random_state=13, shuffle=False)
    env = StockTradingEnv(df_tr[0:1000])

    # for i_episode in range(20):
    #     observation = env.reset()
    #     for t in range(100):
    #         #env.render()
    #         print(type(observation))
    #         print(observation)
    #         action = env.action_space.sample()
    #         observation, reward, done, info = env.step(action)
    #         if done:
    #             print("Episode finished after {} timesteps".format(t+1))
    #             break

    agent,history = train_agent(env)
    print(history)
    agent.save_weights(os.path.join(PROJECT_PATH),'Model/initial_msft_weights.h5f',overwrite=True)
    t = agent.test(env, nb_episodes=10, visualize=True)
    return None

def import_dataset(PATH):
    df = pd.read_csv(PATH)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['time'],inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def main():
    train_model()

if __name__=='__main__':
    main()
