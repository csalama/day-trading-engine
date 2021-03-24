#!/usr/bin/env python

import os
import sys
from dotenv import load_dotenv
load_dotenv()
PROJECT_PATH = os.environ.get('PROJECT_PATH')
sys.path.insert(0,PROJECT_PATH)

import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from stable_baselines.common.vec_env import DummyVecEnv;
from stable_baselines.ddpg.policies import DDPGPolicy;
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec;
from stable_baselines import DDPG,PPO2;
from stable_baselines.ddpg import MlpPolicy

from stock_env import StockTradingEnv

FIN_PATH = os.path.join(PROJECT_PATH,'data/processed/MSFT_1year_feat.csv')

def train_model():
    df = import_dataset(FIN_PATH)

    #df_tr,df_te = train_test_split(df,test_size=.33,random_state=13, shuffle=False)
    #env = StockTradingEnv(df_tr[0:1000])

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
    ### Build environment
    df1=df
    #print(df1)
    #df2=df_tr[1000:1100].reset_index(drop=True)
    env_tr = DummyVecEnv([lambda: StockTradingEnv(df1)])
    #env_val = DummyVecEnv([lambda: StockTradingEnv(df2)])

    # print(df_tr[0:1000].loc[0, 'open'])
    # print(df1.shape)
    # print(df2.shape)

    ### Training
    n_actions = env_tr.action_space.shape[-1]
    param_noise = None
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

    train_timesteps=20000

    # start = time.time()
    # model = DDPG('MlpPolicy',
    #               env_tr,
    #               param_noise=param_noise,
    #               action_noise=action_noise,
    #               actor_lr = .1,
    #               critic_lr = .1,
    #               batch_size = 100000,
    #               verbose=0)
    # model.learn(total_timesteps=train_timesteps)
    # end = time.time()
    # print('Training time (DDPG): ', (end-start)/60,' minutes')
    # model.save(os.path.join(PROJECT_PATH,'model/ddpg_initial') )

    policy_kwargs = dict(net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])])

    start = time.time()
    model = PPO2('MlpPolicy', env_tr, policy_kwargs=policy_kwargs, ent_coef = 0.005, nminibatches = 8)
    model.learn(total_timesteps=train_timesteps)
    end = time.time()
    print('Training time (PPO): ', (end - start) / 60, ' minutes')
    model.save(os.path.join(PROJECT_PATH,'model/ppo2_20kstep') )

    # ### Quickly validating
    # obs_trade = env_val.reset()
    # reward_l = []

    # model = PPO2.load(os.path.join(PROJECT_PATH,'model/ppo2_initial'))
    # obs_trade = env_tr.reset()
    # for i in range(train_timesteps):
    #     action,_states=model.predict(obs_trade)
    #     obs_trade, rewards, dones, info = env_tr.step(action)
    #     reward_l.append(rewards[0])
    #     if i == (train_timesteps-1):
    #         last_state = env_tr.render()
    # plt.plot(range(train_timesteps),reward_l)
    # plt.show()

    return None

def test_model():
    df = import_dataset(FIN_PATH)
    df1=df

    env_tr = DummyVecEnv([lambda: StockTradingEnv(df1)])

    model = PPO2.load(os.path.join(PROJECT_PATH,'model/ppo2_20kstep'))

    obs_trade = env_tr.reset()

    timesteps=2000
    reward_l = []
    for i in range(timesteps):
        action,_states=model.predict(obs_trade)
        obs_trade, rewards, dones, info = env_tr.step(action)
        reward_l.append(rewards[0])
        if i == (timesteps-1):
            last_state = env_tr.render()
    plt.plot(range(len(reward_l)),reward_l)
    plt.show()

    return None

def sample_results():
    df = import_dataset(FIN_PATH)
    base_env = StockTradingEnv(df)

    #env_tr = DummyVecEnv([lambda: StockTradingEnv(df)])
    obs_trade = base_env.reset()

    #print(base_env.action_space.sample())

    reward_l = []
    timesteps=2000
    for i in range(timesteps):
        action = base_env.action_space.sample()
        print('Action to give: {}'.format(action))
        obs_trade, rewards, dones, info = base_env.step(action)
        reward_l.append(rewards+100000)
        if i == (timesteps-1):
            last_state = base_env.render()
    plt.plot(range(len(reward_l)),reward_l)
    plt.plot(range(len(reward_l)),np.ones(len(reward_l))*100000 )
    plt.show()

    return None

def import_dataset(PATH):
    df = pd.read_csv(PATH)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['time'],inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def main():
    sample_results()


if __name__ == '__main__':
    main()
