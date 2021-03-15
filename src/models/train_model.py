#!/usr/bin/env python
import os
import sys
from dotenv import load_dotenv
load_dotenv()
PROJECT_PATH = os.environ.get('PROJECT_PATH')
sys.path.insert(0,PROJECT_PATH)

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from build_model import train_agent, build_dqn, build_sequential
from stock_env import StockTradingEnv
#import src.build_features


FIN_PATH = os.path.join(PROJECT_PATH,'data/processed/MSFT_1year_feat.csv')

def train_model():
    df = import_dataset(FIN_PATH)
    print(df.head())
    #tscv = TimeSeriesSplit(n_splits = 3)
    #for train_index, test_index in tscv.split(df):
    #   print("TRAIN RANGE:", train_index[0], "to", train_index[-1])
    #    print("TEST RANGE:",test_index[0], "to", test_index[-1])

        #df_tr, df_te = df[train_index], df[test_index]
        #print(df_tr)
        #print(df_te)
    #Train test split the time series

    small_test_df = df[0:100]
    env = StockTradingEnv(small_test_df)

    dqn,history = train_agent(env)

    print(history)
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
