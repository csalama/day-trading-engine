#!/usr/bin/env python

import os
import sys

import pandas as pd
import talib

from dotenv import load_dotenv
load_dotenv()
PROJECT_PATH = os.environ.get('PROJECT_PATH')
AGG_PATH = os.path.join(PROJECT_PATH,'data/raw/MSFT_1year_agg.csv')

def build_feature_dataset():
    df = import_dataset()
    #SMA
    df['MA'] = talib.SMA(df['close'])

    #TEMA
    df['TEMA'] = talib.TEMA(df['close'])

    #OBV
    df['OBV'] = talib.OBV(df['close'],df['volume'])

    #CCI
    df['CCI'] = talib.CCI(df['high'],df['low'],df['close'])

    #HT_TRENDLINE
    df['HT_TRENDLINE'] = talib.HT_TRENDLINE(df['close'])

    #BBANDS
    df['BBAND_u'],df['BBAND_m'],df['BBAND_l'] = talib.BBANDS(df['close'])

    #MOM
    df['MOM'] = talib.MOM(df['close'])

    #ATR
    df['ATR'] = talib.ATR(df['high'],df['low'],df['close'])

    #RSI
    df['RSI'] = talib.RSI(df['close'])

    df.to_csv(os.path.join(PROJECT_PATH,'data/processed/MSFT_1year_feat.csv'),index=False)

def import_dataset():
    df = pd.read_csv(AGG_PATH)
    df['time'] = pd.to_datetime(df['time'])
    df.sort_values(by=['time'],inplace=True)
    df.reset_index(drop=True,inplace=True)
    return df

def main():
    build_feature_dataset()

if __name__ == '__main__':
    main()



#########################
# Ignore below
#########################

# df['15 min MA'] = df['close'].rolling(15,min_periods=1).mean()
#
# # RSI
# rsi_n = 14
# delta = df['close'].diff()
# dUp = delta.copy()
# dDown = delta.copy()
# dUp[dUp<0]=0
# dDown[dDown>0]=0
#
# RolUp = pd.Series.rolling(dUp,rsi_n).mean()
# RolDown = pd.Series.rolling(dDown,rsi_n).mean().abs()
#
# RS = RolUp/RolDown
# df['RSI'] = 100 - (100/ (1.0 + RS))
#
# # On-Balance Volume
# df['obv'] = OBV(df['close'],df['volume'])
