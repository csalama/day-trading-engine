#!/usr/bin/env python

import os
import pandas as pd

#Load paths
from dotenv import load_dotenv
load_dotenv()
PROJECT_PATH = os.environ.get('PROJECT_PATH')
RAW_DATA_PATH = os.path.join(PROJECT_PATH,'data/raw/')

MSFT_PATH = os.path.join(RAW_DATA_PATH,'MSFT_intraday_1year/')
AGG_PATH = os.path.join(RAW_DATA_PATH,'MSFT_1year_agg.csv')

def join_dataset():
    msft_files = os.listdir(MSFT_PATH)
    df = pd.read_csv(os.path.join(MSFT_PATH,msft_files[0]))
    for file in msft_files[1:]:
        temp_path = os.path.join(MSFT_PATH,file)
        new_df = pd.read_csv(temp_path)
        df = pd.concat([df,new_df],ignore_index=True)
    df.to_csv(AGG_PATH,index=False)
    return None

def main():
    #Join and save to the correct path
    join_dataset()

    #Test the import
    df = pd.read_csv(AGG_PATH)
    print(df.head())

if  __name__ == "__main__":
    main()

#########################
# Ignore below
#########################

#import requests

#Importing data automatically using alpha-vantage
#def build_av_dataset(symbol,trailing_months):
#
#    av_link = 'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol={symbol}&interval=1min&apikey=82XI361U1X7XJQ4M&datatype=csv'
    #requests.get("")
