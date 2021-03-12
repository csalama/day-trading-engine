

from build_model import train_agent, build_dqn, build_sequential
from stock_env import StockTradingEnv

def train_model():



    #Train test split the time series
    env = StockTradingEnv(df)
    train_agent(env)



def main():
    pass

if __name__=='__main__':
    main()
