# Implement our gym environment into a learning algorithm

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_agent(model,actions):



#Build the deep learning model behind the DQNAgent
def build_nn_model(states,actions):
    model = Sequential() 

    #Input is the total number of states, i.e. HUGE.
    model.add(Flatten( input_shape=(1,states) ))  #### CHECK INPUT VALUES

    #Possibly need to modify activation/number of layers
    model.add(Dense(24, activation = 'relu'))
    model.add(Dense(24, activation = 'relu'))

    #Need to decide on our output states.  Perhaps buy, sell, hold AKA actions?
    model.add(Dense(actions, activation = 'linear'))  #no 'activation' is applied.  Shape is number of actions.
    return model
