# Implement our gym environment into a learning algorithm

import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def train_agent(env):
    #Model Parameters:

    ###Compiler
    learning_rate = 1e-3  #Learning rate of the Adam optimizer (partial SGD)
                          #Adjust to LearningRateSchedule for a schedule.
    ###Early Stopping
    monitor = 'val_binary_crossentropy'   #change to val_loss to track 'metrics'
    min_delta = 1   #Minimum change in 'monitor' to count as an improvement
    patience = 1000  #How many epochs without improvement will we go before stopping
    restore_best_weights = True #Restore weight from the epoch with the best monitored quantity

    ###Agent
    metrics = ['mae'] #Metrics to track
    nb_steps = 50000  #Number of training steps performed
    verbose = 1  #0 for nothing, 1 for interval logging, 2 for episode logging

    #Training
    optimizer = Adam(learning_rate=learning_rate)
    callbacks = [EarlyStopping(monitor = monitor,
                                min_delta = 0,
                                restore_best_weights = restore_best_weights
                                )] #Maybe add TensorBoard for details in the future
    states =
    model = build_sequential(states,actions)
    agent = build_dqn(model,actions) #Actions is undefined
    agent.compile(optimizer,metrics=metrics)
    training_history = agent.fit(env,
                        callbacks=callbacks,
                        nb_steps=nb_steps,
                        visualize=False,
                        verbose=verbose)
    return dqn,training_history


def build_dqn(model,actions):
    """
    The model passed through should be the neural network desired to train this model.
    """
    #Parameters to define:
    learning_rate = 1e-3
    nb_steps_warmup = 0 #100  #Determines how long we wait before we start doing experience replay, which is when we actually start training the network
    seq_memory_limit = 50000 #Maximum size for the memory object. Forgets old details as new things are added to the memory.
    #Add a second target or dueling network to reduce overfitting. May not be necessary with keras-rl
    #enable_double_dqn = False
    #enable_dueling_network = False

    #Build the RL agent, memory, policy, add callback

    #The policy for the agent based on softmax/boltzmann dist
    #Can increase tau to increase exploration
    policy = BoltzmannQPolicy(tau=1.)
    memory = SequentialMemory(limit=seq_memory_limit,window_length=1)
    dqn = DQNAgent(model=model,
                    memory=memory,
                    policy=policy,
                    nb_actions=actions,
                    nb_steps_warmup=nb_steps_warmup)
    return dqn


#Build the deep learning model behind the DQNAgent
def build_sequential(states,actions,dense_shape=[2,24],dense_activation='relu'):
    model = Sequential()

    #Input is the total number of states
    model.add(Flatten( input_shape=(1,states) ))  #### CHECK INPUT VALUES

    #Possibly need to modify activation/number of layers
    for i in dense_shape:
        model.add(Dense(dense_shape[i], activation = dense_activation))

    #Need to decide on our output states.  Perhaps buy, sell, hold AKA actions?
    model.add(Dense(actions, activation = 'linear'))  #no 'activation' is applied.  Shape is number of actions.
    return model
