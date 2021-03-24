# Implement our gym environment into a learning algorithm

import numpy as np

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Flatten, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import RandomUniform

from rl.agents import DQNAgent, DDPGAgent
from rl.policy import BoltzmannQPolicy,EpsGreedyQPolicy
from rl.random import OrnsteinUhlenbeckProcess
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor

def train_agent(env):
    #Model Parameters:

    ###Compiler
    learning_rate = 1e-3  #Learning rate of the Adam optimizer (partial SGD)
                          #Adjust to LearningRateSchedule for a schedule.
    ###Early Stopping
    monitor = 'val_loss'   #change to val_loss to track 'metrics'
    min_delta = 1   #Minimum change in 'monitor' to count as an improvement
    patience = 100  #How many epochs without improvement will we go before stopping
    restore_best_weights = True #Restore weight from the epoch with the best monitored quantity

    ###Agent
    metrics = ['mae'] #Metrics to track
    nb_steps = 100000  #Number of training steps performed #Changable
    verbose = 1  #0 for nothing, 1 for interval logging, 2 for episode logging

    #Training
    optimizer = Adam(learning_rate=learning_rate)
    callbacks = [EarlyStopping(monitor = monitor,
                                min_delta = 0,
                                restore_best_weights = restore_best_weights
                                )] #Maybe add TensorBoard for details in the future

    ##########################################
    #Define our space
    num_states = 1 #env.observation_space.shape[1] #temp setting to 1 for testing
    print("Size of State Space ->  {}".format(num_states))
    num_actions = env.action_space.shape[0]
    print("Size of Action Space ->  {}".format(num_actions))

    upper_bound = env.action_space.high[0]
    lower_bound = env.action_space.low[0]

    print("Max Value of Action ->  {}".format(upper_bound))
    print("Min Value of Action ->  {}".format(lower_bound))

    ##########################################
    actor = build_actor(num_states,num_actions)
    ##########################################
    #Build Critic
    critic,action_input = build_critic(num_states,num_actions)
    ##########################################

    agent = build_ddpg(actor,critic,action_input,num_actions)
    agent.compile(optimizer,metrics=metrics)
    training_history = agent.fit(env,
                        callbacks=callbacks,
                        nb_steps=nb_steps,
                        visualize=False,
                        verbose=verbose)
    return agent,training_history

def build_actor(num_states,num_actions):
    #Build Actor
    print('Building Actor.')
    last_init = RandomUniform(minval=-0.003, maxval=0.003)
    inputs = Input(shape=(num_states,))
    out = Dense(256, activation="relu")(inputs)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(num_actions, activation="tanh", kernel_initializer=last_init)(out)
    #outputs = outputs * upper_bound #tf.math.multiply(outputs, [3.0,1.0])
    actor = Model(inputs, outputs)
    #actor.summary()
    return actor

def build_critic(num_states,num_actions):
    print('Building Critic.')
    # State as input
    state_input = Input(shape=(num_states,))
    state_out = Dense(16, activation="relu")(state_input)
    state_out = Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = Input(shape=(num_actions,))
    action_out = Dense(32, activation="relu")(action_input)

    # Both are passed through seperate layer before concatenating
    concat = Concatenate()([state_out, action_out])

    out = Dense(256, activation="relu")(concat)
    out = Dense(256, activation="relu")(out)
    outputs = Dense(1)(out)

    critic = Model([state_input, action_input], outputs)
    critic.summary()
    return critic,action_input

def build_ddpg(actor,critic,action_input,nb_actions):
    """
    The model passed through should be the neural network desired to train this model.
    """
    #Parameters to define:
    learning_rate = 1e-3
    nb_steps_warmup = 10  #Determines how long before experience replay
    seq_memory_limit = 50000 #Maximum size for the memory object

    #policy = EpsGreedyQPolicy(eps=.1)
    #policy = BoltzmannQPolicy(tau=1.) #Can increase tau to increase exploration
    processor = MultiInputProcessor(14) #Number of inputs
    memory = SequentialMemory(limit=seq_memory_limit,window_length=5)
    random_process = OrnsteinUhlenbeckProcess(size=nb_actions, theta=.15, mu=0., sigma=.3)
    ddpg = TestDDPG(nb_actions=nb_actions,
        actor=actor,
        critic=critic,
        critic_action_input=action_input,
        random_process=random_process,
        memory=memory,
        processor=processor,
        nb_steps_warmup_critic=15, nb_steps_warmup_actor=15, target_model_update=1e-3)
    return ddpg

class TestDDPG(DDPGAgent):
    def __init__(self, nb_actions, actor, critic, critic_action_input, memory,gamma=.99, batch_size=22, nb_steps_warmup_critic=1000, nb_steps_warmup_actor=1000,train_interval=1, memory_interval=1, delta_range=None, delta_clip=np.inf,random_process=None, custom_model_objects={}, target_model_update=.001, **kwargs):
        super().__init__(nb_actions, actor, critic, critic_action_input, memory,gamma=gamma, batch_size=batch_size, nb_steps_warmup_critic=nb_steps_warmup_critic, nb_steps_warmup_actor=nb_steps_warmup_actor,train_interval=train_interval, memory_interval=memory_interval, delta_range=delta_range, delta_clip=delta_clip,random_process=random_process,custom_model_objects=custom_model_objects,target_model_update=target_model_update, **kwargs)

    def select_action(self, state):
        batch = self.process_state_batch([state]) #Trying to take this out again
        print(f'Batch in select action is: {batch}')
        action = self.actor.predict_on_batch(batch).flatten()
        assert action.shape == (self.nb_actions,)

        # Apply noise, if a random process is set.
        if self.training and self.random_process is not None:
            noise = self.random_process.sample()
            assert noise.shape == action.shape
            action += noise

        return action

    def forward(self, observation):
        # Select an action.
        #print(f'Within forward, observation shape is: {observation.shape}')
        state = self.memory.get_recent_state(observation)
        print(f'State passed to select action is: {state}')
        action = self.select_action(state)  # TODO: move this into policy

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action


    def backward(self, reward, terminal=False):
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        can_train_either = self.step > self.nb_steps_warmup_critic or self.step > self.nb_steps_warmup_actor
        if can_train_either and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            action_batch = np.array(action_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert action_batch.shape == (self.batch_size, self.nb_actions)

            # Update critic, if warm up is over.
            if self.step > self.nb_steps_warmup_critic:
                target_actions = self.target_actor.predict_on_batch(state1_batch)
                assert target_actions.shape == (self.batch_size, self.nb_actions)
                if len(self.critic.inputs) >= 3:
                    state1_batch_with_action = state1_batch[:]
                else:
                    state1_batch_with_action = [state1_batch]
                state1_batch_with_action.insert(self.critic_action_input_idx, target_actions)
                #print(f'The state1_batch_with_action is: {state1_batch_with_action}')
                target_q_values = self.target_critic.predict_on_batch(state1_batch_with_action).flatten()
                assert target_q_values.shape == (self.batch_size,)

                # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target ys accordingly,
                # but only for the affected output units (as given by action_batch).
                discounted_reward_batch = self.gamma * target_q_values
                discounted_reward_batch *= terminal1_batch
                assert discounted_reward_batch.shape == reward_batch.shape
                targets = (reward_batch + discounted_reward_batch).reshape(self.batch_size, 1)

                # Perform a single batch update on the critic network.
                if len(self.critic.inputs) >= 3:
                    state0_batch_with_action = state0_batch[:]
                else:
                    state0_batch_with_action = [state0_batch]
                state0_batch_with_action.insert(self.critic_action_input_idx, action_batch)
                metrics = self.critic.train_on_batch(state0_batch_with_action, targets)
                if self.processor is not None:
                    metrics += self.processor.metrics

            # Update actor, if warm up is over.
            if self.step > self.nb_steps_warmup_actor:
                # TODO: implement metrics for actor
                if len(self.actor.inputs) >= 2:
                    inputs = state0_batch[:]
                else:
                    inputs = [state0_batch]
                action_values = self.actor_train_fn(inputs)[0]
                assert action_values.shape == (self.batch_size, self.nb_actions)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics


#Build the actor
# def build_sequential(input_shape,output_shape,dense_shape=[2,24],dense_activation='relu'):
#     model = Sequential()
#     model.add(Flatten(input_shape=(1,) + input_shape.shape ))
#     for i in range(dense_shape[0]):
#         model.add(Dense(dense_shape[1], activation = dense_activation))
#     model.add(Dense(output_shape, activation = 'linear'))
#     return model


#Build the critic
# def build_critic(actions,obs_input,dense_shape=[2,24],dense_activation='relu'):
#     action_input = Input(shape=(actions,), name='action_input')
#     observation_input = Input(shape=(1,) + obs_input.shape, name='observation_input')
#
#     flattened_observation = Flatten()(observation_input)
#     x = Concatenate()([action_input, flattened_observation])
#     for i in range(dense_shape[0]):
#         x = Dense(dense_shape[1])(x)
#         x = Activation('relu')(x)
#     x = Dense(1)(x)
#     x = Activation('linear')(x)
#     critic = Model(inputs=[action_input, observation_input], outputs=x)
#     return critic,action_input
