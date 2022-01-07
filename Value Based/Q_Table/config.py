import gym
env= gym.make('FrozenLake-v1')

episodes= 1000
max_step=100

alpha= 0.5
gamma= 0.99

max_epsilon=epsilon= 1.0
min_epsilon= 1e-2
decay_rate= 0.005