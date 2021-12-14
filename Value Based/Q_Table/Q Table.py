import numpy as np
import random
from config import *

#geting action space ad observation space dimensions for q_table
action_input=env.action_space.n
observation_input=env.observation_space.n

#creating the Q_table
q_table=np.zeros((observation_input,action_input))
rewards=[]
for episode in range(episodes):
    state=env.reset()
    done=False
    total_reward=0
    for step in range(max_step):
        #epsiolon greedy policy
        if np.random.uniform(0,1)>epsilon:
            action=np.argmax(q_table[state,:])
        else:
            action=env.action_space.sample()
        
        next_state,reward,done,_=env.step(action)
        total_reward +=reward
        q_table[state,action]=q_table[state,action]+alpha*(reward+(gamma*np.max(q_table[next_state,:]))-q_table[state,action])
        state=next_state
        if done:
            break
    epsilon=min_epsilon+(max_epsilon-min_epsilon)*np.exp(-1*decay_rate*episode)
    print(total_reward)
print(q_table)
