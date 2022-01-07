import numpy as np
import random
from config import *
import matplotlib.pyplot as plt

#geting action space ad observation space dimensions for q_table
action_input=env.action_space.n
observation_input=env.observation_space.n

#creating the Q_table
q_table=np.zeros((observation_input,action_input))

#iterative Q_learning with epsilon greedy policy to select actions
rewards=[]
e=[]
r=[]
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
        #bellman equation with TD target
        q_table[state,action]=q_table[state,action]+alpha*(reward+(gamma*np.max(q_table[next_state,:]))-q_table[state,action])
        state=next_state
        if done:
            break
    rewards.append(total_reward)
    #decay epsilon to reduce regret in log(t)
    epsilon=min_epsilon+(max_epsilon-min_epsilon)*np.exp(-1*decay_rate*episode)
    if episode%30==0:
        print("episode: ",episode)
        print("Best Reward: ",max(rewards))
        print("Mean reward over last 30: ",np.mean(rewards))
        print("epsilon: ",epsilon)
        e.append(episode)
        r.append(np.mean(rewards))
        rewards=[]

plt.plot(e,r)
plt.show()