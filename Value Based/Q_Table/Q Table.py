import numpy as np
from config import *

#geting action space ad observation space dimensions for q_table
action_input=env.action_space.n
observation_input=env.observation_space.n

#creating the Q_table
q_table=np.zeros((action_input,observation_input))



