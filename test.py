# environment

# buffer


# actor
ACTOR_HIDDEN_DIM = 64
ACTOR_HIDDEN_LAYER_COUNT = 0
ACTOR_LR = 1e-4

# critic
CRITIC_HIDDEN_DIM = 64
CRITIC_HIDDEN_LAYER_COUNT = 0
CRITIC_LR = 1e-3

# training
BATCH_SIZE = 100
EPISODE_COUNT = 200
from environment import Environment

env = Environment()

env.reset()

agent_count = len(env.uavs)
state_dim = env.observation_space.shape[1]
action_dim = env.action_space.shape[1]
from algorithm import Algorithm
from utils import *
import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
utils = Utils(device=device)

algorithm = Algorithm(
    algorithm="MADDPG",
    agent_count=agent_count,
    state_dim=state_dim,
    action_dim=action_dim,
    actor_hidden_dim=ACTOR_HIDDEN_DIM,
    critic_hidden_dim=CRITIC_HIDDEN_DIM,
    actor_hidden_layer_count=ACTOR_HIDDEN_LAYER_COUNT,
    critic_hidden_layer_count=CRITIC_HIDDEN_LAYER_COUNT,
    actor_lr=ACTOR_LR,
    critic_lr=CRITIC_LR,
    device=device,
    utils=utils,
)
import numpy as np

PRINT_INTERVAL = 10
TRAIN_INTERVAL = 10

total_steps = 0
score_history = []
avg_scores = []

for episode in range(EPISODE_COUNT):
    score = algorithm.explore(env=env)
    if total_steps % PRINT_INTERVAL == 0:
        algorithm.train()

    score_history.append(score)
    avg_score = float(np.mean(score_history[-100:]))
    if episode % PRINT_INTERVAL == 0 and episode > 0:
        print(f"Episode {episode}, Average Score: {avg_score:.1f}")
        avg_scores.append(avg_score)
