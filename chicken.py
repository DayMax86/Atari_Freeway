import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt

gym.register_envs(ale_py)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

env = gym.make('ALE/Freeway-v5',
                render_mode='human', # Comment out this line when training 
                mode=0,
                difficulty=1
            )

class Q(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Conv2d(in_channels=210, out_channels=32, kernel_size=2, stride=4, device=device)
        self.layer2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, device=device)
        self.layer3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1, device=device)
        self.layer4 = nn.Linear(in_features=1280, out_features=16, device=device)
        self.layer5 = nn.Linear(in_features=16, out_features=3, device=device)
    
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.layer4(x))
        return self.layer5(x)

# Memory buffer
class MemoryBuffer():
    def __init__(self, max_size):
        self.buffer = deque([], maxlen=max_size)

    def commit(self, memory):
        self.buffer.append(memory)

    def sample(self, number_of_memories):
        return random.sample(self.buffer, number_of_memories)

# Hyperparameters
total_episodes = 500
epsilon = 0.99
epsilon_decay = 0.0025
epsilon_min = 0.05
alpha = 0.05
gamma = 0.99
buffer_size = 64
minibatch_size = 16
steps_per_update = 16

q1 = Q().to(device)
q2 = Q().to(device)
q2.load_state_dict(q1.state_dict())
loss_function = nn.MSELoss()
mb = MemoryBuffer(buffer_size)
optimiser = torch.optim.SGD(q1.parameters(), lr=alpha)

def train(epsilon, from_model=None):
    episode_rewards = []
    avg_episode_return = 0
    losses = []
    for epsiode in range(total_episodes):
        # Initialise initial state
        state, _ = env.reset()
        t_state = torch.tensor(state, dtype=torch.float32, device=device)
        episode_complete = False
        episode_reward_total = 0

        step_counter = 0

        while not episode_complete:

            with torch.no_grad():

                if from_model==None:
                    # Perform epsilon-greedy check
                    if np.random.rand() < epsilon:
                        action = np.random.randint(0,2)  # Either move up or noop
                    else:
                        if step_counter < buffer_size:
                            action = 0
                        else:
                            # Add a dimension to the tensor to account for batch_size
                            y = t_state.unsqueeze(dim=0)
                            action = q1(y).argmax().item()
                        
                else:
                    y = t_state.unsqueeze(dim=0)
                    action = from_model(y).argmax().item()

                t_action = torch.tensor(action, dtype=torch.int64, device=device)

            # Execute action, observe reward etc.
            next_state, reward, episode_complete, _, _ = env.step(t_action.item())
            t_next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            t_reward = torch.tensor(reward, dtype=torch.float32, device=device)
            t_episode_complete = torch.tensor(episode_complete, dtype=torch.int8, device=device)

            if from_model==None:
                # Add to memory buffer for experience replay
                mb.commit((t_state,t_action,t_reward,t_next_state,t_episode_complete))
                
                if mb.buffer.__len__() == buffer_size:
                    minibatch = mb.sample(minibatch_size)
                    losses.append( optimise(minibatch, q1, q2) )
                
                # Every C steps, update weights for both networks
                if step_counter.__mod__(steps_per_update)==0:
                    q1.load_state_dict(q2.state_dict())

                step_counter += 1

            episode_reward_total+=reward
        print("Episode", epsiode ,"total reward = ", episode_reward_total)
        losses.clear()
        episode_rewards.append(episode_reward_total)
        avg_episode_return+=episode_reward_total
        epsilon -= epsilon_decay
        if epsilon < epsilon_min:
            epsilon = epsilon_min
        print("Epsilon = ", epsilon)    
    avg_episode_return = avg_episode_return / total_episodes
    print("Average episode reward = ", avg_episode_return)

    if from_model == None:
        torch.save(q1.state_dict(),"saved_model.pt")

    plotResults(episode_rewards)

def optimise(minibatch, q1, q2):
    s, a, r, ns, ec = zip(*minibatch)
    # Turn minibatch into PyTorch-friendly tensors
    states = torch.stack(s)
    actions = torch.stack(a)
    rewards = torch.stack(r)
    next_states = torch.stack(ns)
    complete_episodes = torch.stack(ec)

    with torch.no_grad():
        q_predictions = q1(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        q_targets = rewards + q2(next_states).max(1)[0].squeeze(0) * gamma * (1 - complete_episodes)
    
        # Perform gradient descent/backpropagation
        loss = loss_function(q_predictions, q_targets)
    optimiser.zero_grad()
    loss.requires_grad = True
    loss.backward()
    optimiser.step()
    return loss

def plotResults(episode_rewards):
    plt.plot(episode_rewards)
    plt.show()

# If not working from a model:

train(epsilon)

# If loading a pre-trained model (comment out the line above and uncomment this block):
'''
FILEPATH = "insert_pretrained_model_filepath_here"
model = Q().to(device)
model.load_state_dict(torch.load(FILEPATH))
model.eval()
train(epsilon, model)
'''

env.close()
