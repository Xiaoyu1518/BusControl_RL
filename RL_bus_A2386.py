#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.stats import gaussian_kde
import time
import torch.nn.functional as F
import sys


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

matplotlib.use('Agg')

# Read file
route_id = 'A2386'  # A2386 A2387 TM85 TM86
travel_time_df = pd.read_excel(f"travel time_norm_{route_id}.xlsx")
stop_id = travel_time_df['start_stop_id']

# Hyperparameters
NUM_AGENTS = 13
STATE_DIM = 4     # [stop_id, headway, occupancy, fleet_order]
ACTION_DIM = 1
TOTAL_STATE_DIM = STATE_DIM * NUM_AGENTS
TOTAL_ACTION_DIM = ACTION_DIM * NUM_AGENTS
HIDDEN_DIM = 256
HIDDEN_LAYERS = [256, 512, 256]

# Network and Training parameters
LR_ACTOR = 1e-4         
LR_CRITIC = 2e-4
LR_DECAY = 0.995
GAMMA = 0.99
TAU = 0.01
BUFFER_SIZE = 20000
BATCH_SIZE = 128
MAX_EPISODES = 400
NUM_STEP = 6 * (len(stop_id) + 1)
UPDATE_ACTOR_EVERY = 5     
WARMUP_EPISODES = 10   
WARMUP_STEPS = 5      

# Reward scale factors & Reward weights
REWARD_SCALE = 0.1       
HEADWAY_SCALE = 2.0    
HOLDING_SCALE = 0.5      

# Parameters
NUM_STOPS = len(stop_id) + 1
TARGET_HEADWAY = 600
MAX_HOLD = 120
MAX_OCCUPANCY = 1.0
BUNCHING_THRESHOLD = 0.1 * TARGET_HEADWAY  # 60s
CAPACITY = 140

# Exploration
EPSILON_START = 0.9     
EPSILON_END = 0.02    
EPSILON_DECAY = 0.99 

# Training stability and Normalization
GRAD_CLIP = 1.0         
WEIGHT_DECAY = 1e-5
MAX_VALUE = 1e6
REWARD_CLIP = 5.0
Q_VALUE_CLIP = 50.0      
TARGET_Q_CLIP = 60.0     

ALIGHT_TIME = 2  # seconds per passenger
BOARD_TIME = 3  # seconds per passenger
DOOR_TIME = 5  # seconds (open/close)
REST_TIME = 180  # seconds (rest)

# Early stop for monitor
PATIENCE = 50
MIN_EPISODES = 100
IMPROVEMENT_THRESHOLD = 0.02

def normalize_state(state):
    """avoid too large or too small value"""
    return np.clip(state, -MAX_VALUE, MAX_VALUE)

def normalize_reward(reward):
    """avoid too large or too small value"""
    return np.clip(reward, -REWARD_CLIP, REWARD_CLIP)

# ==== Actor Network ====
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        
        # Normalize
        self.ln_input = nn.LayerNorm(state_dim)
        
        # Headway feature
        self.headway_processor = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Feature layer
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim + 16, HIDDEN_LAYERS[0]), 
            nn.LayerNorm(HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Residual layer
        self.residual_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[0]),
                nn.LayerNorm(HIDDEN_LAYERS[0]),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[0]),
                nn.LayerNorm(HIDDEN_LAYERS[0])
            ) for _ in range(2)
        ])
        
        # Headway attention
        self.headway_attention = nn.Sequential(
            nn.Linear(HIDDEN_LAYERS[0], 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # output layer
        self.output_net = nn.Sequential(
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[0] // 2),
            nn.LayerNorm(HIDDEN_LAYERS[0] // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[0] // 2, action_dim)
        )
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
    def forward(self, state):
        # Get headway from state (second feature)
        headway_norm = state[:, 1].unsqueeze(-1)
        
        # Headway
        headway_features = self.headway_processor(headway_norm)
        
        # Process all state information
        x = self.ln_input(state)
        x = torch.cat([x, headway_features], dim=-1)  
        x = self.feature_net(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            identity = x
            x = block(x)
            x = F.relu(x + identity)
        
        # Headway attention
        headway_attention_weight = self.headway_attention(x)
        
        # Generate base action
        x = self.output_net(x)
        
        action = torch.sigmoid(x * 3.0)   

        # Dynamically adjust the action scope based on headway severity
        headway_severity = torch.abs(headway_norm)
        
        action_scale = 0.1 + 0.9 * headway_severity  
        action = action * action_scale * headway_attention_weight 

        # if Headway >= Target, action == 0
        zeros = torch.zeros_like(action)
        action = torch.where(headway_norm >= 0, zeros, action)
        
        return action

# ==== Critic Network ====
class CentralCritic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super().__init__()
        
        # encode state
        self.state_encoder = nn.Sequential(
            nn.Linear(total_state_dim, HIDDEN_LAYERS[0]),
            nn.LayerNorm(HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[0]),
            nn.LayerNorm(HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
            nn.LayerNorm(HIDDEN_LAYERS[1]),
            nn.ReLU()
        )
        
        # encode action
        self.action_encoder = nn.Sequential(
            nn.Linear(total_action_dim, HIDDEN_LAYERS[0]),
            nn.LayerNorm(HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[1]),
            nn.LayerNorm(HIDDEN_LAYERS[1])
        )
        
        # estimate Q-value
        self.q_net = nn.Sequential(
            nn.Linear(HIDDEN_LAYERS[1] * 2, HIDDEN_LAYERS[2]),
            nn.LayerNorm(HIDDEN_LAYERS[2]),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(HIDDEN_LAYERS[2], HIDDEN_LAYERS[0]),
            nn.LayerNorm(HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[0], HIDDEN_LAYERS[0] // 2),
            nn.LayerNorm(HIDDEN_LAYERS[0] // 2),
            nn.ReLU(),
            nn.Linear(HIDDEN_LAYERS[0] // 2, 1)
        )
        
        self.q_output = nn.Sequential(
            nn.Tanh()
        )
        
        # initialize
        self.apply(self._init_weights)
        
        nn.init.uniform_(self.q_net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.q_net[-1].bias, -3e-3, 3e-3)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, states, actions):
        # check stability
        states = torch.clamp(states, -MAX_VALUE, MAX_VALUE)
        actions = torch.clamp(actions, -MAX_VALUE, MAX_VALUE)
        
        # encode
        state_features = self.state_encoder(states)
        action_features = self.action_encoder(actions)
        
        # estimate Q-value
        combined = torch.cat([state_features, action_features], dim=1)
        q_raw = self.q_net(combined)
        q_value = self.q_output(q_raw) * Q_VALUE_CLIP  
        return q_value

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)
        self.priorities = deque(maxlen=max_size)  

    def push(self, states, actions, reward, next_states, priority=1.0):
        self.buffer.append((states, actions, reward, next_states))
        self.priorities.append(priority)

    def sample(self, batch_size):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states = map(np.stack, zip(*samples))
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)
        
# ==== Bus Environment ====
class MultiBusSimEnv:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.constant_speed = 33
        self.global_time = np.zeros(num_agents)
        self.arrival_history = {stop_id: [] for stop_id in range(NUM_STOPS)}
        self.last_departure_times = np.zeros(num_agents)
        self.state = self.reset()
        self.step_count = 0
        
    def get_normalized_headway(self, headway):
        """limit headway to [-1, 1]，check stability"""
        normalized = np.clip(1 * (headway - TARGET_HEADWAY) / TARGET_HEADWAY, -1, 1)
        if np.isnan(normalized) or np.isinf(normalized):
            return 0.0  # error
        return normalized

    def reset(self):
        self.step_count = 0
        self.last_departure_times = np.zeros(self.num_agents)
        self.arrival_history = {stop_id: [] for stop_id in range(NUM_STOPS)}
        
        # initialize state
        self.state = np.zeros((self.num_agents, STATE_DIM), dtype=np.float32)
        self.state[:, 0] = 0  # the first stop
        
        # randomly set departure headways
        random_headways = np.random.uniform(0.3 * TARGET_HEADWAY, 0.6 * TARGET_HEADWAY, size=self.num_agents)
        random_headways = np.clip(random_headways, 0.1 * TARGET_HEADWAY, 1.2 * TARGET_HEADWAY)
        # Store normalized headways in state instead of raw values
        for i in range(self.num_agents):
            self.state[i, 1] = self.get_normalized_headway(random_headways[i])
        
        # randomly set initial occupancy
        random_occupancy = np.random.uniform(0, 0.5, size=self.num_agents)
        self.state[:, 2] = random_occupancy 
        
        # the initial time at the first stop (based on headway above)
        self.global_time = np.array([sum(random_headways[:i]) for i in range(self.num_agents)], dtype=np.float32)
        
        # sort by global_time to generate sequential numbers
        order_ids = np.argsort(self.global_time)  # the first departure, id = 0, then 1, 2, 3, ...
        self.state[:, 3] = order_ids  
        
        return self.state.copy()

    def calculate_reward_components(self, headway, hold_time, occupancy):
        try:
            if headway >= TARGET_HEADWAY:
                # No need for control
                if headway <= 1.1 * TARGET_HEADWAY:
                    headway_reward = 1.0  # slight reward, very close to Target
                else:
                    headway_reward = 0.5
                holding_penalty = -1.0 * (hold_time / MAX_HOLD) if hold_time > 0 else 0.0  # punish holding
                return headway_reward, holding_penalty
            else:
                # 1. headway reward
                headway_ratio = headway / TARGET_HEADWAY  # [0-1]
                
                if headway_ratio >= 0.9:  # 540-600s
                    headway_reward = 1.5
                elif headway_ratio >= 0.7:  # 420-540s
                    headway_reward = 0.5 + 0.5 * (headway_ratio - 0.7) / 0.2
                elif headway_ratio >= 0.3:  # 180-420s
                    headway_reward = 0.2 + 0.3 * (headway_ratio - 0.3) / 0.4
                elif headway_ratio >= 0.1:  # 60-180s
                    headway_reward = -0.2 + 0.4 * (headway_ratio - 0.1) / 0.2
                else:  # <60s，bunching
                    headway_reward = -1.2 + 0.8 * headway_ratio / 0.1
                
                # 2. holding reward 
                if hold_time > 0:
                    holding_cost = hold_time / MAX_HOLD    # [0, 1]
                    occupancy_factor = occupancy * 2.0 
                    hold_reward = - holding_cost * occupancy_factor
                else:
                    if headway >= TARGET_HEADWAY:
                        hold_reward = 0.1  # When holding is not necessary, it is right not to hold
                    else:
                        hold_reward = -0.1 * (1 + occupancy)  # When holding is required, there is a slight penalty for not holding
                
                return headway_reward, hold_reward
            
        except Exception as e:
            print(f"Warning: Error in reward calculation: {str(e)}")
            return 0.0, 0.0

    def step(self, actions, training=True):
        rewards = []
        r1s, r2s = [], []
        prev_state = self.state.copy()
        prev_arrival = self.global_time.copy()

        for i, action in enumerate(actions):
            try:
                cur_stop_id = int(prev_state[i][0])
                occupancy = np.clip(prev_state[i][2], 0, 1)
                fleet_order = int(prev_state[i][3])
                
                if cur_stop_id == NUM_STOPS - 1:
                    action[0] = 0.0
                    next_stop = 0
                else:
                    next_stop = cur_stop_id + 1

                # 1. boarding and alighting number
                choices = np.arange(0, 8)
                probs = np.array([0.135, 0.273, 0.271, 0.180, 0.090, 0.036, 0.012, 0.003])
                probs = probs / np.sum(probs)
                alight_passengers = np.random.choice(choices, p=probs)
                board_passengers = np.random.choice(choices, p=probs)
                
                # 2. dwell time
                base_dwell = max(alight_passengers * ALIGHT_TIME, board_passengers * BOARD_TIME) + DOOR_TIME
                hold_time = float(action[0]) * MAX_HOLD if not (cur_stop_id == NUM_STOPS - 1) else 0
                total_dwell = np.clip(base_dwell + hold_time, 0, MAX_HOLD * 2)
                
                # 3. travel time -- more randomness
                data_columns = travel_time_df.columns[3:]
                if cur_stop_id < NUM_STOPS - 1:  # non-final-stop
                    row = travel_time_df[(travel_time_df['start_stop_id'] == stop_id.iloc[cur_stop_id])].iloc[0]
                    mu = float(row["mu"])
                    sigma = float(row["sigma"])
                    if pd.notna(row['mu']) and pd.notna(row['sigma']) and mu > 0 and sigma > 0:
                        base_travel_time = np.random.normal(row['mu'], max(8, row['sigma']))
                    else:
                        max_col_name = row[data_columns].astype(float).idxmax()
                        base_travel_time = np.random.normal(max_col_name, 8)
                    
                    base_travel_time = max(base_travel_time, 30)
                else:  
                    base_travel_time = 0

                # 4. arrival time at next stop
                departure_time = prev_arrival[i] + total_dwell if cur_stop_id != NUM_STOPS - 1 else prev_arrival[i] + base_dwell + REST_TIME
                arrival_time = departure_time + base_travel_time if cur_stop_id != NUM_STOPS - 1 else departure_time
                
                # check whether overtaking happened, to make sure fleet_id = i+1 arrived later than fleet_id = i
                # when entering the current station, it was blocked by the car in front, thus affecting the next station.        
                for j in range(NUM_AGENTS):
                        if j == i:
                            continue
                        if int(prev_state[j][3]) < fleet_order: 
                            if int(prev_state[j][0]) == cur_stop_id:
                                arrive_diff = prev_arrival[j] - prev_arrival[i]
                                if arrive_diff > 0:  # if the front bus is late, make sure no overtaking
                                    arrival_time = max(arrival_time, arrival_time + arrive_diff + 10)
              
                # 5. update state
                new_occupancy = np.clip(
                    (occupancy * CAPACITY + board_passengers - alight_passengers) / CAPACITY,
                    0, 1
                )
                self.state[i] = [
                    next_stop,
                    prev_state[i][1], # up date later
                    new_occupancy,
                    fleet_order  # keep same until restart from the first stop
                ]
                # update
                self.arrival_history[next_stop].append(arrival_time)
                self.global_time[i] = arrival_time
                self.last_departure_times[i] = departure_time

            except Exception as e:
                print(f"CRITICAL: Unhandled error in step calculation for agent {i}: {str(e)}. State might be inconsistent.")
                # keep previous state for safety
                self.state[i] = prev_state[i] if prev_state is not None and len(prev_state) > i else np.zeros(STATE_DIM) 
                continue
        
        # 6. calculate headway and update state
        for i in range(NUM_AGENTS):
            try:
                arrival_time = self.global_time[i]
                next_stop = int(self.state[i][0])
                new_occupancy = self.state[i][2]
                # the arrival record before this bus in next stop
                recent_arrivals = [t for t in self.arrival_history[next_stop] 
                                    if t < arrival_time - 0.1]
                if recent_arrivals:
                    headway = arrival_time - max(recent_arrivals)  # Calculate the time difference with the nearest front bus
                else:
                    # if no record (i.e., the first bus), use TARGET_HEADWAY
                    headway = TARGET_HEADWAY
                
                # clip into reasonable range
                headway = np.clip(headway, 10, TARGET_HEADWAY * 2.0)
                # update state
                self.state[i][1] = self.get_normalized_headway(headway) 
            
            except Exception as e:
                print(f"Warning: Error in headway calculation for agent {i}: {str(e)}")
                headway = TARGET_HEADWAY
            # 7. calculate reward
            try:
                if training:
                    r1, r2 = self.calculate_reward_components(
                        headway,
                        float(actions[i][0]) * MAX_HOLD if next_stop != 0 else 0,
                        new_occupancy
                    )
                    
                    reward = REWARD_SCALE * (
                        HEADWAY_SCALE * r1 + 
                        HOLDING_SCALE * r2
                    )
                    
                    reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)
                    rewards.append(reward)
                    r1s.append(r1)
                    r2s.append(r2)

            except Exception as e:
                print(f"Error in state update for agent {i}: {str(e)}")
                rewards.append(0.0)
                r1s.append(0.0)
                r2s.append(0.0)
                continue        

        self.step_count += 1
        
        # if all buses are at the first station, re-arrange the numbers by arrival_time
        all_at_start = all(int(self.state[i][0]) == 0 for i in range(self.num_agents))
        if all_at_start:
            new_order = np.argsort(self.global_time)  # Whoever arrives first gets the smaller number
            for rank, bus_id in enumerate(new_order):
                self.state[bus_id][3] = rank  # update state[fleet_order]

        while len(rewards) < self.num_agents:
            rewards.append(0.0)
        while len(r1s) < self.num_agents: r1s.append(0.0)
        while len(r2s) < self.num_agents: r2s.append(0.0)

        return self.state.copy(), rewards, self.step_count >= NUM_STEP, (r1s, r2s)

# Utilities
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def plot_training_results(reward_history):
    if not reward_history:  # safety check
        print("Warning: No reward history to plot")
        return
        
    plt.figure(figsize=(12, 6))
    
    # original reward
    plt.plot(reward_history, 'b-', alpha=0.3, label='Raw Reward')
    
    # moving average reward
    window_size = min(10, len(reward_history))
    if len(reward_history) >= window_size:
        moving_avg = pd.Series(reward_history).rolling(window=window_size).mean()
        plt.plot(moving_avg, 'r-', label=f'Moving Average (window={window_size})')
    
    plt.title("Training Reward over Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("training_reward_curve.png")
    # plt.show()

def train_bus_controller():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = MultiBusSimEnv(NUM_AGENTS)
    
    # initialize network
    actors = [Actor(STATE_DIM, ACTION_DIM).to(device) for _ in range(NUM_AGENTS)]
    target_actors = [Actor(STATE_DIM, ACTION_DIM).to(device) for _ in range(NUM_AGENTS)]
    for i in range(NUM_AGENTS):
        target_actors[i].load_state_dict(actors[i].state_dict())
        
    critic = CentralCritic(TOTAL_STATE_DIM, TOTAL_ACTION_DIM).to(device)
    target_critic = CentralCritic(TOTAL_STATE_DIM, TOTAL_ACTION_DIM).to(device)
    target_critic.load_state_dict(critic.state_dict())
    
    # optimizer
    actor_optimizers = []
    actor_schedulers = []
    for actor in actors:
        opt = optim.Adam(actor.parameters(), lr=LR_ACTOR, weight_decay=WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9995)
        actor_optimizers.append(opt)
        actor_schedulers.append(scheduler)

    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
    critic_scheduler = optim.lr_scheduler.ExponentialLR(critic_optimizer, gamma=0.9995)
    
    # initialize replay buffer
    replay_buffer = ReplayBuffer(BUFFER_SIZE)
    
    # initialize monitor
    reward_history = []
    actor_loss = [[] for _ in range(NUM_AGENTS)]
    critic_loss = []
    best_reward = float('-inf')
    best_actors = None
    patience_counter = 0
    last_improvement = 0
    
    q_explosion_count = 0
    max_q_explosion_resets = 3

    print(f"Training Start at {time.strftime('%Y-%m-%d %H:%M:%S')} ")
    print("Episode |  Reward  | Avg(10) |  r1     r2     --   |   Act  |     Critic Loss     |  Q_pred   | Q_target | Nbunch | FHeadway")
    print("-" * 110)

    for episode in range(MAX_EPISODES):
        states = env.reset()
        states = normalize_state(states)
        total_reward_for_episode = 0
        episode_r1, episode_r2 = [], []
        episode_critic_losses, episode_actor_losses = [], []
        episode_actions = []
        episode_q_pred, episode_q_target = [], []
        count_bunch = 0
        
        q_value_stats = {'min': [], 'max': [], 'mean': [], 'std': []}
        
        # Exploration rate
        epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** max(0, episode - WARMUP_EPISODES)))
        
        # track whether the optimizer step was executed
        optimizer_stepped = False
        
        for t in range(NUM_STEP):
            is_warmup = t < WARMUP_STEPS or episode < WARMUP_EPISODES
            actions = []
            for i in range(NUM_AGENTS):
                if is_warmup or np.random.rand() < epsilon:
                    cur_stop_id = int(states[i][0])
                    headway_norm = states[i][1]
                    
                    if cur_stop_id == NUM_STOPS - 1 or headway_norm >= 0:
                        action_val = 0.0
                    else:
                        #action_val = np.random.uniform(0.0, 1.0)
                        # Heuristic explore
                         if occupancy > 0.8:
                            action_val = np.random.uniform(0.1, 0.3)
                        else:
                            if headway_norm <= -0.8:  
                                base_action = 0.8 
                                noise = np.random.uniform(-0.2, 0.2)  
                                action_val = np.clip(base_action + noise, 0.0, 1.0)
                            elif headway_norm <= -0.5:  
                                base_action = 0.5
                                noise = np.random.uniform(-0.2, 0.2)
                                action_val = np.clip(base_action + noise, 0.0, 1.0)
                            else:  
                                action_val = np.random.uniform(0.0, 1.0) 
                       
                    actions.append(np.array([action_val]))
                else:
                    state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        
                        action_val = actors[i](state_tensor).cpu().numpy()[0]
                        
                        if states[i][1] < 0:  
                            #noise_scale = 0.05 * (1 - episode / MAX_EPISODES)  
                            base_scale = 0.2
                            decay_rate = 0.5  
                            noise_scale = max(0.05, base_scale * (1 - episode / (MAX_EPISODES * decay_rate)))
                            action_val += np.random.normal(0, noise_scale)
                            action_val = np.clip(action_val, 0, 1)
                    actions.append(action_val)

            next_states, rewards, done, (r1s, r2s) = env.step(actions)
            next_states = normalize_state(next_states)

            # normalize rewards
            rewards = [normalize_reward(r) for r in rewards]
            mean_reward = np.mean(rewards)
            
            # Priority Buffer
            headways = []
            headways = [states[i][1] for i in range(NUM_AGENTS)]
            
            # 1. bad record
            has_severe_bunching = any(h <= -0.8 for h in headways)
            has_medium_bunching = any(h <= -0.5 for h in headways)
            
            # 2. good record
            has_good_headway = any(-0.2 <= h < 0 for h in headways)
            
            # 3. priority
            if has_severe_bunching:
                priority = 2.5  
            elif has_good_headway:
                priority = 3.0  
            elif has_medium_bunching:
                priority = 2.0  
            else:
                priority = 1.0  

            replay_buffer.push(states, actions, mean_reward, next_states, priority)
            
            if not is_warmup and len(replay_buffer) > BATCH_SIZE:  # non-warmup episode
                try:
                    # update critic
                    states_b, actions_b, rewards_b, next_states_b = replay_buffer.sample(BATCH_SIZE)
                    
                    states_b_tensor = torch.FloatTensor(states_b.reshape(BATCH_SIZE, -1)).to(device)
                    actions_b_tensor = torch.FloatTensor(actions_b.reshape(BATCH_SIZE, -1)).to(device)
                    rewards_b_tensor = torch.FloatTensor(rewards_b).unsqueeze(1).to(device)
                    next_states_b_tensor = torch.FloatTensor(next_states_b.reshape(BATCH_SIZE, -1)).to(device)

                    stop_ids = states_b_tensor[:, ::STATE_DIM]  
                    
                    # do not train at the last stop
                    non_terminal_mask = (stop_ids != (NUM_STOPS - 1)).all(dim=1)  
                    if non_terminal_mask.sum() == 0:
                        continue

                    states_b_tensor_non_terminal = states_b_tensor[non_terminal_mask]
                    actions_b_tensor_non_terminal = actions_b_tensor[non_terminal_mask]
                    rewards_b_tensor_non_terminal = rewards_b_tensor[non_terminal_mask]
                    next_states_b_tensor_non_terminal = next_states_b_tensor[non_terminal_mask]

                    with torch.no_grad():
                        next_actions_non_terminal = torch.cat([
                            target_actors[i](next_states_b_tensor_non_terminal[:, i*STATE_DIM:(i+1)*STATE_DIM])
                            for i in range(NUM_AGENTS)
                        ], dim=1)
                        target_q_non_terminal = target_critic(next_states_b_tensor_non_terminal, next_actions_non_terminal)
                        y = rewards_b_tensor_non_terminal + GAMMA * target_q_non_terminal

                        y = torch.clamp(y, -TARGET_Q_CLIP, TARGET_Q_CLIP)
                    
                    current_q = critic(states_b_tensor_non_terminal, actions_b_tensor_non_terminal)
                    
                    # check Q
                    if torch.isnan(current_q).any() or torch.isinf(current_q).any():
                        print("Warning: NaN or Inf detected in Q values, skipping this batch")
                        continue
                    
                    current_q = torch.clamp(current_q, -Q_VALUE_CLIP, Q_VALUE_CLIP)
                    
                    q_reg = 1e-3 * current_q.pow(2).mean()  
                    critic_loss = F.smooth_l1_loss(current_q, y) + q_reg
                    
                    # check loss
                    if torch.isnan(critic_loss) or torch.isinf(critic_loss) or critic_loss.item() > 100:
                        print(f"Warning: Unstable critic loss {critic_loss.item()}, skipping this batch")
                        continue
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                    critic_optimizer.step()
                    optimizer_stepped = True

                    episode_q_pred.append(current_q.mean().item())
                    episode_q_target.append(y.mean().item())
                    episode_critic_losses.append(critic_loss.item())
                    
                    q_value_stats['min'].append(current_q.min().item())
                    q_value_stats['max'].append(current_q.max().item())
                    q_value_stats['mean'].append(current_q.mean().item())
                    q_value_stats['std'].append(current_q.std().item())
                    
                    # update actor
                    if t % UPDATE_ACTOR_EVERY == 0:
                        for i in range(NUM_AGENTS):
                            current_states = states_b_tensor[:, i*STATE_DIM:(i+1)*STATE_DIM]
                            stop_ids = current_states[:, 0]

                            # do not train at the last stop
                            non_terminal_mask = (stop_ids != (NUM_STOPS - 1))
                            if non_terminal_mask.sum() == 0:
                                continue  
                            curr_action = actors[i](current_states[non_terminal_mask])
                            joint_actions = []

                            for j in range(NUM_AGENTS):
                                state_j = states_b_tensor[:, j * STATE_DIM:(j + 1) * STATE_DIM]
                                if j == i:
                                    with torch.no_grad():
                                        temp_action = actors[i](state_j)
                                    temp_action[non_terminal_mask] = curr_action 
                                    joint_actions.append(temp_action)
                                else:
                                    with torch.no_grad():
                                        joint_actions.append(actors[j](state_j))
                            # actor_actions = actors[i](current_states)
                            all_actions = torch.cat(joint_actions, dim=1)
                            
                            #actor_loss = -critic(states_b_tensor, all_actions).mean()
                            entropy = -torch.mean(curr_action * torch.log(curr_action + 1e-6))
                            actor_loss = -critic(states_b_tensor, all_actions).mean() - 0.5 * entropy

                            actor_optimizers[i].zero_grad()
                            actor_loss.backward()
                            torch.nn.utils.clip_grad_norm_(actors[i].parameters(), GRAD_CLIP)
                            actor_optimizers[i].step()
                            
                            episode_actor_losses.append(actor_loss.item())
                        
                        # soft update
                        soft_update(target_critic, critic, TAU)
                        for i in range(NUM_AGENTS):
                            soft_update(target_actors[i], actors[i], TAU)
                
                except Exception as e:
                    print(f"Error during training: {str(e)}")
                    continue
            
            states = next_states
            total_reward_for_episode += mean_reward
            episode_r1.extend(r1s)
            episode_r2.extend(r2s)
            episode_actions.extend([a[0] for a in actions])
            count_bunch += sum(1 for i in range(NUM_AGENTS) if states[i][1] <= - 0.9)
            
            if done:
                break
        
        # Update the learning rate only after an optimizer step has been performed
        if optimizer_stepped:
            critic_scheduler.step()
            for scheduler in actor_schedulers:
                scheduler.step()
        
        # record and monitor
        reward_history.append(total_reward_for_episode)
        window_size = min(10, len(reward_history))
        avg_reward = np.mean(reward_history[-window_size:])
        
        if avg_reward > best_reward + IMPROVEMENT_THRESHOLD and episode >= max(WARMUP_EPISODES, 10):
            best_reward = avg_reward
            best_actors = [actor.state_dict() for actor in actors]
            for i, state_dict in enumerate(best_actors):
                torch.save(state_dict, f"best_actor_agent{i}.pth")
            last_improvement = episode
            print(f"++ New Best Avg Reward: {best_reward:.2f}, Episode: {last_improvement} ++")
            patience_counter = 0
        else:
            patience_counter += 1
        
        if episode % 5 == 0:
            mean_r1 = np.mean(episode_r1) if episode_r1 else 0
            mean_r2 = np.mean(episode_r2) if episode_r2 else 0
            mean_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
            mean_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
            mean_action = np.mean(episode_actions) if episode_actions else 0.0
            std_action = np.std(episode_actions) if episode_actions else 0.0
            mean_q_pred = np.mean(episode_q_pred) if episode_q_pred else 0.0
            mean_q_target = np.mean(episode_q_target) if episode_q_target else 0.0
            
            # Q-value mornitor
            q_min = np.mean(q_value_stats['min']) if q_value_stats['min'] else 0.0
            q_max = np.mean(q_value_stats['max']) if q_value_stats['max'] else 0.0
            q_std = np.mean(q_value_stats['std']) if q_value_stats['std'] else 0.0
            
            # headway mornitor
            final_states = env.state
            headway_norms = [final_states[i][1] for i in range(NUM_AGENTS)]
            positive_headway_count = sum(1 for h in headway_norms if h >= 0)  
            negative_headway_count = sum(1 for h in headway_norms if h < 0)   
            
            if episode >= max(WARMUP_EPISODES, 10):
                print(f"{episode:5d}  | {total_reward_for_episode:8.2f} | {avg_reward:7.2f} | "
                  f"{mean_r1:6.3f} {mean_r2:6.3f} {'--':>6} | "
                  f"{mean_action:.3f}±{std_action:.3f} | C:{mean_critic_loss:8.4f} A:{mean_actor_loss:8.4f}|"
                  f"{mean_q_pred:8.4f} | {mean_q_target:8.4f} |"
                  f"{count_bunch} | H+:{positive_headway_count} H-:{negative_headway_count} | Q:[{q_min:.2f},{q_max:.2f}]±{q_std:.2f}")
            else:
                print(f"{episode:5d}  | {total_reward_for_episode:8.2f} | {avg_reward:7.2f} | "
                  f"{mean_r1:6.3f} {mean_r2:6.3f} {'--':>6} | "
                  f"{mean_action:.3f}±{std_action:.3f} | C:{mean_critic_loss:8.4f} A:{mean_actor_loss:8.4f} |"
                  f"H+:{positive_headway_count} H-:{negative_headway_count} | Q:[{q_min:.2f},{q_max:.2f}]±{q_std:.2f}")
        
        # early stop check
        if episode > MIN_EPISODES and patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered. No improvement for {PATIENCE} episodes.")
            patience_counter = 0
            # break
        
        # Q-value tracker
        if q_value_stats['max'] and abs(np.mean(q_value_stats['max'])) > Q_VALUE_CLIP * 0.9:  
            q_explosion_count += 1
            print(f" ! Warning: Q values approaching limits! Episode {episode}, Max Q: {np.mean(q_value_stats['max']):.2f}")
            
            if q_explosion_count >= 5: 
                print(f" ! Q-value explosion：reduce learning rate...")
                for optimizer in actor_optimizers:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.8  
                for param_group in critic_optimizer.param_groups:
                    param_group['lr'] *= 0.8
                q_explosion_count = 0  
        else:
            q_explosion_count = max(0, q_explosion_count - 1)
    
    print("\nTraining finished.")
    print(f"Best average reward: {best_reward:.2f}")
    
    # save training curve (reward-episode)
    plot_training_results(reward_history)
    
    return best_actors, reward_history

# Evaluate
def evaluate_trained_policy():
    print("Evaluating saved best policy...")
    eval_actors = [Actor(STATE_DIM, ACTION_DIM) for _ in range(NUM_AGENTS)]
    for i in range(NUM_AGENTS):
        eval_actors[i].load_state_dict(torch.load(f"best_actor_agent{i}.pth"))
        eval_actors[i].eval()

    eval_env = MultiBusSimEnv(NUM_AGENTS)
    states = eval_env.reset()

    total_eval_reward = 0
    agent_rewards = [0.0 for _ in range(NUM_AGENTS)]  # record the cumulative reward of each agent

    for t in range(NUM_STEP):
        state_tensors = [torch.FloatTensor(states[i]).unsqueeze(0) for i in range(NUM_AGENTS)]
        with torch.no_grad():
            actions = [eval_actors[i](state_tensors[i]).numpy()[0] for i in range(NUM_AGENTS)]
        states, rewards, done, (r1s, r2s) = eval_env.step(actions)
        total_eval_reward += sum(rewards)
        for i in range(NUM_AGENTS):
            agent_rewards[i] += rewards[i] 

    # the best agent with the highest reward
    best_agent = int(np.argmax(agent_rewards))
    print(f" Best performing agent: Agent {best_agent} with reward {agent_rewards[best_agent]:.2f}")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()

if __name__ == "__main__":
    log_file = open("RL_log.txt", "w", encoding="utf-8")
    sys.stdout = Tee(sys.stdout, log_file)
    print("Starting bus control training...")
    best_actors, reward_history = train_bus_controller()
    
    # plot training result
    plot_training_results(reward_history)
    
    # assess and pick the best agent network
    if best_actors is not None:
        print("\nEvaluating best policy...")
        evaluate_trained_policy()
    else:
        print("\nNo best policy found during training.")



