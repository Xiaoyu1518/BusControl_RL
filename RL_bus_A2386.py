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
travel_time_df = pd.read_excel(f"TravelTime/travel_time_norm_{route_id}.xlsx")
stop_id = travel_time_df['start_stop_id']

# Hyperparameters
NUM_AGENTS = 12
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
GAMMA = 0.995
TAU = 0.005
BUFFER_SIZE = 100000
BATCH_SIZE = 64
MAX_EPISODES = 400
NUM_STEP = 6 * (len(stop_id) + 1)
UPDATE_ACTOR_EVERY = 2
WARMUP_EPISODES = 30
WARMUP_STEPS = len(stop_id) + 1

# Reward scale factors & Reward weights
REWARD_SCALE = 1.0
HEADWAY_SCALE = 1.0
HOLDING_SCALE = 0.2
BUNCHING_SCALE = 0.8

# Parameters
NUM_STOPS = len(stop_id) + 1
TARGET_HEADWAY = 600
MAX_HOLD = 120
MAX_OCCUPANCY = 1.0
BUNCHING_THRESHOLD = 0.1 * TARGET_HEADWAY
CAPACITY = 140

# Exploration
EPSILON_START = 1.0
EPSILON_END = 0.2
EPSILON_DECAY = 0.9998

# Training stability and Normalization
GRAD_CLIP = 0.5
WEIGHT_DECAY = 1e-5
MAX_VALUE = 1e6
REWARD_CLIP = 5.0

ALIGHT_TIME = 2  # seconds per passenger
BOARD_TIME = 3  # seconds per passenger
DOOR_TIME = 5  # seconds (open/close)
REST_TIME = 180  # seconds (rest)

# Early stop for monitor
PATIENCE = 50
MIN_EPISODES = 150
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
        
        # Feature layer
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_LAYERS[0]),
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
        x = self.ln_input(state)
        x = self.feature_net(x)
        
        # Residual
        for block in self.residual_blocks:
            identity = x
            x = block(x)
            x = F.relu(x + identity)
            
        x = self.output_net(x)
        return torch.sigmoid(x)

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
        return self.q_net(combined)

# ==== Replay Buffer ====
class ReplayBuffer:
    def __init__(self, max_size=BUFFER_SIZE):
        self.buffer = deque(maxlen=max_size)

    def push(self, states, actions, reward, next_states):
        self.buffer.append((states, actions, reward, next_states))

    def sample(self, batch_size):
        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.stack, zip(*samples))
        return states, actions, rewards, next_states

    def __len__(self):
        return len(self.buffer)
        
# ==== Bus Environment ====
class MultiBusSimEnv:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        # self.segment_distances = np.array(STOP_NETWORK)
        self.constant_speed = 33
        self.global_time = np.zeros(num_agents)
        self.arrival_history = {stop_id: [] for stop_id in range(NUM_STOPS)}
        self.last_departure_times = np.zeros(num_agents)
        # self.cumulative_delays = np.zeros(num_agents)
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
        random_headways = np.random.uniform(0.8 * TARGET_HEADWAY, 1.2 * TARGET_HEADWAY, size=self.num_agents)
        random_headways = np.clip(random_headways, 0.8 * TARGET_HEADWAY, 1.2 * TARGET_HEADWAY)
        self.state[:, 1] = random_headways
        
        # randomly set initial occupancy
        random_occupancy = np.random.uniform(0, 0.3, size=self.num_agents)
        self.state[:, 2] = random_occupancy 
        
        # the initial time at the first stop (based on headway above)
        self.global_time = np.array([sum(random_headways[:i]) for i in range(self.num_agents)], dtype=np.float32)
        
        # sort by global_time to generate sequential numbers
        order_ids = np.argsort(self.global_time)  # the first departure, id = 0, then 1, 2, 3, ...
        self.state[:, 3] = order_ids  
        
        return self.state.copy()

    def calculate_reward_components(self, headway, hold_time, occupancy):
        try:
            # 1. Headway deviation reward (r1)
            norm_headway = abs(self.get_normalized_headway(headway))
            if norm_headway <= 0.2:
                r1 = 1.0 - 0.2 * norm_headway # 1.0 - norm_headway
            elif norm_headway <= 0.5:
                r1 = 1.0 - 0.5 * norm_headway # 0.8 * (1 - norm_headway)
            elif norm_headway <= 0.9:
                r1 = 1.0 - 0.9 * norm_headway # 0.5 * (1 - norm_headway)
            else:
                r1 = 1.0 - 1.0 * norm_headway
            
            # 2. Holding and delay penalty (r2)
            norm_hold = np.clip(hold_time / MAX_HOLD, 0, 1)
            # norm_delay = np.clip(delay / TARGET_HEADWAY, 0, 1)
            if headway <= TARGET_HEADWAY:
                r2 = - norm_hold * occupancy * (1 - norm_headway)
            else:
                r2 = - norm_hold * occupancy * (1 + norm_headway)
            
            # 3. Bunching prevention (r3)
            if headway < BUNCHING_THRESHOLD:
                r3 = - 0.2 * (1 - headway / BUNCHING_THRESHOLD)
            else:
                r3 = 0
            
            # smooth with tanh
            r1 = np.tanh(r1)
            r2 = np.tanh(r2)
            r3 = np.tanh(r3)
            
            return r1, r2, r3 
            
        except Exception as e:
            print(f"Warning: Error in reward calculation: {str(e)}")
            return 0.0, 0.0, 0.0

    def step(self, actions, training=True):
        rewards = []
        r1s, r2s, r3s = [], [], []
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
                
                # 3. travel time
                data_columns = travel_time_df.columns[3:]
                if cur_stop_id < NUM_STOPS - 1:  # non-final-stop
                    row = travel_time_df[(travel_time_df['start_stop_id'] == stop_id.iloc[cur_stop_id])].iloc[0]
                    mu = float(row["mu"])
                    sigma = float(row["sigma"])
                    if pd.notna(row['mu']) and pd.notna(row['sigma']) and mu > 0 and sigma > 0:
                        base_travel_time = np.random.normal(row['mu'], min(5, row['sigma']))
                    else:
                        max_col_name = row[data_columns].astype(float).idxmax()
                        base_travel_time = np.random.normal(max_col_name, 5) # sample_from_kde_each_row(row, data_columns)
                    base_travel_time = max(base_travel_time, 30)
                else:  
                    base_travel_time = 0

                # 4. arrival time at next stop
                departure_time = prev_arrival[i] + total_dwell if cur_stop_id != NUM_STOPS - 1 else base_dwell + REST_TIME
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
                    r1, r2, r3 = self.calculate_reward_components(
                        headway,
                        float(actions[i][0]) * MAX_HOLD if next_stop != NUM_STOPS - 1 else 0,
                        new_occupancy
                    )
                    
                    reward = REWARD_SCALE * (
                        HEADWAY_SCALE * r1 + 
                        HOLDING_SCALE * r2 + 
                        BUNCHING_SCALE * r3
                    )
                    
                    reward = np.clip(reward, -REWARD_CLIP, REWARD_CLIP)
                    rewards.append(reward)
                    r1s.append(r1)
                    r2s.append(r2)
                    r3s.append(r3)

            except Exception as e:
                print(f"Error in state update for agent {i}: {str(e)}")
                rewards.append(0.0)
                r1s.append(0.0)
                r2s.append(0.0)
                r3s.append(0.0)
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
        while len(r3s) < self.num_agents: r3s.append(0.0)

        return self.state.copy(), rewards, self.step_count >= NUM_STEP, (r1s, r2s, r3s)

# Utilities
def soft_update(target, source, tau):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)

def plot_training_stats(actor_losses, critic_losses, r1_list, r2_list, r3_list):
    # safety check
    if not all([actor_losses, critic_losses, r1_list, r2_list, r3_list]):
        print("Warning: Some training statistics are empty")
        return
        
    plt.figure(figsize=(14, 8))

    plt.subplot(2, 1, 1)
    for i in range(NUM_AGENTS):
        if actor_losses[i]:  # safety check
            plt.plot(actor_losses[i], marker='o', linestyle='none', markersize=1.5, label=f'Actor {i}')
    if critic_losses:  # safety check
        plt.plot(critic_losses, marker='o', linestyle='none', markersize=1.5, label='Critic', linewidth=2, color='black')
    plt.title("Actor & Critic Losses")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    if r1_list:  # safety check
        plt.plot(r1_list, marker='o', linestyle='none', markersize=1.5, label='r1: Headway Dev')
    if r2_list:  
        plt.plot(r2_list, marker='o', linestyle='none', markersize=1.5, label='r2: Holding Penalty')
    if r3_list:  
        plt.plot(r3_list, marker='o', linestyle='none', markersize=1.5, label='r3: Bunching Penalty')
    plt.title("Reward Components Over Time")
    plt.xlabel("Training Steps")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("Result/training_debug_stats.png")
    # plt.show()

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
    plt.savefig("Result/training_reward_curve.png")
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
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPISODES, eta_min=LR_ACTOR * 0.1)
        actor_optimizers.append(opt)
        actor_schedulers.append(scheduler)

    critic_optimizer = optim.Adam(critic.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
    critic_scheduler = optim.lr_scheduler.CosineAnnealingLR(critic_optimizer, T_max=MAX_EPISODES, eta_min=LR_CRITIC * 0.1)
    
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
    
    print(f"Training Start at {time.strftime('%Y-%m-%d %H:%M:%S')} ")
    print("Episode |  Reward  | Avg(10) |  r1     r2     r3   |   Act  |     Critic Loss     |  Q_pred   | Q_target")
    print("-" * 100)

    for episode in range(MAX_EPISODES):
        states = env.reset()
        states = normalize_state(states)
        total_reward_for_episode = 0
        episode_r1, episode_r2, episode_r3 = [], [], []
        episode_critic_losses, episode_actor_losses = [], []
        episode_actions = []
        
        # calculate epsilon for exploration
        epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY ** max(0, episode - WARMUP_EPISODES)))
        
        # track whether the optimizer step was executed
        optimizer_stepped = False
        
        for t in range(NUM_STEP):
            is_warmup = t < WARMUP_STEPS or episode < WARMUP_EPISODES
            actions = []
            for i in range(NUM_AGENTS):
                if is_warmup or np.random.rand() < epsilon:
                    # Heuristic strategies
                    cur_stop_id = int(states[i][0])
                    headway_norm = states[i][1]
                    occupancy = states[i][2]
                    
                    if cur_stop_id == NUM_STOPS - 1:
                        action_val = 0.0
                    elif headway_norm < -0.8:  # headway too small
                        action_val = np.random.uniform(0.8, 1.0)
                    elif headway_norm > 0.8:  # headway too large
                        action_val = np.random.uniform(0.0, 0.2)
                    else:
                        action_val = np.random.uniform(0.2, 0.8)
                    actions.append(np.array([action_val]))
                else:
                    state_tensor = torch.FloatTensor(states[i]).unsqueeze(0).to(device)
                    with torch.no_grad():
                        action_val = actors[i](state_tensor).cpu().numpy()[0]
                        # add noise for exploration
                        noise_scale = 0.1 * (1 - episode / MAX_EPISODES)
                        action_val += np.random.normal(0, noise_scale)
                        action_val = np.clip(action_val, 0, 1)
                    actions.append(action_val)

            next_states, rewards, done, (r1s, r2s, r3s) = env.step(actions)
            next_states = normalize_state(next_states)
            
            # normalize rewards
            rewards = [normalize_reward(r) for r in rewards]
            mean_reward = np.mean(rewards)
            
            # save to buffer
            replay_buffer.push(states, actions, mean_reward, next_states)
            
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

                        # y = torch.clamp(y, -Q_VALUE_CLIP, Q_VALUE_CLIP)
                    
                    current_q = critic(states_b_tensor_non_terminal, actions_b_tensor_non_terminal)
                    q_reg = 1e-3 * torch.mean(current_q ** 2)
                    critic_loss = F.smooth_l1_loss(current_q, y) + q_reg
                    
                    critic_optimizer.zero_grad()
                    critic_loss.backward()
                    torch.nn.utils.clip_grad_norm_(critic.parameters(), GRAD_CLIP)
                    critic_optimizer.step()
                    optimizer_stepped = True
                    
                    episode_critic_losses.append(critic_loss.item())
                    
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
                            actor_loss = -critic(states_b_tensor, all_actions).mean() - 0.1 * entropy

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
            episode_r3.extend(r3s)
            episode_actions.extend([a[0] for a in actions])
            
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
                torch.save(state_dict, f"Result/best_actor_agent{i}.pth")
            print(f"++ New Best Avg Reward: {best_reward:.2f}")
            last_improvement = episode
            patience_counter = 0
        else:
            patience_counter += 1
        
        if episode % 5 == 0:
            mean_r1 = np.mean(episode_r1) if episode_r1 else 0
            mean_r2 = np.mean(episode_r2) if episode_r2 else 0
            mean_r3 = np.mean(episode_r3) if episode_r3 else 0
            mean_critic_loss = np.mean(episode_critic_losses) if episode_critic_losses else 0
            mean_actor_loss = np.mean(episode_actor_losses) if episode_actor_losses else 0
            mean_action = np.mean(episode_actions) if episode_actions else 0.0
            
            
            if episode >= max(WARMUP_EPISODES, 10):
                print(f"{episode:5d}  | {total_reward_for_episode:8.2f} | {avg_reward:7.2f} | "
                  f"{mean_r1:6.3f} {mean_r2:6.3f} {mean_r3:6.3f} | "
                  f"{mean_action:.3f} | C:{mean_critic_loss:8.4f} A:{mean_actor_loss:8.4f}|"
                  f"{current_q.mean().item():8.4f} | {y.mean().item():8.4f}")
            else:
                print(f"{episode:5d}  | {total_reward_for_episode:8.2f} | {avg_reward:7.2f} | "
                  f"{mean_r1:6.3f} {mean_r2:6.3f} {mean_r3:6.3f} | "
                  f"{mean_action:.3f} | C:{mean_critic_loss:8.4f} A:{mean_actor_loss:8.4f}")
        
        # early stop check
        if episode > MIN_EPISODES and patience_counter >= PATIENCE:
            print(f"\nEarly stopping triggered. No improvement for {PATIENCE} episodes.")
            patience_counter = 0
            # break
    
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
        states, rewards, done, (r1s, r2s, r3s) = eval_env.step(actions)
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



