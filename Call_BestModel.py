import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# Parameters (same as training)
STATE_DIM = 4       # state dimension
ACTION_DIM = 1      # action dimension
NUM_AGENTS = 13
TARGET_HEADWAY = 600
from RL_bus_A2386 import Actor 

def deploy_action_per_vehicle(state_dict, actor_models):
    """
    输入：
        state_dict: dict，键为 bus_id，值为 state 向量（长度为4）
        actor_models: list，长度为 num_agents，每个是一个已加载的 Actor 模型
    返回：
        actions: dict，键为 bus_id，值为输出动作
    """
    actions = {}
    
    for bus_id, state in state_dict.items():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, 4]
        
        # 第四个元素是 agent_id
        agent_id = int(state[3])
        assert 0 <= agent_id < len(actor_models), f"Invalid agent_id {agent_id} for {bus_id}"

        actor = actor_models[agent_id]
        with torch.no_grad():
            action = actor(state_tensor).numpy()[0]

        actions[bus_id] = action
    
    return actions

# 加载模型
actor_models = []

for i in range(NUM_AGENTS):
    model = Actor(STATE_DIM, ACTION_DIM)
    model.load_state_dict(torch.load(f"Result\\247577\\best_actor_agent{i}.pth", map_location="cpu"))
    model.eval()
    actor_models.append(model)

# use normalized headway instead of raw value
def get_normalized_headway(headway):
    """limit headway to [-1, 1]，check stability"""
    normalized = np.clip(1 * (headway - TARGET_HEADWAY) / TARGET_HEADWAY, -1, 1)
    if np.isnan(normalized) or np.isinf(normalized):
        return 0.0  # error
    return normalized


# 测试
states = [
    [5, 100, 0.3, 0],   
    [5, 100, 0.3, 1],   
    [5, 100, 0.3, 2],   
    [5, 100, 0.3, 3],   
    [5, 100, 0.3, 4],
    [5, 100, 0.3, 5],
    [5, 100, 0.3, 6],
    [5, 100, 0.3, 7],
    [5, 100, 0.3, 8],
]
NUM_STATES = len(states)

# normalization
normalized_states = [
    [s[0], get_normalized_headway(s[1]), s[2], s[3]]
    for s in states
]

# 转成带 bus_id 的 state_dict 格式
state_dict = {f"bus_{i}": state for i, state in enumerate(normalized_states)}

actions = deploy_action_per_vehicle(state_dict, actor_models)

for bus, act in actions.items():
    print(f"{bus} → agent {int(state_dict[bus][3])} → action: {act}")
