import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

"""
The model should be called at the time of arrival of the bus.
    The first state: the arriving stop ID.
    The second state: the headway (the time gap of the arrival time of the previous bus at this stop)
    The third state: occupancy
    The fourth state: the bus's position in the fleet of the route (e.g., the first bus, the second bus, etc)

"""

# Parameters (same as training)
STATE_DIM = 4       # state dimension
ACTION_DIM = 1      # action dimension
NUM_AGENTS = 13
TARGET_HEADWAY = 600
from RL_bus_A2386 import Actor 

def deploy_action_per_vehicle(state_dict, actor_models):
    actions = {}
    
    for bus_id, state in state_dict.items():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)  # [1, 4]
        agent_id = int(state[3])
        assert 0 <= agent_id < len(actor_models), f"Invalid agent_id {agent_id} for {bus_id}"

        actor = actor_models[agent_id]
        with torch.no_grad():
            action = actor(state_tensor).numpy()[0]

        actions[bus_id] = action
    
    return actions

# load models
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

# Test
## TODO: whrap this into a FAST API, which makes a get request to GTFS RT every 1 min, estimates bus bunching and sends the result to a Kafka topic
## TODO: This should be replaced with the data from GTFS RT (api/v1/vehicle-positions) 
## TODO: 0.3 represents 30% crowding but in GTFS realtime there are only categories -> need to convert GTFS categories to %
## TODO: I am not sure in the GTFS realtime there is a consequitive number of the vehicle

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

# transform
state_dict = {f"bus_{i}": state for i, state in enumerate(normalized_states)}

actions = deploy_action_per_vehicle(state_dict, actor_models)


for bus, act in actions.items():
    print(f"{bus} → agent {int(state_dict[bus][3])} → action: {act}")
