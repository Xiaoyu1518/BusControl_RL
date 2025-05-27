import torch

# import Actor 
from RL_bus_A2386 import Actor  
TARGET_HEADWAY = 600 # same as training
    
# Parameters (same as training)
STATE_DIM = 4       # staete dimension
ACTION_DIM = 1      # action dimension

# Load the best model (Example)
BEST_AGENT = 3
BEST_MODEL_PATH = f"Result/best_actor_agent{BEST_AGENT}.pth"

shared_actor = Actor(STATE_DIM, ACTION_DIM)
shared_actor.load_state_dict(torch.load(BEST_MODEL_PATH))
shared_actor.eval() 
print("Loaded shared actor model from", BEST_MODEL_PATH)

# use normalized headway instead of raw value
def get_normalized_headway(headway):
        """limit headway to [-1, 1]，check stability"""
        normalized = np.clip(1 * (headway - TARGET_HEADWAY) / TARGET_HEADWAY, -1, 1)
        if np.isnan(normalized) or np.isinf(normalized):
            return 0.0  # error
        return normalized
    
# load states (Example)
NUM_STATES = 5
states = [
    [1, 50, 0.2, 0],   
    [1, 150, 0.3, 1],   
    [1, 300, 0.4, 2],   
    [1, 600, 0.6, 3],   
    [1, 800, 0.8, 4]    
]
# normalization
normalized_states = [
    [s[0], get_normalized_headway(s[1]), s[2], s[3]]
    for s in states
]

# Generate actions
actions = []
for i in range(NUM_STATES):
    state_tensor = torch.FloatTensor(normalized_states[i]).unsqueeze(0)  # (1, STATE_DIM)
    with torch.no_grad():
        action = shared_actor(state_tensor).numpy()[0]
    actions.append(action)
    print(f'state: {states[i]}, action: {action}')

print("Generated policies")
