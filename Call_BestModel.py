import torch

# import Actor 
from RL_5 import Actor  

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

# load states (Example)
NUM_STATES = 5
states = [
    [1, 50, 0.2, 0],   
    [1, 150, 0.3, 1],   
    [1, 300, 0.4, 2],   
    [1, 600, 0.6, 3],   
    [1, 800, 0.8, 4]    
]

# Generate actions
actions = []
for i in range(NUM_STATES):
    state_tensor = torch.FloatTensor(states[i]).unsqueeze(0)  # (1, STATE_DIM)
    with torch.no_grad():
        action = shared_actor(state_tensor).numpy()[0]
    actions.append(action)
    print(f'state: {states[i]}, action: {action}')

print("Generated policies")
