from environment import *
from utils import *
from agent import *
from DQN import *
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

environment = DQNEnvironment(device=device)
model = DQN(num_actions=4)
model.load_the_model()
agent = Agent(model=model,
              device=device,
              gamma=0.99,
              epsilon=1.0,
              min_epsilon=0.05,
              exploration_episodes=5000,
              number_of_actions=4,
              lr=0.000025,
              memory_capacity=100000,
              batch_size=32)

agent.train(env=environment, epochs=25000, batch_identifier=0)

test_environment = DQNEnvironment(device=device, render_mode='human')

agent.test(env=test_environment)

# Display the image
#display_observation_image(state)
