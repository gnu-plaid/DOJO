import pygame
from sympy.core.random import random

from env_dojo.render import Render
from agents.agent import SAC
import numpy as np
import torch

pygame.init()
env_render = Render(scaling_ratio=0.7)
state_size = env_render.state_dim
action_size = env_render.action_dim

agent = SAC(state_size=state_size, action_size=action_size, random_seed=0,
              action_prior="uniform", device='cpu')  # "normal"

# the turns in
turns = 5700

agent.actor_local.load_state_dict(torch.load('../checkpoints/SAC_actor_5700.pth'))
agent.critic1.load_state_dict(torch.load('../checkpoints/SAC_critic_5700.pth'))

for _ in range(100):
    done = False
    env_render.reset(position_required=False, position=[475, 615, 180])
    while not done and env_render.time_since_start < 100:
        # print(game)
        state = env_render.get_state()
        # print(state)
        state_tensor = torch.tensor([state], dtype=torch.float)
        command = agent.act(state)[0].detach().numpy()

        # command = env_render.act_by_rule(loc=10,scale = 3)

        action_tensor = torch.tensor([command], dtype=torch.float)
        q_value = agent.critic1(state_tensor, action_tensor)[0].detach().numpy()

        print(q_value)

        # command = game.act_by_rule(loc=15, scale=3)
        # command = [0,0,0,1]
        # print(game.robo_thrust_inspection())
        env_render.play(command=command)

        done, reward, _ = env_render.check_done()
        env_render.pause(done)