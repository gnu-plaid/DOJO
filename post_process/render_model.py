import pygame
from sympy.core.random import random

from env_dojo.render import Render
from agents.agent import SAC
import numpy as np
import torch

def render(test_model,reward , scaling_ratio = 0.7, print_Qvalue = True,mode = 'model'):
    pygame.init()
    env_render = Render(scaling_ratio=scaling_ratio)
    state_size = env_render.state_dim
    action_size = env_render.action_dim

    agent = SAC(state_size=state_size, action_size=action_size, random_seed=0,
                  action_prior="uniform", device='cpu')  # "normal"

    # using the turn
    agent.actor_local.load_state_dict(torch.load('../checkpoints/SAC_actor_{}_r={}.pth'.format(test_model, reward)))
    agent.critic1.load_state_dict(torch.load('../checkpoints/SAC_critic_{}_r={}.pth'.format(test_model, reward)))

    bad_end_list = ['collision', 'time out', 'out range', 'X']

    for _ in range(100):
        done = False
        env_render.reset(position_required=True, position=[20, 20, 180])
        while not done and env_render.time_since_start < 100:
            # print(game)

            state = env_render.get_state()
            # print(state)
            state_tensor = torch.tensor([state], dtype=torch.float)

            if mode == 'model':
                command = agent.act(state)[0].detach().numpy()

            else:
                command = env_render.act_by_rule(loc=10,scale = 3)


            action_tensor = torch.tensor([command], dtype=torch.float)
            q_value = agent.critic1(state_tensor, action_tensor)[0].detach().numpy()

            if print_Qvalue:
                print(q_value)

            env_render.play(command=command)
            done, reward, _ = env_render.check_done()

            if env_render.ending in bad_end_list:
                done = True
            env_render.pause(done)

if __name__=="__main__":
    render(test_model=20000,reward=-3,mode='model')