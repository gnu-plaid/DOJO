import os
import torch
from env_dojo.env import Dojo
import numpy as np
from agents.agent import SAC
import time
import random
from collections import deque

def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set seed
    seed = 0

    # training episodes
    n_episodes = 5000

    # save per episodes
    print_every = 500

    # initialize Dojo
    env = Dojo()
    np.random.seed(seed)
    state_size = env.state_dim
    action_size = env.action_dim

    #initialize agent
    agent = SAC(state_size=state_size, action_size=action_size, random_seed=seed,action_prior="uniform",device=device)  # "normal"
    start_time = time.time()
    scores_deque = deque(maxlen=print_every)

    # rule hyperparameter
    rule_threshold = 0.01
    for i_episode in range(1, n_episodes + 1):
        env.reset()
        score = 0
        state = env.get_state()
        explore_by_rule = False
        done = False
        DATA_FLAG = 0
        while not done and  env.time_since_start <= env.time_threshold:
            # record the starting point
            if_rule = (env.time_since_start / env.time_threshold) * rule_threshold
            if if_rule > random.random() and not explore_by_rule:
                explore_by_rule = True
                state_dic, oppo_state = env.save()

            # get action
            action = agent.act(state)
            action_v = action[0].numpy()

            # step
            env.step(action_v)

            # get done reward next_state
            done, reward, _ = env.check_done()
            next_state = env.get_state()

            action = torch.from_numpy(action_v).view(1, -1).float()
            agent.step(state, action, reward, next_state, done, DATA_FLAG)

            state = next_state
            score += reward

        if not explore_by_rule and reward == 0:
            env.update_success_rate(_)
        elif explore_by_rule:
            rule_done = False
            env.reset()
            env.load(state_dic,oppo_state)
            # rule data flag
            DATA_FLAG = 1
            while not rule_done and  env.time_since_start <= env.time_threshold:
                state = env.get_state()
                action_v = np.array(env.act_by_rule(loc=10, scale=3))
                env.step(robot_command=action_v)
                rule_done,reward,_ = env.check_done()
                next_state = env.get_state()
                action = torch.from_numpy(action_v).view(1, -1).float()
                agent.step(state, action, reward, next_state, rule_done, DATA_FLAG)

        # update the
        for _ in range(32):
            agent.update()

        scores_deque.append(score)

        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), env.guidance_success_rate,
              end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average100 Score: {:.2f}'.format(i_episode, score,
                                                                                  np.mean(scores_deque)))

            now_path = os.getcwd()
            actor_path = now_path + '/checkpoints/' + 'SAC_actor_{}.pth'.format(i_episode)
            critic_path = now_path + '/checkpoints/' + 'SAC_critic_{}.pth'.format(i_episode)
            torch.save(agent.actor_local.state_dict(), actor_path)
            torch.save(agent.critic1.state_dict(), critic_path)


    end_time = time.time()
    print("Training took: {} min".format((end_time - start_time) / 60))

if __name__ == '__main__':
    train()