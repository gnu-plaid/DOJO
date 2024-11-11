import os
import torch
from env_dojo.env import Dojo
import numpy as np
from agents.agent import SAC
import time
import random
from collections import deque

def check_path():
    folder_path = "./checkpoints"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder'{folder_path}' has been created.")
    else:
        print(f"Folder '{folder_path}' exists.")

def train():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # set seed
    seed = 0

    # episodes
    n_episodes = 20000

    # training per episodes
    n_trains = 32

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
    percentage_deque = deque(maxlen=n_trains)
    demo_deque = deque(maxlen=100)

    # rule hyperparameter
    rule_threshold = 0.01

    # score_list
    score_list = []
    # percentage_list
    percentage_list = []

    now_path = os.getcwd()

    for i_episode in range(1, n_episodes + 1):
        env.reset()
        score = 0
        state = env.get_state()
        explore_by_rule = False
        done = False
        DATA_FLAG = 0
        while not done and  env.time_since_start < env.time_threshold:
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
            done, reward, hitting_point = env.check_done()
            next_state = env.get_state()

            action = torch.from_numpy(action_v).view(1, -1).float()
            agent.step(state, action, reward, next_state, done, DATA_FLAG)

            state = next_state
            score += reward

        if explore_by_rule:
            rule_done = False
            env.reset()
            env.load(state_dic, oppo_state)
            # rule data flag
            DATA_FLAG = 1
            demo_deque.append(1)
            while not rule_done and env.time_since_start < env.time_threshold:
                state = env.get_state()
                action_v = np.array(env.act_by_rule(loc=10, scale=3))
                env.step(robot_command=action_v)
                rule_done, reward, _ = env.check_done()
                next_state = env.get_state()
                action = torch.from_numpy(action_v).view(1, -1).float()
                agent.step(state, action, reward, next_state, rule_done, DATA_FLAG)
        else:
            demo_deque.append(0)
            if reward == 0:
                env.update_success_rate(hitting_point)

        demo_freq = np.mean(demo_deque)
        for _ in range(n_trains):
            agent.update(demo_freq,n_episode=n_episodes)
            percentage_deque.append(agent.data_per)

        # mean score
        scores_deque.append(score)

        # lists
        score_list.append(score)
        percentage_list.append(np.mean(percentage_deque))

        print('\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f} Demo Data percentage： {:.2f}  demo freq: {:.2f}'.format(i_episode, score, np.mean(scores_deque), np.mean(percentage_deque),demo_freq), env.guidance_success_rate,end="")
        if i_episode % print_every == 0:
            print(
                '\rEpisode {} Reward: {:.2f}  Average100 Score: {:.2f} Demo Data percentage： {:.2f}  demo freq: {:.2f}'.format(
                    i_episode, score, np.mean(scores_deque), np.mean(percentage_deque), demo_freq),
                env.guidance_success_rate)
            # save

            actor_path = now_path + '/checkpoints/' + 'SAC_actor_{}.pth'.format(i_episode)
            critic_path = now_path + '/checkpoints/' + 'SAC_critic_{}.pth'.format(i_episode)
            torch.save(agent.actor_local.state_dict(), actor_path)
            torch.save(agent.critic1.state_dict(), critic_path)

    np.savetxt(now_path + '/checkpoints'+ '/score_list_{}.txt'.format(n_episodes), np.array(score_list))
    np.savetxt(now_path + '/checkpoints' + '/demo_percentage_list_{}.txt'.format(n_episodes), np.array(percentage_list))
    end_time = time.time()
    print("Training took: {} min".format((end_time - start_time) / 60))

if __name__ == '__main__':
    check_path()
    train()