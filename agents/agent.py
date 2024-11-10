import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
from agents.model import Actor_SAC,Critic_SAC
from agents.PER_buffer import PrioritizedReplay

class SAC:
    def __init__(self,
                 state_size, action_size,
                 random_seed,
                 device,
                 n_episode,
                 LR_ACTOR = 5e-4,
                 LR_CRITIC = 5e-4,
                 WEIGHT_DECAY = 0,
                 BUFFER_SIZE = 1e6,
                 BATCH_SIZE = 1024,
                 GAMMA = 0.99,
                 TAU = 1e-2,
                 action_prior="uniform"):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.n_episode = n_episode
        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR)
        self._action_prior = action_prior

        print("Using: ", device)

        # Actor Network
        self.actor_local = Actor_SAC(state_size, action_size, random_seed,device).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic_SAC(state_size, action_size, random_seed).to(device)
        self.critic2 = Critic_SAC(state_size, action_size, random_seed).to(device)

        self.critic1_target = Critic_SAC(state_size, action_size, random_seed).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic_SAC(state_size, action_size, random_seed).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        # para
        self.device = device
        self.BATCH_SIZE = BATCH_SIZE
        self.GAMMA = GAMMA
        self.TAU =TAU
        # Replay memory
        self.memory = PrioritizedReplay(capacity=int(BUFFER_SIZE))

        self.data_per = 0

    def step(self, state, action, reward, next_state, done, DATA_FLAG):
        """Save experience in replay memory"""
        self.memory.push(state, action, reward, next_state, done, DATA_FLAG)

    def update(self,demo_priority):
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.BATCH_SIZE:
            experiences = self.memory.sample(self.BATCH_SIZE)
            self.learn(experiences, self.GAMMA,demo_priority)

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).unsqueeze(0).float().to(self.device)
        action = self.actor_local.get_action(state).detach()
        return action

    def learn(self,experiences, gamma, episode):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, idx, weights,DATA_FLAG = experiences
        states = torch.FloatTensor(np.float32(states)).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.cat(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)
        DATA_FLAG = torch.FloatTensor(DATA_FLAG).unsqueeze(1)

        demo_percentage = DATA_FLAG.sum() / DATA_FLAG.numel()
        self.data_per = round(demo_percentage.item(),2)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))
        Q_target2_next = self.critic2_target(next_states.to(self.device), next_action.squeeze(0).to(self.device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next).cpu()

        # Compute Q targets for current states (y_i)
        Q_targets = rewards.cpu() + (
                gamma * (1 - dones.cpu()) * (Q_target_next - self.alpha * log_pis_next.mean(1).unsqueeze(1).cpu()))

        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        td_error1 = Q_targets.detach() - Q_1  # ,reduction="none"
        td_error2 = Q_targets.detach() - Q_2
        critic1_loss = 0.5 * (td_error1.pow(2) * weights).mean()
        critic2_loss = 0.5 * (td_error2.pow(2) * weights).mean()

        _lambda = max(-1, 1 - episode * 2.0 /self.n_episode)

        prios =(
                torch.clamp(torch.abs(td_error1 + td_error2) / 2.0 + _lambda * DATA_FLAG,min = 0)
             + 1e-5).squeeze()

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        self.memory.update_priorities(idx, prios.data.cpu().numpy())

        # ---------------------------- update actor ---------------------------- #

        alpha = torch.exp(self.log_alpha)
        # Compute alpha loss
        actions_pred, log_pis = self.actor_local.evaluate(states)
        alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        self.alpha = alpha
        # Compute actor loss
        if self._action_prior == "normal":
            policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size),
                                              scale_tril=torch.ones(self.action_size).unsqueeze(0))
            policy_prior_log_probs = policy_prior.log_prob(actions_pred)
        elif self._action_prior == "uniform":
            policy_prior_log_probs = 0.0

        actor_loss = ((alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(
            0)).cpu() - policy_prior_log_probs) * weights).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic1, self.critic1_target, self.TAU)
        self.soft_update(self.critic2, self.critic2_target, self.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# wait, can SAC use NStepBackup?
# doesn't make sense on N-step Q-value update
# class NStepBackup():
#   pass