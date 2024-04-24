import os
import torch
import torch.nn as nn
from torch.optim import Adam

from .models import ActorCritic
from .buffer import RolloutBuffer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, total_updates, action_std_init=0.6):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.total_updates = total_updates
        self.action_std = action_std_init
        self.save_interval = max(total_updates // 10, 1)  # Ensure at least one save

        # Directory for saving models
        self.checkpoint_dir = 'tmp/ppo'
        os.makedirs(self.checkpoint_dir, exist_ok=True)  # Simplified folder creation

        self.buffer = RolloutBuffer(device=device)
        self.policy = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.optimizer = Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])
        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        if new_action_std < 0:
            raise ValueError("Standard deviation must be non-negative")
        self.action_std = new_action_std
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        new_action_std = max(self.action_std - action_std_decay_rate, min_action_std)
        if new_action_std != self.action_std:
            print("Updated action std: ", new_action_std)
            self.set_action_std(new_action_std)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        # Ensure that act returns action, action_logprob, and state_val
        action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.store(state, action, action_logprob, state_val, None, None)  # Temporarily store None for reward and done
        return action.detach().cpu().numpy().flatten(), action_logprob, state_val


    def update(self):
        # Prepare Mini-Batch for training from buffer
        states, actions, log_probs, state_vals, rewards, dones = self.buffer.get_data()

        # Calculate discounted rewards and advantages
        rewards_path = [self.calculate_discounted_reward(rewards[i:], dones[i:]) for i in range(len(rewards))]
        rewards = torch.tensor(rewards_path, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        cumulative_loss = 0.0

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            ratios = torch.exp(logprobs - log_probs)
            advantages = rewards - state_values.detach()
            print(f"Ratios shape: {ratios.shape}")
            print(f"Advantages shape: {advantages.shape}")

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            cumulative_loss += loss.mean().item()

        # Update the old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.buffer.clear()

        return cumulative_loss / self.K_epochs  # Return average loss over the epochs


    def calculate_discounted_reward(self, rewards, dones):
        discounted_reward = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
        return discounted_reward

    def save(self, filename):
        torch.save(self.policy.state_dict(), filename)

    def load(self, filename):
        self.policy.load_state_dict(torch.load(filename, map_location=device))
        self.policy_old.load_state_dict(self.policy.state_dict())
