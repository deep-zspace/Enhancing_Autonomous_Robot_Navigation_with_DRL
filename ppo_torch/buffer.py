import torch

class RolloutBuffer:
    def __init__(self, device):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.state_values = []
        self.rewards = []
        self.is_terminals = []
        self.device = device

    def store(self, state, action, logprob, state_value, reward, done):
        action_dim = len(action)  # Assuming action is a list or a numpy array

        state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
        action_tensor = torch.tensor(action, dtype=torch.float32).view(1, action_dim).to(self.device)
        logprob_tensor = torch.tensor(logprob, dtype=torch.float32).to(self.device)
        state_value_tensor = torch.tensor(state_value, dtype=torch.float32).to(self.device)
        reward_tensor = torch.tensor(reward, dtype=torch.float32).to(self.device)
        done_tensor = torch.tensor(done, dtype=torch.bool).to(self.device)

        self.states.append(state_tensor)
        self.actions.append(action_tensor)
        self.logprobs.append(logprob_tensor)
        self.state_values.append(state_value_tensor)
        self.rewards.append(reward_tensor)
        self.is_terminals.append(done_tensor)



            


    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.state_values[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def to_tensor(self):
        # Only convert lists to tensors if not already done in store method
        self.states = torch.stack(self.states).to(self.device) if self.states else torch.empty((0, self.state_dim), device=self.device)
        self.actions = torch.stack(self.actions).to(self.device) if self.actions else torch.empty((0, self.action_dim), device=self.device)
        self.logprobs = torch.stack(self.logprobs).to(self.device) if self.logprobs else torch.empty((0,), device=self.device)
        self.rewards = torch.tensor(self.rewards, dtype=torch.float32).to(self.device) if self.rewards else torch.empty((0,), device=self.device)
        self.state_values = torch.stack(self.state_values).to(self.device) if self.state_values else torch.empty((0,), device=self.device)
        self.is_terminals = torch.tensor(self.is_terminals, dtype=torch.bool).to(self.device) if self.is_terminals else torch.empty((0,), device=self.device)

    def get_data(self):
        """Convert all stored lists to tensors and return them."""
        self.to_tensor()  # Ensure all lists are tensors
        return (
            self.states, self.actions, self.logprobs,
            self.state_values, self.rewards, self.is_terminals
        )
