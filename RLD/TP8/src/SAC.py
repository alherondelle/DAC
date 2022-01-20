import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    def __init__(self, hidden_size, input_size, gamma, alpha, lr, lr_q, batch_size, buffer_length):
        nn.Module.__init__(self)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, 1)
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.lr_q = lr_q
        self.batch_size = batch_size
        self.buffer_length = buffer_length

    def forward(self, input):
        input = F.batch_norm(input)
        x1 = F.leaky_relu(self.linear1(input))
        x1 = F.batch_norm(x1)
        x2 = F.leaky_relu(self.linear2(x1))
        x2 = F.batch_norm(x2)
        x3 = self.linear(x2)
        return x3

class PolicyNetwork(nn.Module):
    def __init__(self, hidden_size, input_size, gamma, alpha, lr, lr_q, batch_size, buffer_length):
        nn.Module.__init__(self)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, hidden_size)
        self.linear3 = torch.nn.Linear(hidden_size, hidden_size//2)
        self.linear4 = torch.nn.Linear(hidden_size//2, 1)
        self.gamma = gamma
        self.alpha = alpha
        self.lr = lr
        self.lr_q = lr_q
        self.batch_size = batch_size
        self.buffer_length = buffer_length

    def forward(self, input):
        x1 = F.relu(self.linear1(input))
        x2 = F.relu(self.linear2(x1))
        x3 = self.linear(x2)
        return x3

class SAC:
    def __init__(self, hidden_size, env):
        self.Q_network = ValueNetwork()
        self.policy_network = ValueNetwork()
        self.replay_buffer = []
        self.done = False
        self.env = env

    def learn(self):
        pass

    def act(self):

        pass

    def update_policy(self):
        pass

    def update_qvalue(self):
        pass

    def getAction(self, state):
        """
        Fonction qui retourne une action et sa probabilité en fonction de l'état précedent.
        Elle projete ensuite cette action dans un espace réduit.
        :param state:
        :return:
        """
        params = self.policy_network(state)
        m = torch.distributions.Normal(*params)
        return m.rsample()

    def store(self, s, a, r, next_s, d):
        self.replay_buffer.append((s, a, next_s, r, d))
        pass