from gridworld.gridworld_env import GridworldEnv
import torch
import numpy as np

class Policy:
    def __init__(self, states, mdp, actions):
        self.states = states
        self.mdp = mdp
        self.actions = actions
        self.type_algo = "policy iteration"
        self.v_value = np.zeros((1, len(states)))
        self.policy = np.random.dirichlet(np.ones(len(actions)), size=len(states))

    def compute_v_value(self, gamma):
        v_value = np.random.rand(len(self.states))
        for s in self.mdp.keys():
            v_value[s] = np.sum(self.policy*[np.sum([l[0]*(l[2]+gamma*self.v_value[-1][l[1]]) for l in self.mdp[s][k]]) for k in self.mdp[s]])
        self.v_value = np.vstack((self.v_value, v_value))

    def policy_iteration(self, epsilon, gamma):
        delta = epsilon + 1
        i = 0
        while np.abs(delta) > epsilon:
            i += 1
            print('Policy iteration : ', i)
            self.compute_v_value(gamma)
            delta = np.mean(self.v_value[-1] - self.v_value[-2])
            print(f'Delta value: {delta}')
