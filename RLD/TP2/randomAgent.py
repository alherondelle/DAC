import matplotlib
import pprint 
matplotlib.use("TkAgg")
import gym
import gridworld
from gym import wrappers, logger
import numpy as np
from torch.utils.tensorboard import SummaryWriter


class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, observation):
        return self.action_space.sample()

class PolicyIterationAgent(object):
    def __init__(self, states, actions, mdp, precision_v, discount, action_space):
        self.states = {states[i]: i for i in range(len(states))}
        self.actions = actions
        self.epsilon = precision_v
        self.gamma = discount
        self.mdp = mdp
        self.policy = self.get_policy(mdp, action_space)


    def act(self, obs):
        st = self.states[str(obs.tolist())]
        return self.policy[st]

    def obs_to_state(self, obs):
        """
        Fonction à revoir
        :param obs: Grille de jeu à l'instant t
        :return: position du joueur dans la grille de jeu qui ne contient pas les murs délimiteurs
        """
        count=0
        for i in obs:
            for j in i:
                if j in list(self.plan_obj.keys()):
                    count+=1
                if j == 2:
                    return count
                else: pass
        return count

    def get_policy(self, P, action_space):
        pi = np.array([action_space.sample() for s in self.states.keys()])
        next_pi = np.array([action_space.sample() for s in self.states.keys()])
        v_value = np.random.rand(len(self.states))
        next_v_value = np.random.rand(len(self.states))
        while not (pi == next_pi).all():
            pi = next_pi
            while np.mean(next_v_value - v_value) > self.epsilon:
                v_value = next_v_value
                next_v_value = np.zeros((len(self.states)))
                for s in P.keys():
                    for l in P[s][pi[s]]:
                        p, s_prime, r, _ = l
                        next_v_value[s] += p * (r + self.gamma*v_value[s_prime])
            reward_estimates = np.zeros((len(self.states), action_space.n))
            for s in P.keys():
                for a in P[s].keys():
                    reward_estimates[s][a] = np.sum([l[0]*(l[2] + self.gamma*next_v_value[l[1]]) for l in P[s][a]])
                    next_pi = np.argmax(reward_estimates, axis=1)
        return next_pi

class ValueIterationAgent(object):
    def __init__(self, states, actions, mdp, precision_v, discount, action_space):
        self.states = {s: i for s, i in zip(states, mdp.keys())}
        self.actions = actions
        self.mdp = mdp
        self.epsilon = precision_v
        self.gamma = discount
        self.action_space = action_space
        self.policy = self.get_policy()

    def get_policy(self):
        V = np.random.rand(len(self.states))
        pi = np.array([self.action_space.sample() for s in self.states.keys()])
        delta = np.mean(V-np.random.rand(len(self.states)))
        i = 0
        while np.abs(delta) > self.epsilon:
            i += 1
            print('Policy iteration : ', i)
            V_new = self.compute_v_value(V)
            delta = np.mean(V_new - V)
            print(f'Delta value: {delta}')
            V = V_new
        for i, s in enumerate(self.states.values()):
            pi[i] = np.argmax(np.sum([[l[0]*(l[2]+self.gamma*V[i]) for l in self.mdp[s][a]] for a in self.mdp[s]], axis=1))
        return pi


    def compute_v_value(self, V):
        V_new = np.zeros(len(self.states))
        for l, i_s in enumerate(self.states.values()):
            values = np.zeros(len(self.actions))
            for a in self.actions.keys():
                for transition in self.mdp[i_s][a]:
                    p, i_s_prime, r, done = transition
                    if not done:
                        l_s_prime = list(self.states.values()).index(i_s_prime)
                        values[a] += p * (r + self.gamma * V[l_s_prime])
                    else:
                        values[a] += p * r
            V_new[l] = np.max(values)
        return V_new

    def act(self, obs):
        st = self.states[str(obs.tolist())]
        idx_s = list(self.mdp.keys()).index(st)
        return self.policy[idx_s]


if __name__ == '__main__':
    env = gym.make("gridworld-v0")
    plan_obj = {0: -0.001, 3: 1, 4: 1, 5: -1, 6: -1}
    discount = 0.99
    precision_v = [i for i in range(10)]
    for p in precision_v:
        env.setPlan("gridworldPlans/plan3.txt", plan_obj)
        env.seed(0)  # Initialise le seed du pseudo-random
        env.render(mode="rgb_array")  # permet de visualiser la grille du jeu
        states, mdp = env.getMDP()  # recupere le mdp et la liste d'etats
        actions = env.actions
        # Execution avec un Agent
        writer = SummaryWriter(f'policy_map_{p}')
        agent = PolicyIterationAgent(states=states, actions=actions, mdp=mdp, precision_v=0.01, discount=0.99,  action_space=env.action_space)
        episode_count = 100
        reward = 0
        done = False
        rsum = 0
        for i in range(episode_count):
            obs = env.reset()
            env.verbose = (i % 100 == 0 and i > 0)  # afficher 1 episode sur 100
            if env.verbose:
                env.render()
            j = 0
            rsum = 0
            while not done:
                action = agent.act(obs)
                obs, reward, done, _ = env.step(action)
                rsum += reward
                j += 1
                if env.verbose:
                    env.render()
            writer.add_scalar(f'reward', rsum, i)
            writer.add_scalar(f'num_actions', j, i)
            print("Episode : " + str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions")
            done = False

        print("done")
        env.close()