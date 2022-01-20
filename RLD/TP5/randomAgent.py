import argparse
import sys
import matplotlib
import random
#matplotlib.use("Qt5agg")
matplotlib.use("TkAgg")
import gym
import gridworld
import torch
import torch.nn as nn
from utils import *
from core import *
from torch.utils.tensorboard import SummaryWriter
#import highway_env
from matplotlib import pyplot as plt
import yaml
from datetime import datetime

### POUR STORE, DEFINIR UNE DATACLASS QUI SERA APPELEE A CHAQUE FOIS

# 1. Agir avec une technique d'exploration : epsilon greedy
# 2. Learn : descente de gradient pour apprendre la question Q -> Calculer la valeur cible et mettre à jour la fonction d'apprentissage

class RandomAgent(object):
    """The world's simplest agent!"""

    def __init__(self, env, opt, epsilon, discount, lr):
        self.opt=opt
        self.gamma = discount
        self.loss = nn.SmoothL1Loss()
        self.env=env
        if opt.fromFile is not None:
            self.load(opt.fromFile)
        self.action_space = env.action_space
        self.featureExtractor = opt.featExtractor(env)
        self.test=False
        self.nbEvents=0
        self.epsilon = epsilon
        self.model = NN(inSize=opt.featExtractor(env).outSize, outSize=env.action_space.n, layers=[env.action_space.n//2])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=lr)


    def act(self, obs):
        ## Obs are the features extracted from the resulting state
        if random.random() < self.epsilon:
            a = random.randint(0, self.action_space.n-1)
        else:
            a = torch.argmax(self.model(obs))
        return a

    # sauvegarde du modèle
    def save(self,outputDir):
        pass

    # chargement du modèle.
    def load(self,inputDir):
        pass


    def learn(self) -> None:
        # Si l'agent est en mode de test, on n'entraîne pas
        if self.test:
            pass
        else:
            R = self.last_reward + self.gamma*torch.max(self.model(self.dest))
            loss = self.loss(R, self.model(self.source)[self.last_action])
            loss.backward()
            self.optim.step()
            self.optim.zero_grad()

    # enregistrement de la transition pour exploitation par learn ulterieure
    def store(self,ob, action, new_ob, reward, done, it) -> None:
        # Si l'agent est en mode de test, on n'enregistre pas la transition
        ### ATTENTION : DOUBLONS A SIMPLIFIER
        if not self.test:
            self.source = ob
            self.dest = new_ob
            self.last_action = action
            self.last_reward = reward
            self.done = done
            # si on atteint la taille max d'episode en apprentissage, alors done ne devrait pas etre a true (episode pas vraiment fini dans l'environnement)
            if it == self.opt.maxLengthTrain:
                print("undone")
                done=False
            tr = (ob, action, reward, new_ob, done)
            self.lastTransition=tr #ici on n'enregistre que la derniere transition pour traitement immédiat, mais on pourrait enregistrer dans une structure de buffer (c'est l'interet de memory.py)

    # retoune vrai si c'est le moment d'entraîner l'agent.
    # Dans cette version retourne vrai tous les freqoptim evenements
    # Mais on pourrait retourner vrai seulement si done pour s'entraîner seulement en fin d'episode
    def timeToLearn(self,done) -> bool:
        if self.test:
            return False
        self.nbEvents+=1
        return self.nbEvents%self.opt.freqOptim == 0

if __name__ == '__main__':
    env, config, outdir, logger = init('./configs/config_random_cartpole.yaml', "RandomAgent")

    freqTest = config["freqTest"]
    freqSave = config["freqSave"]
    nbTest = config["nbTest"]
    env.seed(config["seed"])
    np.random.seed(config["seed"])
    episode_count = config["nbEpisodes"]

    agent = RandomAgent(env,config)


    rsum = 0
    mean = 0
    verbose = True
    itest = 0
    reward = 0
    done = False
    for i in range(episode_count):
        checkConfUpdate(outdir, config
                        )

        rsum = 0
        agent.nbEvents = 0
        ob = env.reset()

        # On souhaite afficher l'environnement (attention à ne pas trop afficher car çà ralentit beaucoup)
        if i % int(config["freqVerbose"]) == 0:
            verbose = True
        else:
            verbose = False

        # C'est le moment de tester l'agent
        if i % freqTest == 0 and i >= freqTest:  ##### Same as train for now
            print("Test time! ")
            mean = 0
            agent.test = True

        # On a fini cette session de test
        if i % freqTest == nbTest and i > freqTest:
            print("End of test, mean reward=", mean / nbTest)
            itest += 1
            logger.direct_write("rewardTest", mean / nbTest, itest)
            agent.test = False

        # C'est le moment de sauver le modèle
        if i % freqSave == 0:
            agent.save(outdir + "/save_" + str(i))

        j = 0
        if verbose:
            env.render()

        new_ob = agent.featureExtractor.getFeatures(ob)
        while True:
            if verbose:
                env.render()

            ob = new_ob
            action= agent.act(ob)
            new_ob, reward, done, _ = env.step(action)
            new_ob = agent.featureExtractor.getFeatures(new_ob)

            j+=1

            # Si on a atteint la longueur max définie dans le fichier de config
            if ((config["maxLengthTrain"] > 0) and (not agent.test) and (j == config["maxLengthTrain"])) or ( (agent.test) and (config["maxLengthTest"] > 0) and (j == config["maxLengthTest"])):
                done = True
                print("forced done!")

            agent.store(ob, action, new_ob, reward, done,j)
            rsum += reward

            if agent.timeToLearn(done):
                agent.learn()
            if done:
                print(str(i) + " rsum=" + str(rsum) + ", " + str(j) + " actions ")
                logger.direct_write("reward", rsum, i)
                agent.nbEvents = 0
                mean += rsum
                rsum = 0

                break

    env.close()
