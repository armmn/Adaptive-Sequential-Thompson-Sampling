import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import copy as copy
from tqdm import tqdm

num_cores = multiprocessing.cpu_count()


class Simulator:
    def __init__(self, num_arms, T, mean_rewards, num_iters, algs, aux_obs_trajectory, parallel=False):
        """
        :param num_arms: number of arms
        :param T: horizon length
        :param mean_rewards: vector of mean rewards
        :param num_iters: number of iterations
        :param algs: algorithms to simulate
        :param aux_obs_trajectory: trajectory of auxiliary observations (list of lists with length T)
        :param parallel: whether to perform multi-processing or not
        """
        self.num_arms = num_arms
        self.T = T
        self.mean_rewards = mean_rewards
        self.num_iters = num_iters
        self.algs = algs
        self.aux_obs_trajectory = aux_obs_trajectory
        self.parallel = parallel

        self.reg_trajectory = None

    def run(self):
        """
        This method runs the simulation for all the iterations
        """
        # generate observations for all iterations
        y = (np.random.rand(self.num_iters, self.T, self.num_arms) <= self.mean_rewards).astype(int)

        if self.parallel:
            reg_trajectories = Parallel(n_jobs=num_cores)(delayed(self.run_one_iter)(iter_, y[iter_, :, :])
                                                          for iter_ in tqdm(range(self.num_iters)))
            self.reg_trajectory = np.concatenate(reg_trajectories, axis=0)
        else:
            self.reg_trajectory = np.zeros((self.num_iters, len(self.algs), self.T))
            for iter_ in tqdm(range(self.num_iters)):
                self.reg_trajectory[iter_, :, :] = np.squeeze(self.run_one_iter(iter_, y[iter_, :, :]), axis=0)

    def run_one_iter(self, iter_, y):
        """
        This method runs the simulation for one single iteration
        :param iter_: iteration number
        :param y: generated observations for this iteration
        """
        print('iteration: {}'.format(iter_), end='\r', flush=True)
        max_mean_reward = np.max(self.mean_rewards)
        # initiate variables for one iteration
        actions = [None for _ in self.algs]
        reg_trajectory_one_iter = np.zeros((1, len(self.algs), self.T))
        algs_one_iter = copy.deepcopy(self.algs)
        for t in range(self.T):
            for i in range(len(self.algs)):
                # get the actions
                actions[i] = algs_one_iter[i].get_action()
                # compute the regrets
                reg_trajectory_one_iter[0, i, t] = max_mean_reward - self.mean_rewards[actions[i]]
                # return feedback to algorithms
                algs_one_iter[i].collect_observation(actions[i], y[t, actions[i]], self.aux_obs_trajectory[t])
        return reg_trajectory_one_iter

    def get_reg_trajectory(self):
        return self.reg_trajectory
