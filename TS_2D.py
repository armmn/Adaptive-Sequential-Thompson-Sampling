import numpy as np


class ThompsonSampling2Dimensional:

    def __init__(self, num_arms, c, alpha_bar_est):
        """
        :param num_arms: number of arms
        :param c: tuning parameter
        :param alpha_bar_est: estimated upper bound on mapping coefficients to be used in the policy
        """
        self.num_arms = num_arms
        self.c = c
        self.alpha_bar_est = alpha_bar_est

        self.num_arm_pulls = np.zeros((num_arms,))
        self.emp_mean_rewards = np.zeros((num_arms,))

        self.num_cum_aux_obs = np.zeros((num_arms,))  # vector of cumulative number of auxiliary observations
        self.emp_mean_aux_obs = np.zeros((num_arms,))  # vector of empirical mean of auxiliary observations

    def collect_observation(self, arm, y, aux_observations):
        """
        :param arm: the arm on which observation is collected by the policy
        :param y: the observation collected by the policy
        :param aux_observations: vector of auxiliary observations
        """
        # incorporate the observation collected by the policy
        self.emp_mean_rewards[arm] = (self.emp_mean_rewards[arm] * self.num_arm_pulls[arm] + y)\
                                     / (self.num_arm_pulls[arm] + 1)
        self.num_arm_pulls[arm] = self.num_arm_pulls[arm] + 1

        # incorporate the auxiliary observations
        num_aux_obs = [len(aux_observations[arm]) for arm in range(self.num_arms)]
        for arm in range(self.num_arms):
            for i in range(num_aux_obs[arm]):
                self.emp_mean_aux_obs[arm] = (self.emp_mean_rewards[arm] * self.num_cum_aux_obs[arm] +
                                              aux_observations[arm][i]) / (self.num_cum_aux_obs[arm] + 1)
                self.num_cum_aux_obs[arm] = self.num_cum_aux_obs[arm] + 1

    def get_action(self):

        # drawing samples from the truncated two-dimensional posteriors through
        # sample discarding from the un-truncated distributions
        # (we might be able to come up with faster sampling using Markov Chain Monte Carlo (MCMC) too!)
        reward_theta = np.zeros((self.num_arms,))
        aux_theta = np.zeros((self.num_arms,))
        for arm in range(self.num_arms):
            flag = True
            while flag or aux_theta[arm] < 0 or self.alpha_bar_est[arm] * aux_theta[arm] < reward_theta[arm]:
                flag = False
                reward_theta[arm] = np.sqrt(self.c / np.maximum(1, self.num_arm_pulls[arm])) * np.random.randn()\
                                    + self.emp_mean_rewards[arm]
                aux_theta[arm] = np.sqrt(self.c / np.maximum(1, self.num_cum_aux_obs[arm])) * np.random.randn()\
                                 + self.emp_mean_aux_obs[arm]

        return np.argmax(reward_theta)
