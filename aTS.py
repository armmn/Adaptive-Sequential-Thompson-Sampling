import numpy as np


class ThompsonSampling:

    def __init__(self, num_arms, c, alpha_est, update_aux=True):
        """
        :param num_arms: number of arms
        :param c: tuning parameter
        :param alpha_est: estimated mapping coefficients to be used in the policy (nd.array)
        :param update_aux: whether to update counters and means based on auxiliary observations; if it is True then we
        have aTS; otherwise a standard TS
        """
        self.num_arms = num_arms
        self.c = c
        self.alpha_est = alpha_est
        self.update_aux = update_aux

        self.num_arm_pulls = np.zeros((num_arms,))
        self.emp_mean_rewards = np.zeros((num_arms,))

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
        if not self.update_aux:
            return
        num_aux_obs = [len(aux_observations[arm]) for arm in range(self.num_arms)]
        for arm in range(self.num_arms):
            for i in range(num_aux_obs[arm]):
                sigma_hat_sigma_ratio_2 = self.alpha_est[arm] ** 2
                self.emp_mean_rewards[arm] = (self.emp_mean_rewards[arm] * self.num_arm_pulls[arm] +
                                              self.alpha_est[arm] * aux_observations[arm][i] / sigma_hat_sigma_ratio_2) /\
                                             (self.num_arm_pulls[arm] + 1 / sigma_hat_sigma_ratio_2)
                self.num_arm_pulls[arm] = self.num_arm_pulls[arm] + 1 / sigma_hat_sigma_ratio_2

    def get_action(self):
        theta = np.sqrt(self.c / np.maximum(1, self.num_arm_pulls)) * np.random.randn(self.num_arms)\
                + self.emp_mean_rewards
        return np.argmax(theta)
