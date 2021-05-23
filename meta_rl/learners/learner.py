import torch
from meta_rl.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)


class Learner(object):
    def __init__(self, sampler, policy, baseline, task_encoder, tau=1.0, device='cpu'):
        self.sampler = sampler
        self.policy = policy
        self.baseline = baseline
        self.task_encoder = task_encoder
        self.tau = tau
        self.device = device

    def inner_loss(self, episodes, embedding, params=None):  # return shape (1, batch_size)
        """
        episedes： 单个任务的所有trajectories
        """
        values = self.baseline(episodes)
        advantages = episodes.gae(values, tau=self.tau)
        advantages = weighted_normalize(advantages, weights=episodes.mask)

        pi = self.policy(episodes.observations, embedding, params=params)
        log_probs = pi.log_prob(episodes.actions)
        if log_probs.dim() > 2:
            log_probs = torch.sum(log_probs, dim=2)
        loss = -weighted_mean(log_probs * advantages, dim=0,
                              weights=episodes.mask)

        return loss

    def adapt(self, episodes, embedding):

        # Fit the baseline to the training episodes
        self.baseline.fit(episodes)
        # Get the loss on the training episodes
        loss = self.inner_loss(episodes, embedding)

        params = self.task_encoder.update_params(loss, self.device)
        return params