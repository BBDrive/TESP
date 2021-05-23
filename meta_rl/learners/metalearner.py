import torch

from meta_rl.utils.torch_utils import (weighted_mean, detach_distribution, weighted_normalize)
from meta_rl.learners.fastlearner import FastLearner
from meta_rl.gru_model.taskencoder import task_embedding
from meta_rl.learners.learner import Learner


class MetaLearner(Learner):

    def __init__(self, sampler, policy, baseline, task_encoder, eta=0.01,
                 tau=1.0, clip_param=0.2, lr_ppo=5e-5, device='cpu'):
        super(MetaLearner, self).__init__(sampler, policy, baseline, task_encoder, tau, device)
        self.eta = eta
        self.clip_param = clip_param
        self.lr_ppo = lr_ppo
        self.to(device)

        self.model_params = list(self.task_encoder.parameters()) + \
                            list(self.policy.parameters()) + \
                            list(self.task_encoder.meta_sgd_lr.values())
        self.optimizer = torch.optim.Adam(self.model_params, lr=self.lr_ppo)

    def sample(self, tasks, fast_update_time=20, te_output_size=16, buffer_size=16, gamma=0.95):
        """Sample trajectories (before and after the update of the parameters) 
        for all the tasks `tasks`.
        """
        buffers = []
        episodes = []
        for task in tasks:
            fast_learner = FastLearner(self.sampler, self.policy, self.baseline, self.task_encoder,
                                       fast_update_time, te_output_size, buffer_size, gamma, self.tau, self.device)
            self.sampler.reset_task(task)
            train_buffer, valid_buffer, train_episode, valid_episode = fast_learner.step()
            buffers.append((train_buffer, valid_buffer))
            episodes.append((train_episode, valid_episode))

        return buffers, episodes

    def surrogate_loss(self, buffers, episodes, old_pis=None):
        # te_params = self.task_encoder.access_params()
        losses, pis = [], []
        if old_pis is None:
            old_pis = [None] * len(episodes)

        for (train_buffers, valid_buffer), (train_episodes, valid_episode), old_pi in zip(buffers, episodes, old_pis):
            # self.task_encoder.store_params(te_params)
            params = None
            for train_buffer, train_episode in zip(train_buffers, train_episodes):
                embedding = task_embedding(train_buffer, self.task_encoder, params=params)
                params = self.adapt(train_episode, embedding)

            embedding = task_embedding(valid_buffer, self.task_encoder, params=params)
            pi = self.policy(valid_episode.observations, embedding)
            pis.append(detach_distribution(pi))

            if old_pi is None:
                old_pi = detach_distribution(pi)

            values = self.baseline(valid_episode)
            advantages = valid_episode.gae(values, tau=self.tau)
            advantages = weighted_normalize(advantages,
                                            weights=valid_episode.mask)

            log_ratio = (pi.log_prob(valid_episode.actions)
                         - old_pi.log_prob(valid_episode.actions))
            if log_ratio.dim() > 2:
                log_ratio = torch.sum(log_ratio, dim=2)
            ratio = torch.exp(log_ratio)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0-self.clip_param, 1.0+self.clip_param) * advantages
            loss = torch.min(surr1, surr2)

            loss = -weighted_mean(loss, dim=0, weights=valid_episode.mask) + self.eta * torch.sum(embedding * embedding)
            losses.append(loss)

        return (torch.sum(torch.stack(losses, dim=0)),
                pis)

    def step(self, buffers, episodes, ppo_update_time=5):

        loss, old_pis = self.surrogate_loss(buffers, episodes)
        for i in range(ppo_update_time):

            # grads = torch.autograd.grad(loss, self.policy.parameters())
            # grads = parameters_to_vector(grads)
            # params = parameters_to_vector(self.policy.parameters())
            # vector_to_parameters(params - self.lr_ppo * grads,
            #                      self.policy.parameters())
            print("------meta_sgd_lr------", i)
            print(self.task_encoder.meta_sgd_lr)
            print("------encoder parameter------", i)
            print(self.task_encoder.access_params())
            print("------policy parameter------", i)
            for name, param in self.policy.named_parameters():
                print("name:{}, param={}".format(name, param))
            print('*'*100)
            self.optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model_params, 0.5)
            self.optimizer.step()
            if i != (ppo_update_time - 1):
                loss, _ = self.surrogate_loss(buffers, episodes, old_pis=old_pis)

    def to(self, device, **kwargs):
        self.policy.to(device, **kwargs)
        self.baseline.to(device, **kwargs)
        self.task_encoder.to(device, **kwargs)
        self.device = device
