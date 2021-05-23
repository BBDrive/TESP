import numpy as np
import torch
import torch.nn.functional as F


class BatchEpisodes(object):
    def __init__(self, batch_size, gamma=0.95, device='cpu'):
        """

        :param batch_size: total trajectories
        :param gamma:
        :param device:
        """
        self.batch_size = batch_size
        self.gamma = gamma
        self.device = device

        self.observations_list = [[] for _ in range(batch_size)]
        self.actions_list = [[] for _ in range(batch_size)]
        self.rewards_list = [[] for _ in range(batch_size)]
        # self.observation_embeddings_list = [[] for _ in range(batch_size)]
        self.mask_list = []

        self._observations = None
        self._actions = None
        self._rewards = None
        # self._observation_embeddings = None
        self._returns = None
        self._mask = None


    @property
    def observations(self):  # return shape (episodes, batch_size, observation_shape)
        if self._observations is None:
            observation_shape = self.observations_list[0][0].shape
            observations = np.zeros((len(self), self.batch_size)
                + observation_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self.observations_list[i])
                observations[:length, i] = np.stack(self.observations_list[i], axis=0)
            self._observations = torch.from_numpy(observations).to(self.device)
        return self._observations

    @property
    def actions(self):  # # return shape (episodes, batch_size, action_shape)
        if self._actions is None:
            action_shape = self.actions_list[0][0].shape
            actions = np.zeros((len(self), self.batch_size)
                + action_shape, dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self.actions_list[i])
                actions[:length, i] = np.stack(self.actions_list[i], axis=0)
            self._actions = torch.from_numpy(actions).to(self.device)
        return self._actions

    @property
    def rewards(self):  # return shape (episodes, batch_size)
        if self._rewards is None:
            rewards = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self.rewards_list[i])
                rewards[:length, i] = np.stack(self.rewards_list[i], axis=0)
            self._rewards = torch.from_numpy(rewards).to(self.device)
        return self._rewards

    # @property
    # def observation_embeddings(self):  # return shape (episodes, batch_size, observation_shape+embedding_shape)
    #     if self._observation_embeddings is None:
    #         observation_embedding_shape = self.observation_embeddings_list[0][0].shape
    #         observation_embeddings = np.zeros((len(self), self.batch_size)
    #                                 + observation_embedding_shape, dtype=np.float32)
    #         for i in range(self.batch_size):
    #             length = len(self.observation_embeddings_list[i])
    #             observation_embeddings[:length, i] = np.stack(self.observation_embeddings_list[i], axis=0)
    #         self._observation_embeddings = torch.from_numpy(observation_embeddings).to(self.device)
    #     return self._observation_embeddings

    @property
    def returns(self):  # return shape (episodes, batch_size) 计算r + gamma * v
        if self._returns is None:
            return_ = np.zeros(self.batch_size, dtype=np.float32)
            returns = np.zeros((len(self), self.batch_size), dtype=np.float32)
            rewards = self.rewards.cpu().numpy()
            mask = self.mask.cpu().numpy()
            for i in range(len(self) - 1, -1, -1):
                return_ = self.gamma * return_ + rewards[i] * mask[i]
                returns[i] = return_
            self._returns = torch.from_numpy(returns).to(self.device)
        return self._returns

    @property
    def mask(self):  # return shape (episodes, batch_size)
        if self._mask is None:
            mask = np.zeros((len(self), self.batch_size), dtype=np.float32)
            for i in range(self.batch_size):
                length = len(self.actions_list[i])
                mask[:length, i] = 1.0
            self._mask = torch.from_numpy(mask).to(self.device)
        return self._mask

    def gae(self, values, tau=1.0):  # return shape (episodes, batch_size)
        # Add an additional 0 at the end of values for
        # the estimation at the end of the episode
        values = values.squeeze(2).detach()
        values = F.pad(values * self.mask, (0, 0, 0, 1))

        deltas = self.rewards + self.gamma * values[1:] - values[:-1]
        advantages = torch.zeros_like(deltas).float()
        gae = torch.zeros_like(deltas[0]).float()
        for i in range(len(self) - 1, -1, -1):
            gae = gae * self.gamma * tau + deltas[i]
            advantages[i] = gae

        return advantages

    def append(self, observations, actions, rewards, batch_ids):
        for observation, action, reward, batch_id in zip(
                observations, actions, rewards, batch_ids):
            if batch_id is None:
                continue
            self.observations_list[batch_id].append(observation.astype(np.float32))
            self.actions_list[batch_id].append(action.astype(np.float32))
            self.rewards_list[batch_id].append(reward.astype(np.float32))
            # self.observation_embeddings_list[batch_id].append(observation_embedding.astype(np.float32))

    def __len__(self):
        return max(map(len, self.rewards_list))

    def assign(self, i, episodes, j):
        '''
        episodes[j] = self.episodes[i]
        '''
        episodes.observations_list[j] = self.observations_list[i]
        episodes.actions_list[j] = self.actions_list[i]
        episodes.rewards_list[j] = self.rewards_list[i]
        # episodes.observation_embeddings_list[j] = self.observation_embeddings_list[i]

        episodes._observations = None
        episodes._actions = None
        episodes._rewards = None
        # episodes._observation_embeddings = None
        episodes._returns = None
        episodes._mask = None
