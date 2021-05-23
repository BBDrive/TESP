import torch

from meta_rl.episode import BatchEpisodes
from meta_rl.gru_model.taskencoder import task_embedding
from meta_rl.learners.learner import Learner


class FastLearner(Learner):
    def __init__(self, sampler, policy, baseline, task_encoder, fast_update_time=20, te_output_size=16, buffer_size=16,
                 gamma=0.95, tau=1.0, device='cpu'):
        super(FastLearner, self).__init__(sampler, policy, baseline, task_encoder, tau, device)
        self.fast_update_time = fast_update_time
        self.te_output_size = te_output_size
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.episode_buffer = BatchEpisodes(self.buffer_size, self.gamma, self.device)

    def update_buffer(self, episodes):
        values, indices = torch.topk(torch.sum(episodes.rewards, dim=0),
                                     self.buffer_size, dim=0, largest=True, sorted=True)  # 当前episodes 升序
        if len(self.episode_buffer.observations_list[self.buffer_size-1]):  # episode buffer 不为空
            b_values, b_indices = torch.sort(torch.sum(self.episode_buffer.rewards, dim=0),
                                             dim=0, descending=False)  # episode buffer 降序
            for i in range(self.buffer_size):
                if values[i] > b_values[i]:
                    episodes.assign(indices[i], self.episode_buffer, b_indices[i])
                else:
                    break
        else:  # episode buffer 为空
            for i, j in zip(indices, range(self.buffer_size)):
                episodes.assign(i, self.episode_buffer, j)

    def step(self):
        params = None
        train_episodes = []
        train_buffers = []
        for i in range(self.fast_update_time+1):
            embedding = task_embedding(self.episode_buffer, self.task_encoder, params=params) if i != 0 \
                else torch.zeros(self.te_output_size).to(self.device)
            episode = self.sampler.sample(self.policy, embedding, gamma=self.gamma, device=self.device)
            if i != 0:
                params = self.adapt(episode, embedding)

                train_buffers.append(self.episode_buffer)
                train_episodes.append(episode)

            self.update_buffer(episode)

        embedding = task_embedding(self.episode_buffer, self.task_encoder, params=params)
        valid_episodes = self.sampler.sample(self.policy, embedding, gamma=self.gamma, device=self.device)
        return train_buffers, self.episode_buffer, train_episodes, valid_episodes
