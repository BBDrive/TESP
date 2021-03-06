import gym
import torch
import numpy as np
import multiprocessing as mp

from meta_rl.envs.subproc_vec_env import SubprocVecEnv
from meta_rl.episode import BatchEpisodes


def make_env(env_name):
    def _make_env():
        return gym.make(env_name)
    return _make_env


class BatchSampler(object):
    def __init__(self, env_name, batch_size, num_workers=mp.cpu_count() - 1):
        """

        :param env_name: 某个环境的名字
        :param batch_size: total trajectories
        :param num_workers: 进程数
        """

        self.env_name = env_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.queue = mp.Queue()
        self.envs = SubprocVecEnv([make_env(env_name) for _ in range(num_workers)],
                                  queue=self.queue)
        self._env = gym.make(env_name)

    # sample一个环境中的batch_size条trajectories
    def sample(self, policy, embedding, params=None, gamma=0.95, device='cpu'):
        episodes = BatchEpisodes(batch_size=self.batch_size, gamma=gamma, device=device)
        for i in range(self.batch_size):
            self.queue.put(i)
        for _ in range(self.num_workers):
            self.queue.put(None)
        observations, batch_ids = self.envs.reset()
        dones = [False]
        while (not all(dones)) or (not self.queue.empty()):
            with torch.no_grad():
                observations_tensor = torch.from_numpy(observations).to(device=device)
                actions_tensor = policy(observations_tensor, embedding, params=params).sample()
                actions = actions_tensor.cpu().numpy()
            new_observations, rewards, dones, new_batch_ids, _ = self.envs.step(actions)
            episodes.append(observations, actions, rewards, batch_ids)
            observations, batch_ids = new_observations, new_batch_ids

        return episodes

    def reset_task(self, task):
        tasks = [task for _ in range(self.num_workers)]  # 对同一任务创建多个进程收集trajectories
        reset = self.envs.reset_task(tasks)
        return all(reset)

    def sample_tasks(self, num_tasks):
        tasks = self._env.unwrapped.sample_tasks(num_tasks)
        return tasks
