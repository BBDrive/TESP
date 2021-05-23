import numpy as np
import multiprocessing as mp
import gym
import sys
is_py2 = (sys.version[0] == '2')  # 判断python版本
if is_py2:
    import Queue as queue
else:
    import queue as queue


class EnvWorker(mp.Process):
    def __init__(self, remote, env_fn, queue, lock):
        """
        针对单个环境的worker
        :param remote: 管道conn2
        :param env_fn: 建立好的一个环境
        :param queue: 队列
        :param lock: 进程锁
        """
        super(EnvWorker, self).__init__()
        self.remote = remote
        self.env = env_fn()
        self.queue = queue
        self.lock = lock
        self.task_id = None  # 放入batch_size的哪个位置
        self.done = False  # 是否batch_size已经不能再放下一个trajectory

    def empty_step(self):
        observation = np.zeros(self.env.observation_space.shape,
                               dtype=np.float32)
        reward, done = 0.0, True
        return observation, reward, done, {}

    def try_reset(self):
        with self.lock:
            try:
                self.task_id = self.queue.get(True)  # True代表当队列空了之后，get 就会阻塞，一直等待队列中有数据后再获取数据
                self.done = (self.task_id is None)
            except queue.Empty:
                self.done = True
        observation = (np.zeros(self.env.observation_space.shape, dtype=np.float32)
                       if self.done else self.env.reset())
        return observation

    def run(self):
        while True:
            command, data = self.remote.recv()
            if command == 'step':
                observation, reward, done, info = (self.empty_step()
                                                   if self.done else self.env.step(data))
                if done and (not self.done):
                    observation = self.try_reset()
                self.remote.send((observation, reward, done, self.task_id, info))
            elif command == 'reset':
                observation = self.try_reset()
                self.remote.send((observation, self.task_id))
            elif command == 'reset_task':
                self.env.unwrapped.reset_task(data)
                self.remote.send(True)
            elif command == 'close':
                self.remote.close()
                break
            elif command == 'get_spaces':
                self.remote.send((self.env.observation_space,
                                 self.env.action_space))
            else:
                raise NotImplementedError()


class SubprocVecEnv(gym.Env):
    def __init__(self, env_factory, queue):
        """
        针对多个同一环境同时操作
        :param env_factory: 建立好的多个相同环境
        :param queue: 队列
        """
        self.lock = mp.Lock()  # 进程锁

        # Pipe方法返回(conn1, conn2)代表一个管道的两个端，
        # Pipe方法有duplex参数，如果duplex参数为True(默认值)，那么这个管道是全双工模式，也就是说conn1和conn2均可收发。
        # duplex为False，conn1只负责接受消息，conn2只负责发送消息。
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_factory])

        self.workers = [EnvWorker(remote, env_fn, queue, self.lock)
                        for (remote, env_fn) in zip(self.work_remotes, env_factory)]
        for worker in self.workers:

            # 如果某个子线程的daemon属性为False，主线程结束时会检测该子线程是否结束，如果该子线程还在运行，则主线程会等待它完成后再退出
            # 如果某个子线程的daemon属性为True，主线程运行结束时不对这个子线程进行检查而直接退出，同时所有daemon值为True的子线程将随主线程一起结束，而不论是否运行完成。
            worker.daemon = True
            worker.start()
        for remote in self.work_remotes:
            remote.close()
        self.waiting = False
        self.closed = False

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        self.observation_space = observation_space
        self.action_space = action_space

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        observations, rewards, dones, task_ids, infos = zip(*results)
        return np.stack(observations), np.stack(rewards), np.stack(dones), task_ids, infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        observations, task_ids = zip(*results)
        return np.stack(observations), task_ids

    def reset_task(self, tasks):
        for remote, task in zip(self.remotes, tasks):
            remote.send(('reset_task', task))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True
