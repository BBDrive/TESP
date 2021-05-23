import numpy as np
import torch
import json
import time

from meta_rl.learners.metalearner import MetaLearner
from meta_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy
from meta_rl.baseline import LinearFeatureBaseline
from meta_rl.sampler import BatchSampler
from meta_rl.gru_model.taskencoder import TaskEncoder

from tensorboardX import SummaryWriter


def total_rewards(episodes_rewards, aggregation=torch.mean):
    rewards = torch.mean(torch.stack([aggregation(torch.sum(rewards, dim=0))
                                      for rewards in episodes_rewards], dim=0))
    return rewards.item()


def main(args):
    continuous_actions = (args.env_name in ['AntVel-v1', 'AntDir-v1',
                                            'AntPos-v0', 'HalfCheetahVel-v1',
                                            'HalfCheetahDir-v1', '2DNavigation-v0'])

    writer = SummaryWriter('./logs/{0}'.format(args.output_folder))
    save_folder = './saves/{0}'.format(args.output_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    with open(os.path.join(save_folder, 'config.json'), 'w') as f:
        config = {k: v for (k, v) in vars(args).items() if k != 'device'}
        config.update(device=args.device.type)
        json.dump(config, f, indent=2)  # indent 缩进

    sampler = BatchSampler(args.env_name, batch_size=args.fast_batch_size,
                           num_workers=args.num_workers)
    if continuous_actions:
        policy = NormalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape))+args.te_output_size,
            int(np.prod(sampler.envs.action_space.shape)),
            hidden_sizes=(args.policy_hidden_size,) * args.policy_num_layers)

        task_encoder = TaskEncoder(input_size=int(np.prod(sampler.envs.observation_space.shape)) +
                                              int(np.prod(sampler.envs.action_space.shape)) + 1,
                                   hidden_size=args.te_hidden_size,
                                   output_size=args.te_output_size, meta_lr=args.meta_lr)
    else:
        policy = CategoricalMLPPolicy(
            int(np.prod(sampler.envs.observation_space.shape))+args.te_output_size,
            sampler.envs.action_space.n,
            hidden_sizes=(args.policy_hidden_size,) * args.policy_num_layers)

        task_encoder = TaskEncoder(input_size=int(np.prod(sampler.envs.observation_space.shape)) +
                                              sampler.envs.action_space.n + 1,
                                   hidden_size=args.te_hidden_size,
                                   output_size=args.te_output_size, meta_lr=args.meta_lr)

    baseline = LinearFeatureBaseline(int(np.prod(sampler.envs.observation_space.shape)))

    meta_learner = MetaLearner(sampler, policy, baseline, task_encoder,
                               eta=args.eta, tau=args.tau, clip_param=args.clip_param,
                               lr_ppo=args.lr_ppo, device=args.device)

    for batch in range(args.meta_update_time):  # meta_update_time: 大循环
        time_start = time.time()

        # te_params = task_encoder.access_params()  # 取出现在的task encoder的参数
        tasks = sampler.sample_tasks(num_tasks=args.meta_batch_size)  # meta_batch_size: 共有多少任务
        buffers, episodes = meta_learner.sample(tasks, fast_update_time=args.fast_update_time,
                                                te_output_size=args.te_output_size, buffer_size=args.buffer_size,
                                                gamma=args.gamma)

        meta_learner.step(buffers, episodes, ppo_update_time=args.ppo_update_time)

        # Tensorboard
        writer.add_scalar('total_rewards', total_rewards([ep.rewards for _, ep in episodes]), batch)

        # Save policy network
        with open(os.path.join(save_folder,
                'policy-{0}.pt'.format(batch)), 'wb') as f:
            torch.save(policy.state_dict(), f)

        print("Training: Batch {} of {} total batch;"
              "Time: {:.2f}s; reward: {:.2f}".format(batch, args.meta_update_time,
                                                     time.time()-time_start,
                                                     total_rewards([ep.rewards for _, ep in episodes])))


if __name__ == '__main__':
    import argparse
    import os
    import multiprocessing as mp

    parser = argparse.ArgumentParser(description='Meta Reinforcement Learning '
                                                 'with Task Embedding and Shared Policy')

    # General
    parser.add_argument('--env-name', type=str,
                        help='name of the environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='value of the discount factor gamma')
    parser.add_argument('--eta', type=float, default=0.01,
                        help='value of the discount factor for task embedding')
    parser.add_argument('--tau', type=float, default=0.97,
                        help='value of the discount factor for GAE')

    # Policy network (relu activation function)
    parser.add_argument('--policy-hidden-size', type=int, default=512,
                        help='number of policy hidden units per layer')
    parser.add_argument('--policy-num-layers', type=int, default=2,
                        help='number of policy hidden layers')

    # Task-specific
    parser.add_argument('--fast-batch-size', type=int, default=25,
                        help='batch size for each individual task')  # batch_size 即为一个任务有多少trajectories
    parser.add_argument('--meta-lr', type=float, default=1e-3,
                        help='Learning rate of Meta SGD ')

    # Task Encoder
    parser.add_argument('--te-hidden-size', type=int, default=256,
                        help='number of TaskEncoder hidden units')
    parser.add_argument('--te-output-size', type=int, default=16,
                        help='dimension of task embeddings')

    # Fast Learner
    parser.add_argument('--buffer-size', type=int, default=16,
                        help='Episode buffer size')

    # Optimization
    parser.add_argument('--meta-update-time', type=int, default=1000,
                        help='number of batches')  # meta_update_time: 大循环
    parser.add_argument('--fast-update-time', type=int, default=3,
                        help='number of fast updates')  # fast_update_time: 小循环
    parser.add_argument('--meta-batch-size', type=int, default=20,
                        help='number of tasks per batch')  # meta_batch_size: 共有多少任务
    parser.add_argument('--ppo-update-time', type=int, default=20,
                        help='maximum number of iterations for line search')
    parser.add_argument('--clip-param', type=float, default=0.15,
                        help='scope of ppo loss ')
    parser.add_argument('--lr-ppo', type=float, default=3e-4,
                        help='the learning rate of ppo')

    # Miscellaneous
    parser.add_argument('--output-folder', type=str, default='maml',
                        help='name of the output folder')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count() - 1,
                        help='number of workers for trajectories sampling')  # 几个进程收集trajectories
    parser.add_argument('--device', type=str, default='cpu',
                        help='set the device (cpu or cuda)')

    args = parser.parse_args()

    # Create logs and saves folder if they don't exist
    if not os.path.exists('./logs'):
        os.makedirs('./logs')
    if not os.path.exists('./saves'):
        os.makedirs('./saves')
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])

    main(args)
