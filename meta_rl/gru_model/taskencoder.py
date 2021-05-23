import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from meta_rl.policies.policy import Policy
from meta_rl.gru_model.gru import GRU


class TaskEncoder(Policy):
    def __init__(self, input_size, hidden_size, output_size, meta_lr):
        super(TaskEncoder, self).__init__(input_size, output_size)
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.gru = GRU(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=False,  # (time_step, batch, input)
        )
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self._define_meta_sgd_lr(meta_lr=meta_lr)

    def forward(self, input, params=None):
        if params is None:
            output, _ = self.gru(input)
            output = self.out(output[-1, :, :])
        else:
            gru_params = []
            for i in range(self.num_layers):
                gru_params.append(params['gru.weight_ih_l{}'.format(i)])
                gru_params.append(params['gru.weight_hh_l{}'.format(i)])
                gru_params.append(params['gru.bias_ih_l{}'.format(i)])
                gru_params.append(params['gru.bias_hh_l{}'.format(i)])

            output, _ = self.gru(input, params=gru_params)
            output = F.linear(output[-1, :, :],
                              weight=params['out.weight'],
                              bias=params['out.bias'])

        return output

    def access_params(self):
        params = OrderedDict()
        for name, param in self.named_parameters():
            params[name] = param
        return params

    def store_params(self, params):
        for name, param in self.named_parameters():
            param = params[name]


def task_embedding(episodes, task_encoder, params=None):
    rewards = torch.unsqueeze(episodes.rewards, dim=2)
    gru_input = torch.cat((episodes.observations, episodes.actions, rewards), dim=2)
    embedding = torch.mean(task_encoder(gru_input, params=params), dim=0)  # shape:(te_ouput_size)
    return embedding
