import torch
import torch.nn as nn

from collections import OrderedDict


def weight_init(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.zero_()


class Policy(nn.Module):
    def __init__(self, input_size, output_size):
        super(Policy, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

    def _define_meta_sgd_lr(self, meta_lr):
        self.meta_sgd_lr = OrderedDict()
        for name, param in self.named_parameters():
            # self.meta_sgd_lr[name] = nn.Parameter(1e-3 * torch.rand(param.size()) *
            #                                       torch.ones_like(param))
            self.meta_sgd_lr[name] = nn.Parameter(meta_lr *
                                                  torch.ones_like(param))

    def update_params(self, loss, device='cpu'):
        """
        update: True 返回更新后的参数；False 返回当前参数
        """
        grads = torch.autograd.grad(loss, self.parameters())
        updated_params = OrderedDict()
        for (name, param), grad in zip(self.named_parameters(), grads):
            lr = self.meta_sgd_lr[name].to(device)
            updated_params[name] = param - lr * grad

        return updated_params
