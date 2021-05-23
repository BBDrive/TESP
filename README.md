# Meta Reinforcement Learning with Task Embedding and Shared Policy

Implementation of TESP in Pytorch.

## Usage
You can use the [`main.py`](main.py) script in order to run meta reinforcement learning experiments with TESP. This script was tested with Python 3.7.
```
python main.py --env-name HalfCheetahDir-v1 --output-folder TESP-halfcheetah-dir --device cuda
```

## References
This repository are based on the paper
> Lan L, Li Z, Guan X, et al. Meta Reinforcement Learning with Task Embedding and Shared Policy[J]. arXiv preprint arXiv:1905.06527, 2019. [[ArXiv](https://arxiv.org/abs/1905.06527)]

If you want to cite this paper
```
@article{lan2019meta,
  title={Meta Reinforcement Learning with Task Embedding and Shared Policy},
  author={Lan, Lin and Li, Zhenguo and Guan, Xiaohong and Wang, Pinghui},
  journal={arXiv preprint arXiv:1905.06527},
  year={2019}
}
```
