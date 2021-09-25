import numpy as np
import torch
from typing import Optional


class Env_Params_Buffer:
    def __init__(self, buffer_size, params_shape, device, n_envs:int = 1):
        self._buffer_size = buffer_size
        self._device = device
        self.pos = 0
        self.n_envs = n_envs
        self._params_size = params_shape

        self.reset()

    def reset(self):
        self.full = False
        self.env_mu_params = np.zeros(shape=(self._buffer_size, self._params_size), dtype=np.float32)
        self.rewards = np.zeros((self._buffer_size, 1), dtype=np.float32)

    def add(self,mu_params, reward=0):
        self.env_mu_params[self.pos] = np.array(mu_params,dtype=np.float64).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.pos += 1

        if self.pos == self._buffer_size:
            self.full = True
            self.pos = 0

    def get(self, batch_size:Optional[int] = None):
        assert self.full, ''
        indices = np.random.permutation(self._buffer_size * self.n_envs)

        if batch_size is None:
            batch_size = self._buffer_size * self.n_envs

        start_idx = 0
        # 上面执行一次
        while start_idx < self._buffer_size * self.n_envs:
            yield self.get_samples(indices[start_idx: start_idx + batch_size])
            start_idx += batch_size

    def get_samples(self, batch_inds: np.ndarray):
        data = self.env_mu_params[batch_inds]

        return torch.tensor(data).to(self._device)

