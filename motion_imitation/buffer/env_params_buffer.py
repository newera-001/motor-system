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
        self.mu_params = np.zeros(shape=(self._buffer_size, self._params_size[0]), dtype=np.float32)
        self.rewards = np.zeros(shape=(self._buffer_size, 1), dtype=np.float32)
        self.q_t_pose = np.zeros((self._buffer_size, self._params_size[1]), dtype=np.float32)
        self.default_q_pose = np.zeros((self._buffer_size, self._params_size[1]), dtype=np.float32)
        self.ref_key_point_pose = np.zeros((self._buffer_size, self._params_size[2]), dtype=np.float32)
        self.current_key_point_pose = np.zeros((self._buffer_size, self._params_size[2]), dtype=np.float32)
        self.motion_one_hot_index = np.zeros((self._buffer_size, self._params_size[3]), dtype=np.float32)


    def add(self,mu_params, q_t_pose, default_q_pose, ref_key_point_pose, current_key_point_pose, rewards, motion_id):
        self.mu_params[self.pos] = np.array(mu_params, dtype=np.float64).copy()
        self.rewards[self.pos] = np.array(rewards, dtype=np.float64).copy()
        self.q_t_pose[self.pos] = np.array(q_t_pose, dtype=np.float64).copy()
        self.default_q_pose[self.pos] = np.array(default_q_pose, dtype=np.float64).copy()
        self.ref_key_point_pose[self.pos] = np.array(ref_key_point_pose, dtype=np.float64).copy()
        self.current_key_point_pose[self.pos] = np.array(current_key_point_pose, dtype=np.float64).copy()
        # 对motion id 做one-hot 然后输入到网络里
        self.motion_one_hot_index[self.pos] = np.eye(self._params_size[3])[motion_id]

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
        data = {'mu_params': torch.tensor(self.mu_params[batch_inds]).to(self._device),
                'rewards': torch.tensor(self.rewards[batch_inds]).to(self._device),
                'q_t_pose': torch.tensor(self.q_t_pose[batch_inds]).to(self._device),
                'default_q_pose': torch.tensor(self.default_q_pose[batch_inds]).to(self._device),
                'ref_key_point_pose': torch.tensor(self.ref_key_point_pose[batch_inds]).to(self._device),
                'current_key_point_pose': torch.tensor(self.current_key_point_pose[batch_inds]).to(self._device),
                'motion_id': torch.tensor(self.motion_one_hot_index[batch_inds]).to(self._device)
                }

        return data
