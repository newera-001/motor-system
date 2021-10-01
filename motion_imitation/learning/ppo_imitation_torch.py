import os
import inspect
import warnings

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import time
import numpy as np
import gym
from gym import spaces
from typing import Any, Dict, Optional, Type, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.policies import ActorCriticPolicy  # 相当于imitation_policies
from stable_baselines3.ppo import ppo
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean, explained_variance
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from motion_imitation.Buffer import env_params_buffer


class PPOImitation(ppo.PPO):
    """
        Proximal Policy Optimization algorithm (PPO) (clip version)
        Paper: https://arxiv.org/abs/1707.06347
        Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
        https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
        and Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)
        Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html
        :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
        :param env: The environment to learn from (if registered in Gym, can be str)
        :param learning_rate: The learning rate, it can be a function
            of the current progress remaining (from 1 to 0)
        :param n_steps: The number of steps to run for each environment per update
            (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
            NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
            See https://github.com/pytorch/pytorch/issues/29372
        :param batch_size: Minibatch size
        :param n_epochs: Number of epoch when optimizing the surrogate loss
        :param gamma: Discount factor
        :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        :param clip_range: Clipping parameter, it can be a function of the current progress
            remaining (from 1 to 0).
        :param clip_range_vf: Clipping parameter for the value function,
            it can be a function of the current progress remaining (from 1 to 0).
            This is a parameter specific to the OpenAI implementation. If None is passed (default),
            no clipping will be done on the value function.
            IMPORTANT: this clipping depends on the reward scaling.
        :param ent_coef: Entropy coefficient for the loss calculation
        :param vf_coef: Value function coefficient for the loss calculation
        :param max_grad_norm: The maximum value for the gradient clipping
        :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
            instead of action noise exploration (default: False)
        :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
            Default: -1 (only sample at the beginning of the rollout)
        :param target_kl: Limit the KL divergence between updates,
            because the clipping is not enough to prevent large update
            see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
            By default, there is no limit on the kl div.
        :param tensorboard_log: the log location for tensorboard (if None, no logging)
        :param create_eval_env: Whether to create a second environment that will be
            used for evaluating the agent periodically. (Only available when passing string for the environment)
        :param policy_kwargs: additional arguments to be passed to the policy on creation
        :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
        :param seed: Seed for the pseudo random generators
        :param device: Device (cpu, cuda, ...) on which the code should be run.
            Setting it to auto, the code will be run on the GPU if possible.
        :param _init_setup_model: Whether or not to build the network at the creation of the instance
        """

    def __init__(self,
                 policy: Union[str, Type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule] = 3e-4,
                 n_steps: int = 2048,
                 batch_size: int = 64,
                 n_epochs: int = 10,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_range: Union[float, Schedule] = 0.2,
                 clip_range_vf: Union[None, float, Schedule] = None,
                 ent_coef: float = 0.0,
                 vf_coef: float = 0.5,
                 max_grad_norm: float = 0.5,
                 use_sde: bool = False,
                 sde_sample_freq: int = -1,
                 target_kl: Optional[float] = None,
                 tensorboard_log: Optional[str] = None,
                 create_eval_env: bool = False,
                 policy_kwargs: Optional[Dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[torch.device, str] = "auto",
                 _init_setup_model: bool = True,
                 env_randomizers=None,
                 encoder: nn.Module = None,
                 type_name = None,
                 z_size:int = 8
                 ):
        super(PPOImitation, self).__init__(policy,
                                           env,
                                           learning_rate=learning_rate,
                                           n_steps=n_steps,
                                           gamma=gamma,
                                           gae_lambda=gae_lambda,
                                           ent_coef=ent_coef,
                                           vf_coef=vf_coef,
                                           max_grad_norm=max_grad_norm,
                                           use_sde=use_sde,
                                           sde_sample_freq=sde_sample_freq,
                                           tensorboard_log=tensorboard_log,
                                           policy_kwargs=policy_kwargs,
                                           verbose=verbose,
                                           device=device,
                                           create_eval_env=create_eval_env,
                                           seed=seed,
                                           _init_setup_model=False,
                                           )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        assert (
                batch_size > 1
        ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps   # 1*2048

            assert (
                    buffer_size > 1
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.target_kl = target_kl
        self.device = device
        self._env_randomizers = env_randomizers if env_randomizers else []
        self.params_buffer = env_params_buffer.Env_Params_Buffer(buffer_size=self.n_steps,
                                                                 device=device,
                                                                 params_shape=self.get_randomizer_shape(self._env_randomizers))

        self.encoder = encoder.to(device)
        self.encoder_optim = self.encoder.optimizer(learning_rate, self.encoder.parameters())
        self.type_name = type_name

        if _init_setup_model:
            self._setup_model()

        """
        重新构建网络 扩展添加隐变量的维度
        160 observation， z_size z-latent
        """
        self.policy.mlp_extractor.value_net = nn.Sequential(
            nn.Linear(in_features=160 + z_size, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU()
        )

        self.policy.mlp_extractor.policy_net = nn.Sequential(
            nn.Linear(in_features=160 + z_size, out_features=512, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=256, bias=True),
            nn.ReLU()
        )

        self.policy.to(device)

    # 构建训练过程
    def train(self):
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)

        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []      # 近似kl散度
            # Do a complete pass on the rollout buffer
            for rollout_data, mu_param in zip(self.rollout_buffer.get(self.batch_size),
                                              self.params_buffer.get(self.batch_size) ):  # data buffer里随机抽batch_size大小的数据出来训练

                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                # TODO: investigate why there is no issue with the gradient
                # if that line is commented (as in SAC)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                latent_param = self.encoder(mu_param.float())
                latent_param_policy = latent_param.clone()
                obs = torch.cat([rollout_data.observations, latent_param_policy.detach()],dim=1).to(self.device)

                # 价值网络 判断动作的好坏
                values, log_prob, entropy = self.policy.evaluate_actions(obs, actions)

                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Calculate encoder loss
                # 提升其概率
                encoder_loss = -torch.mean(F.softmax(latent_param,dim=1) * advantages.reshape(-1,1).detach())

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the different between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + torch.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)                              # 用来更新价值网络
                value_losses.append(value_loss.item())

                # Entropy loss favor exploration
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -torch.mean(-log_prob)
                else:
                    entropy_loss = -torch.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss           # 用来更新策略网络

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html

                # 计算kl散度（正则项，但loss里并为用kl散度来做正则）和 重要性采样比  这里并没有用kl散度做正则项
                with torch.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = torch.mean((torch.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                """
                Optimize Encoder
                """
                self.encoder_optim.zero_grad()
                encoder_loss.backward()
                self.encoder_optim.step()

                """
                Optimize PPO
                """
                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Logs
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/encoder_loss", encoder_loss.item())
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", torch.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "PPOImitation",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        save_iters = 20,
        save_path = None,
    ) -> "PPOImitation":

        iteration = 0
        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name
        )

        callback.on_training_start(locals(), globals())

        while self.num_timesteps < total_timesteps:

            # 收集训练数据 并 初始化环境
            continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer,
                                                      params_buffer=self.params_buffer, n_rollout_steps=self.n_steps)

            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            if total_timesteps % save_iters ==0:
                self.save(save_path)
                self.save_encoder(self.encoder, self.type_name)

            # 调用train 方法来更新
            self.train()

        callback.on_training_end()

        return self


    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        params_buffer: env_params_buffer.Env_Params_Buffer = None
    ) -> bool:
        """
            Collect experiences using the current policy and fill a ``RolloutBuffer``.
            The term rollout here refers to the model-free notion and should not
            be used with the concept of rollout used in model-based RL or planning.

            :param env: The training environment
            :param callback: Callback that will be called at each step
                (and at the beginning and end of the rollout)
            :param rollout_buffer: Buffer to fill with rollouts
            :param n_steps: Number of experiences to collect per environment
            :return: True if function returned with at least `n_rollout_steps`
                collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"

        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()    # 重置buffer里的数据
        params_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        # 开始收集数据(gpu进行)
        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
            # 收集数据时不计算梯度
            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                z_latent = self.encoder(torch.from_numpy(np.array(self._last_mu_params)).to(self.device).float())
                obs = torch.cat([obs_tensor, z_latent], dim=1).to(self.device)

                # 根据当前环境选取动作
                actions, values, log_probs = self.policy.forward(obs)
            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            # 这一步包含了domain randomization （new_obs 包含了all sensors 和参考机器狗未来t 时刻的动作数据）
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # 获取环境参数
            mu_param_values = []
            for env_randomizer in self._env_randomizers:
                mu_params = env_randomizer.get_randomization_parameters()  # 获取µ ∼p(µ) 如果源码中真没做这一工作
                params_values = []
                for key in mu_params.keys():
                    if type(mu_params[key]) == float:
                        params_values.append(mu_params[key])
                    else:
                        params_values.extend(mu_params[key])
                mu_param_values.append(params_values)

            params_buffer.add(mu_param_values)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            self._last_obs = new_obs
            self._last_mu_params = mu_param_values

            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            z_latent = self.encoder(torch.from_numpy(np.array(mu_param_values)).to(self.device).float())
            obs = torch.cat([obs_tensor, z_latent], dim=1).to(self.device)
            _, values, _ = self.policy.forward(obs)

        # advantage
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def get_randomizer_shape(self, env_randomizer):
        params_shape = []
        for env_randomizer in env_randomizer:
            mu_params = env_randomizer.get_randomization_parameters()  # 获取µ ∼p(µ) 如果源码中真没做这一工作
            params_values = []
            for key in mu_params.keys():
                if type(mu_params[key]) == float:
                    params_values.append(mu_params[key])
                else:
                    params_values.extend(mu_params[key])
            params_shape.append(params_values)

        # 这个初始化的环境隐变量可能不好
        self._last_mu_params = params_shape
        return np.array(params_shape).size

    def save_encoder(self, model, type_name):
        torch.save(model,f'output/{type_name}_encoder.pkl')

