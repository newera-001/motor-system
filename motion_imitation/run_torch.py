import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import argparse
from mpi4py import MPI
import numpy as np
import os
import random
import torch

import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.learning import ppo_imitation_torch as ppo_imitation
from motion_imitation.learning import encoder
from stable_baselines3.common.policies import ActorCriticPolicy

from stable_baselines3.common.callbacks import CheckpointCallback


ENABLE_ENV_RANDOMIZER = True


def set_rand_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    seed += 97 * MPI.COMM_WORLD.Get_rank()

    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)



def build_model(env, env_randomizer, z_size, is_load, output_dir, type_name='dog_pace'):
    policy_kwargs = {
        "net_arch": [{"pi": [512, 256],
                      "vf": [512, 256]}],
        "activation_fn": torch.nn.ReLU
    }


    model = ppo_imitation.PPOImitation(
        policy=ActorCriticPolicy,
        env=env,
        device='cuda',
        gamma=0.95,
        policy_kwargs=policy_kwargs,
        tensorboard_log=output_dir,
        verbose=1,
        env_randomizers=env_randomizer,
        # input_size is 107 
        encoder=encoder.Encoder(107, z_size),        # 里面并没有保存encoder模型
        type_name = type_name,
        z_size = z_size,
        is_load= is_load
    )
    return model


def train(model,  total_timesteps, output_dir="",type_name='dog_pace', int_save_freq=0):
    if (output_dir == ""):
        save_path = None
    else:
        save_path = os.path.join(output_dir, f"{type_name}_model.zip")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if (output_dir != ""):
        if (int_save_freq > 0):
            int_dir = os.path.join(output_dir, "intermedate")
            callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                                name_prefix=f'{type_name}_model'))

    model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks,
                policy_load=f'{type_name}policy-checkpoint.pkl',
                encoder_load=f'{type_name}encoder-checkpoint.pkl')



def test(model, encoder_model:torch.nn.Module, env, num_procs, num_episodes=None):

    encoder = torch.load(encoder_model).to('cuda')

    curr_return = 0
    sum_return = 0
    episode_count = 0

    if num_episodes is not None:
        num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
    else:
        num_local_episodes = np.inf

    obs = torch.tensor(env.reset()).to('cuda')
    params = env.get_env_parameters()

    params_values = []
    for key in params.keys():
        if type(params[key]) == float:
            params_values.append(params[key])
        else:
            params_values.extend(params[key])

    while episode_count < num_local_episodes:

        z_latent = encoder(torch.tensor(params_values).float().to('cuda'))
        obs = torch.cat([obs,z_latent], dim=0)
        act, _ = model.predict(obs.detach().unsqueeze(dim=0), deterministic=True)
        new_obs, r, done, info = env.step(act)

        params = env.get_env_parameters()
        params_values = []
        for key in params.keys():
            if type(params[key]) == float:
                params_values.append(params[key])
            else:
                params_values.extend(params[key])

        curr_return += r

        if done:
            obs = torch.tensor(env.reset()).to('cuda')
            sum_return += curr_return
            episode_count += 1
        obs = torch.tensor(new_obs).to('cuda')
    # sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
    # episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

    mean_return = sum_return / episode_count

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Mean Return: " + str(mean_return))
        print("Episode Count: " + str(episode_count))



def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--motion_file", dest="motion_file", type=str,
                            default="dog_spin.txt")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
    arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument('--encoder_file',dest='encoder_file',type=str,default='')
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e7)
    arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int,
                            default=0)  # save intermediate model every n policy steps
    arg_parser.add_argument('--num_envs',dest='num_envs',type=int,default=1)
    arg_parser.add_argument('--z_size', dest='z_size',type=int, default=8)
    arg_parser.add_argument('--type_name',dest='type_name',type=str)
    arg_parser.add_argument('--is_load',dest='is_load',type=bool,default=False)


    args = arg_parser.parse_args()

    enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")

    base_motion_path = 'motion_imitation/data/motions/'
    motion_list = args.motion_file.split('|')
    motion_files = []
    for motion_name in motion_list:
        motion_files.append(base_motion_path+motion_name)

    # 创建训练需要的env环境 仿真世界 agent step reset reward 这些agent去环境交互的借口
    # 还有仿真物理环境参数随机化（每episode随机一次）
    env, randomizer = env_builder.build_imitation_env(motion_files=motion_files,
                                          num_parallel_envs=args.num_envs,
                                          mode=args.mode,
                                          enable_randomizer=enable_env_rand,
                                          enable_rendering=args.visualize)


    # 算法模型 还有 数据buffer等
    model = build_model(env=env,
                        env_randomizer=randomizer,
                        output_dir=args.output_dir,
                        z_size=args.z_size,
                        type_name=args.type_name,
                        is_load=args.is_load)

    if args.model_file != "":
        model.set_parameters(args.model_file)

    if args.mode == "train":
        train(model=model,
              total_timesteps=args.total_timesteps,
              output_dir=args.output_dir,
              int_save_freq=args.int_save_freq,
              type_name=args.type_name)
    elif args.mode == "test":
        test(model=model,
             env=env,
             num_procs=args.num_envs,
             num_episodes=args.num_test_episodes,
             encoder_model=args.encoder_file)
    else:
        assert False, "Unsupported mode: " + args.mode

    return


if __name__ == '__main__':
    main()
