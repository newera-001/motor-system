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
from stable_baselines3.common.policies import ActorCriticPolicy


from stable_baselines3.common.callbacks import CheckpointCallback

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

ENABLE_ENV_RANDOMIZER = True


def set_rand_seed(seed=None):
    if seed is None:
        seed = int(time.time())

    seed += 97 * MPI.COMM_WORLD.Get_rank()

    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)



def build_model(env,  output_dir):
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
        verbose=1)
    return model


def train(model, env, total_timesteps, output_dir="", int_save_freq=0):
    if (output_dir == ""):
        save_path = None
    else:
        save_path = os.path.join(output_dir, "model.zip")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    callbacks = []
    # Save a checkpoint every n steps
    if (output_dir != ""):
        if (int_save_freq > 0):
            int_dir = os.path.join(output_dir, "intermedate")
            callbacks.append(CheckpointCallback(save_freq=int_save_freq, save_path=int_dir,
                                                name_prefix='model'))

    model.learn(total_timesteps=total_timesteps, save_path=save_path, callback=callbacks)



def test(model, env, num_procs, num_episodes=None):
    curr_return = 0
    sum_return = 0
    episode_count = 0

    if num_episodes is not None:
        num_local_episodes = int(np.ceil(float(num_episodes) / num_procs))
    else:
        num_local_episodes = np.inf

    obs = env.reset()
    while episode_count < num_local_episodes:
        act, _ = model.predict(obs, deterministic=True)
        new_obs, r, done, info = env.step(act)
        curr_return += r

        if done:
            obs = env.reset()
            sum_return += curr_return
            episode_count += 1
        obs = new_obs
    sum_return = MPI.COMM_WORLD.allreduce(sum_return, MPI.SUM)
    episode_count = MPI.COMM_WORLD.allreduce(episode_count, MPI.SUM)

    mean_return = sum_return / episode_count

    if MPI.COMM_WORLD.Get_rank() == 0:
        print("Mean Return: " + str(mean_return))
        print("Episode Count: " + str(episode_count))



def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--seed", dest="seed", type=int, default=None)
    arg_parser.add_argument("--mode", dest="mode", type=str, default="train")
    arg_parser.add_argument("--motion_file", dest="motion_file", type=str,
                            default="motion_imitation/data/motions/dog_spin.txt")
    arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
    arg_parser.add_argument("--output_dir", dest="output_dir", type=str, default="output")
    arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=None)
    arg_parser.add_argument("--model_file", dest="model_file", type=str, default="")
    arg_parser.add_argument("--total_timesteps", dest="total_timesteps", type=int, default=2e8)
    arg_parser.add_argument("--int_save_freq", dest="int_save_freq", type=int,
                            default=0)  # save intermediate model every n policy steps

    args = arg_parser.parse_args()


    num_envs = 1
    enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")
    env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                          num_parallel_envs=num_envs,
                                          mode=args.mode,
                                          enable_randomizer=enable_env_rand,
                                          enable_rendering=args.visualize)

    model = build_model(env=env,
                        output_dir=args.output_dir)

    if args.model_file != "":
        model.set_parameters(args.model_file)

    if args.mode == "train":
        train(model=model,
              env=env,
              total_timesteps=args.total_timesteps,
              output_dir=args.output_dir,
              int_save_freq=args.int_save_freq)
    elif args.mode == "test":
        test(model=model,
             env=env,
             num_procs=num_procs,
             num_episodes=args.num_test_episodes)
    else:
        assert False, "Unsupported mode: " + args.mode


if __name__ == '__main__':
    main()
