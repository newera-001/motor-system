import torch

class CheckSaveLoad(object):
    """
    This class is used to save model params and buffer data.
    """
    def __init__(self,type_name, is_load = False):
        self._type_name = type_name
        self._is_load = is_load

    def save_policy(self, model:torch.nn.Module, optimizer:torch.optim, current_progress_remaining,
                    num_timesteps, iterations, n_updates, last_obs, last_mu_params):
        """
        因为buffer的数据是每learn一次 buffer都要重新收集并且环境状态重置 学习的动作也将会随机选取
        因此不需要保存checkpoint点处的env状态、buffer和选取动作的状态
        """
        checkpoint = {'model_state_dict': model.state_dict(),
                       'optimizer_state_dict': optimizer.state_dict(),
                       'current_progress_remaining': current_progress_remaining,
                       'num_timesteps': num_timesteps,
                       'iterations': iterations,
                       'n_updates': n_updates,
                       'last_obs': last_obs,
                       'last_mu_params': last_mu_params}
        torch.save(checkpoint,f'{self._type_name}policy-checkpoint.pkl')

    def save_encoder(self, model: torch.nn.Module, optimizer: torch.optim):
        checkpoint = {'model_state_dict': model.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint,f'{self._type_name}encoder-checkpoint.pkl')

    def load_policy(self, filename):
        checkpoint = torch.load(filename)
        return checkpoint

    def load_encoder(self, filename):
        checkpoint = torch.load(filename)
        return checkpoint