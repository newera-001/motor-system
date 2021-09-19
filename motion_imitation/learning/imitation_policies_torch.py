import warnings
import numpy as np
import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional, Tuple, Type, Union, Callable
import gym

from stable_baselines3.common.distributions import spaces, DiagGaussianDistribution
from stable_baselines3.common.torch_layers import NatureCNN, MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, BaseFeaturesExtractor, FlattenExtractor

from stable_baselines3.common.preprocessing import preprocess_obs
from stable_baselines3.common.vec_env import vec_normalize


Schedule = Callable[[float], float]


def make_proba_dist_type(ac_space):
    """
    return an instance of ProbabilityDistributionType for the correct type of action space
    :param ac_space: (Gym Space) the input action space
    :return: (ProbabilityDistributionType) the appropriate instance of a ProbabilityDistributionType
    """
    if isinstance(ac_space, spaces.Box):
        assert len(ac_space.shape) == 1, "Error: the action space must be a vector"
        return DiagGaussianFixedVarProbabilityDistributionType(ac_space.shape[0])
    else:
        return make_proba_dist_type(ac_space)


# class DiagGaussianFixedVarProbabilityDistributionType(DiagGaussianProbabilityDistributionType):
#     def __init__(self, size):
#         super(DiagGaussianFixedVarProbabilityDistributionType, self).__init__(size)
#         return
#
#     def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector,
#                                         pi_init_scale=1.0, pi_init_bias=0.0, pi_init_std=1.0,
#                                         vf_init_scale=1.0, vf_init_bias=0.0):
#         mean = linear(pi_latent_vector, 'pi', self.size, init_scale=pi_init_scale, init_bias=pi_init_bias)
#         logstd = tf.get_variable(name='pi/logstd', shape=[1, self.size], initializer=tf.constant_initializer(np.log(pi_init_std)), trainable=False)
#         pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
#         q_values = linear(vf_latent_vector, 'q', self.size, init_scale=vf_init_scale, init_bias=vf_init_bias)
#         return self.proba_distribution_from_flat(pdparam), mean, q_values


def linear(input_tensor, n_hidden, init_scale=1., init_bias=0.):
    """
        Creates a fully connected layer for PyTorch
        :param input_tensor: (Tensor) The input tensor for the fully connected layer
        :param scope: (str) The TensorFlow variable scope
        :param n_hidden: (int) The number of hidden neurons
        :param init_scale: (int) The initialization scale
        :param init_bias: (int) The initialization offset bias
        :return: (PyTorch) fully connected layer
        """
    n_input = input_tensor.shape()[1].value
    layer = nn.Linear(n_input, n_hidden, bias=True)
    for p in layer.parameters():
        p.requires_grad = False
    layer.weight = init_scale
    layer.bias = init_bias

    return torch.matmul(input_tensor, torch.FloatTensor(layer.weight)) + layer

# def observation_input(
#     observation_space,
#     batch_size = None,
#     normalize_images: bool = True,
# ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
#     """
#         Preprocess observation to be to a neural network.
#         For images, it normalizes the values by dividing them by 255 (to have values in [0, 1])
#         For discrete observations, it create a one hot vector.
#
#         :param observation_space:
#         :param batch_size: (int) batch size for input
#         (default is None, so that resulting input placeholder can take tensors with any batch size)
#         :param normalize_images: Whether to normalize images or not
#             (True by default)
#         :return: processed_input_tensor
#         """
#     if isinstance(observation_space,gym.spaces.Box):
#         if normalize_images:
#             preocessed_observatons = (observation_space - observation_space.low)









# def conv_to_fc(input_tensor):
#     """
#        Reshapes a Tensor from a convolutional network to a Tensor for a fully connected network
#
#        :param input_tensor: (Torch Tensor) The convolutional input tensor
#        :return: (Torch Tensor) The fully connected output tensor
#     """
#     n_hidden = int(np.prod(input_tensor.size()))
#     input_tensor = torch.reshape(input_tensor,shape=[-1,n_hidden])
#     return input_tensor
#
# def nature_cnn(scaled_images, **kwargs):
#     """
#     CNN from Nature paper.
#
#     :param scaled_images: (Torch Tensor) Image input placeholder
#     :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
#     :return: (Torch) The CNN output layer
#     """
#     activ = nn.ReLU()
#     layer1 = nn.Conv2d(scaled_images,32,8,4)
#     layer1.weight = torch.full(layer1.weight.shape,fill_value=np.sqrt(2))
#     layer2 = nn.Conv2d(activ(layer1), 64, 4,2)
#     layer2.weight = torch.full(layer2.weight.shape,fill_value=np.sqrt(2))
#     layer3 = nn.Conv2d(activ(layer2),64,3,1)
#     layer3.weight = torch.full(layer3.weight.shape,fill_value=np.sqrt(2))
#     layer3 = conv_to_fc(layer3)
#     return activ(linear(layer3,n_hidden=512,init_scale=np.sqrt(2)))


class DiagGaussianFixedVarProbabilityDistributionType(DiagGaussianDistribution):
    def __init__(self, size):
        super(DiagGaussianFixedVarProbabilityDistributionType, self).__init__(size)

    def proba_distribution_from_latent(self, pi_latent_vector, vf_latent_vector,
                                       pi_init_scale=1.0, pi_init_bias=0., pi_init_std=1.0,
                                       vf_init_scale=1., vf_init_bias=0.):
        mean = linear(pi_latent_vector, self.action_dim, init_scale=pi_init_scale, init_bias=pi_init_bias)
        logstd = nn.Parameter(torch.full(size=[1, self.action_dim], fill_value=np.log(pi_init_std)),
                              requires_grad=False)
        q_values = linear(vf_latent_vector, self.action_dim, init_scale=vf_init_scale, init_bias=vf_init_bias)
        return self.log_prob_from_params(mean, logstd), mean, q_values


class FeedForwardPolicy(ActorCriticPolicy):
    """
       Policy class for actor-critic algorithms (has both policy and value prediction).
       Used by A2C, PPO and the likes.

       :param observation_space: Observation space
       :param action_space: Action space
       :param lr_schedule: Learning rate schedule (could be constant)
       :param net_arch: The specification of the policy and value networks.
       :param activation_fn: Activation function
       :param ortho_init: Whether to use or not orthogonal initialization
       :param use_sde: Whether to use State Dependent Exploration or not
       :param log_std_init: Initial value for the log standard deviation
       :param full_std: Whether to use (n_features x n_actions) parameters
           for the std instead of only (n_features,) when using gSDE
       :param sde_net_arch: Network architecture for extracting features
           when using gSDE. If None, the latent features from the policy will be used.
           Pass an empty list to use the states as features.
       :param use_expln: Use ``expln()`` function instead of ``exp()`` to ensure
           a positive standard deviation (cf paper). It allows to keep variance
           above zero and prevent it from growing too fast. In practice, ``exp()`` is usually enough.
       :param squash_output: Whether to squash the output using a tanh function,
           this allows to ensure boundaries when using gSDE.
       :param features_extractor_class: Features extractor to use.
       :param features_extractor_kwargs: Keyword arguments
           to pass to the features extractor.
       :param normalize_images: Whether to normalize images or not,
            dividing by 255.0 (True by default)
       :param optimizer_class: The optimizer to use,
           ``th.optim.Adam`` by default
       :param optimizer_kwargs: Additional keyword arguments,
           excluding the learning rate, to pass to the optimizer
       """

    def __init__(
            self,
            ob_space,
            ac_space,
            observation_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            squash_output: bool = False,
            features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
            features_extractor_kwargs: Optional[Dict[str, Any]] = None,
            normalize_images: bool = True,
            optimizer_class: Type[torch.optim.Optimizer] = torch.optim.Adam,
            optimizer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == torch.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super(ActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output,
        )