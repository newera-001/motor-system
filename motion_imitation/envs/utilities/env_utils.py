"""Utility functions to manipulate environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from gym import spaces
import numpy as np

def flatter_observations(observation_dict, observation_excluded=()):
    """Flattens the observation dictionary to an array.
    If observation_excluded is passed in, it will still return a dictionary,
    which includes all the (key, observation_dict[key]) in observation_excluded,
    and ('other': the flattened array).
    Args:
      observation_dict: A dictionary of all the observations.
      observation_excluded: A list/tuple of all the keys of the observations to be
        ignored during flattening.
    Returns:
      An array or a dictionary of observations based on whether
        observation_excluded is empty.
    """
    if not isinstance(observation_excluded, (list, tuple)):
        observation_excluded = [observation_excluded]
    observation = []
    for key, value in observation_dict.items():
        if key not in observation_excluded:
            observation.append(np.asarray(value).flatten())
    flat_observations = np.concatenate(observation)

    if not observation_excluded:
        return flat_observations
    else:
        observation_dict_after_flatten = {'other':flat_observations}
        for key in observation_excluded:
            observation_dict_after_flatten[key] = observation_excluded[key]
        return collections.OrderedDict(list(observation_dict_after_flatten.items()))


def flatten_observation_spaces(observation_spaces, observation_excluede=()):
    """Flattens the dictionary observation spaces to gym.spaces.Box.
    If observation_excluded is passed in, it will still return a dictionary,
    which includes all the (key, observation_spaces[key]) in observation_excluded,
    and ('other': the flattened Box space).
    Args:
      observation_spaces: A dictionary of all the observation spaces.
      observation_excluded: A list/tuple of all the keys of the observations to be
        ignored during flattening.
    Returns:
      A box space or a dictionary of observation spaces based on whether
        observation_excluded is empty.
    """
    if not isinstance(observation_excluede,(list, tuple)):
        observation_excluede = [observation_excluede]
    lower_bound = []
    upper_bound = []
    for key, value in observation_spaces.spaces.items():
        lower_bound.append(np.asarray(value.low).flatten())
        upper_bound.append(np.asarray(value.high).flatten())
    lower_bound = np.concatenate(lower_bound)
    upper_bound = np.concatenate(upper_bound)

    observation_space = spaces.Box(lower_bound,upper_bound,dtype=np.float32)
    if not observation_excluede:
        return observation_space
    else:
        observation_space_after_flatten = {'other':observation_space}
        for key in observation_excluede:
            observation_space_after_flatten[key] = observation_excluede[key]
            return spaces.Dict(observation_space_after_flatten)

