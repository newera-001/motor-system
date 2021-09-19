"""Converts a list of sensors to gym space."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import gym
from gym import spaces
import numpy as np
import typing

from . import sensor

class UnsupportedConversionError(Exception):
    """An exception when the function cannot convert sensors to the gym space."""

class AmbiguousDataTyperError(Exception):
    """An exception when the function cannot determine the data type"""


def convert_sensors_to_gym_space(sensors : typing.List[sensor.Sensor]) -> gym.Space:
    """Convert a list of sensors to the corresponding gym space.
    Args:
      sensors: a list of the current sensors
    Returns:
      space: the converted gym space
    Raises:
      UnsupportedConversionError: raises when the function cannot convert the
        given list of sensors.
    """
    if all([isinstance(s, sensor.BoxSpaceSensor) and s.get_dimension() ==1 for s in sensors]):
        return convert_1d_box_sensors_to_gym_space(sensors)
    raise UnsupportedConversionError('sensors = '+ str(sensors))

def convert_1d_box_sensors_to_gym_space(sensors:typing.List[sensor.Sensor]) -> gym.Space:
    """Convert a list of 1D BoxSpaceSensors to the corresponding gym space.
    Args:
      sensors: a list of the current sensors
    Returns:
      space: the converted gym space
    Raises:
      UnsupportedConversionError: raises when the function cannot convert the
        given list of sensors.
      AmbiguousDataTypeError: raises when the function cannot determine the
        data types because they are not uniform.
    """
    # check if all sensors are 1D BoxSpaceSensors
    if not all([isinstance(s, sensor.BoxSpaceSensor) and s.get_dimension() == 1 for s in sensors]):
        raise UnsupportedConversionError('sensors = '+ str(sensors))

    # Check if all sensor have the same data type
    sensors_dtypes = [s.get_dtype() for s in sensors]
    if sensors_dtypes.count(sensors_dtypes[0]) != len(sensors_dtypes):
        raise AmbiguousDataTyperError('sensor datatypes are inhomogeneous')

    """
    针对全体sensor建的gym-space
    """
    lower_bound = np.concatenate([s.get_lower_bound() for s in sensors])
    upper_bound = np.concatenate([s.get_upper_bound() for s in sensors])
    observation_space = spaces.Box(lower_bound,upper_bound,dtype=np.float32)

    return observation_space

def convert_sensors_to_gym_space_dictionary(sensors:typing.List[sensor.Sensor]) -> gym.Space:
    """Convert a list of sensors to the corresponding gym space dictionary.
    Args:
      sensors: a list of the current sensors （包含仿真环境和机器狗上的）
    Returns:
      space: the converted gym space dictionary
    Raises:
      UnsupportedConversionError: raises when the function cannot convert the
        given list of sensors.
    """

    """
    一个传感器创建自己对应的gym-space dict的索引是sensor的名字 value 是对应的gym-space
    """
    gym_space_dict = {}
    for s in sensors:
        if isinstance(s,sensor.BoxSpaceSensor):
            gym_space_dict[s.get_name()] = spaces.Box(np.array(s.get_lower_bound()),
                                                      np.array(s.get_upper_bound()),
                                                      dtype=np.float32)
        else:
            raise UnsupportedConversionError('sensors = ' + str(sensors))

    return spaces.Dict(gym_space_dict)


