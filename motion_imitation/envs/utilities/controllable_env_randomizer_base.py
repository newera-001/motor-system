"""Base class for controllable environment randomizer."""

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

from motion_imitation.envs.utilities import env_randomizer_base


class ControllableEnvRandomizerBase(env_randomizer_base.EnvRandomizerBase):
  """Base class for environment randomizer that can be manipulated explicitly.
  Randomizes physical parameters of the objects in the simulation and adds
  perturbations to the stepping of the simulation.
  """
  def get_randomization_parameters(self):
    """Get the parameters of the randomization."""
    raise NotImplementedError

  def set_randomization_from_parameters(self, env, randomization_parameters):
    """Set the parameters of the randomization."""
    raise NotImplementedError

  