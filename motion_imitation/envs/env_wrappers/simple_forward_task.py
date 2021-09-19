"""A simple locomotion task and termination condition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

class SimpleForwardTask(object):
    """Default empy task"""
    def __init__(self):
        """Initializers the task"""
        self.current_base_pose = np.zeros(3)
        self.last_base_pose = np.zeros(3)

    def __call__(self, env, *args, **kwargs):
        return self.reward(env)

    def reset(self,env):
        """Resets the internal state of the task."""
        self._env = env
        self.last_base_pose = env.robot.GetBasePosition()
        self.current_base_pose = self.last_base_pose

    def update(self, env):
        """Updates the internal state of the task."""
        self.last_base_pose = self.current_base_pose
        self.current_base_pose = env.robot.GetBasePosition()

    def done(self, env):
        """Checks if the episode is over.
           If the robot base becomes unstable (based on orientation), the episode
           terminates early.
        """
        rot_quat = env.robot.GetBaseOrientation()
        rot_mat = env.pybullet_client.getMatrixFromQuaternion(rot_quat)
        return rot_mat[-1] < 0.85

    def reward(self, env):
        """Get the reward without side effects."""
        del env
        return self.current_base_pose[0] - self.last_base_pose[0]

