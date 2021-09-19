"""A gin-config class for locomotion_gym_env.
This should be identical to locomotion_gym_config.proto.
"""
import attr
import typing
from .. import robots

@attr.s
class SimulationParameters(object):
  """Parameters specific for the pyBullet simulation."""
  sim_time_step_s = attr.ib(type=float, default=0.001)         # 执行一个对产生环境影响的操作所需的单位时间长度
  num_action_repeat = attr.ib(type=int, default=33)            # 一个动作会环境重复执行几次
  enable_hard_reset = attr.ib(type=bool, default=False)        # 重新初始化机器狗的状态（并包括环境的物理状态）
  enable_rendering = attr.ib(type=bool, default=False)         # 是否渲染训练过程
  enable_rendering_gui = attr.ib(type=bool, default=True)      # 渲染的方式
  robot_on_rack = attr.ib(type=bool, default=False)            # ？
  camera_distance = attr.ib(type=float, default=1.0)           # 相机距离机器人的距离
  camera_yaw = attr.ib(type=float, default=0)                  # 相机的角度
  camera_pitch = attr.ib(type=float, default=-30)
  render_width = attr.ib(type=int, default=480)                # 渲染窗口的尺寸
  render_height = attr.ib(type=int, default=360)
  egl_rendering = attr.ib(type=bool, default=False)
  motor_control_mode = attr.ib(type=int,
                               default=robots.robot_config.MotorControlMode.POSITION)    # 运动控制的方式
  reset_time = attr.ib(type=float, default=-1)
  enable_action_filter = attr.ib(type=bool, default=True)                # 是否做动作裁剪
  enable_action_interpolation = attr.ib(type=bool, default=True)         # 是否做动作插值
  allow_knee_contact = attr.ib(type=bool, default=False)                 # ？
  enable_clip_motor_commands = attr.ib(type=bool, default=True)          # 对电机的命名做裁剪


@attr.s
class ScalarField(object):
  """A named scalar space with bounds."""
  # 针对的是机器人腿上的电机 用name来区分它们 并且设置其对应的控制上下限
  name = attr.ib(type=str)
  upper_bound = attr.ib(type=float)
  lower_bound = attr.ib(type=float)


@attr.s
class LocomotionGymConfig(object):
  """Grouped Config Parameters for LocomotionGym."""
  simulation_parameters = attr.ib(type=SimulationParameters)
  log_path = attr.ib(type=typing.Text, default=None)
  profiling_path = attr.ib(type=typing.Text, default=None)

