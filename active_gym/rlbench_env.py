# Copyright (c) Jinghuan Shang.

import os
import copy
import imageio
import random
import functools
import warnings

from collections import deque
from PIL import Image
from typing import Tuple, Union, Callable

import cv2
import torch
import numpy as np

from transforms3d.quaternions import (
    mat2quat,
    quat2mat
)

import gymnasium as gym
from gymnasium.spaces import (
    Box, 
    Dict,
    Discrete, 
)

from pyrep.const import RenderMode
from pyrep.objects.dummy import Dummy
from pyrep.objects.vision_sensor import VisionSensor

from active_gym.fov_env import (
    RecordWrapper
)

from active_gym.utils import (
    euler_to_rotation_matrix
) 

from gymnasium.envs.registration import register
from rlbench.backend.task import TASKS_PATH
from rlbench.utils import name_to_task_class
from rlbench.gym.rlbench_env import RLBenchEnv
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.environment import Environment
from rlbench.observation_config import (
    ObservationConfig, 
    CameraConfig
)


def ignore_warnings(category: Warning):
    def ignore_warnings_decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=category)
                return func(*args, **kwargs)
        return wrapper
    return ignore_warnings_decorator

# for gynmaniusm compabaility
class RLBenchGymnasiumEnv(RLBenchEnv):
    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, task_class, observation_mode='state',
                 render_mode: Union[None, str] = None):
        self._observation_mode = observation_mode
        self._render_mode = render_mode
        obs_config = ObservationConfig()
        if observation_mode == 'state':
            obs_config.set_all_high_dim(False)
            obs_config.set_all_low_dim(True)
        elif observation_mode == 'vision':
            obs_config.set_all(True)
        else:
            raise ValueError('Unrecognised observation_mode: %s.' % observation_mode)

        action_mode = MoveArmThenGripper(JointVelocity(), Discrete())
        self.env = Environment(
            action_mode, obs_config=obs_config, headless=True)
        self.env.launch()
        self.task = self.env.get_task(task_class)

        _, obs = self.task.reset()

        self.action_space = Box(
            low=-1.0, high=1.0, shape=self.env.action_shape
        )

        if observation_mode == 'state':
            self.observation_space = Box(
                low=-np.inf, high=np.inf, shape=obs.get_low_dim_data().shape, dtype=np.float32
            )
        elif observation_mode == 'vision':
            self.observation_space = Dict({
                "state": Box(
                    low=-np.inf, high=np.inf,
                    shape=obs.get_low_dim_data().shape, dtype=np.float32),
                "left_shoulder_rgb": Box(
                    low=0., high=1., shape=obs.left_shoulder_rgb.shape, dtype=np.float32),
                "right_shoulder_rgb": Box(
                    low=0., high=1., shape=obs.right_shoulder_rgb.shape, dtype=np.float32),
                "wrist_rgb": Box(
                    low=0., high=1., shape=obs.wrist_rgb.shape, dtype=np.float32),
                "front_rgb": Box(
                    low=0., high=1., shape=obs.front_rgb.shape, dtype=np.float32),
                })

    def step(self, action):
        obs, reward, terminate = self.task.step(action)
        truncated = False
        return self._extract_obs(obs), reward, terminate, truncated, {}
        
    
    def reset(self, seed=None, options=None):
        descriptions, obs = self.task.reset()
        obs = self._extract_obs(obs)
        info = {"task_descriptions": descriptions}
        return obs, info

    def _extract_obs(self, obs):
        if self._observation_mode == 'state':
            return obs.get_low_dim_data()
        elif self._observation_mode == 'vision':
            return {
                "state": obs.get_low_dim_data().astype(np.float32),
                "left_shoulder_rgb": obs.left_shoulder_rgb.astype(np.float32) / 255.,
                "right_shoulder_rgb": obs.right_shoulder_rgb.astype(np.float32) / 255.,
                "wrist_rgb": obs.wrist_rgb.astype(np.float32) / 255.,
                "front_rgb": obs.front_rgb.astype(np.float32) / 255.,
            }

@ignore_warnings(category=UserWarning)
def register_rlbench_envs():
    TASKS = [t for t in os.listdir(TASKS_PATH)
            if t != '__init__.py' and t.endswith('.py')]

    for task_file in TASKS:
        task_name = task_file.split('.py')[0]
        task_class = name_to_task_class(task_name)
        register(
            id='%s-vision-v0' % task_name,
            entry_point='active_gym.rlbench_env:RLBenchGymnasiumEnv',
            kwargs={
                'task_class': task_class,
                'observation_mode': 'vision',
                'render_mode': 'rgb_array'
            }
        )

register_rlbench_envs()

class RLBenchCameraMover:

    def __init__(self, camera):
        self.camera = camera

    def move_camera(self, direction, scale=5.0):
        camera_pose = self.camera.get_pose()
        camera_pos, camera_rot = camera_pose[:3], camera_pose[3:]
        pos = camera_pos + direction * scale
        self.camera.set_position(pos)

    def rotate_camera(self, pyr:np.ndarray, scale=1.0):
        """
        pry: in degree, not rad
        """
        camera_pose = self.camera.get_pose()
        camera_pos, camera_rot = camera_pose[:3], camera_pose[3:]
        camera_rot = euler_to_rotation_matrix(camera_rot)[:3, :3]
        pyr = np.pi * pyr * scale / 180.0 # convert to rad
        R = euler_to_rotation_matrix(pyr)
        camera_pose = np.zeros((4, 4))
        camera_pose[:3, :3] = camera_rot
        camera_pose[:3, 3] = camera_pos
        camera_pose = camera_pose @ R

        # Update camera pose
        quat = mat2quat(camera_pose[:3, :3])
        self.camera.set_quaternion(quat)


class RLBenchActiveEnv(gym.Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.args = args
        # self.init_fov_pos = env.activeview_camera_init_pos
        # self.init_fov_quat = env.activeview_camera_init_quat
        self.sensory_action_mode = args.sensory_action_mode
        self.sensory_action_dim = 6 # 6-DOF: pos, ang
        sensory_action_space_high = 1. * np.ones(self.sensory_action_dim, dtype=np.float32)
        sensory_action_space_low = -sensory_action_space_high
        self.sensory_action_space = Box(low=sensory_action_space_low, 
                                       high=sensory_action_space_high, 
                                       dtype=np.float32)
        self.action_space = Dict({
            "motor_action": self.env.action_space,
            "sensory_action": self.sensory_action_space,
        })
        self.active_camera = self.add_active_camera()
        self.active_camera_mover = RLBenchCameraMover(self.active_camera)
        self.active_camera_intrinsic = None

        self.fov_pos = None
        self.fov_quat = None

        self.return_camera_matrix = args.return_camera_matrix
        self.fixed_cam_extrinsic = None
        self.movable_cam_extrinsic = None

        self.record = args.record # not sure to exposed here for improve efficiency

        # try to make gym not complain types
        self.observation_space = Dict({
            "state": Box(
                low=np.full(self.observation_space["state"].shape, -np.inf, dtype=np.float32), 
                high=np.full(self.observation_space["state"].shape, np.inf, dtype=np.float32),
                shape=self.observation_space["state"].shape,
                
            ),
            "left_shoulder_rgb": Box(
                low=np.full(self.observation_space["left_shoulder_rgb"].shape, 0, dtype=np.float32), 
                high=np.full(self.observation_space["left_shoulder_rgb"].shape, 1, dtype=np.float32), 
                shape=self.observation_space["left_shoulder_rgb"].shape
            ),
            "right_shoulder_rgb": Box(
                low=np.full(self.observation_space["right_shoulder_rgb"].shape, 0, dtype=np.float32), 
                high=np.full(self.observation_space["right_shoulder_rgb"].shape, 1, dtype=np.float32), 
                shape=self.observation_space["right_shoulder_rgb"].shape
            ),
            "wrist_rgb": Box(
                low=np.full(self.observation_space["wrist_rgb"].shape, 0, dtype=np.float32), 
                high=np.full(self.observation_space["wrist_rgb"].shape, 1, dtype=np.float32), 
                shape=self.observation_space["wrist_rgb"].shape
            ),
            "front_rgb": Box(
                low=np.full(self.observation_space["front_rgb"].shape, 0, dtype=np.float32), 
                high=np.full(self.observation_space["front_rgb"].shape, 1, dtype=np.float32), 
                shape=self.observation_space["front_rgb"].shape
            ),
            "active_rgb": Box(
                low=np.full(self.args.obs_size + (3,), 0., dtype=np.float32), 
                high=np.full(self.args.obs_size + (3,), 1., dtype=np.float32), 
                shape=self.args.obs_size + (3,)
            ),
        })

        self.cameras = {
            "left_shoulder_camera": self.env.unwrapped.env._scene._cam_over_shoulder_left,
            "right_shoulder_camera": self.env.unwrapped.env._scene._cam_over_shoulder_left,
            "wrist_camera": self.env.unwrapped.env._scene._cam_over_shoulder_left,
            "front_camera": self.env.unwrapped.env._scene._cam_over_shoulder_left,
            "active_camera": self.active_camera
        }

    def add_active_camera(self):
        cam_placeholder = Dummy(self.args.init_view) # e.g.'cam_cinematic_placeholder'
        active_cam = VisionSensor.create(self.args.obs_size)
        active_cam.set_pose(cam_placeholder.get_pose())
        active_cam.set_render_mode(RenderMode.OPENGL3)
        return active_cam

    def _get_camera_extrinsic_matrix(self, camera_name: str) -> np.ndarray:
        return self.cameras[camera_name].get_extrinsic_matrix()
    
    def _get_camera_intrinsic_matrix(self, camera_name: str) -> np.ndarray:
        return self.cameras[camera_name].get_intrinsic_matrix()

    def _get_all_fixed_cam_extrinsic(self):
        all_extrinsic = {}
        for camera_name in self.cameras:
            if "active" in camera_name or "wrist" in camera_name: # skip movable cameras
                continue
            extrinsic = self._get_camera_extrinsic_matrix(camera_name=camera_name)
            all_extrinsic[camera_name] = extrinsic
        return all_extrinsic
    
    def _get_all_movable_cam_extrinsic(self):
        all_extrinsic = {}
        for camera_name in self.cameras:
            if "active" in camera_name or "wrist" in camera_name:
                extrinsic = self._get_camera_extrinsic_matrix(camera_name=camera_name)
                all_extrinsic[camera_name] = extrinsic
        return all_extrinsic

    def _sensory_step(self, sensory_action):
        if np.all(sensory_action == 0):
            pass
        else:
            movement, rotation = sensory_action[:3], sensory_action[3:]
            # print ("movement, rotation", movement, rotation)
            if len(rotation) < 3:
                rotation = np.concatenate([rotation, np.zeros((3-len(rotation), ), dtype=rotation.dtype)])
            
            self.active_camera_mover.move_camera(direction=movement, scale=0.01)
            self.active_camera_mover.rotate_camera(pyr=rotation, scale=1.0) # +up, +left, +clcwise

        if self.return_camera_matrix:
            activeview_camera_pos, activeview_camera_quat = self.active_camera_mover.get_camera_pose()
            self.fov_pos, self.fov_quat = activeview_camera_pos, activeview_camera_quat
            self.movable_cam_extrinsic = self._get_all_movable_cam_extrinsic()

    def _get_active_camera_obs(self):
        active_rgb = self.active_camera.capture_rgb()
        return active_rgb
    
    def step(self, action):
        """
        Args:
            action : {"motor_action":
                    "sensory_action": }
        """
        # first change this camera and then step
        # in order to return the observation from modified camera
        sensory_action = action["sensory_action"]
        self._sensory_step(sensory_action)
        obs, reward, done, truncated, info = self.env.step(action=action["motor_action"])
        obs["active_rgb"] = self._get_active_camera_obs()

        if self.return_camera_matrix:
            info["fov_pos"] = self.fov_pos.copy()
            info["fov_quat"] = self.fov_quat.copy()
            info["movable_cam_extrinsic"] = self.movable_cam_extrinsic.copy()

        return obs, reward, done, truncated, info
    
    def _init_active_camera(self):
        cam_placeholder = Dummy(self.args.init_view) # e.g.'cam_cinematic_placeholder'
        active_cam = VisionSensor.create(self.args.obs_size)
        active_cam.set_pose(cam_placeholder.get_pose())

        if self.return_camera_matrix:
            activeview_camera_pos, activeview_camera_quat = self.get_camera_pose()
            self.fov_pos, self.fov_quat = activeview_camera_pos, activeview_camera_quat
            self.movable_cam_extrinsic = self._get_all_movable_cam_extrinsic()

    def get_camera_pose(self):
        pose = self.active_camera.get_pose()
        pos, quat = pose[:3], pose[3:]
        return pos, quat
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs["active_rgb"] = self._get_active_camera_obs()
        self._init_active_camera()
        
        if self.return_camera_matrix:
            self.fixed_cam_extrinsic = self._get_all_fixed_cam_extrinsic()
            self.active_camera_intrinsic = self._get_camera_intrinsic_matrix("active_view")

            info["fov_pos"] = self.fov_pos.copy()
            info["fov_quat"] = self.fov_quat.copy()
            info["cam_intrinsic"] = self.active_camera_intrinsic.copy()
            info["fixed_cam_extrinsic"] = copy.deepcopy(self.fixed_cam_extrinsic)
            info["movable_cam_extrinsic"] = copy.deepcopy(self.movable_cam_extrinsic)
        return obs, info

    def save_record_to_file(self, file_path: str, selected_cameras=["active_rgb"]):
        if self.record:
            video_path = file_path.replace(".pt", ".mp4")
            size = self.prev_record_buffer["state"][0][selected_cameras[0]].shape[:2][::-1]
            fps = 10
            # print (size)
            video_writer = imageio.get_writer(video_path, fps=fps)
            # video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            video_len = len(self.prev_record_buffer["state"])
            # print ("video length", video_len)
            for i in range(video_len):
                frames = [self.prev_record_buffer["state"][i][camera] for camera in selected_cameras]
                frames = (np.moveaxis(np.hstack(frames), 0, -1)*255.).astype(np.uint8)
                video_writer.append_data(frames)
            # video_writer.release()
            video_writer.close()
            # empty buffer
            self.prev_record_buffer["rgb"] = video_path
            # self.prev_record_buffer["state"] = [0] * len(self.prev_record_buffer["reward"])
            torch.save(self.prev_record_buffer, file_path)

class RLBenchEnvArgs:
    """
    args for ActiveRobosuiteEnv
    """
    def __init__(self, task, seed, obs_size: Tuple[int, int], **kwargs):
        self.env_backend = "rlbench"

        self.device = None
        self.seed = seed
        self.max_episode_length = 1000
        self.task = task
        self.frame_stack = 4
        self.action_repeat = 4
        self.obs_size = obs_size
        self.record = False
        self.clip_reward = False
        self.sensory_action_mode = "relative"
        self.return_camera_matrix = False

        # active env kwargs but pass into robosuite make
        self.init_view = "cam_cinematic_placeholder"

        # rlbench kwargs pass into make
        for k, v in kwargs.items():
            self.__setattr__(k, v)

# def get_rlbench_kwargs(args: RLBenchEnvArgs):
#     rlbench_kwargs = {
#         "task": args.task
#     }
#     return rlbench_kwargs

def make_active_rlbench_env(args: RLBenchEnvArgs):
    """
    create active rlbench envs
    """
    # rlbench_kwargs = get_rlbench_kwargs(args)
    env = gym.make(args.task.replace("active_", ""))
    env = RecordWrapper(env, args)
    env = RLBenchActiveEnv(env, args)
    return env


def make_base_rlbench_env(args: RLBenchEnvArgs):
    """
    alternative to create base rlbench envs
    be compatible with make_active_rlbench_env
    """
    env = gym.make(args.task.replace("active_", ""))
    env = RecordWrapper(env, args)
    return env


if __name__ == "__main__":
    # some variables for tests that I'm reluctant to remove
    training_steps = 120
    episode_length = 40

    # env = gym.make('reach_target-state-v0', render_mode="rgb_array")

    # test for basic rlbench
    # env = gym.make('reach_target-vision-v0', render_mode="rgb_array")

    
    # for i in range(1):
    #     if i % episode_length == 0:
    #         print('Reset Episode')
    #         obs = env.reset()
    #     obs, reward, terminate, _ = env.step(env.action_space.sample())
    #     print (obs, type(obs))

    # # print('Done')
    # env.close()

    # test for my adaptation of base rlbench
    # env_args = RLBenchEnvArgs(
    #     task="reach_target-vision-v0",
    #     seed=0,
    #     obs_size=(84, 84),
    # )
    # env = make_base_rlbench_env(env_args)
    # for i in range(1):
    #     if i % episode_length == 0:
    #         print('Reset Episode')
    #         obs, info = env.reset()
    #     obs, reward, terminate, truncated, _ = env.step(env.action_space.sample())
    #     print (type(obs))
    
    # env.close()

    # test active rl bench
    task = "reach_target-vision-v0"
    env_args = RLBenchEnvArgs(
        task=task,
        seed=0,
        obs_size=(84, 84)
    )
    env = make_active_rlbench_env(task, env_args)
    for i in range(1):
        if i % episode_length == 0:
            print('Reset Episode')
            obs, info = env.reset()
        print (list(obs.keys()))
        for j in range(5):
            obs, reward, terminate, truncated, _ = env.step(env.action_space.sample())
            print (obs["active_rgb"].shape, obs["active_rgb"].dtype)
            im = Image.fromarray((obs["active_rgb"]*255.).astype(np.uint8))
            im.save(f"tmp/{j:02d}.png")
        
    env.close()
