# Copyright (c) Jinghuan Shang.

import os

from collections import deque
import copy
from PIL import Image
from typing import Tuple, Union

import cv2
import imageio
import numpy as np
import torch

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import robosuite
from robosuite.controllers import load_controller_config
import robosuite.environments.manipulation as manipulation
from robosuite.models.tasks import Task
from robosuite.utils.camera_utils import (
    CameraMover,
    get_camera_intrinsic_matrix,
    get_camera_extrinsic_matrix
)
from robosuite.utils.mjcf_utils import (
    find_elements, 
    find_parent
)
import robosuite.utils.transform_utils as T
from robosuite.wrappers import Wrapper

from active_gym.fov_env import (
    RecordWrapper
)

from active_gym.utils import (
    euler_to_rotation_matrix
)

class RobosuiteCameraMover(CameraMover):

    def rotate_camera(self, pyr, scale=5.0):
        """
        angle: [-1, 1]
        """
        camera_pos = np.array(self.env.sim.data.get_mocap_pos(self.mover_body_name))
        camera_rot = T.quat2mat(T.convert_quat(self.env.sim.data.get_mocap_quat(self.mover_body_name), to="xyzw"))
        pyr = np.pi * pyr * scale / 180.0
        R = euler_to_rotation_matrix(pyr)
        camera_pose = np.zeros((4, 4))
        camera_pose[:3, :3] = camera_rot
        camera_pose[:3, 3] = camera_pos
        camera_pose = camera_pose @ R
        # Update camera pose
        pos, quat = camera_pose[:3, 3], T.mat2quat(camera_pose[:3, :3])
        self.set_camera_pose(pos=pos, quat=quat)

        return pos, quat


class RobosuiteGymWrapper(Wrapper, gym.Env):
    metadata = None
    render_mode = None
    """
    Initializes the Gym wrapper. Modified from Robosuite's original gymwrapper

    Args:
        env (MujocoEnv): The environment to wrap.
        keys (None or list of str): If provided, each observation will
            consist of concatenated keys from the wrapped environment's
            observation dictionary. Defaults to proprio-state and object-state.

    Raises:
        AssertionError: [Object observations must be enabled if no keys]
    """

    def __init__(self, env, keys=None):
        # Run super method
        super().__init__(env=env)
        # Create name for gym
        robots = "".join([type(robot.robot_model).__name__ for robot in self.env.robots])
        self.name = robots + "_" + type(self.env).__name__

        # Get reward range
        self.reward_range = (0, self.env.reward_scale)

        if keys is None:
            keys = []
            # Add object obs if requested
            if self.env.use_object_obs:
                keys += ["object-state"]
            # Add image obs if requested
            if self.env.use_camera_obs:
                keys += [f"{cam_name}_image" for cam_name in self.env.camera_names]
            # Iterate over all robots to add to state
            for idx in range(len(self.env.robots)):
                keys += ["robot{}_proprio-state".format(idx)]
        self.keys = keys

        # Gym specific attributes
        self.env.spec = None

        # set up observation and action spaces
        obs, _ = self.reset()
        obs_space_dict = {}
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        for modality_name in self.modality_dims:
            space_base = 1. if "image" in modality_name else np.inf
            high = space_base * np.ones(self.modality_dims[modality_name])
            low = -high
            modality_space = Box(low, high)
            obs_space_dict[modality_name] = modality_space
        self.observation_space = Dict(obs_space_dict)
        # print (self.observation_space)

        low, high = self.env.action_spec
        self.action_space = Box(low, high)

    def _filter_obs_by_keys(self, obs):
        return {k:obs[k] for k in self.keys}
    
    def _normalize_image_obs(self, obs):
        for k in self.keys:
            if "image" in k:
                obs[k] = obs[k].astype(np.float32)/255.
                obs[k] = obs[k][::-1, ...] # !important to have this change, don't know why it is upside down
                obs[k] = np.moveaxis(obs[k], -1, 0)
                # print (obs[k].shape)
        return obs
    
    def _process_obs(self, obs):
        obs = self._filter_obs_by_keys(obs)
        obs = self._normalize_image_obs(obs)
        return obs

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return observation of normal OrderedDict and optionally resets seed
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        obs = self.env.reset()
        obs = self._process_obs(obs)
        return obs, {}

    def step(self, action):
        """
        Extends vanilla step() function call to return observation
        directly return the ordered dict
        images are normalized by dividing 255.
        """
        obs, reward, terminated, info = self.env.step(action)
        obs = self._process_obs(obs)
        return obs, reward, terminated, False, info

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
    
    def render(self, **kwargs):
        return 0

    def close(self):
        self.env.close()


class RobosuiteActiveEnv(gym.Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.args = args
        self.init_fov_pos = env.activeview_camera_init_pos
        self.init_fov_quat = env.activeview_camera_init_quat
        self.sensory_action_mode = args.sensory_action_mode
        self.sensory_action_dim = 6 # 6-DOF: pos, ang
        sensory_action_space_high = 1. * np.ones(self.sensory_action_dim)
        sensory_action_space_low = -sensory_action_space_high
        self.sensory_action_space = Box(low=sensory_action_space_low, 
                                       high=sensory_action_space_high, 
                                       dtype=np.float32)
        self.action_space = Dict({
            "motor_action": self.env.action_space,
            "sensory_action": self.sensory_action_space,
        })
        self.active_camera_mover = None
        self.active_camera_intrinsic = None

        self.fov_pos = None
        self.fov_quat = None

        self.return_camera_matrix = args.return_camera_matrix
        self.fixed_cam_extrinsic = None
        self.movable_cam_extrinsic = None

        self.record = args.record # not sure to exposed here for improve efficiency

        self.reset()

    def _get_camera_extrinsic_matrix(self, camera_name: str) -> np.ndarray:
        return get_camera_extrinsic_matrix(self.env.unwrapped.sim, 
                                           camera_name)
    
    def _get_camera_intrinsic_matrix(self, camera_name: str) -> np.ndarray:
        return get_camera_intrinsic_matrix(self.env.unwrapped.sim, 
                                           camera_name, 
                                           self.env.unwrapped.camera_heights[0],
                                           self.env.unwrapped.camera_widths[0])

    def _get_all_fixed_cam_extrinsic(self):
        all_extrinsic = {}
        for camera_name in self.env.unwrapped.camera_names:
            if "active" in camera_name or "eye" in camera_name: # skip movable cameras
                continue
            extrinsic = self._get_camera_extrinsic_matrix(camera_name=camera_name)
            all_extrinsic[camera_name] = extrinsic
        return all_extrinsic
    
    def _get_all_movable_cam_extrinsic(self):
        all_extrinsic = {}
        for camera_name in self.env.unwrapped.camera_names:
            if "active" in camera_name or "eye" in camera_name:
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

    def step(self, action):
        """
        Args:
            action : {"motor_action":
                    "sensory_action": }
        """
        # in robomimic, we need to first change this camera and then step
        # in order to return the observation from modified camera
        sensory_action = action["sensory_action"]
        self._sensory_step(sensory_action)
        state, reward, done, truncated, info = self.env.step(action=action["motor_action"])

        if self.return_camera_matrix:
            info["fov_pos"] = self.fov_pos.copy()
            info["fov_quat"] = self.fov_quat.copy()
            info["movable_cam_extrinsic"] = self.movable_cam_extrinsic.copy()
        
        # use active_view_image for the image observation from active camera
        # others are also returned for convienience
        return state, reward, done, truncated, info
    
    def _init_active_camera(self):
        self.active_camera_mover = RobosuiteCameraMover(
            env=self.env.unwrapped,
            camera="active_view",
        )
        if self.return_camera_matrix:
            activeview_camera_pos, activeview_camera_quat = self.active_camera_mover.get_camera_pose()
            self.fov_pos, self.fov_quat = activeview_camera_pos, activeview_camera_quat
            self.movable_cam_extrinsic = self._get_all_movable_cam_extrinsic()
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed, options)
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

    def save_record_to_file(self, file_path: str, selected_cameras=["active_view_image"]):
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
            

    # def render(self, **kwargs):
    #     return 0
    
    @property
    def unwrapped(self):
        """
        Grabs unwrapped environment

        Returns:
            env (MujocoEnv): Unwrapped environment
        """
        if hasattr(self.env, "unwrapped"):
            return self.env.unwrapped
        else:
            return self.env
        
def make_robosuite_active(load_model):
    """
    The decorator to add an controllable camera to robosuite
    """
    def wrapper(self, *args, **kwargs):
        load_model(self)
        # get the original arena
        mujoco_arena = self.model.mujoco_arena

        # get a default init active camera if not provided
        
        if self.init_view is None:
            if self.activeview_camera_init_pos is None or \
                self.activeview_camera_init_quat is None:
                self.init_view = "sideview"
        
        camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": self.init_view}, return_first=True)
        self.activeview_camera_init_pos = np.array([float(x) for x in camera.get("pos").split(" ")])
        self.activeview_camera_init_quat = np.array([float(x) for x in camera.get("quat").split(" ")])
        # print (self.activeview_camera_init_pos, self.activeview_camera_init_quat)

        # add a camera
        mujoco_arena.set_camera(
            camera_name="active_view", 
            pos=self.activeview_camera_init_pos,
            quat=self.activeview_camera_init_quat
        )

        # re-set the model using modifyed arena
        self.model = Task(
            mujoco_arena=mujoco_arena,
            mujoco_robots=self.model.mujoco_robots,
            mujoco_objects=self.model.mujoco_objects,
        )
    return wrapper

"""
Define active version of all robosuite environments
"""
class ActiveLift(robosuite.environments.manipulation.lift.Lift):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveStack(manipulation.stack.Stack):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveNutAssembly(manipulation.nut_assembly.NutAssembly):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None,
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveNutAssemblySingle(manipulation.nut_assembly.NutAssemblySingle):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveNutAssemblySquare(manipulation.nut_assembly.NutAssemblySquare):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveNutAssemblyRound(manipulation.nut_assembly.NutAssemblyRound):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActivePickPlace(manipulation.pick_place.PickPlace):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActivePickPlaceSingle(manipulation.pick_place.PickPlaceSingle):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActivePickPlaceMilk(manipulation.pick_place.PickPlaceMilk):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActivePickPlaceBread(manipulation.pick_place.PickPlaceBread):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActivePickPlaceCereal(manipulation.pick_place.PickPlaceCereal):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActivePickPlaceCan(manipulation.pick_place.PickPlaceCan):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveDoor(manipulation.door.Door):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveWipe(manipulation.wipe.Wipe):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveToolHang(manipulation.tool_hang.ToolHang):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveTwoArmLift(manipulation.two_arm_lift.TwoArmLift):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveTwoArmPegInHole(manipulation.two_arm_peg_in_hole.TwoArmPegInHole):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveTwoArmHandover(manipulation.two_arm_handover.TwoArmHandover):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

class ActiveTwoArmTransport(manipulation.two_arm_transport.TwoArmTransport):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        init_view=None,
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        self.init_view = init_view
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    @make_robosuite_active
    def _load_model(self):
        super()._load_model()

#------------ Vanilla way, kept for understanding/experimental purpose
class VanillaActiveDoor(manipulation.door.Door):
    def __init__(self, 
                activeview_camera_init_pos=None, 
                activeview_camera_init_quat=None, 
                **kwargs):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        # ! important note for pose if using CameraMover
        # the quat used here is inconsistent with the quat get from camera_mover
        # we need to shift the axes to make it right
        # shift [[3,0,1,2]] if use camera mover
        self.activeview_camera_init_quat = activeview_camera_init_quat 
        super().__init__(**kwargs)

    # override _load_model to add camera at this stage
    def _load_model(self):
        """
        Loads an xml model, puts it in self.model
        """
        super()._load_model()

        # get the original arena
        mujoco_arena = self.model.mujoco_arena

        # get a default init active camera if not provided
        if self.activeview_camera_init_pos is None or \
            self.activeview_camera_init_quat is None:
            camera = find_elements(root=mujoco_arena.worldbody, tags="camera", attribs={"name": "sideview"}, return_first=True)
            self.activeview_camera_init_pos = np.array([float(x) for x in camera.get("pos").split(" ")])
            self.activeview_camera_init_quat = np.array([float(x) for x in camera.get("quat").split(" ")])
            # print (self.activeview_camera_init_pos, self.activeview_camera_init_quat)

        # add a camera
        mujoco_arena.set_camera(
            camera_name="active_view", 
            pos=self.activeview_camera_init_pos,
            quat=self.activeview_camera_init_quat
        )

        # re-set the model using modifyed arena
        self.model = Task(
            mujoco_arena=mujoco_arena,
            mujoco_robots=self.model.mujoco_robots,
            mujoco_objects=self.model.mujoco_objects,
        )
#------------            

class RobosuiteEnvArgs:
    """
    args for ActiveRobosuiteEnv
    """
    def __init__(self, task, seed, obs_size: Tuple[int, int], **kwargs):
        self.env_backend = "robosuite"
        self.device = None
        self.seed = seed
        self.max_episode_length = 1000
        self.task = task
        self.frame_stack = 4
        self.action_repeat = 4
        self.obs_size = obs_size
        self.record = False
        self.clip_reward = False
        self.selected_obs_names = ["sideview_image", "active_view_image"]
        self.sensory_action_mode = "relative"
        self.return_camera_matrix = False

        # active env kwargs but pass into robosuite make
        self.init_view = "sideview"

        # robosuite kwargs pass into make
        self.robots = "Panda"
        self.controller_configs = load_controller_config(default_controller="OSC_POSE")
        self.has_renderer = False
        self.has_offscreen_renderer = True
        self.horizon = 500
        self.use_object_obs = False
        self.use_camera_obs = True
        self.camera_names = ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand', "active_view"]
        self.reward_shaping = True
        self.render_gpu_device_id = int(os.environ.get("MUJOCO_EGL_DEVICE_ID", -1))

        for k, v in kwargs.items():
            self.__setattr__(k, v)

def get_robosuite_kwargs(args: RobosuiteEnvArgs):
    robosuite_kwargs = {
        "robots": args.robots,
        "controller_configs": args.controller_configs,
        "has_renderer": args.has_renderer,
        "has_offscreen_renderer": args.has_offscreen_renderer,
        "horizon": args.horizon,
        "use_object_obs": args.use_object_obs,
        "use_camera_obs": args.use_camera_obs,
        "camera_names": args.camera_names,
        "camera_heights": args.obs_size[0],
        "camera_widths": args.obs_size[1],
        "reward_shaping": args.reward_shaping,
        "init_view": args.init_view,
        # "render_gpu_device_id": args.render_gpu_device_id,
    }
    return robosuite_kwargs

def make_active_robosuite_env(args: RobosuiteEnvArgs):
    """
    create active robosuite envs
    it is auto compatible with base (non-active) env if we ignore the active_view camera
    and ignore the sensory action
    """
    robosuite_kwargs = get_robosuite_kwargs(args)
    env = robosuite.make(env_name="Active"+args.task, **robosuite_kwargs)
    env = RobosuiteGymWrapper(env, keys=args.selected_obs_names)
    env = RecordWrapper(env, args)
    env = RobosuiteActiveEnv(env, args)
    return env


def make_base_robosuite_env(args: RobosuiteEnvArgs):
    """
    alternative to create base robosuite envs
    be compatible with make_active_robosuite_env
    """
    robosuite_kwargs = get_robosuite_kwargs(args)
    # double check it does not include that
    if "init_view" in robosuite_kwargs:
        del robosuite_kwargs["init_view"]
    if "active_view" in args.camera_name:
        args.camera_names.remove("active_view")
        args.selected_obs_names.remove("active_view_image")
    env = robosuite.make(env_name=args.task, **robosuite_kwargs)
    env = RobosuiteGymWrapper(env, keys=args.selected_obs_names)
    env = RecordWrapper(env, args)
    return env


# test
if __name__ == "__main__":
    # show available envs
    print (list(robosuite.environments.base.REGISTERED_ENVS.keys()))
    # ['Lift', 'Stack', 'NutAssembly', 'NutAssemblySingle', 'NutAssemblySquare', 
    # 'NutAssemblyRound', 'PickPlace', 'PickPlaceSingle', 'PickPlaceMilk', 'PickPlaceBread', 
    # 'PickPlaceCereal', 'PickPlaceCan', 'Door', 'Wipe', 'ToolHang', 'TwoArmLift', 
    # 'TwoArmPegInHole', 'TwoArmHandover', 'TwoArmTransport']

    # test base env and modification for active env
    # load default controller parameters for Operational Space Control (OSC)
    controller_config = load_controller_config(default_controller="OSC_POSE")
    camera_names = ['frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand']

    active = True
    if active:
        camera_names += ["active_view"]
        env = ActiveNutAssemblyRound(
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=True,
            use_object_obs=False,
            use_camera_obs=True,
            camera_names=camera_names,
            camera_heights=84,
            camera_widths=84,
            reward_shaping=True
        )
        

    # print (env.sim.model.camera_names)
    # ('frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand')
    obs = env.reset()
    active_camera_mover = RobosuiteCameraMover(
        env=env,
        camera="active_view",
    )

    obss_dict = {}
    for i in range(5):
        print (f"step {i}")
        obs, reward, done, info = env.step(np.random.random(env.action_dim))
        print (list(obs.keys()))
        for k in obs:
            print (obs[k].shape)
        # print (type(obs))
        active_camera_mover.move_camera(direction=[1.0, 1.0, 1.0], scale=0.05)
        active_camera_mover.rotate_camera(pyr=np.ndarray([0.0, 0.0, 1.0])) # +up, +left, +clcwise
        activeview_camera_pos, activeview_camera_quat = active_camera_mover.get_camera_pose()
        # print (activeview_camera_pos, activeview_camera_quat)
        # print (type(obs), obs.shape, obs.max(), obs.min())
        for cam_id, cam_name in enumerate(camera_names):
            obs_cam = obs[f"{cam_name}_image"]
            if cam_name not in obss_dict:
                obss_dict[cam_name] = []
            obss_dict[cam_name].append(obs_cam[::-1, ..., ::-1]) # !important to have this change, don't know why it is upside down
            # print (obs_cam.shape)
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    for cam_name in obss_dict:
        cam_obss = np.hstack(obss_dict[cam_name])
        cv2.imwrite(f"tmp/robosuite_env_test_{cam_name}.png", cam_obss)
    env.close()

    # test gym & active wrapped robosuite env
    env_args = RobosuiteEnvArgs(
        task="TwoArmTransport",
        robots=["Panda", "Panda"],
        seed=0,
        obs_size=(84, 84), # other params are default param
    )
    env = make_active_robosuite_env(env_args)
    obs, info = env.reset()
    obss_dict = {}
    for i in range(5):
        print (f"step {i}")
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        print (list(obs.keys()))
        # print (activeview_camera_pos, activeview_camera_quat)
        # print (type(obs), obs.shape, obs.max(), obs.min())
        for obs_name in obs:
            obs_cam = obs[f"{obs_name}"]
            if obs_name not in obss_dict:
                obss_dict[obs_name] = []
            obss_dict[obs_name].append(obs_cam[..., ::-1]*255.) # !important to have this change, don't know why it is upside down
            # print (obs_cam.shape)
    if not os.path.exists("tmp"):
        os.mkdir("tmp")
    for obs_name in obss_dict:
        cam_obss = np.hstack(obss_dict[obs_name])
        cv2.imwrite(f"tmp/robosuite_env_wrapped_test_{obs_name}.png", cam_obss)
    env.close()


    print (f"perf test {env_args.task}")
    import time
    env = make_active_robosuite_env(env_args)
    obs, info = env.reset()
    start = time.time()
    i = 0
    while i < 100:
        # print (f"step {i}")
        obs, reward, done, truncated, info = env.step(env.action_space.sample())
        i += 1
        if done:
            env.reset()
    print (f"{i/(time.time()-start)} FPS")
    env.close()
