import os

from collections import deque
from PIL import Image
from typing import Tuple, Union

import cv2
import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import robosuite
from robosuite.controllers import load_controller_config
import robosuite.environments.manipulation as manipulation
from robosuite.models.tasks import Task
from robosuite.utils.camera_utils import CameraMover
from robosuite.utils.mjcf_utils import find_elements, find_parent
from robosuite.wrappers import Wrapper

from active_gym.fov_env import (
    RecordWrapper
)

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
        obs = self.env.reset()
        obs_space_dict = {}
        self.modality_dims = {key: obs[key].shape for key in self.keys}
        for modality_name in self.modality_dims:
            space_base = 1. if "image" in modality_name else np.inf
            high = space_base * np.ones(self.modality_dims[modality_name])
            low = -high
            modality_space = Box(low, high)
            obs_space_dict[modality_name] = modality_space

        low, high = self.env.action_spec
        self.action_space = Box(low, high)

    def _filter_obs_by_keys(self, obs):
        return {k:obs[k] for k in self.keys}
    
    def _normalize_image_obs(self, obs):
        for k in self.keys:
            if "image" in k:
                obs[k] = obs[k].astype(np.float32)/255.
                obs[k] = obs[k][::-1, ...] # !important to have this change, don't know why it is upside down
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
    
    def close(self):
        self.env.close()


class RobosuiteActiveEnv(gym.Wrapper):
    def __init__(self, env, args):
        super().__init__(env)
        self.init_fov_pos = env.activeview_camera_init_pos
        self.init_fov_quat = env.activeview_camera_init_quat
        self.visual_action_mode = args.visual_action_mode
        self.visual_action_dim = 6 # 6-DOF: pos, ang
        visual_action_space_high = 1. * np.ones(self.visual_action_dim)
        visual_action_space_low = -visual_action_space_high
        self.visual_action_space = Box(low=visual_action_space_low, 
                                       high=visual_action_space_high, 
                                       dtype=np.float32)
        self.action_space = Dict({
            "physical_action": self.env.action_space,
            "visual_action": self.visual_action_space,
        })
        self.active_camera_mover = None

        self.reset()

    def _sensory_step(self, sensory_action):
        movement, rotation = sensory_action[:3], sensory_action[3:]
        self.active_camera_mover.move_camera(direction=movement, scale=0.05)
        self.active_camera_mover.rotate_camera(point=None, axis=rotation, angle=5) # +up, +left, +clcwise
        activeview_camera_pos, activeview_camera_quat = self.active_camera_mover.get_camera_pose()
        self.fov_pos, self.fov_quat = activeview_camera_pos, activeview_camera_quat

    def step(self, action):
        """
        Args:
            action : {"physical_action":
                    "visual_action": }
        """
        # in robomimic, we need to first change this camera and then step
        # in order to return the observation from modified camera
        sensory_action = action["visual_action"]
        self._sensory_step(sensory_action)
        state, reward, done, truncated, info = self.env.step(action=action["physical_action"])

        info["fov_pos"] = self.fov_pos.copy()
        info["fov_quat"] = self.fov_quat.copy()
        if self.record:
            if not done:
                self.save_transition(info["fov_loc"])
                self.save_transition(info["fov_quat"])
        
        # use active_view_image for the image observation from active camera
        # others are also returned for convienience
        return state, reward, done, truncated, info
    
    def _init_active_camera(self):
        self.active_camera_mover = CameraMover(
            env=self.env.unwrapped,
            camera="active_view",
        )
        activeview_camera_pos, activeview_camera_quat = self.active_camera_mover.get_camera_pose()
        self.fov_pos, self.fov_quat = activeview_camera_pos, activeview_camera_quat
    
    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed, options)
        self._init_active_camera()
        info["fov_pos"] = self.fov_pos.copy()
        info["fov_quat"] = self.fov_quat.copy()
        if self.record:
            if not done:
                self.save_transition(info["fov_loc"])
                self.save_transition(info["fov_quat"])
        return obs, info
    
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
    return wrapper

"""
Define active version of all robosuite environments
"""
class ActiveLift(robosuite.environments.manipulation.lift.Lift):
    def __init__(
        self, 
        activeview_camera_init_pos=None, 
        activeview_camera_init_quat=None, 
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        **kwargs
    ):
        self.activeview_camera_init_pos = activeview_camera_init_pos
        self.activeview_camera_init_quat = activeview_camera_init_quat 
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
        self.visual_action_mode = "relative"

        # robosuite kwargs
        self.robots = "Panda"
        self.controller_configs = load_controller_config(default_controller="OSC_POSE")
        self.has_renderer = False
        self.has_offscreen_renderer = True
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
        "use_object_obs": args.use_object_obs,
        "use_camera_obs": args.use_camera_obs,
        "camera_names": args.camera_names,
        "camera_heights": args.obs_size[0],
        "camera_widths": args.obs_size[1],
        "reward_shaping": args.reward_shaping,
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
    if "active_view" in args.camera_name:
        args.camera_name.remove("active_view")
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
    active_camera_mover = CameraMover(
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
        active_camera_mover.rotate_camera(point=None, axis=[0.0, 0.0, 1.0], angle=20) # +up, +left, +clcwise
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