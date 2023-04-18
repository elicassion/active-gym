# Copyright (c) Jinghuan Shang.
# reference: dmc2gym: https://github.com/denisyarats/dmc2gym

import copy
import random

from collections import deque
from PIL import Image
from typing import Tuple, Union

import cv2
import gym
from gym.spaces import Box

import numpy as np
import torch

from dm_control import suite
from dm_env import specs

from .fov_env import (
    FixedFovealEnv, 
    FlexibleFovealEnv, 
    FixedFovealPeripheralEnv,
    FlexibleFovealEnvActionType
)

def _spec_to_box(spec, dtype) -> gym.Space:
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = np.int_(np.prod(s.shape))
        if type(s) == specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) == specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return Box(low, high, dtype=dtype)


def _flatten_obs(obs) -> np.ndarray:
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)

class DMCEnvArgs:
    def __init__(self, domain_name: str, task_name: str, seed: int, 
                        obs_size: Tuple[int, int], **kwargs):
        self.seed = seed
        self.domain_name = domain_name
        self.task_name = task_name
        self.obs_size = obs_size
        self.task_kwargs = {}
        self.visualize_reward = False
        self.from_pixels = True
        self.grey = True
        self.camera_id = 0
        self.action_repeat = 4 # == action repeat
        self.frame_stack = 3
        self.mask_out = False
        self.environment_kwargs = {}
        self.clip_reward = False
        self.record = False
        for k, v in kwargs.items():
            self.__setattr__(k, v)

class DMCBaseEnv(gym.Env):
    def __init__(
        self,
        args: DMCEnvArgs
    ):
        self.args = args
        self.seed_num           = args.seed
        self.from_pixels: bool  = args.from_pixels
        self.grey: bool         = args.grey
        self.obs_size: Tuple[int, int] = args.obs_size
        self.camera_id: int     = args.camera_id
        self.action_repeat: int    = args.action_repeat
        self.frame_stack: int   = args.frame_stack

        self.task_kwargs = args.task_kwargs
        self.task_kwargs['random'] = self.seed_num
        
        self.clip_reward = args.clip_reward
        
        self.state_buffer = deque([], maxlen=args.frame_stack)

        # create task
        self.dmc_env = suite.load(
            domain_name=args.domain_name,
            task_name=args.task_name,
            task_kwargs=self.task_kwargs,
            visualize_reward=args.visualize_reward,
            environment_kwargs=args.environment_kwargs
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self.dmc_env.action_spec()], np.float32)
        self._norm_action_space = Box(
            low=-1.0,
            high=1.0,
            shape=self._true_action_space.shape,
            dtype=np.float32
        )

        # create observation space
        if self.from_pixels:
            if self.grey:
                shape = (self.frame_stack, ) + self.obs_size
            else:
                shape = (self.frame_stack, 3) + self.obs_size
            self._observation_space = Box(
                low=-1., high=1., shape=shape, dtype=np.float32
            )
        else:
            self._observation_space = _spec_to_box(
                self.dmc_env.observation_spec().values(),
                np.float32
            )
            
        self._state_space = _spec_to_box(
            self.dmc_env.observation_spec().values(),
            np.float32
        )
        
        self.current_state = None

        # set seed
        self.seed(seed=self.seed_num)

        # init reward and episode len counter
        self.cumulative_reward = 0
        self.ep_len = 0
        self.record_buffer = None
        self.prev_record_buffer = None
        self.record = args.record

    def __getattr__(self, name):
        return getattr(self.dmc_env, name)
    
    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self.action_repeat

    def seed(self, seed):
        self._true_action_space.seed(seed)
        self._norm_action_space.seed(seed)
        self._observation_space.seed(seed)

    def _convert_action(self, action) -> np.ndarray:
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    def _get_obs(self, time_step):
        if self.from_pixels:
            height, width = self.obs_size
            obs = self.dmc_env.physics.render(
                    height=height, width=width, camera_id=self.camera_id
                )
            if self.grey:
                obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
            obs = obs.astype(np.float32) / 255.
        else:
            obs = _flatten_obs(time_step.observation)
        return obs
    
    def _get_info(self, time_step):
        return {"internal_state": self.dmc_env.physics.get_state().copy(),
                "discount": time_step.discount,
                "ep_len": self.ep_len}
    
    def _reset_record_buffer(self):
        self.prev_record_buffer = copy.deepcopy(self.record_buffer)
        self.record_buffer = {"rgb": [], "state":[] , "action": [], "reward": [], "done": [], 
                              "truncated": [], "info": [], "return_reward": []}

    def _reset_buffer(self):
        for _ in range(self.frame_stack):
            self.state_buffer.append(np.zeros(self.obs_size))

    def _reset(self):
        self._reset_buffer()
        self.ep_len = 0
        self.cumulative_reward = 0

        time_step = self.dmc_env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        info = self._get_info(time_step)
        self.state_buffer.append(obs)
        state = np.stack(self.state_buffer, axis=0)

        if self.record:
            rgb = self.render()
            self._reset_record_buffer()
            self._save_transition(obs, done=False, info=info, rgb=rgb)
        return state, info

    def _step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0

        # frame skip / action repeat
        for _ in range(self.action_repeat):
            time_step = self.dmc_env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)

        # for recording
        self.ep_len += 1
        self.cumulative_reward += reward

        # variables to be returned
        self.state_buffer.append(obs)
        return_reward = np.sign(reward) if self.clip_reward else reward
        state = np.stack(self.state_buffer, axis=0)
        info = self._get_info(time_step)
        truncated = False
        
        if self.record:
            rgb = self.render()
            self._save_transition(state, action, reward, done, False, info, rgb=rgb, return_reward=return_reward)
        return state, return_reward, done, truncated, info
    
    def reset(self):
        state, info = self._reset()
        return state, info

    def step(self, action):
        return self._step(action)

    def render(self, mode='rgb_array', obs_size=None, camera_id=0) -> np.ndarray:
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        height, width = obs_size if obs_size is not None else self.obs_size
        camera_id = camera_id or self.camera_id
        rgb = self.dmc_env.physics.render(
            height=height, width=width, camera_id=camera_id
        )
        return rgb
    
    def close():
        pass

    def _save_transition(self, state, action=None, reward=None, done=None, 
                            truncated=None, info=None, rgb=None,
                            return_reward=None):
        if (done is not None) and (not done):
            self.record_buffer["state"].append(state)
            # print (np.sum(rgb))
            self.record_buffer["rgb"].append(rgb)
        if action is not None:
            self.record_buffer["action"].append(action)
        if reward is not None:
            self.record_buffer["reward"].append(reward)
        if done is not None and len(self.record_buffer["state"]) > 1:
            self.record_buffer["done"].append(done) 
        if truncated is not None:
            self.record_buffer["truncated"].append(truncated)
        if info is not None and len(self.record_buffer["state"]) > 1:
            self.record_buffer["info"].append(info)
        if return_reward is not None:
            self.record_buffer["return_reward"].append(return_reward)
    
    def save_record_to_file(self, file_path: str):
        if self.record:
            video_path = file_path.replace(".pt", ".mp4")
            size = self.prev_record_buffer["rgb"][0].shape[:2][::-1]
            fps = 30
            # print (size)
            video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            for frame in self.prev_record_buffer["rgb"]:
                video_writer.write(frame)
            self.prev_record_buffer["rgb"] = video_path
            self.prev_record_buffer["state"] = [0] * len(self.prev_record_buffer["reward"])
            torch.save(self.prev_record_buffer, file_path)
            video_writer.release()

def DMCFixedFovealEnv(args: DMCEnvArgs) -> gym.Wrapper:
    base_env = DMCBaseEnv(args)
    wrapped_env = FixedFovealEnv(base_env, args)
    return wrapped_env

def DMCFlexibleFovealEnv(args: DMCEnvArgs)-> gym.Wrapper:
    base_env = DMCBaseEnv(args)
    wrapped_env = FlexibleFovealEnv(base_env, args)
    return wrapped_env
    
def DMCFixedFovealPeripheralEnv(args: DMCEnvArgs)-> gym.Wrapper:
    base_env = DMCBaseEnv(args)
    wrapped_env = FixedFovealPeripheralEnv(base_env, args)
    return wrapped_env
    
# test
if __name__ == "__main__":
    env_args = DMCEnvArgs(domain_name="reacher", task_name="easy", seed=42, obs_size=(84, 84), 
                          from_pixels=True, gray=True,
                        )
    
    # print(suite.ALL_TASKS)

    # dmc backbone test
    print ("dmc backbone test")
    backbone_env = suite.load(
        domain_name=env_args.domain_name,
        task_name=env_args.task_name,
    )

    obs = backbone_env.reset()
    print (obs)

    
    print ("dmc base env test")
    ori_env = DMCBaseEnv(env_args)
    ori_env.reset()
    for i in range(5):
        obs, reward, done, _, _ = ori_env.step(ori_env.action_space.sample())
        print (obs.shape)
    ori_env.save_record_to_file("test_env_record_file.pt")

    print ("test fixed foveal, flexible foveal, foveal peripheral envs")
    print ("fixed foveal")
    env_args = DMCEnvArgs(domain_name="reacher", task_name="easy", seed=42, obs_size=(84, 84), 
                            fov_size=(50, 50),
                            fov_init_loc=(0, 0),
                            visual_action_mode="absolute",
                            visual_action_space=(-10.0, 10.0),
                            resize_to_full=False,
                            from_pixels=True, gray=True,
                        )
    fixed_foveal_env = DMCFixedFovealEnv(env_args)
    fixed_foveal_env.reset()
    while not done:
        obs, reward, done, _, _ = fixed_foveal_env.step({
                                "physical_action": fixed_foveal_env.action_space["physical_action"].sample(), 
                                "visual_action": np.array([random.randint(0,10), random.randint(10,50)])})
        print (obs.shape, fixed_foveal_env.fov_loc)

    print ("flexible foveal")
    env_args = DMCEnvArgs(domain_name="reacher", task_name="easy", seed=42, obs_size=(84, 84), 
                            fov_size=(50, 50),
                            fov_init_loc=(0, 0),
                            visual_action_mode="absolute",
                            visual_action_space=(-10.0, 10.0),
                            resize_to_full=False,
                            from_pixels=True, gray=True,
                        )
    flexible_foveal_env = DMCFlexibleFovealEnv(env_args)
    flexible_foveal_env.reset()
    done = False
    while not done:
            action_type = random.choice(list(FlexibleFovealEnvActionType))
            if action_type == FlexibleFovealEnvActionType.FOV_LOC:
                action = np.array([random.randint(0,10), random.randint(10,50)])
            elif action_type == FlexibleFovealEnvActionType.FOV_RES:
                action = np.array([random.randint(10, 40), random.randint(15, 70)])
            
            ac = {
                "physical_action": flexible_foveal_env.action_space["physical_action"].sample(), 
                "visual_action": action,
                "visual_action_type": np.array((action_type.value,))
            }
            # print (ac)
            obs, reward, done, _, _ = flexible_foveal_env.step(ac)
            print (obs.shape, flexible_foveal_env.fov_loc, flexible_foveal_env.fov_res)

    print ("foveal peripheral")
    env_args = DMCEnvArgs(domain_name="reacher", task_name="easy", seed=42, obs_size=(84, 84), 
                            fov_size=(50, 50),
                            fov_init_loc=(0, 0),
                            peripheral_res=(20, 20),
                            visual_action_mode="absolute",
                            visual_action_space=(-10.0, 10.0),
                            resize_to_full=False,
                            record=True,
                            from_pixels=True
                        )
    foveal_peripheral_env = DMCFixedFovealPeripheralEnv(env_args)
    foveal_peripheral_env.reset()
    done = False
    while not done:
        obs, reward, done, _, _ = foveal_peripheral_env.step({
                                "physical_action": foveal_peripheral_env.action_space["physical_action"].sample(), 
                                "visual_action": np.array((random.randint(0,10), random.randint(10,50)))
                                })
        print (obs.shape, foveal_peripheral_env.fov_loc)
        # print (obs)
        obs_image = Image.fromarray((obs*255).astype(np.uint8)[-1])
        obs_image.save("test_env_obs.png")