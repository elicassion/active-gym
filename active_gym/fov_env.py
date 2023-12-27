# Copyright (c) Jinghuan Shang.

import copy
from enum import IntEnum
from typing import Tuple, Union

import cv2
import gymnasium as gym
from gymnasium.spaces import Box, Discrete, Dict

import numpy as np
import torch
from torchvision.transforms import Resize

class RecordWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, args):
        super().__init__(env)
        
        # init reward and episode len counter
        self.args = args
        self.cumulative_reward = 0
        self.ep_len = 0
        self.record_buffer = None
        self.prev_record_buffer = None
        self.record = args.record
        # if self.record:
        #     self._reset_record_buffer()

    def _add_info(self, info):
        info["reward"] = self.cumulative_reward
        info["ep_len"] = self.ep_len
        return info
    
    def _reset_record_buffer(self):
        self.prev_record_buffer = copy.deepcopy(self.record_buffer)
        self.record_buffer = {"rgb": [], "state":[] , "action": [], "reward": [], "done": [], 
                              "truncated": [], "info": [], "return_reward": []}

    def reset(self, seed=None, options=None):
        if hasattr(self.args, "env_backend"):
            if self.args.env_backend == "rlbench":
                state, info = self.env.reset()
            else:
                state, info = self.env.reset(seed, options)
        else:
            state, info = self.env.reset(seed, options)
        # state, info = self.env.reset(seed, options)
        self.cumulative_reward = 0
        self.ep_len = 0
        info = self._add_info(info)
        if self.record:
            rgb = self.env.render()
            # print ("_reset: reset record buffer")
            self._reset_record_buffer()
            self._save_transition(state, done=False, info=info, rgb=rgb)
        return state, info

    def step(self, action):
        state, return_reward, done, truncated, info = self.env.step(action)
        # for recording
        self.ep_len += 1
        self.cumulative_reward += info.get("raw_reward", return_reward)
        info = self._add_info(info)
        if self.record:
            rgb = self.env.render()
            self._save_transition(state, action, self.cumulative_reward, done, truncated, info, rgb=rgb, return_reward=return_reward)
        return state, return_reward, done, truncated, info

    
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

    def render(self, **kwargs):
        return self.env.render(**kwargs)

class FixedFovealEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, args):
        super().__init__(env)
        self.fov_size: Tuple[int, int] = args.fov_size
        self.fov_init_loc: Tuple[int, int] = args.fov_init_loc
        assert (np.array(self.fov_size) < np.array(self.obs_size)).all()

        self.sensory_action_mode: str = args.sensory_action_mode # "absolute", "relative"
        if self.sensory_action_mode == "relative":
            self.sensory_action_space = np.array(args.sensory_action_space)
        elif self.sensory_action_mode == "absolute":
            self.sensory_action_space = np.array(self.obs_size) - np.array(self.fov_size)

        self.resize: Resize = Resize(self.env.obs_size) if args.resize_to_full else None

        self.mask_out: bool = args.mask_out

        # set gym.Env attribute
        self.action_space = Dict({
            "motor_action": self.env.action_space,
            "sensory_action": Box(low=self.sensory_action_space[0], 
                                 high=self.sensory_action_space[1], dtype=int),
        })

        if args.mask_out:
            self.observation_space = Box(low=-1., high=1., 
                                         shape=(self.env.frame_stack,)+self.env.obs_size, 
                                         dtype=np.float32)
        elif args.resize_to_full:
            self.observation_space = Box(low=-1., high=1., 
                                         shape=(self.env.frame_stack,)+self.env.obs_size, 
                                         dtype=np.float32)
        else:
            self.observation_space = Box(low=-1., high=1., 
                                         shape=(self.env.frame_stack,)+self.fov_size, 
                                         dtype=np.float32)

        # init fov location
        # The location of the upper left corner of the fov image, on the original observation plane
        self.fov_loc: np.ndarray = np.empty_like(self.fov_init_loc)  # 
        self._init_fov_loc()

    def _init_fov_loc(self):
        self.fov_loc = np.rint(np.array(self.fov_init_loc, copy=True)).astype(np.int32)

    def reset_record_buffer(self):
        self.env.record_buffer["fov_size"] = self.fov_size
        self.env.record_buffer["fov_loc"] = []
    
    def reset(self):
        full_state, info = self.env.reset()
        self._init_fov_loc()
        fov_state = self._get_fov_state(full_state)
        info["fov_loc"] = self.fov_loc.copy()
        if self.env.record:
            self.reset_record_buffer()
            self.save_transition(info["fov_loc"])
        return fov_state, info

    def _clip_to_valid_fov(self, loc):
        return np.rint(np.clip(loc, 0, np.array(self.env.obs_size) - np.array(self.fov_size))).astype(int)

    def _clip_to_valid_sensory_action_space(self, action):
        return np.rint(np.clip(action, *self.sensory_action_space)).astype(int)
    
    def _get_fov_state(self, full_state):
        fov_state = full_state[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                                    self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]]

        if self.mask_out:
            mask = np.zeros_like(full_state)
            mask[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                    self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]] = fov_state
            fov_state = mask
        elif self.resize:
            fov_state = self.resize(torch.from_numpy(fov_state))
            fov_state = fov_state.numpy()

        return fov_state

    def _fov_step(self, full_state, action):
        if type(action) is torch.Tensor:
            action = action.detach().cpu().numpy()
        elif type(action) is Tuple:
            action = np.array(action)

        if self.sensory_action_mode == "absolute":
            action = self._clip_to_valid_fov(action)
            self.fov_loc = action
        elif self.sensory_action_mode == "relative":
            action = self._clip_to_valid_sensory_action_space(action)
            fov_loc = self.fov_loc + action
            self.fov_loc = self._clip_to_valid_fov(fov_loc)

        fov_state = self._get_fov_state(full_state)
        
        return fov_state

    def save_transition(self, fov_loc):
        # print ("saving one transition")
        self.env.record_buffer["fov_loc"].append(fov_loc)

    def step(self, action):
        """
        action : {"motor_action":
                  "sensory_action": }
        """
        # print ("in env", action, action["motor_action"], action["sensory_action"])
        state, reward, done, truncated, info = self.env.step(action=action["motor_action"])
        fov_state = self._fov_step(full_state=state, action=action["sensory_action"])
        info["fov_loc"] = self.fov_loc.copy()
        if self.record:
            if not done:
                self.save_transition(info["fov_loc"])
        return fov_state, reward, done, truncated, info
    
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

class FlexibleFovealEnvActionType(IntEnum):
    FOV_LOC = 0
    FOV_RES = 1

class FlexibleFovealEnv(FixedFovealEnv):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.action_space["sensory_action_type"] = Discrete(len(FlexibleFovealEnvActionType))
        self.fov_init_res: Tuple[int, int] = args.fov_size
        self.fov_res: np.ndarray = np.empty_like(self.fov_init_res)
        self._init_fov_res()

        self.interpolate_resize_to = Resize(args.fov_size)

    def _init_fov_res(self):
        self.fov_res = np.rint(np.array(self.fov_init_res, copy=True)).astype(np.int32)

    def reset_record_buffer(self):
        self.record_buffer["fov_size"] = self.fov_size
        self.record_buffer["fov_loc"] = []
        self.record_buffer["fov_res"] = []
    
    def reset(self):
        full_state, info = self.env.reset()
        self._init_fov_loc()
        self._init_fov_res()
        fov_state = self._get_fov_state(full_state)
        info["fov_loc"] = self.fov_loc.copy()
        info["fov_res"] = self.fov_res.copy()
        if self.record:
            self.reset_record_buffer()
            self.save_transition(info["fov_loc"], info["fov_res"])
        return fov_state, info

    def _clip_to_valid_fov(self, loc):
        return np.rint(np.clip(loc, 0, np.array(self.env.obs_size) - np.array(self.fov_res))).astype(int)

    def _clip_to_valid_sensory_action_space(self, action):
        return np.rint(np.clip(action, *self.sensory_action_space)).astype(int)
    
    def _interpolate_to_fov_size(self, s):
        s = self.interpolate_resize_to(torch.from_numpy(s))
        interpolate_resize_back = Resize(tuple(self.fov_res))
        s = interpolate_resize_back(s)
        return s.numpy()

    
    def _get_fov_state(self, full_state):
        fov_state = full_state[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_res[0],
                                    self.fov_loc[1]:self.fov_loc[1]+self.fov_res[1]]
        if self.fov_res[0] > self.fov_size[0]:
            fov_state = self._interpolate_to_fov_size(fov_state)

        if self.mask_out:
            mask = np.zeros_like(full_state)
            mask[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_res[0],
                    self.fov_loc[1]:self.fov_loc[1]+self.fov_res[1]] = fov_state
            fov_state = mask
        elif self.resize:
            fov_state = self.resize(torch.from_numpy(fov_state))
            fov_state = fov_state.numpy()

        return fov_state

    def _fov_step(self, full_state, action, action_type=0):
        if type(action) is torch.Tensor:
            action = action.detach().cpu().numpy()
        elif type(action) is Tuple:
            action = np.array(action)

        if type(action_type) is torch.Tensor:
            action_type = action_type.detach().cpu().item()
        elif type(action_type) is Tuple:
            action_type = action_type[0]
        elif type(action_type) is np.ndarray:
            action_type = action_type.tolist()[0]
        action_type = FlexibleFovealEnvActionType(action_type)

        if action_type == FlexibleFovealEnvActionType.FOV_LOC:
            if self.sensory_action_mode == "absolute":
                action = self._clip_to_valid_fov(action)
                self.fov_loc = action
            elif self.sensory_action_mode == "relative":
                action = self._clip_to_valid_sensory_action_space(action)
                fov_loc = self.fov_loc + action
                self.fov_loc = self._clip_to_valid_fov(fov_loc)
        elif action_type == FlexibleFovealEnvActionType.FOV_RES:
            self.fov_res = action.copy()
            self.fov_loc = self._clip_to_valid_fov(self.fov_loc)
        else:
            raise NotImplementedError

        fov_state = self._get_fov_state(full_state)
        
        return fov_state

    def save_transition(self, fov_loc, fov_res):
        # print ("saving one transition")
        self.env.record_buffer["fov_loc"].append(fov_loc)
        self.env.record_buffer["fov_res"].append(fov_res)

    def step(self, action):
        """
        action : {"motor_action":
                  "sensory_action": 
                  "sensory_action_type": }
        action["action_type"]: FoveaFOVAtariEnvActionType.FOV_LOC (0): fov_loc
                               FoveaFOVAtariEnvActionType.FOV_RES (1): fov_res
        """
        # print ("in env", action, action["motor_action"], action["sensory_action"])
        state, reward, done, truncated, info = self.env.step(action=action["motor_action"])
        fov_state = self._fov_step(full_state=state, 
                                   action=action["sensory_action"], 
                                   action_type=action["sensory_action_type"])
        info["fov_loc"] = self.fov_loc.copy()
        info["fov_res"] = self.fov_res.copy()
        if self.record:
            if not done:
                self.save_transition(info["fov_loc"], fov_res=info["fov_res"])
        return fov_state, reward, done, truncated, info


class FixedFovealPeripheralEnv(FixedFovealEnv):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.mask_out = False
        self.resize_to_full = True
        self.observation_space = Box(low=-1., high=1., 
                                     shape=(self.env.frame_stack,)+self.env.obs_size, dtype=np.float32)
        self.peripheral_res: Tuple[int, int] = args.peripheral_res # the resolution applied on the whole background
        self.squeeze_and_expand = torch.nn.Sequential(
                            Resize(self.peripheral_res),
                            Resize(self.env.obs_size))

    def reset_record_buffer(self):
        self.env.record_buffer["fov_size"] = self.fov_size
        self.env.record_buffer["peripheral_res"] = self.peripheral_res
        self.env.record_buffer["fov_loc"] = []
        
    def _squeeze_to_peripheral_size(self, s) -> np.ndarray:
        s = self.squeeze_and_expand(torch.from_numpy(s))
        return s.numpy()
    
    def _get_fov_state(self, full_state) -> np.ndarray:
        fov_state = full_state[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                                    self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]]
        
        peripheral_state = self._squeeze_to_peripheral_size(full_state)

        peripheral_state[..., self.fov_loc[0]:self.fov_loc[0]+self.fov_size[0],
                    self.fov_loc[1]:self.fov_loc[1]+self.fov_size[1]] = fov_state

        return peripheral_state