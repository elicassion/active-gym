from enum import IntEnum
from typing import Tuple, Union

import cv2
import gym
from gym.spaces import Box, Discrete, Dict

import numpy as np
import torch
from torchvision.transforms import Resize

class FixedFovealEnv(gym.Wrapper):
    def __init__(self, env: gym.Env, args):
        super().__init__(env)
        self.fov_size: Tuple[int, int] = args.fov_size
        self.fov_init_loc: Tuple[int, int] = args.fov_init_loc
        assert (np.array(self.fov_size) < np.array(self.obs_size)).all()

        self.visual_action_mode: str = args.visual_action_mode # "absolute", "relative"
        if self.visual_action_mode == "relative":
            self.visual_action_space = np.array(args.visual_action_space)
        elif self.visual_action_mode == "absolute":
            self.visual_action_space = np.array(self.obs_size) - np.array(self.fov_size)

        self.resize: Resize = Resize(self.env.obs_size) if args.resize_to_full else None

        self.mask_out: bool = args.mask_out

        # set gym.Env attribute
        self.action_space = Dict({
            "physical_action": self.env.action_space,
            "visual_action": Box(low=self.visual_action_space[0], 
                                 high=self.visual_action_space[1], dtype=int),
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
        full_state, info = self.env._reset()
        self._init_fov_loc()
        fov_state = self._get_fov_state(full_state)
        info["fov_loc"] = self.fov_loc.copy()
        if self.env.record:
            self.reset_record_buffer()
            self.save_transition(info["fov_loc"])
        return fov_state, info

    def _clip_to_valid_fov(self, loc):
        return np.rint(np.clip(loc, 0, np.array(self.env.obs_size) - np.array(self.fov_size))).astype(int)

    def _clip_to_valid_visual_action_space(self, action):
        return np.rint(np.clip(action, *self.visual_action_space)).astype(int)
    
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

        if self.visual_action_mode == "absolute":
            action = self._clip_to_valid_fov(action)
            self.fov_loc = action
        elif self.visual_action_mode == "relative":
            action = self._clip_to_valid_visual_action_space(action)
            fov_loc = self.fov_loc + action
            self.fov_loc = self._clip_to_valid_fov(fov_loc)

        fov_state = self._get_fov_state(full_state)
        
        return fov_state

    def save_transition(self, fov_loc):
        # print ("saving one transition")
        self.env.record_buffer["fov_loc"].append(fov_loc)

    def step(self, action):
        """
        action : {"physical_action":
                  "visual_action": }
        """
        # print ("in env", action, action["physical_action"], action["visual_action"])
        state, reward, done, truncated, info = self.env._step(action=action["physical_action"])
        fov_state = self._fov_step(full_state=state, action=action["visual_action"])
        info["fov_loc"] = self.fov_loc.copy()
        if self.record:
            if not done:
                self.save_transition(info["fov_loc"])
        return fov_state, reward, done, truncated, info

class FlexibleFovealEnvActionType(IntEnum):
    FOV_LOC = 0
    FOV_RES = 1

class FlexibleFovealEnv(FixedFovealEnv):
    def __init__(self, env, args):
        super().__init__(env, args)
        self.action_space["visual_action_type"] = Discrete(len(FlexibleFovealEnvActionType))
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
        full_state, info = self.env._reset()
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

    def _clip_to_valid_visual_action_space(self, action):
        return np.rint(np.clip(action, *self.visual_action_space)).astype(int)
    
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
            if self.visual_action_mode == "absolute":
                action = self._clip_to_valid_fov(action)
                self.fov_loc = action
            elif self.visual_action_mode == "relative":
                action = self._clip_to_valid_visual_action_space(action)
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
        action : {"physical_action":
                  "visual_action": 
                  "visual_action_type": }
        action["action_type"]: FoveaFOVAtariEnvActionType.FOV_LOC (0): fov_loc
                               FoveaFOVAtariEnvActionType.FOV_RES (1): fov_res
        """
        # print ("in env", action, action["physical_action"], action["visual_action"])
        state, reward, done, truncated, info = self.env._step(action=action["physical_action"])
        fov_state = self._fov_step(full_state=state, 
                                   action=action["visual_action"], 
                                   action_type=action["visual_action_type"])
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