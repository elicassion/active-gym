# Copyright (c) Jinghuan Shang.

import random

from collections import deque
from PIL import Image
from typing import Tuple

import cv2
import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import numpy as np

import atari_py

from .fov_env import (
    RecordWrapper,
    FixedFovealEnv, 
    FlexibleFovealEnv, 
    FixedFovealPeripheralEnv,
    FlexibleFovealEnvActionType
)

class AtariEnvArgs:
    def __init__(self, game, seed, obs_size: Tuple[int, int], **kwargs):
        self.device = None
        self.seed = seed
        self.max_episode_length = 108e3
        self.game = game
        self.frame_stack = 4
        self.action_repeat = 4
        self.obs_size = obs_size
        self.mask_out = False
        self.record = False
        self.clip_reward = False
        for k, v in kwargs.items():
            self.__setattr__(k, v)

class AtariEnv(gym.Env):
    def __init__(self, args):
        self.seed_num = args.seed
        self.ale = atari_py.ALEInterface()
        self.ale.setInt('random_seed', args.seed)
        self.ale.setInt('max_num_frames_per_episode', args.max_episode_length)
        self.ale.setFloat('repeat_action_probability', 0)  # Disable sticky actions
        self.ale.setInt('frame_skip', 0)
        self.ale.setBool('color_averaging', False)
        self.ale.loadROM(atari_py.get_game_path(args.game))  # ROM loading must be done after setting options
        actions = self.ale.getMinimalActionSet()
        self.actions = dict([i, e] for i, e in zip(range(len(actions)), actions))
        self.lives = 0  # Life counter (used in DeepMind training)
        self.life_termination = False  # Used to check if resetting only from loss of life
        self.frame_stack = args.frame_stack # Number of frames to concatenate
        self.action_repeat = args.action_repeat
        self.state_buffer = deque([], maxlen=args.frame_stack)
        self.training = True  # Consistent with model training mode
        self.obs_size = args.obs_size
        self.clip_reward = args.clip_reward

        self.metadata = {"render_modes": []}
        # define render_mode if your environment supports rendering
        self.render_mode = None
        self.reward_range = (-float("inf"), float("inf"))
        self.spec = None

        # Set attributes for gym.Env
        self.action_space = Discrete(len(actions))
        self.observation_space = Box(low=-1., high=1., shape=(self.frame_stack,)+self.obs_size, dtype=np.float32)
        

    def _get_state(self):
        state = cv2.resize(self.ale.getScreenGrayscale(), self.obs_size, interpolation=cv2.INTER_LINEAR)
        return state.astype(np.float32) / 255.

    def _get_info(self, raw_reward=0):
        return {"raw_reward": raw_reward}

    def _reset_buffer(self):
        for _ in range(self.frame_stack):
            self.state_buffer.append(np.zeros(self.obs_size))

    def _reset(self):
        # episodic life
        if self.life_termination:
            self.life_termination = False  # Reset flag
            self.ale.act(0)  # Use a no-op after loss of life
        else:
            # reset internals
            self._reset_buffer()
            self.ale.reset_game()

            # no-op reset
            # perform up to 30 random no-ops before starting
            for _ in range(random.randrange(30)):
                self.ale.act(0)  # Assumes raw action 0 is always no-op
                if self.ale.game_over():
                    self.ale.reset_game()

        # fire reset
        if len(self.actions) >= 3:
            self.ale.act(1)
            if self.ale.game_over():
                self.ale.reset_game()
                self.ale.act(2)
            if self.ale.game_over():
                self.ale.reset_game()

        # Process and return "initial" state
        observation = self._get_state()
        self.state_buffer.append(observation)
        self.lives = self.ale.lives()
        state = np.stack(self.state_buffer, axis=0)
        info = self._get_info()

        return state, info

    def _step(self, action):
        # Repeat action 4 times, max pool over last 2 frames
        frame_buffer = np.zeros((2, *self.obs_size))
        reward, done = 0, False
        for t in range(self.action_repeat):
            reward += self.ale.act(self.actions.get(action))
            if t == 2:
                frame_buffer[0] = self._get_state()
            elif t == 3:
                frame_buffer[1] = self._get_state()
            done = self.ale.game_over()
            if done:
                break
        observation = frame_buffer.max(0)
        self.state_buffer.append(observation)
        # Detect loss of life as terminal in training mode
        if self.training:
            lives = self.ale.lives()
            if lives < self.lives and lives > 0:  # Lives > 0 for Q*bert
                self.life_termination = not done  # Only set flag when not truly done
                done = True
            self.lives = lives

        # Return state, reward, done
        state = np.stack(self.state_buffer, axis=0)
        return_reward = np.sign(reward) if self.clip_reward else reward
        truncated = False
        info = self._get_info(raw_reward=reward)
        
        return state, return_reward, done, truncated, info

    def reset(self):
        state, info = self._reset()
        return state, info

    def step(self, action):
        return self._step(action)

    # Uses loss of life as terminal signal
    def train(self):
        self.training = True

    # Uses standard terminal signal
    def eval(self):
        self.training = False

    def render(self, mode='rgb_array', obs_size=None) -> np.ndarray:
        assert mode == 'rgb_array', 'only support rgb_array mode, given %s' % mode
        size = obs_size if obs_size else (256, 256)
        rgb_for_record = cv2.resize(self.ale.getScreenRGB(), size, interpolation=cv2.INTER_LINEAR)
        return rgb_for_record

    def close(self):
        pass

def AtariBaseEnv(args: AtariEnvArgs) -> gym.Wrapper:
    base_env = AtariEnv(args)
    wrapped_env = RecordWrapper(base_env, args)
    return wrapped_env

def AtariFixedFovealEnv(args: AtariEnvArgs) -> gym.Wrapper:
    base_env = AtariBaseEnv(args)
    wrapped_env = FixedFovealEnv(base_env, args)
    return wrapped_env

def AtariFlexibleFovealEnv(args: AtariEnvArgs)-> gym.Wrapper:
    base_env = AtariBaseEnv(args)
    wrapped_env = FlexibleFovealEnv(base_env, args)
    return wrapped_env
    
def AtariFixedFovealPeripheralEnv(args: AtariEnvArgs) -> gym.Wrapper:
    base_env = AtariBaseEnv(args)
    wrapped_env = FixedFovealPeripheralEnv(base_env, args)
    return wrapped_env

# test
if __name__ == "__main__":
    env_args = AtariEnvArgs(game="breakout", seed=42, obs_size=(84, 84), 
                        fov_size=(50, 50),
                        fov_init_loc=(33, 33),
                        visual_action_mode="relative",
                        visual_action_space=(-10.0, 10.0),
                        resize_to_full=True,
                        )
    ori_env = AtariBaseEnv(env_args)
    
    ori_env.reset()
    for i in range(5):
        obs, reward, done, _, _ = ori_env.step(ori_env.action_space.sample())
        print (obs.shape)

    # ---------

    print ("test fixed foveal, flexible foveal, foveal peripheral envs")
    print ("fixed foveal")
    env_args = AtariEnvArgs(game="breakout", seed=42, obs_size=(84, 84), 
                        fov_size=(50, 50),
                        fov_init_loc=(0, 0),
                        visual_action_mode="absolute",
                        visual_action_space=(-10.0, 10.0),
                        resize_to_full=False,
                        record=True)
    def make_env(env_name, seed, **kwargs):
        def thunk():
            env_args = AtariEnvArgs(
                game=env_name, seed=seed, obs_size=(84, 84), **kwargs
            )
            env = AtariFixedFovealEnv(env_args)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk
    fov_env = [make_env(env_args.game, env_args.seed, frame_stack=3, action_repeat=1, 
                            fov_size=env_args.fov_size, 
                            fov_init_loc=env_args.fov_init_loc,
                            visual_action_mode=env_args.visual_action_mode,
                            visual_action_space=env_args.visual_action_space,
                            resize_to_full=env_args.resize_to_full,
                            mask_out=True,
                            training=False,
                            record=True)]
    fov_env = gym.vector.SyncVectorEnv(fov_env)
    obs, _ = fov_env.reset()
    # print (fov_env.envs)
    print (fov_env.action_space)
    done = False
    ep_len_counter = 0
    while not done:
        obs, reward, done, _, _ = fov_env.step({
                                "physical_action": fov_env.action_space["physical_action"].sample(), 
                                "visual_action": np.array([[random.randint(0,10), random.randint(10,50)]])})
        ep_len_counter += 1
        print (obs.shape, fov_env.envs[0].fov_loc)
    fov_env.envs[0].save_record_to_file("test_env_record_file.pt")

    print ("flexible foveal")
    def make_env(env_name, seed, **kwargs):
        def thunk():
            env_args = AtariEnvArgs(
                game=env_name, seed=seed, obs_size=(84, 84), **kwargs
            )
            env = AtariFlexibleFovealEnv(env_args)
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            return env

        return thunk
    fovea_fov_env = [make_env(env_args.game, env_args.seed, frame_stack=3, action_repeat=1, 
                            fov_size=env_args.fov_size, 
                            fov_init_loc=env_args.fov_init_loc,
                            visual_action_mode=env_args.visual_action_mode,
                            visual_action_space=env_args.visual_action_space,
                            resize_to_full=env_args.resize_to_full,
                            mask_out=True,
                            training=False,
                            record=True)]
    fovea_fov_env = gym.vector.SyncVectorEnv(fovea_fov_env)
    for i in range (10):
        obs, _ = fovea_fov_env.reset()
        print (fovea_fov_env.envs)
        done = False
        ep_len_counter = 0
        while not done:
            action_type = random.choice(list(FlexibleFovealEnvActionType))
            if action_type == FlexibleFovealEnvActionType.FOV_LOC:
                action = np.array([[random.randint(0,10), random.randint(10,50)]])
            elif action_type == FlexibleFovealEnvActionType.FOV_RES:
                action = np.array([[random.randint(10, 40), random.randint(15, 70)]])
            
            ac = {
                "physical_action": fovea_fov_env.action_space["physical_action"].sample(), 
                "visual_action": action,
                "visual_action_type": np.array((action_type.value,))
            }
            # print (ac)
            obs, reward, done, _, _ = fovea_fov_env.step(ac)
            ep_len_counter += 1
            print (obs.shape, fovea_fov_env.envs[0].fov_loc, fovea_fov_env.envs[0].fov_res)
    fovea_fov_env.envs[0].save_record_to_file("test_env_record_file.pt")
    

    print ("foveal peripheral")
    env_args = AtariEnvArgs(game="kung_fu_master", seed=42, obs_size=(84, 84), 
                        fov_size=(50, 50),
                        fov_init_loc=(0, 0),
                        peripheral_res=(20, 20),
                        visual_action_mode="absolute",
                        visual_action_space=(-10.0, 10.0),
                        resize_to_full=False,
                        record=True)
    fov_env = AtariFixedFovealPeripheralEnv(env_args)
    fov_env.reset()
    done = False
    ep_len_counter = 0
    while not done:
        obs, reward, done, _, _ = fov_env.step({
                                "physical_action": fov_env.action_space["physical_action"].sample(), 
                                "visual_action": np.array((random.randint(0,10), random.randint(10,50)))})
        ep_len_counter += 1
        print (obs.shape, fov_env.fov_loc)
        # print (obs)
        obs_image = Image.fromarray((obs*255).astype(np.uint8)[-1])
        obs_image.save("test_env_obs.png")
    fov_env.reset()