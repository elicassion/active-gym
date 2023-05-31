# Active-Gym

<img src="./_doc/media/boxing_merge.gif">

Active-Gym is a collection of environments for **Active Reinforcement Learning**. It builds on top of existing common visual RL environments including [Atari](https://github.com/openai/atari-py), [DeepMind Control suite](https://github.com/deepmind/dm_control), and [Robosuite](https://github.com/ARISE-Initiative/robosuite). All environments are compatible with [Gymasium](https://github.com/Farama-Foundation/Gymnasium) interface.

What is Active-RL?
It is a process that allows an embodied agent to actively gather new information in an environment. This library particularly focus on actively managing its visual observation, for example an image crop in a 2D environment or a viewpoint in a 3D environment. The agent takes motor actions (like manipulations, physical interactions) and sensory actions (like eye movements) together to finish a task.


## Installation

`pip install git+https://github.com/elicassion/active-gym.git`

or clone and install it in editable mode

```
git clone https://github.com/elicassion/active-gym.git
cd active-gym
pip install -e .
```

If you are interested in using robot environments (like Robosuite), you can do
```
pip install git+https://github.com/elicassion/active-gym.git[robosuite]
```
or follow the installation of [Robosuite](https://github.com/ARISE-Initiative/robosuite)

## Quick Start
```
from active_gym import AtariBaseEnv, AtariEnvArgs
env_args = AtariEnvArgs(
    game=env_name, 
    seed=seed, 
    obs_size=(84, 84), 
    fov_size=(30, 30),              # the partial observation size
    fov_init_loc=(0, 0),            # the initial partial observation location
    visual_action_mode="absolute",  # change the observation location by abs coordinates
    record=False,                   # it integrates recording, if needed
)
env = AtariFixedFovealEnv(env_args)
obs, info = env.reset()
for i in range(5):
    obs, reward, done, truncated, info = env.step(env.action_space.sample())
    # action is in the form of gymnasium.spaces.Dict
    # {"motor_action": np.ndarray, "sensory_action": np.ndarray}
```
More examples can be found at `if __name__ == "__main__"` part in `atari_env.py`, `dmc_env.py`, and `robosuite_env.py`.

## Supported Types of Environments
- Atari
```
AtariBaseEnv                        # Base environment with full observation
AtariFixedFovealEnv                 # Active environment with limited foveal visual observation, at fixed size
AtariFlexibleFovealEnv              # Active environment with options to change foveal observation size
AtariFixedFovealPeripheralEnv       # Active environment with both foveal and peripheral observations
```

- DMC
```
DMCBaseEnv
DMCFixedFovealEnv
DMCFlexibleFovealEnv
DMCFixedFovealPeripheralEnv
```

- Robosuite
```
# All robosuite environments are supported by adding a movable camera.
make_active_robosuite_env(task="<robosuite_env_name>", <more_args>)
```

## TODO List
All suggestions on bug-fix, new features, and improvmets are welcomed. Please open an issue or contact jishang\[at\]cs.stonybrook.edu
- [ ] Detailed documentation
- [ ] (Long term) more environments

## Citation
Please consider cite us if you find this repo helpful.

## Acknowledgement
We thank the implementation of the base environments used in this repo: [Atari](https://github.com/openai/atari-py), [DeepMind Control suite](https://github.com/deepmind/dm_control), and [Robosuite](https://github.com/ARISE-Initiative/robosuite).
