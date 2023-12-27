# Copyright (c) Jinghuan Shang.

from .atari_env import (
    AtariBaseEnv,
    AtariFixedFovealEnv,
    AtariFlexibleFovealEnv,
    AtariFixedFovealPeripheralEnv,
    AtariEnvArgs
)
try:
    from .dmc_env import (
        DMCBaseEnv,
        DMCFixedFovealEnv,
        DMCFlexibleFovealEnv,
        DMCFixedFovealPeripheralEnv,
        DMCEnvArgs
    )
except Exception as e:
    print ("DMC loading error. Please check MUJOCO or GL")

try:
    from .robosuite_env import (
        ActiveLift,
        ActiveStack,
        ActiveNutAssembly,
        ActiveNutAssemblySingle,
        ActiveNutAssemblySquare,
        ActiveNutAssemblyRound,
        ActivePickPlace,
        ActivePickPlaceSingle,
        ActivePickPlaceMilk,
        ActivePickPlaceBread,
        ActivePickPlaceCereal,
        ActivePickPlaceCan,
        ActiveDoor,
        ActiveWipe,
        ActiveToolHang,
        ActiveTwoArmLift,
        ActiveTwoArmPegInHole,
        ActiveTwoArmHandover,
        ActiveTwoArmTransport,
        RobosuiteEnvArgs,
        make_active_robosuite_env,
    )
except Exception as e:
    print ("Robosuite loading error. Ignore this if you do not use robosuite.")

try:
    from .rlbench_env import (
        make_active_rlbench_env,
        make_base_rlbench_env,
        register_rlbench_envs
    )
    register_rlbench_envs
except Exception as e:
    print ("RLBench loding error. Ignore this if you do not use RLBench.")

from .fov_env import (
    RecordWrapper,
    FixedFovealEnv,
    FlexibleFovealEnv,
    FlexibleFovealEnvActionType,
    FixedFovealPeripheralEnv
)

