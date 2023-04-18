from setuptools import find_packages, setup

setup(
    name='active_gym',
    version='0.1',
    author='Jinghuan Shang',
    description=('Gym-like wrapper to implement Active Reinforcement Learning environments based on existing environments'),
    keywords='active-rl rl gym atari deepmind',
    packages=find_packages(),
    install_requires=[
        "mujoco",
        "mujoco-py>=2.1.2.14"
        "atari-py"
        "dm-control==1.0.11"
        "gym"
    ],
)