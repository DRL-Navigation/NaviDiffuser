import random
import yaml
import time
import sys
drl_nav_path = "../../"
sys.path.append(sys.path[0] + "/" + drl_nav_path)

from envs import make_env, read_yaml
from testing.algorithms import DWAPolicy4Nav


if __name__ == "__main__":
    cfg = read_yaml(drl_nav_path + 'envs/cfg/gazebo_cfg/turtlebot3_barndataset.yaml')
    env = make_env(cfg)
    state = env.reset()
    dwa_policy = DWAPolicy4Nav(env.robot_total, cfg)
    while 1:
        state, reward, done, info = env.step(dwa_policy.gen_action(state))
