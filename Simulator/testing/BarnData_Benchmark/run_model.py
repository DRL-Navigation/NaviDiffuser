import random
import yaml
import time
from envs import make_env, read_yaml



if __name__ == "__main__":
    cfg = read_yaml('envs/cfg/gazebo_cfg/jackal_clearpath_barndataset.yaml')
    env = make_env(cfg)
    model = MPTModel() # TODO

    state = env.reset()
    print(state)
    while True:
        action = model(state)
        state, reward, done, info = env.step(action)
        print(state)


