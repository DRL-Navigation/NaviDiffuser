import sys
sys.path.append(sys.path[0] + "/../../../")

import random
import yaml
import time

from envs import make_env, read_yaml
from testing.algorithms import DWAPolicy4Nav



if __name__ == "__main__":
    import sys
    tmp = len(sys.argv)
    if tmp == 2:
        cfg = read_yaml(sys.argv[1])
    else:
        cfg = read_yaml('envs/cfg/test.yaml')
    print(cfg)
    env = make_env(cfg)
    # env2 = make_env(cfg)
    # time.sleep(1)
    dwa_policy = DWAPolicy4Nav(env.robot_total, cfg)
    # test continuous action

    state = env.reset()
    while 1:
        a = time.time()
        state, reward, done, info = env.step(dwa_policy.gen_action(state))
        # env2.step(random_policy.gen_action())
        # print(time.time() - a)
        # time.sleep(0.1)

