import numpy, sys, os, math

from env.envs import make_env, read_yaml
from dwa import dwa_control
from dwa import Config as DWAConfig


def scan_to_obs(scan, min_angle = -math.pi/2, max_angle = math.pi/2, max_range = 10, filter_scale = 10):
    """
    激光转障碍物list
    :param scan:
    :param min_angle:
    :param max_angle:
    :param max_range:
    """
    obs_list = []
    range_total = len(scan) // filter_scale
    angle_step = (max_angle - min_angle) / range_total
    cur_angle = min_angle
    for i in range(range_total):
        j = i * filter_scale
        if scan[j] <= max_range:
            x = scan[j] * math.cos(cur_angle)
            y = scan[j] * math.sin(cur_angle)
            obs_list.append([x,y])
        cur_angle += angle_step
    return numpy.array(obs_list)


output = './output/'
logname = 'eval.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = read_yaml('./cfg/BARN.yaml')
env = make_env(cfg)
dwa_cfg = DWAConfig()

states = env.reset()
while True:
    x = numpy.array([0.0, 0.0, 0.0, states[1][0][-2], states[1][0][-1]])
    goal = numpy.array([states[1][0][0], states[1][0][1]])
    obs = scan_to_obs(states[0][0][0])
    u, t = dwa_control(x, dwa_cfg, goal, obs)
    t = t[:,:2]
    states, reward, done, info = env.step([u,], path=[t,])