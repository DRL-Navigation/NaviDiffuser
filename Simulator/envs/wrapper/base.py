import torch
import gym
import numpy as np
import math
import yaml
import time
import sys
import pickle, pymongo, random

import rospy
from std_msgs.msg import Float64MultiArray, MultiArrayDimension

from typing import *
from collections import deque
from copy import deepcopy


from envs.state import ImageState
from envs.action import *
from envs.utils import BagRecorder


class StatePedVectorWrapper(gym.ObservationWrapper):
    avg = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0])
    std = np.array([6.0, 6.0, 0.6, 0.9, 0.50, 0.5, 6.0])

    def __init__(self, env, cfg=None):
        super(StatePedVectorWrapper, self).__init__(env)

    def observation(self, state: ImageState):
        self._normalize_ped_state(state.ped_vector_states)
        return state

    def _normalize_ped_state(self, peds):

        for robot_i_peds in peds:
            for j in range(int(robot_i_peds[0])): # j: ped index
                robot_i_peds[1 + j * 7:1 + (j + 1) * 7] = (robot_i_peds[1 + j * 7:1 + (j + 1) * 7] - self.avg) / self.std


class VelActionWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(VelActionWrapper, self).__init__(env)
        if cfg['discrete_action']:
            self.actions: DiscreteActions = DiscreteActions(cfg['discrete_actions'])

            self.f = lambda x: self.actions[int(x)] if np.isscalar(x) else ContinuousAction(*x)
        else:
            clip_range = cfg['continuous_actions']

            def tmp_f(x):
                y = []
                for i in range(len(x)):
                    y.append(np.clip(x[i], clip_range[i][0], clip_range[i][1]))
                return ContinuousAction(*y)
            # self.f = lambda x: ContinuousAction(*x)
            self.f = tmp_f

    def step(self, action: np.ndarray, **kwargs):
        action = self.action(action)
        state, reward, done, info = self.env.step(action, **kwargs)
        info['speeds'] = np.array([a.reverse()[:2] for a in action])
        return state, reward, done, info

    def action(self, actions: np.ndarray) -> List[ContinuousAction]:
        return list(map(self.f, actions))

    def reverse_action(self, actions):

        return actions


class MultiRobotCleanWrapper(gym.Wrapper):
    """
    有一些robot撞了之后，动作可以继续前向，但是送给网络的数据要过滤掉。
    """
    is_clean : list

    def __init__(self, env, cfg):
        super(MultiRobotCleanWrapper, self).__init__(env)
        self.is_clean = np.array([True] * cfg['agent_num_per_env'])

    def step(self, action, **kwargs):
        state, reward, done, info = self.env.step(action, **kwargs)
        info['is_clean'] = deepcopy(self.is_clean)
        reward[~info['is_clean']] = 0
        info['speeds'][~info['is_clean']] = np.zeros(2)
        # for i in range(len(done)):
        #     if done[i]:
        #         self.is_clean[i]=False
        self.is_clean = np.where(done>0, False, self.is_clean)
        return state, reward, done, info

    def reset(self, **kwargs):
        state = self.env.reset(**kwargs)
        self.is_clean = np.array([True] * len(self.is_clean))
        return state



class StateBatchWrapper(gym.Wrapper):
    batch_dict: Dict

    def __init__(self, env, cfg):
        print(cfg,flush=True)
        super(StateBatchWrapper, self).__init__(env)
        self.q_sensor_maps = deque([], maxlen=cfg['image_batch']) if cfg['image_batch']>0 else None
        self.q_vector_states = deque([], maxlen=cfg['state_batch']) if cfg['state_batch']>0 else None
        self.q_lasers = deque([], maxlen=max(cfg['laser_batch'], 1) ) if cfg['laser_batch']>= 0 else None
        self.batch_dict = {
            "sensor_maps": self.q_sensor_maps,
            "vector_states": self.q_vector_states,
            "lasers": self.q_lasers,
        }

    def step(self, action, **kwargs):
        state, reward, done, info = self.env.step(action, **kwargs)
        return self.batch_state(state), reward, done, info

    def _concate(self, b: str, t: np.ndarray):
        q = self.batch_dict[b]
        if q is None:
            return t
        else:
            t = np.expand_dims(t, axis=1)
        # start situation
        while len(q) < q.maxlen:
            q.append(np.zeros_like(t))
        q.append(t)
        #  [n(Robot), k(batch), 84, 84]
        return np.concatenate(list(q), axis=1)

    def batch_state(self, state):
        # TODO transpose. print
        state.sensor_maps = self._concate("sensor_maps", state.sensor_maps)
        # print('sensor_maps shape; ', state.sensor_maps.shape)

        # [n(robot), k(batch), state_dim] -> [n(robot), k(batch) * state_dim]
        tmp_ = self._concate("vector_states", state.vector_states)
        state.vector_states = tmp_.reshape(tmp_.shape[0], tmp_.shape[1] * tmp_.shape[2])
        # print("vector_states shape", state.vector_states.shape)
        state.lasers = self._concate("lasers", state.lasers)
        # print("lasers shape:", state.lasers.shape)
        return state

    def _clear_queue(self):
        for q in self.batch_dict.values():
            if q is not None:
                q.clear()

    def reset(self, **kwargs):
        self._clear_queue()
        state = self.env.reset(**kwargs)
        return self.batch_state(state)


class SensorsPaperRewardWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(SensorsPaperRewardWrapper, self).__init__(env)

        self.robot_r = cfg['robot_radius']
        self.laser_norm = cfg['laser_norm']
        self.laser_max = cfg['laser_max']
        self.control_dt = cfg['control_hz']
        self.min_dist = []
        self.last_obs_dist = []
        self.last_v = []
        self.last_a = []

    def reset(self, **kwargs):
        states = self.env.reset(**kwargs)
        self.min_dist = [math.sqrt(vs[0]**2+vs[1]**2) for vs in states.vector_states]
        self.last_obs_dist = [min(list(laser.reshape(-1))) * (self.laser_max if self.laser_norm else 1.0) for laser in states.lasers]
        self.last_v = [[0, 0] for _ in states.vector_states]
        self.last_a = [[0, 0] for _ in states.vector_states]
        return states

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        return states, self.reward(reward, states), done, info

    def _each_r(self, states: ImageState, index: int):
        closer_reward = obs_reward = jerk_reward = reach_reward = coll_reward = stop_reward = 0.0
        closer_reward_factor = 160.0
        obs_reward_factor = -80.0
        jerk_reward_factor = -10.0 / (16.0 * 16.0)
        closer_bias = 0.0
        obs_bias = 5.0
        jerk_bias = 10.0

        min_dist = self.min_dist[index]
        last_obs_dist = self.last_obs_dist[index]
        vector_state = states.vector_states[index]
        dist_to_goal = math.sqrt(vector_state[0] ** 2 + vector_state[1] ** 2)
        dist_to_obs = min(list(states.lasers[index].reshape(-1))) * (self.laser_max if self.laser_norm else 1.0)
        is_collision = states.is_collisions[index]
        is_arrive = states.is_arrives[index]
        last_v, last_a = self.last_v[index], self.last_a[index]
        v = [vector_state[-2]*math.cos(vector_state[2]), -vector_state[-2]*math.sin(vector_state[2])]
        a = [(v[0]-last_v[0])/self.control_dt, (v[1]-last_v[1])/self.control_dt]
        j = [(a[0]-last_a[0])/self.control_dt, (a[1]-last_a[1])/self.control_dt]

        if is_collision > 0:
            coll_reward = -1000
        elif dist_to_obs <= self.robot_r:
            coll_reward = -500
        if is_arrive:
            reach_reward = 1000
        if vector_state[-2] <= 0.04:
            stop_reward = -10

        dist_to_obs = max(dist_to_obs, 0.1)
        obs_reward = obs_reward_factor / dist_to_obs * (last_obs_dist - dist_to_obs)**2
        obs_reward = max(obs_reward, -5)
        if dist_to_obs >= last_obs_dist:
            obs_reward = -obs_reward
        self.last_obs_dist[index] = dist_to_obs

        if dist_to_goal < min_dist:
            closer_reward = closer_reward_factor * (min_dist - dist_to_goal)**2
            closer_reward = min(closer_reward, 10)
            self.min_dist[index] = dist_to_goal
        else:
            closer_reward = 0.0

        jerk_reward = jerk_reward_factor * ((j[0])**2+(j[1])**2)
        jerk_reward = -10.0 if jerk_reward < -10.0 else jerk_reward
        self.last_v[index], self.last_a[index] = v, a

        closer_reward = closer_reward + closer_bias
        obs_reward = obs_reward + obs_bias
        jerk_reward = jerk_reward + jerk_bias
            
        return [closer_reward, obs_reward, jerk_reward, stop_reward, reach_reward, coll_reward]

    def reward(self, reward, states):
        rewards = []
        for i in range(len(states)):
            rewards.append(self._each_r(states, i))

        return np.array(rewards)

class NeverStopWrapper(gym.Wrapper):
    """
        NOTE !!!!!!!!!!!
        put this in last wrapper.
    """
    def __init__(self, env, cfg):
        super(NeverStopWrapper, self).__init__(env)

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        if info['all_down'][0]:
            states = self.env.reset(**info)

        return states, reward, done, info


# time limit
class TimeLimitWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(TimeLimitWrapper, self).__init__(env)
        self._max_episode_steps = cfg['time_max']
        robot_total = cfg['robot']['total']
        self._elapsed_steps = np.zeros(robot_total, dtype=np.uint8)

    def step(self, ac, **kwargs):
        observation, reward, done, info = self.env.step(ac, **kwargs)
        self._elapsed_steps += 1
        done = np.where(self._elapsed_steps > self._max_episode_steps, 1, done)
        info['dones_info'] = np.where(self._elapsed_steps > self._max_episode_steps, 10, info['dones_info'])
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)###


class InfoLogWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(InfoLogWrapper, self).__init__(env)
        self.robot_total = cfg['robot']['total']
        self.tmp = np.zeros(self.robot_total, dtype=np.uint8)
        self.ped: bool = cfg['ped_sim']['total'] > 0 and cfg['env_type'] == 'robot_nav'

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        info['arrive'] = states.is_arrives
        info['collision'] = states.is_collisions

        info['dones_info'] = np.where(states.is_collisions > 0, states.is_collisions, info['dones_info'])
        info['dones_info'] = np.where(states.is_arrives == 1, 5, info['dones_info'])
        info['all_down'] = self.tmp + sum(np.where(done>0, 1, 0)) == len(done)

        if self.ped:
            # when robot get close to human, record their speeds.
            info['bool_get_close_to_human'] = np.where(states.ped_min_dists < 1 , 1 , 0)

        return states, reward, done, info


class BagRecordWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(BagRecordWrapper, self).__init__(env)
        self.reward_res = []
        self.reward_res_pub = rospy.Publisher("/" + cfg['env_name'] + str(cfg['node_id']) + "/reward_res", Float64MultiArray, queue_size=1)

        self.bag_recorder = BagRecorder(cfg["bag_record_output_name"])
        self.record_epochs = int(cfg['bag_record_epochs'])
        self.episode_res_topic = "/" + cfg['env_name'] + str(cfg['node_id']) + "/episode_res"
        self.episode_res_topic += " " + "/" + cfg['env_name'] + str(cfg['node_id']) + "/reward_res"
        print("epi_res_topic", self.episode_res_topic, flush=True)
        self.cur_record_epoch = 0

        self.bag_recorder.record(self.episode_res_topic)

    def _trans2string(self, dones_info):
        o: List[str] = []
        for int_done in dones_info:
            if int_done == 10:
                o.append("stuck")
            elif int_done == 5:
                o.append("arrive")
            elif 0 < int_done < 4:
                o.append("collision")
            else:
                raise ValueError
        print(o, flush=True)
        return o

    def reset(self, **kwargs):
        if self.cur_record_epoch == self.record_epochs:
            time.sleep(10)
            self.bag_recorder.stop()
        if kwargs.get('dones_info') is not None: # first reset not need
            self.reward_res = np.array(self.reward_res)
            msg = Float64MultiArray()
            msg.layout.dim = [MultiArrayDimension(), MultiArrayDimension(), MultiArrayDimension()]
            msg.layout.dim[0].label = "seq"
            msg.layout.dim[0].size = self.reward_res.shape[0]
            msg.layout.dim[0].stride = self.reward_res.shape[0]*self.reward_res.shape[1]*self.reward_res.shape[2]
            msg.layout.dim[1].label = "robot"
            msg.layout.dim[1].size = self.reward_res.shape[1]
            msg.layout.dim[1].stride = self.reward_res.shape[1]*self.reward_res.shape[2]
            msg.layout.dim[2].label = "reward"
            msg.layout.dim[2].size = self.reward_res.shape[2]
            msg.layout.dim[2].stride = self.reward_res.shape[2]
            msg.data = self.reward_res.flatten().tolist()
            self.reward_res_pub.publish(msg)
            self.reward_res = []

            self.env.end_ep(self._trans2string(kwargs['dones_info']))
            self.cur_record_epoch += 1
        """
                done info:
                10: timeout
                5:arrive
                1: get collision with static obstacle
                2: get collision with ped
                3: get collision with other robot
                """
        print(self.cur_record_epoch, flush=True)
        return self.env.reset(**kwargs)
    
    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        self.reward_res.append(reward)
        return states, reward, done, info
    

class TimeControlWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(TimeControlWrapper, self).__init__(env)
        self.dt = cfg['control_hz']

    def step(self, action, **kwargs):
        start_time = time.time()
        states, reward, done, info = self.env.step(action, **kwargs)
        while time.time() - start_time < self.dt:
            time.sleep(0.02)
        return states, reward, done, info

class RewardSumWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(RewardSumWrapper, self).__init__(env)

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        reward = (reward - np.array([0, 5, 10, 0, 0, 0])) * np.array([4, 2, 0, 0, 1, 1])
        reward = np.sum(reward, axis=-1, keepdims=False)
        return states, reward, done, info
    
class ExpCollectWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(ExpCollectWrapper, self).__init__(env)
        self.node_id = cfg['node_id']
        self.client = pymongo.MongoClient('mongodb://localhost:27017/')
        self.database = self.client['DataBase']
        self.collection = self.database['Collection']
        self.num = 0
        self.sum = cfg['exp_sum']
        self.exp_id_bias = cfg['exp_id_bias']
        self.exp = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        self.finish = False

    @classmethod
    def __decode_data(cls, bytes_data: bytes) -> List[np.ndarray]:
        return pickle.loads(bytes_data)

    @classmethod
    def __encode_data(cls, np_list_data: List[np.ndarray]) -> bytes:
        return pickle.dumps(np_list_data)

    def reset(self, **kwargs):
        if self.finish == False:
            if self.sum > self.num:
                if len(self.exp['action']) > 0:
                    self.exp['action'].append(np.zeros_like(self.exp['action'][-1]))
                    self.exp['reward'].append(np.zeros_like(self.exp['reward'][-1]))
                    self.collection.insert_one({'_id' : int(self.node_id*self.sum+self.num+self.exp_id_bias),
                                                'laser' : ExpCollectWrapper.__encode_data(self.exp['laser']),
                                                'vector' : ExpCollectWrapper.__encode_data(self.exp['vector']),
                                                'action' : ExpCollectWrapper.__encode_data(self.exp['action']),
                                                'reward' : ExpCollectWrapper.__encode_data(self.exp['reward'])
                                                })
                    self.num += 1
                    if self.num % 10000 == 0:
                        print("[{}]Got {}W Exp from Node {}".format(time.strftime('%H:%M:%S',time.localtime(time.time())), int(self.num/10000), self.node_id), flush=True)
            else:
                self.client.close()
                self.database = self.collection = None
                self.finish = True
                print("[{}]Node {} Finish Exp Generate .".format(self.node_id, time.strftime('%H:%M:%S',time.localtime(time.time()))), flush=True)
        states = self.env.reset(**kwargs)
        self.exp = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        self.exp['laser'].append(states.lasers)
        self.exp['vector'].append(states.vector_states)
        return states

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        if self.finish == False:
            self.exp['action'].append(info['feedback_action'])
            self.exp['reward'].append(reward)
            self.exp['laser'].append(states.lasers)
            self.exp['vector'].append(states.vector_states)
        return states, reward, done, info
    
class ActionRandomWrapper(gym.Wrapper):
    def __init__(self, env, cfg):
        super(ActionRandomWrapper, self).__init__(env)
        self.range = cfg["discrete_actions"]
        self.turn_on = cfg["random_action"]
        self.rate = cfg["random_rate"]

    def random_action(self, robot_action):
        random_action = self.range[random.randint(0, len(self.range)-1)]
        return ContinuousAction(*random_action)

    @classmethod
    def unzip_action(cls, vector_states):
        return [vector_states[-2], vector_states[-1]]

    def step(self, action, **kwargs):
        if self.turn_on:
            if random.random() < self.rate:
                action = list(map(self.random_action, action))
        state, reward, done, info = self.env.step(action, **kwargs)
        info['feedback_action'] = np.array(list(map(ActionRandomWrapper.unzip_action, state.vector_states.tolist())))
        return state, reward, done, info