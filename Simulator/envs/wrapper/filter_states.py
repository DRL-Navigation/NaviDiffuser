import gym

from envs.state import ImageState


class ObsStateTmp(gym.ObservationWrapper):
    def __init__(self, env, cfg):
        super(ObsStateTmp, self).__init__(env)

    def observation(self, states: ImageState):

        return [states.sensor_maps, states.vector_states, states.ped_maps]


class ObsLaserStateTmp(gym.Wrapper):
    def __init__(self, env, cfg):
        super(ObsLaserStateTmp, self).__init__(env)

    def observation(self, states: ImageState):
        return [states.lasers, states.vector_states, states.ped_maps]
    
    def reset(self, **kwargs):
        return self.observation(self.env.reset(**kwargs))

    def step(self, action, **kwargs):
        states, reward, done, info = self.env.step(action, **kwargs)
        return self.observation(states), reward, done, info

