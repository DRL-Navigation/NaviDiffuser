import torch, numpy, sys, os, math

from env.envs import make_env, read_yaml
from utils import EnveDoubleMoActorAgent
import numpy as np

def generate_w(num_prefence, pref_param, fixed_w=None):
    if fixed_w is not None and num_prefence>1:
        sigmas = torch.Tensor([0.01]*len(pref_param))
        w = torch.distributions.normal.Normal(torch.Tensor(pref_param), sigmas)
        w = w.sample(torch.Size((num_prefence-1,))).numpy()
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence-1, 1)
        return np.concatenate(([fixed_w], w))
    elif fixed_w is not None and num_prefence==1:
        return np.array([fixed_w])
    else:
        sigmas = torch.Tensor([0.01]*len(pref_param))
        w = torch.distributions.normal.Normal(torch.Tensor(pref_param), sigmas)
        w = w.sample(torch.Size((num_prefence,))).numpy()
        w = np.abs(w) / np.linalg.norm(w, ord=1, axis=1).reshape(num_prefence, 1)
        return w
    return w

if __name__ == '__main__':

    output = './output/'
    logname = 'eval.log'
    if not os.path.exists(output): os.makedirs(output)
    sys.stdout = open(output+logname, 'w+')

    cfg = read_yaml('./cfg/BARN.yaml')
    env = make_env(cfg)
    loadpath = './model/70.pt'

    # get enviroment information
    input_size = cfg['Agent']['input_size']
    output_size = cfg['Agent']['output_size']
    reward_size = cfg['Agent']['reward_size']

    action_list = cfg['discrete_actions']

    agent = EnveDoubleMoActorAgent(**cfg['Agent'])

    agent.model.load_state_dict(torch.load(loadpath))

    # set agent model as eval since we won't update its weights.
    agent.model.eval()

    states = []
    state = env.reset()
    laser = state[0].reshape(1, -1)
    vector = state[1]
    state_formed = np.concatenate((vector, laser), axis=1)
    states.append(state_formed)
    states = np.stack(states)
    
    pref_param = np.array([4, 2, 0., 0., 1., 1.])

    explore_w = generate_w(1, pref_param)

    while True:

        while True:
            new_states = []
            action = agent.get_action(states, explore_w)
            state, reward, done, info = env.step([action_list[action[0]]])

            if info['all_down'][0]: break
            laser = state[0].reshape(1, -1)
            vector = state[1]
            state_formed = np.concatenate((vector, laser), axis=1)
            new_states.append(state_formed)
            states = np.stack(new_states)

