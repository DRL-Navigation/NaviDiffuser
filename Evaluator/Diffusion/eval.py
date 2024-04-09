import torch, numpy, sys, os, math, copy

from env.envs import make_env, read_yaml
from nn import NaviDiffusion, ClassifyFreeDDIM


def motion(x, u, dt):
    y = numpy.zeros_like(x)

    y[2] = x[2] + u[1] * dt
    if u[1] != 0:
        y[0] = x[0] + u[0]/u[1] * (math.sin(y[2])-math.sin(x[2]))
        y[1] = x[1] + u[0]/u[1] * (math.cos(x[2])-math.cos(y[2]))
    else:
        y[0] = x[0] + u[0] * math.cos(x[2]) * dt  
        y[1] = x[1] + u[0] * math.sin(x[2]) * dt  
    y[3] = u[0]
    y[4] = u[1]

    return y

def predict_trajectory(action_list, dt):
    x = numpy.zeros(5)
    trajectory = copy.deepcopy(x)
    for i in range(len(action_list)):
        x = motion(x, action_list[i], dt)
        trajectory = numpy.vstack((trajectory, x))

    return trajectory[:,:2]

class ExpHandle:
    def __init__(self, history_max_len, future_max_len, reward_want, reward_label):
        self.history_max_len = history_max_len
        self.future_max_len = future_max_len
        self.reward_want = reward_want
        self.reward_label = reward_label

    def _to_device(self, data, device):
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = self._to_device(data[i], device)
        elif isinstance(data, dict):
            for key in data.keys():
                data[key] = self._to_device(data[key], device)
        elif isinstance(data, torch.Tensor):
            data = data.to(device=device, non_blocking=True)
        else:
            raise ValueError
        return data

    def collate_fn(self, batch, device):
        history = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        history_mask = []
        condition = []
        condition_label = []
        future = []

        for exp in batch:
            elen = len(exp['laser'])
            tlen = min(self.history_max_len+1, elen)
            si = elen-tlen

            for key in history.keys():
                history[key].append(numpy.concatenate(exp[key][si:si+tlen], axis=0))
                history[key][-1] = numpy.concatenate([numpy.zeros((self.history_max_len+1-tlen,)+history[key][-1].shape[1:]), history[key][-1]], axis=0)
                history[key][-1] = history[key][-1].reshape((1,)+history[key][-1].shape)
            history_mask.append(numpy.concatenate([numpy.zeros((1, self.history_max_len+1-tlen)), numpy.ones((1, tlen))], axis=1))

        for key in history.keys():
            history[key] = torch.from_numpy(numpy.concatenate(history[key], axis=0)).to(dtype=torch.float32)
        history_mask = torch.from_numpy(numpy.concatenate(history_mask, axis=0)).to(dtype=torch.bool)
        condition = torch.from_numpy(numpy.array(self.reward_want)*self.future_max_len).unsqueeze(0).repeat(history_mask.shape[0], 1).to(dtype=torch.float32)
        condition_label = torch.from_numpy(numpy.array(self.reward_label)).unsqueeze(0).repeat(history_mask.shape[0], 1).to(dtype=torch.int64)
        future = torch.randn((history_mask.shape[0], self.future_max_len, 2)).to(dtype=torch.float32)

        # cut off useless information
        history['laser']  = history['laser'].squeeze(2)
        history['vector'] = history['vector'][:,:,:3]
        history['reward'] = history['reward'][:,:,:3]
        condition = condition[:,:3]
        condition_label = condition_label[:,:3]
        max_vector = torch.max(torch.abs(history['vector'][:,:,:2]))
        if max_vector >= 5.0:
            history['vector'][:,:,:2] *= (5.0/max_vector)
        history['action'][:,-2,:] *= 0.0

        x = {}
        x['action'] = future
        x['condition'] = condition
        x['condition_label'] = condition_label
        c = history
        c['history_mask'] = history_mask

        return self._to_device([x, c], device)


output = './output/'
logname = 'eval.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = read_yaml('./cfg/BARN.yaml')
env = make_env(cfg)
net = NaviDiffusion(**cfg['Network']).cuda().float()
net.load_state_dict(torch.load('./model/last_model_shared.pt'))
net = ClassifyFreeDDIM(net, **cfg['Diffusion'])
exp_handle = ExpHandle(**cfg['CollateFN'])

states = env.reset()
while True:
    exp = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
    exp['laser'].append(states[0])
    exp['vector'].append(states[1])
    exp['action'].append(numpy.zeros((1, 2)))
    exp['reward'].append(numpy.zeros((1, 6)))
    
    while True:
        data = exp_handle.collate_fn([exp,], 'cuda')
        future = net.sample(*data)['action'].cpu().numpy()
        action = list(future[0][0])
        path = predict_trajectory(future[0], cfg['control_hz'])
        states, reward, done, info = env.step([action,], path=[path,])
        if info['all_down'][0]: break

        if len(exp['laser']) > exp_handle.history_max_len+1:
            for key in exp.keys():
                exp[key].pop(0)

        exp['laser'].append(states[0])
        exp['vector'].append(states[1])
        exp['action'].insert(-1, numpy.array([[states[1][0][-2], states[1][0][-1]]]))
        exp['reward'].insert(-1, reward)
