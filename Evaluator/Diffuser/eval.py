import torch, numpy, sys, os, math

from env.envs import make_env, read_yaml
from nn import GaussianDiffusion, TemporalUnet, ValueFunction, ValueDiffusion

from sampling import ValueGuide, n_step_guided_p_sample

output = './output/'
logname = 'eval.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = read_yaml('./cfg/BARN.yaml')
env = make_env(cfg)
loadpath = './model/state_49_5.pt'
value_loadpath = './model/state_value_49_2.pt'
model = TemporalUnet(**cfg['TemporalUnet'])
Diffusion_model = GaussianDiffusion(model=model, **cfg['GaussianDiffusion'])
value_func = ValueFunction(**cfg['ValueFunction'])
Value_model = ValueDiffusion(model=value_func, **cfg['ValueDiffusion'])
data = torch.load(loadpath)
Diffusion_model.load_state_dict(data['model'])
value_data = torch.load(value_loadpath)
Value_model.load_state_dict(value_data['model'])
guide = ValueGuide(Value_model)

states = env.reset()
while True:
    exp = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
    laser = states[0].reshape(1, -1)
    vector = states[1][:, :3]
    action = numpy.zeros((1, 2))
    action_dim = action.shape[1]
    
    while True:
        data = numpy.concatenate((action, vector, laser), axis=1)
        data = torch.from_numpy(data).to(dtype=torch.float32)
        cond = {0: data[:, action_dim:]}
        samples = Diffusion_model(cond, guide=guide, verbose=False, sample_fn=n_step_guided_p_sample, **cfg['Sampling'])
        traj = samples.trajectories
        action = traj[0][0][:action_dim]
        states, reward, done, info = env.step([action,])
        if info['all_down'][0]: break

        laser = states[0].reshape(1, -1)
        vector = states[1][:, :3]
        action = numpy.array([[states[1][0][-2], states[1][0][-1]]])
        reward = reward[:, :3]
