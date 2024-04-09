import torch, numpy, sys, os, yaml, pymongo, pickle
from typing import List

from nn import NaviDiffusion, ClassifyFreeDDIM
from data import NaviDataset


def read_yaml(file: str) -> dict:
    with open(file, 'r', encoding="utf-8") as f:
        data = yaml.load(f.read(), Loader=yaml.FullLoader)
    return data

class ExpHandle:
    def __init__(self, dataset, batch, history_max_len, future_max_len):
        self.dataset = dataset
        self.dataset_len = dataset.__len__()
        self.dataset_pt = 0
        self.batch = batch
        self.history_max_len = history_max_len
        self.future_max_len = future_max_len

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
    
    def _randn(self, data):
        if isinstance(data, list):
            for i in range(len(data)):
                data[i] = self._randn(data[i])
        elif isinstance(data, dict):
            for key in data.keys():
                data[key] = self._randn(data[key])
        elif isinstance(data, torch.Tensor):
            data = torch.randn_like(data)
        else:
            raise ValueError
        return data

    def collate_fn(self):
        history = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        history_mask = []
        condition = []
        future = {'laser':[], 'vector':[], 'action':[], 'reward':[]}

        batch_i = 0
        while self.dataset_pt < self.dataset_len and batch_i < self.batch:
            exp = self.dataset.__getitem__(self.dataset_pt)
            self.dataset_pt += 1
            if exp['reward'][-2][0][-1] >= 1-1000: continue
            elen = len(exp['laser'])
            if elen <= self.future_max_len: continue
            si = max(0, elen-self.future_max_len-self.history_max_len)
            tlen = elen-self.future_max_len-si+1

            for key in history.keys():
                history[key].append(numpy.concatenate(exp[key][si:si+tlen], axis=0))
                history[key][-1] = numpy.concatenate([numpy.zeros((self.history_max_len+1-tlen,)+history[key][-1].shape[1:]), history[key][-1]], axis=0)
                history[key][-1] = history[key][-1].reshape((1,)+history[key][-1].shape)
                future[key].append(numpy.concatenate(exp[key][si+tlen-1:si+tlen-1+self.future_max_len], axis=0))
                future[key][-1] = future[key][-1].reshape((1,)+future[key][-1].shape)
            history_mask.append(numpy.concatenate([numpy.zeros((1, self.history_max_len+1-tlen)), numpy.ones((1, tlen))], axis=1))
            condition.append(numpy.zeros((1, 1)) if future['reward'][-1][0][-2][-1] < 1-1000 else numpy.ones((1, 1)))
            batch_i += 1

        if batch_i == 0:
            return None

        for key in history.keys():
            history[key] = torch.from_numpy(numpy.concatenate(history[key], axis=0)).to(dtype=torch.float32)
            future[key] = torch.from_numpy(numpy.concatenate(future[key], axis=0)).to(dtype=torch.float32)
        history_mask = torch.from_numpy(numpy.concatenate(history_mask, axis=0)).to(dtype=torch.bool)
        condition = torch.from_numpy(numpy.concatenate(condition, axis=0)).to(dtype=torch.float32)

        # cut off useless information
        # history['laser'] = 1.0 / (torch.clamp(history['laser'], min=0.0) + 0.01)
        history['vector'] = history['vector'][:,:,:3]
        history['reward'] = history['reward'][:,:,:3]
        future['vector'] = future['vector'][:,:,:3]
        future['reward'] = future['reward'][:,:,:3]

        x = future
        c = history
        c['history_mask'] = history_mask
        c['condition'] = condition

        return [x, c]

class OutputDataset:
    def __init__(self, host, database='DataBase', collection='Collection'):
        super(OutputDataset, self).__init__()
        self.client = pymongo.MongoClient(host)
        self.database = self.client[database]
        self.collection = self.database[collection]
        self.num = 0

    @classmethod
    def __decode_data(cls, bytes_data: bytes) -> List[numpy.ndarray]:
        return pickle.loads(bytes_data)

    @classmethod
    def __encode_data(cls, np_list_data: List[numpy.ndarray]) -> bytes:
        return pickle.dumps(np_list_data)
    
    def output(self, data):
        x, c = data[0], data[1]
        
        c['vector'] = torch.nn.functional.pad(c['vector'], (2, 0, 0, 0, 0, 0), 'constant', 0)
        c['reward'] = torch.nn.functional.pad(c['reward'], (3, 0, 0, 0, 0, 0), 'constant', 0)
        x['vector'] = torch.nn.functional.pad(x['vector'], (2, 0, 0, 0, 0, 0), 'constant', 0)
        x['reward'] = torch.nn.functional.pad(x['reward'], (3, 0, 0, 0, 0, 0), 'constant', 0)

        for i in range(c['laser'].shape[0]):
            exp = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
            mask_pt = torch.where(c['history_mask'][i]==True)[0]
            for key in exp.keys():
                exp[key] = list(torch.cat([c[key][i][mask_pt], x[key][i]], dim=0).unsqueeze(1).cpu().numpy())
            self.collection.insert_one({'_id' : int(self.num),
                                                'laser' : self.__encode_data(exp['laser']),
                                                'vector' : self.__encode_data(exp['vector']),
                                                'action' : self.__encode_data(exp['action']),
                                                'reward' : self.__encode_data(exp['reward'])
                                                })
            self.num += 1


output = './output/'
logname = 'generate.log'
if not os.path.exists(output): os.makedirs(output)
sys.stdout = open(output+logname, 'w+')

cfg = read_yaml('./cfg/generate.yaml')
dataset = NaviDataset(cfg['Dataset']['input'])
output = OutputDataset(cfg['Dataset']['output'])
net = NaviDiffusion(**cfg['Network']).cuda().float()
net.load_state_dict(torch.load('./model/last_model_shared.pt'))
net = ClassifyFreeDDIM(net, **cfg['Diffusion'])
exp_handle = ExpHandle(dataset, cfg['Dataset']['batch'], **cfg['Dataset']['CollateFN'])

while exp_handle.dataset_pt < exp_handle.dataset_len:
    data = exp_handle.collate_fn()
    if data is None: break
    data[0] = exp_handle._randn(data[0])
    data[0]['condition'] = data[1].pop('condition')
    data = exp_handle._to_device(data, 'cuda')
    future = net.sample(*data)
    data[0] = future
    output.output(data)
