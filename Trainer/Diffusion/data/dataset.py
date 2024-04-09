import random, pickle, pymongo, torch, math
import numpy as np
from torch.utils.data import Dataset
from typing import List

class NaviDataset(Dataset):
    def __init__(self, host, database='DataBase', collection='Collection'):
        super(NaviDataset, self).__init__()
        self.client = pymongo.MongoClient(host)
        self.database = self.client[database]
        self.collection = self.database[collection]
        self.Len = self.collection.estimated_document_count()

    @classmethod
    def __decode_data(cls, bytes_data: bytes) -> List[np.ndarray]:
        return pickle.loads(bytes_data)

    @classmethod
    def __encode_data(cls, np_list_data: List[np.ndarray]) -> bytes:
        return pickle.dumps(np_list_data)

    @classmethod
    def __data_to_exp(cls, data):
        exp = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        for key in exp.keys():
            exp[key] = NaviDataset.__decode_data(data[key])
        return exp

    def __len__(self):
        return self.Len
    
    def __getitem__(self, index):
        return NaviDataset.__data_to_exp(self.collection.find_one({'_id': int(index)}))
    
    def __getitems__(self, index_list):
        return list(map(NaviDataset.__data_to_exp, self.collection.find({'_id': {'$in': index_list}})))

class NaviCollateFN:
    def __init__(self, history_max_len, future_max_len, exp_sample_num, reward_max, reward_min, up_reward_topk):
        self.history_max_len = history_max_len
        self.future_max_len = future_max_len
        self.exp_sample_num = exp_sample_num
        self.reward_max = reward_max
        self.reward_min = reward_min
        self.up_reward_topk = up_reward_topk
    
    def collate_fn(self, batch):
        data = {'laser':[], 'vector':[], 'action':[], 'reward':[], 'history_mask':[], 'condition':[], 'condition_label':[], 'future':[]}

        for exp in batch:
            elen = len(exp['laser'])
            if elen <= self.future_max_len: continue
            for _ in range(self.exp_sample_num):
                si = random.randint(0, elen-self.future_max_len)
                tlen = min(random.randint(1, self.history_max_len+1), elen-self.future_max_len-si+1)

                for key in ['laser', 'vector', 'action', 'reward']:
                    data[key].append(np.concatenate(exp[key][si:si+tlen], axis=0))
                    data[key][-1] = np.concatenate([np.zeros((self.history_max_len+1-tlen,)+data[key][-1].shape[1:]), data[key][-1]], axis=0)
                    data[key][-1] = data[key][-1].reshape((1,)+data[key][-1].shape)
                data['history_mask'].append(np.concatenate([np.zeros((1, self.history_max_len+1-tlen)), np.ones((1, tlen))], axis=1))
                data['condition'].append(np.sum(np.stack(exp['reward'][si+tlen-1:si+tlen-1+self.future_max_len]), axis=0))
                data['condition_label'].append(np.ones_like(data['condition'][-1]))
                data['future'].append(np.concatenate(exp['action'][si+tlen-1:si+tlen-1+self.future_max_len], axis=0))
                data['future'][-1] = data['future'][-1].reshape((1,)+data['future'][-1].shape)

        _type = [torch.float32, torch.float32, torch.float32, torch.float32, torch.bool, torch.float32, torch.int64, torch.float32]
        for key, type in zip(data.keys(), _type):
            data[key] = np.concatenate(data[key], axis=0)
            data[key] = torch.from_numpy(data[key]).to(dtype=type)
            if key in ['vector', 'reward']:
                data[key] = data[key][:,:,:3]
            elif key in ['condition', 'condition_label']:
                data[key] = data[key][:,:3]
            else: pass

        batch_size, reward_type = data['condition'].shape[0], data['condition'].shape[1]
        up_batch = math.ceil(batch_size*self.up_reward_topk)
        _, indices = data['condition'].topk(up_batch, dim=0)
        indices = indices.permute(1, 0).repeat(1, reward_type).reshape(-1)
        
        for key in data.keys():
            if key == 'condition':
                _tensor0 = data[key][indices]
                _mask = torch.eye(reward_type).unsqueeze(1).repeat(1, up_batch*reward_type, 1).reshape((-1, reward_type))
                _noise = torch.rand_like(_tensor0)
                _max = (torch.Tensor(self.reward_max)*self.future_max_len).unsqueeze(0).repeat(_tensor0.shape[0], 1)
                _min = (torch.Tensor(self.reward_min)*self.future_max_len).unsqueeze(0).repeat(_tensor0.shape[0], 1)
                _tensor1 = _tensor0 + (_min+(_max-_min)*_noise-_tensor0)
                data[key] = torch.cat([data[key], _tensor1], dim=0)

                _tensor = data['condition_label'][indices]
                _tensor = _tensor + (torch.sign(_tensor1-_tensor0)).to(dtype=torch.int64)
                data['condition_label'] = torch.cat([data['condition_label'], _tensor], dim=0)
            elif key == 'condition_label': pass
            else: data[key] = torch.cat([data[key], data[key][indices]], dim=0)

        data_split = {}  
        data_split['action'] = data.pop('future')
        data_split['condition'] = data.pop('condition')
        data_split['condition_label'] = data.pop('condition_label')

        info = {}
        info['batch'] = batch_size
        info['up_batch'] = up_batch * reward_type * reward_type
        info['topk'] = self.up_reward_topk

        return [data_split, data, info]