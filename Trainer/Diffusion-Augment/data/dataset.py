import random, copy, pickle, pymongo, torch, math
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
        # return NaviDataset.__data_to_exp(self.collection.aggregate([{'$sample': {'size': 1}}]).next())

    def __getitems__(self, index_list):
        return list(map(NaviDataset.__data_to_exp, self.collection.find({'_id': {'$in': index_list}})))

class NaviCollateFN:
    def __init__(self, history_max_len, future_max_len, exp_sample_num):
        self.history_max_len = history_max_len
        self.future_max_len = future_max_len
        self.exp_sample_num = exp_sample_num

    def collate_fn(self, batch):
        history = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        history_mask = []
        condition = []
        future = {'laser':[], 'vector':[], 'action':[], 'reward':[]}

        for exp in batch:
            elen = len(exp['laser'])
            if elen <= self.future_max_len: continue
            for _ in range(self.exp_sample_num):
                if exp['reward'][-2][0][-1] < 1-1000:
                    si = max(0, elen-self.future_max_len-self.history_max_len)
                    tlen = elen-self.future_max_len-si+1
                else:
                    si = random.randint(0, elen-self.future_max_len)
                    tlen = min(random.randint(1, self.history_max_len+1), elen-self.future_max_len-si+1)

                for key in history.keys():
                    history[key].append(np.concatenate(exp[key][si:si+tlen], axis=0))
                    history[key][-1] = np.concatenate([np.zeros((self.history_max_len+1-tlen,)+history[key][-1].shape[1:]), history[key][-1]], axis=0)
                    history[key][-1] = history[key][-1].reshape((1,)+history[key][-1].shape)
                    future[key].append(np.concatenate(exp[key][si+tlen-1:si+tlen-1+self.future_max_len], axis=0))
                    future[key][-1] = future[key][-1].reshape((1,)+future[key][-1].shape)
                history_mask.append(np.concatenate([np.zeros((1, self.history_max_len+1-tlen)), np.ones((1, tlen))], axis=1))
                condition.append(np.zeros((1, 1)) if future['reward'][-1][0][-2][-1] < 1-1000 else np.ones((1, 1)))

        for key in history.keys():
            history[key] = torch.from_numpy(np.concatenate(history[key], axis=0)).to(dtype=torch.float32)
            future[key] = torch.from_numpy(np.concatenate(future[key], axis=0)).to(dtype=torch.float32)
        history_mask = torch.from_numpy(np.concatenate(history_mask, axis=0)).to(dtype=torch.bool)
        condition = torch.from_numpy(np.concatenate(condition, axis=0)).to(dtype=torch.float32)

        # cut off useless information
        # history['laser'] = 1.0 / (torch.clamp(history['laser'], min=0.0) + 0.01)
        history['vector'] = history['vector'][:,:,:3]
        history['reward'] = history['reward'][:,:,:3]
        future['vector'] = future['vector'][:,:,:3]
        future['reward'] = future['reward'][:,:,:3]

        x = future
        c = history
        c['history_mask'] = history_mask
        x['condition'] = condition

        return [x, c]