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
    def __init__(self, horizon):
        self.horizen = horizon
    
    def collate_fn(self, batch):
        data = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        # print(batch)

        for exp in batch:
            elen = len(exp['laser'])
            if elen <= self.horizen: continue
            si = random.randint(0, elen-self.horizen)

            for key in ['laser', 'vector', 'action', 'reward']:
                data[key].append(np.concatenate(exp[key][si:si+self.horizen], axis=0))

        _type = [torch.float32, torch.float32, torch.float32, torch.float32]
        for key, type in zip(data.keys(), _type):
            data[key] = np.array(data[key])
            data[key] = torch.from_numpy(data[key]).to(dtype=type)
            if key in ['vector', 'reward']:
                data[key] = data[key][:,:,:3]
            else: pass
        
        # batch_size = data['action'].shape[0]
        # seq_length = data['action'].shape[1]
        action_dim = data['action'].shape[2]
        # x = torch.stack([data['action'], data['vector'], data['laser'], data['reward']], dim=1).permute(0, 2, 1, 3).reshape(batch_size, (1+1+1+1)*seq_length, -1)
        x = torch.cat([data['action'], data['vector'], data['laser']], dim=-1)

        cond = {0: x[:, 0, action_dim:]}

        return [x, cond]
    
class NaviValueCollateFN:
    def __init__(self, horizon):
        self.horizen = horizon
    
    def collate_fn(self, batch):
        data = {'laser':[], 'vector':[], 'action':[], 'reward':[]}
        # print(batch)

        for exp in batch:
            elen = len(exp['laser'])
            if elen <= self.horizen: continue
            si = random.randint(0, elen-self.horizen)

            for key in ['laser', 'vector', 'action', 'reward']:
                seq = np.concatenate(exp[key][si:si+self.horizen], axis=0)
                if key == 'reward':
                    count = np.zeros((1,))
                    for i in range(seq.shape[0]):
                        temp = (seq[i, :] - np.array([0, 5, 10, 0, 0, 0])) * np.array([4, 2, 0, 0, 1, 1])
                        temp = np.sum(temp, axis=-1, keepdims=False)
                        count = count + temp
                    seq = count             
                data[key].append(seq)

        _type = [torch.float32, torch.float32, torch.float32, torch.float32]
        for key, type in zip(data.keys(), _type):
            data[key] = np.array(data[key])
            data[key] = torch.from_numpy(data[key]).to(dtype=type)
            if key in ['vector']:
                data[key] = data[key][:,:,:3]
            else: pass
        
        # batch_size = data['action'].shape[0]
        # seq_length = data['action'].shape[1]
        action_dim = data['action'].shape[2]
        x = torch.cat([data['action'], data['vector'], data['laser']], dim=-1)
        target = data['reward']

        cond = {0: x[:, 0, action_dim:]}

        return [x, cond, target]
