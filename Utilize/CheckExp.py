import pickle, random, copy, pymongo, time
import numpy as np
from typing import List

class DataBase:
    def __init__(self, host, database, collection):
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
        exp = {'laser':[], 'vector':[], 'ped':[], 'action':[], 'reward':[]}
        for key in exp.keys():
            exp[key] = DataBase.__decode_data(data[key])
        length = len(exp['reward'])
        exp['return_to_go'] = copy.deepcopy(exp['reward'])
        for j in range(2, length+1):
            exp['return_to_go'][length-j] = np.add(exp['return_to_go'][length-j], exp['return_to_go'][length-j+1])
        return exp

    def random_get(self, index=None):
        if index == None:
            index = random.randint(0, self.Len-1)
        document = self.collection.find_one({'_id': int(index)})
        return DataBase.__data_to_exp(document)

    def close(self):
        if self.client != None:
            self.client.close()
            self.client = self.database = self.collection = None

host = 'mongodb://localhost:27017/'
database = 'DataBase'
collection = 'Collection'

db = DataBase(host, database, collection)
Time = time.time()
exp = db.random_get()
Time = time.time() - Time
print('DataBase:', db.Len, '\n')
print('GetOneCost:', Time, '\n')
db.close()



for si in range(len(exp['action'])):
    print('action:', exp['action'][si], '\n')
    print('reward:', exp['reward'][si], '\n')

print('All_RTG:', exp['return_to_go'][0], '\n')
print('length:', len(exp['action']), '\n')
ti = random.randint(0, len(exp['action'])-1)
print('sample_state:\n', 'laser:', exp['laser'][ti], '\n', 'vector:', exp['vector'][ti], '\n', 'ped:', exp['ped'][ti], '\n')