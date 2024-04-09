import tqdm
from pymongo import MongoClient

def copy_documents(source_client, source_db, source_collection_name, target_client, target_db, target_collection_name, start_id, end_id):
    # 连接到数据库 B
    source_db = source_client[source_db]
    source_collection = source_db[source_collection_name]

    # 获取数据库 B 中的要复制的文档
    cursor = source_collection.find({'_id': {'$gte': start_id, '$lte': end_id}})

    # 连接到数据库 A
    target_db = target_client[target_db]
    target_collection = target_db[target_collection_name]

    # 复制文档到数据库 A
    # for document in tqdm.tqdm(cursor, '', ncols=120, unit='Document'):
    #     target_collection.update_one({'_id': document['_id']}, {'$set': document}, upsert=True)

    Len = int(target_collection.estimated_document_count())
    for document in tqdm.tqdm(cursor, '', ncols=120, unit='Document'):
        document['_id'] = int(Len)
        target_collection.insert_one(document)
        Len += 1

    print("Finish move！")

if __name__ == "__main__":
    # 连接到数据库 B
    source_client = MongoClient('localhost', 27017)  # 替换为数据库 B 的连接信息
    source_db_name = 'DataBase'  # 替换为数据库 B 的名称
    source_collection_name = 'Collection'  # 替换为数据库 B 中的集合名称

    # 连接到数据库 A
    target_client = MongoClient('localhost', 27018)  # 替换为数据库 A 的连接信息
    target_db_name = 'DataBase'  # 替换为数据库 A 的名称
    target_collection_name = 'Collection'  # 替换为数据库 A 中的集合名称

    start_id = 0
    end_id = 200000-1

    copy_documents(source_client, source_db_name, source_collection_name, target_client, target_db_name, target_collection_name, start_id, end_id)