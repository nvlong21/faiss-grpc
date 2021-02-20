import argparse
from argparse import Namespace
from typing import List, Union

import grpc
import numpy as np
from google.protobuf.empty_pb2 import Empty

import faiss_grpc.faiss_pb2 as faiss_pb2 # isort:skip
import faiss_grpc.faiss_pb2_grpc as faiss_pb2_grpc # isort:skip
VectorLike = Union[List[float], np.ndarray]

class GrpcClient:
    def __init__(self, server_host : str = '0.0.0.0' , server_post:int = 50051) -> None:
        channel = grpc.insecure_channel('{}:{}'.format(server_host, server_post))
        self._stub = faiss_pb2_grpc.FaissServiceStub(channel)

    @property
    def stub(self) -> faiss_pb2_grpc.FaissServiceStub:
        return self._stub

    def search(self, query: VectorLike, k: int) -> None:
        queries: List[faiss_pb2.Vector] = []
        for i in range(query.shape[0]):
            vec = faiss_pb2.Vector(val=query[i])
            queries.append(vec)
        
        req = faiss_pb2.SearchRequest(collection = "test", queries=queries, k=k)
        res = self.stub.Search(req)
        result = []
        for _, mn in enumerate(res.multi_neighbors):
            sub = []
            for i, n in enumerate(mn.neighbors):
                sub.append({'top': i, 'id': n.id, 'score': n.score})
            result.append(sub)
        return result

    def insert(self, qbs: List[dict]) -> str:
        queries: List[faiss_pb2.Vector] = []
        mul_entities: List[faiss_pb2.Entities] = []
        ids: List[int] = []
        for q in qbs:
            vec = faiss_pb2.Vector(val=q["vector"])
            queries.append(vec)
            entities: List[faiss_pb2.Entity] = []
            _dict_entity = q.get('entities', {})
            for e_k in _dict_entity.keys():
                en = faiss_pb2.Entity(name = e_k, val = _dict_entity[e_k])
                entities.append(en)
                
            mul_entities.append(faiss_pb2.Entities(entities=entities))
            ids.append(q['id'])
        
        req = faiss_pb2.InsertRequest(collection = "test", vectors=queries, ids=ids, multi_entities=mul_entities)
        res = self.stub.Insert(req)
        list_ids = []
        for _id in res.id:
            list_ids.append(_id)
        return "OK", list_ids
    
    def remove(self, ids: VectorLike) -> list:
        req = faiss_pb2.RemoveRequest(collection = "test", ids=ids)
        res = self.stub.Remove(req)
        list_ids = []
        for _id in res.ids:
            list_ids.append(_id)
        return "OK", list_ids
    
    def create_collection(self, data: dict) -> None:
        req = faiss_pb2.CreateRequest(collection_name= data.get('collection_name', None), dim = data.get('dim', None))
        res = self.stub.CreateCollection(req)
        print(res)

    def search_by_id(self, request_id: int, k: int) -> None:
        req = faiss_pb2.SearchByIdRequest(id=request_id, k=k)
        res = self.stub.SearchById(req)

        print(f'requested id {res.request_id}')
        for i, n in enumerate(res.neighbors):
            print(f'#{i}, id: {n.id}, score: {n.score}')

    def heatbeat(self) -> None:
        res = self.stub.Heatbeat(Empty())
        print(f'message {res.message}')


def heatbeat(_: Namespace) -> None:
    client = GrpcClient()
    client.heatbeat()

def create_collection(args: Namespace) -> None:
    client = GrpcClient()
    client.create_collection({"collection_name": "test2", "dim": 512})

def search(args: Namespace) -> None:
    client = GrpcClient()
    query = np.random.rand(3,512)
    # query = np.ones(512, dtype=np.float32)
    client.search(query, args.k)

def insert(args: Namespace) -> None:
    client = GrpcClient()
    # query = np.random.rand(3, 512)
    qbs = []
    data = cv2.imread('/home/mdt4/Pictures/123367863_1375636152791241_5754739064388602868_o.jpg')
    for i in range(5):
        temp_dict = {"vector": np.random.rand(512),
                    'entities':
                        {
                            "name": "long"
                        },
                    "id": i
                    }
        qbs.append(temp_dict)
    client.insert(qbs)

def remove(args: Namespace) -> None:
    client = GrpcClient()
    a = [0, 4]
    client.remove(a)

def search_by_id(args: Namespace) -> None:
    client = GrpcClient()
    client.search_by_id(args.id, args.k)

def run() -> None:
    parser = argparse.ArgumentParser(description='gRPC client example')
    sub_parser = parser.add_subparsers(title='subcommands')

    parser_heatbeat = sub_parser.add_parser('heatbeat', description='heatbeat')
    parser_heatbeat.set_defaults(handler=heatbeat)

    parser_search = sub_parser.add_parser(
        'search',
        description=(
            'search nearest neighbors of given query. '
            'in this example query is prepared as identity vector.'
        ),
    )
    parser_search.add_argument('k', type=int)
    parser_search.set_defaults(handler=search)

    parser_search = sub_parser.add_parser(
        'insert',
        description=(
            'search nearest neighbors of given query. '
            'in this example query is prepared as identity vector.'
        ),
    )
    parser_search.set_defaults(handler=insert)

    parser_search = sub_parser.add_parser(
        'remove',
        description=(
            'search nearest neighbors of given query. '
            'in this example query is prepared as identity vector.'
        ),
    )
    parser_search.set_defaults(handler=remove)
    parser_search = sub_parser.add_parser(
        'create',
        description=(
            'search nearest neighbors of given query. '
            'in this example query is prepared as identity vector.'
        ),
    )
    parser_search.set_defaults(handler=create_collection)
    
    parser_seach_by_id = sub_parser.add_parser(
        'search-by-id', description='search nearest neighbors of given id'
    )
    parser_seach_by_id.add_argument('id', type=int)
    parser_seach_by_id.add_argument('k', type=int)
    parser_seach_by_id.set_defaults(handler=search_by_id)

    args = parser.parse_args()

    if hasattr(args, 'handler'):
        args.handler(args)
    else:
        print('subcommand is required one of {heatbeat, search, search-by-id}')


if __name__ == "__main__":
    run()
