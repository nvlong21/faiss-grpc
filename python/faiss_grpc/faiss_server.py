import os
import cv2
import time

from concurrent import futures
from dataclasses import dataclass
from typing import List, Optional

import faiss
import grpc
import numpy as np
import glob

pwd = os.path.dirname(os.path.realpath(__file__))
import datetime
from copy import deepcopy
import config_with_yaml as yaml_config
from easydict import EasyDict as edict
from contextlib import contextmanager
import json
import pyssdb
import base64 
from faiss_grpc.faiss_pb2 import (
    SimpleResponse,
    Neighbor,
    SearchByIdResponse,
    SearchResponse,
    InsertResponse,
    RemoveResponse,
    MultipleNeighbor,
    Entity
)
from faiss_grpc.faiss_pb2_grpc import (
    FaissServiceServicer,
    add_FaissServiceServicer_to_server,
)
from environs import Env

# import faiss

def sanitize(x):
    return np.ascontiguousarray(x, dtype='float32')

@contextmanager
def timer(name):
    t0 = time.time()
    yield
    print("[{}] done in {:.3f} s".format(name, time.time() - t0))

@dataclass(eq=True, frozen=True)
class ServerConfig:
    host: str = '[::]'
    port: int = 50051
    max_workers: int = 10


@dataclass(eq=True, frozen=True)
class FaissServiceConfig:
    nprobe: Optional[int] = None
    normalize_query: bool = False


env = Env()
env.read_env()

SSDB_HOST = env.str('SSDB_HOST', '[::]')

class FaissServiceServicer(FaissServiceServicer):
    
    collection_default = "test"
    def __init__(self, config: FaissServiceConfig, yaml_config:str, storage:str) -> None:
        self.config = config
        self.yaml_config = yaml_config
        self.storages = storage
        if self.config.nprobe:
            self.index.nprobe = self.config.nprobe

        self.__config__()
        self.collections = edict()
        # self.vector_dim = self.config.vector_dim
        if self.cfg.getProperty("Faiss.MetaData.Enable"):
            _P = os.environ.get("SSDB_PORT", self.cfg.getProperty("Faiss.MetaData.Port"))
            SSDB_PORT = int(_P)
            self.ssdb_client = pyssdb.Client(host=SSDB_HOST, port = SSDB_PORT)

        self.status = {}
        self._load_all_collection()
        
    def _load_all_collection(self):
        list_collection = os.listdir(self.storages)
        list_collection.remove(self.collection_default)
        self.load_collection(self.collection_default)
        for cl in list_collection:
            self.load_collection(self.collection_default)
        
    def __config__(self):
        self.cfg = yaml_config.load(self.yaml_config)
        self.gpu = False
        if self.cfg.getProperty("Faiss.Gpu.Enable"):
            self.gpu = True
            if len(self.cfg.getProperty("Faiss.Gpu.Device"))>1:
                self.devices = [int(g) for g in self.cfg.getProperty("Faiss.Gpu.Device").split(',')]
            elif len(self.cfg.getProperty("Faiss.Gpu.Device"))==1:
                self.devices = [int(self.cfg.getProperty("Faiss.Gpu.Device"))]

            else:
                self.devices = [0]
        else:
            self.devices = [-1]
    
    def Search(self, request, context) -> SearchResponse:
        collection_name = request.collection
        if collection_name is None:
            collection_name = self.collection_default
        if self.status[collection_name]:
            queries = []
            for q in request.queries:
                query = np.array(q.val, dtype=np.float32)
                queries.append(query)
                # if query.shape[1] != self.index.d:
                #     context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                #     msg = (
                #         'query vector dimension mismatch '
                #         f'expected {self.index.d} but passed {query.shape[1]}'
                #     )
                #     context.set_details(msg)
                #     return SearchResponse()
            
            queries = np.atleast_2d(queries)
            
            if not getattr(self.collections, collection_name).ntotal:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                msg = (
                    'database vector empty expected not empty'
                )
                context.set_details(msg)
                return SearchResponse()

            if isinstance(queries, list):
                queries = np.asarray(queries).astype('float32')

            if self.config.normalize_query:
                queries = self.normalize(queries)
            D, I = getattr(self.collections, collection_name).search(queries, request.k)
            multi_neighbors: List[MultipleNeighbor] = []
            for i, (distances, ids) in enumerate(zip(D, I)):
                neighbors: List[Neighbor] = []

                for d, i in zip(distances, ids):
                    if i != -1:
                        entities: List[Entity] = []
                        if self.cfg.getProperty("Faiss.MetaData.Enable"):
                            fields = [field.strip() for field in self.cfg.getProperty("Faiss.MetaData.Files").split('|')]

                            for fi in fields:
                                val_fill = self.ssdb_client.hget(str(i), fi)
                                if isinstance(val_fill, bytes):
                                    val_fill = val_fill.decode('utf-8')
                                elif val_fill is None:
                                    val_fill = ''
                                else:
                                    val_fill = str(val_fill)
                                entities.append(Entity(name = fi, val = val_fill))
                        neighbors.append(Neighbor(id=i, score=d, entities= entities))
                    else:
                        neighbors.append(Neighbor(id=-1, score=2.0, entities= []))

                multi_neighbors.append(MultipleNeighbor(neighbors = neighbors))
            print("There are %d data, RSS %d kiB] " % (
                getattr(self.collections, collection_name).ntotal,
                faiss.get_mem_usage_kb()))
            return SearchResponse(multi_neighbors=multi_neighbors)


    def Insert(self, request, context) -> InsertResponse:
        collection_name = request.collection
        count = 0
        max_batch = 10000
        embeddings = []
        for vt in request.vectors:
            vec = np.array(vt.val, dtype=np.float32)
            embeddings.append(vec)
            # if query.shape[1] != self.index.d:
            #     context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            #     msg = (
            #         'query vector dimension mismatch '
            #         f'expected {self.index.d} but passed {query.shape[1]}'
            #     )
            #     context.set_details(msg)
            #     return SearchResponse()
        embeddings = np.atleast_2d(embeddings)
        ids = request.ids
        multi_entities = request.multi_entities
        if collection_name is None:
            collection_name = self.collection_default
        
        if len(embeddings):
            embeddings = sanitize(np.asarray(embeddings, dtype=np.float32))
            ids = np.asarray(ids, dtype=np.int)
        passed_ids = []
        # if self.cfg.getProperty("Faiss.MetaData.Enable"):
        #     fields = [field.strip() for field in self.cfg.getProperty("Faiss.MetaData.Files").split('|')]
        while count < embeddings.shape[0]:
            
            batch_embedding = embeddings[count:max_batch]
            batch_id = ids[count:max_batch]
            batch_entities = multi_entities[count:max_batch]
            getattr(self.collections, collection_name).add_with_ids(batch_embedding, batch_id)
            for id, eb, en in zip(batch_id, batch_embedding, batch_entities):
                try:
                    collection_path = os.path.join(self.storages, collection_name)
                    os.makedirs(collection_path, exist_ok = True)
                    data_collection_path = os.path.join(collection_path, "data")
                    os.makedirs(data_collection_path, exist_ok = True)
                    vector_path = os.path.join(data_collection_path, "{}.npy".format(id))
                    np.save(vector_path, eb)
                    # if "image" in rc.keys():
                    #     image_path = os.path.join(data_collection_path, "{}.jpg".format(id))
                    #     cv2.imwrite(image_path, rc['image'])
                    # metadata_path = os.path.join(data_collection_path, "{}.json".format(id))
                    meta_data = {}
                    for e in en.entities:
                        meta_data[str(e.name)] = str(e.val)   
                        self.ssdb_client.hset(str(id), str(e.name), str(e.val))
                    with open(os.path.join(data_collection_path, "{}.json".format(id)), 'w', encoding='utf8') as json_file:
                        json.dump(meta_data, json_file, ensure_ascii=False)
                    passed_ids.append(id)
                except Exception as e:
                    print(e)

                time.sleep(0.2)
            count+=max_batch
        return InsertResponse(message="ok", id = passed_ids)

    def Remove(self, request, context) -> RemoveResponse:
        collection_name = request.collection
        if collection_name is None:
            collection_name = self.collection_default

        ids = request.ids
        max_batch = 10000
        count = 0
        pass_ids:List[int] = []
        while count < len(ids):
            batch_id = ids[count:max_batch]
            batch_id = np.array(batch_id)
            batch_id = np.asarray(batch_id, dtype=np.int)
            for id in batch_id:
                try:
                    collection_path = os.path.join(self.storages, collection_name)
                    data_collection_path = os.path.join(collection_path, "data")
                    vector_path = os.path.join(data_collection_path, "{}.npy".format(id))
                    os.remove(vector_path) 
                    image_path = os.path.join(data_collection_path, "{}.jpg".format(id))
                    if os.path.exists(image_path):
                        os.remove(image_path)
                    else:
                        print("Can not delete the file as it doesn't exists")

                    metadata_path = os.path.join(data_collection_path, "{}.json".format(id))
                    if os.path.exists(metadata_path):
                        os.remove(metadata_path)
                    else:
                        print("Can not delete the file as it doesn't exists")

                    self.ssdb_client.hclear(str(id))
                    pass_ids.append(id)
                except Exception as e:
                    print(e)
                time.sleep(0.05)
            count += max_batch
        self.load_collection(collection_name)
        return RemoveResponse(message = 'OK', ids = pass_ids)

    def CreateCollection(self, request, context) -> SimpleResponse:
        collection_name = request.collection_name
        dim = int(request.dim)
        self._create_collection(collection_name, dim)
        return SimpleResponse(message='OK')

    def Heatbeat(self, request, context) -> SimpleResponse:
        return SimpleResponse(message='OK')

    @staticmethod
    def normalize(vec: np.ndarray, axis: int = 1) -> np.ndarray:
        norm = np.linalg.norm(vec, axis=axis, keepdims=True)
        output = vec / norm
        return output

    # @staticmethod
    def exists_index(self, collection_name, name):
        if collection_name is None:
            collection_name = self.collection_default
        os.makedirs(os.path.join(self.storages, collection_name), exist_ok = True)
        if os.path.exists(os.path.join(self.storages, collection_name, name)):
            return True, os.path.join(self.storages, collection_name, name)
        else:
            return False, os.path.join(self.storages, collection_name, name)

    def list_collections(self):
        collection_names = ', '.join(i for i in dir(self.collections) if not i.startswith('__'))
        return collection_names

    def len_collections(self):
        collection_names = ', '.join(i for i in dir(self.collections) if not i.startswith('__'))
        return len(collection_names)

    def _collection_to_gpu(self, collection):
        with timer("activate GPU"):
            if len(self.devices)>1:
                def make_vres_vdev():
                    " return vectors of device ids and resources useful for gpu_multiple"
                    ngpu = faiss.get_num_gpus()
                    vres = faiss.GpuResourcesVector()
                    vdev = faiss.IntVector()
                    i0 = min(self.devices)
                    i1 = max(max(self.devices), ngpu)
                    if i1 == -1:
                        i1 = ngpu
                    for i in range(i0, i1):
                        res = faiss.StandardGpuResources()
                        tempmem  = self.cfg.getProperty("Faiss.Gpu.TempMemory")
                        if tempmem >= 0:
                            res.setTempMemory(tempmem)
                        vdev.push_back(i)
                        vres.push_back(res)
                    return vres, vdev

                co = faiss.GpuClonerOptions()
                # here we are using a 64-byte PQ, so we must set the lookup tables to
                # 16 bit float (this is due to the limited temporary memory).
                co.useFloat16 = False
                vres, vdev = make_vres_vdev()
                _collection = faiss.index_cpu_to_gpu_multiple(vres,  vdev, collection, co)
            else:
                # co = faiss.GpuClonerOptions()
                # here we are using a 64-byte PQ, so we must set the lookup tables to
                # 16 bit float (this is due to the limited temporary memory).
                # co.useFloat16 = False
            
                res = faiss.StandardGpuResources()
                tempmem  = self.cfg.getProperty("Faiss.Gpu.TempMemory")
                if tempmem >= 0:
                    res.setTempMemory(tempmem)
                _collection  = faiss.index_cpu_to_gpu(res, self.devices[0], collection)
        return _collection

    def _create_collection(self, collection_name, dim = 512):
        if collection_name is None:
            collection_name = self.collection_default
        _collection = faiss.IndexHNSWFlat(dim, 32)
        _collection.hnsw.efConstruction = 40
        _collection.verbose = True
        _collection = faiss.IndexIDMap2(_collection)
        _, path = self.exists_index(collection_name, "%s_trained.index" % collection_name)
        faiss.write_index(_collection, path)
        setattr(self.collections, collection_name, _collection)
        return _collection

    @staticmethod
    def save_collection(collection, path):
        faiss.write_index(collection, path)

    def load_collection(self, collection_name, load_trained = True, auto_creat =True, empty_index = False):
        if collection_name is None:
            collection_name = self.collection_default
        _collection = None
        with timer("load index {}".format(collection_name)):
            if load_trained:
                collection_name_index = collection_name + '_trained'
            else:
                collection_name_index = collection_name
            p_exists, path = self.exists_index(collection_name, "%s.index" % collection_name_index)
            if p_exists:
                _collection = faiss.read_index(path)
                
            else:
                if auto_creat:
                    _collection = self._create_collection(collection_name)
                else:
                    raise Exception("Collection %s is not exists!"%collection_name)
            if empty_index:
                return _collection
            if self.gpu:
                _collection = self._collection_to_gpu(_collection)
            
            collection_path = os.path.join(self.storages, collection_name)
            if os.path.exists(collection_path):
                data_collection_path = os.path.join(collection_path, "data")
                if os.path.exists(data_collection_path):
                    data_list = os.listdir(data_collection_path)
                    ids = []
                    vts = []
                    for dt in data_list:
                        try:
                            name, ext = os.path.splitext(dt)
                            if ext == '.npy':
                                c_id = int(name)
                                vt = np.load(os.path.join(data_collection_path, dt))
                                ids.append(c_id)
                                vts.append(vt)
                        except:
                            continue
                    if len(vts) and len(ids):
                        vts = sanitize(vts)
                        ids = np.array(ids)
                        _collection.add_with_ids(vts, ids)
                        _collection.search(vts[0:2], 1)
                    del vts, ids
            print("There are %d data, RSS %d kiB] " % ( _collection.ntotal, faiss.get_mem_usage_kb()))
        self.status[collection_name] = False
        setattr(self.collections, collection_name, _collection)
        self.status[collection_name] = True
        
class Server:
    def __init__(
        self,
        # index_path: str,
        server_config: ServerConfig,
        service_config: FaissServiceConfig,
        yaml_config:str, 
        storage:str
    ) -> None:
        
        # index = faiss.read_index(index_path)
        self.server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=server_config.max_workers)
        )
        add_FaissServiceServicer_to_server(
            FaissServiceServicer(service_config, yaml_config, storage), self.server
        )
        self.server.add_insecure_port(
            f'{server_config.host}:{server_config.port}'
        )

    def serve(self) -> None:
        self.server.start()
        self.server.wait_for_termination()
