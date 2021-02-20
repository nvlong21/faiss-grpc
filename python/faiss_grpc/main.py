from environs import Env

from faiss_grpc.faiss_server import FaissServiceConfig, Server, ServerConfig

env = Env()
env.read_env()


def main(yaml_config:str, storage:str) -> None:
    server_config = ServerConfig(
        host=env.str('FAISS_GRPC_HOST', '[::]'),
        port=env.int("FAISS_GRPC_PORT", 50051),
        max_workers=env.int("FAISS_GRPC_MAX_WORKERS", 10),
    )
    service_config = FaissServiceConfig(
        nprobe=env.int("FAISS_GRPC_NPROBE", None),
        normalize_query=env.bool("FAISS_GRPC_NORMALIZE_QUERY", False),
    )

    server = Server(server_config, service_config, yaml_config, storage)
    server.serve()

if __name__ == "__main__":
    main()
