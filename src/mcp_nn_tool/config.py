BASE_URL = "http://localhost:8090"
DOWNLOAD_URL = "./trained_model"
MCP_PORT = 8090
from pathlib import Path


def get_download_url(path:str):
    return f"{BASE_URL}/download/file/{Path(path).relative_to(DOWNLOAD_URL).as_posix()}"

def get_static_url(path:str):
    return f"{BASE_URL}/static/{Path(path).relative_to(DOWNLOAD_URL).as_posix()}"