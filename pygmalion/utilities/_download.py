import requests
import pathlib
from io import IOBase, BytesIO
from tqdm import tqdm


def download(file_path: str, url: str):
    """
    Download google drive a file from the given url to the disk.
    If the directory does not exists raise an error.
    If the file already exists skip it.

    Parameters
    ----------
    directory : str
        directory in which the file is saved
    file_name : str
        name of the file
    url : str
        url to download it from
    """
    file_path = pathlib.Path(file_path)
    # test if path are valid
    directory = file_path.parent
    if not directory.is_dir():
        raise NotADirectoryError(f"The directory '{directory}' does not exists")
    if file_path.is_file():
        print(f"skipping file '{file_path.name}' as it already exists", flush=True)
        return
    with open(file_path, "wb") as f:
        _download_to_stream(url, f)


def download_bytes(url: str) -> BytesIO:
    """
    Download a google drive file to a BytesIO
    """
    stream = BytesIO()
    _download_to_stream(url, stream)
    stream.seek(0)
    return stream


def _download_to_stream(url: str, stream: IOBase, file_name: str="Download"):
    """
    Download the bytes of the give google drive file into the given IO stream
    """
    session = requests.Session()
    response = session.get(_direct_url(url), stream=True)
    if response.status_code >= 400:
        raise RuntimeError(f"http error: {response.status_code}")
    total_size = int(response.headers['content-length'])
    CHUNK_SIZE = 4096
    with tqdm(unit="B", total=total_size, unit_scale=True, unit_divisor=1000) as pbar:
        for chunk in response.iter_content(CHUNK_SIZE):
            stream.write(chunk)
            pbar.update(len(chunk))


def _direct_url(url: str) -> str:
    """
    Converts a googledrive 'share' url to a direct download url

    Parameters
    ----------
    url : str
        the link of of a shared googledrive file

    Returns
    -------
    str :
        the direct download url
    """
    id = url.split("/")[-2]
    return f"https://docs.google.com/uc?export=download&confirm=t&id={id}"
