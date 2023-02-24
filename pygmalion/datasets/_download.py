import requests
import pathlib
from typing import List


def download_all(directory: str, folder_name: str,
                 file_names: List[str], urls: List[str]):
    """
    Download a series of files in an existing folder.

    Parameters
    ----------
    directory : str
        the base directory where the dataset must be stored
    folder_name : str
        the folder to create in the directory
    file_names : list of str
        the list of file names to create
    urls : list of str
        the list of urls to download them from
    """
    # test if path are valid
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"'{directory}' is not an existing directory")
    directory = directory / folder_name
    if not directory.is_dir():
        directory.mkdir(parents=True, exist_ok=True)
    # download each file
    for file_name, url in zip(file_names, urls):
        download(directory, file_name, url)


def download(directory: str, file_name: str, url: str):
    """
    Download a file from the given url to the disk.
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
    # test if path are valid
    directory = pathlib.Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"The directory '{directory}' does not exists")
    path = directory / file_name
    if path.is_file():
        print(f"skipping file '{file_name}' as it already exists", flush=True)
        return
    # make the request
    session = requests.Session()
    response = session.get(_direct_url(url), stream=True)
    if response.status_code >= 400:
        raise RuntimeError(f"http error: {response.status_code}")
    # get a confirmation token for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(_direct_url(url), params={'confirm': value},
                                   stream=True)
            break
    # save on disk
    CHUNK_SIZE = 32768
    with open(path, "wb") as f:
        print(f"{file_name}: 0. kB", end="", flush=True)
        for i, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            f.write(chunk)
            n_bytes = (i+1)*CHUNK_SIZE / 8.
            for j, unit in enumerate(['Bytes', 'kB', 'MB', 'GB', 'TB']):
                if n_bytes < 1024**(j+1):
                    break
            progress = n_bytes / 10024**j
            print(f"\r{file_name}: {progress:.1f} {unit}"+" "*10,
                  end="", flush=True)
    print()


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
    return f"https://docs.google.com/uc?export=download&id={id}"
