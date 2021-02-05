import requests
import pathlib
from typing import List


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
        raise NotADirectoryError(f"The path '{directory}' is not an existing directory")
    path = directory / file_name
    if path.is_file():
        print(f"skipping file '{file_name}' as it already exists", flush=True)
        return
    # make the request
    session = requests.Session()
    response = session.get(direct_url(url), stream=True)
    if response.status_code >= 400:
        raise RuntimeError(f"http error: {response.status_code}")
    # get a confirmation token for large files
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(direct_url(url), params={'confirm': value},
                                   stream=True)
            break
    # save on disk
    CHUNK_SIZE = 32768
    with open(path, "wb") as f:
        print(f"{file_name}: 0. kB", end="", flush=True)
        for i, chunk in enumerate(response.iter_content(CHUNK_SIZE)):
            f.write(chunk)
            n_bytes = (i+1)*CHUNK_SIZE / 8.
            for j, unit in enumerate(['bytes', 'kB', 'MB', 'GB', 'TB']):
                if n_bytes < 10**(3*(j+1)):
                    break
            progress = n_bytes / 10**(3*j)
            print(f"\r{file_name}: {progress:.2g} {unit}"+" "*10,
                  end="", flush=True)
    print()


def direct_url(url: str) -> str:
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


def downloads(directory: str, folder_name: str,
              file_names: List[str], urls: List[str]):
    """
    Download a series of files in a new folder.

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


def boston_housing(directory: str):
    """downloads 'boston_housing.csv' in the given directory"""
    download(directory, "boston_housing.csv", "https://drive.google.com/file/d/1fTWYixdKF4tWyhD3V-qCDSZmReN_6LzP/view?usp=sharing")


def iris(directory: str):
    """downloads 'iris.csv' in the given directory"""
    download(directory, "iris.csv", "https://drive.google.com/file/d/1S1AHfTBtnW1SxsMskRmUcnoDehMbCj0R/view?usp=sharing")


def titanic(directory: str):
    """downloads 'titanic.csv' in the given directory"""
    download(directory, "titanic.csv", "https://drive.google.com/file/d/1LYjbHW3wyJSMzGMMCmaOFNA_RIKqxRoI/view?usp=sharing")


def fashion_mnist(directory: str):
    """downloads the 'fashion MNIST' dataset in the given directory"""
    file_names = ["classes.txt", "test_images.npy", "test_labels.npy",
                  "train_images.npy", "train_labels.npy"]
    urls = ["https://drive.google.com/file/d/1hrQfvcIbpe48JUIS870Ubah0wAFJvcEI/view?usp=sharing",
            "https://drive.google.com/file/d/1wLuRtkRIb1z7A92V86CZxP4L80K2L7ip/view?usp=sharing",
            "https://drive.google.com/file/d/11mb6eMSGbsgvuEpDzdITMcAPlsJP6Oxt/view?usp=sharing",
            "https://drive.google.com/file/d/1Z9Tir5UwuiY4paqos4wSBYnkPZJIorOw/view?usp=sharing",
            "https://drive.google.com/file/d/1MRIs2UhmfiT29NkPCv0qclc5KtOm5kuG/view?usp=sharing"]
    downloads(directory, "fashion-MNIST", file_names, urls)


def cityscapes(directory: str):
    """downloads the 'cityscapes' dataset in the given directory"""
    file_names = ["classes.json", "test_images.npy", "test_segmented.npy",
                  "train_images.npy", "train_segmented.npy"]
    urls = ["https://drive.google.com/file/d/1T89BJ0U9AtPXjfON7Du9Sbd1BniM2Q_T/view?usp=sharing",
            "https://drive.google.com/file/d/10Yfsfc_V0SMXh4cAlXHCnwJHtaWUvrK-/view?usp=sharing",
            "https://drive.google.com/file/d/1Gu9-hUZUhNYHsRc_PqwT-AupCuCBE5UU/view?usp=sharing",
            "https://drive.google.com/file/d/13DzRX1yUlDW8oXNiWEa-Bu02_myttlem/view?usp=sharing",
            "https://drive.google.com/file/d/1BmhXfQraa37rwsnvcub0x-FjPhir6RsJ/view?usp=sharing"]
    downloads(directory, "cityscapes", file_names, urls)
