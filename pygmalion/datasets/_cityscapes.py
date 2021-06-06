from ._download import download_all


def cityscapes(directory: str):
    """downloads the modified 'cityscapes' dataset in the given directory"""
    file_names = ["class_fractions.json", "classes.json", "test_images.npy",
                  "test_segmented.npy", "train_images.npy",
                  "train_segmented.npy"]
    urls = ["https://drive.google.com/file/d/"
            "1WVUVJgoaovv-rHeIjztcY7J98oB3Hgmq/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1T89BJ0U9AtPXjfON7Du9Sbd1BniM2Q_T/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "10Yfsfc_V0SMXh4cAlXHCnwJHtaWUvrK-/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1Gu9-hUZUhNYHsRc_PqwT-AupCuCBE5UU/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "13DzRX1yUlDW8oXNiWEa-Bu02_myttlem/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1BmhXfQraa37rwsnvcub0x-FjPhir6RsJ/view?usp=sharing"]
    download_all(directory, "cityscapes", file_names, urls)
