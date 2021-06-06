from ._download import download_all


def fashion_mnist(directory: str):
    """downloads the 'fashion MNIST' dataset in the given directory"""
    file_names = ["classes.txt", "test_images.npy", "test_labels.npy",
                  "train_images.npy", "train_labels.npy"]
    urls = ["https://drive.google.com/file/d/"
            "1hrQfvcIbpe48JUIS870Ubah0wAFJvcEI/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1wLuRtkRIb1z7A92V86CZxP4L80K2L7ip/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "11mb6eMSGbsgvuEpDzdITMcAPlsJP6Oxt/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1Z9Tir5UwuiY4paqos4wSBYnkPZJIorOw/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1MRIs2UhmfiT29NkPCv0qclc5KtOm5kuG/view?usp=sharing"]
    download_all(directory, "fashion-MNIST", file_names, urls)
