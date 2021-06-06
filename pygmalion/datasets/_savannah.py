from ._download import download_all


def savannah(directory: str):
    """downloads the 'savannah' dataset in the given directory"""
    file_names = ["class_fractions.json",
                  "test_bounding_boxes.json",
                  "test_images.npy",
                  "train_bounding_boxes.json",
                  "train_images.npy",
                  "val_bounding_boxes.json",
                  "val_images.npy"]
    urls = ["https://drive.google.com/file/d/"
            "1_uOo0lcjP_Oop2B4l3wqg-8ffv9KpEkw/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1m9zYDZHqgMAtZjLPD71p16u_45Z_jeRg/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1tGhPMEBjHKMSlD8osDeoINdNisWfUlwf/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1mq6brJeDThNMAX1l04_IjWy_hMO57Z6H/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1WF5A9xdgXoq092kI9w0dLZm1sTbSNFHE/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1ByMZlJ-_k6L8OnPjNXe97g_MGE9PkKlw/view?usp=sharing",
            "https://drive.google.com/file/d/"
            "1Sd5JO6oBmlzQzydoaAGWR_ZXxAXWqFKq/view?usp=sharing"]
    download_all(directory, "savannah", file_names, urls)
