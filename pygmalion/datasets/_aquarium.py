from ._download import download_all


def aquarium(directory: str):
    """downloads the 'roboflow' aquarium adatset in the given directory"""
    file_names = ["class_fractions.json",
                  "test_bounding_boxes.json",
                  "test_images.npy",
                  "train_bounding_boxes.json",
                  "train_images.npy",
                  "val_bounding_boxes.json",
                  "val_images.npy"]
    urls = ["https://drive.google.com/file/d/1IHKPYSciPV6T20-dtJn3BR_jpMFAAtZ3/view?usp=sharing",
            "https://drive.google.com/file/d/1GZ8xYMGhiyAufb6iCOZi_Y44aq80iTQj/view?usp=sharing",
            "https://drive.google.com/file/d/1eFLyp48QHEbkwSxwLT-hFIzFCyvgFQm5/view?usp=sharing",
            "https://drive.google.com/file/d/1hkE8J2NwCUmF55qgdmUSR3U3CTsd2di8/view?usp=sharing",
            "https://drive.google.com/file/d/1RuqoyX2ZQN4AjuVo12tEyweS7zlkTpXY/view?usp=sharing",
            "https://drive.google.com/file/d/1cKefoCsgQv7DUSrmF_Mg7pItwzOTSQMN/view?usp=sharing",
            "https://drive.google.com/file/d/1xTXiFWKt23d6JYvSoOVtY6EPAE4ZHiju/view?usp=sharing"]
    download_all(directory, "aquarium", file_names, urls)
