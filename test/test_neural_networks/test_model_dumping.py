import sys
import pandas as pd
import numpy as np
import pygmalion.neural_networks as nn

rtol, atol = 1.0E-4, 0


def test_DenseClassifier():
    inputs = ["x1", "x2", "x3", "x4", "x5"]
    classes = ["1", "2", "3"]
    layers = [{"channels": 8}, {"channels": 16}, {"channels": 8}]
    m1 = nn.DenseClassifier(inputs, classes, layers)
    m2 = nn.DenseClassifier.from_dump(m1.dump)
    x = pd.DataFrame(data=np.random.uniform(size=(100, len(inputs))),
                     columns=inputs)
    assert all([p1 == p2 for p1, p2 in zip(m1(x), m2(x))])


def test_DenseRegressor():
    inputs = ["x1", "x2", "x3", "x4", "x5"]
    layers = [{"channels": 8}, {"channels": 16}, {"channels": 8}]
    m1 = nn.DenseRegressor(inputs, layers)
    m2 = nn.DenseRegressor.from_dump(m1.dump)
    x = pd.DataFrame(data=np.random.uniform(size=(100, len(inputs))),
                     columns=inputs)
    assert np.allclose(m1(x), m2(x), rtol=rtol, atol=atol)


def test_ImageClassifier():
    in_features = 3
    classes = ["1", "2", "3"]
    conv = [{"window": (3, 3), "channels": 4},
            {"window": (3, 3), "channels": 4},
            {"window": (3, 3), "channels": 4}]
    pooling = [(2, 2), (2, 2)]
    dense = [{"channels": 4}]
    m1 = nn.ImageClassifier(in_features, classes,
                            convolutions=conv,
                            pooling=pooling,
                            dense=dense,)
    m2 = nn.ImageClassifier.from_dump(m1.dump)
    x = np.random.randint(0, 255, size=(100, 8, 8, 3), dtype="uint8")
    assert all([p1 == p2 for p1, p2 in zip(m1(x), m2(x))])


# def test_ObjectDetector():
#     in_features = 3
#     boxes_per_cell = 3
#     classes = ["1", "2", "3"]
#     down = [{"window": (3, 3), "channels": 4},
#             {"window": (3, 3), "channels": 4},
#             {"window": (3, 3), "channels": 4}]
#     pooling = [(2, 2), (2, 2), (2, 2)]
#     dense = [{"window": (3, 3), "channels": 4},
#              {"window": (3, 3), "channels": 4}]
#     m1 = nn.ObjectDetector(in_features, classes,
#                            boxes_per_cell,
#                            downsampling=down,
#                            pooling=pooling,
#                            dense=dense)
#     m2 = nn.ObjectDetector.from_dump(m1.dump)
#     x = np.random.randint(0, 255, size=(100, 8, 8, 3), dtype="uint8")
#     assert all([p1 == p2 for p1, p2 in zip(m1(x), m2(x))])


def test_SemanticSegmenter():
    in_features = 3
    colors = {"1": [255, 0, 0], "2": [0, 255, 0], "3": [0, 0, 255]}
    downward = [{"window": (3, 3), "channels": 4},
                {"window": (3, 3), "channels": 4},
                {"window": (3, 3), "channels": 4}]
    pooling = [(2, 2), (2, 2), (2, 2)]
    upward = [{"window": (3, 3), "channels": 4},
              {"window": (3, 3), "channels": 4},
              {"window": (3, 3), "channels": 4}]
    m1 = nn.SemanticSegmenter(in_features, colors,
                              downsampling=downward,
                              pooling=pooling,
                              upsampling=upward)
    m2 = nn.SemanticSegmenter.from_dump(m1.dump)
    x = np.random.randint(0, 255, size=(100, 8, 8, 3), dtype="uint8")
    assert np.allclose(m1(x), m2(x), rtol=rtol, atol=atol)


def test_Traductor():
    embedding_dim = 64
    lexicon_in = ["a", "b", "c", "d", "e"]
    lexicon_out = ["f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p"]
    sentence = "a b d a e c b e e a"
    projection_dim = 16
    n_heads = 4
    layers = [{"channels": 128}, {"channels": 128}]
    n_stages = 3
    m1 = nn.Traductor(embedding_dim, lexicon_in, lexicon_out, projection_dim,
                      n_heads, layers, n_stages)
    m2 = nn.Traductor.from_dump(m1.dump)
    assert m1(sentence) == m2(sentence)


if __name__ == "__main__":
    module = sys.modules[__name__]
    for attr in dir(module):
        if not attr.startswith("test_"):
            continue
        print(attr, end="")
        func = getattr(module, attr)
        try:
            func()
        except AssertionError:
            print(": Failed")
        else:
            print(": Passed")
