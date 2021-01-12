import sys
import IPython
import pathlib
import machine_learning as ml
import machine_learning.agnostic as agn
import pandas as pd
import numpy as np

data_path = pathlib.Path(__file__).parents[1] / "data"


def test_fit():
    """
    Montgomery, D. C., & Runger, G. C. (2010). Applied statistics and
    probability for engineers. John Wiley & Sons.
    Example 12-1
    """
    df = pd.read_csv(data_path / "WireBondData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2"]
    model = ml.LinearModel()
    model.fit(df[x], df[y])
    res = "y = 2.26379 + 2.74427*X1 + 0.0125278*X2"
    assert model.get_model_string(precision=6) == res


def test_weighted():
    df = pd.read_csv(data_path / "WireBondData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2"]
    model = ml.LinearModel()
    model.fit(df[x], df[y], weights=df["X4"])
    res = "y = 1.08305 + 2.92825*X1 + 0.0112528*X2"
    assert model.get_model_string(precision=6) == res


def test_without_bias():
    df = pd.read_csv(data_path / "WireBondData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2"]
    model = ml.LinearModel()
    model.fit(df[x], df[y], bias=False)
    res = "y = + 2.82327*X1 + 0.0161177*X2"
    assert model.get_model_string(precision=6) == res


def test_predict():
    df = pd.read_csv(data_path / "WireBondData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2"]
    model = ml.LinearModel()
    model.fit(df[x], df[y])
    X = df[x]
    Y_true = [8.38, 25.60, 33.95, 36.60, 27.91, 15.75, 12.45, 8.40, 28.21,
              27.98, 18.40, 37.46, 41.46, 12.26, 15.81, 18.25, 64.67, 12.34,
              36.47, 46.56, 47.06, 52.56, 56.31, 19.98, 21.00]
    Y_predicted = np.round(model.predict(X), decimals=2)
    assert (Y_true == Y_predicted).all()


def test_Ttest():
    """
    Montgomery, D. C., & Runger, G. C. (2010). Applied statistics and
    probability for engineers. John Wiley & Sons.
    Example 12-4
    """
    df = pd.read_csv(data_path / "WireBondData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2"]
    model = ml.LinearModel()
    p, T = model._T_test(df[x], df[y], "X2")
    assert (round(T, 3), round(p, 4)) == (4.477, 0.0002)


def test_Ftest():
    """
    Montgomery, D. C., & Runger, G. C. (2010). Applied statistics and
    probability for engineers. John Wiley & Sons.
    Example 12-6
    """
    df = pd.read_csv(data_path / "WireBondData.csv", sep=";", decimal=",")
    y = "Y"
    x_full = ["X1", "X2", "X3", "X4"]
    x_sub = ["X1", "X2"]
    model = ml.LinearModel()
    p, F = model._partial_F_test(df, df[y], x_full, x_sub)
    assert (round(F, 3), round(p, 3)) == (4.047, 0.033)


def test_forward():
    df = pd.read_csv(data_path / "WineQualityData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2", "X3", "X4", "X5"]
    model = ml.LinearModel()
    model.fit_forward(df[x], df[y], p=0.1, verbose=False)
    model_str = model.get_model_string(precision=5)
    assert model_str == "y = 6.4672 + 1.1997*X4 - 0.60232*X5 + 0.58012*X2"


def test_backward():
    """
    Montgomery, D. C., & Runger, G. C. (2010). Applied statistics and
    probability for engineers. John Wiley & Sons.
    Example 12-15
    """
    df = pd.read_csv(data_path / "WineQualityData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2", "X3", "X4", "X5"]
    model = ml.LinearModel()
    model.fit_backward(df[x], df[y], p=0.15, verbose=False)
    model_str = model.get_model_string(precision=5)
    assert model_str == "y = 6.4672 + 0.58012*X2 + 1.1997*X4 - 0.60232*X5"


def test_stepwise():
    """
    Montgomery, D. C., & Runger, G. C. (2010). Applied statistics and
    probability for engineers. John Wiley & Sons.
    Example 12-15
    """
    df = pd.read_csv(data_path / "WineQualityData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2", "X3", "X4", "X5"]
    model = ml.LinearModel()
    model.fit_stepwise(df[x], df[y], p=0.1, verbose=False)
    model_str = model.get_model_string(precision=5)
    assert model_str == "y = 6.4672 + 1.1997*X4 - 0.60232*X5 + 0.58012*X2"


def test_confidence():
    """
    Montgomery, D. C., & Runger, G. C. (2010). Applied statistics and
    probability for engineers. John Wiley & Sons.
    Example 12-9
    """
    df = pd.read_csv(data_path / "WireBondData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2"]
    model = ml.LinearModel()
    model.fit(df[x], df[y])
    x = pd.DataFrame(data=[[8., 275.]], columns=["X1", "X2"])
    dy = model.uncertainty(x, alpha=0.05)
    assert round(dy[0], 2) == round(27.66-22.81, 2)


def test_dump():
    df = pd.read_csv(data_path / "WineQualityData.csv", sep=";", decimal=",")
    y = "Y"
    x = ["X1", "X2"]
    model = ml.LinearModel()
    model.fit(df[x], df[y])
    loaded = agn.LinearModel(model.dump)
    eval_model = model.predict(df)
    eval_loaded = loaded(df)
    assert np.isclose(eval_model, eval_loaded, atol=0.).all()


if __name__ == "__main__":
    module = sys.modules[__name__]
    for attr in dir(module):
        if "test_" not in attr:
            continue
        func = getattr(module, attr)
        print(attr)
        func()
    IPython.embed()
