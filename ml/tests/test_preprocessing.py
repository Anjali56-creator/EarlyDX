"""Preprocessing building blocks."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone

from ml.common.preprocessing import ZeroToNaN, build_preprocessor


def test_zero_to_nan_only_targets_named_columns():
    df = pd.DataFrame({"a": [0, 1, 2], "b": [0, 0, 5]})
    out = ZeroToNaN(["b"]).fit_transform(df)
    assert out["a"].tolist() == [0, 1, 2]
    assert np.isnan(out["b"][0]) and np.isnan(out["b"][1])
    assert out["b"][2] == 5


def test_zero_to_nan_is_cloneable():
    t = ZeroToNaN(["b"])
    clone(t)  # must not raise (sklearn param contract)


def test_preprocessor_imputes_and_scales_on_fit_only():
    train = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    pre = build_preprocessor(["x", "y"], scale=True)
    pre.fit(train)
    # unseen row with a NaN -> imputed with the TRAIN median, then scaled
    test = pd.DataFrame({"x": [np.nan], "y": [25.0]})
    out = pre.transform(test)
    assert out.shape == (1, 2)
    assert not np.isnan(out).any()
