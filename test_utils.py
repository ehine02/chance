import numpy as np
import pytest
import utils

import pandas as pd


def test_perform_oversampling():
    unbalanced = pd.DataFrame({'x': [1, 2, 3, 4, 5],
                               'y': [5, 4, 3, 2, 1],
                               'chance': [True, False, False, False, False]})

    balanced = utils.perform_oversampling(unbalanced)

    assert balanced.chance.shape[0] == 8

    assert balanced.loc[balanced.chance == True].shape[0] == 4

    assert balanced.loc[balanced.chance == False].shape[0] == 4


def test_euclidean_distance():
    assert utils.euclidean_distance((1, 1), (2, 2)) == pytest.approx(1.4142, 0.01)

    assert utils.euclidean_distance((1, 1), (1, 1)) == pytest.approx(0.0, 0.01)

    assert utils.euclidean_distance((0, 0), (0, 0)) == pytest.approx(0.0, 0.01)


def test_list_if_not_nan():
    assert type(utils.list_if_not_nan('[1, 2, 3, 4, 5]')) == list

    assert utils.list_if_not_nan('[1, 2, 3, 4, 5]') == [1, 2, 3, 4, 5]

    assert utils.list_if_not_nan(np.nan) is np.nan


def test_split_location():
    assert utils.split_location(np.nan) == (np.nan, np.nan)

    assert utils.split_location([119.9, 39.9]) == (120, 40)
