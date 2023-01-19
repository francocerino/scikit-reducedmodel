import os
import pathlib

import numpy as np

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

WAVES_PATH = PATH / "waveforms"

@pytest.fixture
def ts_test():
    path = WAVES_PATH / 'ts_test_1d-seed_eq_1.npy'
    return np.load(path)

@pytest.fixture
def times():
    path = WAVES_PATH / 'times_1d-seed_eq_1.npy'
    return np.load(path)
