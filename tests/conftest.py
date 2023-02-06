# =============================================================================
# IMPORTS
# =============================================================================


import os
import pathlib

import numpy as np

import pytest


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
WAVES_PATH = PATH / "waveforms"

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def parameters_train():
    path = WAVES_PATH / 'q_train_1d-seed_eq_1.npy'
    return np.load(path)

@pytest.fixture
def parameters_test():
    path = WAVES_PATH / 'q_test_1d-seed_eq_1.npy'
    return np.load(path)

@pytest.fixture
def ts_train():
    path = WAVES_PATH / 'ts_train_1d-seed_eq_1.npy'
    return np.load(path)

@pytest.fixture
def ts_test():
    path = WAVES_PATH / 'ts_test_1d-seed_eq_1.npy'
    return np.load(path)

@pytest.fixture
def times():
    path = WAVES_PATH / 'times_1d-seed_eq_1.npy'
    return np.load(path)
