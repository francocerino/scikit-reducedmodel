"""Fixtures for pytest."""

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
WAVES_PATH = PATH / "../examples/waveforms"


PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

BESSEL_PATH = PATH / "bessel"

# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def ts_train():
    """
    Training set.

    Complex gravitational waves, associated to a 1d parameter space.
    """
    path = WAVES_PATH / "ts_train_1d-seed_eq_1.npy"
    return np.load(path)


@pytest.fixture
def ts_test():
    """
    Test set.

    Complex gravitational waves, associated to a 1d parameter space.
    """
    path = WAVES_PATH / "ts_test_1d-seed_eq_1.npy"
    return np.load(path)


@pytest.fixture
def parameters_test():
    """Parameters associated to 'ts_test', which has a 1d parameter space."""
    path = WAVES_PATH / "q_test_1d-seed_eq_1.npy"
    return np.load(path)


@pytest.fixture
def parameters_train():
    """Parameters associated to 'ts_train', which has a 1d parameter space."""
    path = WAVES_PATH / "q_train_1d-seed_eq_1.npy"
    return np.load(path)


@pytest.fixture
def times():
    """Physical points of 'ts_train' and 'ts_test' waveforms."""
    path = WAVES_PATH / "times_1d-seed_eq_1.npy"
    return np.load(path)


@pytest.fixture
def training_set():
    """Training set for Bessel example."""
    path = BESSEL_PATH / "bessel_training.txt"
    return np.loadtxt(path)


@pytest.fixture
def physical_points():
    """Physical points for Bessel example."""
    path = BESSEL_PATH / "physical_points.txt"
    return np.loadtxt(path)


@pytest.fixture
def rom_parameters(training_set, physical_points):
    """ROM inputs from Bessel exapmple."""
    parameter_points = np.linspace(0, 10, len(training_set))
    params = {
        "training_set": training_set,
        "physical_points": physical_points,
        "parameter_points": parameter_points,
    }
    return params
