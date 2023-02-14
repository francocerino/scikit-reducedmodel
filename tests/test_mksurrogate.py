import numpy as np

import pytest

from skreducedmodel.reducedbasis import ReducedBasis

from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

from skreducedmodel.mksurrogate import mksurrogate, InputError


def test_error_non_trained_rb_only():
    with pytest.raises(TypeError):
        rb = ReducedBasis()
        mksurrogate(instance=rb)


def test_instance_class():
    with pytest.raises(InputError):
        eim = EmpiricalInterpolation()
        mksurrogate(instance=2)


def test_eim_and_train_data_given(ts_train, parameters_train, times):
    with pytest.raises(InputError):
        eim = EmpiricalInterpolation()
        mksurrogate(
            parameters=parameters_train[:, 0],
            training_set=ts_train,
            physical_points=times,
            instance=eim,
        )


def test_rb_and_train_data_given(ts_train, parameters_train, times):
    with pytest.raises(InputError):
        rb = ReducedBasis()
        mksurrogate(
            parameters=parameters_train[:, 0],
            training_set=ts_train,
            physical_points=times,
            instance=rb,
        )


def test_train_with_only_train_data(ts_train, parameters_train, times):
    surrogate = mksurrogate(
        parameters=parameters_train[:, 0],
        training_set=ts_train,
        physical_points=times,
    )
    assert surrogate.is_trained


def test_train_with_train_data_and_hyperparameter(
    ts_train, parameters_train, times
):
    surrogate = mksurrogate(
        lmax=1,
        parameters=parameters_train[:, 0],
        training_set=ts_train,
        physical_points=times,
    )
    assert surrogate.is_trained


def test_rb_given_and_hyperparameters_of_it(ts_train, parameters_train, times):
    with pytest.raises(InputError):
        rb = ReducedBasis()
        rb.fit(
            parameters=parameters_train[:, 0],
            training_set=ts_train,
            physical_points=times,
        )
        mksurrogate(rb, lmax=1)
