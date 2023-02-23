import pytest

from skreducedmodel.reducedbasis import ReducedBasis
from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation
from skreducedmodel.mksurrogate import mksurrogate, InputError


def test_error_non_trained_rb_only():
    with pytest.raises(TypeError):
        rb = ReducedBasis()
        mksurrogate(instance=rb)


def test_wrong_parameter():
    # debe devolver: InputError("something must not be given. It
    # is not a parameter to use.)"
    with pytest.raises(InputError):
        mksurrogate(something=2)

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

def test_rb_given_all_ok(ts_train, parameters_train, times):
    rb = ReducedBasis()
    rb.fit(
        parameters=parameters_train[:, 0],
        training_set=ts_train,
        physical_points=times,
    )
    surrogate = mksurrogate(rb)
    surrogate.predict(1)

def test_eim_given_all_ok(ts_train, parameters_train, times):
    rb = ReducedBasis()
    rb.fit(
        parameters=parameters_train[:, 0],
        training_set=ts_train,
        physical_points=times,
    )
    eim = EmpiricalInterpolation(rb)
    surrogate = mksurrogate(eim)
    surrogate.predict(1)

def test_eim_with_no_training_data_fails():
    with pytest.raises(InputError):
        rb = ReducedBasis()
        eim = EmpiricalInterpolation(rb)
        surrogate = mksurrogate(eim)
