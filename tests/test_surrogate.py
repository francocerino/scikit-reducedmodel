import numpy as np
import sys

from skreducedmodel.reducedbasis import ReducedBasis, _error
from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation
from skreducedmodel.surrogate import Surrogate

from arby import ReducedOrderModel as ROM

from sklearn.tree import DecisionTreeRegressor

from scipy.special import jv as BesselJ

import gwtools


def test_rom_rb_interface(rom_parameters):
    """Test API consistency."""

    bessel = ROM(
        rom_parameters["training_set"],
        rom_parameters["physical_points"],
        rom_parameters["parameter_points"].reshape(-1, 1),
        greedy_tol=1e-14,
    )
    basis = bessel.basis_.data
    errors = bessel.greedy_errors_
    projection_matrix = bessel.projection_matrix_
    greedy_indices = bessel.greedy_indices_
    eim = bessel.eim_

    assert len(basis) == 10
    assert len(errors) == 10
    assert len(projection_matrix) == 101
    assert len(greedy_indices) == 10
    assert np.allclose(eim.interpolant, bessel.basis_.eim_.interpolant)

    rb = ReducedBasis(greedy_tol=1e-14)
    rb.fit(
        training_set=rom_parameters["training_set"],
        parameters=rom_parameters["parameter_points"].reshape(-1, 1),
        physical_points=rom_parameters["physical_points"],
    )
    eim = EmpiricalInterpolation(reduced_basis=rb)
    eim.fit()
    rom = Surrogate(eim=eim)
    rom.fit()

    leaf = rom.eim.reduced_basis.tree
    rom_errors = leaf.errors
    rom_projection_matrix = leaf.projection_matrix
    rom_greedy_indices = leaf.indices

    assert bessel.basis_.data.shape == leaf.basis.shape
    assert np.allclose(rom_errors, errors)
    assert np.allclose(rom_projection_matrix, projection_matrix)
    assert rom_greedy_indices == greedy_indices


"""
def test_predictions_dataset_real_dataset(
    ts_train, ts_test, parameters_train, parameters_test, times
):

    ts_train_real = np.real(ts_train)
    ts_test_real = np.real(ts_test)
    
    # comparacion con arby
    # orden de datos para splines tener en cuenta
    f_model = ROM(
        training_set=ts_train_real[np.argsort(parameters_train[:, 0])],
        physical_points=times,
        parameter_points=np.sort(parameters_train[:, 0]),
    )

    errors_f_model = []
    for h, q in zip(ts_test_real, parameters_test):
        h_rom = f_model.surrogate(q[0])
        errors_f_model.append(_error(h, h_rom, times))

    # skrm

    #parameters_train = parameters_train[:,0].reshape(-1,1)
    #parameters_test = parameters_test[:,0].reshape(-1,1)

    rb = ReducedBasis()
    rb.fit(
        training_set=ts_train_real,
        parameters=parameters_train[:, 0],
        physical_points=times,
    )
    eim = EmpiricalInterpolation(reduced_basis=rb)
    eim.fit()
    rom = Surrogate(eim=eim)
    rom.fit()

    errors_rom = []
    for h, q in zip(ts_test_real, parameters_test):
        h_rom = rom.predict(q).reshape(-1)
        errors_rom.append(_error(h, h_rom, times))

    assert errors_f_model == errors_rom
"""


def test_surrogate_accuracy():
    """Test surrogate accuracy for Bessel functions."""

    physical_points = np.linspace(0, 1, 1000)

    parameters_train = np.linspace(1, 10, num=101).reshape(-1, 1)
    parameters_validation = np.linspace(1, 10, num=1001)

    train = np.array([BesselJ(nn, physical_points) for nn in parameters_train])
    test = np.array(
        [BesselJ(nn, physical_points) for nn in parameters_validation]
    )

    rb = ReducedBasis()
    rb.fit(train, parameters_train, physical_points)
    eim = EmpiricalInterpolation(reduced_basis=rb)
    eim.fit()
    rom = Surrogate(eim=eim)
    rom.fit()

    bessel_surrogate = np.array(
        [rom.predict(nn) for nn in parameters_validation]
    )

    np.testing.assert_allclose(
        test, bessel_surrogate.reshape(1001, 1000), rtol=1e-4, atol=1e-5
    )


def test_consistency_complex_and_real_cases(
    ts_train, ts_test, parameters_train, parameters_test, times
):
    ts_train_abs = np.abs(ts_train)
    ts_test_abs = np.abs(ts_test)
    parameters = parameters_train[:, 0].reshape(-1, 1)
    rb = ReducedBasis()
    rb.fit(
        training_set=ts_train_abs,
        parameters=parameters,
        physical_points=times,
    )
    eim = EmpiricalInterpolation(rb)
    eim.fit()
    surrogate = Surrogate(eim)
    surrogate.fit()

    rb_complex = ReducedBasis()
    rb_complex.fit(
        training_set=ts_train_abs + 1j * 1e-16,
        parameters=parameters,
        physical_points=times,
    )
    eim_complex = EmpiricalInterpolation(rb_complex)
    eim_complex.fit()
    surrogate_complex = Surrogate(eim_complex)
    surrogate_complex.fit()

    assert surrogate_complex.eim.reduced_basis.complex_dataset
    assert not surrogate.eim.reduced_basis.complex_dataset

    errors_rom = []
    errors_rom_complex = []
    for h, q in zip(ts_test_abs, parameters_test):
        q = q[:1]
        h_rom = surrogate.predict(q).T
        h_rom_complex = surrogate_complex.predict(q).T

        errors_rom.append(_error(h, h_rom, times))
        errors_rom_complex.append(_error(h, h_rom_complex, times))

    assert np.allclose(errors_rom_complex, errors_rom)


def test_no_errors_in_prediction_with_a_different_regression_algorithm(
    ts_train, parameters_train, parameters_test, times
):
    ts_train_amp = np.array([gwtools.amp(wf) for wf in ts_train])

    rb_amp_rforest = ReducedBasis(greedy_tol=1e-14)
    rb_amp_rforest.fit(ts_train_amp, parameters_train[:, 0], times)
    eim_amp_rforest = EmpiricalInterpolation(reduced_basis=rb_amp_rforest)
    eim_amp_rforest.fit()
    rom_amp_rforest = Surrogate(
        eim=eim_amp_rforest, regression_model=DecisionTreeRegressor
    )
    rom_amp_rforest.fit()

    surrogates_amp_rforest = np.array(
        [rom_amp_rforest.predict(q) for q in parameters_test[:, 0]]
    )
    surrogates_amp_rforest = surrogates_amp_rforest.reshape(100, -1)


def test_no_errors_in_prediction_with_new_hyperparameters_for_gpr(
    ts_train, parameters_train, parameters_test, times
):
    ts_train_amp = np.array([gwtools.amp(wf) for wf in ts_train])

    regression_hyperparameters = {"alpha": 1e-14, "n_restarts_optimizer": 100}
    ts_train_amp = np.array([gwtools.amp(wf) for wf in ts_train])

    rb_amp = ReducedBasis(greedy_tol=1e-14)
    rb_amp.fit(ts_train_amp, parameters_train[:, 0], times)
    eim_amp = EmpiricalInterpolation(reduced_basis=rb_amp)
    eim_amp.fit()
    rom_amp = Surrogate(
        eim=eim_amp, regression_hyperparameters=regression_hyperparameters
    )
    rom_amp.fit()

    surrogates_amp = np.array(
        [rom_amp.predict(q) for q in parameters_test[:, 0]]
    )
    surrogates_amp = surrogates_amp.reshape(100, -1)


# resultado de errores menores a arby
def test_improvement_with_gaussian_process(
    ts_train, ts_test, parameters_train, parameters_test, times
):

    ts_train_amp = np.array([gwtools.amp(wf) for wf in ts_train])
    ts_test_amp = np.array([gwtools.amp(wf) for wf in ts_test])

    arby_surr = ROM(
        ts_train_amp[np.argsort(parameters_train[:, 0])],
        times,
        np.sort(parameters_train[:, 0]),
        greedy_tol=1e-14,
    )
    arby_surr = arby_surr.surrogate

    surrogates_arby = np.array([arby_surr(q) for q in parameters_test[:, 0]])

    regression_hyperparameters = {"alpha": 1e-14, "n_restarts_optimizer": 100}
    rb = ReducedBasis(greedy_tol=1e-14)
    rb.fit(
        ts_train_amp[np.argsort(parameters_train[:, 0])],
        np.sort(parameters_train[:, 0]),
        times,
    )
    eim = EmpiricalInterpolation(reduced_basis=rb)
    eim.fit()
    rom_amp = Surrogate(
        eim=eim, regression_hyperparameters=regression_hyperparameters
    )
    rom_amp.fit()

    norm = rom_amp.eim.reduced_basis.tree.integration.norm

    surrogates_amp = np.array(
        [rom_amp.predict(q) for q in parameters_test[:, 0]]
    )
    surrogates_amp = surrogates_amp.reshape(100, -1)

    errors_arby = []
    for h_pred, h_test in zip(surrogates_arby, ts_test_amp):
        errors_arby.append(norm(h_test - h_pred.reshape(-1)) / norm(h_test))

    errors_skrm_amp = []
    for h_pred, h_test in zip(surrogates_amp, ts_test_amp):
        errors_skrm_amp.append(
            norm(h_test - h_pred.reshape(-1)) / norm(h_test)
        )

    assert np.max(errors_skrm_amp) < np.max(errors_arby)


# testear caso de output de solo regresiones
