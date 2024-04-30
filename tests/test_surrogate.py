import numpy as np

from skreducedmodel.reducedbasis import ReducedBasis, _error
from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation
from skreducedmodel.surrogate import Surrogate

from arby import ReducedOrderModel as ROM

from scipy.special import jv as BesselJ


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
    assert eim == bessel.basis_.eim_

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
    assert (rom_errors == errors).all()
    assert (rom_projection_matrix == projection_matrix).all()
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
