# import numpy as np

from skreducedmodel.surrogate import Surrogate

from arby import ReducedOrderModel as ROM

import arby

import numpy as np

from skreducedmodel.reducedbasis import error

from scipy.special import jv as BesselJ


def test_rom_rb_interface(rom_parameters):
    """Test API consistency."""

    bessel = ROM(
        rom_parameters["training_set"],
        rom_parameters["physical_points"],
        rom_parameters["parameter_points"],
        greedy_tol=1e-14
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

    rom = Surrogate(greedy_tol=1e-14, lmax = 0)

    rom.fit(training_set = rom_parameters["training_set"],
            parameters = rom_parameters["parameter_points"],
            physical_points = rom_parameters["physical_points"],
        )      

    leaf = rom.base.tree
    rom_errors = leaf.errors
    rom_projection_matrix = leaf.projection_matrix
    rom_greedy_indices = leaf.indices

    assert bessel.basis_.data.shape == leaf.basis.shape
    assert (rom_errors == errors).all()
    assert (rom_projection_matrix == projection_matrix).all()
    assert rom_greedy_indices == greedy_indices

def test_predictions_dataset_real_dataset(ts_train,
                                          ts_test,
                                          parameters_train,
                                          parameters_test,
                                          times):
    
    ts_train_real = np.real(ts_train)
    ts_test_real = np.real(ts_test)

    #orden de datos para splines tener en cuenta
    f_model = ROM(training_set=ts_train_real[np.argsort(parameters_train[:,0])],
                physical_points=times,
                parameter_points=np.sort(parameters_train[:,0])
                )

    errors_f_model = []
    for h, q in zip(ts_test_real, parameters_test):
        h_rom = f_model.surrogate(q[0])
        errors_f_model.append(error(h, h_rom, times))

    rom = Surrogate(lmax = 0,
                    nmax = np.inf,
                    normalize = False
                )

    rom.fit(training_set = ts_train_real,
                parameters = parameters_train[:,0],
                physical_points = times
        )
    errors_rom = []
    for h, q in zip(ts_test_real, parameters_test):
        h_rom = rom.predict(q[0])
        errors_rom.append(error(h, h_rom, times))

    assert errors_f_model == errors_rom


def test_surrogate_accuracy():
    """Test surrogate accuracy for Bessel functions."""

    parameter_points = np.linspace(1, 10, num=101)
    nu_validation = np.linspace(1, 10, num=1001)
    physical_points = np.linspace(0, 1, 1001)

    # build training space
    training = np.array(
        [BesselJ(nn, physical_points) for nn in parameter_points]
    )

    # build reduced basis
    rom = Surrogate(greedy_tol=1e-15)
    rom.fit(training_set = training,
            physical_points = physical_points,
            parameters = parameter_points
            )

    bessel_test = [BesselJ(nn, physical_points) for nn in nu_validation]
    bessel_surrogate = [rom.predict(nn) for nn in nu_validation]

    np.testing.assert_allclose(
        bessel_test, bessel_surrogate, rtol=1e-5, atol=1e-5
    )
