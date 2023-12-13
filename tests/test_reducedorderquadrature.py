import numpy as np

import pytest
from skreducedmodel.integrals import Integration
from skreducedmodel.reducedbasis import ReducedBasis
from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation
from skreducedmodel.reducedorderquadrature import ReducedOrderQuadrature


def test_input_reduced_basis_class():
    with pytest.raises(ValueError):
        rb = ReducedBasis()
        ReducedOrderQuadrature(rb)


def test_dim_weights_equal_number_eim_nodes(
    ts_train, parameters_train, times, ts_test
):
    # number of weights must be the same than the number of empirical nodes
    rb = ReducedBasis(
        greedy_tol=1e-16,
        lmax=1,
        normalize=True,
    )
    rb.fit(ts_train, parameters_train, times)
    eim = EmpiricalInterpolation(rb)
    eim.fit()
    roq = ReducedOrderQuadrature(eim)
    roq.fit(times=times, data=ts_test[0])

    parameters_test = np.array([1, 10])
    for leaf, parameter in zip(eim.reduced_basis.tree.leaves, parameters_test):
        len(leaf.nodes) == len(roq.predict(parameter))


def test_accuracy_linear_weigths(
    ts_train, parameters_train, parameters_test, times, ts_test
):
    q_train = parameters_train
    q_test = parameters_test
    hyperparameters = {
        "index_seed_global_rb": 0,
        "greedy_tol": 1e-12,
        "lmax": 7,
        "nmax": 10,
        "normalize": False,
        "integration_rule": "riemann",
    }

    integration = Integration(times, rule="riemann")

    linear_rb = ReducedBasis(**hyperparameters)

    linear_rb.fit(
        training_set=ts_train, parameters=q_train, physical_points=times
    )

    # build the empirical interpolator
    linear_eim = EmpiricalInterpolation(reduced_basis=linear_rb)
    linear_eim.fit()

    # loop para validar que todas las ondas tienen overlap bien aproximado

    tree = linear_eim.reduced_basis.tree

    for index_wave_new in range(len(ts_test)):
        # onda de la cual se van a inferir sus parametros
        wave_new = ts_test[index_wave_new]
        paramter_wave_new = q_test[index_wave_new]

        linear_roq = ReducedOrderQuadrature(linear_eim)
        linear_roq.fit(times, data=wave_new)

        for index_wave_train in range(len(ts_train)):
            train_wave = ts_train[index_wave_train]
            paramter_train_wave = q_train[index_wave_train]

            leaf = linear_eim.reduced_basis.search_leaf(
                parameters=paramter_train_wave, node=tree
            )
            print(leaf.name)
            assert leaf.is_leaf
            eim_train_wave = linear_eim.transform(
                q=paramter_train_wave, h=train_wave
            )
            eim_train_wave_at_eim_nodes = eim_train_wave[leaf.nodes]

            overlap_fiducial = integration.integral(
                train_wave * np.conjugate(wave_new)
            )

            w_weights = linear_roq.predict(paramter_train_wave)

            # overlap calculado a lo ROQ
            overlap_roq = np.dot(w_weights, eim_train_wave_at_eim_nodes)
            print(overlap_roq, overlap_fiducial)
            assert np.allclose(overlap_roq, overlap_fiducial)


def test_accuracy_quadratic_weigths(ts_train, parameters_train, times):
    q_train = parameters_train
    hyperparameters = {
        "index_seed_global_rb": 0,
        "greedy_tol": 1e-12,
        "lmax": 7,
        "nmax": 10,
        "normalize": False,
        "integration_rule": "riemann",
    }

    integration = Integration(times, rule="riemann")

    quadratic_ts_train = np.real(ts_train * np.conjugate(ts_train))

    # quadratic roq
    quadratic_rb = ReducedBasis(**hyperparameters)

    quadratic_rb.fit(
        training_set=quadratic_ts_train,
        parameters=q_train,
        physical_points=times,
    )

    # we built the empirical interpolator
    quadratic_eim = EmpiricalInterpolation(reduced_basis=quadratic_rb)
    quadratic_eim.fit()

    quadratic_roq = ReducedOrderQuadrature(quadratic_eim)
    quadratic_roq.fit(
        times=times, quadratic_weights=True
    )  # revisar que esta bien ingresado el input
    # capaz conviene cambiar la forma de ingresar

    tree = quadratic_eim.reduced_basis.tree

    for index_wave_train in range(len(ts_train)):
        train_wave = ts_train[index_wave_train]
        paramter_train_wave = q_train[index_wave_train]

        leaf = quadratic_eim.reduced_basis.search_leaf(
            parameters=paramter_train_wave, node=tree
        )
        # leaf.nodes # ver que el nodo elegido está bien

        eim_train_wave = quadratic_eim.transform(
            q=paramter_train_wave, h=train_wave
        )
        eim_train_wave_at_eim_nodes = eim_train_wave[leaf.nodes]

        h_new_norm_fiducial = np.real(
            integration.integral(
                np.real(train_wave * np.conjugate(train_wave))
            )
        )

        # quadratic_bj = quadratic_roq.predict(2)
        # = np.real(integration.integral(quadratic_bj))

        c_weights = quadratic_roq.predict(2)

        h_new_norm_roq = np.real(
            np.dot(
                c_weights,
                eim_train_wave_at_eim_nodes
                * np.conjugate(eim_train_wave_at_eim_nodes),
            )
        )
        assert np.allclose(h_new_norm_roq, h_new_norm_fiducial)
