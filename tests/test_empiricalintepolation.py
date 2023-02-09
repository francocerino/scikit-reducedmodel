from scipy.integrate import odeint

from skreducedmodel.reducedbasis import ReducedBasis

from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

from skreducedmodel.empiricalinterpolation import InputDataError

import numpy as np

import pytest


def pend(y, t, b, λ):
    θ, ω = y
    dydt = [ω, -b * ω - λ * np.sin(θ)]

    return dydt


def test_EmpiricalInterpolationit():

    b = 0.2
    y0 = [np.pi / 2, 0.0]

    param = np.linspace(1, 5, 101)
    times = np.linspace(0, 50, 1001)

    training = []
    for λ in param:
        sol = odeint(pend, y0, times, (b, λ))
        training.append(sol[:, 0])

    training_set = np.array(training)
    parameters = param
    physical_points = times
    nmax = 10

    model = ReducedBasis(
        index_seed_global_rb=0, greedy_tol=1e-10, lmax=0, normalize=False
    )

    model.fit(
        training_set=training_set,
        parameters=parameters,
        physical_points=physical_points,
    )

    ti = EmpiricalInterpolation(reduced_basis=model)
    ti.fit()

    assert ti.reduced_basis.tree.nodes[0] == 0
    assert ti.reduced_basis.tree.nodes[5] == 167
    assert ti.reduced_basis.tree.nodes[19] == 816


def test_interpolator(ts_train, parameters_train, times):
    """Test that interpolate method works as true projectors."""

    random = np.random.default_rng(seed=42)
    basis = ReducedBasis(
        index_seed_global_rb=0,
        greedy_tol=1e-12,
        lmax=0,
        nmax=np.inf,
        normalize=True,
    )

    basis.fit(
        training_set=ts_train,
        parameters=parameters_train,
        physical_points=times,
    )

    eim = EmpiricalInterpolation(basis)
    eim.fit()

    for _ in range(10):
        # compute a random index to test Proj_operator^2 = Proj_operator

        random_index = random.integers(len(ts_train))
        sample = ts_train[random_index]  # random.choice(ts_train)
        parameter_sample = parameters_train[random_index]

        interp_fun = eim.transform(sample, parameter_sample)
        re_interp_fun = eim.transform(interp_fun, parameter_sample)
        np.testing.assert_allclose(
            interp_fun, re_interp_fun, rtol=1e-5, atol=1e-8
        )

        leaf = eim.reduced_basis.search_leaf(
            parameter_sample, node=eim.reduced_basis.tree
        )
        assert leaf.is_leaf
        # test if interpolation is true
        np.testing.assert_allclose(
            interp_fun[leaf.nodes], sample[leaf.nodes], rtol=1e-5, atol=1e-8
        )


def test_EmpiricalInterpolationit_error_no_info():
    # usa lineas de "elif reduced_basis is None:" y
    # "if not "tree" in vars(self.reduced_basis):".
    # Comprueba que no hay errores al instanciacion,
    # fit y transform de EIM en el caso particular.
    b = 0.2
    y0 = [np.pi / 2, 0.0]

    param = np.linspace(1, 5, 101)
    times = np.linspace(0, 50, 1001)

    training = []
    for λ in param:
        sol = odeint(pend, y0, times, (b, λ))
        training.append(sol[:, 0])

    training_set = np.array(training)
    parameters = param
    physical_points = times

    rb = ReducedBasis()
    rb.fit(
        training_set=training_set,
        parameters=parameters,
        physical_points=physical_points,
    )
    eim_model = EmpiricalInterpolation(reduced_basis=rb)
    eim_model.fit()
    eim_model.transform(training_set[0], parameters[0])
    assert True
