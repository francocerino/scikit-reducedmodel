from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

from scipy.integrate import odeint

from skreducedmodel.reducedbasis import ReducedBasis

import numpy as np


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
    ti.fit

    print(ti.nodes)

    assert ti.nodes[0] == 0
    assert ti.nodes[5] == 167
    assert ti.nodes[19] == 816
