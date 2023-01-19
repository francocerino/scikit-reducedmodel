from scipy.integrate import odeint
import numpy as np
from skreducedmodel.reducedbasis import ReducedBasis
from skreducedmodel.reducedbasis import normalize_set
from skreducedmodel import integrals
# from scipy.special import jv as BesselJ
#
#
# def test_dim_rb_with_nmax():
#    # import data of 1d gravitational waves
#    q_train = np.load("q_train_1d_seed=1.npy")
#    ts_train = np.load("ts_train_1d_seed=1.npy")
#    # q_test = np.load("q_test_1d_seed=1.npy")
#    # ts_test = np.load("ts_test_1d_seed=1.npy")
#    times = np.load("times_1d_seed=1.npy")
#
#    # para nmax == 0 tiene que saltar el assert
#    for nmax in range(1, 3):
#        model = ReducedModel(
#                    seed_global_rb=0,
#                    greedy_tol=1e-16,
#                    lmax=0,
#                    nmax=nmax,
#                    normalize=True
#                    )
#
#        rb = model.fit(
#                training_set=ts_train,
#                parameters=q_train,
#                physical_points=times
#                )
#
#        assert len(rb.indices) == nmax


def pend(y, t, b, λ):
    θ, ω = y
    dydt = [ω, -b * ω - λ * np.sin(θ)]

    return dydt

def test_normalize_set(ts_test,times):

    ts_test_normalized = normalize_set(ts_test,times)
    integration = integrals.Integration(times, 'riemann')
    for i in range(10):
        norms = integration.norm(ts_test_normalized[i,:])
        assert np.allclose(norms,1,1e-10)

def test_transform():

    b = 0.2
    y0 = [np.pi / 2, 0.0]

    parameters = np.linspace(1, 5, 101)
    times = np.linspace(0, 50, 1001)

    training = []
    for λ in parameters:
        sol = odeint(pend, y0, times, (b, λ))
        training.append(sol[:, 0])

    training_set = np.array(training)
    physical_points = times
    nmax = 10

    model = ReducedBasis(index_seed_global_rb=0,
                         greedy_tol=1e-16,
                         lmax=0,
                         nmax=nmax,
                         normalize=True
                        )

    model.fit(training_set=training_set,
              parameters=parameters,
              physical_points=physical_points,
             )

    wave1 = model.tree.basis[0]
    wave_transform = model.transform(wave1,parameters)

    assert wave1.all() == wave_transform.all()


def test_ReducedModelFit():

    b = 0.2
    y0 = [np.pi / 2, 0.0]

    parameters = np.linspace(1, 5, 101)
    times = np.linspace(0, 50, 1001)

    training = []
    for λ in parameters:
        sol = odeint(pend, y0, times, (b, λ))
        training.append(sol[:, 0])

    training_set = np.array(training)
    physical_points = times
    nmax = 10

    model = ReducedBasis(index_seed_global_rb=0,
                         greedy_tol=1e-16,
                         lmax=0,
                         nmax=nmax,
                         normalize=True
                        )

    model.fit(training_set=training_set,
              parameters=parameters,
              physical_points=physical_points,
             )

    print(model.tree.errors[nmax - 1], model.tree.errors[0])

    assert model.tree.errors[0] > model.tree.errors[nmax - 1]
    assert model.tree.errors[5] > model.tree.errors[nmax - 1]
    assert len(model.tree.indices) == nmax
    assert len(model.tree.indices) == nmax
    # todos los numeros salieron del ejemplo del Pendulo
    assert model.tree.indices[9] == 92


def test_rmfit_parameters():

    b = 0.2
    y0 = [np.pi / 2, 0.0]

    parameters = np.linspace(1, 5, 101)
    times = np.linspace(0, 50, 1001)

    training = []
    for λ in parameters:
        sol = odeint(pend, y0, times, (b, λ))
        training.append(sol[:, 0])

    training_set = np.array(training)
    physical_points = times
    # nmax = 10

    model1 = ReducedBasis(
        index_seed_global_rb=0,
        greedy_tol=1e-1,
        lmax=0,
    )

    model2 = ReducedBasis(
        index_seed_global_rb=0,
        greedy_tol=1e-16,
        lmax=0,
    )

    model1.fit(
        training_set=training_set,
        parameters=parameters,
        physical_points=physical_points,
    )

    model2.fit(
        training_set=training_set,
        parameters=parameters,
        physical_points=physical_points,
    )

    assert len(model1.tree.indices) < len(model2.tree.indices)



"""
def test_rom_rb_interface(rom_parameters):
    ""Test API consistency.""
    training_set = rom_parameters["training_set"]
    physical_points = rom_parameters["physical_points"]
    parameter_points = rom_parameters["parameter_points"]

    model = ReducedModel(greedy_tol=1e-14)

    bessel = model.fit(training_set=training_set,
                       physical_points=physical_points,
                       parameters=parameter_points
                       )

    # bessel = ReducedOrderModel(
    #    training_set, physical_points, parameter_points, greedy_tol=1e-14
    # )
    basis = bessel.basis.data
    errors = bessel.errors
    projection_matrix = bessel.projection_matrix
    greedy_indices = bessel.indices
    # eim = bessel.eim_

    assert len(basis) == 10
    assert len(errors) == 10
    assert len(projection_matrix) == 101
    assert len(greedy_indices) == 10
    # assert eim == bessel.basis_.eim
"""

def test_partition():
    # test para para los índices de los parametros de entrenamiento de
    # los subespacios resultantes.
    # la interseccion tiene que dar vacia.
    # la union da los índices parametros de entrenamiento del espacio original
    
    b = 0.2
    y0 = [np.pi / 2, 0.0]

    parameters = np.linspace(1, 5, 101)
    times = np.linspace(0, 50, 1001)

    training = []
    for λ in parameters:
        sol = odeint(pend, y0, times, (b, λ))
        training.append(sol[:, 0])

    training_set = np.array(training)
    physical_points = times
    nmax = 10
    lmax = 1
    
    model = ReducedBasis(index_seed_global_rb=0,
                         greedy_tol=1e-16,
                         lmax=lmax,
                         nmax=nmax,
                         normalize=True
                        )

    model.fit(training_set=training_set,
              parameters=parameters,
              physical_points=physical_points,
             )

    idxs_subspace1 = model.tree.idxs_subspace0
    idxs_subspace2 = model.tree.idxs_subspace1
    assert model.tree.height == 1
    assert set(idxs_subspace2) & set(idxs_subspace1) == set()
    assert set(idxs_subspace2) | set(idxs_subspace1) == set(range(len(parameters)))
