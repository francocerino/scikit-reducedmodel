import numpy as np
import pytest
from skreducedmodel.reducedbasis import ReducedBasis
from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation
from skreducedmodel.reducedorderquadrature import ReducedOrderQuadrature

# hacer que eim este entrenado al entrenar?


def test_input_reduced_basis_class():
    with pytest.raises(ValueError):
        rb = ReducedBasis()
        roq = ReducedOrderQuadrature(rb)


def test_dim_weights_equal_number_eim_nodes(
    ts_train, parameters_train, times, ts_test
):
    # dimension de array de pesos igual que cantidad de nodos empíricos
    rb = ReducedBasis(
        greedy_tol=1e-16,
        lmax=1,
        normalize=True,
    )
    rb.fit(ts_train, parameters_train, times)
    eim = EmpiricalInterpolation(rb)
    eim.fit()
    roq = ReducedOrderQuadrature(eim)
    roq.fit(ts_test[0])

    parameters_test = np.array([1, 10])
    for leaf, parameter in zip(eim.reduced_basis.tree.leaves, parameters_test):
        len(leaf.nodes) == len(roq.predict(parameter))
