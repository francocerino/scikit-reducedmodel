import numpy as np

from .empiricalinterpolation import EmpiricalInterpolation
from .integrals import Integration


class ReducedOrderQuadrature:
    def __init__(self, eim) -> None:
        self.eim = eim
        if not isinstance(self.eim, EmpiricalInterpolation):
            raise ValueError("EmpiricalInterpolation object is expected")

    # simplified version of ROQ. no "t_c", delta_t, "W", and more.
    def fit(self, times, data=None, quadratic_weights=False) -> None:
        # data = None is used for the case of computing weights for the norm
        # compute weights of each partition defined by hp-greedy algorithm

        integration = Integration(times, rule="riemann")

        self.quadratic_weights = quadratic_weights
        for leaf in self.eim.reduced_basis.tree.leaves:
            if not self.quadratic_weights:
                w_weights = []
                for bj in leaf.interpolant.T:
                    integrand = bj * np.conjugate(data)
                    # assert integrand.shape == (31300,)
                    w = integration.integral(integrand)
                    w_weights.append(w)
                # assert len(integrands)==len(np.conjugate(leaf.interpolant).T)

                w_weights = np.array(w_weights)
                leaf._roq_weights = w_weights

            else:
                c_weights = []
                for quadratic_bj in leaf.interpolant.T:
                    c_weight = np.real(integration.integral(quadratic_bj))
                    c_weights.append(c_weight)

                c_weights = np.array(c_weights)
                leaf._roq_weights = c_weights

    def predict(self, parameter):
        # return weigths of a partition defined by hp-greedy algorithm
        leaf = self.eim.reduced_basis.search_leaf(
            parameter, node=self.eim.reduced_basis.tree
        )
        return leaf._roq_weights
