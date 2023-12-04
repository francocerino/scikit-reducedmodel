from .empiricalinterpolation import EmpiricalInterpolation


class ReducedOrderQuadrature:
    def __init__(self, eim) -> None:
        self.eim = eim
        if not isinstance(self.eim, EmpiricalInterpolation):
            raise ValueError

    # simplified version of ROQ. no "t_c", "W", and more.
    def fit(self, data) -> None:
        # compute weigths of each partition defined by hp-greedy algorithm
        for leaf in self.eim.reduced_basis.tree.leaves:
            leaf._roq_weights = data @ leaf.interpolant

    def predict(self, parameter):
        # return weigths of a partition defined by hp-greedy algorithm
        leaf = self.eim.reduced_basis.search_leaf(
            parameter, node=self.eim.reduced_basis.tree
        )
        return leaf._roq_weights
