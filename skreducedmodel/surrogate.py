from skreducedmodel.reducedbasis import ReducedBasis

from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

import numpy as np

from scipy.interpolate import splev, splrep

class Surrogate:
    def __init__(self,
                 poly_deg = 3,
                 **kwargs
                ) -> None:
        self.base = ReducedBasis(**kwargs)
        self.eim = EmpiricalInterpolation(reduced_basis=self.base)
        self.poly_deg = poly_deg

    def fit(self,
            training_set=None,
            parameters=None,
            physical_points=None
            ) -> None:
        if "tree" not in vars(self.base):
            # build tree if it does not exist
            self.base.fit(training_set, parameters, physical_points)
            self.eim.fit()

        for leaf in self.base.tree.leaves:
            self._spline_model(leaf,training_set,parameters)

    def _spline_model(self,leaf,training_set,parameters):

        training_compressed = training_set[:, leaf.nodes]

        h_in_nodes_splined = [
            splrep(
                np.sort(parameters),
                training_compressed[:, i][np.argsort(parameters)],
                k=self.poly_deg,
            )
            for i, _ in enumerate(leaf.basis)
        ]

        leaf._cached_spline_model = h_in_nodes_splined

    def predict(self, parameter):
        """Evaluate the surrogate model at parameter/s.

        Build a surrogate model valid for the entire parameter domain.
        The building stage is performed only once for the first function call.
        For subsequent calls, the method invokes the already fitted model and
        just evaluates it. The output is an array storing surrogate evaluations
        at the parameter/s.

        Parameters
        ----------
        param : float or array_like(float)
            Point or set of parameters inside the parameter domain.

        Returns
        -------
        h_surrogate : numpy.ndarray
            The evaluated surrogate function for the given parameters.

        """

        leaf = self.base.search_leaf(parameter, node=self.base.tree)

        fitted_model = leaf._cached_spline_model

        h_surr_at_nodes = np.array(
            [splev(parameter, spline) for spline in fitted_model]
        )
        h_surrogate = leaf.interpolant @ h_surr_at_nodes

        return h_surrogate