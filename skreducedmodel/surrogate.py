"""Surrogate module."""

import numpy as np

from scipy.interpolate import splev, splrep

from .empiricalinterpolation import EmpiricalInterpolation


class Surrogate:
    """Build reduced order models.

    This class comprises a set of tools to build and handle reduced bases,
    empirical interpolants and predictive models from pre-computed training
    set of functions. The underlying or ground truth model describing the
    training set is a real function g(v,x) parameterized by a *training*
    parameter v. The *physical* variable x belongs to a domain for which an
    inner product can defined. The surrogate model is built bringing together
    the Reduced Basis (RB) greedy algorithm and the Empirical Interpolation
    Method (EIM) to work in synergy towards a predictive model for the ground
    truth model.

    Parameters
    ----------
    eim : EmpiricalInterpolation, optional
        .
    poly_deg: int, optional
        Degree <= 5 of polynomials used for splines. Default = 3.

    Attributes
    ----------
    _trained : bool
        .
    poly_deg : int
        .
    eim :
        .
    """

    def __init__(self, eim=None, poly_deg=3) -> None:
        """Initialize the class.

        This methods initialize the Surrogate class.
        """
        self.poly_deg = poly_deg
        self.eim = EmpiricalInterpolation() if eim is None else eim
        self._trained = False

    def fit(self) -> None:
        """Build a surrogate model from EmpiricalInterpolation class.

        Parameters
        ----------
        parameter1 : numpy.ndarray
           Explanation
        """
        # train surrogate stage
        for leaf in self.eim.reduced_basis.tree.leaves:
            if np.any(np.iscomplex(leaf.training_set)):
                leaf.complex_dataset_bool = True

                amp_training_set = np.abs(leaf.training_set)

                phase_training_set = np.angle(leaf.training_set)

                leaf._cached_spline_model_amp = self._spline_model(
                    leaf, amp_training_set, leaf.train_parameters
                )

                leaf._cached_spline_model_phase = self._spline_model(
                    leaf, phase_training_set, leaf.train_parameters
                )

            else:
                leaf.complex_dataset_bool = False
                leaf._cached_spline_model = self._spline_model(
                    leaf, leaf.training_set, leaf.train_parameters
                )

        self._trained = True

    @property
    def is_trained(self):
        """Return True only if the instance is trained, False otherwise."""
        return self._trained

    def _spline_model(self, leaf, training_set, parameters):
        training_compressed = training_set[:, leaf.nodes]
        h_in_nodes_splined = [
            splrep(
                np.sort(parameters),
                training_compressed[:, i][np.argsort(parameters)],
                k=self.poly_deg,
            )
            for i, _ in enumerate(leaf.basis)
        ]
        return h_in_nodes_splined

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
        leaf = self.eim.reduced_basis.search_leaf(
            parameter, node=self.eim.reduced_basis.tree
        )

        if not leaf.complex_dataset_bool:
            h_surrogate_at_nodes = self._prediction_real_dataset(
                parameter, leaf._cached_spline_model
            )

        else:
            h_surrogate_at_nodes = self._prediction_real_dataset(
                parameter, leaf._cached_spline_model_amp
            ) * np.exp(
                1j
                * self._prediction_real_dataset(
                    parameter, leaf._cached_spline_model_phase
                )
            )

        h_surrogate = leaf.interpolant @ h_surrogate_at_nodes

        return h_surrogate

    def _prediction_real_dataset(self, parameter, fitted_model):

        h_surrogate_at_nodes = np.array(
            [splev(parameter, spline) for spline in fitted_model]
        )

        return h_surrogate_at_nodes
