"""Surrogate module."""

import gwtools

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor

from .empiricalinterpolation import EmpiricalInterpolation, _error


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
    eim :
        Instance of EmpiricalInterpolation.
    """

    def __init__(
        self,
        eim=None,
        # poly_deg=3,
        regression_model=GaussianProcessRegressor,
        regression_hyperparameters=None,
    ) -> None:
        """Initialize the class.

        This methods initialize the Surrogate class.
        """
        # self.poly_deg = poly_deg
        self.eim = EmpiricalInterpolation() if eim is None else eim
        self._trained = False
        self.regression_model = regression_model
        self.regression_hyperparameters = regression_hyperparameters

    def get_params(self):
        return {
            "eim": self.eim,
            "regression_model": self.regression_model,
            "regression_hyperparameters": self.regression_hyperparameters,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self) -> None:
        """Construct the model.

        Build a surrogate model valid for the entire parameter domain.
        Regressions at empirical times are trained to build the
        surrogate model.
        """
        # train surrogate stage
        for leaf in self.eim.reduced_basis.tree.leaves:
            if self.eim.reduced_basis.complex_dataset:
                # leaf.complex_dataset_bool = True
                amp_training_set = np.array(
                    [gwtools.amp(h) for h in leaf.training_set]
                )  # np.abs(leaf.training_set)

                phase_training_set = np.array(
                    [gwtools.phase(h) for h in leaf.training_set]
                )  # np.angle(leaf.training_set)

                leaf._cached_regression_model_amp = self._regression_model(
                    leaf, amp_training_set, leaf.train_parameters
                )

                leaf._cached_regression_model_phase = self._regression_model(
                    leaf, phase_training_set, leaf.train_parameters
                )

            else:
                # leaf.complex_dataset_bool = False
                leaf._cached_regression_model = self._regression_model(
                    leaf, leaf.training_set, leaf.train_parameters
                )

        self._trained = True

    @property
    def is_trained(self):
        """Return True only if the instance is trained, False otherwise."""
        return self._trained

    def _regression_model(self, leaf, training_set, parameters):
        training_compressed = training_set[:, leaf.empirical_nodes]

        """
        h_in_nodes_regression = [
            splrep(
                np.sort(parameters),
                training_compressed[:, i][np.argsort(parameters)],
                k=self.poly_deg,
                )
                for i, _ in enumerate(leaf.basis)
        ]
        """
        rb = self.eim.reduced_basis
        if self.regression_hyperparameters is None:
            h_in_nodes_regression = [
                self.regression_model().fit(
                    parameters.reshape(-1, rb.parameter_dimension),
                    training_compressed[:, i],
                )
                for i, _ in enumerate(leaf.basis)
            ]
        else:
            # the same as above, but adding new hyperparameters to the model
            h_in_nodes_regression = [
                self.regression_model(**self.regression_hyperparameters).fit(
                    parameters.reshape(-1, rb.parameter_dimension),
                    training_compressed[:, i],
                )
                for i, _ in enumerate(leaf.basis)
            ]

        return h_in_nodes_regression

    def predict(self, parameter, only_regressions=False):
        """Evaluate the surrogate model at a given parameter.

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

        if (
            not self.eim.reduced_basis.complex_dataset
        ):  # if not leaf.complex_dataset_bool:
            h_surrogate_at_nodes = self._prediction_real_dataset(
                parameter, leaf._cached_regression_model
            )

        else:
            h_surrogate_at_nodes = self._prediction_real_dataset(
                parameter, leaf._cached_regression_model_amp
            ) * np.exp(
                1j
                * self._prediction_real_dataset(
                    parameter, leaf._cached_regression_model_phase
                )
            )
        if not only_regressions:
            h_surrogate = leaf.interpolant @ h_surrogate_at_nodes
            return h_surrogate
        else:
            return h_surrogate_at_nodes

    def _prediction_real_dataset(self, parameter, fitted_models):
        # ver si shape de splines y gpr devueltos son los mismos.
        # todo esta codeado para splines

        """
        h_surrogate_at_nodes = np.array(
            [splev(parameter, spline) for spline in fitted_models]
        )
        """
        rb = self.eim.reduced_basis

        h_surrogate_at_nodes = np.array(
            [
                model.predict(parameter.reshape(1, rb.parameter_dimension))
                for model in fitted_models
            ]
        )

        return h_surrogate_at_nodes

    def score(self, h1, h2, domain, rule="riemann"):
        return _error(h1, h2, domain, rule)
