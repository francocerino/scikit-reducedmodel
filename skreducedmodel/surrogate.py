from skreducedmodel.reducedbasis import ReducedBasis

from skreducedmodel.empiricalinterpolation import EmpiricalInterpolation

from skreducedmodel.empiricalinterpolation import InputDataError

# import gwtools.gwtools as gwtools

import numpy as np

from scipy.interpolate import splev, splrep

class Surrogate:
    def __init__(self,
                 poly_deg=3,
                 reduced_basis=None,
                 eim=None,
                 **kwargs
                ) -> None:

        self.poly_deg = poly_deg
        if reduced_basis == None and eim == None:
            self.base = ReducedBasis(**kwargs)
            self.eim = EmpiricalInterpolation(reduced_basis=self.base)

        elif eim is not None:
            if reduced_basis is not None:
                raise InputDataError(
                    "reduced_basis != None and not taken in account "
                    + "because a eim instance is given. Only "
                    + "one must be given."
                )
            elif  kwargs != {}:
                raise InputDataError(
                    "kwargs != { } and not taken in account "
                    + "because a eim instance is given. Only "
                    + "one must be given."
                )
            self.eim = eim

        elif reduced_basis is not None:
            if kwargs != {}:
                raise InputDataError(
                    "**kwargs != { } and not taken in account "
                    + "because a reduced basis is given"
                )
            self.base = reduced_basis
        
        elif reduced_basis is None:
            self.base = ReducedBasis(**kwargs)


    def fit(self,
            training_set=None,
            parameters=None,
            physical_points=None
            ) -> None:

        #if not (
        #        training_set is None
        #        and parameters is None
        #        and physical_points is None
        #    ):
        #       raise InputDataError(
        #            "Reduced Basis is already trained. "
        #            + "'training_set' or 'parameters' or"
        #            + "'physical_points' not needed"
        #        )

        # build reduced basis or eim if there are not created
        if "tree" not in vars(self.base):
            # build tree if it does not exist
            self.base.fit(training_set, parameters, physical_points)
            self.eim.fit()
        elif "eim" not in vars(self):
            self.eim = EmpiricalInterpolation(reduced_basis=self.base)
            self.eim.fit()

        # train surrogate stage
        for leaf in self.base.tree.leaves:
            if np.any(np.iscomplex(leaf.training_set)):
                leaf.complex_dataset_bool = True
                amp_training_set, phase_training_set = _amp_phase_set(leaf.training_set)
                
                leaf._cached_spline_model_amp = self._spline_model(leaf,
                                                                   amp_training_set,
                                                                   leaf.train_parameters
                                                                   )

                leaf._cached_spline_modela_phase = self._spline_model(leaf,
                                                                      phase_training_set,
                                                                      leaf.train_parameters
                                                                      )

            else:
                leaf.complex_dataset_bool = False
                leaf._cached_spline_model = self._spline_model(leaf,
                                                                leaf.training_set,
                                                                parameters
                                                                )

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

        leaf = self.base.search_leaf(parameter, node=self.base.tree)

        if not leaf.complex_dataset_bool:

            h_surrogate = self._prediction_real_dataset(leaf,parameter, leaf._cached_spline_model)

        else:

            h_surrogate = self._prediction_real_dataset(leaf,parameter, leaf._cached_spline_model_amp)*\
                np.exp(1j*self.prediction_real_dataset(leaf,parameter, leaf._cached_spline_model_phase))

        return h_surrogate

    def _prediction_real_dataset(self, leaf, parameter, model):
        fitted_model = model

        h_surr_at_nodes = np.array(
            [splev(parameter, spline) for spline in fitted_model]
        )
        h_surrogate = leaf.interpolant @ h_surr_at_nodes

        return h_surrogate

    def _amp_phase_set(self,training_set):
        amp_training_set = np.zeros(training_set.shape)
        phase_training_set = np.zeros(training_set.shape)
        for idx, f in enumerate(training_set):
            amp_training_set[idx], phase_training_set[idx] = gwtools.amp_phase(f)
        return amp_training_set, phase_training_set
