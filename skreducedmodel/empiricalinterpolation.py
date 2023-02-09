"""Empirical Interpolation Methods."""


# import functools

import numpy as np

from skreducedmodel.reducedbasis import ReducedBasis

# import logging
# import attr
# from . import integrals


# =================================
# CONSTANTS
# =================================


# logger = logging.getLogger("arby.basis")


class InputDataError(ValueError):
    pass


class EmpiricalInterpolation:
    """Empirital interpolation functions and methods.

    This class is used to build empirical interpolants.

    Parameters
    ----------
    reduced_basis : ...
    """

    # Se inicializa con la clase base reducida
    def __init__(
        self,
        reduced_basis=None,
        **kwargs,
    ) -> None:
        """Initialize the class.
        This methods initialize the EmpiritalInterpolation class.
        """
        if reduced_basis is not None:
            if kwargs != {}:
                raise InputDataError(
                    "**kwargs != { } and not taken in account "
                    + "because a reduced basis is given"
                )
            self.base = reduced_basis
        elif reduced_basis is None:
            self.base = ReducedBasis(**kwargs)
        self._trained = False

    # def fit(self):
    #    print(self.basis.indices)

    # @functools.lru_cache(maxsize=None)
    # [fc] para que lo de arriba?
    # @property

    def fit(
        self, training_set=None, parameters=None, physical_points=None
    ) -> None:
        """Implement EIM algorithm.

        The Empirical Interpolation Method (EIM)
        introspects the basis and selects a set of interpolation ``nodes`` from
        the physical domain for building an ``interpolant`` matrix using the
        basis and the selected nodes. The ``interpolant`` matrix can be used to
        approximate a field of functions for which the span of the basis is a
        good approximant.
        Returns: skreducemodel.eim
        Container for EIM data. Contains (``interpolant``, ``nodes``).
        """
        if "tree" not in vars(self.base):
            # build tree if it does not exist
            self.base.fit(training_set, parameters, physical_points)

        elif not (
            training_set is None
            and parameters is None
            and physical_points is None
        ):
            raise InputDataError(
                "Reduced Basis is already trained. "
                + "'training_set' or 'parameters' or"
                + "'physical_points' not needed"
            )

        for leaf in self.base.tree.leaves:
            nodes = []
            v_matrix = None
            first_node = np.argmax(np.abs(leaf.basis[0]))
            nodes.append(first_node)

            nbasis = len(leaf.indices)

            # logger.debug(first_node)

            for i in range(1, nbasis):
                v_matrix = self._next_vandermonde(leaf.basis, nodes, v_matrix)
                base_at_nodes = [leaf.basis[i, t] for t in nodes]
                invv_matrix = np.linalg.inv(v_matrix)
                step_basis = leaf.basis[:i]
                basis_interpolant = base_at_nodes @ invv_matrix @ step_basis
                residual = leaf.basis[i] - basis_interpolant
                new_node = np.argmax(abs(residual))

                # logger.debug(new_node)

                nodes.append(new_node)

            v_matrix = np.array(
                self._next_vandermonde(leaf.basis, nodes, v_matrix)
            )
            invv_matrix = np.linalg.inv(v_matrix.T)
            interpolant = leaf.basis.T @ invv_matrix

            leaf.interpolant = interpolant
            leaf.nodes = nodes

        self._trained = True

    @property
    def is_trained(self):
        return self._trained

    def _next_vandermonde(self, data, nodes, vandermonde=None):
        """Build the next Vandermonde matrix from the previous one."""
        if vandermonde is None:
            vandermonde = [[data[0, nodes[0]]]]
            return vandermonde

        n = len(vandermonde)
        new_node = nodes[-1]
        for i in range(n):
            vandermonde[i].append(data[i, new_node])

        vertical_vector = [data[n, nodes[j]] for j in range(n)]
        vertical_vector.append(data[n, new_node])
        vandermonde.append(vertical_vector)
        return vandermonde

    def transform(self, h, q):
        """Interpolate a function h at EIM nodes.

        This method uses the basis and associated EIM nodes
        (see the ``arby.Basis.eim_`` method) for interpolation.

        Parameters
        ----------
        h : np.ndarray
            Function or set of functions to be interpolated.

        Returns
        -------
        h_interpolated : np.ndarray
            Interpolated function at EIM nodes.
        """

        # search leaf and use the basis associated
        leaf = self.base.search_leaf(q, node=self.base.tree)
        # print(f"node name: {leaf.name}. is root: {leaf.is_leaf}")
        # print(np.sort(leaf.train_parameters[:,0])[0],np.sort(leaf.train_parameters[:,0])[-1])

        h = h.T
        h_at_nodes = h[leaf.nodes]
        h_interpolated = leaf.interpolant @ h_at_nodes
        return h_interpolated.T
