"""Empirical Interpolation Methods."""

import numpy as np

from skreducedmodel.reducedbasis import ReducedBasis, _error


# =================================
# CONSTANTS
# =================================


class EmpiricalInterpolation:
    """Empirital interpolation functions and methods.

    Implements EIM algorithm:

        The Empirical Interpolation Method (EIM) (TiglioAndVillanueva2021)
        introspects the basis and selects a set of interpolation ``nodes`` from
        the physical domain for building an ``interpolant`` matrix using the
        basis and the selected nodes. The ``interpolant`` matrix can be used to
        approximate a field of functions for which the span of the basis is a
        good approximant.

    Parameters
    ----------
    reduced_basis : instance of ReducedBasis
    """

    # Se inicializa con la clase base reducida
    def __init__(self, reduced_basis=None) -> None:
        """Initialize the class.

        This method initializes the EmpiritalInterpolation class.

        Parameters
        ----------
        reduced_basis : ReducedBasis, optional
            instance of a reduced basis, by default None
        """
        self.reduced_basis = (
            ReducedBasis() if reduced_basis is None else reduced_basis
        )
        self._trained = False

    def get_params(self):

        return {
            "reduced_basis": self.reduced_basis,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self) -> None:
        """Implement EIM algorithm.

        The Empirical Interpolation Method (EIM)
        introspects the basis and selects a set of interpolation ``nodes`` from
        the physical domain for building an ``interpolant`` matrix using the
        basis and the selected nodes. The ``interpolant`` matrix can be used to
        approximate a field of functions for which the span of the basis is a
        good approximant.

        Container for EIM data. Contains (``interpolant``, ``nodes``).
        """
        for leaf in self.reduced_basis.tree.leaves:
            nodes = []
            v_matrix = None
            first_node = np.argmax(np.abs(leaf.basis[0]))
            nodes.append(first_node)

            nbasis = len(leaf.indices)

            for i in range(1, nbasis):
                v_matrix = self._next_vandermonde(leaf.basis, nodes, v_matrix)
                base_at_nodes = [leaf.basis[i, t] for t in nodes]
                invv_matrix = np.linalg.inv(v_matrix)
                step_basis = leaf.basis[:i]
                basis_interpolant = base_at_nodes @ invv_matrix @ step_basis
                residual = leaf.basis[i] - basis_interpolant
                new_node = np.argmax(abs(residual))

                nodes.append(new_node)

            v_matrix = np.array(
                self._next_vandermonde(leaf.basis, nodes, v_matrix)
            )
            invv_matrix = np.linalg.inv(v_matrix.T)
            interpolant = leaf.basis.T @ invv_matrix

            leaf.invv_matrix = invv_matrix
            leaf.v_matrix = v_matrix.T
            leaf.interpolant = interpolant
            leaf.empirical_nodes = nodes

        self._trained = True

    @property
    def is_trained(self):
        """Return True only if the instance is trained, False otherwise.

        Returns
        -------
        Bool
        """
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

        Parameters
        ----------
        h : np.ndarray
            Function or set of functions to be interpolated.

        Returns
        -------
        h_interpolated : np.ndarray
            Interpolated function at EIM nodes.
        """
        # search leaf and use the associated basis.
        leaf = self.reduced_basis.search_leaf(q, node=self.reduced_basis.tree)

        h = h.T
        h_at_nodes = h[leaf.empirical_nodes]
        h_interpolated = leaf.interpolant @ h_at_nodes
        return h_interpolated.T

    def score(self, h1, h2, domain, rule="riemann"):
        return _error(h1, h2, domain, rule)
