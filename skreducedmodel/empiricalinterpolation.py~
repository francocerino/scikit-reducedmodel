"""Empirical Interpolation Methods."""


import functools

import numpy as np

# import logging
# import attr
# from . import integrals


# =================================
# CONSTANTS
# =================================


# logger = logging.getLogger("arby.basis")


class EmpiricalInterpolation:

    """Class with the empirical interpolation functions
    and methods

    Parameters
    ----------

    reduced_basis : ...

    """

    # Se inicializa con la clase base reducida
    def __init__(self, reduced_basis):
        self.base = reduced_basis

    # def fit(self):
    #    print(self.basis.indices)

    @property
    @functools.lru_cache(maxsize=None)
    def fit(self):
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
        nodes = []
        v_matrix = None
        first_node = np.argmax(np.abs(self.base.tree.basis[0]))
        nodes.append(first_node)

        nbasis = len(self.base.tree.indices)

        # logger.debug(first_node)

        for i in range(1, nbasis):
            # print(i)
            v_matrix = self._next_vandermonde(
                self.base.tree.basis, nodes, v_matrix
            )
            base_at_nodes = [self.base.tree.basis[i, t] for t in nodes]
            invv_matrix = np.linalg.inv(v_matrix)
            step_basis = self.base.tree.basis[:i]
            basis_interpolant = base_at_nodes @ invv_matrix @ step_basis
            residual = self.base.tree.basis[i] - basis_interpolant
            new_node = np.argmax(abs(residual))

            # logger.debug(new_node)

            nodes.append(new_node)

        v_matrix = np.array(
            self._next_vandermonde(self.base.tree.basis, nodes, v_matrix)
        )
        invv_matrix = np.linalg.inv(v_matrix.T)
        interpolant = self.base.tree.basis.T @ invv_matrix

        self.interpolant = interpolant
        self.nodes = nodes
        # return EIM(interpolant=interpolant, nodes=nodes)

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
