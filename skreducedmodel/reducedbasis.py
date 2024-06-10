"""Reduced Basis module."""

from anytree import Node

import numpy as np

from . import integrals

# import logging

# logger = logging.getLogger("arby.basis")


class ReducedBasis:
    """Class for building a reduced basis (RB) using the RB greedy algorithm.

    The reduced basis is built from the training data set with a user
    specified tolerance by linear combinations of its elements. The
    reduced basis can be also constructed with a  domain decomposition
    of the parameter space, building local basis in each subspace.

    Parameters
    ----------
    index_seed_global_rb : int, optional
        The seed for construct the reduced basis, by default 0.
    lmax : int, optional
        The maximum number domain partitions performed.
    nmax : int, optional
        The maximum number of basis functions to be used, by default np.inf.
    greedy_tol : float, optional
        The greedy tolerance, by default 1e-12.
    normalize : bool, optional
        Indicates if the training set should be normalized, by default False.
    integration_rule : str, optional
        The integration rule to be used, by default "riemann".

    Attributes
    ----------
    tree : Node
        The tree data structure for the reduced basis.
    """

    def __init__(
        self,
        index_seed_global_rb=0,
        lmax=0,
        nmax=np.inf,
        greedy_tol=1e-12,
        normalize=False,
        integration_rule="riemann",
    ) -> None:
        """summary.

        Parameters
        ----------
        index_seed_global_rb : int, optional
            Index of the training_set element that is going to be the seed
              of the greedy algorithm, by default 0
        lmax : int, optional
            Maximum depth of the tree built by the partitioning of the
              parameter space, by default 0
        nmax : _type_, optional
            Maximum dimension of the reduced basis, by default np.inf
        greedy_tol : _type_, optional
            Precision to reach by the greedy algorithm, by default 1e-12
        normalize : bool, optional
            Normalize the training set, by default False
        integration_rule : str, optional
            By default "riemann"
        """

        # the default seed is the first of the array "parameters"
        self.index_seed_global_rb = index_seed_global_rb
        self.lmax = lmax
        self.nmax = nmax
        self.greedy_tol = greedy_tol
        self.normalize = normalize
        self.integration_rule = integration_rule

        assert self.nmax > 0 and self.lmax >= 0

    def get_params(self):

        return {
            "index_seed_global_rb": self.index_seed_global_rb,
            "lmax": self.lmax,
            "nmax": self.nmax,
            "greedy_tol": self.greedy_tol,
            "normalize": self.normalize,
            "integration_rule": self.integration_rule,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    # comenzamos la implementacion de reduced_basis
    # la idea es acoplar esto al método fit de ReducedModel.
    def fit(
        self,
        training_set,
        parameters,
        physical_points,
    ) -> None:
        """Build a reduced basis from training data.

        This function implements the Reduced Basis (RB) greedy algorithm to
        build an orthonormalized reduced basis out from training data. The
        basis is built to reproduce the training functions with the user
        specified tolerance by linear combinations
        of its elements.

        Parameters
        ----------
        training_set : numpy.ndarray
           Training set functions.
        parameters : numpy.ndarray
           Associated parameters to the training set functions.
        physical_points : numpy.ndarray
           Physical points for quadrature rules.
        """
        self.__first_iteration = True
        self._trained = False

        self.complex_dataset = np.any(np.iscomplex(training_set))
        if len(parameters.shape) == 1:
            parameters = parameters.reshape(-1, 1)
        assert len(parameters.shape) == 2
        self.parameter_dimension = parameters.shape[1]

        # parameters = X_train[0]
        # physical_points = X_train[1]

        self._fit(
            training_set,
            parameters,
            physical_points,
            index_seed=self.index_seed_global_rb,
            parent=None,
            node_idx=0,
            deep=0,
        )

        self._trained = True

    def _fit(
        self,
        training_set,
        parameters,
        physical_points,
        # Los siguientes son parametros internos para la función self.fit().
        index_seed,
        parent,
        node_idx,
        deep,
    ) -> None:
        _validate_parameters(parameters)
        _validate_physical_points(physical_points)
        _validate_training_set(training_set)

        if self.__first_iteration is True:
            assert parent is None and node_idx == 0 and deep == 0
            self.__first_iteration = False

        # Create a node for the tree.
        # If the tree does not exists, create it.
        if parent is not None:
            node = Node(
                name=parent.name + (node_idx,),
                parent=parent,
                train_parameters=parameters,
            )
        else:
            self.tree = Node(name=(node_idx,), train_parameters=parameters)
            node = self.tree

        integration = integrals.Integration(
            physical_points, rule=self.integration_rule
        )

        # useful constants
        ntrain = training_set.shape[0]
        nsamples = training_set.shape[1]
        max_rank = min(ntrain, nsamples)

        # validate inputs
        if nsamples != np.size(integration.weights_):
            raise ValueError(
                "Number of samples is inconsistent with quadrature rule."
            )

        if np.allclose(np.abs(training_set), 0, atol=1e-30):
            raise ValueError("Null training set!")

        # ====== Seed the greedy algorithm and allocate memory ======

        # memory allocation
        greedy_errors = np.empty(max_rank, dtype=np.float64)
        proj_matrix = np.empty((max_rank, ntrain), dtype=training_set.dtype)
        basis_data = np.empty((max_rank, nsamples), dtype=training_set.dtype)

        norms = integration.norm(training_set)
        if self.normalize:
            # normalize training set
            training_set = np.array(
                [
                    h if np.allclose(h, 0, atol=1e-15) else h / norms[i]
                    for i, h in enumerate(training_set)
                ]
            )

            # choose seed
            next_index = index_seed
            seed = training_set[next_index]

            aux = 0
            while aux < ntrain - 1:
                if np.allclose(np.abs(seed), 0):
                    if next_index < ntrain - 1:
                        next_index += 1
                    else:
                        next_index = 0

                    seed = training_set[next_index]
                    aux += 1
                else:
                    break

            greedy_indices = [next_index]
            basis_data[0] = training_set[next_index]
            proj_matrix[0] = integration.dot(basis_data[0], training_set)
            sq_errors = _sq_errs_rel
            errs = sq_errors(np.ones(ntrain), proj_matrix[0])

        else:
            # choose seed
            next_index = index_seed  # old version: np.argmax(norms)
            greedy_indices = [next_index]
            basis_data[0] = training_set[next_index] / norms[next_index]
            proj_matrix[0] = integration.dot(basis_data[0], training_set)
            # unitary vectors, then use absolute value, not relative:
            sq_errors = _sq_errs_abs
            errs, diff_training = sq_errors(
                proj_matrix[0], basis_data[0], integration.dot, training_set
            )

        next_index = np.argmax(errs)
        greedy_errors[0] = errs[next_index]
        sigma = greedy_errors[0]

        # ====== Start greedy loop ======

        # logger.debug("\n Step", "\t", "Error")
        nn = 0
        # print(nn, sigma, next_index)
        while sigma > self.greedy_tol and self.nmax > nn + 1:
            if next_index in greedy_indices:
                break

            nn += 1
            greedy_indices.append(next_index)
            basis_data[nn], _ = _gs_one_element(
                training_set[greedy_indices[nn]],
                basis_data[:nn],
                integration,
            )
            proj_matrix[nn] = integration.dot(basis_data[nn], training_set)
            if self.normalize:
                errs = sq_errors(errs, proj_matrix[nn])
            else:
                errs, diff_training = sq_errors(
                    proj_matrix[nn],
                    basis_data[nn],
                    integration.dot,
                    diff_training,
                )
            next_index = np.argmax(errs)
            greedy_errors[nn] = errs[next_index]

            sigma = errs[next_index]
            # print(nn, sigma, next_index)
            # logger.debug(nn, "\t", sigma)

        # Prune excess allocated entries
        greedy_errors, proj_matrix = _prune(greedy_errors, proj_matrix, nn + 1)
        if self.normalize:
            # restore proj matrix
            proj_matrix = norms * proj_matrix

        node.basis_error = greedy_errors[-1]
        node.indices = greedy_indices
        node.idx_anchor_0 = node.indices[0]
        if len(node.indices) > 1:
            node.idx_anchor_1 = node.indices[1]

        if (
            deep < self.lmax
            and self.greedy_tol < node.basis_error
            and len(node.indices) > 1
        ):
            idxs_subspace0, idxs_subspace1 = self.partition(
                parameters, node.idx_anchor_0, node.idx_anchor_1
            )

            # agrego estos atributos para usarlos al testear
            node.idxs_subspace0 = idxs_subspace0
            node.idxs_subspace1 = idxs_subspace1

            self._fit(
                training_set[idxs_subspace0],
                parameters[idxs_subspace0],
                physical_points,
                parent=node,
                node_idx=0,
                deep=deep + 1,
                index_seed=0,
            )

            self._fit(
                training_set[idxs_subspace1],
                parameters[idxs_subspace1],
                physical_points,
                parent=node,
                node_idx=1,
                deep=deep + 1,
                index_seed=0,
            )
        else:
            # estos datos se guardan solo cuando el nodo
            # es hoja del árbol.

            node.basis = basis_data[: nn + 1]
            node.errors = greedy_errors
            node.integration = integration
            node.train_parameters = parameters
            node.projection_matrix = proj_matrix.T
            node.training_set = training_set

    @property
    def is_trained(self):
        """Return True only if the instance is trained, False otherwise.

        Returns
        -------
        Bool
        """

        if hasattr(self, "_trained"):
            return self._trained
        else:
            return False

    def search_leaf(self, parameters, node):
        """Search Leaf.

        This function finds the leaf node in a tree by recursively
        evaluating each node using the select_child_node function.
        If a node is a leaf, the node is returned, otherwise the
        search continues on the selected child node.

        Parameters
        ----------
        parameters : np.ndarray
            Set of parameters to search in the tree.

        Returns
        -------
        node : np.ndarray
            Set of nodes where the parameters are located in the tree.
        """
        # parameters: conjunto de parametros que se quiere buscar sus
        # respectivas bases reducidas.

        if not node.is_leaf:
            child = select_child_node(parameters, node)
            return self.search_leaf(parameters, child)
        else:
            return node

    def transform(self, test_set, parameters, s=(None,)):
        # previous version: def project(self, h, s=(None,)):
        """Project a function h onto the basis.

        This method represents the action of projecting the function h onto the
        span of the basis.

        Parameters
        ----------
        h : np.ndarray
            Function or set of functions to be projected.
        s : tuple, optional
            Slice the basis. If the slice is not provided, the whole basis is
            considered. Default = (None,)

        Returns
        -------
        projected_function : np.ndarray
            Projection of h onto the basis.
        """
        # search leaf
        leaf = self.search_leaf(parameters, node=self.tree)

        # leafs = [self.search_leaf(parameter, node=self.tree)\
        #  for parameter in parameters]
        # projected_functions = []
        # for leaf, test_function in zip(leafs, test_set):
        #     s = slice(*s)
        #     projected_function = 0.0
        #     for e in leaf.basis[s]:
        #         projected_function += np.tensordot(
        #         leaf.integration.dot(e, test_function), e, axes=0
        #          )
        #     projected_functions.append(projected_function)
        # projected_functions = np.array(projected_functions)
        # return projected_function

        # use basis associated to leaf
        s = slice(*s)
        projected_function = 0.0
        for e in leaf.basis[s]:
            projected_function += np.tensordot(
                leaf.integration.dot(e, test_set), e, axes=0
            )
        return projected_function

    def partition(self, parameters, idx_anchor_0, idx_anchor_1):
        """Partition the parameter space.

        This code partitions a list of parameters into two subspaces
        based on their distances to two anchors. The function calculates
        the distances and returns two arrays of indices representing
        the parameters in each subspace. For equal distances the function
        make a random decision.

        Parameters
        ----------
        parameters : np.ndarray
            array of parameter from the domain of problem
        idx_anchor_0 : float, integer
            first greedy parameter of the space to divide
        idx_anchor_1 : float, integer
            second greedy parameter of the space to divide

        Returns
        -------
        np.ndarray
            indices of parameter that correspond to each subspace

        References
        ----------
        [CerinoAndTiglio2023] An automated parameter domain decomposition
        approach for gravitational wave surrogates using hp-greedy
        refinement. Cerino, F. and Tiglio M. arXiv:2212.08554 (2023)
        """
        anchor_0 = parameters[idx_anchor_0]
        anchor_1 = parameters[idx_anchor_1]

        assert not np.array_equal(anchor_0, anchor_1)

        seed = 12345
        rng = np.random.default_rng(seed)

        # caso de arby con ts normalizado:
        # la semilla es el primer elemento,
        # por lo tanto, si quiero que los anchors vayan primero:

        # #idxs_subspace0 = []
        # #idxs_subspace1 = []
        idxs_subspace_0 = [idx_anchor_0]
        idxs_subspace_1 = [idx_anchor_1]

        # y usar a continuación del for --> if idx != idx_anchor0
        # and idx != idx_anchor1:
        # sirve para el caso de usar normalize = True en
        # reduced_basis()
        # da error en splines porque scipy
        # pide los parametros de forma ordenada

        for idx, parameter in enumerate(parameters):
            if idx != idx_anchor_0 and idx != idx_anchor_1:
                dist_anchor_0 = np.linalg.norm(anchor_0 - parameter)  # 2-norm
                dist_anchor_1 = np.linalg.norm(anchor_1 - parameter)
                if dist_anchor_0 < dist_anchor_1:
                    idxs_subspace_0.append(idx)
                elif dist_anchor_0 > dist_anchor_1:
                    idxs_subspace_1.append(idx)
                else:
                    # para distancias iguales se realiza
                    # una elección aleatoria.
                    # tener en cuenta que se puede agregar el
                    # parametro a ambos subespacios!
                    if rng.integers(2):
                        idxs_subspace_0.append(idx)
                    else:
                        idxs_subspace_1.append(idx)

        return np.array(idxs_subspace_0), np.array(idxs_subspace_1)

    def score(self, h1, h2, domain, rule="riemann"):
        return _error(h1, h2, domain, rule)  # hacer con herencia de una clase?


def _prune(greedy_errors, proj_matrix, num):
    """Prune arrays to have size num."""
    return greedy_errors[:num], proj_matrix[:num]


def _sq_errs_rel(errs, proj_vector):
    """Square of projection errors from precomputed projection coefficients.

    This function takes advantage of an orthonormalized basis and a normalized
    training set to compute fewer floating-point operations than in the
    non-normalized case.

    Parameters
    ----------
    errs : numpy.array
        Projection errors.
    proj_vector : numpy.ndarray
        Stores the projection coefficients of the training set onto the actual
        basis element.

    Returns
    -------
    proj_errors : numpy.ndarray
        Squared projection errors.
    """
    return np.subtract(errs, np.abs(proj_vector) ** 2)


def _sq_errs_abs(proj_vector, basis_element, dot_product, diff_training):
    """Square of projection errors from precomputed projection coefficients.

    Since the training set is not a-priori normalized, this function computes
    errors computing the squared norm of the difference between training set
    and the approximation. This method trades accuracy by memory.

    Parameters
    ----------
    proj_vector : numpy.ndarray
        Stores projection coefficients of training functions onto the actual
        basis.
    basis_element : numpy.ndarray
        Actual basis element.
    dot_product : arby.Integration.dot
        Inherited dot product.
    diff_training : numpy.ndarray
        Difference between training set and projected set aiming to be
        actualized.

    Returns
    -------
    proj_errors : numpy.ndarray
        Squared projection errors.
    diff_training : numpy.ndarray
        Actualized difference training set and projected set.
    """
    diff_training = np.subtract(
        diff_training, np.tensordot(proj_vector, basis_element, axes=0)
    )
    return np.real(dot_product(diff_training, diff_training)), diff_training


def _gs_one_element(h, basis, integration, max_iter=3):
    """Orthonormalize a function against an orthonormal basis."""
    norm = integration.norm(h)
    e = h / norm

    for _ in range(max_iter):
        for b in basis:
            e -= b * integration.dot(b, e)
        new_norm = integration.norm(e)
        if new_norm / norm > 0.5:
            break
        norm = new_norm
    else:
        raise StopIteration("Max number of iterations reached ({max_iter}).")

    return e / new_norm, new_norm


seed1 = 12345
rng1 = np.random.default_rng(seed1)


def select_child_node(parameter, node):
    """Select child node.

    The function selects a child node from a binary tree structure,
    given a parameter. The function calculates the distances
    between the parameter and two anchors, stored in the node,
    and returns the child node with the closest anchor. If the
    distances are equal, a random choice is made.

    Parameters
    ----------
    parameter : np.ndarray
        parameter to evaluate by the sustitute model in a subspace
    node : np.ndarray
        node where the parameter is located in the subspace
    """
    # node : se da la raiz del arbol binario para realizar la evaluación.
    # parameter : parámetro a evaluar por el modelo sustituto de un subespacio.

    anchor_0 = node.train_parameters[node.idx_anchor_0]
    anchor_1 = node.train_parameters[node.idx_anchor_1]

    dist_anchor_0 = np.linalg.norm(anchor_0 - parameter)  # 2-norm used.
    dist_anchor_1 = np.linalg.norm(anchor_1 - parameter)  # 2-norm used.

    if dist_anchor_0 < dist_anchor_1:
        if node.children[0].name[-1] == 0:
            child = node.children[0]
        else:
            child = node.children[1]
    elif dist_anchor_0 > dist_anchor_1:
        # child = node.children[1]
        if node.children[0].name[-1] == 1:
            child = node.children[0]
        else:
            child = node.children[1]
    else:
        # para distancias iguales se realiza una elección aleatoria.
        if rng1.integers(2):
            child = node.children[0]
        else:
            child = node.children[1]
    return child


def normalize_set(array, domain, rule="riemann"):
    """Normalize set.

    Normalize a set of functions or arrays.

    Parameters
    ----------
    array :  np.ndarray
        Set of functions to normalize.
    domain :  np.ndarray
        Physical domain of the set of functions.
    rule : str, optional
        Integration rule used to calculate the norm (default="riemann"),
            by default "riemann"

    Returns
    -------
    np.ndarray
        Normalized set of functions.
    """

    integration = integrals.Integration(domain, rule)
    norms = integration.norm(array)

    return np.array(
        [
            h if np.allclose(h, 0, atol=1e-15) else h / norms[i]
            for i, h in enumerate(array)
        ]
    )


def _error(h1, h2, domain, rule="riemann"):
    """Error function.

    The error is computed in the L2 norm (continuous case) or the 2-norm
    (discrete case), that is, ||h1 - h2||^2.

    Parameters
    ----------
    h1 : np.ndarray
    h2 : np.ndarray
    domain : np.ndarray
    rule : integration rule used to compute the error (default="riemann")
    """
    integration = integrals.Integration(domain, rule)
    diff = h1 - h2
    return np.real(integration.dot(diff, diff))


def _validate_parameters(input_value):
    """Validate parameters.

    This function validate the parameters
    """
    if not isinstance(input_value, np.ndarray):
        raise TypeError(
            "The input parameters must be a numpy array."
            "Got instead type {}".format(type(input_value))
        )


def _validate_training_set(input_value):
    """Validate training set.

    This function validate the parameters
    """
    if not isinstance(input_value, np.ndarray):
        raise TypeError(
            "The input training_set must be a numpy array."
            "Got instead type {}".format(type(input_value))
        )


def _validate_physical_points(input_value):
    """Validate physical points.

    This function validate the parameters
    """
    if not isinstance(input_value, np.ndarray):
        raise TypeError(
            "The input physical_points must be a numpy array."
            "Got instead type {}".format(type(input_value))
        )
