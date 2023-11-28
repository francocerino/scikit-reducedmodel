""" construccion de rb y surrogate con particion de dominio (hp-greedy).

no implementa eim, regresiones en espacio de parámetros y predicciones

funciones:
	-partition: realiza particion de dominio
	-hpgreedy: rb con particion de dominio. algoritmo recursivo.

code with "##": implementation for ROM only. 

this code uses arby.
there are 2 != ways: with ReducedOrderModel and reduced_basis.
ReducedOrderModel by design does not allow normalize the training space (TS)
"""

import numpy as np
import arby
from anytree import Node, RenderTree


def hpgreedy(
    ts,
    parameters,
    times,
    greedy_tol,
    N_max,
    L_max,
    normalize,
    parent=None,
    node_idx=0,
    l=0,
    integration_rule="riemann",
):
    # hpgreedy recursivo con arby
    # la variable 'normalize' queda obsoleta en el caso de usar ROM y no RB.

    assert N_max > 0  # no tiene sentido N_max == 0.

    # create node
    if parent != None:
        node = Node(
            name=parent.name + (node_idx,),
            parent=parent,
            parameters_ts=parameters,
        )
    else:
        node = Node(name=(node_idx,), parameters_ts=parameters)

    # rb for given domain
    rb = arby.basis.reduced_basis(
        training_set=ts,
        physical_points=times,
        integration_rule=integration_rule,
        greedy_tol=greedy_tol,
        normalize=normalize,
    )

    ##model = arby.rom.ReducedOrderModel(ts, times, parameters,
    ##                                   integration_rule = integration_rule,
    ##                                   greedy_tol = greedy_tol,
    ##                                   poly_deg = 3)

    # tomo error de rb con N_max elementos o menos
    shape_rb_no_N_max = rb.basis.data.shape[0]
    ##shape_rb_no_N_max = model.basis_.data.shape[0]

    if shape_rb_no_N_max >= N_max:
        shape_rb_with_N_max = N_max
        err_Nmax = rb.errors[N_max - 1]
        ##err_Nmax = model.greedy_errors_[N_max-1]
    else:
        shape_rb_with_N_max = shape_rb_no_N_max
        err_Nmax = rb.errors[-1]
        ##err_Nmax = model.greedy_errors_[-1]

    setattr(node, "idx_seed", rb.indices[0])

    # hp greedy recursion
    if l < L_max and greedy_tol < err_Nmax and shape_rb_with_N_max >= 2:
        # partición de domminio y rb para cada uno de ellos
        # print(node.name," -> l=", l ,"err=",err_Nmax, "shape_rb_with_N_max=",shape_rb_with_N_max)
        idx_anchor0 = rb.indices[
            0
        ]  # rb.indices -> indices de los elementos del ts que van a la rb
        idx_anchor1 = rb.indices[1]
        ##idx_anchor0 = model.greedy_indices_[0]
        ##idx_anchor1 = model.greedy_indices_[1]

        setattr(node, "idx_anchor0", idx_anchor0)
        setattr(node, "idx_anchor1", idx_anchor1)

        idxs_subspace0, idxs_subspace1 = partition(
            parameters, idx_anchor0, idx_anchor1
        )

        child0 = hpgreedy(
            ts[idxs_subspace0],
            parameters[idxs_subspace0],
            times,
            greedy_tol,
            N_max,
            L_max,
            normalize,
            parent=node,
            node_idx=0,
            l=l + 1,
        )

        child1 = hpgreedy(
            ts[idxs_subspace1],
            parameters[idxs_subspace1],
            times,
            greedy_tol,
            N_max,
            L_max,
            normalize,
            parent=node,
            node_idx=1,
            l=l + 1,
        )

        child0.parent = node
        child1.parent = node
        return node

    else:
        # no hay particion de dominio (el nodo es una hoja)
        # se toma como máximo N_max elementos
        # en caso de querer agregar rb en todo nodo, esto va antes del if de hpgreedy.

        basis_domain = rb.basis.data[:N_max]
        parameters_domain = rb.indices[:N_max]
        rb_errors = rb.errors[:N_max]
        ##basis_domain = model.basis_.data[:N_max]
        ##parameters_domain = model.greedy_indices_[:N_max]
        ##rb_errors = model.greedy_errors_[:N_max]
        # print("leaf:  ",node.name," -> l=", l ,"err=",err_Nmax, "shape_rb_with_N_max=",shape_rb_with_N_max)
        setattr(node, "rb", basis_domain)
        setattr(node, "rb_parameters_idxs", parameters_domain)
        setattr(node, "rb_errors", rb_errors)
        setattr(node, "err_Nmax", err_Nmax)
        # for rb
        setattr(node, "rb_data", rb)
        # for rom:
        ##setattr(node,"model",model)

        return node


######


def partition(parameters, idx_anchor0, idx_anchor1):
    # devuelve indices de parametros que corresponden a cada subespacio

    # parameters: array of parameters from the domain of the problem
    # anchor1: first greedy parameter of the space to divide.
    # anchor2: second greedy parameter of the space to divide.

    anchor0 = parameters[idx_anchor0]
    anchor1 = parameters[idx_anchor1]

    assert not np.array_equal(anchor0, anchor1)

    seed = 12345
    rng = np.random.default_rng(seed)

    # caso de arby con ts normalizado:
    # la semilla es el primer elemento,
    # por lo tanto, si quiero que los anchors vayan primero:

    ##idxs_subspace0 = []
    ##idxs_subspace1 = []
    idxs_subspace0 = [idx_anchor0]
    idxs_subspace1 = [idx_anchor1]

    # y usar a continuación del for --> if idx != idx_anchor0 and idx != idx_anchor1:
    # sirve para el caso de usar normalize = True en reduced_basis()
    # da error en splines porque scipy pide los parametros de forma ordenada

    for idx, parameter in enumerate(parameters):
        if idx != idx_anchor0 and idx != idx_anchor1:
            dist_anchor0 = np.linalg.norm(anchor0 - parameter)  # 2-norm
            dist_anchor1 = np.linalg.norm(anchor1 - parameter)
            if dist_anchor0 < dist_anchor1:
                idxs_subspace0.append(idx)
            elif dist_anchor0 > dist_anchor1:
                idxs_subspace1.append(idx)
            else:  # para distancias iguales se realiza una elección aleatoria.
                if rng.integers(2):
                    idxs_subspace0.append(idx)
                else:
                    idxs_subspace1.append(idx)

    return np.array(idxs_subspace0), np.array(idxs_subspace1)


def test_partition(parameters, idxs_subspace1, idxs_subspace2):
    # test para para los ts de subespacios resultantes.
    # interseccion vacia
    # union da el ts del espacio original.

    return set(idxs_subspace2) & set(idxs_subspace1) == set() and set(
        idxs_subspace2
    ) | set(idxs_subspace1) == set(range(len(parameters)))


######

# show some info about trees with these functions


def visual_tree(tree):
    for pre, fill, node in RenderTree(tree):
        print("%s%s" % (pre, node.name))


def leaves_rb_size(tree):
    print("rb size for each leave:")
    sum_ = 0
    for i in range(len(tree.leaves)):
        # print(np.sort(tree.leaves[i].parameters_ts))
        len_ = len(tree.leaves[i].rb_parameters_idxs)
        err_Nmax = tree.leaves[i].err_Nmax
        print(
            f"{len_}, err_Nmax = {err_Nmax}, compression rate = {round(len(tree.leaves[i].parameters_ts)/len_,4)}"
        )
        print(f"rb errors : {tree.leaves[i].rb_errors}")
        print("\n")
        sum_ += len_

    print(f"total of elements chosen by hpgreedy = {sum_}\n")


######

# funciones para realizar una evaluacion en hpgreedy
# se recorre el arbol hasta encontrar el subespacio que le corresponde
# y se evalua el modelo sustituto del subespacio en el parametro dado.


def select_child_node(node, parameter):
    # node : se da la raiz del arbol binario para realizar la evaluación.
    # parameter : parámetro a evaluar por el modelo sustituto de un subespacio.

    seed = 12345
    rng = np.random.default_rng(seed)

    anchor0 = node.parameters_ts[node.idx_anchor0]
    anchor1 = node.parameters_ts[node.idx_anchor1]

    dist_anchor0 = np.linalg.norm(anchor0 - parameter)  # 2-norm.
    dist_anchor1 = np.linalg.norm(anchor1 - parameter)

    if dist_anchor0 < dist_anchor1:
        if node.children[0].name[-1] == 0:
            child = node.children[0]
        else:
            child = node.children[1]
    elif dist_anchor0 > dist_anchor1:
        # child = node.children[1]
        if node.children[0].name[-1] == 1:
            child = node.children[0]
        else:
            child = node.children[1]
    else:
        # para distancias iguales se realiza una elección aleatoria.
        if rng.integers(2):
            child = node.children[0]
        else:
            child = node.children[1]
    return child


def prediction_hpgreedy(node, parameter):
    if not node.is_leaf:
        # compare parameter with two anchors
        child = select_child_node(node, parameter)
        return prediction_hpgreedy(child, parameter)
    else:
        result = node.model.surrogate(parameter)
        return result


def proj_error_hpgreedy(node, parameter, gw):
    # N: proyectar sobre los N primeros elementos de la rb (armada sin N_max).
    # error: square of the projection error (arby).

    # busca el subespacio correspondiente a la solución dada a través
    # del árbol y luego imprime el error de proyección.

    if not node.is_leaf:
        # compare parameter with two anchors
        child = select_child_node(node, parameter)
        return proj_error_hpgreedy(child, parameter, gw)
    else:
        ##result = node.model.basis_.projection_error(gw,(N,))
        #         if len(node.rb_errors)<node.rb_data.basis.Nbasis_: print(True)
        result = node.rb_data.basis.projection_error(
            gw, (len(node.rb_errors),)
        )  # Square of proj. err.
        # if not result == node.rb_data.basis.projection_error(gw,(N,)):
        # 	print("Warning: different errors. check code.")
        return result


#######


def node_anchors(node):
    return (
        node.parameters_ts[node.idx_anchor0],
        node.parameters_ts[node.idx_anchor1],
    )


def print_anchors_tree(tree):
    print("\ntree.depth = ", tree.depth)
    # print(node_anchors(tree))

    anchors = {}
    for i in range(tree.height):
        anchors[i] = []

    anchors[tree.depth].append(node_anchors(tree))
    # print("len(node.children) = ", len(tree.children))
    for child in tree.children:
        if len(child.children) == 2:
            print_anchors_tree(child, anchors)


def print_compressions(tree):
    # tasa de compresion
    print("local:")
    ts_chosen_for_rb = 0
    for leaf in tree.leaves:
        ts_chosen_for_rb += len(leaf.rb_parameters_idxs)
        local_parameters_ts = len(leaf.parameters_ts)
        local_parameters_rb = len(leaf.rb_parameters_idxs)
        print(
            local_parameters_ts,
            local_parameters_rb,
            local_parameters_ts / local_parameters_rb,
        )
    print("\nglobal:")
    print(n_train, ts_chosen_for_rb, n_train / ts_chosen_for_rb)


def dim_rbs_hpgreedy(tree):
    dim_rbs = []
    for leaf in tree.leaves:
        elems = len(leaf.rb_parameters_idxs)
        # print(f" #rb = {elems}, err = {leaf.rb_errors[-1]}")
        dim_rbs.append(elems)
    return dim_rbs


def errors_hpgreedy(parameters, waveforms, tree):
    errors = []
    for q, wf in zip(parameters, waveforms):
        error = np.real(proj_error_hpgreedy(tree, q, wf))
        errors.append(error)
    return errors


def errors_hpgreedy_leaves(parameters, waveforms, tree):
    errors_rb_leaves = {}
    for q, wf in zip(parameters, waveforms):
        error, node_name, rb_dim = proj_error_hpgreedy_(tree, q, wf)
        error = np.real(error)
        if node_name in errors_rb_leaves:
            errors_rb_leaves[node_name]["errors"].append(error)
            errors_rb_leaves[node_name]["errors"] = [
                np.max(errors_rb_leaves[node_name]["errors"])
            ]
        else:
            errors_rb_leaves[node_name] = {}
            errors_rb_leaves[node_name]["errors"] = [error]
            errors_rb_leaves[node_name]["rb_dim"] = rb_dim
    return errors_rb_leaves


def max_errs_leaves_train_data(tree, ts_train, q_train):
    max_errs_leaves = []
    for leaf in tree.leaves:
        params = leaf.parameters_ts
        errs_leaf = []
        for q in params:
            wf = ts_train[idx_parameter(q, q_train)]
            err = proj_error_hpgreedy(tree, q, wf)
            errs_leaf.append(err)
        max_errs_leaves.append(np.real(np.max(errs_leaf)))
    return max_errs_leaves


def leaves_depth(tree):
    depths = []
    for leaf in tree.leaves:
        depths.append(leaf.depth)
    return depths


def idx_parameter(param, array):
    for idx, p in enumerate(array):
        if np.array_equal(p, param):
            return idx
