# import numpy as np

from skreducedmodel.surrogate import Surrogate

import arby

def test_rom_rb_interface(rom_parameters):
    """Test API consistency."""

    bessel = arby.ReducedOrderModel(
        rom_parameters["training_set"],
        rom_parameters["physical_points"],
        rom_parameters["parameter_points"],
        greedy_tol=1e-14
    )
    basis = bessel.basis_.data
    errors = bessel.greedy_errors_
    projection_matrix = bessel.projection_matrix_
    greedy_indices = bessel.greedy_indices_
    eim = bessel.eim_

    assert len(basis) == 10
    assert len(errors) == 10
    assert len(projection_matrix) == 101
    assert len(greedy_indices) == 10
    assert eim == bessel.basis_.eim_

    rom = Surrogate(greedy_tol=1e-14, lmax = 0)

    rom.fit(training_set = rom_parameters["training_set"],
            parameters = rom_parameters["parameter_points"],
            physical_points = rom_parameters["physical_points"],
        )      

    leaf = rom.base.tree
    rom_errors = leaf.errors
    rom_projection_matrix = leaf.projection_matrix
    rom_greedy_indices = leaf.indices

    assert bessel.basis_.data.shape == leaf.basis.shape
    assert (rom_errors == errors).all()
    assert (rom_projection_matrix == projection_matrix).all()
    assert rom_greedy_indices == greedy_indices
