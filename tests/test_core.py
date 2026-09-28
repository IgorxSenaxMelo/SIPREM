import numpy as np

from core.fuzzy_numbers import (
    trap_add, trap_sub, trap_mul_geldermann, defuzz_coa
)
from core.fahp import (
    crisp_to_fuzzy_trap_interp,
    calc_fahp_buckley,
    weights_from_reference_row,
)


def test_fuzzy_basic():
    assert np.allclose(trap_add([1,2,3,4], [1,1,1,1]), [2,3,4,5])
    assert np.allclose(trap_sub([2,3,4,5], [1,1,1,1]), [1,2,3,4])


def test_fahp_scale():
    assert np.allclose(crisp_to_fuzzy_trap_interp(1), [1,1,1,1])
    assert np.allclose(crisp_to_fuzzy_trap_interp(5), [4,4.5,5.5,6])


def test_fahp_e1_shape():
    wc, wf, A, mf = weights_from_reference_row(
        ["Igual importância",
         "Moderadamente mais importante",
         "Moderadamente a fortemente mais importante",
         "Fortemente mais importante",
         "Muito fortemente mais importante"]
    )
    assert wc.shape == (5,)
    assert wf.shape == (5,4)
    assert A.shape == (5,5)
    assert mf.shape == (5,5,4)
    assert np.isclose(wc.sum(), 1.0)
