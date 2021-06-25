from ..tensor_contraction_layers_plus import TCLplus
import tensorly as tl
from tensorly import testing
tl.set_backend('pytorch')


def test_tcl_plus():
    random_state = 12345
    rng = tl.check_random_state(random_state)
    batch_size = 2
    in_shape = (4, 5, 6)
    out_shape = (2, 3, 5)
    sum_terms = 4
    data = tl.tensor(rng.random_sample((batch_size, ) + in_shape))

    expected_shape = (batch_size, ) + out_shape
    tcl_plus = TCLplus(input_shape=in_shape, rank=out_shape, sum_terms=sum_terms, bias=False)
    res = tcl_plus(data)
    testing.assert_(res.shape==expected_shape, 
                    msg=f'Wrong output size of TCL, expected {expected_shape} but got {res.shape}')
