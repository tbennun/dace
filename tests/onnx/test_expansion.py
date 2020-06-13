import dace
from dace.libraries.onnx.nodes.onnx_op import Conv

def test_expansion():
    sdfg = dace.SDFG("test_expansion")
    sdfg.add_array("X_arr", (2, 2), dace.float32)
    state = sdfg.add_state()
    access = state.add_access("X_arr")
    c = Conv("Conv")
    state.add_node(c)
    state.add_edge(access, None, c, "X", sdfg.get_array_memlet("X_arr"))
    sdfg.compile()

if __name__ == '__main__':
    test_expansion()
