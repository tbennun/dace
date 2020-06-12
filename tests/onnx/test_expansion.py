import dace
from dace.libraries.onnx.nodes.onnx_op import Conv

def test_expansion():
    sdfg = dace.SDFG("test_expansion")
    state = sdfg.add_state()
    state.add_node(Conv("Conv"))
    sdfg.compile()

if __name__ == '__main__':
    test_expansion()
