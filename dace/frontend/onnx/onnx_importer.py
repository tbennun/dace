from itertools import chain, repeat

import numpy as np
import onnx
from onnx import numpy_helper

import dace
from dace.sdfg import SDFG, SDFGState
from dace.libraries.onnx.converters import convert_attribute_proto, onnx_tensor_type_to_dace_type
from dace.libraries.onnx import get_onnx_node, has_onnx_node, ONNXParameterType
from dace.dtypes import AccessType
import dace.sdfg.nodes as nd


def _nested_HasField(obj, full_attr):
    """Performs a nested hasattr check, separating attr on dots."""
    attrs = full_attr.split(".")
    for attr in attrs:
        if obj.HasField(attr):
            obj = getattr(obj, attr)
        else:
            return False
    return True


class ONNXModel:
    """Loads an ONNX model into an SDFG."""
    def __init__(self, name, model: onnx.ModelProto):
        """
        Constructs a new ONNXImporter
        :param name: the name for the SDFG
        :param model: the model to import
        """

        graph: onnx.GraphProto = model.graph

        self.sdfg = SDFG(name)
        self.state = self.sdfg.add_state()

        # Add all values to the SDFG, check for unsupported ops
        ##########################################

        self.value_infos = {}

        self.inputs = set()
        self.outputs = set()

        for value, is_input in chain(
                zip(graph.input, repeat(True)), zip(graph.output,
                                                   repeat(False))):
            if not value.HasField("name"):
                raise ValueError("Got input or output without name")
            if is_input:
                self.inputs.add(value.name)
            else:
                self.outputs.add(value.name)

            self.value_infos[value.name] = value
            self._add_value_info(value)

        for value in graph.value_info:
            if not value.HasField("name"):
                raise ValueError("Got input or output without name")
            if value.name not in self.value_infos:
                self.value_infos[value.name] = value
                self._add_value_info(value)

        # add weights
        self.weights = {}
        for init in graph.initializer:
            self._add_constant_tensor(init)

        access_nodes = {}
        for i, node in enumerate(graph.node):
            if not has_onnx_node(node.op_type):
                raise ValueError("Unsupported ONNX operator: '{}'".format(
                    node.op_type))

            # extract the op attributes
            # TODO check HasField name here
            op_attributes = {
                attribute_proto.name: convert_attribute_proto(attribute_proto)
                for attribute_proto in node.attribute
            }

            if node.HasField("name"):
                node_name = node.name
            else:
                node_name = node.op_type + "_" + str(i)

            # construct the dace node
            op_node = get_onnx_node(node.op_type)(node_name, **op_attributes)
            self.state.add_node(op_node)

            for param_idx, (name, is_input) in chain(
                    enumerate(zip(node.input, repeat(True))),
                    enumerate(zip(node.output, repeat(False)))):
                if self._clean_array_name(name) not in self.sdfg.arrays: #and self._clean_array_name(name) not in self.sdfg.constants_prop:
                    raise ValueError(
                        "Could not find array with name '{}'".format(name))


                # get the access node
                if name in access_nodes:
                    access = access_nodes[name]
                    self._update_access_type(access, is_input)
                else:
                    access = nd.AccessNode(
                        self._clean_array_name(name), AccessType.ReadOnly
                        if is_input else AccessType.WriteOnly)
                    self.state.add_node(access)
                    access_nodes[name] = access

                # get the connector name
                params = op_node.schema.inputs if is_input else op_node.schema.outputs
                params_len = len(params)
                if param_idx > params_len:
                    # this is a variadic parameter. Then the last parameter of the parameter must be variadic.
                    if params[-1].param_type != ONNXParameterType.Variadic:
                        raise ValueError(
                            "Expected the last {i_or_o} parameter to be variadic,"
                            " since the {i_or_o} with idx {param_idx} has more parameters than the schema ({params_len})"
                            .format(i_or_o="input" if is_input else "output",
                                    param_idx=param_idx,
                                    params_len=params_len))
                    conn_name = params[-1].name + "__" + str(param_idx -
                                                             params_len)
                elif params[param_idx].param_type == ONNXParameterType.Variadic:
                    # this is a variadic parameter, and it is within the range of params, so it must be the first
                    # instance of a variadic parameter
                    conn_name = params[param_idx].name + "__0"
                else:
                    conn_name = params[param_idx].name

                if self._clean_array_name(name) in self.sdfg.arrays:
                    data_desc = self.sdfg.arrays[self._clean_array_name(name)]
                #else:
                #    data_desc, _ = self.sdfg.constants_prop[self._clean_array_name(name)]

                # add the connector if required, and add an edge
                if is_input:
                    if conn_name not in op_node.in_connectors:
                        op_node.add_in_connector(conn_name)
                    self.state.add_edge(
                        access, None, op_node, conn_name,
                        dace.Memlet.from_array(
                            self._clean_array_name(name), data_desc))
                else:
                    if conn_name not in op_node.out_connectors:
                        op_node.add_out_connector(conn_name)

                    self.state.add_edge(
                        op_node, conn_name, access, None,
                        dace.Memlet.from_array(
                            self._clean_array_name(name), data_desc))

    @staticmethod
    def _update_access_type(node: dace.nodes.AccessNode, is_input: bool):
        if node.access == AccessType.ReadOnly and not is_input:
            node.access = AccessType.ReadWrite
        elif node.access == AccessType.WriteOnly and is_input:
            node.access = AccessType.ReadWrite

    def _add_constant_tensor(self, tensor: onnx.TensorProto):
        if not tensor.HasField("name"):
            raise ValueError("Got tensor without name")


        if not tensor.HasField("data_type"):
            raise ValueError("Initializer tensor '{}' has no type".format(tensor.name))

        if len(tensor.dims) == 0:
            # this is a scalar
            dims = ()
        else:
            dims = [d for d in tensor.dims]

        self.weights[tensor.name] = numpy_helper.to_array(tensor)
        self.sdfg.add_array(self._clean_array_name(tensor.name), dims, onnx_tensor_type_to_dace_type(tensor.data_type))

    def _add_value_info(self, value_info: onnx.ValueInfoProto):
        if not value_info.HasField("name"):
            raise ValueError("Got value without name")

        name = value_info.name

        if (not _nested_HasField(value_info, "type.tensor_type.shape"))\
                or (len(value_info.type.tensor_type.shape.dim) == 0):
            raise ValueError(
                "Value '{}' does not have a shape in this graph."
                " Please run shape inference before importing.".format(name))

        tensor_type = value_info.type.tensor_type

        if not tensor_type.HasField("elem_type"):
            raise ValueError(
                "Value '{}' does not have a type in this graph."
                " Please run type inference before importing.".format(name))

        shape = []
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                shape.append(d.dim_value)
            elif d.HasField("dim_param"):
                raise NotImplementedError(
                    "Symbolic dimensions are currently not supported.")
            else:
                raise ValueError(
                    "Value '{}' does not have a shape in this graph."
                    " Please run shape inference before importing.".format(
                        name))

        if len(shape) == 0:
            self.sdfg.add_scalar(self._clean_array_name(name),
                                 dtype=onnx_tensor_type_to_dace_type(
                                     tensor_type.elem_type))
        else:
            self.sdfg.add_array(self._clean_array_name(name),
                                shape=shape,
                                dtype=onnx_tensor_type_to_dace_type(
                                    tensor_type.elem_type))

    def get_sdfg_arrays(self):
        """ Return arrays that need to be passed to the SDFG before running. This includes weights and zero initialized
            intermediate values
        """

        # these values will be zeroed out
        intermediate_values = set(self.value_infos).difference(self.inputs).difference(self.outputs).difference(self.weights)
        result = {}
        for val in intermediate_values:
            clean_name = self._clean_array_name(val)
            arr = self.sdfg.arrays[clean_name]
            result[clean_name] = np.zeros(arr.shape, dtype=arr.dtype.as_numpy_dtype())

        # add the weights
        result.update({
        })
        for name, arr in self.weights.items():
            if len(arr.shape) == 0:
                result[self._clean_array_name(name)] = arr.copy().reshape([1,])
            else:
                result[self._clean_array_name(name)] =arr.copy()

        return result


    @staticmethod
    def _clean_array_name(name: str) -> str:
        """Modifies a onnx array name that is potentially invalid in dace
           to make it valid"""
        return "ONNX_" + name.replace(".", "__")