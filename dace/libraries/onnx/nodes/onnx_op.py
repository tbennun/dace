from abc import ABC, abstractmethod
from itertools import chain, repeat

import onnx

import dace
import dace.sdfg.nodes as nd
from dace import SDFG, SDFGState
from dace.properties import make_properties, Property, ListProperty
from dace.transformation.pattern_matching import ExpandTransformation
from dace.libraries.standard.nodes.code import _get_inputs_and_outputs
from dace.libraries.onnx.schema import ONNXSchema, ONNXAttributeType, _ATTR_TYPE_TO_PYTHON_TYPE, ONNXParameterType
from dace.libraries.onnx import ONNXRuntime


def get_missing_arguments_message(function_name, missing_arguments,
                                  argument_type):
    names = list(map(lambda x: "'" + x + "'", missing_arguments))

    if len(missing_arguments) == 1:
        arglist = names[0]
    else:
        arglist = ", ".join(names[:-1]) + ", and " + names[-1]

    return "{function_name} missing {num_missing} required {argument_type}{s}: {arglist}'".format(
        function_name=function_name,
        num_missing=len(missing_arguments),
        argument_type=argument_type,
        s='' if len(missing_arguments) == 1 else 's',
        arglist=arglist)


def parse_variadic_param(param):
    split = param.split('__')
    if len(split) != 2:
        raise ValueError(
            "Unable to parse variadic parameter '{}'".format(param))
    name = split[0]
    number = split[1]

    if number[0] == '0':
        raise ValueError(
            "Variadic parameters must not be numbered with leading zeroes, got: '{}'"
            .format(number))

    number = int(number)
    if number < 0:
        raise ValueError(
            "Variadic parameters numberings must be greater than zero, got: '{}'"
            .format(number))
    return name, number


class ONNXOp(nd.LibraryNode, ABC):
    """ Abstract superclass for all ONNX ops"""

    # Global properties
    # these two are filled out in the generated constructor
    implementations = {}
    default_implementation = None

    # Object fields
    schema = Property(dtype=ONNXSchema,
                      desc="The operator's ONNX OpSchema",
                      allow_none=True)

    def validate(self, sdfg: SDFG, state: SDFGState):
        in_edges = state.in_edges(self)
        out_edges = state.out_edges(self)

        # check that we have all required in_edges
        ##########################################
        required_inputs = {
            inp.name
            for inp in schema.inputs
            if inp.param_type == ONNXParameterType.Single
        }
        passed_inputs = {inp.dst_conn
                         for inp in in_edges if '__' not in inp
                         }  # we will test variadic inputs separately
        known_inputs = {inp.name for inp in schema.inputs}

        missing_inputs = required_inputs.difference(passed_inputs)
        if len(missing_inputs) > 0:
            raise ValueError(
                get_missing_arguments_message(self.schema.name, missing_inputs,
                                              "input"))

        # check that we have all required out_edges
        ##########################################
        required_outputs = {
            outp.name
            for outp in schema.outputs
            if outp.param_type == ONNXParameterType.Single
        }
        passed_outputs = {
            outp.src_conn
            for outp in out_edges if '__' not in outp
        }  # we will test variadic inputs separately
        known_outputs = {outp.name for outp in schema.outputs}

        missing_outputs = required_outputs.difference(passed_outputs)
        if len(missing_outputs) > 0:
            raise ValueError(
                get_missing_arguments_message(self.schema.name, missing_outputs,
                                              "output"))

        # check that we have no unknown in edges
        ##########################################
        unknown_inputs = passed_inputs.difference(known_inputs)
        if len(unknown_inputs) > 0:
            raise TypeError("{} got an unexpected argument '{}'".format(
                self.schema.name,
                list(unknown_inputs)[0]))

        # check that we have no unknown out edges
        ##########################################
        unknown_outputs = passed_outputs.difference(known_outputs)
        if len(unknown_outputs) > 0:
            raise TypeError("{} got an unexpected argument '{}'".format(
                self.schema.name,
                list(unknown_outputs)[0]))

        # check variadic params
        ##########################################
        variadic_inputs = {
            inp.name
            for inp in schema.inputs
            if inp.param_type == ONNXParameterType.Variadic
        }
        passed_variadic_inputs = {
            edge.dst_conn
            for edge in in_edges if '__' in edge
        }

        for param in passed_variadic_inputs:
            name, number = parse_variadic_param(param)
            if name not in variadic_inputs:
                raise ValueError(
                    "{} got an unexpected variadic argument '{}'".format(
                        self.schema.name, param))

        variadic_outputs = {
            outp.name
            for outp in schema.outputs
            if outp.param_type == ONNXParameterType.Variadic
        }
        passed_variadic_outputs = {
            edge.src_conn
            for edge in out_edges if '__' in edge
        }

        for param in passed_variadic_outputs:
            name, number = parse_variadic_param(param)
            if name not in variadic_outputs:
                raise ValueError(
                    "{} got an unexpected variadic argument '{}'".format(
                        self.schema.name, param))

        # check that type params solve
        ##########################################

        assigned_params = {}
        for edge, is_input in chain(zip(in_edges, repeat(True)),
                                    zip(out_edges, repeat(False))):
            conn_name = edge.dst_conn if is_input else edge.src_conn
            matching = [
                inp for inp in (
                    self.schema.inputs if is_input else self.schema.outputs)
                if inp.name == conn_name
            ]

            if len(matching) != 1:
                raise ValueError(
                    "Expected to find one {} parameter in schema with name '{}', but found {}"
                    .format("input" if is_input else "output", conn_name,
                            len(matching)))
            matched = matching[0]

            edge_data = edge.data.data
            edge_dtype = sdfg.arrays[edge_data].dtype
            if matched.type_str in assigned_params and assigned_params[
                    matched.type_str] != edge_dtype:
                raise ValueError(
                    "Could not solve type constraints on {name}; excepted type '{expected}' for {param_type} '{conn_name}', got type '{actual}'"
                    .format(name=self.schema.name,
                            expected=assigned_params[matched.type_str],
                            param_type="input" if is_input else "output",
                            conn_name=matched.name,
                            actual=edge_dtype))

            # otherwise, matched.type_str was not assigned a type yet: try to assign it
            cons = schema.type_constraints[matched.type_str]
            if edge_dtype not in cons.types:
                raise ValueError(
                    "Unexpected type on {name}; expected type in '{possible}' for {param_type} '{conn_name}', got type '{actual}'"
                    .format(name=self.schema.name,
                            possible=cons.types,
                            param_type="input" if is_input else "output",
                            conn_name=matched.name,
                            actual=edge_dtype))
            assigned_params[matched.type_str] = edge_dtype

        # check that we have all required attributes
        ##########################################
        required_attrs = {
            name
            for name, attr in dace_schema.attributes.items() if attr.required
        }
        for attr in required_attrs:
            if getattr(self, attr) is None:
                raise ValueError(
                    "Expected value for required attribute '{}' on {}, got None"
                    .format(attr, self.schema.name))


for schema in onnx.defs.get_all_schemas():
    try:
        dace_schema = ONNXSchema.from_onnx_proto(schema)
    except Exception as e:
        print("Import of {} failed: {}".format(schema.name, e))
        continue

    docstring = dace_schema.doc
    attrs = {}
    attrs['__doc__'] = docstring
    attrs['schema'] = dace_schema


    # add properties for each op attribute
    for name, attr in dace_schema.attributes.items():
        if attr.type in [
                ONNXAttributeType.Int, ONNXAttributeType.String,
                ONNXAttributeType.Float
        ]:
            attrs[name] = Property(dtype=_ATTR_TYPE_TO_PYTHON_TYPE[attr.type],
                                   desc=attr.description,
                                   allow_none=True,
                                   default=None if attr.default_value is None
                                   else attr.default_value)
        elif attr.type in [
                ONNXAttributeType.Ints, ONNXAttributeType.Strings,
                ONNXAttributeType.Floats
        ]:
            attrs[name] = ListProperty(
                element_type=_ATTR_TYPE_TO_PYTHON_TYPE[attr.type],
                desc=attr.description,
                allow_none=True,
                default=None
                if attr.default_value is None else attr.default_value)
        else:
            raise NotImplementedError(
                "Got unsupported ONNXAttributeType: {}".format(attr.type))

    required_attrs = {
        name
        for name, attr in dace_schema.attributes.items() if attr.required
    }

    def __init__(self, name, *args, location=None, **op_attributes):
        super(ONNXOp, self).__init__(
            name,
            location=location,
            # add required parameters as in/out connectors
            inputs={
                inp.name
                for inp in self.schema.inputs
                if inp.param_type == ONNXParameterType.Single
            },
            outputs={
                out.name
                for out in self.schema.outputs
                if out.param_type == ONNXParameterType.Single
            })

        if len(args) > 0:
            raise TypeError(
                "__init__() takes 2 positional arguments but {} were given".
                format(2 + len(args)))

        missing_arguments = required_attrs.difference(op_attributes)
        if len(missing_arguments) > 0:

            raise TypeError(
                get_missing_arguments_message("__init__()", missing_arguments,
                                              "keyword-only argument"))

        unknown_attrs = set(op_attributes).difference(self.schema.attributes)
        if len(unknown_attrs) > 0:
            raise TypeError(
                "__init__() got an unexpected keyword argument '{}'".format(
                    list(unknown_attrs)[0]))

        for name, attr in op_attributes.items():
            setattr(self, name, attr)

        # Inline the class such that "self" is included in the expansion
        @dace.library.expansion
        class Expansion(ExpandTransformation):
            environments = [ONNXRuntime]

            @staticmethod
            def expansion(node, state: SDFGState, sdfg: SDFG):
                # Extract input and output array views (as generated by memlets)
                inputs, outputs = _get_inputs_and_outputs(sdfg, state, node)
                # Generate the appropriate code
                # Replace this node with a C++ tasklet
                return nd.Tasklet('onnx_code',
                                  set(inputs.keys()),
                                  set(outputs.keys()),
                                  "onnx:NodeProto proto;",
                                  language=dace.dtypes.Language.CPP)

        self.implementations['default'] = Expansion
        Expansion._match_node = self
        self.implementation = 'default'
        self.register_implementation('default', Expansion)

    attrs['__init__'] = __init__

    cls = type(dace_schema.name, (ONNXOp, ), attrs)

    vars()[schema.name] = dace.library.node(cls)
