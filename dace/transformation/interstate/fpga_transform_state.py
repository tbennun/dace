""" Contains inter-state transformations of an SDFG to run on an FPGA. """

import copy
import itertools

import dace
from dace import data, memlet, dtypes, sdfg as sd, subsets, symbolic
from dace.graph import edges, nodes, nxutil
from dace.transformation import pattern_matching


def fpga_update(sdfg, state, depth):
    scope_dict = state.scope_dict()
    for node in state.nodes():
        if (isinstance(node, nodes.AccessNode)
                and node.desc(sdfg).storage == dtypes.StorageType.Default):
            nodedesc = node.desc(sdfg)
            if depth >= 2:
                nodedesc.storage = dtypes.StorageType.FPGA_Local
            else:
                if scope_dict[node]:
                    nodedesc.storage = dtypes.StorageType.FPGA_Local
                else:
                    nodedesc.storage = dtypes.StorageType.FPGA_Global
        if (hasattr(node, "schedule")
                and node.schedule == dace.dtypes.ScheduleType.Default):
            node.schedule = dace.dtypes.ScheduleType.FPGA_Device
        if isinstance(node, nodes.NestedSDFG):
            for s in node.sdfg.nodes():
                fpga_update(node.sdfg, s, depth + 1)


class FPGATransformState(pattern_matching.Transformation):
    """ Implements the FPGATransformState transformation. """

    _state = sd.SDFGState()

    @staticmethod
    def expressions():
        return [nxutil.node_path_graph(FPGATransformState._state)]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        state = graph.nodes()[candidate[FPGATransformState._state]]

        for node in state.nodes():

            if (isinstance(node, nodes.AccessNode)
                    and node.desc(sdfg).storage != dtypes.StorageType.Default):
                return False

            if not isinstance(node, nodes.MapEntry):
                continue

            map_entry = node
            candidate_map = map_entry.map

            # No more than 3 dimensions
            if candidate_map.range.dims() > 3: return False

            # Map schedules that are disallowed to transform to FPGAs
            if (candidate_map.schedule == dtypes.ScheduleType.MPI
                    or candidate_map.schedule == dtypes.ScheduleType.GPU_Device
                    or
                    candidate_map.schedule == dtypes.ScheduleType.FPGA_Device
                    or candidate_map.schedule ==
                    dtypes.ScheduleType.GPU_ThreadBlock):
                return False

            # Recursively check parent for FPGA schedules
            sdict = state.scope_dict()
            current_node = map_entry
            while current_node is not None:
                if (current_node.map.schedule == dtypes.ScheduleType.GPU_Device
                        or current_node.map.schedule ==
                        dtypes.ScheduleType.FPGA_Device
                        or current_node.map.schedule ==
                        dtypes.ScheduleType.GPU_ThreadBlock):
                    return False
                current_node = sdict[current_node]

        return True

    @staticmethod
    def match_to_str(graph, candidate):
        state = graph.nodes()[candidate[FPGATransformState._state]]

        return state.label

    def apply(self, sdfg):
        state = sdfg.nodes()[self.subgraph[FPGATransformState._state]]

        # Find source/sink (data) nodes
        input_nodes = nxutil.find_source_nodes(state)
        output_nodes = nxutil.find_sink_nodes(state)

        fpga_data = {}

        if input_nodes:

            pre_state = sd.SDFGState('pre_' + state.label, sdfg)

            for node in input_nodes:

                if (not isinstance(node, dace.graph.nodes.AccessNode)
                        or not isinstance(node.desc(sdfg), dace.data.Array)):
                    # Only transfer array nodes
                    # TODO: handle streams
                    continue

                array = node.desc(sdfg)
                if node.data in fpga_data:
                    fpga_array = fpga_data[node.data]
                else:
                    fpga_array = sdfg.add_array(
                        'fpga_' + node.data,
                        array.shape,
                        array.dtype,
                        materialize_func=array.materialize_func,
                        transient=True,
                        storage=dtypes.StorageType.FPGA_Global,
                        allow_conflicts=array.allow_conflicts,
                        access_order=array.access_order,
                        strides=array.strides,
                        offset=array.offset)
                    fpga_data[node.data] = fpga_array
                # fpga_node = type(node)(fpga_array)

                pre_node = pre_state.add_read(node.data)
                pre_fpga_node = pre_state.add_write('fpga_' + node.data)
                full_range = subsets.Range(
                    [(0, s - 1, 1) for s in array.shape])
                mem = memlet.Memlet(node.data, full_range.num_elements(),
                                    full_range, 1)
                pre_state.add_edge(pre_node, None, pre_fpga_node, None, mem)

                fpga_node = state.add_read('fpga_' + node.data)
                nxutil.change_edge_src(state, node, fpga_node)
                state.remove_node(node)

            sdfg.add_node(pre_state)
            nxutil.change_edge_dest(sdfg, state, pre_state)
            sdfg.add_edge(pre_state, state, edges.InterstateEdge())

        if output_nodes:

            post_state = sd.SDFGState('post_' + state.label, sdfg)

            for node in output_nodes:

                if (not isinstance(node, dace.graph.nodes.AccessNode)
                        or not isinstance(node.desc(sdfg), dace.data.Array)):
                    # Only transfer array nodes
                    # TODO: handle streams
                    continue

                array = node.desc(sdfg)
                if node.data in fpga_data:
                    fpga_array = fpga_data[node.data]
                else:
                    fpga_array = sdfg.add_array(
                        'fpga_' + node.data,
                        array.shape,
                        array.dtype,
                        materialize_func=array.materialize_func,
                        transient=True,
                        storage=dtypes.StorageType.FPGA_Global,
                        allow_conflicts=array.allow_conflicts,
                        access_order=array.access_order,
                        strides=array.strides,
                        offset=array.offset)
                    fpga_data[node.data] = fpga_array
                # fpga_node = type(node)(fpga_array)

                post_node = post_state.add_write(node.data)
                post_fpga_node = post_state.add_read('fpga_' + node.data)
                full_range = subsets.Range(
                    [(0, s - 1, 1) for s in array.shape])
                mem = memlet.Memlet('fpga_' + node.data,
                                    full_range.num_elements(), full_range, 1)
                post_state.add_edge(post_fpga_node, None, post_node, None, mem)

                fpga_node = state.add_write('fpga_' + node.data)
                nxutil.change_edge_dest(state, node, fpga_node)
                state.remove_node(node)

            sdfg.add_node(post_state)
            nxutil.change_edge_src(sdfg, state, post_state)
            sdfg.add_edge(state, post_state, edges.InterstateEdge())

        for src, _, dst, _, mem in state.edges():
            if mem.data is not None and mem.data in fpga_data:
                mem.data = 'fpga_' + mem.data

        fpga_update(sdfg, state, 0)


pattern_matching.Transformation.register_stateflow_pattern(FPGATransformState)
