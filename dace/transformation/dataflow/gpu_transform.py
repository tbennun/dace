""" Contains the GPU Transform Map transformation. """

import copy
import itertools

from dace import data, dtypes, sdfg as sd, subsets as sbs, symbolic
from dace.graph import nodes, nxutil
from dace.transformation import pattern_matching
from dace.properties import Property, make_properties
from dace.config import Config


@make_properties
class GPUTransformMap(pattern_matching.Transformation):
    """ Implements the GPUTransformMap transformation.

        Converts a single map to a GPU-scheduled map and creates GPU arrays
        outside it, generating CPU<->GPU memory copies automatically.
    """

    _maps_transformed = 0
    _arrays_removed = 0

    fullcopy = Property(
        desc="Copy whole arrays rather than used subset",
        dtype=bool,
        default=False)

    _map_entry = nodes.MapEntry(nodes.Map("", [], []))
    _reduce = nodes.Reduce('lambda: None', None)

    @staticmethod
    def expressions():
        return [
            nxutil.node_path_graph(GPUTransformMap._map_entry),
            nxutil.node_path_graph(GPUTransformMap._reduce)
        ]

    @staticmethod
    def can_be_applied(graph, candidate, expr_index, sdfg, strict=False):
        if expr_index == 0:
            map_entry = graph.nodes()[candidate[GPUTransformMap._map_entry]]
            candidate_map = map_entry.map

            # Map schedules that are disallowed to transform to GPUs
            if (candidate_map.schedule == dtypes.ScheduleType.MPI
                    or candidate_map.schedule == dtypes.ScheduleType.GPU_Device
                    or candidate_map.schedule ==
                    dtypes.ScheduleType.GPU_ThreadBlock):
                return False

            # Recursively check parent for GPU schedules
            sdict = graph.scope_dict()
            current_node = map_entry
            while current_node is not None:
                if (current_node.map.schedule == dtypes.ScheduleType.GPU_Device
                        or current_node.map.schedule ==
                        dtypes.ScheduleType.GPU_ThreadBlock):
                    return False
                current_node = sdict[current_node]

            # Ensure that map does not include internal arrays that are allocated
            # on non-default space
            subgraph = graph.scope_subgraph(map_entry)
            for node in subgraph.nodes():
                if (isinstance(node, nodes.AccessNode) and
                        node.desc(sdfg).storage != dtypes.StorageType.Default
                        and node.desc(sdfg).storage !=
                        dtypes.StorageType.Register):
                    return False

            return True
        elif expr_index == 1:
            reduce = graph.nodes()[candidate[GPUTransformMap._reduce]]

            # Map schedules that are disallowed to transform to GPUs
            if (reduce.schedule == dtypes.ScheduleType.MPI
                    or reduce.schedule == dtypes.ScheduleType.GPU_Device
                    or reduce.schedule == dtypes.ScheduleType.GPU_ThreadBlock):
                return False

            # Recursively check parent for GPU schedules
            sdict = graph.scope_dict()
            current_node = sdict[reduce]
            while current_node is not None:
                if (current_node.map.schedule == dtypes.ScheduleType.GPU_Device
                        or current_node.map.schedule ==
                        dtypes.ScheduleType.GPU_ThreadBlock):
                    return False
                current_node = sdict[current_node]

            return True

    @staticmethod
    def match_to_str(graph, candidate):
        if GPUTransformMap._reduce in candidate:
            return str(graph.nodes()[candidate[GPUTransformMap._reduce]])
        else:
            map_entry = graph.nodes()[candidate[GPUTransformMap._map_entry]]
            return str(map_entry)

    def apply(self, sdfg):
        graph = sdfg.nodes()[self.state_id]
        if self.expr_index == 0:
            cnode = graph.nodes()[self.subgraph[GPUTransformMap._map_entry]]
            node_schedprop = cnode.map
            exit_nodes = graph.exit_nodes(cnode)
        else:
            cnode = graph.nodes()[self.subgraph[GPUTransformMap._reduce]]
            node_schedprop = cnode
            exit_nodes = [cnode]

        # Change schedule
        node_schedprop._schedule = dtypes.ScheduleType.GPU_Device
        if Config.get_bool("debugprint"):
            GPUTransformMap._maps_transformed += 1

        gpu_storage_types = [
            dtypes.StorageType.GPU_Global,
            dtypes.StorageType.GPU_Shared,
            dtypes.StorageType.GPU_Stack  #, dtypes.StorageType.CPU_Pinned
        ]

        #######################################################
        # Add GPU copies of CPU arrays (i.e., not already on GPU)

        # First, understand which arrays to clone
        all_out_edges = []
        for enode in exit_nodes:
            all_out_edges.extend(list(graph.out_edges(enode)))
        in_arrays_to_clone = set()
        out_arrays_to_clone = set()
        out_streamarrays = {}
        for e in graph.in_edges(cnode):
            data_node = sd.find_input_arraynode(graph, e)
            if isinstance(data_node.desc(sdfg), data.Scalar):
                continue
            if data_node.desc(sdfg).storage not in gpu_storage_types:
                in_arrays_to_clone.add(data_node)
        for e in all_out_edges:
            data_node = sd.find_output_arraynode(graph, e)
            if isinstance(data_node.desc(sdfg), data.Scalar):
                continue
            if data_node.desc(sdfg).storage not in gpu_storage_types:
                # Stream directly connected to an array
                if sd.is_array_stream_view(sdfg, graph, data_node):
                    datadesc = data_node.desc(sdfg)
                    if datadesc.transient is False:
                        raise TypeError('Non-transient stream-array view are '
                                        'unsupported')
                    # Add parent node to clone
                    out_arrays_to_clone.add(graph.out_edges(data_node)[0].dst)
                    out_streamarrays[graph.out_edges(data_node)[0]
                                     .dst] = data_node

                    # Do not clone stream
                    continue

                out_arrays_to_clone.add(data_node)
        if Config.get_bool("debugprint"):
            GPUTransformMap._arrays_removed += len(in_arrays_to_clone) + len(
                out_arrays_to_clone)

        # Second, create a GPU clone of each array
        cloned_arrays = {}
        in_cloned_arraynodes = {}
        out_cloned_arraynodes = {}
        for array_node in in_arrays_to_clone:
            array = array_node.desc(sdfg)
            if array_node.data in cloned_arrays:
                cloned_array = cloned_arrays[array_node.data]
            else:
                cloned_array = array.clone()
                cloned_array.storage = dtypes.StorageType.GPU_Global
                cloned_array.transient = True
                sdfg.add_datadesc('gpu_' + array_node.data, cloned_array)
                cloned_arrays[array_node.data] = 'gpu_' + array_node.data
            cloned_node = type(array_node)('gpu_' + array_node.data)

            in_cloned_arraynodes[array_node.data] = cloned_node
        for array_node in out_arrays_to_clone:
            array = array_node.desc(sdfg)
            if array_node.data in cloned_arrays:
                cloned_array = cloned_arrays[array_node.data]
            else:
                cloned_array = array.clone()
                cloned_array.storage = dtypes.StorageType.GPU_Global
                cloned_array.transient = True
                sdfg.add_datadesc('gpu_' + array_node.data, cloned_array)
                cloned_arrays[array_node.data] = 'gpu_' + array_node.data
            cloned_node = type(array_node)('gpu_' + array_node.data)

            out_cloned_arraynodes[array_node.data] = cloned_node

        # Third, connect the cloned arrays to the originals
        # TODO(later): Shift indices and create only the necessary sub-arrays
        for array_name, node in in_cloned_arraynodes.items():
            graph.add_node(node)
            for edge in graph.in_edges(cnode):
                if edge.data.data == array_name:
                    graph.remove_edge(edge)
                    newmemlet = copy.copy(edge.data)
                    newmemlet.data = node.data
                    graph.add_edge(node, edge.src_conn, edge.dst,
                                   edge.dst_conn, newmemlet)

                    if self.fullcopy:
                        edge.data.subset = sbs.Range.from_array(
                            node.desc(sdfg))
                    edge.data.other_subset = edge.data.subset
                    graph.add_edge(edge.src, None, node, None, edge.data)
        for array_name, node in out_cloned_arraynodes.items():
            graph.add_node(node)
            for edge in all_out_edges:
                if edge.data.data == array_name:
                    graph.remove_edge(edge)
                    newmemlet = copy.copy(edge.data)
                    newmemlet.data = node.data
                    graph.add_edge(edge.src, edge.src_conn, node,
                                   edge.dst_conn, newmemlet)
                    edge.data.wcr = None
                    if self.fullcopy:
                        edge.data.subset = sbs.Range.from_array(
                            node.desc(sdfg))
                    edge.data.other_subset = edge.data.subset
                    graph.add_edge(node, None, edge.dst, None, edge.data)

        # Reconnect stream-arrays
        for array_node, streamnode in out_streamarrays.items():
            # Set stream storage to GPU
            streamnode.desc(sdfg).storage = dtypes.StorageType.GPU_Global

            cloned_node = out_cloned_arraynodes[array_node.data]

            e = graph.out_edges(streamnode)[0]
            graph.remove_edge(e)
            newmemlet = copy.copy(e.data)
            newmemlet.data = cloned_node.data
            # stream -> cloned array
            graph.add_edge(e.src, e.src_conn, cloned_node, e.dst_conn,
                           newmemlet)
            # cloned array -> array
            graph.add_nedge(cloned_node, array_node, e.data)

        # Fourth, replace memlet arrays as necessary
        if self.expr_index == 0:
            scope_subgraph = graph.scope_subgraph(cnode)
            for edge in scope_subgraph.edges():
                if (edge.data.data is not None
                        and edge.data.data in cloned_arrays):
                    edge.data.data = cloned_arrays[edge.data.data]

    def modifies_graph(self):
        return True

    @staticmethod
    def print_debuginfo():
        print("Automatically cloned {} arrays for the GPU.".format(
            GPUTransformMap._arrays_removed))
        print("Automatically changed {} maps for the GPU.".format(
            GPUTransformMap._maps_transformed))


pattern_matching.Transformation.register_pattern(GPUTransformMap)
