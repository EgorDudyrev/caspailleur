import heapq
from collections.abc import Iterator, Callable
from typing import TypeVar, Literal, NamedTuple

from bitarray import bitarray
from bitarray.util import zeros as bazeros
from tqdm.auto import tqdm

from caspailleur.registries import register_line_layout

T = TypeVar('T')
Coordinate = tuple[float, float]


@register_line_layout('nx-BFS')
def nx_bfs_layout(nodes: set[T], edges: set[tuple[T, T]], start_: T = None) -> dict[T, tuple[float, float]]:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    start_ = next(nx.topological_sort(graph)) if start_ is None else start_

    pos = nx.layout.bfs_layout(graph, start_, align='horizontal')
    return {node: (float(x), float(y)) for node, (x, y) in pos.items()}


@register_line_layout('nx-Multipartite')
def nx_multipartite_layout(nodes: set[T], edges: set[tuple[T, T]], y_position: dict[T, float] = None) -> dict[T, tuple[float, float]]:
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)

    if y_position is None:
        start_ = next(nx.topological_sort(graph))
        y_position = nx.single_source_shortest_path_length(graph, start_)

    for node in graph.nodes: graph.nodes[node]['y_position'] = y_position[node]
    pos = nx.layout.multipartite_layout(graph, align='horizontal', subset_key='y_position')
    return {node: (float(x), float(y)) for node, (x, y) in pos.items()}




# function to check if point q lies on line segment 'pr'
def onSegment(p: Coordinate, q: Coordinate, r: Coordinate) -> bool:
    return q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])

# function to find orientation of ordered triplet (p, q, r)
# 0 --> p, q and r are collinear
# 1 --> Clockwise
# 2 --> Counterclockwise
def orientation(p: Coordinate, q: Coordinate, r: Coordinate) -> Literal['colinear', 'clockwise', 'counterclockwise']:
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])

    # collinear
    if val == 0:
        return 'colinear'

    # clock or counterclock wise
    # 1 for clockwise, 2 for counterclockwise
    return 'clockwise' if val > 0 else 'counterclockwise'


# function to check if two line segments intersect
def doIntersect(line_a: tuple[Coordinate, Coordinate], line_b: tuple[Coordinate, Coordinate]):
    # find the four orientations needed
    # for general and special cases
    o1 = orientation(line_a[0], line_a[1], line_b[0])
    o2 = orientation(line_a[0], line_a[1], line_b[1])
    o3 = orientation(line_b[0], line_b[1], line_a[0])
    o4 = orientation(line_b[0], line_b[1], line_a[1])

    # general case
    if o1 != o2 and o3 != o4:
        return True

    # special cases
    # p1, q1 and p2 are collinear and p2 lies on segment p1q1
    if o1 == 'colinear' and onSegment(line_a[0], line_b[0], line_a[1]):
        return True

    # p1, q1 and q2 are collinear and q2 lies on segment p1q1
    if o2 == 'colinear' and onSegment(line_a[0], line_b[1], line_a[1]):
        return True

    # p2, q2 and p1 are collinear and p1 lies on segment p2q2
    if o3 == 'colinear' and onSegment(line_b[0], line_a[0], line_b[1]):
        return True

    # p2, q2 and q1 are collinear and q1 lies on segment p2q2
    if o4 == 'colinear' and onSegment(line_b[0], line_a[1], line_b[1]):
        return True

    return False


@register_line_layout('column_sofia')
def sofia_column_layout(
        nodes: set[T],
        edges: set[tuple[T, T]],
        n_columns_max: int = None, n_overlaps: int = None, n_best_solutions: int = None,
        y_position: Callable[[T], float] | dict[T, float] = None,
        use_tqdm: bool = False,
) -> dict[T, tuple[float, float]]:
    TMeasure = NamedTuple('TMeasure', [('n_overlaps', int), ('n_columns', int)])

    nodes_idx_map: dict[T, int] = dict()
    subnodes_relation: list[bitarray] = []

    def preprocess_node_subnode(node, subnodes):
        node_idx = nodes_idx_map[node]  # len(nodes_idx_map)
        subnodes_ba = bazeros(node_idx)
        for subnode in subnodes:
            subnode_idx = nodes_idx_map[subnode]
            subnodes_ba[subnode_idx] = True
            subnodes_ba[:subnode_idx] |= subnodes_relation[subnode_idx]
        return node_idx, subnodes_ba

    def expand_solutions(solutions_: Iterator[tuple[int,...]]) -> Iterator[tuple[tuple[int,...], tuple[int,...]]]:
        for old_solution in solutions_:
            n_columns = max(old_solution, default=-1)+1
            # place the new node to one of the existing columns
            for column_idx in range(n_columns):
                new_solution = old_solution + (column_idx,)
                yield new_solution, old_solution

            # place the new node to the left of everything
            new_solution = tuple([x+1 for x in old_solution]) + (0,)
            yield new_solution, old_solution
            # place the new node to the right of everything
            new_solution = old_solution + (n_columns,)
            yield new_solution, old_solution

            # place the new node between the existing columns
            for column_idx in range(1, n_columns):
                new_solution = tuple([x if x < column_idx else (x+1) for x in old_solution]) + (column_idx,)
                yield new_solution, old_solution

    def update_overlap(x_pos: tuple[int, ...], old_overlaps: int) -> int:
        def select_direct_subnodes(idx) -> bitarray:
            direct_subnodes = bitarray(subnodes_relation[idx])
            while (idx := direct_subnodes.find(True, 0, idx, right=True)) >= 0:
                direct_subnodes[:len(subnodes_relation[idx])] &= ~subnodes_relation[idx]
            return direct_subnodes

        last_added_idx = len(x_pos)-1
        connected_nodes = select_direct_subnodes(last_added_idx)
        n_overlaps = old_overlaps
        for subnode_i in connected_nodes.search(True):
            new_line = (x_pos[subnode_i], y_positions[subnode_i]), (x_pos[last_added_idx], y_positions[last_added_idx])

            target_nodes_to_check = ~subnodes_relation[last_added_idx]
            idx = len(target_nodes_to_check)
            while (idx := target_nodes_to_check.find(True, 0, idx, right=True)) >= 0:
                if y_positions[idx] <= y_positions[subnode_i]:
                    target_nodes_to_check[:len(subnodes_relation[idx])] &= ~subnodes_relation[idx]

            for target_node_idx in target_nodes_to_check.search(True):
                for source_node_idx in select_direct_subnodes(target_node_idx):
                    old_line = (x_pos[source_node_idx], y_positions[source_node_idx]), (x_pos[target_node_idx], y_positions[target_node_idx])

                    n_overlaps += int(doIntersect(old_line, new_line))
        return n_overlaps

    def evaluate_solution(solution: tuple[int, ...], old_eval: TMeasure) -> TMeasure:
        return TMeasure(update_overlap(solution, old_eval[0]), max(solution, default=-1)+1)

    def filter_solution(evals: TMeasure) -> bool:
        if n_columns_max is not None and evals.n_columns > n_columns_max:
            return False
        if n_overlaps is not None and evals.n_overlaps > n_overlaps:
            return False
        return True

    y_positions: list[float] = []
    solutions: dict[tuple[int, ...], TMeasure] = {tuple(): TMeasure(0, 0)}

    import networkx as nx
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    node_toposort = list(nx.topological_sort(graph))
    nodes_idx_map = {node: idx for idx, node in enumerate(node_toposort)}
    direct_subnodes = {node: set() for node in node_toposort}
    for source, target in edges:
        if nodes_idx_map[source] < nodes_idx_map[target]:
            direct_subnodes[target].add(source)
        else:
            direct_subnodes[source].add(target)
    nodes_and_subnodes = [(node, direct_subnodes[node]) for node in node_toposort]
    print([(nodes_idx_map[n], [nodes_idx_map[sn] for sn in sns]) for n, sns in nodes_and_subnodes])

    for node, subnodes in tqdm(nodes_and_subnodes, disable=not use_tqdm):
        node_idx, subnodes_ba = preprocess_node_subnode(node, subnodes)
        nodes_idx_map[node] = node_idx
        subnodes_relation.append(subnodes_ba)

        y_positions.append(y_position[node] if isinstance(y_position, dict) else y_position(node) if y_position is not None else subnodes_ba.count())

        new_solutions_generator = expand_solutions(solutions.keys())
        new_solutions = ((new_solution, evaluate_solution(new_solution, solutions[old_solution]))
                         for new_solution, old_solution in new_solutions_generator)
        new_solutions = [(new_solution, evals) for new_solution, evals in new_solutions
                         if filter_solution(evals)]

        if n_best_solutions is not None and len(new_solutions) > n_best_solutions:
            new_solutions = heapq.nsmallest(n_best_solutions, new_solutions, key=lambda xpos_evals: xpos_evals[1])
        solutions = dict(new_solutions)

    xpos_final = min(solutions, key=lambda x_pos: solutions[x_pos])
    return {el: (xpos_final[idx], y_positions[idx]) for el, idx in nodes_idx_map.items()}
