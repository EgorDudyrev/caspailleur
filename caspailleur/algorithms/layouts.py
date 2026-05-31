import heapq
from collections.abc import Iterable, Iterator
from typing import TypeVar, Literal
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
        nodes: set[T], edges: set[tuple[T, T]],
        n_columns_max: int = None, n_edge_overlaps_max: int = None, n_best_solutions: int = None,
        nodes_ascending_order: list[T] = None,
        y_position: dict[T, float] = None,
        use_tqdm: bool = False,
) -> dict[T, tuple[float, float]]:
    if nodes_ascending_order is None or y_position is None:
        import networkx as nx
        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_edges_from(edges)

    if nodes_ascending_order is None:
        nodes_ascending_order = list(nx.topological_sort(graph))

    if y_position is None:
        y_position = nx.single_source_shortest_path_length(graph, nodes_ascending_order[0])

    TLayout = tuple[tuple[T, ...], ...]
    TMeasures = tuple[int, int]

    def layout2pos(layout: TLayout) -> dict[T, tuple[float, float]]:
        pos = dict()
        for column_idx, column in enumerate(layout):
            for el in column:
                pos[el] = (column_idx, y_position[el])
        return pos

    def update_overlap(layout: TLayout, new_node: T, old_score: int) -> int:
        pos = layout2pos(layout)

        n_overlaps = old_score
        shown_edges = [edge for edge in edges if edge[0] in pos and edge[1] in pos]
        old_edges = [edge for edge in shown_edges if new_node not in edge]
        new_edges = set(shown_edges) - set(old_edges)
        for old_edge in old_edges:
            old_line = [pos[el] for el in old_edge]
            for new_edge in new_edges:
                new_line = [pos[el] for el in new_edge]
                n_overlaps += int(doIntersect(old_line, new_line))
        return n_overlaps

    def expand_layouts(layouts: Iterable[TLayout], new_node: T) -> Iterator[tuple[TLayout, TLayout]]:
        for old_layout in layouts:
            # option 1: add new node to the end of an existing column
            for column_idx in range(len(old_layout)):
                new_layout = [colvals for colvals in old_layout]
                new_layout[column_idx] = new_layout[column_idx]+(new_node,)
                yield tuple(new_layout), old_layout

            # option 2: insert new node into a new column in between the layouts
            for column_idx in range(len(old_layout)):
                new_layout = old_layout[:column_idx] + ((new_node,),) + old_layout[column_idx:]
                yield new_layout, old_layout

            new_layout = old_layout + ((new_node,),)
            yield new_layout, old_layout
            new_layout = ((new_node,),) + old_layout
            yield new_layout, old_layout


    layouts: dict[TLayout, TMeasures] = {tuple(): (0, 0)}
    for node in tqdm(nodes_ascending_order, disable=not use_tqdm):
        new_layouts_generator = tqdm(expand_layouts(layouts, node), disable=not use_tqdm, leave=False, unit_scale=True)
        eval_layouts = {new_layout:
                            (update_overlap(new_layout, node, layouts[old_layout][0]), len(new_layout))
                        for new_layout, old_layout in new_layouts_generator}
        layouts = {layout: score for layout, score in eval_layouts.items() if score[0] <= n_edge_overlaps_max and score[1] <= n_columns_max}

        if n_best_solutions is not None and len(layouts) > n_best_solutions:
            layouts = heapq.nsmallest(n_best_solutions, layouts.items(), key=lambda layout: layout[1])
            layouts = dict(layouts)

    layouts = sorted(layouts, key=lambda layout: layouts[layout])
    return layout2pos(layouts[0])
