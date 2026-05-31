from typing import TypeVar

from caspailleur.registries import register_line_layout

T = TypeVar('T')


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
