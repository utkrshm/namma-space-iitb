"""Measured links only. A temporal adjacency is not evidence of covisibility."""


def connected_components(nodes, edges):
    adjacency = {node: set() for node in nodes}
    for edge in edges:
        adjacency[edge.a].add(edge.b)
        adjacency[edge.b].add(edge.a)
    components, remaining = [], set(nodes)
    while remaining:
        pending = [min(remaining)]
        component = set()
        while pending:
            node = pending.pop()
            if node in component:
                continue
            component.add(node)
            pending.extend(adjacency[node] - component)
        remaining -= component
        components.append(sorted(component))
    return components


def build_graph(selected, sequential_edges, revisit_edges):
    nodes = [f.frame_id for f in selected if f.pool == "geometry"]
    edges = list(sequential_edges) + list(revisit_edges)
    if any(e.a not in nodes or e.b not in nodes for e in edges):
        raise ValueError("Edge endpoint absent from geometry pool")
    return {
        "nodes": nodes,
        "edges": [e.to_dict() for e in edges],
        "components": connected_components(nodes, edges),
    }
