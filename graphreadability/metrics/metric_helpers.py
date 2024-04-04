from itertools import combinations
from ..utils import helpers


def _get_triangles(G):
    """Get all triangles and edges from a graph."""
    # Create a list of sets of three nodes that form a triangle
    triangles = set()
    for u, v in G.edges():
        for t in G.neighbors(u):
            if v in G.neighbors(t):
                # Use tuple (u, v, t) as it is immutable
                triangle = tuple(sorted([u, v, t]))
                triangles.add(triangle)

    # Create a set of the edges from the set of triangles
    edges = set()
    for triangle in triangles:
        # Use tuples for edges as they are immutable
        u, v, t = triangle
        edges.add(tuple(sorted([u, v])))
        edges.add(tuple(sorted([v, t])))
        edges.add(tuple(sorted([u, t])))

    return triangles, edges


def _count_impossible_triangle_crossings(G):
    """Count the number of impossible triangle crossings in a graph.

    "Pairs of triangles can only cross at most six times, as opposed to the nine calculated by only
    using node degree. We do however have to account for triangles with shared edges and nodes, as
    these cases are partially handled by node degree. Additionally non-adjacent edges to a triangle
    can only cross at most two of the triangle's edges." Source: Mooney et. al.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate impossible triangle crossings for.

    Returns
    -------
    total_impossible : int
        The total number of impossible triangle crossings in the graph.
    """
    triangles, triangle_edges = _get_triangles(G)

    # Count the number of impossible triangles
    total_impossible = 0
    for triangle in triangles:
        # Count the edges in G that are not in or adjacent to the triangle
        u, v, t = triangle
        for a, b in G.edges():
            if {a, b} in triangle_edges:
                continue

            if a in triangle or b in triangle:
                continue

            total_impossible += 1

    # Count the number of impossible crossings between pairs of triangles
    for t1, t2 in combinations(triangles, 2):

        u, v, t = t1
        a, b, c = t2
        # Triangles share an edge (two nodes)
        if len({u, v, t} & {a, b, c}) == 2:
            total_impossible += 1
        # Triangles share a node
        elif len({u, v, t} & {a, b, c}) == 1:
            total_impossible += 2
        # Triangles do not share any nodes
        else:
            total_impossible += 3

    return total_impossible


def _count_4_cycles(G):
    """Count the number of 4-cycles in a graph."""
    num_4_cycles = 0
    for u, v in G.edges():
        for w in G.neighbors(v):
            if w == u:
                continue
            for x in G.neighbors(w):
                if x == v:
                    continue
                if x in G.neighbors(u):
                    num_4_cycles += 1
    num_4_cycles //= 4

    return num_4_cycles


def _calculate_edge_crossings(G, save_edge_attributes=True):
    """Calculate all edge crossings in a graph and save them to the edge data.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate edge crossings for.
    save_edge_attributes : bool
        Whether to save the edge crossings to the edge data.

    Returns
    -------
    crossings : set((edge1, edge2))
        A set of all edge crossings in the graph.
    angles : dict((edge1, edge2), angle)
        A dictionary of angles between edges.
    """
    crossings = set()
    angles = {}
    if save_edge_attributes:
        edge_crossings = {edge: {"count": 0, "angles": []} for edge in G.edges}

    # Iterate over all pairs of edges and check for intersections
    edges = list(G.edges)
    for i, edge1 in enumerate(edges):
        for edge2 in edges[i + 1 :]:
            # Check for intersections
            line_a = (
                (G.nodes[edge1[0]]["x"], G.nodes[edge1[0]]["y"]),
                (G.nodes[edge1[1]]["x"], G.nodes[edge1[1]]["y"]),
            )
            line_b = (
                (G.nodes[edge2[0]]["x"], G.nodes[edge2[0]]["y"]),
                (G.nodes[edge2[1]]["x"], G.nodes[edge2[1]]["y"]),
            )
            # Skip edges that share a node
            if len(set(line_a) & set(line_b)) > 0:
                continue
            if helpers._intersect(line_a, line_b):
                crossings.add((edge1, edge2))
                # Calculate angle between edges
                v1 = helpers.edge_vector(line_a)
                v2 = helpers.edge_vector(line_b)
                angle = helpers.calculate_angle_between_vectors(v1, v2)
                angles[edge1, edge2] = angle
                if save_edge_attributes:
                    edge_crossings[edge1]["count"] += 1
                    edge_crossings[edge2]["count"] += 1
                    edge_crossings[edge1]["angles"].append(angle)
                    edge_crossings[edge2]["angles"].append(angle)

    # Save edge crossings to edge data
    if save_edge_attributes:
        import networkx as nx

        nx.set_edge_attributes(G, edge_crossings, "edge_crossings")
    return crossings, angles
