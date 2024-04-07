"""
This module contains all metric functions. A metric should be a function that takes a NetworkX graph as the
first argument and returns a float. It may also take additional arguments, which should be specified in the docstring.
"""

import random as rand
import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull as __ConvexHull
from ..utils import helpers
from ..utils import crosses_promotion
from ..metrics.metric_helpers import (
    _count_impossible_triangle_crossings,
    _count_4_cycles,
    _calculate_edge_crossings,
)


def edge_crossing(G, subtract_tri=True, subtract_4=True):
    """Calculate the metric for the number of edge_crossing, scaled against the total
    number of possible crossings.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.
    verbose : bool
        Whether to print additional information about the metric.

    Returns
    -------
    float
        The edge crossing metric.
    """
    # Estimate for the upper bound for the number of edge crossings
    m = G.number_of_edges()
    c_all = (m * (m - 1)) / 2

    # Calculate the number of impossible crossings based on the node degrees
    degree = np.array([degree[1] for degree in G.degree()])
    c_deg = np.dot(degree, degree - 1) / 2

    # Calculate the maximum number of possible crossings
    c_mx = c_all - c_deg

    if subtract_tri:
        c_mx -= _count_impossible_triangle_crossings(G)

    if subtract_4:
        c_mx -= _count_4_cycles(G)

    # Retrieve the edge crossings from the graph if they have been calculated, otherwise calculate
    if not nx.get_edge_attributes(G, "edge_crossings"):
        crossings, angles = _calculate_edge_crossings(G)
    else:
        edge_crossings = nx.get_edge_attributes(G, "edge_crossings")
        crossings = set()
        for edge, crossing in edge_crossings.items():
            if crossing["count"] > 0:
                crossings.add(edge)

    # Calculate the number of edge crossings
    c = len(crossings)

    return 1 - helpers.divide_or_zero(c, c_mx)


def edge_orthogonality(G):
    """Calculate the metric for edge orthogonality.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.
    optimal_angle : float
        The optimal angle for edge orthogonality.

    Returns
    -------
    float
        The edge orthogonality metric.
    """
    ortho_list = []

    # Iterate over each edge and get it's minimum angle relative to the orthogonal grid
    for e in G.edges:
        source = e[0]
        target = e[1]

        x1, y1 = G.nodes[source]["x"], G.nodes[source]["y"]
        x2, y2 = G.nodes[target]["x"], G.nodes[target]["y"]

        try:
            gradient = (y2 - y1) / (x2 - x1)
        except ZeroDivisionError:
            gradient = 0

        angle = np.degrees(np.arctan(abs(gradient)))

        edge_ortho = min(angle, abs(90 - angle), 180 - angle) / 45
        ortho_list.append(edge_ortho)

    # Return 1 minus the average of minimum angles
    return 1 - (sum(ortho_list) / G.number_of_edges())


def angular_resolution(G, all_nodes=False):
    """Calculate the metric for angular resolution.

    This metric captures how evenly the edges leaving a node are distributed. If all_nodes is True, include
    nodes with degree 1, for which the angle will always be perfect.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.
    all_nodes : bool
        Whether to include all nodes in the calculation.

    Returns
    -------
    float
        The angular resolution metric.
    """
    angles_sum = 0
    nodes_count = 0
    for node in G.nodes:
        if G.degree[node] <= 1:
            continue

        nodes_count += 1
        ideal = (
            360 / G.degree[node]
        )  # Each node has an ideal angle for adjacent edges, based on the number of adjacent edges

        x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
        actual_min = 360

        # Iterate over adjacent edges and calculate the difference of the minimum angle from the ideal angle
        for adj in G.neighbors(node):
            x2, y2 = G.nodes[adj]["x"], G.nodes[adj]["y"]
            angle1 = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

            for adj2 in G.neighbors(node):
                if adj == adj2:
                    continue

                x3, y3 = G.nodes[adj2]["x"], G.nodes[adj2]["y"]
                angle2 = np.degrees(np.arctan2((y3 - y1), (x3 - x1)))

                diff = abs(angle2 - angle1)

                if diff < actual_min:
                    actual_min = diff

        angles_sum += abs((ideal - actual_min) / ideal)

    # Return 1 minus the average of minimum angles
    return (
        1 - (angles_sum / G.number_of_nodes())
        if all_nodes
        else 1 - (angles_sum / nodes_count)
    )


def crossing_angle(G, crossing_limit=1e6):
    """Calculate the metric for the edge crossings angle.

    The edge crossings angle metric compares the angle of a crossing to an ideal angle. crossing_limit specifies
    the maximum number of crossings allowed, which is limited due to long execution times.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.
    crossing_limit : int
        The maximum number of crossings allowed.

    Returns
    -------
    float
        The edge crossings angle metric.

    Raises
    ------
    ValueError
        If the number of edges exceeds the crossing limit.
    """
    if G.number_of_edges() > crossing_limit:
        raise ValueError(
            f"Number of edges exceeds the crossing limit of {crossing_limit}"
        )

    # Check if graph edges have edge_crossings attribute
    if not nx.get_edge_attributes(G, "edge_crossings"):
        _calculate_edge_crossings(G)

    edge_crossings = nx.get_edge_attributes(G, "edge_crossings")

    angles_sum = 0
    for crossing in edge_crossings.values():
        ideal = 180 / (
            crossing["count"] + 1
        )  # Each crossing adds an additional edge, so the ideal angle is 180 / (count + 1)
        angles_sum += sum(
            [abs((ideal - angle) % ideal) / ideal for angle in crossing["angles"]]
        )
    return 1 - helpers.divide_or_zero(angles_sum, len(edge_crossings))


def __crossing_angle_old(G, crossing_limit=1e6):
    """Calculate the metric for the edge crossings angle. crossing_limit specifies the maximum number of crossings allowed,
    which is limited due to long execution times."""

    angles_sum = 0
    num_minor_nodes = 0
    for node in G.nodes:
        # Only crosses promoted nodes should be counted
        if not crosses_promotion._is_minor(node, G):
            continue

        num_minor_nodes += 1
        ideal = (
            360 / G.degree[node]
        )  # This should always be 90 degrees, except in rare cases where multiple edges intersect at the exact same point

        x1, y1 = G.nodes[node]["x"], G.nodes[node]["y"]
        actual_min = 360

        # Iterate over adjacent edges and calculate the difference of the minimum angle from the ideal angle
        for adj in G.neighbors(node):
            x2, y2 = G.nodes[adj]["x"], G.nodes[adj]["y"]
            angle1 = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))

            for adj2 in G.neighbors(node):
                if adj == adj2:
                    continue

                x3, y3 = G.nodes[adj2]["x"], G.nodes[adj2]["y"]
                angle2 = np.degrees(np.arctan2((y3 - y1), (x3 - x1)))

                diff = abs(angle1 - angle2)

                if diff < actual_min:
                    actual_min = diff

        angles_sum += abs((ideal - actual_min) / ideal)

    if num_minor_nodes == 0:
        print("Warning: No minor nodes found. Did you run crosses promotion?")
        return 1

    # Return 1 minus the average of minimum angles
    return 1 - (angles_sum / num_minor_nodes) if num_minor_nodes > 0 else 1


def node_orthogonality(G):
    """Calculate the metric for node orthogonality."""
    coord_set = []

    # Start with random node
    first_node = rand.sample(list(G.nodes), 1)[0]
    min_x, min_y = (
        G.nodes[first_node]["x"],
        G.nodes[first_node]["y"],
    )

    # Find minimum x and y positions
    for node in G.nodes:
        x = G.nodes[node]["x"]
        y = G.nodes[node]["y"]

        if x < min_x:
            min_x = x
        elif y < min_y:
            min_y = y

    x_distance = abs(0 - float(min_x))
    y_distance = abs(0 - float(min_y))

    # Adjust graph so node with minimum coordinates is at 0,0
    for node in G.nodes:
        G.nodes[node]["x"] = float(G.nodes[node]["x"]) - x_distance
        G.nodes[node]["y"] = float(G.nodes[node]["y"]) - y_distance

    # Start with random node
    first_node = rand.sample(list(G.nodes), 1)[0]

    min_x, min_y = (
        G.nodes[first_node]["x"],
        G.nodes[first_node]["y"],
    )
    max_x, max_y = (
        G.nodes[first_node]["x"],
        G.nodes[first_node]["y"],
    )

    for node in G.nodes:
        x, y = G.nodes[node]["x"], G.nodes[node]["y"]

        coord_set.append(x)
        coord_set.append(y)

        # Get GCD of node positions
        gcd = int(float(coord_set[0]))
        for coord in coord_set[1:]:
            gcd = np.gcd(int(float(gcd)), int(float(coord)))

        # Get maximum and minimum coordinates
        if x > max_x:
            max_x = x
        elif x < min_x:
            min_x = x

        if y > max_y:
            max_y = y
        elif y < min_y:
            min_y = y

    # Get size of unit grid
    h = abs(max_y - min_y)
    w = abs(max_x - min_x)

    reduced_h = h / gcd
    reduced_w = w / gcd

    A = (reduced_w + 1) * (reduced_h + 1)

    # Return number of nodes on the unit grid weighted against the number of positions on the unit grid
    return len(G.nodes) / A


def node_resolution(G):
    """Calculate the metric for node resolution.

    Node resolution is the ratio of the smallest and largest distance between any pair of nodes.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.

    Returns
    -------
    float
        The node resolution metric.
    """
    # Start with two random nodes
    first_node, second_node = rand.sample(list(G.nodes), 2)
    a = G.nodes[first_node]["x"], G.nodes[first_node]["y"]
    b = G.nodes[second_node]["x"], G.nodes[second_node]["y"]

    min_dist = helpers._euclidean_distance(a, b)
    max_dist = min_dist

    # Iterate over every pair of nodes, keeping track of the maximum and minimum distances between them
    nodes = list(G.nodes)
    for idx, i in enumerate(nodes):
        for j in nodes[idx + 1 :]:

            a = G.nodes[i]["x"], G.nodes[i]["y"]
            b = G.nodes[j]["x"], G.nodes[j]["y"]

            d = helpers._euclidean_distance(a, b)

            if d < min_dist:
                min_dist = d

            if d > max_dist:
                max_dist = d

    return min_dist / max_dist


def edge_length(G, ideal_edge_length=None):
    """Calculate the edge length metric.

    The edge length metric compares the edge lengths to an ideal length. Default ideal is average of all edge lengths.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.
    ideal : float
        The ideal edge length.

    Returns
    -------
    float
        The edge length metric.
    """
    if not ideal_edge_length:
        # For unweighted graphs, set the ideal edge length to the average edge length
        ideal_edge_length = 0
        for edge in G.edges:
            a = G.nodes[edge[0]]["x"], G.nodes[edge[0]]["y"]
            b = G.nodes[edge[1]]["x"], G.nodes[edge[1]]["y"]

            ideal_edge_length += helpers._euclidean_distance(a, b)
        ideal_edge_length = ideal_edge_length / G.number_of_edges()

    edge_length_sum = 0
    for edge in G.edges:
        a = G.nodes[edge[0]]["x"], G.nodes[edge[0]]["y"]
        b = G.nodes[edge[1]]["x"], G.nodes[edge[1]]["y"]
        edge_length_sum += (
            abs(ideal_edge_length - helpers._euclidean_distance(a, b))
            / ideal_edge_length
        )

    # Remove negatives
    if edge_length_sum > G.number_of_edges():
        return 1 - abs(1 - (edge_length_sum / G.number_of_edges()))

    return 1 - (edge_length_sum / G.number_of_edges())


def gabriel_ratio(G):
    """Calculate the metric for the gabriel ratio.

    A graph is a Gabriel graph if no node falls within the area of any circles constructed using each edge as its diameter.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.

    Returns
    -------
    float
        The gabriel ratio metric.
    """

    # Initial upper bound on number of nodes which could potentially be violating nodes
    possible_non_conforming = (G.number_of_edges() * G.number_of_nodes()) - (
        G.number_of_edges() * 2
    )

    num_non_conforming = 0

    # Iterate over each edge
    for edge in G.edges:

        # Get the equation of the circle with the edge as its diameter
        a = G.nodes[edge[0]]["x"], G.nodes[edge[0]]["y"]
        b = G.nodes[edge[1]]["x"], G.nodes[edge[1]]["y"]

        r = helpers._euclidean_distance(a, b) / 2
        center_x, center_y = helpers._midpoint(edge[0], edge[1], G)

        # Check if any nodes fall with within the circle and increment the counter if they do
        for node in G.nodes:
            if edge[0] == node or edge[1] == node:
                continue

            x, y = G.nodes[node]["x"], G.nodes[node]["y"]

            if helpers._in_circle(x, y, center_x, center_y, r):
                num_non_conforming += 1
                # If the node is adjacent to either node in the current edge reduce total by 1,
                # since the nodes cannot both simultaneously be in each others circle
                if node in G.neighbors(edge[0]):
                    possible_non_conforming -= 1
                if node in G.neighbors(edge[1]):
                    possible_non_conforming -= 1

    # Return 1 minus the ratio of non conforming nodes to the upper bound on possible non conforming nodes.
    return (
        1 - (num_non_conforming / possible_non_conforming)
        if possible_non_conforming > 0
        else 1
    )


def __stress(G):
    """Calculate the metric for stress.

    Stress is a measure of how well the graph preserves the pairwise distances between nodes.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.

    Returns
    -------
    float
        The stress metric.
    """
    # Create a single matrix of all node locations
    X = np.array([[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in G.nodes()])
    N = len(X)

    # Create a sorted dictionary of the shortest path lengths between all pairs of nodes
    all_pairs_shortest = dict(nx.all_pairs_shortest_path_length(G))
    all_pairs_shortest = dict(sorted(all_pairs_shortest.items()))

    # Create a matrix of the shortest path lengths between all pairs of nodes
    d = np.zeros((N, N))
    for i, k in enumerate(all_pairs_shortest):
        all_pairs_shortest[k] = dict(sorted(all_pairs_shortest[k].items()))
        d[i] = [float(v) for v in all_pairs_shortest[k].values()]

    from math import comb

    ss = (X * X).sum(axis=1)

    diff = np.sqrt(abs(ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X, X.T)))

    np.fill_diagonal(diff, 0)

    def stress_func(a):
        return np.sum(
            np.square(np.divide((a * diff - d), d, out=np.zeros_like(d), where=d != 0))
        ) / comb(N, 2)

    from scipy.optimize import minimize_scalar

    min_a = minimize_scalar(stress_func)

    if not min_a.success:
        raise ValueError(f"Failed to minimize stress function: {min_a.message}")

    return stress_func(a=min_a.x)


def aspect_ratio(G):
    """Calculate the metric for aspect ratio.

    Aspect ratio is the ratio of the width to the height of the smallest bounding box that contains all nodes.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.

    Returns
    -------
    float
        The aspect ratio metric.
    """
    bbox = helpers._get_bounding_box(G)

    width = bbox[1, 0] - bbox[0, 0]
    height = bbox[1, 1] - bbox[0, 1]

    if width > height:
        return height / width
    else:
        return width / height


def node_uniformity(G):
    """Calculate the metric for node uniformity.

    Node uniformity is the ratio of the number of nodes to the number of cells in a grid that contains all nodes.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.

    Returns
    -------
    float
        The node uniformity metric.
    """

    points = helpers._graph_to_points(G)
    bbox = helpers._bounding_box(points)
    x_min, y_min, x_max, y_max = bbox.flatten().tolist()

    num_points = len(points)
    num_cells = int(np.sqrt(num_points))

    cell_width = (x_max - x_min) / num_cells
    cell_height = (y_max - y_min) / num_cells

    grid = [[0 for _ in range(num_cells)] for _ in range(num_cells)]

    for i in range(num_cells):
        for j in range(num_cells):
            for point in points:
                square = (
                    (x_min + (i * cell_width)),
                    (y_min + (j * cell_height)),
                ), (
                    (x_min + ((i + 1) * cell_width)),
                    (y_min + ((j + 1) * cell_height)),
                )
                # print(square)
                if helpers._is_point_inside_square(
                    *point,
                    square[0][0],
                    square[0][1],
                    square[1][0],
                    square[1][1],
                ):
                    grid[i][j] += 1

    total_cells = num_cells * num_cells
    average_points_per_cell = num_points / total_cells
    evenness = sum(
        abs(cell - average_points_per_cell) for row in grid for cell in row
    ) / (2 * total_cells)
    return 1 - evenness if evenness < 1 else 0


def neighbourhood_preservation(G, k=None):
    """Calculate the metric for neighbourhood preservation.

    Neighbourhood preservation is the average of the ratio of the number of neighbors by edges to the number
    of neighbors by k-nearest neighbors. This metric attempts to capture how well the geometry of the graph
    preserves the topology of the graph.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.
    k : int
        The number of nearest neighbours to consider.

    Returns
    -------
    float
        The neighbourhood preservation metric.
    """
    N = G.number_of_nodes()

    # Default to average degree
    if k is None:
        k = np.floor(helpers.avg_degree(G)).astype(int)

    adj = nx.to_numpy_array(G)
    K = np.zeros_like(adj)

    # Get node positions
    points = helpers._graph_to_points(G)

    # Build KD tree
    tree = helpers._build_kd_tree(points)

    # Find k nearest neighbours for each node
    for i, u in enumerate(G.nodes()):
        nearest = helpers._find_k_nearest_points(points[i], k + 1, tree=tree)
        for j in nearest[1:]:
            K[i][j] = 1

    # Remove diagonal
    np.fill_diagonal(K, 0)

    # Calculate the ratio of neighbours to k-nearest neighbours
    intersection = np.logical_and(adj, K)
    union = np.logical_or(adj, K)
    return intersection.sum() / union.sum()


def __count_crossings(G, crosses_limit=1e6):
    """
    Count the number of edge crossings in a graph.

    Parameters
    ----------
    G : nx.Graph
        The graph to calculate the metric for.

    Returns
    -------
    int
        The number of edge crossings in the graph.
    """

    covered = []  # List to keep track of covered edges
    c = 0  # Counter for edge crossings

    for e in G.edges:
        a_p1 = (G.nodes[e[0]]["x"], G.nodes[e[0]]["y"])  # Position of source node of e
        a_p2 = (G.nodes[e[1]]["x"], G.nodes[e[1]]["y"])  # Position of target node of e
        line_a = (a_p1, a_p2)  # Line segment of edge e

        for e2 in G.edges:
            if c > crosses_limit:
                raise ValueError(
                    f"Number of edge crossings exceeds the limit of {crosses_limit}"
                )

            if e == e2:
                continue  # Skip if the edges are the same

            b_p1 = (
                G.nodes[e2[0]]["x"],
                G.nodes[e2[0]]["y"],
            )  # Position of source node of e2
            b_p2 = (
                G.nodes[e2[1]]["x"],
                G.nodes[e2[1]]["y"],
            )  # Position of target node of e2
            line_b = (b_p1, b_p2)  # Line segment of edge e2

            if helpers._intersect(line_a, line_b) and (line_a, line_b) not in covered:
                covered.append((line_b, line_a))  # Mark the edges as covered
                c += 1  # Increment the counter for edge crossings

    return c


def _symmetry(
    G=None,
    num_crossings=None,
    show_sym=False,
    crosses_limit=1e6,
    threshold=1,
    tolerance=0.1,
):
    """
    Calculate the symmetry metric."""
    if num_crossings is None:
        num_crossings = __count_crossings(G, crosses_limit)

    axes = helpers._find_bisectors(G)

    total_area = 0
    total_sym = 0

    for a in axes:

        num_mirror = 0
        sym_val = 0
        subgraph = []
        covered = []

        for e1 in G.edges:
            for e2 in G.edges:
                if e1 == e2 or (e1, e2) in covered:
                    continue

                if helpers._mirror(a, e1, e2, G, tolerance) == 1:
                    num_mirror += 1
                    sym_val += helpers._sym_value(e1, e2, G)
                    subgraph.append(e1)
                    subgraph.append(e2)

                covered.append((e2, e1))

        # Compare number of mirrored edges to specified threshold
        if num_mirror >= threshold:

            points = helpers._graph_to_points(G, subgraph)

            if len(points) <= 2:
                break

            # Add area of local symmetry to total area and add to total symmetry
            conv_hull = __ConvexHull(points, qhull_options="QJ")
            sub_area = conv_hull.volume
            total_area += sub_area

            total_sym += (sym_val * sub_area) / (len(subgraph) / 2)

            # Debug info
            if show_sym:
                ag = nx.Graph()
                ag.add_edges_from(subgraph)

                for node in ag:
                    if node in G:
                        ag.nodes[node]["x"] = G.nodes[node]["x"]
                        ag.nodes[node]["y"] = G.nodes[node]["y"]
                helpers.draw_graph(ag)

    # Get the are of the convex hull of the graph
    whole_area_points = helpers._graph_to_points(G)

    whole_hull = __ConvexHull(whole_area_points)
    whole_area = whole_hull.volume

    # Return the symmetry weighted against either the area of the convex hull of the graph or the combined area of all local symmetries
    return total_sym / max(whole_area, total_area)
