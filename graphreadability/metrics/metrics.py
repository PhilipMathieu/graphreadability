"""
This module contains all metric functions. A metric should be a function that takes a NetworkX graph as the
first argument and returns a float. It may also take additional arguments, which should be specified in the docstring.
"""
import random as rand
import numpy as np
import networkx as nx
from scipy.spatial import ConvexHull
from ..utils import helpers
from ..utils import crosses_promotion


def count_impossible_triangle_crossings(G):
    triangles = []
    for u, v in G.edges():
        for t in G.neighbors(u):
            if v in G.neighbors(t) and {u, v, t} not in triangles:
                triangles.append({u, v, t})

    triangle_edges = []
    for u, v, t in triangles:
        if {u, v} not in triangle_edges:
            triangle_edges.append({u, v})
        if {v, t} not in triangle_edges:
            triangle_edges.append({v, t})
        if {t, u} not in triangle_edges:
            triangle_edges.append({t, u})

    total_impossible = 0
    for u, v, t in triangles:
        bubble = []
        bubble.extend(G.edges(u))
        bubble.extend(G.edges(v))
        bubble.extend(G.edges(t))

        subG = nx.Graph(bubble)

        for a, b in G.edges():
            if (a, b) in subG.edges() or (b, a) in subG.edges():
                continue

            if {a, b} in triangle_edges:
                continue

            total_impossible += 1

    covered_triangles = []
    for u, v, t in triangles:
        for a, b, c in triangles:
            if {u, v, t} in covered_triangles or {a, b, c} in covered_triangles:
                continue

            if {u, v, t} == {a, b, c}:
                continue

            covered_triangles.append({u, v, t})
            # Triangles share an edge
            if (
                ({u, v} == {a, b} or {u, v} == {b, c} or {u, v} == {c, a})
                or ({v, t} == {a, b} or {v, t} == {b, c} or {v, t} == {c, a})
                or ({t, u} == {a, b} or {t, u} == {b, c} or {t, u} == {c, a})
            ):

                total_impossible += 1
                continue

            # Triangles share a node
            if (
                (u == a or u == b or u == c)
                or (v == a or v == b or v == c)
                or (t == a or t == b or t == c)
            ):
                total_impossible += 2
                continue

            total_impossible += 3

    num_4_cycles = 0
    for u, v in G.edges():
        for t in G.neighbors(u):
            if t == v:
                continue

            for w in G.neighbors(v):
                if w == t or w == u:
                    continue

                if w in G.neighbors(t):
                    square = G.subgraph([u, v, t, w])
                    num_adj = 0

                    for su, sv in square.edges():
                        if {su, sv} in triangle_edges:
                            num_adj += 1

                    if num_adj < 2:
                        num_4_cycles += 1

    return total_impossible + (num_4_cycles // 4)

def calculate_edge_crossings(G):
        crossings = set()
        angles = {}
        edges_checked = set()
        edge_crossings = {edge: {"count": 0, "angles": []} for edge in G.edges}

        for edge1 in G.edges:
            for edge2 in G.edges:
                if edge1 != edge2 and (edge2, edge1) not in edges_checked:
                    edges_checked.add((edge1, edge2))
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
                        edge_crossings[edge1]["count"] += 1
                        edge_crossings[edge2]["count"] += 1
                        edge_crossings[edge1]["angles"].append(angle)
                        edge_crossings[edge2]["angles"].append(angle)

        # Save edge crossings to edge data
        nx.set_edge_attributes(G, edge_crossings, "edge_crossings")
        return crossings

def edge_crossing(G, verbose=False):
    """Calculate the metric for the number of edge_crossing, scaled against the total
    number of possible crossings."""
    # Estimate for the upper bound for the number of edge crossings
    m =  G.number_of_edges()
    c_all = (m * (m - 1))/2

    degree = np.array([degree[1] for degree in G.degree()])
    c_impossible = np.dot(degree, degree - 1) / 2

    c_mx = c_all - c_impossible

    c_tri = count_impossible_triangle_crossings(G)

    c_mx_no_tri = c_all - c_tri

    c_mx_no_tri_no_deg = c_all - c_impossible - c_tri

    if verbose:
        print(f"Total Upper bound: {c_all:.0f}")
        print(f"Impossible by degree: {c_impossible:.0f}")
        print(f"Impossible by triangle: {c_tri:.0f}")
        print(f"Upper bound removing degree: {c_mx:.0f}")
        print(f"Upper bound removing triangles: {c_mx_no_tri:.0f}")
        print(f"Upper bound removing degree and triangles: {c_mx_no_tri_no_deg:.0f}")

    # Check if graph edges have edge_crossings attribute
    if not nx.get_edge_attributes(G, "edge_crossings"):
        calculate_edge_crossings(G)

    c = np.sum([crossing["count"] for crossing in nx.get_edge_attributes(G, "edge_crossings").values()]) / 2 # Each crossing is counted twice

    if verbose:
        print(f"Num Crossings: {c}")
        print(f"Original EC: {1 - (c / c_mx) if c_mx > 0 else 1}")
        print(f"EC without triangles: {1 - (c / c_mx_no_tri) if c_mx_no_tri > 0 else 1}")
        print(f"EC without triangles and degrees: {1 - (c / c_mx_no_tri_no_deg) if c_mx_no_tri_no_deg > 0 else 1}")

    return 1 - helpers.divide_or_zero(c, c_mx)


def edge_orthogonality(G):
    """Calculate the metric for edge orthogonality."""
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
    """Calculate the metric for angular resolution. If all_nodes is True, include nodes with degree 1, for which the angle will always be perfect."""
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

def crossing_angle(G):
     # Check if graph edges have edge_crossings attribute
    if not nx.get_edge_attributes(G, "edge_crossings"):
        calculate_edge_crossings(G)

    edge_crossings = nx.get_edge_attributes(G, "edge_crossings")

    angles_sum = 0
    for crossing in edge_crossings.values():
        ideal = 180 / (crossing["count"] + 1) # Each crossing adds an additional edge, so the ideal angle is 180 / (count + 1)
        angles_sum += sum([abs((ideal - angle) % ideal) / ideal for angle in crossing["angles"]])
    return 1 - helpers.divide_or_zero(angles_sum, len(edge_crossings))

def crossing_angle_old(G, crossing_limit=1e6):
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
    """Calulate the metric for node resolution, which is the ratio of the smallest and largest distance between any pair of nodes."""

    # Start with two random nodes
    first_node, second_node = rand.sample(list(G.nodes), 2)
    a = G.nodes[first_node]["x"], G.nodes[first_node]["y"]
    b = G.nodes[second_node]["x"], G.nodes[second_node]["y"]

    min_dist = helpers._euclidean_distance(a, b)
    max_dist = min_dist

    # Iterate over every pair of nodes, keeping track of the maximum and minimum distances between them
    for i in G.nodes:
        for j in G.nodes:
            if i == j:
                continue

            a = G.nodes[i]["x"], G.nodes[i]["y"]
            b = G.nodes[j]["x"], G.nodes[j]["y"]

            d = helpers._euclidean_distance(a, b)

            if d < min_dist:
                min_dist = d

            if d > max_dist:
                max_dist = d

    return min_dist / max_dist


def edge_length(G, ideal=None):
    """Calculate the edge length metric by comparing the edge lengths to an ideal length. Default ideal is average of all edge lengths."""

    ideal_edge_length = 0
    for edge in G.edges:
        a = G.nodes[edge[0]]["x"], G.nodes[edge[0]]["y"]
        b = G.nodes[edge[1]]["x"], G.nodes[edge[1]]["y"]

        ideal_edge_length += helpers._euclidean_distance(a, b)

    if not ideal:
        # For unweighted graphs, set the ideal edge length to the average edge length
        ideal_edge_length = ideal_edge_length / G.number_of_edges()
    else:
        ideal_edge_length = ideal

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
    """Calculate the metric for the gabriel ratio. A graph is a Gabriel graph if no node falls within the area of any circles constructed using each edge as its diameter."""

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


def get_stress(G):
    X = np.array([[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in G.nodes()])

    apsp = dict(nx.all_pairs_shortest_path_length(G))
    apsp = dict(sorted(apsp.items()))

    d = [[] for _ in range(G.number_of_nodes())]

    for i, k in enumerate(apsp):
        apsp[k] = dict(sorted(apsp[k].items()))
        d[i] = [float(v) for v in apsp[k].values()]

    d = np.array(d)

    from math import comb

    N = len(X)
    ss = (X * X).sum(axis=1)

    diff = np.sqrt(abs(ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X, X.T)))

    np.fill_diagonal(diff, 0)

    stress = lambda a: np.sum(
        np.square(np.divide((a * diff - d), d, out=np.zeros_like(d), where=d != 0))
    ) / comb(N, 2)

    from scipy.optimize import minimize_scalar

    min_a = minimize_scalar(stress)

    return stress(a=min_a.x)


def stress_not_normal(G):

    X = np.array([[float(G.nodes[n]["x"]), float(G.nodes[n]["y"])] for n in G.nodes()])

    apsp = dict(nx.all_pairs_shortest_path_length(G))
    apsp = dict(sorted(apsp.items()))

    d = [[] for _ in range(G.number_of_nodes())]

    for i, k in enumerate(apsp):
        apsp[k] = dict(sorted(apsp[k].items()))
        d[i] = [float(v) for v in apsp[k].values()]

    d = np.array(d)

    from math import comb

    N = len(X)
    ss = (X * X).sum(axis=1)

    diff = np.sqrt(abs(ss.reshape((N, 1)) + ss.reshape((1, N)) - 2 * np.dot(X, X.T)))

    np.fill_diagonal(diff, 0)
    stress = lambda a: np.sum(
        np.square(np.divide((a * diff - d), d, out=np.zeros_like(d), where=d != 0))
    ) / comb(N, 2)

    from scipy.optimize import minimize_scalar

    min_a = minimize_scalar(stress)
    # print("a is ",min_a.x)
    return stress(a=min_a.x)


def aspect_ratio(G):

    points = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()]

    x_min = min(point[0] for point in points)
    y_min = min(point[1] for point in points)
    x_max = max(point[0] for point in points)
    y_max = max(point[1] for point in points)
    width = x_max - x_min
    height = y_max - y_min

    if height > width:
        return width / height
    else:
        return height / width


def node_uniformity(G):

    points = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()]
    x_min = min(point[0] for point in points)
    y_min = min(point[1] for point in points)
    x_max = max(point[0] for point in points)
    y_max = max(point[1] for point in points)

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
                    square[0][0], square[0][1],
                    square[1][0], square[1][1],
                ):
                    grid[i][j] += 1

    total_cells = num_cells * num_cells
    average_points_per_cell = num_points / total_cells
    evenness = sum(
        abs(cell - average_points_per_cell) for row in grid for cell in row
    ) / (2 * total_cells)
    return 1 - evenness if evenness < 1 else 0


def neighbourhood_preservation(G, k=None):

    if k == None:
        k = np.floor(helpers.avg_degree(G)).astype(int)

    adj = nx.to_numpy_array(G)

    K = np.zeros((G.number_of_nodes(), G.number_of_nodes()))

    points = [(G.nodes[n]["x"], G.nodes[n]["y"]) for n in G.nodes()]

    for i, u in enumerate(G.nodes()):
        for j, v in enumerate(G.nodes()):

            # shortest_paths = nx.shortest_path_length(G, source=u)

            # if shortest_paths[v] <= k:
            #     K[i][j] = 1

            p = (G.nodes[u]["x"], G.nodes[u]["y"])
            nearest = helpers._find_k_nearest_points(p, points, k + 1)

            q = (G.nodes[v]["x"], G.nodes[v]["y"])

            if q in nearest:
                K[i][j] = 1

    np.fill_diagonal(K, 0)
    # print(K)
    intersection = np.logical_and(adj, K)
    union = np.logical_or(adj, K)
    return intersection.sum() / union.sum()


def count_crossings(G):
    """
    Count the number of edge crossings in a graph.

    Parameters:
    - G: NetworkX graph object

    Returns:
    - c: Number of edge crossings
    """

    covered = []  # List to keep track of covered edges
    c = 0  # Counter for edge crossings

    for e in G.edges:
        a_p1 = (G.nodes[e[0]]["x"], G.nodes[e[0]]["y"])  # Position of source node of e
        a_p2 = (G.nodes[e[1]]["x"], G.nodes[e[1]]["y"])  # Position of target node of e
        line_a = (a_p1, a_p2)  # Line segment of edge e

        for e2 in G.edges:
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

def symmetry(G=None, num_crossings=None, show_sym=False, crosses_limit=1e6, threshold=1, tolerance=0.1):
    """Calculate the symmetry metric."""
    if num_crossings is None:
        num_crossings = count_crossings(G)
    if num_crossings > crosses_limit:
        return 0

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
            conv_hull = ConvexHull(points, qhull_options="QJ")
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

    whole_hull = ConvexHull(whole_area_points)
    whole_area = whole_hull.volume

    # Return the symmetry weighted against either the area of the convex hull of the graph or the combined area of all local symmetries
    return total_sym / max(whole_area, total_area)