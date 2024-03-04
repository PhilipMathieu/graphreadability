import math
import random as rand
from scipy.spatial import KDTree
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def _rel_point_line_dist(gradient, y_intercept, x, y):
    """Helper function to get the relative distance between a bisector and a point."""
    gradient *= -1
    y_intercept *= -1

    x = gradient * float(x)
    denom = math.sqrt(gradient**2 + 1)
    return (x + float(y) + float(y_intercept)) / denom


def _same_position(n1, n2, G, tolerance=0):
    """Helper function to determine if two nodes are in the same postion, with some tolerance."""
    x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
    x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]

    if tolerance == 0:
        return x1 == x2 and y1 == y2

    return _in_circle(x1, y1, x2, y2, tolerance)


def _is_positive(self, x):
    """Return true if x is postive."""
    return x > 0


def _are_collinear(a, b, c, G):
    """Returns true if the three points are collinear, by checking if the determinant is 0."""
    return (
        (G.nodes[a]["x"] * G.nodes[b]["y"])
        + (G.nodes[b]["x"] * G.nodes[c]["y"])
        + (G.nodes[c]["x"] * G.nodes[a]["y"])
        - (G.nodes[a]["x"] * G.nodes[c]["y"])
        - (G.nodes[b]["x"] * G.nodes[a]["y"])
        - (G.nodes[c]["x"] * G.nodes[b]["y"])
    ) == 0


def _mirror(axis, e1, e2, G, tolerance=0):
    """Helper function to determine if two edges are mirrored about a bisector."""
    e1_p1_x, e1_p1_y = G.nodes[e1[0]]["x"], G.nodes[e1[0]]["y"]
    e1_p2_x, e1_p2_y = G.nodes[e1[1]]["x"], G.nodes[e1[1]]["y"]

    e2_p1_x, e2_p1_y = G.nodes[e2[0]]["x"], G.nodes[e2[0]]["y"]
    e2_p2_x, e2_p2_y = G.nodes[e2[1]]["x"], G.nodes[e2[1]]["y"]

    # The end nodes of edge1 are P and Q
    # The end nodes of edge2 are X and Y
    P, Q, X, Y = e1[0], e1[1], e2[0], e2[1]

    if axis[0] == "x":
        p = axis[1] - e1_p1_y
        q = axis[1] - e1_p2_y
        x = axis[1] - e2_p1_y
        y = axis[1] - e2_p2_y
    elif axis[0] == "y":
        p = axis[1] - e1_p1_x
        q = axis[1] - e1_p2_x
        x = axis[1] - e2_p1_x
        y = axis[1] - e2_p2_x
    else:
        p = _rel_point_line_dist(axis[0], axis[1], e1_p1_x, e1_p1_y)
        q = _rel_point_line_dist(axis[0], axis[1], e1_p2_x, e1_p2_y)
        x = _rel_point_line_dist(axis[0], axis[1], e2_p1_x, e2_p1_y)
        y = _rel_point_line_dist(axis[0], axis[1], e2_p2_x, e2_p2_y)

    if e1 == e2:
        # Same edge
        return 0
    elif p == 0 and q == 0:
        # Edge on axis
        return 0
    elif y == 0 and x == 0:
        # Edge on other axis
        return 0
    elif _same_position(P, X, G, tolerance) and (
        _same_rel_position(p, 0, tolerance) and _same_rel_position(x, 0, tolerance)
    ):
        if _same_rel_position(q, y, tolerance) and (_is_positive(q) != _is_positive(y)):
            if not _are_collinear(Q, P, Y, G):
                # Shared node on axis but symmetric
                return 1
    elif _same_position(P, Y, G, tolerance) and (
        _same_rel_position(p, 0, tolerance) and _same_rel_position(y, 0, tolerance)
    ):
        if _same_rel_position(q, x, tolerance) and (_is_positive(q) != _is_positive(x)):
            if not _are_collinear(Q, P, X, G):
                # Shared node on axis but symmetric
                return 1
    elif _same_position(Q, Y, G, tolerance) and (
        _same_rel_position(q, 0, tolerance) and _same_rel_position(y, 0, tolerance)
    ):
        if _same_rel_position(p, x, tolerance) and (_is_positive(x) != _is_positive(p)):
            if not _are_collinear(P, Q, X, G):
                # Shared node on axis but symmetric
                return 1
    elif _same_position(Q, X, G, tolerance) and (
        _same_rel_position(q, 0, tolerance) and _same_rel_position(x, 0, tolerance)
    ):
        if _same_rel_position(p, y, tolerance) and (_is_positive(p) != _is_positive(y)):
            if not _are_collinear(P, Q, Y, G):
                # Shared node on axis but symmetric
                return 1
    elif _is_positive(p) != _is_positive(q):
        # Edge crosses axis
        return 0
    elif _is_positive(x) != _is_positive(y):
        # Other edge crosses axis
        return 0
    elif (
        (_same_rel_position(p, x, tolerance) and _same_rel_position(q, y, tolerance))
        and (_is_positive(p) != _is_positive(x))
        and (_is_positive(q) != _is_positive(y))
    ):
        # Distances are equal and signs are different
        x1, y1 = G.nodes[P]["x"], G.nodes[P]["y"]
        x2, y2 = G.nodes[X]["x"], G.nodes[X]["y"]
        x3, y3 = G.nodes[Q]["x"], G.nodes[Q]["y"]
        x4, y4 = G.nodes[Y]["x"], G.nodes[Y]["y"]

        dist1 = _euclidean_distance((x1, y1), (x2, y2))
        dist2 = _euclidean_distance((x3, y3), (x4, y4))
        axis_dist1 = abs(p) * 2
        axis_dist2 = abs(q) * 2
        if _same_distance(axis_dist1, dist1) and _same_distance(axis_dist2, dist2):
            return 1

    elif (
        (_same_rel_position(p, y, tolerance) and _same_rel_position(x, q, tolerance))
        and (_is_positive(p) != _is_positive(y))
        and (_is_positive(x) != _is_positive(q))
    ):
        # Distances are equal and signs are different
        x1, y1 = G.nodes[P]["x"], G.nodes[P]["y"]
        x2, y2 = G.nodes[Y]["x"], G.nodes[Y]["y"]
        x3, y3 = G.nodes[Q]["x"], G.nodes[Q]["y"]
        x4, y4 = G.nodes[X]["x"], G.nodes[X]["y"]

        dist1 = _euclidean_distance((x1, y1), (x2, y2))
        dist2 = _euclidean_distance((x3, y3), (x4, y4))
        axis_dist1 = abs(p) * 2
        axis_dist2 = abs(q) * 2
        if _same_distance(axis_dist1, dist1) and _same_distance(axis_dist2, dist2):
            return 1
    else:
        return 0


def _same_rel_position(a, b, tolerance=0):
    """Helper function to determine if two nodes are in the same postion (regardless of sign compared to the bisector), with some tolerance."""
    if tolerance == 0:
        return abs(a) == abs(b)
    else:
        return abs(abs(a) - abs(b)) <= tolerance


def _same_distance(a, b, tolerance=0.5):
    """Helper function to determine if two distances are the same, with some tolerance."""
    return abs(abs(a) - abs(b)) <= tolerance


def _graph_to_points(G, edges=None):
    """Helper function for convex hulls which converts a graph's nodes to a list of points."""
    points = []

    if edges is None:
        for n in G.nodes:
            p1_x, p1_y = G.nodes[n]["x"], G.nodes[n]["y"]
            points.append((p1_x, p1_y))

    else:
        for e in edges:
            p1_x, p1_y = G.nodes[e[0]]["x"], G.nodes[e[0]]["y"]
            p2_x, p2_y = G.nodes[e[1]]["x"], G.nodes[e[1]]["y"]
            points.append((p1_x, p1_y))
            points.append((p2_x, p2_y))

    return points


def get_bounding_box(G):
    """Helper function to get the bounding box of the graph."""

    # Start with a random node
    first_node = rand.sample(list(G.nodes), 1)[0]
    min_x, min_y = G.nodes[first_node]["x"], G.nodes[first_node]["y"]
    max_x, max_y = G.nodes[first_node]["x"], G.nodes[first_node]["y"]

    # Get the maximum and minimum x and y positions.
    for node in G.nodes:
        x = G.nodes[node]["x"]
        y = G.nodes[node]["y"]

        if x > max_x:
            max_x = x

        if x < min_x:
            min_x = x

        if y > max_y:
            max_y = y

        if y < min_y:
            min_y = y

    return ((min_x, min_y), (max_x, max_y))


def _euclidean_distance(a, b):
    """Helper function to get the euclidean distance between two points a and b."""
    return math.sqrt(((b[0] - a[0]) ** 2) + ((b[1] - a[1]) ** 2))


def _midpoint(a, b, G=None):
    """Given two nodes and the graph they are in, return the midpoint between them"""
    if G is None:
        G = self.graph

    x1, y1 = G.nodes[a]["x"], G.nodes[a]["y"]
    x2, y2 = G.nodes[b]["x"], G.nodes[b]["y"]

    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2

    return (mid_x, mid_y)


def _in_circle(x, y, center_x, center_y, r):
    """Return true if the point x, y is inside or on the perimiter of the circle with center center_x, center_y and radius r"""
    return ((x - center_x) ** 2 + (y - center_y) ** 2) <= r**2


def _circles_intersect(x1, y1, x2, y2, r1, r2):
    """Returns true if two circles touch or intersect."""
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2) <= (r1 + r2) * (r1 + r2)


def _is_point_inside_square(point, square_bottom_left, square_top_right):
    x, y = point
    x1, y1 = square_bottom_left
    x2, y2 = square_top_right

    return x1 <= x <= x2 and y1 <= y <= y2


def avg_degree(G):
    degs = []
    for n in G.nodes():
        degs.append(G.degree(n))

    return sum(degs) / G.number_of_nodes()


def _find_k_nearest_points(p, points, k):
    tree = KDTree(points)
    distances, indices = tree.query(p, k=k)
    return [points[i] for i in indices]


def pretty_print_nodes(G):
    """Prints the nodes in the graph and their attributes"""
    for n in G.nodes(data=True):
        print(n)


def _on_opposite_sides(a, b, line):
    """Check if two lines pass the on opposite sides test. Return True if they do."""
    g = (line[1][0] - line[0][0]) * (a[1] - line[0][1]) - (line[1][1] - line[0][1]) * (
        a[0] - line[0][0]
    )
    h = (line[1][0] - line[0][0]) * (b[1] - line[0][1]) - (line[1][1] - line[0][1]) * (
        b[0] - line[0][0]
    )
    return g * h <= 0.0 and (
        a != line[1] and b != line[0] and a != line[0] and b != line[1]
    )


def _bounding_box(line_a, line_b):
    """Check if two lines pass the bounding box test. Return True if they do."""
    x1 = min(line_a[0][0], line_a[1][0])
    x2 = max(line_a[0][0], line_a[1][0])
    x3 = min(line_b[0][0], line_b[1][0])
    x4 = max(line_b[0][0], line_b[1][0])

    y1 = min(line_a[0][1], line_a[1][1])
    y2 = max(line_a[0][1], line_a[1][1])
    y3 = min(line_b[0][1], line_b[1][1])
    y4 = max(line_b[0][1], line_b[1][1])

    return x4 >= x1 and y4 >= y1 and x2 >= x3 and y2 >= y3


def _intersect(line_a, line_b):
    """Check if two lines intersect by checking the on opposite sides and bounding box
    tests. Return True if they do."""
    return (
        _on_opposite_sides(line_a[0], line_a[1], line_b)
        and _on_opposite_sides(line_b[0], line_b[1], line_a)
        and _bounding_box(line_a, line_b)
    )


def draw_graph(G, flip=True):
    """Draws the graph using standard NetworkX methods with matplotlib. Due to the nature of the coordinate systems used,
    graphs will be flipped on the X axis. To see the graph the way it would be drawn in yEd, set flip to True (default=True).
    """

    if flip:
        pos = {
            k: np.array((v["x"], 0 - float(v["y"])), dtype=np.float32)
            for (k, v) in [u for u in G.nodes(data=True)]
        }
    else:
        pos = {
            k: np.array((v["x"], v["y"]), dtype=np.float32)
            for (k, v) in [u for u in G.nodes(data=True)]
        }

    nx.draw(G, pos=pos, with_labels=True)
    plt.show()
