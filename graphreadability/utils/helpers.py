from scipy.spatial import KDTree
import numpy as np
import networkx as nx


# MATH HELPERS #
def _is_positive(x: int | float | np.ndarray) -> bool:
    """Return true if x is positive."""
    return x > 0


def divide_or_zero(
    a: int | float | np.ndarray, b: int | float | np.ndarray
) -> int | float | np.ndarray:
    """Return 0 if b is 0, otherwise divide a by b."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0.0)


# GEOMETRY HELPERS #
"""
Functions in this section should work on any number of dimensions, but are primarily used in 2D.
"""


def edge_vector(edge):
    """Convert an edge or line to a vector."""
    return np.array(edge[1]) - np.array(edge[0])


def calculate_angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate the angle between two vectors."""
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    return np.degrees(angle)


def _in_circle(x, y, center_x, center_y, r):
    """Return true if the point x, y is inside or on the perimeter of the circle with center center_x, center_y and radius r"""
    return np.square(x - center_x) + np.square(y - center_y) <= np.square(r)


def _are_collinear_points(a, b, c):
    """Return true if the three points are collinear."""
    # Check that all three points are (x, y) pairs
    if not all(isinstance(p, (list, np.ndarray)) for p in [a, b, c]):
        raise TypeError(
            f"Expected a, b, and c to be a list or numpy array, got {type(a)}, {type(b)}, and {type(c)}"
        )
    simplex = np.array([a, b, c])
    simplex = np.column_stack((simplex, np.ones(3)))
    return np.isclose(np.linalg.det(simplex), 0)


def _rel_point_line_dist(axis, x, y):
    """Return the relative distance of a point to a line."""
    gradient = (
        (axis[1][1] - axis[0][1]) / (axis[1][0] - axis[0][0])
        if axis[1][0] - axis[0][0] != 0
        else np.inf
    )
    y_intercept = axis[0][1] - gradient * axis[0][0]
    if gradient == 0:
        return np.abs(y - y_intercept)
    if np.isinf(gradient):
        return np.abs(x - axis[0][0])
    return np.abs(y - gradient * x - y_intercept) / np.sqrt(1 + gradient**2)


def _euclidean_distance(a, b):
    """Helper function to get the euclidean distance between two points a and b."""
    return np.linalg.norm(np.array(a) - np.array(b))


def _same_distance(a, b, tolerance=0.5):
    """Helper function to determine if two values are the same, with some tolerance, regardless of sign."""
    return np.isclose(np.abs(a) - np.abs(b), 0, atol=tolerance)


def _bounding_box(points):
    """Return the bounding cube of a set of points in any number of dimensions."""
    return np.array([np.min(points, axis=0), np.max(points, axis=0)])


def _midpoint_nd(a, b):
    """Return the midpoint between two points in any number of dimensions."""
    return (a + b) / 2


def _circles_intersect_nd(c1, c2, r1, r2):
    """Return true if two balls intersect."""
    return np.linalg.norm(c1 - c2) <= r1 + r2


def _circles_intersect(x1, y1, x2, y2, r1, r2):
    """Returns true if two circles touch or intersect."""
    return _circles_intersect_nd(np.array([x1, y1]), np.array([x2, y2]), r1, r2)


def _in_rectangle(point, minima, maxima):
    """Return true if the point is inside the rectangle defined by the given points."""
    return np.all(np.logical_and(minima <= point, point <= maxima))


def _is_point_inside_square(x, y, x1, y1, x2, y2):
    """Return true if the point x, y is inside the square defined by the points x1, y1, x2, y2."""
    return _in_rectangle(np.array([x, y]), np.array([x1, y1]), np.array([x2, y2]))


def _on_opposite_sides(a, b, line):
    """Check if two points are on opposite sides of a line. Return True if they are."""
    x1, y1 = line[0]
    x2, y2 = line[1]
    x3, y3 = a
    x4, y4 = b

    # Return false if either point is on the line
    if np.isclose((x1 - x2) * (y3 - y1), (y1 - y2) * (x3 - x1)):
        return False

    return np.all(
        np.sign((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1))
        != np.sign((x2 - x1) * (y4 - y1) - (y2 - y1) * (x4 - x1))
    )


def _bounding_box_lines(line_a, line_b):
    """Check if the bounding boxes of two lines intersect. Return True if they do."""
    x1 = np.minimum(line_a[0][0], line_a[1][0])
    x2 = np.maximum(line_a[0][0], line_a[1][0])
    x3 = np.minimum(line_b[0][0], line_b[1][0])
    x4 = np.maximum(line_b[0][0], line_b[1][0])

    y1 = np.minimum(line_a[0][1], line_a[1][1])
    y2 = np.maximum(line_a[0][1], line_a[1][1])
    y3 = np.minimum(line_b[0][1], line_b[1][1])
    y4 = np.maximum(line_b[0][1], line_b[1][1])

    return np.logical_and(
        x4 >= x1, np.logical_and(y4 >= y1, np.logical_and(x2 >= x3, y2 >= y3))
    )


def _build_kd_tree(points):
    """Create a KDTree from a set of points."""
    return KDTree(points)


def _find_k_nearest_points(p, k, points=None, tree=None):
    """Find the k nearest points to a given point p."""
    # Promote points to numpy array
    if tree is None:
        tree = _build_kd_tree(points)
    distances, indices = tree.query(p, k=k)
    if points:
        return points[indices.astype(int)]
    return indices


def lines_intersect(line_a, line_b):
    """Check if two lines (each defined by two points) intersect."""
    p1, p2, p3, p4 = line_a[0], line_a[1], line_b[0], line_b[1]
    # Calculate parts of the determinants
    det1 = (p1[0] - p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (p3[0] - p4[0])
    det2 = (p1[0] * p2[1] - p1[1] * p2[0]) * (p3[0] - p4[0]) - (p1[0] - p2[0]) * (
        p3[0] * p4[1] - p3[1] * p4[0]
    )
    det3 = (p1[0] * p2[1] - p1[1] * p2[0]) * (p3[1] - p4[1]) - (p1[1] - p2[1]) * (
        p3[0] * p4[1] - p3[1] * p4[0]
    )
    det1_zero = np.isclose(det1, 0)
    x = np.where(det1_zero, 0, det2 / det1)
    y = np.where(det1_zero, 0, det3 / det1)
    # Check if intersection point is on both line segments
    line1_x_range = np.sort([p1[0], p2[0]])
    line1_y_range = np.sort([p1[1], p2[1]])
    line2_x_range = np.sort([p3[0], p4[0]])
    line2_y_range = np.sort([p3[1], p4[1]])
    return (
        np.logical_and(line1_x_range[0] <= x, x <= line1_x_range[1])
        & np.logical_and(line2_x_range[0] <= x, x <= line2_x_range[1])
        & np.logical_and(line1_y_range[0] <= y, y <= line1_y_range[1])
        & np.logical_and(line2_y_range[0] <= y, y <= line2_y_range[1])
    )


def _intersect(line_a, line_b):
    """Check if two lines intersect by checking the on opposite sides and bounding box tests. Return True if they do."""
    return np.logical_and(
        _on_opposite_sides(line_a[0], line_a[1], line_b),
        np.logical_and(
            _on_opposite_sides(line_b[0], line_b[1], line_a),
            _bounding_box_lines(line_a, line_b),
        ),
    )


def compute_intersection(p1, q1, p2, q2):
    """Compute the intersection point of two lines."""
    x1, y1 = p1
    x2, y2 = q1
    x3, y3 = p2
    x4, y4 = q2
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
        (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    )

    px[px == -0.0] = 0
    py[py == -0.0] = 0.0
    return px, py


# NODE HELPERS #
"""
Functions in this section are used to determine properties of nodes in a graph. They may assume (when needed):
1. The graph is a class extending the NetworkX Graph() class.
2. The nodes have attributes "x" and "y" which represent their coordinates.
3. The nodes have attributes "width" and "height" which represent their dimensions.
4. Other parameters are passed as scalers or numpy arrays.
"""


def _same_position(n1, n2, G, tolerance=0):
    """Helper function to determine if two nodes are in the same position, with some tolerance."""
    x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
    x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]

    if tolerance == 0:
        return x1 == x2 and y1 == y2

    return _in_circle(np.array([x1, y1]), np.array([x2, y2]), tolerance)


def _are_collinear(n1, n2, n3, G):
    """Returns true if the three points are collinear, by checking if the determinant is 0."""
    x1, y1 = G.nodes[n1]["x"], G.nodes[n1]["y"]
    x2, y2 = G.nodes[n2]["x"], G.nodes[n2]["y"]
    x3, y3 = G.nodes[n3]["x"], G.nodes[n3]["y"]

    return _are_collinear_points((x1, y1), (x2, y2), (x3, y3))


def _check_shared_node_symmetric(P, X, Q, Y, tolerance, G):
    """Helper function to determine if the edges are symmetric about the axis."""
    # Check if one of the nodes is shared
    if (P == X or Q == X) != (P == Y or Q == Y):
        return False

    # Get the coordinates of the nodes
    P_x, P_y = G.nodes[P]["x"], G.nodes[P]["y"]
    X_x, X_y = G.nodes[X]["x"], G.nodes[X]["y"]
    Q_x, Q_y = G.nodes[Q]["x"], G.nodes[Q]["y"]
    Y_x, Y_y = G.nodes[Y]["x"], G.nodes[Y]["y"]

    # Calculate the distances between the nodes
    p = _euclidean_distance((P_x, P_y), (X_x, X_y))
    q = _euclidean_distance((Q_x, Q_y), (Y_x, Y_y))
    x = _euclidean_distance((P_x, P_y), (Y_x, Y_y))
    y = _euclidean_distance((Q_x, Q_y), (X_x, X_y))

    # Check if the distances are the same
    return _same_distance(p, y, tolerance) and _same_distance(q, x, tolerance)


def _is_minor(node, G):
    """Returns True if a node was created by crosses promotion."""
    try:
        return G.nodes[node]["type"] == "minor"
    except KeyError:
        return False


def _sym_value(e1, e2, G):
    """Helper function to calculate the level of symmetry between two edges, based on whoch nodes were crosses promoted."""
    # The end nodes of edge1 are P and Q
    # The end nodes of edge2 are X and Y
    P, Q, X, Y = e1[0], e1[1], e2[0], e2[1]

    if _is_minor(P, G) == _is_minor(X, G) and _is_minor(Q, G) == _is_minor(Y, G):
        # P=X and Q=Y
        return 1
    elif _is_minor(P, G) == _is_minor(Y, G) and _is_minor(Q, G) == _is_minor(X, G):
        # P=Y and X=Q
        return 1
    elif _is_minor(P, G) == _is_minor(X, G) and _is_minor(Q, G) != _is_minor(Y, G):
        # P=X but Q!=Y
        return 0.5
    elif _is_minor(P, G) == _is_minor(Y, G) and _is_minor(Q, G) != _is_minor(X, G):
        # P=Y but Q!=X
        return 0.5
    elif _is_minor(P, G) != _is_minor(X, G) and _is_minor(Q, G) == _is_minor(Y, G):
        # P!=X but Q==Y
        return 0.5
    elif _is_minor(P, G) != _is_minor(Y, G) and _is_minor(Q, G) == _is_minor(X, G):
        # P!=Y but Q==X
        return 0.5
    elif _is_minor(P, G) != _is_minor(X, G) and _is_minor(Q, G) != _is_minor(Y, G):
        # P!=X and Q!=Y
        return 0.25
    elif _is_minor(P, G) != _is_minor(Y, G) and _is_minor(Q, G) != _is_minor(X, G):
        # P!=Y and Q!=X
        return 0.25


def _find_bisectors(G):
    """Returns the set of perpendicular bisectors between every pair of nodes"""
    bisectors = []
    covered = []

    # For each pair of nodes
    for n1 in G.nodes:
        for n2 in G.nodes:
            if n1 == n2 or (n1, n2) in covered:
                continue
            n1_x, n1_y = G.nodes[n1]["x"], G.nodes[n1]["y"]
            n2_x, n2_y = G.nodes[n2]["x"], G.nodes[n2]["y"]

            # Get the midpoint between the two nodes
            midpoint_x = (n2_x + n1_x) / 2
            midpoint_y = (n2_y + n1_y) / 2

            # Get the gradient of perpendicular bisector
            try:
                initial_gradient = (n2_y - n1_y) / (n2_x - n1_x)
                perp_gradient = (1 / initial_gradient) * -1
                c = midpoint_y - (perp_gradient * midpoint_x)

            except ZeroDivisionError:
                if n2_x == n1_x:
                    perp_gradient = "x"
                    c = midpoint_y

                elif n2_y == n1_y:
                    perp_gradient = "y"
                    c = midpoint_x

            grad_c = (perp_gradient, c)

            # Convert to a pair of points
            axis = np.array([(0, c), (1, perp_gradient + c)])
            # Move to midpoint
            axis[:, 0] += midpoint_x
            axis[:, 1] += midpoint_y

            bisectors.append(axis)
            covered.append((n2, n1))

    return bisectors


def _mirror(axis, e1, e2, G, tolerance=0):
    """
    Determine if two edges are mirrored about a bisecting axis.

    Parameters:
    axis (str): The axis to check for mirroring. Can be "x" or "y".
    e1 (tuple): The first edge represented as a tuple of node indices.
    e2 (tuple): The second edge represented as a tuple of node indices.
    G (networkx.Graph): The graph containing the nodes and edges.
    tolerance (float, optional): The tolerance for comparing distances. Defaults to 0.

    Returns:
    bool: True if the edges are mirrored about the axis, False otherwise.
    """

    # Check if the same edge
    if np.array_equal(e1, e2):
        return False

    if isinstance(axis, str):
        # If axis is "x" or "y", then the bisector is a vertical or horizontal line
        if axis == "x":
            axis = np.array([(0, 0), (0, 1)])
        elif axis == "y":
            axis = np.array([(0, 0), (1, 0)])
        else:
            raise ValueError("Axis must be 'x' or 'y' or numpy array.")

    # Get the coordinates of the nodes of edge1
    e1_p1 = np.array([G.nodes[e1[0]]["x"], G.nodes[e1[0]]["y"]])
    e1_p2 = np.array([G.nodes[e1[1]]["x"], G.nodes[e1[1]]["y"]])

    # Get the coordinates of the nodes of edge2
    e2_p1 = np.array([G.nodes[e2[0]]["x"], G.nodes[e2[0]]["y"]])
    e2_p2 = np.array([G.nodes[e2[1]]["x"], G.nodes[e2[1]]["y"]])

    # The end nodes of edge1 are P and Q
    # The end nodes of edge2 are X and Y
    P, Q, X, Y = e1[0], e1[1], e2[0], e2[1]

    # Calculate the vector distances of the nodes to the axis
    p = _rel_point_line_dist(axis, e1_p1[0], e1_p1[1])
    q = _rel_point_line_dist(axis, e1_p2[0], e1_p2[1])
    x = _rel_point_line_dist(axis, e2_p1[0], e2_p1[1])
    y = _rel_point_line_dist(axis, e2_p2[0], e2_p2[1])

    if (p == 0 and q == 0) or (x == 0 and y == 0):
        # One or both edges are on the axis
        return False

    # Check if the edges cross the axis
    if (np.sign(p) != np.sign(q)) or (np.sign(x) != np.sign(y)):
        # One or both edges cross the axis
        return False

    # Check if the edges are mirrored about the axis
    if (_same_distance(p, x, tolerance) and _same_distance(q, y, tolerance)) or (
        _same_distance(p, y, tolerance) and _same_distance(q, x, tolerance)
    ):
        return True

    # Default to False
    return False


def _graph_to_points(G, edges=None):
    """Helper function for convex hulls which converts a graph's nodes to a list of points. If edges is not None, returns only points from the edges."""
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


def _get_bounding_box(G):
    """Helper function to get the bounding box of the graph."""
    points = _graph_to_points(G)
    return _bounding_box(points)


def _midpoint(a, b, G):
    """Given two nodes and the graph they are in, return the midpoint between them"""
    x1, y1 = G.nodes[a]["x"], G.nodes[a]["y"]
    x2, y2 = G.nodes[b]["x"], G.nodes[b]["y"]
    return _midpoint_nd(np.array([x1, y1]), np.array([x2, y2]))


def avg_degree(G):
    """Return the average degree of a graph."""
    degs = np.array([G.degree(n) for n in G.nodes()])
    return np.mean(degs)


def pretty_print_nodes(G):
    """Prints the nodes in the graph and their attributes"""
    for n in G.nodes(data=True):
        print(n)


def draw_graph(G, flip=True, ax=None, **kwargs):
    """Draws the graph using standard NetworkX methods with matplotlib. Due to the nature of the coordinate systems used,
    graphs will be flipped on the X axis. To see the graph the way it would be drawn in yEd, set flip to True (default=True).
    """
    default_kwargs = {
        "edge_color": "#BDCCB4",
        "node_color": "#778181",
        "linewidths": 3,
        "node_size": 300,
        "width": 5,
        "edgecolors": "#333333",
        "with_labels": False,
    }
    for key, value in default_kwargs.items():
        if key not in kwargs:
            kwargs[key] = value

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

    nx.draw(G, pos=pos, ax=ax, **kwargs)
