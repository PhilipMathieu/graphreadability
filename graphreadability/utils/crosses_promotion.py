import networkx as nx
from .helpers import _intersect, compute_intersection


def crosses_promotion(G):
    """
    Promote crossings in a graph to nodes, creating a new graph with no edge crossings.

    Parameters:
    - G: NetworkX graph object

    Returns:
    - H: NetworkX graph object
    """
    H = G.copy()  # Create a copy of the input graph

    for n in H.nodes():
        H.nodes[n]["type"] = "major"  # Set the "type" attribute of each node to "major"

    covered = []  # List to keep track of covered edges
    intersections = {}  # Dictionary to store intersections between edges
    for u, v in H.edges():
        for x, y in H.edges():
            if (u, v) == (x, y):
                continue  # Skip if the edges are the same

            if ((u, v), (x, y)) in covered:
                continue  # Skip if the edges have already been covered

            line_a = (
                (H.nodes[u]["x"], H.nodes[u]["y"]),
                (H.nodes[v]["x"], H.nodes[v]["y"]),
            )  # Line segment of edge (u, v)
            line_b = (
                (H.nodes[x]["x"], H.nodes[x]["y"]),
                (H.nodes[y]["x"], H.nodes[y]["y"]),
            )  # Line segment of edge (x, y)

            if _intersect(line_a, line_b):  # Check if the line segments intersect
                try:
                    intersection = compute_intersection(
                        line_a[0], line_a[1], line_b[0], line_b[1]
                    )  # Compute the intersection point
                    if (u, v) not in intersections.keys():
                        intersections[(u, v)] = (
                            []
                        )  # Initialize the list of intersections for edge (u, v)

                    if (x, y) not in intersections.keys():
                        intersections[(x, y)] = (
                            []
                        )  # Initialize the list of intersections for edge (x, y)

                    intersections[(u, v)].append(
                        (intersection[0], intersection[1])
                    )  # Add the intersection point to the list
                    intersections[(x, y)].append(
                        (intersection[0], intersection[1])
                    )  # Add the intersection point to the list
                except ZeroDivisionError:
                    pass

                covered.append(((x, y), (u, v)))  # Mark the edges as covered

    intersections_covered = []  # List to keep track of covered intersections

    for k, v in intersections.items():
        H.remove_edge(
            k[0], k[1]
        )  # Remove the original edge (k[0], k[1]) from the graph

        node_list = []  # List to store the nodes involved in the crossing

        points = sorted(
            v, key=lambda v: v[0]
        )  # Sort the intersection points by x-coordinate

        if H.nodes[k[0]]["x"] < points[0][0]:
            node_list.append(
                k[0]
            )  # Add the source node of the original edge to the node list
        else:
            node_list.append(
                k[1]
            )  # Add the target node of the original edge to the node list

        for x, y in points:
            if (x, y) not in intersections_covered:
                new_node = "c" + str(len(H.nodes()))  # Generate a new node label
                H.add_node(new_node)  # Add the new node to the graph
                H.nodes[new_node]["label"] = "\n"  # Set the label of the new node
                H.nodes[new_node][
                    "shape_type"
                ] = "ellipse"  # Set the shape type of the new node
                H.nodes[new_node]["x"] = x  # Set the x-coordinate of the new node
                H.nodes[new_node]["y"] = y  # Set the y-coordinate of the new node
                H.nodes[new_node][
                    "type"
                ] = "minor"  # Set the "type" attribute of the new node to "minor"
                H.nodes[new_node][
                    "color"
                ] = "#3BC6E5"  # Set the color of the new node to blue
                node_list.append(new_node)  # Add the new node to the node list
                intersections_covered.append((x, y))  # Mark the intersection as covered
            else:
                node = [
                    a for a, b in H.nodes(data=True) if b["x"] == x and b["y"] == y
                ]  # Find the existing node with the same coordinates
                node_list.append(node[0])  # Add the existing node to the node list

        if H.nodes[k[0]]["x"] < points[0][0]:
            node_list.append(
                k[1]
            )  # Add the target node of the original edge to the node list
        else:
            node_list.append(
                k[0]
            )  # Add the source node of the original edge to the node list

        for i in range(len(node_list) - 1):
            H.add_edge(
                node_list[i], node_list[i + 1]
            )  # Add edges between consecutive nodes in the node list

        H.remove_edges_from(
            nx.selfloop_edges(H)
        )  # Remove self-loop edges from the graph

    return H  # Return the modified graph
