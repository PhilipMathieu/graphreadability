"""
This module is based on:

    C. Dunne, S. I. Ross, B. Shneiderman, and M. Martino, “Readability metric feedback for aiding
    node-link visualization designers,” IBM Journal of Research and Development, vol. 59, no. 2/3,
    p. 14:1-14:16, Mar. 2015, doi: 10.1147/JRD.2015.2411412.
"""

import networkx as nx
import numpy as np
from src.utils.helpers import (
    divide_or_zero,
    lines_intersect,
    calculate_angle_between_vectors,
)


class ReadabilityGraph(nx.Graph):
    def __init__(self, data=None, **attr):
        super().__init__(data, **attr)
        self.edge_crossing_list = (
            None  # Store computed edge crossings to avoid recomputation
        )
        self.edge_angle_dict = None  # Store computed edge angles to avoid recomputation
        self.node_overlap_list = (
            None  # Store computed node overlaps to avoid recomputation
        )

    # HELPER FUNCTIONS #
    def edge_vector(self, edge):
        """Calculate the vector of an edge given its nodes' positions."""
        pos1, pos2 = self.nodes[edge[0]]["pos"], self.nodes[edge[1]]["pos"]
        return np.array(pos2) - np.array(pos1)

    def calculate_edge_crossings(self):
        positions = nx.get_node_attributes(
            self, "pos"
        )  # Assuming 'pos' contains node positions
        crossings = set()
        angles = {}
        edges_checked = set()
        edge_crossings = {edge: 0 for edge in self.edges}

        for edge1 in self.edges:
            for edge2 in self.edges:
                if edge1 != edge2 and (edge2, edge1) not in edges_checked:
                    edges_checked.add((edge1, edge2))
                    line1 = (positions[edge1[0]], positions[edge1[1]])
                    line2 = (positions[edge2[0]], positions[edge2[1]])
                    # Skip edges that share a node
                    if len(set(edge1) & set(edge2)) > 0:
                        continue
                    if lines_intersect(line1, line2):
                        crossings.add((edge1, edge2))
                        # Calculate angle between edges
                        angle = calculate_angle_between_vectors(
                            self.edge_vector(edge1), self.edge_vector(edge2)
                        )
                        angles[edge1, edge2] = angle
                        edge_crossings[edge1] += 1
                        edge_crossings[edge2] += 1

        # Save edge angles to edge data
        self.edge_angle_dict = angles

        # Save edge crossings to edge data
        nx.set_edge_attributes(self, edge_crossings, "crossings_count")
        return crossings

    def calculate_node_node_overlap(self):
        """ """
        # Placeholder for overlaps
        overlaps = set()

        # Placeholder for overlap counts
        overlap_count = {node: 0 for node in self.nodes}

        # Iterate over all pairs of nodes to check for overlap
        for node1, data1 in self.nodes(data=True):
            for node2, data2 in self.nodes(data=True):
                if node1 < node2:  # Ensure pairs are processed only once
                    pos1 = np.array(data1["pos"])  # Position of node1
                    pos2 = np.array(data2["pos"])  # Position of node2
                    size1 = data1["size"]  # Size of node1
                    size2 = data2["size"]  # Size of node2

                    # Calculate distance between node centers
                    distance = np.linalg.norm(pos1 - pos2)

                    # Calculate sum of radii
                    sum_of_radii = size1 / 2.0 + size2 / 2.0

                    # Check for overlap
                    if distance < sum_of_radii:
                        # Add overlap to set
                        overlaps.add((node1, node2))
                        # Increment overlap count for both nodes
                        overlap_count[node1] += 1
                        overlap_count[node2] += 1

        # Save overlap counts to node data
        nx.set_node_attributes(self, overlap_count, "overlap_count")

        return overlaps

    # METRICS #
    def node_overlap_global(self):
        # Implement computation for global overlap metric
        pass

    def node_overlap_node(self):
        """
        Node overlap is the proportion of the node's representation that is not obscured by other nodes.
        This method currently does not compute the actual overlap, but simply returns True if more than 0
        overlaps are found, and False otherwise.
        """
        if self.node_overlap_list is None:
            self.node_overlap_list = self.calculate_node_node_overlap()

        return {
            node: overlaps > 0
            for node, overlaps in nx.get_node_attributes(self, "overlap_count").items()
        }

    def edge_crossings_global(self):
        """Return the number of edge crossings in the graph."""
        if self.edge_crossing_list is None:
            self.edge_crossing_list = self.calculate_edge_crossings()

        c = len(self.edge_crossing_list)
        m = len(self.edges)
        c_all = m * (m - 1) / 2
        degree = np.array([degree[1] for degree in self.degree()])
        c_impossible = np.dot(degree, degree - 1) / 2
        c_max = c_all - c_impossible

        return 1 - divide_or_zero(c, c_max)

    def edge_crossings_edge(self):
        """Return the number of edge crossings in the graph."""
        if self.edge_crossing_list is None:
            self.edge_crossing_list = self.calculate_edge_crossings()

        m = len(self.edges)
        c_all = m - 1

        crossings = {}
        for edge in self.edges:
            src = edge[0]
            dst = edge[1]
            c_impossible = self.degree(src) + self.degree(dst) - 2
            c_max = c_all - c_impossible
            crossings[edge] = 1 - divide_or_zero(
                self.edges[edge]["crossings_count"], c_max
            )

        return crossings

    def edge_crossings_node(self):
        """ """
        if self.edge_crossing_list is None:
            self.edge_crossing_list = self.calculate_edge_crossings()

        # Get all the edges for each node
        node_edges = {node: [] for node in self.nodes}
        for edge in self.edges:
            node_edges[edge[0]].append(edge)
            node_edges[edge[1]].append(edge)
            node_edges[edge[0]].sort()
            node_edges[edge[1]].sort()

        # Compute edge crossings for each node
        crossings = {}
        for node in self.nodes:
            edges = node_edges[node]
            m = len(edges)
            c_max = sum(
                [m + 1 - self.degree(edge[0]) - self.degree(edge[1]) for edge in edges]
            )
            crossings[node] = 1 - divide_or_zero(
                sum([self.edges[edge]["crossings_count"] for edge in edges]), c_max
            )

        return crossings

    def edge_crossing_angles_edge(self, ideal_angle=70):
        """
        Edge crossing angle is defined for any edge as the average deviation of its individual edge
        crossing angles from an ideal angle of 70 degrees."""
        if self.edge_crossing_list is None:
            self.edge_crossing_list = self.calculate_edge_crossings()

        deviation = {edge: 0 for edge in self.edges}
        deviation_max = {edge: 0 for edge in self.edges}
        for edge1, edge2 in self.edge_crossing_list:
            deviation[edge1] += self.edge_angle_dict[edge1, edge2] - ideal_angle
            deviation[edge2] += self.edge_angle_dict[edge1, edge2] - ideal_angle
            deviation_max[edge1] += ideal_angle
            deviation_max[edge2] += ideal_angle

        return [
            1 - divide_or_zero(deviation[edge], deviation_max[edge])
            for edge in self.edges
        ]

    def edge_crossing_angles_global(self, ideal_angle=70):
        """"""
        if self.edge_crossing_list is None:
            self.edge_crossing_list = self.calculate_edge_crossings()

        deviation = 0
        deviation_max = 0
        for edge1, edge2 in self.edge_crossing_list:
            deviation += self.edge_angle_dict[edge1, edge2] - ideal_angle
            deviation_max += ideal_angle

        return 1 - divide_or_zero(deviation, deviation_max)

    def compute_metrics(self):
        # Return a dictionary of all metrics
        metrics = {
            "node_overlap_node": self.node_overlap_node(),
            "edge_crossings_global": self.edge_crossings_global(),
            "edge_crossings_edge": self.edge_crossings_edge(),
            "edge_crossings_node": self.edge_crossings_node(),
            "edge_crossing_angles_edge": self.edge_crossing_angles_edge(),
            "edge_crossing_angles_global": self.edge_crossing_angles_global(),
        }
        return metrics


if __name__ == "__main__":
    # Example usage
    G = ReadabilityGraph()
    G.add_nodes_from(
        [
            (1, {"pos": (0, 0), "size": 1}),
            (2, {"pos": (1, 1), "size": 2}),
            (3, {"pos": (1, -1), "size": 0.5}),
            (4, {"pos": (2, 0), "size": 1}),
        ]
    )
    G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)])
    print(G.compute_metrics())
    print(G.edge_crossing_list)
    print(G.edge_angle_dict)
    print(G.node_overlap_list)
    import matplotlib.pyplot as plt

    nx.draw(
        G,
        pos=nx.get_node_attributes(G, "pos"),
        with_labels=True,
    )
    plt.show()
