import unittest
import networkx as nx
from graphreadability import ReadabilityGraph


class TestReadabilityGraph(unittest.TestCase):
    def setUp(self):
        self.graph = ReadabilityGraph()

    def test_edge_vector(self):
        self.graph.add_node("A", pos=(0, 0))
        self.graph.add_node("B", pos=(1, 1))
        vector = self.graph.edge_vector(("A", "B"))
        self.assertEqual(vector.tolist(), [1, 1])

    def test_calculate_edge_crossings(self):
        self.graph.add_node("A", pos=(0, 0))
        self.graph.add_node("B", pos=(1, 1))
        self.graph.add_node("C", pos=(2, 0))
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        crossings = self.graph.calculate_edge_crossings()
        self.assertEqual(len(crossings), 0)

    def test_calculate_node_node_overlap(self):
        self.graph.add_node("A", pos=(0, 0), size=1)
        self.graph.add_node("B", pos=(2, 0), size=1)
        self.graph.add_node("C", pos=(1, 0), size=1)
        overlaps = self.graph.calculate_node_node_overlap()
        self.assertEqual(len(overlaps), 0)

    def test_node_overlap_node(self):
        self.graph.add_node("A", pos=(0, 0), size=1)
        self.graph.add_node("B", pos=(1, 0), size=1)
        self.graph.add_edge("A", "B")
        overlap = self.graph.node_overlap_node()
        self.assertEqual(overlap["A"], False)
        self.assertEqual(overlap["B"], False)

    def test_edge_crossings_global(self):
        self.graph.add_node("A", pos=(0, 0))
        self.graph.add_node("B", pos=(1, 1))
        self.graph.add_node("C", pos=(2, 0))
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        crossings = self.graph.edge_crossings_global()
        self.assertEqual(crossings, 0)

    def test_edge_crossings_edge(self):
        self.graph.add_node("A", pos=(0, 0))
        self.graph.add_node("B", pos=(1, 1))
        self.graph.add_node("C", pos=(2, 0))
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        crossings = self.graph.edge_crossings_edge()
        self.assertEqual(crossings[("A", "B")], 0)
        self.assertEqual(crossings[("B", "C")], 0)

    def test_edge_crossings_node(self):
        self.graph.add_node("A", pos=(0, 0))
        self.graph.add_node("B", pos=(1, 1))
        self.graph.add_node("C", pos=(2, 0))
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        crossings = self.graph.edge_crossings_node()
        self.assertEqual(crossings["A"], 0)
        self.assertEqual(crossings["B"], 0)
        self.assertEqual(crossings["C"], 0)

    def test_edge_crossing_angles_edge(self):
        self.graph.add_node("A", pos=(0, 0))
        self.graph.add_node("B", pos=(1, 1))
        self.graph.add_node("C", pos=(2, 0))
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        angles = self.graph.edge_crossing_angles_edge()
        self.assertEqual(len(angles), 2)

    def test_edge_crossing_angles_global(self):
        self.graph.add_node("A", pos=(0, 0))
        self.graph.add_node("B", pos=(1, 1))
        self.graph.add_node("C", pos=(2, 0))
        self.graph.add_edge("A", "B")
        self.graph.add_edge("B", "C")
        angle = self.graph.edge_crossing_angles_global()
        self.assertEqual(angle, 0)


if __name__ == "__main__":
    unittest.main()
