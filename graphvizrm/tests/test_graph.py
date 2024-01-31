import unittest
from graphvizrm import Graph


class TestGraph(unittest.TestCase):
    def setUp(self):
        # Setup code, run before each test
        self.graph = Graph()

    def test_graph_initialization(self):
        # Test graph initialization (e.g., empty graph)
        self.assertEqual(len(self.graph.nodes), 0)
        self.assertEqual(len(self.graph.edges), 0)

    def test_add_node(self):
        # Test adding a node
        self.graph.add_node("A")
        self.assertIn("A", self.graph.nodes)

    def test_add_edge(self):
        # Test adding an edge
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_edge("A", "B")
        self.assertIn(("A", "B"), self.graph.edges)

    def test_remove_node(self):
        # Test removing a node
        self.graph.add_node("A")
        self.graph.remove_node("A")
        self.assertNotIn("A", self.graph.nodes)

    def test_remove_edge(self):
        # Test removing an edge
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_edge("A", "B")
        self.graph.remove_edge("A", "B")
        self.assertNotIn(("A", "B"), self.graph.edges)

    def test_set_node_position(self):
        # Test set_node_position function
        self.graph.add_node("A")
        self.graph.set_node_position("A", 0, 0)
        self.assertEqual(self.graph.layout["A"]["x"], 0)
        self.assertEqual(self.graph.layout["A"]["y"], 0)

    def test_set_node_size(self):
        # Test set_node_size function
        self.graph.add_node("A")
        self.graph.set_node_size("A", 100, 100)
        self.assertEqual(self.graph.nodes["A"]["width"], 100)
        self.assertEqual(self.graph.nodes["A"]["height"], 100)

    def test_is_valid(self):
        # Test is_valid function
        self.graph.add_node("A")
        self.graph.add_node("B")
        self.graph.add_node("C")
        self.graph.set_node_position("A", 0, 0)
        self.graph.set_node_position("B", 0, 0)
        self.graph.set_node_position("C", 0, 0)
        self.assertTrue(self.graph.is_valid())

    def tearDown(self):
        # Teardown code, run after each test
        pass


if __name__ == "__main__":
    unittest.main()
