import unittest
import networkx as nx
import graphvizrm


class TestGraph(unittest.TestCase):
    """Basic graph tests."""

    def setUp(self):
        # Setup code, run before each test
        self.graph = nx.Graph()

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


class TestGraphMonkeyPatching(unittest.TestCase):
    """Tests for the monkey-patched graph class."""

    def setUp(self):
        # Setup a graph instance for each test
        self.G = nx.Graph()

    def test_layout_positions(self):
        # Assuming set_layout_positions was monkey patched onto nx.Graph
        positions = {1: (0, 0), 2: (1, 1)}
        self.G.set_layout_positions(positions)
        self.assertEqual(
            self.G.layout_positions, positions, "Layout positions not set correctly"
        )

    def test_is_cartesian_grid(self):
        # Assuming is_cartesian_grid was monkey patched to include a setter and getter
        self.G.is_cartesian_grid = True
        self.assertTrue(
            self.G.is_cartesian_grid, "Graph should be marked as Cartesian grid"
        )

        self.G.is_cartesian_grid = False
        self.assertFalse(
            self.G.is_cartesian_grid, "Graph should not be marked as Cartesian grid"
        )

    def test_metadata(self):
        # Assuming add_metadata and get_metadata were monkey patched onto nx.Graph
        metadata = {"description": "Example graph", "year": 2024}
        self.G.add_metadata(**metadata)

        for key, value in metadata.items():
            self.assertEqual(
                self.G.get_metadata(key), value, f"Metadata {key} not set correctly"
            )

        # Test default value for non-existent metadata
        self.assertIsNone(
            self.G.get_metadata("nonexistent_key"),
            "Default value for nonexistent metadata key should be None",
        )


if __name__ == "__main__":
    unittest.main()
