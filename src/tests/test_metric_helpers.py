import networkx as nx
from ..metrics import metric_helpers
import unittest
from collections import defaultdict


class MetricsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        # Complete graph with 4 nodes arranged in a unit square
        G1 = nx.complete_graph(4)
        nx.set_node_attributes(
            G1,
            {
                0: {"x": 0, "y": 0},
                1: {"x": 1, "y": 0},
                2: {"x": 1, "y": 1},
                3: {"x": 0, "y": 1},
            },
        )
        # Cycle graph with 8 nodes arranged in a unit circle
        G2 = nx.cycle_graph(8)
        nx.set_node_attributes(
            G2,
            {
                0: {"x": 1, "y": 0},
                1: {"x": 0.707, "y": 0.707},
                2: {"x": 0, "y": 1},
                3: {"x": -0.707, "y": 0.707},
                4: {"x": -1, "y": 0},
                5: {"x": -0.707, "y": -0.707},
                6: {"x": 0, "y": -1},
                7: {"x": 0.707, "y": -0.707},
            },
        )
        # Six-pointed star graph
        G3 = nx.Graph()
        G3.add_nodes_from([0, 1, 2, 3, 4, 5])
        nx.set_node_attributes(
            G3,
            {
                0: {"x": 0, "y": 0},
                1: {"x": 1, "y": 0},
                2: {"x": 0.5, "y": 1},
                3: {"x": 0, "y": 0.5},
                4: {"x": 0.5, "y": -0.5},
                5: {"x": 1, "y": 0.5},
            },
        )
        G3.add_edges_from(
            [
                (0, 1),
                (1, 2),
                (2, 0),
                (3, 4),
                (4, 5),
                (5, 3),
            ]
        )
        self.G1 = G1
        self.G2 = G2
        self.G3 = G3

    def test_get_triangles_complete_graph(self):
        # Test Case 1: Complete Graph
        triangles, edges = metric_helpers._get_triangles(self.G1)
        self.assertEqual(
            triangles,
            {(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)},
        )
        self.assertEqual(
            edges,
            {
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 2),
                (1, 3),
                (2, 3),
            },
        )

    def test_get_triangles_cycle_graph(self):
        # Test Case 2: Cycle Graph
        triangles, edges = metric_helpers._get_triangles(self.G2)
        self.assertEqual(triangles, set())
        self.assertEqual(edges, set())

    def test_get_triangles_six_pointed_star_graph(self):
        # Test Case 3: Six-pointed Star Graph
        triangles, edges = metric_helpers._get_triangles(self.G3)
        self.assertEqual(
            triangles,
            {(0, 1, 2), (3, 4, 5)},
        )
        self.assertEqual(
            edges,
            {
                (0, 1),
                (0, 2),
                (1, 2),
                (3, 4),
                (3, 5),
                (4, 5),
            },
        )

    def test_count_impossible_triangle_crossings_complete_graph(self):
        # Test Case 1: Complete Graph
        crossings = metric_helpers._count_impossible_triangle_crossings(self.G1)
        self.assertEqual(crossings, 6)

    def test_count_impossible_triangle_crossings_cycle_graph(self):
        # Test Case 2: Cycle Graph
        crossings = metric_helpers._count_impossible_triangle_crossings(self.G2)
        self.assertEqual(crossings, 0)

    def test_count_impossible_triangle_crossings_six_pointed_star_graph(self):
        # Test Case 3: Six-pointed Star Graph
        crossings = metric_helpers._count_impossible_triangle_crossings(self.G3)
        self.assertEqual(crossings, 9)

    def test_count_4_cycles_complete_graph(self):
        # Test Case 1: Complete Graph
        cycles = metric_helpers._count_4_cycles(self.G1)
        self.assertEqual(cycles, 3)

    def test_count_4_cycles_cycle_graph(self):
        # Test Case 2: Cycle Graph
        cycles = metric_helpers._count_4_cycles(self.G2)
        self.assertEqual(cycles, 0)

    def test_count_4_cycles_six_pointed_star_graph(self):
        # Test Case 3: Six-pointed Star Graph
        cycles = metric_helpers._count_4_cycles(self.G3)
        self.assertEqual(cycles, 0)

    def test_calculate_edge_crossings_complete_graph(self):
        # Test Case 1: Complete Graph
        crossings, angles = metric_helpers._calculate_edge_crossings(self.G1)
        self.assertEqual(crossings, set({((0, 2), (1, 3))}))
        self.assertEqual(angles, {((0, 2), (1, 3)): 90})
        # Check that edge crossings were saved as attributes
        expected_crossings = defaultdict(
            lambda: {"count": 0, "angles": []}
        )  # Fix: Pass a callable lambda function as the first argument
        expected_crossings[(0, 2)] = {"count": 1, "angles": [90]}
        expected_crossings[(1, 3)] = {"count": 1, "angles": [90]}
        crossings = nx.get_edge_attributes(self.G1, "edge_crossings")
        for key, value in crossings.items():
            self.assertEqual(value, expected_crossings[key])

    def test_calculate_edge_crossings_cycle_graph(self):
        # Test Case 2: Cycle Graph
        crossings, angles = metric_helpers._calculate_edge_crossings(self.G2)
        self.assertEqual(crossings, set())
        self.assertEqual(angles, dict())
        # Check that edge crossings were saved as attributes
        crossings = nx.get_edge_attributes(self.G2, "edge_crossings")
        for k, v in crossings.items():
            self.assertEqual(v, {"count": 0, "angles": []})

    def test_calculate_edge_crossings_six_pointed_star_graph(self):
        # Test Case 3: Six-pointed Star Graph
        crossings, angles = metric_helpers._calculate_edge_crossings(self.G3)
        expected_crossings = set(
            {
                ((0, 1), (3, 4)),
                ((0, 1), (4, 5)),
                ((0, 2), (3, 4)),
                ((0, 2), (3, 5)),
                ((1, 2), (3, 5)),
                ((1, 2), (4, 5)),
            }
        )
        self.assertEqual(crossings, expected_crossings)
        expected_angles = {
            ((0, 1), (3, 4)): 63.43494882292201,
            ((0, 1), (4, 5)): 63.43494882292201,
            ((0, 2), (3, 4)): 126.86989764584402,
            ((0, 2), (3, 5)): 63.43494882292201,
            ((1, 2), (3, 5)): 116.56505117707799,
            ((1, 2), (4, 5)): 53.13010235415599,
        }
        for key, value in angles.items():
            self.assertAlmostEqual(value, expected_angles[key])
        # Check that edge crossings were saved as attributes
        expected_crossings = defaultdict(lambda: {"count": 0, "angles": []})
        for crossing, angle in expected_angles.items():
            for e1, e2 in crossing:
                expected_crossings[(e1, e2)]["count"] += 1
                expected_crossings[(e1, e2)]["angles"].append(angle)


if __name__ == "__main__":
    unittest.main()
