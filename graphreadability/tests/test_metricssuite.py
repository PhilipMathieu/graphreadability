"""
Tests in this module are designed to test the MetricsSuite class in src/core/metricssuite.py. The MetricsSuite class
is a wrapper class that allows for the calculation of multiple metrics on a graph, and the combination of these metrics
into a single value. The MetricsSuite class is designed to be used in the context of graph layout optimization, where
multiple metrics are calculated on a graph, and then combined into a single value that can be used to evaluate the
quality of the layout.

This module tests all methods of the MetricsSuite class. It uses the following approach:
1. Create a test graph with known properties.
2. Test the MetricsSuite class with the test graph.

The tests manually set the values of the metrics in the MetricsSuite object. This allows for testing the combination
of metrics without having to rely on the actual calculation of the metrics, which are implemented elsewhere.
"""

import os
import sys
from io import StringIO
import unittest
import networkx as nx
from src.core.metricssuite import MetricsSuite


class TestMetricsSuite(unittest.TestCase):

    def graphs_equivalent_except_positions(self, G1, G2):
        # Verify that all node attributes except position (x and y) are the same
        for node in G1.nodes():
            attr_expected, attr_actual = (
                {k: v for k, v in G1.nodes[node].items() if k not in ["x", "y"]},
                {k: v for k, v in G2.nodes[node].items() if k not in ["x", "y"]},
            )
            if attr_expected != attr_actual:
                return False
        # Verify that all edges are the same
        return G1.edges() == G2.edges()

    def graphs_positions_equivalent(self, G1, G2):
        # Verify that all node attributes except position (x and y) are the same
        for node in G1.nodes():
            attr_expected, attr_actual = (
                {k: v for k, v in G1.nodes[node].items() if k in ["x", "y"]},
                {k: v for k, v in G2.nodes[node].items() if k in ["x", "y"]},
            )
            if attr_expected != attr_actual:
                return False
        return True

    def setUp(self) -> None:
        super().setUp()
        # Complete graph with 4 nodes arranged in a unit square
        self.G = nx.complete_graph(4)
        nx.set_node_attributes(
            self.G,
            {
                0: {"x": 0, "y": 0},
                1: {"x": 1, "y": 0},
                2: {"x": 1, "y": 1},
                3: {"x": 0, "y": 1},
            },
        )

    def test_load_graph_test_default(self):
        metrics_suite = MetricsSuite()
        G = nx.sedgewick_maze_graph()
        self.assertTrue(self.graphs_equivalent_except_positions(G, metrics_suite.G))

    def test_load_graph_test(self):
        metrics_suite = MetricsSuite(self.G)
        self.assertTrue(
            self.graphs_equivalent_except_positions(self.G, metrics_suite.G)
        )

    def test_copy(self):
        metrics_suite = MetricsSuite(G=self.G)
        # Ensure that metrics_suite.G is a reference to self.G
        self.G.add_node(4)
        self.assertTrue(4 in metrics_suite.G.nodes())
        # Create a copy of metrics_suite
        metrics_suite_deep = metrics_suite.copy()
        metrics_suite_shallow = metrics_suite.copy(deep=False)
        # Ensure that metrics_suite_copy.G is a deep copy of self.G
        self.G.add_node(5)
        self.assertFalse(5 in metrics_suite_deep.G.nodes())
        self.assertTrue(5 in metrics_suite_shallow.G.nodes())

    def test_set_weights(self):
        metrics_suite = MetricsSuite(G=self.G)
        metric_weights = {
            "edge_crossing": 1,
            "edge_orthogonality": 0,
            "crossing_angle": 2,
        }
        expected_weights = {"edge_crossing": 1, "crossing_angle": 2}
        weights = metrics_suite.set_weights(metric_weights)

        self.assertEqual(weights, expected_weights)

    def test_apply_layout(self):
        metrics_suite = MetricsSuite(G=self.G)
        pos = nx.spring_layout(self.G)
        metrics_suite.apply_layout(pos)
        self.assertTrue(self.graphs_positions_equivalent(self.G, metrics_suite.G))
        # Test again, this time using copy
        metrics_suite = MetricsSuite(G=self.G, copy=True)
        pos_spring = nx.spectral_layout(self.G)
        metrics_suite.apply_layout(pos_spring)
        self.assertFalse(self.graphs_positions_equivalent(self.G, metrics_suite.G))

    def test_calculate_metric(self):
        metrics_suite = MetricsSuite(G=self.G)
        # Check that it hasn't been calculated yet
        self.assertIsNone(metrics_suite.metrics["edge_crossing"]["value"])
        metrics_suite.calculate_metric("edge_crossing")
        self.assertIsNotNone(metrics_suite.metrics["edge_crossing"]["value"])

    def test_calculate_metrics(self):
        metrics_suite = MetricsSuite(G=self.G)
        # Check that no metrics have been calculated yet
        for metric in metrics_suite.metrics:
            self.assertIsNone(metrics_suite.metrics[metric]["value"])
        metrics_suite.calculate_metrics()
        # Check that all metrics have been calculated
        for metric in metrics_suite.metrics:
            self.assertIsNotNone(metrics_suite.metrics[metric]["value"])

    def test_calculate_metrics_some_none(self):
        metrics_suite = MetricsSuite(
            G=self.G, metric_weights={"edge_crossing": 0.5, "angular_resolution": 1}
        )
        # Check that no metrics have been calculated yet
        for metric in metrics_suite.metrics:
            self.assertIsNone(metrics_suite.metrics[metric]["value"])
        metrics_suite.calculate_metrics()
        # Check that only the specified metrics have been calculated
        for metric in metrics_suite.metrics:
            if metric in ["edge_crossing", "angular_resolution"]:
                self.assertIsNotNone(metrics_suite.metrics[metric]["value"])
            else:
                self.assertIsNone(metrics_suite.metrics[metric]["value"])
        # Check that the calculate_all flag works
        metrics_suite.calculate_metrics(calculate_all=True)
        for metric in metrics_suite.metrics:
            self.assertIsNotNone(metrics_suite.metrics[metric]["value"])

    def test_reset_metrics(self):
        metrics_suite = MetricsSuite(G=self.G)
        metrics_suite.calculate_metrics()
        for metric in metrics_suite.metrics:
            self.assertIsNotNone(metrics_suite.metrics[metric]["value"])
        metrics_suite.reset_metrics()
        for metric in metrics_suite.metrics:
            self.assertIsNone(metrics_suite.metrics[metric]["value"])

    def test_weighted_prod(self):
        metrics_suite = MetricsSuite(
            G=self.G, metric_weights={"edge_crossing": 0.5, "angular_resolution": 1}
        )
        metrics_suite.metrics["edge_crossing"]["value"] = 10
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_result = 4.0  # 10 * 0.5 * 0.8 * 1
        result = metrics_suite.weighted_prod()
        self.assertEqual(result, expected_result)

    def test_weighted_sum(self):
        metrics_suite = MetricsSuite(
            G=self.G, metric_weights={"edge_crossing": 0.5, "angular_resolution": 1}
        )
        metrics_suite.metrics["edge_crossing"]["value"] = 10
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_result = 5.8 / 1.5  # 10 * 0.5 + 0.8 * 1 / (0.5 + 1)
        result = metrics_suite.weighted_sum()
        self.assertEqual(result, expected_result)

    def test_combine_metrics_default(self):
        metrics_suite = MetricsSuite(
            G=self.G, metric_weights={"edge_crossing": 0.5, "angular_resolution": 1}
        )
        metrics_suite.metrics["edge_crossing"]["value"] = 10
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_result = 5.8 / 1.5  # 10 * 0.5 + 0.8 * 1
        result = metrics_suite.combine_metrics(calculate=False)
        self.assertEqual(result, expected_result)

    def test_combined_metrics_prod(self):
        metrics_suite = MetricsSuite(
            G=self.G,
            metric_weights={"edge_crossing": 0.5, "angular_resolution": 1},
            metric_combination_strategy="weighted_prod",
        )
        metrics_suite.metrics["edge_crossing"]["value"] = 10
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_result = 4  # 10 * 0.5 * 0.8 * 1
        result = metrics_suite.combine_metrics(calculate=False)
        self.assertEqual(result, expected_result)

    def test_pretty_print_metrics(self):
        metrics_suite = MetricsSuite(
            G=self.G, metric_weights={"edge_crossing": 0.5, "angular_resolution": 1}
        )
        metrics_suite.metrics["edge_crossing"]["value"] = 1
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        # Load expected output
        filepath = os.path.join(
            os.path.dirname(__file__), "test_data", "pretty_print_output.txt"
        )
        with open(filepath, "r") as f:
            expected_output = f.read()
        # Redirect stdout
        captured_output = StringIO()
        sys.stdout = captured_output
        metrics_suite.pretty_print_metrics()
        sys.stdout = sys.__stdout__
        self.maxDiff = None
        self.assertEqual(captured_output.getvalue(), expected_output)

    def test_metric_table(self):
        metrics_suite = MetricsSuite(
            G=self.G, metric_weights={"edge_crossing": 0.5, "angular_resolution": 1}
        )
        metrics_suite.metrics["edge_crossing"]["value"] = 1
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_output = {
            "edge_crossing": 1,
            "angular_resolution": 0.8,
            "combined": 0.8666666666666667,
        }
        result = metrics_suite.metric_table()
        for k, v in result.items():
            if k in expected_output:
                self.assertEqual(expected_output[k], v)
            else:
                self.assertIsNone(v)


if __name__ == "__main__":
    unittest.main()
