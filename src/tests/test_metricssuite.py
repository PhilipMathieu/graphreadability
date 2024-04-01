import io
import unittest
import networkx as nx
from src.core.metricssuite import MetricsSuite


class TestMetricsSuite(unittest.TestCase):
    def setUp(self):
        self.graph = nx.Graph()
        self.graph.add_nodes_from([1, 2, 3])
        self.graph.add_edges_from([(1, 2), (2, 3)])

    def test_set_weights(self):
        metrics_suite = MetricsSuite(graph=self.graph)
        metric_weights = {
            "edge_crossing": 1,
            "edge_orthogonality": 0,
            "node_orthogonality": 2,
        }
        expected_weights = {"edge_crossing": 1, "node_orthogonality": 2}
        weights = metrics_suite.set_weights(metric_weights)
        self.assertEqual(weights, expected_weights)

    def test_weighted_prod(self):
        metrics_suite = MetricsSuite(graph=self.graph)
        metrics_suite.metrics["edge_crossing"]["value"] = 10
        metrics_suite.metrics["node_orthogonality"]["value"] = 0.5
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_result = 4.0  # 10 * 0.5 * 0.8
        result = metrics_suite._weighted_prod()
        self.assertEqual(result, expected_result)

    def test_weighted_sum(self):
        metrics_suite = MetricsSuite(graph=self.graph)
        metrics_suite.metrics["edge_crossing"]["value"] = 10
        metrics_suite.metrics["node_orthogonality"]["value"] = 0.5
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_result = 5.0  # (10 * 1 + 0.5 * 1 + 0.8 * 0) / (1 + 1 + 0)
        result = metrics_suite._weighted_sum()
        self.assertEqual(result, expected_result)

    def test_calculate_metric(self):
        metrics_suite = MetricsSuite(graph=self.graph)
        metrics_suite.calculate_metric("edge_crossing")
        self.assertIsNotNone(metrics_suite.metrics["edge_crossing"]["value"])

    def test_calculate_metrics(self):
        metrics_suite = MetricsSuite(graph=self.graph)
        metrics_suite.calculate_metrics()
        for metric in metrics_suite.metrics:
            if metrics_suite.metrics[metric]["weight"] != 0:
                self.assertIsNotNone(metrics_suite.metrics[metric]["value"])

    def test_combine_metrics(self):
        metrics_suite = MetricsSuite(graph=self.graph)
        metrics_suite.metrics["edge_crossing"]["value"] = 10
        metrics_suite.metrics["node_orthogonality"]["value"] = 0.5
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_result = 5.0  # (10 * 1 + 0.5 * 1 + 0.8 * 0) / (1 + 1 + 0)
        result = metrics_suite.combine_metrics()
        self.assertEqual(result, expected_result)

    def test_pretty_print_metrics(self):
        metrics_suite = MetricsSuite(graph=self.graph)
        metrics_suite.metrics["edge_crossing"]["value"] = 10
        metrics_suite.metrics["node_orthogonality"]["value"] = 0.5
        metrics_suite.metrics["angular_resolution"]["value"] = 0.8
        expected_output = "----------------------------------------\nMetric              Value\tWeight\n----------------------------------------\nedge_crossing       10.000\t1\nnode_orthogonality  0.500\t0\nangular_resolution  0.800\t0\n----------------------------------------\nEvaluation using weighted_sum: 5.00000\n----------------------------------------\n"
        with unittest.mock.patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
            metrics_suite.pretty_print_metrics()
            self.assertEqual(mock_stdout.getvalue(), expected_output)

    def test_load_graph_test(self):
        metrics_suite = MetricsSuite()
        graph = metrics_suite.load_graph_test()
        self.assertIsInstance(graph, nx.Graph)


if __name__ == "__main__":
    unittest.main()
