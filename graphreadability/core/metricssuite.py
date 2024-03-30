import math
import time
from typing import Optional, Union, List
from collections import defaultdict
import networkx as nx

from ..metrics.metrics import *

class MetricsSuite:
    """A suite for calculating several metrics for graph drawing aesthetics, as well as methods for combining these into a single cost function.
    Takes as an argument a path to a GML or GraphML file, or a NetworkX Graph object. Also takes as an argument a dictionary of metric:weight key/values.
    Note: to prevent unnecessary calculations, omit metrics with weight 0."""

    def __init__(
        self,
        graph: Union[nx.Graph, str] = None,
        metric_weights: Optional[dict] = None,
        metric_combination_strategy: str = "weighted_sum",
        sym_threshold: Union[int, float] = 2,
        sym_tolerance: Union[int, float] = 3,
        file_type: str = "GraphML",
    ):
        # Dictionary mapping metric combination strategies to their functions
        self.metric_combination_strategies = {
            "weighted_sum": self._weighted_sum,
            "weighted_prod": self._weighted_prod,
        }
        # Placeholder for version of graph with crosses promoted to nodes
        self.graph_cross_promoted = None
        # Dictionary mapping metric names to their functions, values, and weights
        self.metrics = defaultdict(lambda: {"func": None, "value": None, "weight": 0}, {
            "edge_crossing": {"func": edge_crossing, "num_crossings": None},
            "edge_orthogonality": {"func": edge_orthogonality},
            "node_orthogonality": {"func": node_orthogonality},
            "angular_resolution": {"func": angular_resolution},
            "symmetry": {"func": symmetry},
            "node_resolution": {"func": node_resolution},
            "edge_length": {"func": edge_length},
            "gabriel_ratio": {"func": gabriel_ratio},
            "crossing_angle": {"func": crossing_angle},
            "stress": {"func": get_stress},
            "neighbourhood_preservation": {"func": neighbourhood_preservation},
            "aspect_ratio": {"func": aspect_ratio},
            "node_uniformity": {"func": node_uniformity},
        })

        # Check all metrics given are valid and assign weights
        if metric_weights:
            self.initial_weights = self.set_weights(metric_weights)
        else:
            self.initial_weights = {"edge_crossing": 1}

        # Check metric combination strategy is valid
        assert (
            metric_combination_strategy in self.metric_combination_strategies
        ), f"Unknown metric combination strategy: {metric_combination_strategy}. Available strategies: {list(self.metric_combination_strategies.keys())}"
        self.metric_combination_strategy = metric_combination_strategy

        if graph is None:
            self.graph = self.load_graph_test()
        elif isinstance(graph, str):
            self.fname = graph
            self.graph = self.load_graph(graph, file_type=file_type)
        elif isinstance(graph, nx.Graph):
            self.fname = ""
            self.graph = graph
        else:
            raise TypeError(
                f"'graph' must be a string representing a path to a GML or GraphML file, or a NetworkX Graph object, not {type(graph)}"
            )

        if sym_tolerance < 0:
            raise ValueError(f"sym_tolerance must be positive.")

        self.sym_tolerance = sym_tolerance

        if sym_threshold < 0:
            raise ValueError(f"sym_threshold must be positive.")

        self.sym_threshold = sym_threshold

    def set_weights(self, metric_weights: List[int, float]):
        metrics_to_remove = [metric for metric, weight in metric_weights.items() if weight <= 0]

        if any(metric_weights[metric] < 0 for metric in metric_weights):
            raise ValueError("Metric weights must be positive.")

        for metric in metrics_to_remove:
            metric_weights.pop(metric)

        self.metrics.update({metric: {"weight": weight} for metric, weight in metric_weights.items() if weight > 0})

        return {metric: weight for metric, weight in metric_weights.items() if weight > 0}
    
    def _weighted_prod(self):
        """Returns the weighted product of all metrics. Should NOT be used as a cost function - may be useful for comparing graphs."""
        return math.prod(
            self.metrics[metric]["value"] * self.metrics[metric]["weight"]
            for metric in self.initial_weights
        )

    def _weighted_sum(self):
        """Returns the weighted sum of all metrics. Can be used as a cost function."""
        total_weight = sum(self.metrics[metric]["weight"] for metric in self.metrics)
        return (
            sum(
                self.metrics[metric]["value"] * self.metrics[metric]["weight"]
                for metric in self.initial_weights
            )
            / total_weight
        )

    def load_graph_test(self, nxg=nx.sedgewick_maze_graph):
        """Loads a test graph with a random layout."""
        G = nxg()
        pos = nx.random_layout(G)
        for k, v in pos.items():
            pos[k] = {"x": v[0], "y": v[1]}

        nx.set_node_attributes(G, pos)
        return G
    
    def calculate_metric(self, metric: str):
        """Calculate the value of the given metric by calling the associated function."""
        self.metrics[metric]["value"] = self.metrics[metric]["func"](self.graph)

    def calculate_metrics(self):
        """Calculates the values of all metrics with non-zero weights."""
        start_time = time.perf_counter()
        for metric in self.metrics:
            if self.metrics[metric]["weight"] != 0:
                self.calculate_metric(metric)
        end_time = time.perf_counter()
        print(f"Metrics calculation took: {end_time - start_time}")

    def combine_metrics(self):
        """Combine several metrics based on the given multiple criteria decision analysis technique."""
        # Important to loop over initial weights to avoid checking the weight of all metrics when they are not needed
        [self.calculate_metric(metric) for metric in self.initial_weights]
        return self.metric_combination_strategies[self.metric_combination_strategy]()

    def pretty_print_metrics(self):
        """Prints all metrics and their values in an easily digestible view."""
        print("-" * 40)
        print("{:<20s}Value\tWeight".format("Metric"))
        print("-" * 40)
        for k, v in self.metrics.items():
            if v["value"]:
                val_str = f"{v['value']:.3f}"
                print(f"{k:<20s}{val_str:<5s}\t{v['weight']}")
            else:
                print(f"{k:<20s}{str(v['value']):<5s}\t{v['weight']}")
        print("-" * 40)
        print(f"Evaluation using {self.metric_combination_strategy}: {self.combine_metrics():.5f}")
        print("-" * 40)