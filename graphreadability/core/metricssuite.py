import math
import time
from typing import Optional, Union
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
        mcdat: str = "weighted_sum",
        sym_threshold: Union[int, float] = 2,
        sym_tolerance: Union[int, float] = 3,
        file_type: str = "GraphML",
    ):
        # Dictionary mapping metric combination strategies to their functions
        self.mcdat_dict = {
            "weighted_sum": self._weighted_sum,
            "weighted_prod": self._weighted_prod,
        }
        # Placeholder for version of graph with crosses promoted to nodes
        self.graph_cross_promoted = None
        # Dictionary mapping metric names to their functions, values, and weights
        self.metrics = {
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
        }
        for key, value in self.metrics.items():
            value["value"] = None
            value["weight"] = 0

        # Check all metrics given are valid and assign weights
        if metric_weights:
            self.initial_weights = self.set_weights(metric_weights)
        else:
            self.initial_weights = {"edge_crossing": 1}

        # Check metric combination strategy is valid
        assert (
            mcdat in self.mcdat_dict
        ), f"Unknown mcdat: {mcdat}. Available mcats: {list(self.mcdat_dict.keys())}"
        self.mcdat = mcdat

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

        if type(sym_threshold) != int and type(sym_threshold) != float:
            raise TypeError(
                f"sym_threshold must be a number, not {type(sym_threshold)}"
            )

        if sym_threshold < 0:
            raise ValueError(f"sym_threshold must be positive.")

        self.sym_threshold = sym_threshold

    def set_weights(self, metric_weights):
        metrics_to_remove = []
        initial_weights = {}
        for metric in metric_weights:
            # Check metric is valid
            assert (
                metric in self.metrics
            ), f"Unknown metric: {metric}. Available metrics: {list(self.metrics.keys())}"
            # Check weight is a number
            if (
                type(metric_weights[metric]) != int
                and type(metric_weights[metric]) != float
            ):
                raise TypeError(
                    f"Metric weights must be a number, not {type(metric_weights[metric])}"
                )
            # Check weight is positive
            if metric_weights[metric] < 0:
                raise ValueError(f"Metric weights must be positive.")

            # Remove metrics with 0 weight
            if metric_weights[metric] == 0:
                metrics_to_remove.append(metric)
            else:
                # Assign weight to metric
                self.metrics[metric]["weight"] = metric_weights[metric]
                initial_weights[metric] = metric_weights[metric]

        # Remove 0 weighted metrics
        for metric in metrics_to_remove:
            metric_weights.pop(metric)

        return initial_weights

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

    def calculate_metric(self, metric):
        """Calculate the value of the given metric by calling the associated function."""
        self.metrics[metric]["value"] = self.metrics[metric]["func"](self.graph)

    def calculate_metrics(self):
        """Calculates the values of all metrics with non-zero weights."""
        t1 = time.time()
        for metric in self.metrics:
            if self.metrics[metric]["weight"] != 0:
                self.calculate_metric(metric)
        t2 = time.time()
        print(f"Took: {t2-t1}")

    def combine_metrics(self):
        """Combine several metrics based on the given multiple criteria descision analysis technique."""
        # Important to loop over initial weights to avoid checking the weight of all metrics when they are not needed
        for metric in self.initial_weights:
            self.calculate_metric(metric)

        return self.mcdat_dict[self.mcdat]()

    def pretty_print_metrics(self):
        """Prints all metrics and their values in an easily digestible view."""
        self.calculate_metrics()
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
        print(f"Evaluation using {self.mcdat}: {self.combine_metrics():.5f}")
        print("-" * 40)
