import math
import time
from typing import Optional, Union, Sequence
import networkx as nx
from ..metrics import metrics

# Get all the functions in the metrics module
_metric_functions = [
    func
    for func in dir(metrics)
    if callable(getattr(metrics, func)) and not func.startswith("__")
]

# Generate the DEFAULT_WEIGHTS dictionary
DEFAULT_WEIGHTS = {func: 1 for func in _metric_functions}

# Generate the METRICS dictionary
METRICS = {func: {"func": getattr(metrics, func)} for func in _metric_functions}


class MetricsSuite:
    """A suite for calculating several metrics for graph drawing aesthetics, as well as methods for combining these into a single cost function.
    Takes as an argument a path to a GML or GraphML file, or a NetworkX Graph object. Also takes as an argument a dictionary of metric:weight key/values.
    Note: to prevent unnecessary calculations, omit metrics with weight 0."""

    def __init__(
        self,
        graph: Union[nx.Graph, str] = None,
        metric_weights: Optional[dict] = DEFAULT_WEIGHTS,
        metric_combination_strategy: str = "weighted_sum",
        sym_threshold: Union[int, float] = 2,
        sym_tolerance: Union[int, float] = 3,
        file_type: str = "GraphML",
    ):
        # Dictionary mapping metric combination strategies to their functions
        self.metric_combination_strategies = {
            "weighted_sum": self.weighted_sum,
            "weighted_prod": self.weighted_prod,
        }
        # Placeholder for version of graph with crosses promoted to nodes
        self.graph_cross_promoted = None
        # Dictionary mapping metric names to their functions, values, and weights
        self.metrics = METRICS.copy()
        for k in self.metrics.keys():
            self.metrics[k].update({"weight": 0, "value": None, "is_calculated": False})

        # Check all metrics given are valid and assign weights
        self.initial_weights = self.set_weights(metric_weights)

        # Check metric combination strategy is valid
        assert (
            metric_combination_strategy in self.metric_combination_strategies
        ), f"Unknown metric combination strategy: {metric_combination_strategy}. Available strategies: {list(self.metric_combination_strategies.keys())}"
        self.metric_combination_strategy = metric_combination_strategy

        if graph is None:
            self._filename = ""
            self._graph = self.load_graph_test()
        elif isinstance(graph, str):
            self._filename = graph
            self._graph = self.load_graph(graph, file_type=file_type)
        elif isinstance(graph, nx.Graph):
            self._filename = ""
            self._graph = graph
        else:
            raise TypeError(
                f"'graph' must be a string representing a path to a GML or GraphML file, or a NetworkX Graph object, not {type(graph)}"
            )

        if sym_tolerance < 0:
            raise ValueError("sym_tolerance must be positive.")

        self.sym_tolerance = sym_tolerance

        if sym_threshold < 0:
            raise ValueError("sym_threshold must be positive.")

        self.sym_threshold = sym_threshold

    def set_weights(self, metric_weights: Sequence[float]):
        metrics_to_remove = [
            metric for metric, weight in metric_weights.items() if weight <= 0
        ]

        if any(metric_weights[metric] < 0 for metric in metric_weights):
            raise ValueError("Metric weights must be positive.")

        for metric in metrics_to_remove:
            metric_weights.pop(metric)

        for metric in metric_weights:
            self.metrics[metric]["weight"] = metric_weights[metric]

        return {
            metric: weight for metric, weight in metric_weights.items() if weight > 0
        }

    def weighted_prod(self):
        """Returns the weighted product of all metrics. Should NOT be used as a cost function - may be useful for comparing graphs."""
        return math.prod(
            self.metrics[metric]["value"] * self.metrics[metric]["weight"]
            for metric in self.initial_weights
        )

    def weighted_sum(self):
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

    def reset_metrics(self):
        for metric in self.metrics:
            self.metrics[metric]["value"] = None
            self.metrics[metric]["is_calculated"] = False

    def calculate_metric(self, metric: str = None):
        """Calculate the value of the given metric by calling the associated function."""
        if metric is None:
            raise ValueError(
                "No metric provided. Did you mean to call calculate_metrics()?"
            )

        if not self.metrics[metric]["is_calculated"]:
            self.metrics[metric]["value"] = self.metrics[metric]["func"](self._graph)
            self.metrics[metric]["is_calculated"] = True
        else:
            pass
            # print(f"Metric {metric} already calculated. Skipping.")

    def calculate_metrics(self):
        """Calculates the values of all metrics with non-zero weights."""
        start_time = time.perf_counter()
        n_metrics = 0
        for metric in self.metrics:
            if self.metrics[metric]["weight"] != 0:
                self.calculate_metric(metric)
                n_metrics += 1
        end_time = time.perf_counter()
        print(
            f"Calculated {n_metrics} metrics in {end_time - start_time:0.3f} seconds."
        )

    def combine_metrics(self):
        """Combine several metrics based on the given multiple criteria decision analysis technique."""
        # Important to loop over initial weights to avoid checking the weight of all metrics when they are not needed
        [self.calculate_metric(metric) for metric in self.initial_weights]
        return self.metric_combination_strategies[self.metric_combination_strategy]()

    def pretty_print_metrics(self):
        """Prints all metrics and their values in an easily digestible view."""
        combined = self.combine_metrics()
        print("-" * 50)
        print("{:<30s}Value\tWeight".format("Metric"))
        print("-" * 50)
        for k, v in self.metrics.items():
            if v["value"]:
                val_str = f"{v['value']:.3f}"
                print(f"{k:<30s}{val_str:<5s}\t{v['weight']}")
            else:
                print(f"{k:<30s}{str(v['value']):<5s}\t{v['weight']}")
        print("-" * 50)
        print(f"Evaluation using {self.metric_combination_strategy}: {combined:.5f}")
        print("-" * 50)

    def metric_table(self):
        """Returns a dictionary of metrics and their values. Designed to work with pandas from_records() method."""
        combined = self.combine_metrics()
        metrics = {}
        for k, v in self.metrics.items():
            metrics[k] = v["value"]
        metrics["Combined"] = combined
        return metrics
