import math
from typing import Optional, Union, Sequence
import networkx as nx
from ..metrics import metrics

# Generate the list of metric functions
METRIC_FUNCS = [
    func
    for func in dir(metrics)
    if callable(getattr(metrics, func)) and not func.startswith("_")
]

# Set default weights for all metrics
DEFAULT_WEIGHTS = {func: 1 for func in METRIC_FUNCS}

# Generate the dictionary of metric functions
METRICS = {func: {"func": getattr(metrics, func)} for func in METRIC_FUNCS}


class MetricsSuite:
    """
    A suite for calculating several metrics for graph drawing aesthetics, as well as methods for
    combining these into a single cost function.

    Parameters
    ----------
    graph : Union[nx.Graph, str], optional
        The graph to be analyzed. Can be a NetworkX Graph object or a path to a GML or GraphML file.
    metric_weights : Optional[dict], optional
        Dictionary of metric:weight key/values. Default is DEFAULT_WEIGHTS.
    metric_combination_strategy : str, optional
        The multiple criteria decision analysis technique to use for combining metrics. Default is "weighted_sum".
    sym_threshold : Union[int, float], optional
        The threshold for symmetry detection. Default is 2.
    sym_tolerance : Union[int, float], optional
        The tolerance for symmetry detection. Default is 3.
    file_type : str, optional
        The file type of the graph file. Default is "GraphML".
    """

    def __init__(
        self,
        G: Union[nx.Graph, str] = None,
        metric_weights: Optional[dict] = DEFAULT_WEIGHTS,
        metric_combination_strategy: str = "weighted_sum",
        sym_threshold: Union[int, float] = 2,
        sym_tolerance: Union[int, float] = 3,
        file_type: str = "GraphML",
        copy=False,
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
            self.metrics[k].update({"weight": None, "value": None})

        # Check all metrics given are valid and assign weights
        self.initial_weights = self.set_weights(metric_weights)

        # Check metric combination strategy is valid
        assert (
            metric_combination_strategy in self.metric_combination_strategies
        ), f"Unknown metric combination strategy: {metric_combination_strategy}. Available strategies: {list(self.metric_combination_strategies.keys())}"
        self.metric_combination_strategy = metric_combination_strategy

        if G is None:
            self._filename = ""
            self.G = self.load_graph_test()
        elif isinstance(G, str):
            self._filename = G
            self.G = self.load_graph(G, file_type=file_type)
        elif isinstance(G, nx.Graph):
            self._filename = ""
            if copy:
                self.G = G.copy()
            else:
                self.G = G
        else:
            raise TypeError(
                f"'graph' must be a string representing a path to a GML or GraphML file, or a NetworkX Graph object, not {type(G)}"
            )

        if sym_tolerance < 0:
            raise ValueError("sym_tolerance must be positive.")

        self.sym_tolerance = sym_tolerance

        if sym_threshold < 0:
            raise ValueError("sym_threshold must be positive.")

        self.sym_threshold = sym_threshold

    def __repr__(self):
        """Return a detailed string representation of the MetricsSuite object."""
        return (
            f"MetricsSuite(graph={self._filename}, metric_weights={self.initial_weights}, "
            f"metric_combination_strategy={self.metric_combination_strategy}, sym_threshold={self.sym_threshold}, "
            "sym_tolerance={self.sym_tolerance})"
        )

    def __str__(self):
        """Return a concise string representation of the MetricsSuite object."""
        return (
            f"MetricsSuite({self._filename}) object with {len(self.metrics)} metrics."
        )

    def __copy__(self):
        """Return a shallow copy of the MetricsSuite object."""
        return MetricsSuite(
            G=self.G,
            metric_weights=self.initial_weights,
            metric_combination_strategy=self.metric_combination_strategy,
            sym_threshold=self.sym_threshold,
            sym_tolerance=self.sym_tolerance,
        )

    def __deepcopy__(self, memo):
        """Return a deep copy of the MetricsSuite object."""
        return MetricsSuite(
            G=self.G.copy(),
            metric_weights=self.initial_weights,
            metric_combination_strategy=self.metric_combination_strategy,
            sym_threshold=self.sym_threshold,
            sym_tolerance=self.sym_tolerance,
        )

    def load_graph_test(self, nxg=nx.sedgewick_maze_graph):
        """Loads a test graph with a random layout."""
        G = nxg()
        pos = nx.random_layout(G)
        for k, v in pos.items():
            pos[k] = {"x": v[0], "y": v[1]}

        nx.set_node_attributes(G, pos)
        return G

    def copy(self, deep=True, memo=None):
        """Return a copy of the MetricsSuite object, defaulting to a deep copy."""
        if deep is True or memo is not None:
            return self.__deepcopy__(memo)
        else:
            return self.__copy__()

    def set_weights(self, metric_weights: Sequence[float]):
        """Set the weights of the metrics in the MetricsSuite object.

        Parameters
        ----------
        metric_weights : dict
            Dictionary of metric:weight key/values.

        Returns
        -------
        dict
            Dictionary of metric:weight key/values for metrics with non-zero weights.
        """
        metric_weights = {
            metric: weight for metric, weight in metric_weights.items() if weight > 0
        }

        if any(metric_weights[metric] < 0 for metric in metric_weights):
            raise ValueError("Metric weights must be positive.")

        for metric in self.metrics:
            self.metrics[metric]["weight"] = (
                metric_weights[metric] if metric in metric_weights else None
            )

        return {
            metric: weight for metric, weight in metric_weights.items() if weight > 0
        }

    def apply_layout(self, pos):
        """Applies the given layout to the graph.

        Parameters
        ----------
        pos : dict(node_id, tuple(float, float))
            Dictionary of node positions.

        Returns
        -------
        None
        """
        # Convert to x and y attributes
        xy = {k: {"x": v[0], "y": v[1]} for k, v in pos.items()}
        nx.set_node_attributes(self.G, xy)

    def calculate_metric(self, metric: str = None):
        """Calculate the value of the given metric by calling the associated function."""
        if metric is None:
            raise ValueError(
                "No metric provided. Did you mean to call calculate_metrics()?"
            )

        try:
            self.metrics[metric]["value"] = self.metrics[metric]["func"](self.G)
        except Exception as e:
            print(f"Error calculating metric {metric}: {e}")
            self.metrics[metric]["value"] = None

    def calculate_metrics(self, calculate_all=False):
        """Calculates the values of all metrics with non-zero weights."""
        n_metrics = 0
        for metric in self.metrics:
            if self.metrics[metric]["weight"] is not None or calculate_all:
                self.calculate_metric(metric)
                n_metrics += 1

    def reset_metrics(self):
        """Resets all metric values and is_calculated flags to None and False, respectively."""
        for metric in self.metrics:
            self.metrics[metric]["value"] = None
            self.metrics[metric]["is_calculated"] = False

    def weighted_prod(self):
        """Returns the weighted product of all metrics. Should NOT be used as a cost function - may be useful for comparing graphs."""
        used_metrics = [
            metric
            for metric in self.metrics.keys()
            if self.metrics[metric]["weight"] is not None
        ]
        for metric in used_metrics:
            if self.metrics[metric]["value"] is None:
                self.calculate_metric(metric)
        return math.prod(
            self.metrics[metric]["value"] * self.metrics[metric]["weight"]
            for metric in used_metrics
        )

    def weighted_sum(self):
        """Returns the weighted sum of all metrics. Can be used as a cost function."""
        used_metrics = [
            metric
            for metric in self.metrics.keys()
            if self.metrics[metric]["weight"] is not None
        ]
        total_weight = sum(self.metrics[metric]["weight"] for metric in used_metrics)
        return (
            sum(
                self.metrics[metric]["value"] * self.metrics[metric]["weight"]
                for metric in used_metrics
            )
            / total_weight
        )

    def combine_metrics(self, calculate=True):
        """Combine several metrics based on the given multiple criteria decision analysis technique."""
        # Important to loop over initial weights to avoid checking the weight of all metrics when they are not needed
        if calculate:
            self.calculate_metrics()
        return self.metric_combination_strategies[self.metric_combination_strategy]()

    def pretty_print_metrics(self, calculate=False):
        """Prints all metrics and their values in an easily digestible view."""
        combined = self.combine_metrics(calculate=calculate)
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

    def metric_table(self, calculate=False):
        """Returns a dictionary of metrics and their values. Designed to work with pandas from_records() method."""
        combined = self.combine_metrics(calculate)
        metrics = {}
        for k, v in self.metrics.items():
            metrics[k] = v["value"]
        metrics["combined"] = combined
        return metrics
