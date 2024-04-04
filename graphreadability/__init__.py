__version__ = "0.0.1"
__author__ = "Philip Mathieu"
__license__ = "Apache-2.0"
__description__ = "A Python package for calculating readability metrics for graph and network visualizations."

# Import core modules
from .core.metricssuite import MetricsSuite
from .core.readabilitygraph import ReadabilityGraph

# Import layout algorithms
from .layout.layout_algorithms import naive_optimizer

# Import utils
from .utils import helpers
from .utils import crosses_promotion

# Import tests
from . import tests

__all__ = [
    "MetricsSuite",
    "ReadabilityGraph",
    "naive_optimizer",
    "helpers",
    "crosses_promotion",
    "tests",
]
