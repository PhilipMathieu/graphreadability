__version__ = "0.0.1"
__author__ = "Philip Mathieu"
__license__ = "Apache-2.0"
__description__ = "A Python package for calculating readability metrics for graph and network visualizations."

# Import core modules
from .core.metricssuite import MetricsSuite

# Import layout algorithms
from .layout import layout_algorithms

# Import utils
from .utils import helpers
from .utils import crosses_promotion

# Import tests
from . import tests

__all__ = [
    "MetricsSuite",
    "ReadabilityGraph",
    "layout_algorithms",
    "helpers",
    "crosses_promotion",
    "tests",
]
