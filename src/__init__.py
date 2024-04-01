__version__ = "0.0.1"
__author__ = "Philip Mathieu"
__license__ = "Apache-2.0"
__description__ = "A Python package for calculating readability metrics for graph and network visualizations."

# Import core modules
from .core.metricssuite import MetricsSuite
from .core.readabilitygraph import ReadabilityGraph

import utils.helpers as helpers
import utils.crosses_promotion as crosses_promotion

# Import tests
import tests

__all__ = ["MetricsSuite", "ReadabilityGraph", "helpers", "crosses_promotion", "tests"]
