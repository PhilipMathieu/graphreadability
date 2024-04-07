# graphreadability

Python module for applying readability metrics to network and graph visualizations.

## Credits
Created by Philip Englund Mathieu, MS DS '24, as part of Northeastern University's Khoury College of Computer Sciences Research Apprenticeship program. Advised by Prof. Cody Dunne. For additional attributions, see [the dedicated references page](references.md).

## Installation

```
pip install graphreadability
```

## Usage

```python
# Suggested import syntax
import networkx as nx
import graphreadability as gr

# Create a basic graph using NetworkX
G = nx.Graph()
G.add_nodes_from(
    [
        (1, {"x": 1, "y": 1}),
        (2, {"x": -1, "y": 1}),
        (3, {"x": -1, "y": -1}),
        (4, {"x": 1, "y": -1}),
        (5, {"x": 2, "y": 1}),
    ]
)
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 2), (1, 3)])

# Create a MetricsSuite to calculate readability metrics
M = gr.MetricsSuite(G)

# Calculate readability metrics
M.calculate_metrics()

# Print results
M.pretty_print_metrics()
```

Expected Output:

```
--------------------------------------------------
Metric                        Value     Weight
--------------------------------------------------
angular_resolution            0.312     1
aspect_ratio                  0.667     1
crossing_angle                1.000     1
edge_crossing                 1.000     1
edge_length                   0.829     1
edge_orthogonality            0.600     1
gabriel_ratio                 0.556     1
neighbourhood_preservation    0.333     1
node_orthogonality            0.417     1
node_resolution               0.277     1
node_uniformity               0.812     1
--------------------------------------------------
Evaluation using weighted_sum: 0.61855
--------------------------------------------------
```
