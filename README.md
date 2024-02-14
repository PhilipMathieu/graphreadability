# `graphreadability`

Python module for applying readability metrics to network and graph visualizations. This project is a work in progress that is being developed by Philip Mathieu (MS DS Student) as part of the Research Apprenticeship program at Northeastern University's Khoury College of Computer Science.

## Usage

The basic functionality of this module is to extend NetworkX to introduce the ability to calculate readability metrics. For example, to calculate edge crossings on a simple graph consisting of two pairs of nodes with an "x" of edges:

```python
from graphreadability import ReadabilityGraph

# Create a graph object which extends networkx.Graph
G = ReadabilityGraph()

# Create a square of nodes
graph.add_node("A", pos=(0, 0))
graph.add_node("B", pos=(1, 1))
graph.add_node("C", pos=(1, 0))
graph.add_node("D", pos=(0, 1))

# Add diagonal edges
graph.add_edge("A", "C")
graph.add_edge("B", "D")

# Calculate the crossings metric
crossings = graph.edge_crossings_global()
```

## Utilities

### Graph Digitizer

This utility is a python package using MatPlotLib to show and image and allowing the user to click to add nodes, right click to delete nodes, and click two nodes sequentially to add edges.

```sh
python graphreadability/utils/digitize_graphs.py -h
usage: digitize_graphs.py [-h] [-i IMAGE] [-o OUTPUT]

Create a graph from an image.

options:
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        Path to the image to create a graph from.
  -o OUTPUT, --output OUTPUT
                        Path to save the graph to.
```

## Sources Cited

Metric definitions are derived from:
- C. Dunne, S. I. Ross, B. Shneiderman, and M. Martino. "Readability metric feedback for aiding node-link visualization designers," IBM Journal of Research and Development, 59(2/3) pages 14:1--14:16, 2015.

Initial inspiration was taken from [rpgove/greadability.js](https://github.com/rpgove/greadability/).

## License

All rights reserved for now (likely to be open sourced shortly).
