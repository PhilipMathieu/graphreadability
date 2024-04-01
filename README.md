# `graphreadability`

Python module for applying readability metrics to network and graph visualizations. This project is a work in progress that is being developed by Philip Mathieu (MS DS Student) as part of the Research Apprenticeship program at Northeastern University's Khoury College of Computer Science.

## Usage

```python
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
M.calculate_metrics()
M.pretty_print_metrics()
```

## Utilities

### Graph Digitizer

This utility is a python package using [`matplotlib`](https://matplotlib.org/) to show and image and allowing the user to click to add nodes, right click to delete nodes, and click two nodes sequentially to add edges.

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

Code in [`graphreadability/metrics/`](graphreadability/metrics/) is in part derived from code originally published at https://github.com/gavjmooney/graph_metrics/ associated with the following publication:
```
@Conference{citekey,
  author       = "Gavin J. Mooney, Helen C. Purchase, Michael Wybrow, Stephen G. Kobourov",
  title        = "The Multi-Dimensional Landscape of Graph Drawing Metrics",
  booktitle    = "2024 IEEE 17th Pacific Visualization Symposium (PacificVis)",
  year         = "2024",
}
```

## License

Apache 2.0 - see [LICENSE.txt](./LICENSE.txt)
