"""
This module extends the networkx Graph class to include additional functionality.
"""

import networkx as nx


# Extend networkx Graph to include layout_positions
def set_layout_positions(self, positions):
    """
    Assigns layout positions to nodes.
    :param positions: dict, A dictionary with node keys and position values (tuples).
    """
    self.layout_positions = positions


nx.Graph.set_layout_positions = set_layout_positions


# Extend networkx Graph to include a property for Cartesian grid
@property
def is_cartesian_grid(self):
    """
    Checks if the graph is intended to be on a Cartesian grid.
    """
    return getattr(self, "_is_cartesian_grid", False)


@is_cartesian_grid.setter
def is_cartesian_grid(self, value):
    setattr(self, "_is_cartesian_grid", value)


nx.Graph.is_cartesian_grid = is_cartesian_grid


# Extend networkx Graph to include custom metadata
def add_metadata(self, **metadata):
    """
    Adds custom metadata to the graph.
    :param metadata: key-value pairs to be added to the graph's metadata.
    """
    if not hasattr(self, "_metadata"):
        self._metadata = {}
    self._metadata.update(metadata)


def get_metadata(self, key, default=None):
    """
    Retrieves a metadata value by key.
    :param key: The key of the metadata to retrieve.
    :param default: The default value to return if the key does not exist.
    """
    return getattr(self, "_metadata", {}).get(key, default)


nx.Graph.add_metadata = add_metadata
nx.Graph.get_metadata = get_metadata

# Example usage
if __name__ == "__main__":
    G = nx.Graph()
    G.set_layout_positions({1: (0, 0), 2: (1, 1)})
    G.is_cartesian_grid = True
    G.add_metadata(description="Example graph", year=2024)

    print(G.layout_positions)
    print(G.is_cartesian_grid)
    print(G.get_metadata("description"))
