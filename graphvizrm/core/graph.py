class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.layout = {}

    def add_node(self, node_id):
        self.nodes[node_id] = {}

    def add_edge(self, source, target):
        if source not in self.nodes or target not in self.nodes:
            raise ValueError("Source or target node does not exist")
        self.edges[(source, target)] = {}

    def remove_node(self, node_id):
        if node_id not in self.nodes:
            raise ValueError("Node does not exist")
        for edge in list(self.edges):
            if node_id in edge:
                del self.edges[edge]
        del self.nodes[node_id]

    def remove_edge(self, source, target):
        if (source, target) not in self.edges:
            raise ValueError("Edge does not exist")
        del self.edges[(source, target)]

    def set_node_position(self, node_id, x, y):
        if node_id not in self.nodes:
            raise ValueError("Node does not exist")
        self.layout[node_id] = {"x": x, "y": y}

    def set_node_size(self, node_id, width, height):
        if node_id not in self.nodes:
            raise ValueError("Node does not exist")
        self.nodes[node_id]["width"] = width
        self.nodes[node_id]["height"] = height

    def is_valid(self):
        for node_id in self.nodes:
            if node_id not in self.layout:
                return False
        return True

    # Add other helper functions here if needed
