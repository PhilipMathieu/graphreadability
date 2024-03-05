import matplotlib.pyplot as plt
import networkx as nx

from utils.helpers import draw_graph

# Default image path
DEFAULT_IMAGE_PATH = "figs/Dunne et al 2015 Figure 1.jpg"
DEFAULT_FILE_NAME = "graphs/graph.graphml"
NEXT_NODE = 1


# Generate a node label with the next available integer
def get_next_id():
    global NEXT_NODE
    label = "Node " + str(NEXT_NODE)
    NEXT_NODE += 1
    return label


# Check if the node is near an existing node
def nearby_nodes(nodes, x, y, radius=15):
    for label, node in nodes:
        if ((node["x"] - x) ** 2 + (node["y"] - y) ** 2) ** 0.5 < radius:
            return label, node
    return None, None


def main(args):
    # Prompt for an image path
    if args.image:
        image_path = args.image
        print(f"Using image path: {image_path}")
    else:
        image_path = input(f"Enter the path to the image: [{DEFAULT_IMAGE_PATH}] ")
        if not image_path:
            image_path = DEFAULT_IMAGE_PATH

    # Load the image
    try:
        img = plt.imread(image_path)
        print("Image loaded successfully!")
    except FileNotFoundError:
        print("Image not found!")
        exit()

    # Display the image
    fig, ax = plt.subplots()
    fig.tight_layout()
    ax.imshow(img)
    ax.set_xticks([])

    # Label with the filename
    ax.set_title(image_path)

    # Add a caption with instructions
    ax.text(
        0.5,
        0,
        "Click to add nodes. Click two existing nodes to add an edge.\nRight click to delete a node and its edges. Press 'q' to quit.",
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="center",
        fontsize=10,
        color="black",
        backgroundcolor=(1, 1, 1, 0.5),
    )

    # Set up the graph
    nodes = []
    edges = []

    # Set up the click event handler
    IN_EDGE_CREATION = False
    EDGE_START = None

    def on_click(event):
        """Handle single mouse clicks on the plot."""
        # Check if the click is in the plot
        if event.inaxes is None:
            return
        nonlocal IN_EDGE_CREATION, EDGE_START

        if event.button == 1:
            # Left click

            x, y = int(event.xdata), int(event.ydata)

            # Check if the click is on an existing node
            label, existing_node = nearby_nodes(nodes, x, y)

            if not IN_EDGE_CREATION:
                if existing_node:
                    # Start edge creation mode
                    IN_EDGE_CREATION = True
                    EDGE_START = (label, existing_node)
                    print(f"Creating edge from {label}")
                else:
                    # Add a new node
                    label = get_next_id()
                    nodes.append((label, {"x": x, "y": y}))
                    print(f"Added node {label} at ({x}, {y})")
                    # Update the plot
                    ax.scatter(x, y, color="r")
                    ax.annotate(label, (x, y))
                    plt.draw()
            else:
                # Finish edge creation mode
                IN_EDGE_CREATION = False
                if existing_node:
                    print(f"Creating edge from {EDGE_START[0]} to {label}")
                    edges.append((EDGE_START[0], label))
                    # Update the plot
                    ax.plot(
                        [EDGE_START[1]["x"], existing_node["x"]],
                        [EDGE_START[1]["y"], existing_node["y"]],
                        color="b",
                    )
                    plt.draw()
                else:
                    # Add a node and complete the edge
                    label = get_next_id()
                    nodes.append((label, {"x": x, "y": y}))
                    print(f"Added node {label} at ({x}, {y})")
                    edges.append((EDGE_START[0], label))
                    # Update the plot
                    ax.plot(
                        [EDGE_START[1]["x"], x],
                        [EDGE_START[1]["y"], y],
                        color="b",
                    )
                    ax.scatter(x, y, color="r")
                    ax.annotate(label, (x, y))
                    plt.draw()

        elif event.button == 3:
            # Right click

            # Negate edge creation mode
            IN_EDGE_CREATION = False
            EDGE_START = None

            x, y = int(event.xdata), int(event.ydata)

            # Check if the click is on an existing node
            key, existing_node = nearby_nodes(nodes, x, y)

            if existing_node:
                # Delete associated edges
                for edge in list(edges):
                    if existing_node in edge:
                        edges.remove(edge)
                        print(
                            f"Deleting edge from {edge[0]['label']} to {edge[1]['label']}"
                        )

                # Delete the node and re-draw the plot
                print(f"Deleting node {existing_node['label']}")
                del nodes[key]
                ax.cla()
                ax.imshow(img)
                for _, node in nodes.items():
                    ax.scatter(node["x"], node["y"], color="r")
                    ax.annotate(node["label"], (node["x"], node["y"]))
                for edge in edges:
                    ax.plot(
                        [edge[0]["x"], edge[1]["x"]],
                        [edge[0]["y"], edge[1]["y"]],
                        color="b",
                    )
                plt.draw()

    def on_key(event):
        """Handle single key presses on the plot."""
        # Close on q
        if event.key == "q":
            plt.close()
        # Cancel edge creation on escape
        elif event.key == "escape":
            nonlocal IN_EDGE_CREATION, EDGE_START
            IN_EDGE_CREATION = False
            EDGE_START = None

    # Connect the click event handler
    cid = fig.canvas.mpl_connect("button_press_event", on_click)
    qid = fig.canvas.mpl_connect("key_press_event", on_key)

    # Display the plot and wait for clicks
    plt.show()

    # Disconnect event handler when plot is closed
    plt.disconnect(cid)

    # Create a graph from the nodes and edges
    G = nx.Graph()
    # Convert nodes to list of tuples
    G.add_nodes_from(nodes)
    # Convert edges to list of tuples
    G.add_edges_from(edges)

    # Display the graph with networkx
    plt.clf()
    draw_graph(G, ax=ax)
    ax.set_title("Captured Graph")

    # Disconnect event handler when plot is closed
    plt.disconnect(qid)

    # Prompt to save the graph
    save_graph = input("Save the graph? (y/[n]): ")
    if save_graph.lower() == "y":
        if args.output:
            output_path = args.output
            print(f"Using output path: {output_path}")
        else:
            # Prompt for a path
            output_path = input(f"Enter output path: [{DEFAULT_FILE_NAME}] ")
            if not output_path:
                output_path = DEFAULT_FILE_NAME
        nx.write_graphml(G, output_path)
        print("Graph saved successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Create a graph from an image.")
    parser.add_argument(
        "-i",
        "--image",
        type=str,
        help="Path to the image to create a graph from.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Path to save the graph to.",
    )
    args = parser.parse_args()

    main(args)
