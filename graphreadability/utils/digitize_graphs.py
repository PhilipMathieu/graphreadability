import matplotlib.pyplot as plt
import networkx as nx

# Default image path
DEFAULT_IMAGE_PATH = "figs/Dunne et al 2015 Figure 1.jpg"
DEFAULT_FILE_NAME = "graphs/graph.graphml"


# Check if the node is near an existing node
def nearby_nodes(nodes, x, y, radius=10):
    for _, node in nodes.items():
        if ((node["pos"][0] - x) ** 2 + (node["pos"][1] - y) ** 2) ** 0.5 < radius:
            return node
    return False


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
    ax.imshow(img)

    # Set up the graph
    nodes = {}
    edges = []

    # Set up the click event handler
    IN_EDGE_CREATION = False
    EDGE_START = None

    def on_click(event):
        """Handle mouse clicks on the plot."""
        nonlocal IN_EDGE_CREATION, EDGE_START
        x, y = int(event.xdata), int(event.ydata)

        # Check if the click is on an existing node
        existing_node = nearby_nodes(nodes, x, y)

        if not IN_EDGE_CREATION:
            if existing_node:
                # Start edge creation mode
                IN_EDGE_CREATION = True
                EDGE_START = existing_node
                print(f"Creating edge from {existing_node}")
            else:
                # Add a new node
                nodes[(x, y)] = {"label": "Node " + str(len(nodes) + 1), "pos": (x, y)}
                print(f"Added node {nodes[(x, y)]['label']} at ({x}, {y})")
                # Update the plot
                ax.scatter(x, y, color="r")
                ax.annotate(nodes[(x, y)]["label"], (x, y))
                plt.draw()
        else:
            # Finish edge creation mode
            IN_EDGE_CREATION = False
            if existing_node:
                print(f"Creating edge from {EDGE_START} to {existing_node}")
                edges.append((EDGE_START, existing_node))
                # Update the plot
                ax.plot(
                    [EDGE_START["pos"][0], existing_node["pos"][0]],
                    [EDGE_START["pos"][1], existing_node["pos"][1]],
                    color="b",
                )
                plt.draw()
            else:
                # Add a node and complete the edge
                nodes[(x, y)] = {"label": "Node " + str(len(nodes) + 1), "pos": (x, y)}
                print(f"Added node {nodes[(x, y)]['label']} at ({x}, {y})")
                edges.append((EDGE_START, nodes[(x, y)]))
                # Update the plot
                ax.plot(
                    [EDGE_START["pos"][0], x],
                    [EDGE_START["pos"][1], y],
                    color="b",
                )
                ax.scatter(x, y, color="r")
                ax.annotate(nodes[(x, y)]["label"], (x, y))
                plt.draw()

    # Connect the click event handler
    cid = fig.canvas.mpl_connect("button_press_event", on_click)

    # Display the plot and wait for clicks
    plt.show()

    # Disconnect event handler when plot is closed
    plt.disconnect(cid)

    # Print the created graph structure
    G = nx.Graph()
    G.add_nodes_from(nodes)
    print(G.nodes())
    print(G.edges())

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
