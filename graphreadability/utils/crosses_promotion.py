import networkx as nx
from utils.helpers import _intersect, compute_intersection
import copy


def crosses_promotion3(G):
    H = G.copy()

    edge_list = copy.deepcopy(H.edges())

    for u, v in edge_list:
        for x, y in edge_list:
            if (u, v) == (x, y):
                continue

            line_a = ((H.nodes[u]['x'], H.nodes[u]['y']), (H.nodes[v]['x'], H.nodes[v]['y']))
            line_b = ((H.nodes[x]['x'], H.nodes[x]['y']), (H.nodes[y]['x'], H.nodes[y]['y']))

            if _intersect(line_a, line_b):
                intersection = compute_intersection(line_a[0], line_a[1], line_b[0], line_b[1])

                new_node = "c" + str(len(H.nodes()) + 1)                

                H.add_node(new_node)

                # Add some default attributes to the node
                H.nodes[new_node]["label"] = '\n'
                H.nodes[new_node]["shape_type"] = "ellipse"
                H.nodes[new_node]["x"] = intersection[0]
                H.nodes[new_node]["y"] = intersection[1]
                H.nodes[new_node]["type"] = "minor" # Only nodes added in crosses promotion are minor

                # Add edges between the new node and end points of e and e2, then remove e and e2
                H.add_edge(u, new_node)
                H.add_edge(new_node, v)

                H.add_edge(x, new_node)
                H.add_edge(new_node, y)

                try:
                    H.remove_edge(u, v)
                    H.remove_edge(x, y)
                except:
                    pass

    for n in H.nodes(data=True):
        print(n)

    return H

def crosses_promotion2(G):
    H = G.copy()

    for n in H.nodes():
        H.nodes[n]["type"] = "major"

    nodes_to_add = []
    edges_to_add = []
    edges_to_remove = []
    covered = []

    for u, v in H.edges():
        for x, y in H.edges():
            if (u, v) == (x, y):
                continue

            if ((u, v), (x, y)) in covered:
                continue

            line_a = ((H.nodes[u]['x'], H.nodes[u]['y']), (H.nodes[v]['x'], H.nodes[v]['y']))
            line_b = ((H.nodes[x]['x'], H.nodes[x]['y']), (H.nodes[y]['x'], H.nodes[y]['y']))

            if _intersect(line_a, line_b):
                intersection = compute_intersection(line_a[0], line_a[1], line_b[0], line_b[1])

                new_node = "c" + str(len(H.nodes()) + len(nodes_to_add) + 1)

                nodes_to_add.append((new_node, intersection[0], intersection[1]))

                edges_to_add.append((u, new_node))
                edges_to_add.append((new_node, v))
                edges_to_add.append((x, new_node))
                edges_to_add.append((new_node, y))

                edges_to_remove.append((u, v))
                edges_to_remove.append((x, y))

                covered.append(((x, y), (u, v)))

    
    for n in set(nodes_to_add):
        H.add_node(n[0])
        H.nodes[n[0]]["label"] = '\n'
        H.nodes[n[0]]["shape_type"] = "ellipse"
        H.nodes[n[0]]["x"] = n[1]
        H.nodes[n[0]]["y"] = n[2]
        H.nodes[n[0]]["type"] = "minor" # Only nodes added in crosses promotion are minor

    
    for c, d in set(edges_to_add):
        H.add_edge(c, d)

    for a, b in set(edges_to_remove):
        H.remove_edge(a, b)

    # !! THIS CODE IS REPLACED BY COVERED
    # nodes_to_remove = []
    # covered = []
    # for u in H.nodes():
    #     for v in H.nodes():
    #         if u == v:
    #             continue
            
    #         if (u, v) in covered or (v, u) in covered:
    #             continue

    #         if H.nodes[u]['x'] == H.nodes[v]['x'] and H.nodes[u]['y'] == H.nodes[v]['y']:
    #             nodes_to_remove.append(v)
    #             covered.append((u, v))

    # for n in nodes_to_remove:
    #     H.remove_node(n)

    for n in H.nodes(data=True):
        print(n)


    return H


def crosses_promotion(G):
    H = G.copy()

    for n in H.nodes():
        H.nodes[n]["type"] = "major"

    covered = []
    intersections = {}
    for u, v in H.edges():
        for x, y in H.edges():
            if (u, v) == (x, y):
                continue

            if ((u, v), (x, y)) in covered:
                continue

            line_a = ((H.nodes[u]['x'], H.nodes[u]['y']), (H.nodes[v]['x'], H.nodes[v]['y']))
            line_b = ((H.nodes[x]['x'], H.nodes[x]['y']), (H.nodes[y]['x'], H.nodes[y]['y']))

            if _intersect(line_a, line_b):
                try:
                    intersection = compute_intersection(line_a[0], line_a[1], line_b[0], line_b[1])
                    if (u, v) not in intersections.keys():
                        intersections[(u, v)] = []
                
                    if (x, y) not in intersections.keys():
                        intersections[(x, y)] = []
                    
                    intersections[(u, v)].append((intersection[0], intersection[1]))
                    intersections[(x, y)].append((intersection[0], intersection[1]))
                except:
                    pass

                

                covered.append(((x, y), (u, v)))


    intersections_covered = []

    for k, v in intersections.items():
        #print(f"{k}: {v}")

        H.remove_edge(k[0], k[1])

        node_list = []

        points = sorted(v, key=lambda v: v[0])
        
        if H.nodes[k[0]]['x'] < points[0][0]:
            node_list.append(k[0])
        else:
            node_list.append(k[1])

        for x, y in points:
            if (x, y) not in intersections_covered:
                new_node = "c" + str(len(H.nodes()))
                H.add_node(new_node)
                H.nodes[new_node]["label"] = '\n'
                H.nodes[new_node]["shape_type"] = "ellipse"
                H.nodes[new_node]["x"] = x
                H.nodes[new_node]["y"] = y
                H.nodes[new_node]["type"] = "minor" # Only nodes added in crosses promotion are minor
                H.nodes[new_node]["color"] = "#3BC6E5" # Blue color to visually distinquish crossing nodes
                node_list.append(new_node)
                intersections_covered.append((x, y))
            else:
                node = [a for a,b in H.nodes(data=True) if b['x']==x and b['y'] == y]
                node_list.append(node[0])

        if H.nodes[k[0]]['x'] < points[0][0]:
            node_list.append(k[1])
        else:
            node_list.append(k[0])

        for i in range(len(node_list) - 1):
            H.add_edge(node_list[i], node_list[i+1])

        H.remove_edges_from(nx.selfloop_edges(H))

    # for n in H.nodes():
    #     print(n)

    return H


def count_crossings(G):

    covered = []
    c = 0
    for e in G.edges:
        
        a_p1 = (G.nodes[e[0]]["x"], G.nodes[e[0]]["y"]) # Position of source node of e
        a_p2 = (G.nodes[e[1]]["x"], G.nodes[e[1]]["y"]) # Position of target node of e
        line_a = (a_p1, a_p2)
        
        for e2 in G.edges:
            if e == e2:
                continue
            
            b_p1 = (G.nodes[e2[0]]["x"], G.nodes[e2[0]]["y"]) # Position of source node of e2
            b_p2 = (G.nodes[e2[1]]["x"], G.nodes[e2[1]]["y"]) # Position of target node of e2
            line_b = (b_p1, b_p2)
            
            if _intersect(line_a, line_b) and (line_a, line_b) not in covered:
                covered.append((line_b, line_a))                  
                c += 1
    return c

def main():
    pass


if __name__ == "__main__":
    main()