from search import *
from notebook import psource, heatmap, gaussian_kernel, show_map, final_path_colors, display_visual, plot_NQueens
brest_map = UndirectedGraph(dict(
    Avignon = dict(Grenoble = 227, Lyon = 104, Montpellier = 121),
    Bordeaux = dict(Limoges = 220,Nantes = 329,Toulouse = 253),
    Brest = dict(Rennes = 244),
    Caen = dict(Calais = 120,Paris = 241, Rennes = 176),
    Calais = dict(Caen = 120,Nancy = 534, Paris = 297),
    Dijon = dict(Nancy = 201, Paris = 313, Strasbourg = 335),
    Grenoble = dict (Avignon = 227, Lyon = 104),
    Limoges = dict(Bordeaux = 220,Lyon = 389,Nantes = 329,Paris = 396,Toulouse = 313),
    Lyon = dict(Dijon = 192,Grenoble = 104,Limoges = 389 ),
    Marseille = dict(Avignon = 99, Nice = 188),
    Montpellier = dict(Avignon = 121,Toulouse = 240),
    Nancy = dict(Calais = 534, Dijon = 201,Paris = 372, Strasbourg = 145),
    Nantes = dict(Bordeaux = 329, Limoges = 329,Rennes = 107),
    Nice = dict(Marseille = 188),
    Paris = dict(Caen = 241,Calais = 297,Dijon = 313,Limoges = 396, Nancy = 372,Rennes = 348),
    Rennes = dict(Brest = 244,Caen = 176,Nantes = 107, Paris = 348),
    Strasbourg = dict(Dijon = 335, Nancy = 145),
    Toulouse = dict(Bordeaux = 253,Limoges = 313,Montpellier = 240 )
))

brest_map.locations=dict(Calais=(240,530),Caen=(220,510),Nancy=(280,480),
                         Strasbourg=(300,500),Rennes=(200,480),Brest=(190,500),Paris=(250,470),Dijon=(280,450),
                         Lyon=(280,400),Nantes=(200,425),Limoges=(230,410),Bordeaux=(200,370),Grenoble=(300,370),
                         Avignon=(280,350),Montpellier=(250,350),Toulouse=(215,350),Marseille=(290,320),Nice=(320,330)
                        )

santa_barbara_map = UndirectedGraph(dict(
    Barstow=dict(Riverside=75, Santa_Barbara=45),
    El_Cajon=dict(San_Diego=15),
    Los_Angeles=dict(Malibu=20, Riverside=25, San_Diego=100),
    Malibu=dict(Los_Angeles=20, Santa_Barbara=45),
    Palm_Springs=dict(Riverside=75),
    Riverside=dict(Barstow=75, Los_Angeles=25, Palm_Springs=75, San_Diego=90),
    Santa_Barbara=dict(Barstow=45, Malibu=45, Los_Angeles=30),
    San_Diego=dict(El_Cajon=15, Los_Angeles=100, Riverside=90)))

santa_barbara_map.locations = dict(
    Barstow=(240, 530), El_Cajon=(270, 300), Los_Angeles=(120, 420),
    Malibu=(80, 450), Palm_Springs=(280, 450), Riverside=(200, 420),
    Santa_Barbara=(131, 530), San_Diego=(210, 300))

# node colors, node positions and node label positions
node_colors = {node: 'white' for node in santa_barbara_map.locations.keys()}
node_positions = santa_barbara_map.locations
node_label_pos = {k: [v[0], v[1] - 10] for k, v in santa_barbara_map.locations.items()}
edge_weights = {(k, k2): v2 for k, v in santa_barbara_map.graph_dict.items() for k2, v2 in v.items()}

santa_barbara_graph_data = {'graph_dict': santa_barbara_map.graph_dict,
                            'node_colors': node_colors,
                            'node_positions': node_positions,
                            'node_label_positions': node_label_pos,
                            'edge_weights': edge_weights
                            }


def tree_depth_search_for_vis(problem):
    """Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Don't worry about repeated paths to a state. [Figure 3.7]"""

    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}

    # Adding first node to the stack
    frontier = [Node(problem.initial)]

    node_colors[Node(problem.initial).state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    while frontier:
        # Popping first node of stack
        node = frontier.pop()

        # modify the currently searching node to red
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            # modify goal node to green after reaching the goal
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        frontier.extend(node.expand(problem))

        for n in node.expand(problem):
            node_colors[n.state] = "orange"
            iterations += 1
            all_node_colors.append(dict(node_colors))

        # modify the color of explored nodes to gray
        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))

    return None


# romania_problem = GraphProblem('Arad', 'Bucharest', romania_map)
#
#
# k = iterations, all_node_colors, node = tree_depth_search_for_vis(romania_problem)
# print(k)


def best_first_graph_search_for_vis(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    # we use these two variables at the time of visualisations
    iterations = 0
    all_node_colors = []
    node_colors = {k: 'white' for k in problem.graph.nodes()}

    f = memoize(f, 'f')
    node = Node(problem.initial)
    node_colors[node.state] = "red"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    if problem.goal_test(node.state):
        node_colors[node.state] = "green"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        return (iterations, all_node_colors, node)

    frontier = PriorityQueue('min', f)
    frontier.append(node)

    node_colors[node.state] = "orange"
    iterations += 1
    all_node_colors.append(dict(node_colors))

    explored = set()
    while frontier:
        node = frontier.pop()
        print(node)
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))

        if problem.goal_test(node.state):
            node_colors[node.state] = "green"
            iterations += 1
            all_node_colors.append(dict(node_colors))
            return (iterations, all_node_colors, node)

        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                node_colors[child.state] = "orange"
                iterations += 1
                all_node_colors.append(dict(node_colors))
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < incumbent:
                    del frontier[child]
                    frontier.append(child)
                    node_colors[child.state] = "orange"
                    iterations += 1
                    all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
    return None

# node colors, node positions and node label positions
node_colors = {node: 'white' for node in brest_map.locations.keys()}
node_positions = brest_map.locations
node_label_pos = { k:[v[0],v[1]-10]  for k,v in brest_map.locations.items() }
edge_weights = {(k, k2) : v2 for k, v in brest_map.graph_dict.items() for k2, v2 in v.items()}

brest_graph_data = {  'graph_dict' : brest_map.graph_dict,
                        'node_colors': node_colors,
                        'node_positions': node_positions,
                        'node_label_positions': node_label_pos,
                         'edge_weights': edge_weights
                     }




all_node_colors = []
problem = GraphProblem('Barstow', 'El_Cajon', santa_barbara_map)
h = memoize(None or problem.h, 'h')
iterations, all_node_colors, node = best_first_graph_search_for_vis(problem,
                                                                lambda n: n.path_cost)