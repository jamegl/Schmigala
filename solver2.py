from edge import Edge
from graph import Graph
from node import Node
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

class Solver2:
    def __init__(self, graph):
        self.graph = graph
        self.result_triangles = []
        self.max_number_of_solutions = 50
        self.first_edge = "" 
        self.drawn_solutions = []
        self.sol_count = 0

    # generate edges, that don't exist, sets weight to 0
    def generate_all_edges(self):
        for i in self.graph.nodes:
            for j in self.graph.nodes:
                if i == j:
                    continue
                if self.graph.get_edge_by_node_names(i.name, j.name) is None:
                    self.graph.add_edge(Edge(i, j, 0))

    # get nodes with max sum weight for edge
    def get_max_for_edge(self, edge, used_nodes):
        nodes_sum_weights_map = {}

        for node in self.graph.nodes:
            if node == edge.first_node or node == edge.second_node or used_nodes.count(node) != 0:
                continue
            sum_weight = self.get_sum_weight(edge, node)
            nodes_sum_weights_map[node] = sum_weight

        max_sum_weight = 0
        if nodes_sum_weights_map:
            max_sum_weight = max(nodes_sum_weights_map.values())
        else:
            return None, None
        
        max_nodes = [x for x, y in nodes_sum_weights_map.items() if y == max_sum_weight]
        return max_nodes, max_sum_weight


    # get sum weight for edge for specific node
    def get_sum_weight(self, edge, node):
        first_node_edge = self.graph.get_edge(edge.first_node, node)
        first_node_weight = first_node_edge.weight

        second_node_edge = self.graph.get_edge(edge.second_node, node)
        second_node_weight = second_node_edge.weight

        return first_node_weight + second_node_weight

    def get_triangles_with_common_node(self, node, triangles, free_edges):
        triangles_strings = []
        triangles_string = ""
        for triangle in triangles:
            triangle_string = "".join(set(triangle[0].first_node.name + triangle[0].second_node.name +
                            triangle[1].first_node.name + triangle[1].second_node.name +
                            triangle[2].first_node.name + triangle[2].second_node.name))
            triangles_strings.append(triangle_string)
            triangles_string += triangle_string

        number_of_triangles_with_node = 0
        for string in triangles_strings:
            if node.name in string:
                number_of_triangles_with_node += 1

        unique_nodes = ""
        for i in triangles_string:
            count = 0
            for j in triangles_string:
                if i == j:
                    count += 1
                if count > 1:
                    break
            if count == 1:
                unique_nodes += i

        if number_of_triangles_with_node == 5 and len(unique_nodes) == 2:

            edge_with_unique_nodes = self.graph.get_edge_by_node_names(unique_nodes[0], unique_nodes[1])
            first_edge_to_remove = self.graph.get_edge_by_node_names(unique_nodes[0], node.name)
            second_edge_to_remove = self.graph.get_edge_by_node_names(unique_nodes[1], node.name)
            third_edge_to_remove = self.graph.get_edge_by_node_names(node.name, unique_nodes[0])
            forth_edge_to_remove = self.graph.get_edge_by_node_names(node.name, unique_nodes[1])
            free_edges.append(edge_with_unique_nodes)
            if free_edges.count(first_edge_to_remove) != 0:
                free_edges.remove(first_edge_to_remove)
            if free_edges.count(second_edge_to_remove) != 0:
                free_edges.remove(second_edge_to_remove)
            if free_edges.count(third_edge_to_remove) != 0:
                free_edges.remove(third_edge_to_remove)
            if free_edges.count(forth_edge_to_remove) != 0:
                free_edges.remove(forth_edge_to_remove)

            triangles.append([edge_with_unique_nodes, first_edge_to_remove, second_edge_to_remove])

    def add_triangle(self, free_edges_of_triangles, used_nodes, triangles, edge, node):
        second_edge = self.graph.get_edge(edge.first_node, node)
        third_edge = self.graph.get_edge(edge.second_node, node)
        triangle = [edge, second_edge, third_edge]
        triangles.append(triangle)

        used_nodes.append(edge.first_node)
        used_nodes.append(edge.second_node)
        used_nodes.append(node)

        if free_edges_of_triangles.count(edge) != 0:
            free_edges_of_triangles.remove(edge)
        else:
            free_edges_of_triangles.append(edge)

        if free_edges_of_triangles.count(second_edge) != 0:
            free_edges_of_triangles.remove(second_edge)
        else:
            free_edges_of_triangles.append(second_edge)

        if free_edges_of_triangles.count(third_edge) != 0:
            free_edges_of_triangles.remove(third_edge)
        else:
            free_edges_of_triangles.append(third_edge)

        if len(triangles) >= 5:
            for i in self.graph.nodes:
                self.get_triangles_with_common_node(i, triangles, free_edges_of_triangles)

    def add_edge_to_solution(self, free_edges, used_nodes, triangles, edge, node):
        second_edge = self.graph.get_edge(edge.first_node, node)
        third_edge = self.graph.get_edge(edge.second_node, node)
        print("Edge: ")
        print(edge)
        print("Node: ")
        print(node.name)
        print("Free edges")
        for i in free_edges:
            print(i)
        print("Triangles")
        for i in triangles:
            for j in i:
                print(j)
            print("|")

        print("Second edge weight: " + str(second_edge.get_weight()))
        print("Third edge weight: " + str(third_edge.get_weight()))

        if second_edge.get_weight() != 0 and third_edge.get_weight() != 0:
            triangle = [edge, second_edge, third_edge]
            triangles.append(triangle)
            
            if free_edges.count(edge) != 0:
                number = 0
                for i in triangles:
                    if edge in i:
                        number = number + 1
                if number == 2:
                    free_edges.remove(edge)
            else:
                free_edges.append(edge)

            if free_edges.count(second_edge) != 0:
                number = 0
                for i in triangles:
                    if second_edge in i:
                        number = number + 1
                if number == 2:
                    free_edges.remove(second_edge)
            else:
                free_edges.append(second_edge)

            if free_edges.count(third_edge) != 0:
                number = 0
                for i in triangles:
                    if third_edge in i:
                        number = number + 1
                if number == 2:
                    free_edges.remove(third_edge)
            else:
                free_edges.append(third_edge)

            if len(triangles) >= 5:
                for i in self.graph.nodes:
                    self.get_triangles_with_common_node(i, triangles, free_edges)

        elif second_edge.get_weight() != 0:
            print("Adding second edge")
            free_edges.append(second_edge)
            print("Free Edges after adding")
            for i in free_edges:
                print(i)

        elif third_edge.get_weight() != 0:
            free_edges.append(third_edge)

        used_nodes.append(edge.first_node)
        used_nodes.append(edge.second_node)
        used_nodes.append(node)


    def get_free_edges_max_nodes(self, free_edges_of_triangles, used_nodes):
        edges_max_nodes = {}
        for edge in free_edges_of_triangles:
            max_nodes, max_sum_weight = self.get_max_for_edge(edge, used_nodes)
            if max_nodes is None:
                continue
            edges_max_nodes[edge] = [max_nodes, max_sum_weight]
        max_weight = 0
        if edges_max_nodes:
            max_weight = max(edges_max_nodes[key][1] for key in edges_max_nodes)
        max_edges = [x for x, y in edges_max_nodes.items() if y[1] == max_weight]
        result_map = {}
        for i in max_edges:
            result_map[i] = edges_max_nodes[i]
        return result_map

    def find_next_node(self, free_edges_of_triangles, used_nodes, triangles):
        edges_max_nodes = self.get_free_edges_max_nodes(free_edges_of_triangles, used_nodes)
        
        if len(self.result_triangles) == self.max_number_of_solutions:
            return
        
        print("Used nodes: ")
        for i in set(used_nodes):
            print(i.name)

        print("Graph nodes: ")
        for i in set(self.graph.nodes):
            print(i.name)

        print("Is subset:")
        print(set(self.graph.nodes).issubset(set(used_nodes)))
        
        if set(self.graph.nodes).issubset(set(used_nodes)):
            edges_without_triangles = []
            edges_with_triangles = []
            for i in free_edges_of_triangles:
                for j in triangles:
                    if i in j:
                        edges_with_triangles.append(i)
                        break
                if i not in edges_with_triangles:
                    edges_without_triangles.append(i)

            print("Edges with triangles:")
            for i in edges_with_triangles:
                print(i)

            print("Edges withput triangles:")
            for i in edges_without_triangles:
                print(i)

            for i in edges_without_triangles:
                for j in edges_with_triangles:
                    if i.first_node == j.first_node:
                        third_edge = self.graph.get_edge(i.second_node, j.second_node)
                        triangles.append([i, j, third_edge])
                        edges_with_triangles.append(third_edge)
                        edges_with_triangles.append(i)
                        break
                    elif i.first_node == j.second_node:
                        third_edge = self.graph.get_edge(i.second_node, j.first_node)
                        triangles.append([i, j, third_edge])
                        edges_with_triangles.append(third_edge)
                        edges_with_triangles.append(i)
                        break
                    elif i.second_node == j.first_node:
                        third_edge = self.graph.get_edge(i.first_node, j.second_node)
                        triangles.append([i, j, third_edge])
                        edges_with_triangles.append(third_edge)
                        edges_with_triangles.append(i)
                        break
                    elif i.second_node == j.second_node:
                        third_edge = self.graph.get_edge(i.first_node, j.first_node)
                        triangles.append([i, j, third_edge])
                        edges_with_triangles.append(third_edge)
                        edges_with_triangles.append(i)
                        break

            print("Final Triangles")
            for i in triangles:
                for j in i:
                    print(j)
                print("|")

            self.result_triangles.append(triangles)
            self.sol_count = self.sol_count + 1
            self.draw_solution(triangles, self.sol_count)
            return


        if edges_max_nodes:
            for x, y in edges_max_nodes.items():
                for node in y[0]:
                    if node in used_nodes:
                        continue
                    new_triangles = triangles.copy()
                    new_free_edges_of_triangles = free_edges_of_triangles.copy()
                    new_used_nodes = used_nodes.copy()
                    self.add_edge_to_solution(new_free_edges_of_triangles, new_used_nodes, new_triangles, x, node)
                    self.find_next_node(new_free_edges_of_triangles, new_used_nodes, new_triangles)
        else:
            self.result_triangles.append(triangles)

    def contains_edge(self, list, edge):
        for i in list:
            if i.equals(edge):
                return True
        return False

    def get_path_length(self, graph_edges, start, goal):
        explored = []

        queue = [[start]]

        if start == goal:
            print("Same Node")
            return

        while queue:
            path = queue.pop(0)
            node = path[-1]

            if node not in explored:
                neighbours = graph_edges[node]

                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)

                    if neighbour == goal:
                        return len(new_path) - 1
                explored.append(node)

    def get_nodes_connections(self, solution):
        edges = []
        for triangle in solution:
            for edge in triangle:
                edges.append(edge)

        nodes_connections = {}
        for node in self.graph.nodes:
            node_connections = []
            for edge in edges:
                if edge.first_node == node:
                    node_connections.append(edge.second_node.name)
                elif edge.second_node == node:
                    node_connections.append(edge.first_node.name)
            nodes_connections[node.name] = list(dict.fromkeys(node_connections))
        return nodes_connections

    def calculate_solution_value(self, solution):
        result = 0
        for edge in self.graph.edges:
            if edge.weight != 0:
                path_length = self.get_path_length(self.get_nodes_connections(solution), edge.first_node.name, edge.second_node.name)
                result += path_length * edge.weight
        return result


    def draw_solution(self, solution, count):
        triangles = []
        coordinates_array = [[1, 1], [2, 2.5], [3, 1]]
        for triangle in solution:
            triangle_name = ""
            first_edge_node = ""
            for edge in triangle:
                if triangle == solution[0] and edge == self.first_edge:
                    first_edge_node = edge.second_node.name
                    triangle_name += edge.first_node.name
                else:
                    triangle_name += edge.first_node.name + edge.second_node.name
            if triangle == solution[0]:
                first_triangle_name = triangle_name.replace(first_edge_node, "") + first_edge_node
                triangles.append("".join(OrderedDict.fromkeys(first_triangle_name)))
            else:
                triangles.append("".join(OrderedDict.fromkeys(triangle_name)))
        nodes_coordinates = {triangles[0][0]: [1, 1], triangles[0][1]: [2, 2.5], triangles[0][2]: [3, 1]}
        solution_already_drawn = False

        for drawn_solution in self.drawn_solutions:
            if all(item in drawn_solution for item in triangles):
                solution_already_drawn = True

        if solution_already_drawn:
            return
        
        self.drawn_solutions.append(triangles)
        while len(nodes_coordinates) < len(self.graph.nodes):
            for triangle in triangles:
                if nodes_coordinates.get(triangle[0]) is not None and nodes_coordinates.get(triangle[1]) is not None \
                        and nodes_coordinates.get(triangle[2]) is not None:
                    continue
                if nodes_coordinates.get(triangle[0]) is not None and nodes_coordinates.get(triangle[1]) is not None \
                        and nodes_coordinates.get(triangle[2]) is None:
                    found_triangle = list(filter(lambda x: triangle[0] in x and triangle[1] in x, triangles))[0]
                    third_node = found_triangle.replace(triangle[0], "").replace(triangle[1], "")
                    try:
                        x = nodes_coordinates[triangle[0]][0] + nodes_coordinates[triangle[1]][0] - nodes_coordinates[third_node][0]
                        y = nodes_coordinates[triangle[0]][1] + nodes_coordinates[triangle[1]][1] - nodes_coordinates[third_node][1]
                        nodes_coordinates[triangle[2]] = [x, y]
                        coordinates_array.append(nodes_coordinates[triangle[0]])
                        coordinates_array.append(nodes_coordinates[triangle[1]])
                        coordinates_array.append([x, y])
                    except KeyError:
                        continue
                elif nodes_coordinates.get(triangle[0]) is not None and nodes_coordinates.get(triangle[1]) is None \
                        and nodes_coordinates.get(triangle[2]) is not None:
                    found_triangle = list(filter(lambda x: triangle[0] in x and triangle[2] in x, triangles))[0]
                    third_node = found_triangle.replace(triangle[0], "").replace(triangle[2], "")
                    try:
                        x = nodes_coordinates[triangle[0]][0] + nodes_coordinates[triangle[2]][0] - nodes_coordinates[third_node][0]
                        y = nodes_coordinates[triangle[0]][1] + nodes_coordinates[triangle[2]][1] - nodes_coordinates[third_node][1]
                        nodes_coordinates[triangle[1]] = [x, y]
                        coordinates_array.append(nodes_coordinates[triangle[0]])
                        coordinates_array.append(nodes_coordinates[triangle[2]])
                        coordinates_array.append([x, y])
                    except KeyError:
                        continue
                elif nodes_coordinates.get(triangle[0]) is None and nodes_coordinates.get(triangle[1]) is not None \
                        and nodes_coordinates.get(triangle[2]) is not None:
                    found_triangle = list(filter(lambda x: triangle[1] in x and triangle[2] in x, triangles))[0]
                    third_node = found_triangle.replace(triangle[1], "").replace(triangle[2], "")
                    try:
                        x = nodes_coordinates[triangle[1]][0] + nodes_coordinates[triangle[2]][0] - nodes_coordinates[third_node][0]
                        y = nodes_coordinates[triangle[1]][1] + nodes_coordinates[triangle[2]][1] - nodes_coordinates[third_node][1]
                        nodes_coordinates[triangle[0]] = [x, y]
                        coordinates_array.append(nodes_coordinates[triangle[1]])
                        coordinates_array.append(nodes_coordinates[triangle[2]])
                        coordinates_array.append([x, y])
                    except KeyError:
                        continue


        X = np.array(list(nodes_coordinates.values()))

        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], s=1)
        for node, coordinates in nodes_coordinates.items():
            plt.text(coordinates[0], coordinates[1], node, size=14, color='red')

        newX = np.array(coordinates_array)

        num = 0

        while num + 3 <= len(coordinates_array):
            t1 = plt.Polygon(newX[num:num+3, :], ec='k', lw=1, fill=False)
            plt.gca().add_patch(t1)
            num += 3

        plt.axis('off')

        plt.savefig('solutions/solution' + str(count) + "_" + str(self.calculate_solution_value(solution)) + '.png') # changed i to solution, need to test

        plt.close()

    def find_solutions(self):
        self.generate_all_edges()

        max_edge = self.graph.get_max_edge()
        self.first_edge = max_edge
        used_nodes = []
        max_nodes, max_sum_weight = self.get_max_for_edge(max_edge, used_nodes)

        for first_triangle_node in max_nodes:  # multiple solutions if max edge has few nodes with max sum weight
            triangles = []
            free_edges = []
            used_nodes = []
            free_edges.append(max_edge)
            self.add_edge_to_solution(free_edges, used_nodes, triangles, max_edge, first_triangle_node)
            #self.add_triangle(free_edges, used_nodes, triangles, max_edge, first_triangle_node)

            self.find_next_node(free_edges, used_nodes, triangles)

        count = 1
        for i in self.result_triangles:
            self.draw_solution(i, count)
            count += 1