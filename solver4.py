from edge import Edge
from graph import Graph
from node import Node
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import threading
import os
import time
from PyQt6.QtCore import QSize, Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import QApplication, QDialog, QProgressBar, QAbstractItemView, QTableWidgetItem, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QTableWidget, QHBoxLayout, QHeaderView

class Solver4(QDialog):
    def __init__(self, graph):
        super().__init__()

        self.graph = graph
        self.result_triangles = []
        self.max_number_of_solutions = 50
        self.first_edge = "" 
        self.drawn_solutions = []
        self.number_of_solutions = 0
        self.test = 0
        self.first_thread_min_cost = 0
        self.second_thread_min_cost = 0
        self.third_thread_min_cost = 0
        self.fourth_thread_min_cost = 0
        self.first_thread_solutions = []
        self.second_thread_solutions = []
        self.third_thread_solutions = []
        self.fourth_thread_solutions = []
        self.first_thread_number_of_solutions = 0
        self.second_thread_number_of_solutions = 0
        self.third_thread_number_of_solutions = 0
        self.fourth_thread_number_of_solutions = 0

        # self.test_multi()

    # generate edges, that don't exist, sets weight to 0
    def generate_all_edges(self):
        for i in self.graph.nodes:
            for j in self.graph.nodes:
                if i == j:
                    continue
                if self.graph.get_edge_by_node_names(i.name, j.name) is None:
                    self.graph.add_edge(Edge(i, j, 0))

    def cls(self):
        os.system('cls' if os.name=='nt' else 'clear')

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

    def get_triangles_with_common_node(self, node, triangles, free_edges_of_triangles):
        triangles_strings = []
        triangles_string = ""
        for triangle in triangles:
            triangle_string = "".join(set(triangle[0].first_node.name + triangle[0].second_node.name +
                            triangle[1].first_node.name + triangle[1].second_node.name +
                            triangle[2].first_node.name + triangle[2].second_node.name))
            triangles_strings.append(triangle_string)
            triangles_string += triangle_string

        number_of_triangles_with_node = 0
        triangles_with_node = ""
        for string in triangles_strings:
            if node.name in string:
                number_of_triangles_with_node += 1
                triangles_with_node = triangles_with_node + string

        unique_nodes = ""
        for i in triangles_with_node:
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
            free_edges_of_triangles.append(edge_with_unique_nodes)
            if free_edges_of_triangles.count(first_edge_to_remove) != 0:
                free_edges_of_triangles.remove(first_edge_to_remove)
            if free_edges_of_triangles.count(second_edge_to_remove) != 0:
                free_edges_of_triangles.remove(second_edge_to_remove)
            if free_edges_of_triangles.count(third_edge_to_remove) != 0:
                free_edges_of_triangles.remove(third_edge_to_remove)
            if free_edges_of_triangles.count(forth_edge_to_remove) != 0:
                free_edges_of_triangles.remove(forth_edge_to_remove)

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

        if edges_max_nodes:
            for x, y in edges_max_nodes.items():
                for node in y[0]:
                    new_triangles = triangles.copy()
                    new_free_edges_of_triangles = free_edges_of_triangles.copy()
                    new_used_nodes = used_nodes.copy()
                    self.add_triangle(new_free_edges_of_triangles, new_used_nodes, new_triangles, x, node)
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
            number_of_same_triangles = 0
            for drawn_triangle in drawn_solution:
                for solution_triangle in triangles:
                    if sorted(drawn_triangle) == sorted(solution_triangle):
                        number_of_same_triangles = number_of_same_triangles + 1
            
            if number_of_same_triangles == len(triangles):
                solution_already_drawn = True


        if solution_already_drawn:
            return
        
        self.drawn_solutions.append(triangles)
        while len(nodes_coordinates) < len(self.graph.nodes):
            for triangle in triangles:
                if nodes_coordinates.get(triangle[0]) is not None and nodes_coordinates.get(triangle[1]) is not None \
                        and nodes_coordinates.get(triangle[2]) is not None:
                    coordinates_array.append(nodes_coordinates[triangle[0]])
                    coordinates_array.append(nodes_coordinates[triangle[1]])
                    coordinates_array.append(nodes_coordinates[triangle[2]])
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
            free_edges_of_triangles = []
            used_nodes = []
            self.add_triangle(free_edges_of_triangles, used_nodes, triangles, max_edge, first_triangle_node)

            self.find_next_node(free_edges_of_triangles, used_nodes, triangles)

        count = 1
        for i in self.result_triangles:
            self.draw_solution(i, count)
            count += 1

    def check_all_combos(self, start_position, end_position, thread_number):

        # A B C D E

        # 2 nodes must be in the same triangle and the third one must be not in triangles

        # ABC ABD BDE 

        for i in range(start_position, end_position):
            for j in range(0, len(self.graph.nodes)):
                if i == j:
                    continue
                for k in range(0, len(self.graph.nodes)):
                    if k == i or k == j:
                        continue
                    solution_triangles = []
                    used_nodes = []
                    free_edges = []
                    first_node = self.graph.nodes[i]
                    second_node = self.graph.nodes[j]
                    third_node = self.graph.nodes[k]
                    first_edge = self.graph.get_edge(first_node, second_node)
                    second_edge = self.graph.get_edge(second_node, third_node)
                    third_edge = self.graph.get_edge(first_node, third_node)
                    first_triangle = [first_edge, second_edge, third_edge]
                    used_nodes.append(first_node)
                    used_nodes.append(second_node)
                    used_nodes.append(third_node)
                    free_edges.append(first_edge)
                    free_edges.append(second_edge)
                    free_edges.append(third_edge)
                    solution_triangles.append(first_triangle)
                    self.generate_next_triangle(solution_triangles.copy(), used_nodes.copy(), free_edges.copy(), thread_number)

    def generate_next_triangle(self, solution_triangles, used_nodes, free_edges, thread_number):
        for i in free_edges:
            for j in self.graph.nodes:
                if j in used_nodes:
                    continue
                
                total_number_of_solutions = self.first_thread_number_of_solutions + self.second_thread_number_of_solutions + self.third_thread_number_of_solutions + self.fourth_thread_number_of_solutions
                if total_number_of_solutions > 2000000:
                    return
                second_edge = self.graph.get_edge(i.first_node, j)
                third_edge = self.graph.get_edge(i.second_node, j)
                triangle = [i, second_edge, third_edge]

                used_nodes_copy = used_nodes.copy()
                used_nodes_copy.append(j)
                solution_triangles_copy = solution_triangles.copy()
                solution_triangles_copy.append(triangle)

                free_edges_copy = free_edges.copy()

                if free_edges_copy.count(i) != 0:
                    free_edges_copy.remove(i)
                else:
                    free_edges_copy.append(i)

                if free_edges_copy.count(second_edge) != 0:
                    free_edges_copy.remove(second_edge)
                else:
                    free_edges_copy.append(second_edge)

                if free_edges_copy.count(third_edge) != 0:
                    free_edges_copy.remove(third_edge)
                else:
                    free_edges_copy.append(third_edge)

                if len(solution_triangles_copy) >= 5:
                    for e in self.graph.nodes:
                        self.get_triangles_with_common_node(e, solution_triangles_copy, free_edges_copy)

                if set(self.graph.nodes).issubset(set(used_nodes_copy)):
                    solution_cost = self.calculate_solution_value(solution_triangles_copy)
                    
                    if thread_number == 1:
                        if len(self.first_thread_solutions) == 0:
                            self.first_thread_min_cost = solution_cost
                        if solution_cost == self.first_thread_min_cost and len(self.first_thread_solutions) < self.max_number_of_solutions:
                            self.first_thread_solutions.append(solution_triangles_copy)
                        elif solution_cost < self.first_thread_min_cost:
                            self.first_thread_solutions = []
                            self.first_thread_solutions.append(solution_triangles_copy)
                            self.first_thread_min_cost = solution_cost
                        self.first_thread_number_of_solutions = self.first_thread_number_of_solutions + 1
                    elif thread_number == 2:
                        if len(self.second_thread_solutions) == 0:
                            self.second_thread_min_cost = solution_cost
                        if solution_cost == self.second_thread_min_cost and len(self.second_thread_solutions) < self.max_number_of_solutions:
                            self.second_thread_solutions.append(solution_triangles_copy)
                        elif solution_cost < self.second_thread_min_cost:
                            self.second_thread_solutions = []
                            self.second_thread_solutions.append(solution_triangles_copy)
                            self.second_thread_min_cost = solution_cost
                        self.second_thread_number_of_solutions = self.second_thread_number_of_solutions + 1
                        
                    elif thread_number == 3:
                        if len(self.third_thread_solutions) == 0:
                            self.third_thread_min_cost = solution_cost
                        if solution_cost == self.third_thread_min_cost and len(self.third_thread_solutions) < self.max_number_of_solutions:
                            self.third_thread_solutions.append(solution_triangles_copy)
                        elif solution_cost < self.third_thread_min_cost:
                            self.third_thread_solutions = []
                            self.third_thread_solutions.append(solution_triangles_copy)
                            self.third_thread_min_cost = solution_cost
                        self.third_thread_number_of_solutions = self.third_thread_number_of_solutions + 1
                        print(self.third_thread_number_of_solutions)
                    elif thread_number == 4:
                        if len(self.fourth_thread_solutions) == 0:
                            self.fourth_thread_min_cost = solution_cost
                        if solution_cost == self.fourth_thread_min_cost and len(self.fourth_thread_solutions) < self.max_number_of_solutions:
                            self.fourth_thread_solutions.append(solution_triangles_copy)
                        elif solution_cost < self.fourth_thread_min_cost:
                            self.fourth_thread_solutions = []
                            self.fourth_thread_solutions.append(solution_triangles_copy)
                            self.fourth_thread_min_cost = solution_cost
                        self.fourth_thread_number_of_solutions = self.fourth_thread_number_of_solutions + 1
                    break
                else:
                    self.generate_next_triangle(solution_triangles_copy, used_nodes_copy, free_edges_copy, thread_number)

    def test_cost(self):
        edge_63 = self.graph.get_edge_by_node_names("F", "C")
        edge_61 = self.graph.get_edge_by_node_names("F", "A")
        edge_31 = self.graph.get_edge_by_node_names("C", "A")
        edge_35 = self.graph.get_edge_by_node_names("C", "E")
        edge_15 = self.graph.get_edge_by_node_names("A", "E")
        edge_14 = self.graph.get_edge_by_node_names("A", "D")
        edge_54 = self.graph.get_edge_by_node_names("E", "D")
        edge_58 = self.graph.get_edge_by_node_names("E", "H")
        edge_54 = self.graph.get_edge_by_node_names("E", "D")
        edge_48 = self.graph.get_edge_by_node_names("D", "H")
        edge_87 = self.graph.get_edge_by_node_names("H", "G")
        edge_47 = self.graph.get_edge_by_node_names("D", "G")
        edge_82 = self.graph.get_edge_by_node_names("H", "B")
        edge_72 = self.graph.get_edge_by_node_names("G", "B")

        solution_triangles = []
        solution_triangles.append([edge_63, edge_61, edge_31])
        solution_triangles.append([edge_31, edge_35, edge_15])
        solution_triangles.append([edge_15, edge_14, edge_54])
        solution_triangles.append([edge_58, edge_54, edge_48])
        solution_triangles.append([edge_48, edge_47, edge_87])
        solution_triangles.append([edge_87, edge_82, edge_72])

        cost = self.calculate_solution_value(solution_triangles)
        print(str(cost))

    def test_multi(self):
        t1 = None

        if len(self.graph.nodes) == 4:
            t1 = threading.Thread(target=self.check_all_combos, args=(0, 4, 1))
        elif len(self.graph.nodes) == 5:
            t1 = threading.Thread(target=self.check_all_combos, args=(0, 5, 1))
        elif len(self.graph.nodes) == 6:   
            t1 = threading.Thread(target=self.check_all_combos, args=(0, 6, 1))
        elif len(self.graph.nodes) == 7:   
            t1 = threading.Thread(target=self.check_all_combos, args=(0, 7, 1))
        elif len(self.graph.nodes) == 8:   
            t1 = threading.Thread(target=self.check_all_combos, args=(2, 6, 1))
        elif len(self.graph.nodes) == 9:   
            t1 = threading.Thread(target=self.check_all_combos, args=(2, 6, 1))
        elif len(self.graph.nodes) == 10:   
            t1 = threading.Thread(target=self.check_all_combos, args=(2, 6, 1))

        t1.start()

        t1.join()

        # min_costs = [self.first_thread_min_cost, self.second_thread_min_cost, self.third_thread_min_cost, self.fourth_thread_min_cost]

        # min_cost = min(min_costs)

        # all_min_solutions = []

        # if self.first_thread_min_cost == min_cost:
        #     all_min_solutions.extend(self.first_thread_solutions)
        # if self.second_thread_min_cost == min_cost:
        #     all_min_solutions.extend(self.second_thread_solutions)
        # if self.third_thread_min_cost == min_cost:
        #     all_min_solutions.extend(self.third_thread_solutions)
        # if self.fourth_thread_min_cost == min_cost:
        #     all_min_solutions.extend(self.fourth_thread_solutions)

        count = 1
        for i in range(0, 50):
            if i < len(self.first_thread_solutions):
                self.draw_solution(self.first_thread_solutions[i], count)
                count += 1
            else:
                break
