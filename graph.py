class Graph:
    def __init__(self):
        self.edges = []
        self.nodes = []

    def get_edges(self):
        return self.edges

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

    def get_node(self, node_name):
        for node in self.nodes:
            if node.name == node_name:
                return node
        return None

    def get_edge(self, first_node, second_node):
        for edge in self.edges:
            if (edge.first_node == first_node and edge.second_node == second_node) \
                    or (edge.first_node == second_node and edge.second_node == first_node):
                return edge
        return None

    def get_edge_by_node_names(self, first_node, second_node):
        for edge in self.edges:
            if edge.first_node.name == first_node and edge.second_node.name == second_node:
                return edge
        return None

    def get_max_edge(self):
        max_weight = max(self.edges, key=lambda x: x.weight).weight  # max weight in graph
        max_edges = [obj for obj in self.edges if obj.weight == max_weight]  # all edges with max weight
        return max(max_edges, key=lambda x: self.get_number_of_connections(x))  # return edge with max number of connections

    def get_number_of_connections(self, edge):
        number = 0
        for obj in self.edges:
            if obj.equals(edge):
                continue
            if obj.has_node(edge.first_node) or obj.has_node(edge.second_node):
                number += 1
        return number
